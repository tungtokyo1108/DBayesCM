#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:04:41 2024

@author: tungbioinfo
"""

import tempfile

#import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import torch

import warnings
from math import ceil, floor
import numpy as np
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    Sampler,
    SequentialSampler,
)

import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.functional import one_hot
from torch import nn
from torch.nn import ModuleList
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import collections
from collections.abc import Callable, Iterable
from typing import Iterable, List
import math
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score

import lightning.pytorch as pl
from lightning.pytorch.strategies.ddp import DDPStrategy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks import RichProgressBar
from scipy import special

# ============================================================================
# CORRECTED vMF HELPER FUNCTIONS (Based on Taghia et al. 2014)
# ============================================================================

def bessel_ratio_stable(nu, kappa):
    """
    Compute I_{nu+1}(kappa) / I_nu(kappa) = ∂/∂κ log I_nu(κ)

    CORRECTED: Fixed numerical overflow by using only exponentially-scaled Bessel functions

    Args:
        nu: Bessel order (D/2 - 1 for D-dimensional vMF)
        kappa: concentration parameter [K] or scalar

    Returns:
        ratio: Bessel ratio [K] or scalar (clipped to [0, 1] for stability)
    """
    if isinstance(kappa, torch.Tensor):
        kappa_np = kappa.detach().cpu().numpy()
        device = kappa.device
        dtype = kappa.dtype
    else:
        kappa_np = np.array(kappa)
        device = 'cpu'
        dtype = torch.float32

    # Clip kappa to reasonable range to prevent overflow
    kappa_np = np.clip(kappa_np, 1e-10, 700.0)  # exp(700) is near float max

    nu_np = float(nu)
    ratio = np.zeros_like(kappa_np)

    # Small kappa: ratio ≈ κ/(2(ν+1))
    small_mask = kappa_np < 1.0
    ratio[small_mask] = kappa_np[small_mask] / (2.0 * (nu_np + 1.0))

    # Large kappa: ratio ≈ 1 - (ν+0.5)/κ (asymptotic)
    large_mask = kappa_np >= 50.0
    ratio[large_mask] = 1.0 - (nu_np + 0.5) / kappa_np[large_mask]

    # Medium kappa: use ONLY exponentially-scaled Bessel (no exp multiplication!)
    medium_mask = ~(small_mask | large_mask)
    if np.any(medium_mask):
        kappa_med = kappa_np[medium_mask]
        # ive(nu, x) is ALREADY exponentially scaled: ive(nu,x) = exp(-|x|) * iv(nu,x)
        # So ratio = ive(nu+1, x) / ive(nu, x) directly!
        i_nu = special.ive(nu_np, kappa_med)
        i_nu_plus = special.ive(nu_np + 1, kappa_med)
        ratio[medium_mask] = i_nu_plus / (i_nu + 1e-10)

    # Clip ratio to valid range [0, 1] for stability
    ratio = np.clip(ratio, 0.0, 1.0)

    return torch.tensor(ratio, dtype=dtype, device=device)


def log_bessel_stable(nu, kappa):
    """
    Compute log I_nu(κ) stably

    CORRECTED: Fixed overflow by using log(ive) + kappa correctly

    Args:
        nu: Bessel order
        kappa: concentration [K] or scalar

    Returns:
        log_i_nu: log I_nu(κ) [K] or scalar
    """
    if isinstance(kappa, torch.Tensor):
        kappa_np = kappa.detach().cpu().numpy()
        device = kappa.device
        dtype = kappa.dtype
    else:
        kappa_np = np.array(kappa)
        device = 'cpu'
        dtype = torch.float32

    # Clip kappa to prevent overflow
    kappa_np = np.clip(kappa_np, 1e-10, 700.0)

    nu_np = float(nu)
    log_bessel = np.zeros_like(kappa_np)

    # Small κ: log I_nu(κ) ≈ nu*log(κ/2) - log Γ(nu+1)
    small_mask = kappa_np < 1.0
    if np.any(small_mask):
        log_bessel[small_mask] = (
            nu_np * np.log(kappa_np[small_mask] / 2.0) -
            special.gammaln(nu_np + 1.0)
        )

    # Large κ: log I_nu(κ) ≈ κ - 0.5*log(2πκ)
    large_mask = kappa_np >= 50.0
    if np.any(large_mask):
        log_bessel[large_mask] = (
            kappa_np[large_mask] -
            0.5 * np.log(2.0 * np.pi * kappa_np[large_mask])
        )

    # Medium κ: use log(ive) + kappa
    # Since ive(nu,x) = exp(-|x|) * iv(nu,x), we have log(iv) = log(ive) + |x|
    medium_mask = ~(small_mask | large_mask)
    if np.any(medium_mask):
        kappa_med = kappa_np[medium_mask]
        ive_vals = special.ive(nu_np, kappa_med)
        # Avoid log(0) by adding small epsilon
        log_bessel[medium_mask] = np.log(np.maximum(ive_vals, 1e-300)) + kappa_med

    return torch.tensor(log_bessel, dtype=dtype, device=device)


def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()

# ============================================================================
# vMF SAMPLING AND KL DIVERGENCE (Davidson et al., 2018 — S-VAE)
# ============================================================================

def sample_vmf_weight(kappa, D, max_iter=100):
    """
    Sample omega from the marginal density:
        g(omega | kappa, D) ∝ exp(kappa*omega) * (1 - omega^2)^((D-3)/2), omega in [-1, 1]
    using the Wood (1994) acceptance-rejection scheme.

    Args:
        kappa: [B] concentration parameters (positive tensors)
        D:     int, ambient dimension of the sphere S^{D-1}
        max_iter: maximum number of rejection-sampling iterations

    Returns:
        omega: [B] accepted samples (no gradient — sampling is treated as constant)
    """
    device = kappa.device
    dtype  = kappa.dtype
    B      = kappa.shape[0]
    m      = float(D)

    # Wood (1994) auxiliary parameters
    sqrt_term = torch.sqrt(4.0 * kappa ** 2 + (m - 1.0) ** 2)
    b  = (m - 1.0) / (2.0 * kappa + sqrt_term)          # [B]
    x0 = (1.0 - b) / (1.0 + b)                          # [B]
    c  = kappa * x0 + (m - 1.0) * torch.log(1.0 - x0 ** 2 + 1e-10)  # [B]

    omega    = torch.zeros(B, device=device, dtype=dtype)
    accepted = torch.zeros(B, dtype=torch.bool, device=device)

    with torch.no_grad():
        b_d      = b.detach()
        c_d      = c.detach()
        kappa_d  = kappa.detach()
        x0_d     = x0.detach()
        beta_a   = (m - 1.0) / 2.0

        for _ in range(max_iter):
            # Sample t ~ Beta((D-1)/2, (D-1)/2)
            t = torch.distributions.Beta(
                torch.full((B,), beta_a, device=device, dtype=dtype),
                torch.full((B,), beta_a, device=device, dtype=dtype),
            ).sample()

            # Transform t -> w in [-1, 1]
            w = (1.0 - (1.0 + b_d) * t) / (1.0 - (1.0 - b_d) * t + 1e-10)
            w = torch.clamp(w, -1.0 + 1e-7, 1.0 - 1e-7)

            # Log acceptance criterion
            u         = torch.rand(B, device=device, dtype=dtype)
            log_ratio = kappa_d * w + (m - 1.0) * torch.log(1.0 - x0_d * w + 1e-10) - c_d
            accept    = (log_ratio >= torch.log(u + 1e-10)) & (~accepted)

            omega    = torch.where(accept, w, omega)
            accepted = accepted | accept

            if accepted.all():
                break

    return omega   # [B], no grad


def vmf_householder_transform(z_prime, mu):
    """
    Apply the Householder reflection that maps e_1 to mu, i.e. H(mu) * e_1 = mu.

    H(u) z = z - 2 * (u^T z) * u,   where u = (e_1 - mu) / ||e_1 - mu||

    Args:
        z_prime: [B, D]  samples built around e_1 (first axis)
        mu:      [B, D]  target unit-direction vectors

    Returns:
        z:       [B, D]  rotated samples distributed as vMF(mu, kappa)
    """
    e1      = torch.zeros_like(mu)
    e1[:, 0] = 1.0

    u       = e1 - mu                                          # [B, D]
    u_norm  = torch.norm(u, p=2, dim=-1, keepdim=True)        # [B, 1]
    u_hat   = u / (u_norm + 1e-8)                             # [B, D]

    dot = torch.sum(u_hat * z_prime, dim=-1, keepdim=True)    # [B, 1]
    z   = z_prime - 2.0 * dot * u_hat                         # [B, D]
    return z


def sample_vmf(mu, kappa):
    """
    Draw one sample z ~ vMF(mu, kappa) per row using Ulrich (1984) / Algorithm 1 of
    Davidson et al. (2018).

    Gradient flows through mu via the differentiable Householder transform.
    Gradient through kappa is carried by the KL term (see vmf_kl_divergence).

    Args:
        mu:    [B, D]  mean directions (unit vectors on S^{D-1})
        kappa: [B]     concentration parameters (>= 0)

    Returns:
        z:     [B, D]  samples on the unit sphere S^{D-1}
    """
    B, D   = mu.shape
    device = mu.device
    dtype  = mu.dtype

    # Step 1: sample the axial component omega (no gradient)
    omega = sample_vmf_weight(kappa, D)   # [B]

    # Step 2: sample a uniform direction on S^{D-2}
    if D > 1:
        v = F.normalize(torch.randn(B, D - 1, device=device, dtype=dtype), p=2, dim=-1)
    else:
        v = torch.zeros(B, 0, device=device, dtype=dtype)

    # Step 3: build z' = (omega, sqrt(1 - omega^2) * v^T)
    sqrt_omg = torch.sqrt(torch.clamp(1.0 - omega ** 2, min=1e-10)).unsqueeze(-1)  # [B,1]
    z_prime  = torch.cat([omega.unsqueeze(-1), sqrt_omg * v], dim=-1)              # [B, D]

    # Step 4: Householder rotation e_1 -> mu  (differentiable w.r.t. mu)
    z = vmf_householder_transform(z_prime, mu)   # [B, D]
    return z


class _VmfKLFunction(torch.autograd.Function):
    """
    Custom autograd function for KL( vMF(mu, kappa) || U(S^{D-1}) ).

    Eq. (5) of Davidson et al. (2018):
        KL = kappa * A_m(kappa)
           + (D/2 - 1) * log(kappa)
           - log I_{D/2-1}(kappa)
           - log Gamma(D/2)

    where A_m(kappa) = I_{D/2}(kappa) / I_{D/2-1}(kappa)  (Bessel ratio).

    The forward uses numerically-stable scipy Bessel functions.
    The backward uses the analytic formula:
        d KL / d kappa = A_m * (1 - kappa * A_m) + (D/2 - 1) / kappa
    derived from  d A_m / d kappa = A_m / kappa - A_m^2.
    """

    @staticmethod
    def forward(ctx, kappa, D):
        nu      = D / 2.0 - 1.0                         # Bessel order
        A_m     = bessel_ratio_stable(nu, kappa)         # I_{D/2} / I_{D/2-1}  [B]
        log_I_nu = log_bessel_stable(nu, kappa)          # log I_{D/2-1}(κ)     [B]

        kl = (kappa * A_m
              + (D / 2.0 - 1.0) * torch.log(kappa + 1e-10)
              - log_I_nu
              - math.lgamma(D / 2.0))

        ctx.save_for_backward(kappa, A_m)
        ctx.D = D
        return kl

    @staticmethod
    def backward(ctx, grad_output):
        kappa, A_m = ctx.saved_tensors
        D = ctx.D
        # d KL / d kappa = A_m*(1 - kappa*A_m) + (D/2 - 1)/kappa
        grad_kappa = grad_output * (
            A_m * (1.0 - kappa * A_m) + (D / 2.0 - 1.0) / (kappa + 1e-10)
        )
        return grad_kappa, None


def vmf_kl_divergence(kappa, D):
    """
    KL( vMF(mu, kappa) || U(S^{D-1}) ) — independent of mu.

    Args:
        kappa: [B]  concentration parameters
        D:     int  latent dimension

    Returns:
        kl:    [B]  per-sample KL divergence (differentiable w.r.t. kappa)
    """
    return _VmfKLFunction.apply(kappa, D)

def stick_breaking(v: torch.Tensor):

    """
    Stick-breaking construction for Dirichlet Process
    
    Args:
        v: Beta distributed random variables with shape [K]
        
    Returns:
        pi: mixture weights that sum to 1, shape [K]
    """
    cumprod_v = torch.cumprod(1 - v, dim=0)
    cumprod_v = torch.cat([torch.ones(1, device=v.device), cumprod_v[:-1]], dim=0)
    pi = v * cumprod_v
    return pi

def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)

class FCLayers(nn.Module):
    
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        cat_dim = sum(self.n_cat_list)
        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer {i}",
                        nn.Sequential(
                            nn.Linear(
                                n_in + cat_dim * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow
                            # implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:], strict=True)
                    )
                ]
            )
        )
        
    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond
    
    def set_online_update_hooks(self, hook_first_layer=True):
        """Set online update hooks."""
        self.hooks = []

        def _hook_fn_weight(grad):
            categorical_dims = sum(self.n_cat_list)
            new_grad = torch.zeros_like(grad)
            if categorical_dims > 0:
                new_grad[:, -categorical_dims:] = grad[:, -categorical_dims:]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if isinstance(layer, nn.Linear):
                    if self.inject_into_layer(i):
                        w = layer.weight.register_hook(_hook_fn_weight)
                    else:
                        w = layer.weight.register_hook(_hook_fn_zero_out)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)
                    
    def forward(self, x: torch.Tensor, *cat_list: int):
        
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError("nb. categorical args provided doesn't match init. params.")
            
        for n_cat, cat in zip(self.n_cat_list, cat_list, strict=False):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = nn.functional.one_hot(cat.squeeze(-1), n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
                
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            if (
                                x.device.type == "mps"
                            ):  # TODO: remove this when MPS supports for loop.
                                x = torch.cat(
                                    [(layer(slice_x.clone())).unsqueeze(0) for slice_x in x], dim=0
                                )
                            else:
                                x = torch.cat(
                                    [layer(slice_x).unsqueeze(0) for slice_x in x], dim=0
                                )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand((x.size(0), o.size(0), o.size(1)))
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x
    
        
class MaskedLinear(nn.Linear):
    """ 
    same as Linear except has a configurable mask on the weights 
    """
    
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', mask)
        
    def forward(self, input):
        #mask = Variable(self.mask, requires_grad=False)
        if self.bias is None:
            return F.linear(input, self.weight*self.mask)
        else:
            return F.linear(input, self.weight*self.mask, self.bias)

class MaskedLinearLayers(FCLayers):

    def __init__(
        self, 
        n_in: int,
        n_out: int,
        mask: torch.Tensor = None,
        mask_first: bool = True,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.ReLU
        ):
            
        super().__init__(
            n_in=n_in,
            n_out=n_out,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_activation=use_activation,
            bias=bias,
            inject_covariates=inject_covariates,
            activation_fn=activation_fn
            )

        self.mask = mask 

        if mask is not None:
            if mask_first:
                layers_dim = [n_in] + [mask.shape[0]] + (n_layers - 1) * [n_hidden] + [n_out]
            else:
                layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [mask.shape[0]] + [n_out]
        else:    
            layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        cat_dim = sum(self.n_cat_list)

        # concatnat one hot encoding to mask if available
        if cat_dim>0:
            mask_input = torch.cat((self.mask, torch.ones(cat_dim, self.mask.shape[1])), dim=0)
        else:
            mask_input = self.mask        

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(
                                n_in + cat_dim * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )
        if mask is not None:
            if mask_first:
                # change the first layer to be MaskedLinear
                self.fc_layers[0] = nn.Sequential(
                                            MaskedLinear(
                                                layers_dim[0] + cat_dim * self.inject_into_layer(0),
                                                layers_dim[1],
                                                mask_input,
                                                bias=bias,
                                            ),
                                            # non-default params come from defaults in original Tensorflow implementation
                                            nn.BatchNorm1d(layers_dim[1], momentum=0.01, eps=0.001)
                                            if use_batch_norm
                                            else None,
                                            nn.LayerNorm(layers_dim[1], elementwise_affine=False)
                                            if use_layer_norm
                                            else None,
                                            activation_fn() if use_activation else None,
                                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                                            )
            else:
                # change the last layer to be MaskedLinear
                self.fc_layers[-1] = nn.Sequential(
                                            MaskedLinear(
                                                layers_dim[-2] + cat_dim * self.inject_into_layer(0),
                                                layers_dim[-1],
                                                torch.transpose(mask_input,0,1),
                                                bias=bias,
                                            ),
                                            # non-default params come from defaults in original Tensorflow implementation
                                            nn.BatchNorm1d(layers_dim[-1], momentum=0.01, eps=0.001)
                                            if use_batch_norm
                                            else None,
                                            nn.LayerNorm(layers_dim[-1], elementwise_affine=False)
                                            if use_layer_norm
                                            else None,
                                            activation_fn() if use_activation else None,
                                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                                            )


    def forward(self, x: torch.Tensor, *cat_list: int):
        """
        Forward computation on ``x``.
        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample
        x: torch.Tensor
        Returns
        -------
        py:class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError(
                "nb. categorical args provided doesn't match init. params."
            )
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if (isinstance(layer, nn.Linear) or isinstance(layer, MaskedLinear)) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x        
        
        
class EncoderDBGCM(nn.Module):
    
    def __init__(
            self, 
            n_input_microbiome: int,
            n_input_metabolite: int,
            n_latent: int,
            n_cat_list: Iterable[int] = None,
            mask: torch.Tensor = None,
            mask_first: bool = True,
            n_layers: int = 2,
            n_hidden: int = 256,
            n_layers_individual: int = 1,
            n_layers_shared: int = 2,
            dropout_rate: float = 0.1,
            use_batch_norm: bool = True,
            use_layer_norm: bool = False,
            use_LayerNorm: bool = False,
            combine_method: str = "add",
    ):
        
        super().__init__()
        
        self.use_LayerNorm = use_LayerNorm
        
        # Add layer normalization for better gradient flow
        if self.use_LayerNorm:
            self.input_norm_microbiome = nn.LayerNorm(n_input_microbiome)
            self.input_norm_metabolite = nn.LayerNorm(n_input_metabolite)
            
        self.combine_method = combine_method

        """
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        """
        
        self.encoder_microbiome = MaskedLinearLayers(
                    n_in=n_input_microbiome,
                    n_out=n_hidden,
                    n_cat_list=n_cat_list,
                    mask=mask,
                    mask_first=mask_first,
                    n_layers=n_layers_individual,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm,
        )
        
        self.encoder_metabolite = MaskedLinearLayers(
                    n_in=n_input_metabolite,
                    n_out=n_hidden,
                    n_cat_list=n_cat_list,
                    mask=mask,
                    mask_first=mask_first,
                    n_layers=n_layers_individual,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm,
        )
        
        if self.combine_method == 'concat':
            dim_encoder_shared = n_hidden + n_hidden
        elif self.combine_method == 'add':
            dim_encoder_shared = n_hidden
        else:
            raise ValueError("combine method must choose from concat or add") 
        
        self.encoder_shared = FCLayers(
            n_in=dim_encoder_shared,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers_shared,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        
        # vMF encoder heads
        # Mean direction: output in R^D, then L2-normalised to S^{D-1}
        self.z_mean_encoder = nn.Linear(n_hidden, n_latent)
        nn.init.xavier_uniform_(self.z_mean_encoder.weight, gain=0.01)

        # Concentration parameter κ (scalar per sample, enforced positive via softplus)
        self.z_kappa_encoder = nn.Linear(n_hidden, 1)
        nn.init.xavier_uniform_(self.z_kappa_encoder.weight, gain=0.01)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, *cat_list: int):
        """
        Encode paired omics inputs as a von Mises-Fisher (vMF) posterior.

        Returns:
            qz_mu:    [B, D]  mean direction on S^{D-1}  (unit vectors)
            qz_kappa: [B]     concentration parameter κ   (>= 0)
            z:        [B, D]  reparameterised sample from vMF(qz_mu, qz_kappa)
        """
        if self.use_LayerNorm:
            x = self.input_norm_microbiome(x)
            y = self.input_norm_metabolite(y)

        q_x = self.encoder_microbiome(x, *cat_list)
        q_y = self.encoder_metabolite(y, *cat_list)

        if self.combine_method == 'concat':
            q = torch.cat([q_x, q_y], dim=-1)
        elif self.combine_method == 'add':
            q = (q_x + q_y) / 2.
        else:
            raise ValueError("combine method must choose from concat or add")

        q = self.encoder_shared(q, *cat_list)

        # Mean direction: L2-normalise to unit sphere S^{D-1}
        qz_mu = F.normalize(self.z_mean_encoder(q), p=2, dim=-1)  # [B, D]

        # Concentration parameter: softplus ensures κ > 0
        qz_kappa = F.softplus(self.z_kappa_encoder(q)).squeeze(-1)  # [B]
        qz_kappa = torch.clamp(qz_kappa, min=1e-3)

        # Reparameterised vMF sample (gradient through mu via Householder transform)
        z = sample_vmf(qz_mu, qz_kappa)   # [B, D] on unit sphere

        return qz_mu, qz_kappa, z

class DecoderDBGCM_vMF(nn.Module):
    """
    Decoder combining von Mises-Fisher (vMF) Mixture Model for clustering
    with Spike-and-Slab Group Horseshoe (SS-GH) for feature sparsity.

    This decoder integrates:
    1. vMF Mixture Model (from v2.4_vMF):
       - Dirichlet prior for mixture weights with automatic pruning (small alpha_0)
       - Von Mises-Fisher distributions for directional clustering on unit sphere
       - Gamma priors for concentration parameters (κ)
       - Variational inference for cluster assignments

    2. Spike-and-Slab Group Horseshoe (from v2.3_diagonal_SS-GH):
       - Node inclusion indicators (spike-and-slab) for feature selection
       - Hierarchical horseshoe prior for adaptive shrinkage:
         * Local scales (τ² = α*β) per feature per cluster
         * Global scale (ζ² = ζ_a*ζ_b) shared across features
       - Regularized horseshoe: τ̃² = (c²*τ²)/(c² + τ²*ζ²)

    Key Architectural Features:
    - Input z is normalized to unit sphere for vMF clustering
    - Separate SS-GH priors for microbiome and metabolite features
    - Batch-wise parameter sampling for variability
    - Numerical stability through clamping and log-space operations

    Mathematical Framework:
    - Clustering: p(z|μ_k,κ_k) ∝ exp(κ_k * μ_k^T * z)  [vMF on unit sphere]
    - Sparsity: w_k ~ z_k * N(0, σ₀²*τ̃²)  [spike-and-slab with horseshoe]
    - Reconstruction: x ~ DirichletMultinomial(exp(θ^T * W))

    Args:
        n_clusters: Number of mixture components (K)
        n_input: Latent dimension (D)
        n_output_microbiome: Number of microbiome features
        n_output_metabolite: Number of metabolite features
        pip0: Prior inclusion probability for spike-and-slab (default: 0.1)
        alpha_0: Dirichlet concentration (small for automatic pruning, default: 0.01)
        c_reg: Regularization parameter for horseshoe (default: 1.0)
        d0: Scale parameter for global horseshoe (default: 1.0)
        sigma0: Base standard deviation for weights (default: 1.0)
    """

    def __init__(
            self,
            n_clusters: int,
            n_input: int,
            n_output_microbiome: int,
            n_output_metabolite: int,
            pip0 = 0.1,
            v0 = 1,
            alpha_0: float = 0.01,  # Small Dirichlet prior for automatic pruning
            c_reg: float = 1.0,
            d0: float = 1.0,
            sigma0: float = 1.0,
            ):

        super().__init__()

        self.n_clusters = n_clusters # clusters
        self.n_output_microbiome = n_output_microbiome # microbiome
        self.n_output_metabolite = n_output_metabolite # metabolite
        self.n_input = n_input # latent_dim

        # ===== vMF Mixture Components =====
        # Dirichlet prior for mixture weights
        self.alpha_0 = alpha_0
        self.register_buffer('alpha_0_vec', torch.ones(n_clusters) * alpha_0)

        # Variational parameters for mixture weights (Dirichlet posterior)
        self.log_alpha = nn.Parameter(torch.log(torch.ones(n_clusters) * (alpha_0 + 1.0)))

        # vMF prior parameters for mean directions
        init_means = torch.randn(n_clusters, n_input)
        self.register_buffer('m_0', F.normalize(init_means, p=2, dim=1))
        self.register_buffer('beta_0', torch.ones(n_clusters) * 0.01)

        # Gamma prior for concentrations
        self.register_buffer('a_0', torch.ones(n_clusters) * 1.0)
        self.register_buffer('b_0', torch.ones(n_clusters) * 0.1)

        # Variational posterior for mean directions: vMF(μ_k, β_k*κ_k)
        self.means = nn.Parameter(torch.randn(n_clusters, n_input) * 0.1)
        self.log_beta = nn.Parameter(torch.log(torch.ones(n_clusters) * 1.0))

        # Variational posterior for concentrations: Gamma(a_k, b_k)
        self.log_a = nn.Parameter(torch.log(torch.ones(n_clusters) * 5.0))
        self.log_b = nn.Parameter(torch.log(torch.ones(n_clusters) * 1.0))

        # CORRECTED: Taylor expansion point for Bessel approximation (Eq. 28)
        # Initialize with mode of Gamma prior: (a_0-1)/b_0 if a_0>1, else a_0/b_0
        initial_kappa = torch.where(
            self.a_0 > 1.0,
            (self.a_0 - 1.0) / self.b_0,
            self.a_0 / self.b_0
        )
        self.register_buffer('lambda_bar', initial_kappa.clone())

        # ===== Spike-and-slab Group Horseshoe variational prior =====
        self.prior_inclusion_prob = pip0
        self.c_reg = c_reg
        self.sigma0 = sigma0
        self.d0 = d0
        self.eps = 1e-6
        
        # Variational parameters of SS-GHS for microbiome
        # Prior parameters
        self.logit_0 = nn.Parameter(torch.logit(torch.ones(1) * pip0, eps=1e-6), requires_grad=False)
        self.bias_d = nn.Parameter(torch.zeros(1, self.n_output_microbiome))
        
        # Weight parameters
        self.slab_mean = nn.Parameter(torch.randn(self.n_clusters, self.n_output_microbiome) * 0.1)
        self.slab_lnvar = nn.Parameter(torch.ones(self.n_clusters, self.n_output_microbiome) * (-1.0))
        
        # Node inclusion probability logits (for variational distribution)
        self.spike_logit = nn.Parameter(torch.zeros(self.n_clusters, self.n_output_microbiome) * self.logit_0)
        
        # Local horseshoe parameters (per node)
        # Each cluster has its own horseshoe local parameters (τ² = βα)
        self.alpha_mu = nn.Parameter(torch.zeros(self.n_clusters, self.n_output_microbiome))
        self.alpha_log_var = nn.Parameter(torch.ones(self.n_clusters, self.n_output_microbiome) * (-6.0))
        self.beta_mu = nn.Parameter(torch.zeros(self.n_clusters, self.n_output_microbiome))
        self.beta_log_var = nn.Parameter(torch.ones(self.n_clusters, self.n_output_microbiome) * (-6.0))
        
        # Variational parameters of SS-GHS for metabolite
        # Prior parameters
        self.logit_0_me = nn.Parameter(torch.logit(torch.ones(1) * pip0, eps=1e-6), requires_grad=False)
        self.bias_d_me = nn.Parameter(torch.zeros(1, self.n_output_metabolite))
        
        # Weight parameters
        self.slab_mean_me = nn.Parameter(torch.randn(self.n_clusters, self.n_output_metabolite) * 0.1)
        self.slab_lnvar_me = nn.Parameter(torch.ones(self.n_clusters, self.n_output_metabolite) * (-1.0))
        
        # Node inclusion probability logits (for variational distribution)
        self.spike_logit_me = nn.Parameter(torch.zeros(self.n_clusters, self.n_output_metabolite) * self.logit_0_me)
        
        # Local horseshoe parameters (per node)
        # Each cluster has its own horseshoe local parameters (τ² = βα)
        self.alpha_mu_me = nn.Parameter(torch.zeros(self.n_clusters, self.n_output_metabolite))
        self.alpha_log_var_me = nn.Parameter(torch.ones(self.n_clusters, self.n_output_metabolite) * (-6.0))
        self.beta_mu_me = nn.Parameter(torch.zeros(self.n_clusters, self.n_output_metabolite))
        self.beta_log_var_me = nn.Parameter(torch.ones(self.n_clusters, self.n_output_metabolite) * (-6.0))

        # Global horseshoe parameters (shared across all nodes) (ζ² = ζ_a ζ_b)
        self.zeta_a_mu = nn.Parameter(torch.zeros(1))
        self.zeta_a_log_var = nn.Parameter(torch.ones(1) * (-6.0))
        self.zeta_b_mu = nn.Parameter(torch.zeros(1))
        self.zeta_b_log_var = nn.Parameter(torch.ones(1) * (-6.0))
        
        # Log softmax operations
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def normalize_means(self):
        """Ensure mean directions are on unit sphere for vMF"""
        with torch.no_grad():
            self.means.data = F.normalize(self.means.data, p=2, dim=1)

    def approx_log_bessel(self, kappa, nu):
        """
        Approximate log I_nu(kappa) using asymptotic expansion
        Based on Taylor approximation approach
        """
        # For large κ: log I_ν(κ) ≈ κ - 0.5*log(2πκ)
        # For small κ: use series expansion
        large_kappa = kappa > 10

        # Large κ approximation (asymptotic)
        log_bessel_large = kappa - 0.5 * torch.log(2 * math.pi * (kappa + 1e-10))

        # Small κ approximation (series expansion)
        nu_tensor = torch.tensor(nu, dtype=kappa.dtype, device=kappa.device)
        log_bessel_small = nu * torch.log((kappa + 1e-10) / 2) - torch.lgamma(nu_tensor + 1)

        # Combine based on threshold
        log_bessel = torch.where(large_kappa, log_bessel_large, log_bessel_small)

        return log_bessel

    def vmf_log_likelihood(self, z: torch.Tensor):
        """
        CORRECTED: Compute E[log p(z_i | μ_k, κ_k)] for vMF distribution

        Implements Equation (36) with Taylor bound (Equations 19-21) from Taghia et al. 2014

        Key Fixes:
        1. Uses E[κ] and E[log κ] from Gamma posterior
        2. Uses Taylor bound for E[log I_nu(κ)] around λ̄_k for monotonic ELBO
        3. Stable Bessel function computation via scipy

        Args:
            z: [B, D] L2-normalized latent vectors

        Returns:
            log_lik: [B, K] E[log p(z_i|μ_k,κ_k)] for each component
        """
        B, D = z.shape
        K = self.n_clusters
        nu = D / 2.0 - 1.0  # Bessel order

        # Get posterior parameters
        a = torch.exp(self.log_a).clamp(min=0.1)  # [K]
        b = torch.exp(self.log_b).clamp(min=0.1)  # [K]

        # === E[κ] and E[log κ] under Gamma(a, b) posterior ===
        E_kappa = a / b  # [K]
        E_log_kappa = torch.digamma(a) - torch.log(b)  # [K]

        # Normalize mean directions to unit sphere
        mu = F.normalize(self.means, p=2, dim=1)  # [K, D]

        # === Compute E[log C_D(κ)] using Taylor bound (Eq. 19) ===
        # From Eq. (2): log C_D(κ) = (D/2-1)*log(κ) - (D/2)*log(2π) - log I_{D/2-1}(κ)

        lambda_bar = self.lambda_bar.clamp(min=0.1)  # [K]

        # log I_nu(λ̄_k) using stable computation
        log_I_bar = log_bessel_stable(nu, lambda_bar)  # [K]

        # ∂/∂λ log I_nu(λ̄_k) = I_{nu+1}(λ̄_k) / I_nu(λ̄_k)
        ratio_bar = bessel_ratio_stable(nu, lambda_bar)  # [K]

        # Taylor bound (Eq. 19): E[log I_nu(κ)] ≤ log I_nu(λ̄) + ratio_bar * E[κ - λ̄]
        E_log_I_nu = log_I_bar + ratio_bar * (E_kappa - lambda_bar)  # [K]

        # E[log C_D(κ)] = (D/2-1)*E[log κ] - (D/2)*log(2π) - E[log I_nu(κ)]
        log_2pi_term = -(D / 2.0) * math.log(2.0 * math.pi)
        E_log_C_D = nu * E_log_kappa + log_2pi_term - E_log_I_nu  # [K]

        # === Compute E[κ * μ^T * z] ===
        mu_dot_z = torch.matmul(z, mu.t())  # [B, K]
        E_kappa_mu_z = E_kappa.unsqueeze(0) * mu_dot_z  # [B, K]

        # === Complete E[log p(z|μ,κ)] (Eq. 36) ===
        log_lik = E_log_C_D.unsqueeze(0) + E_kappa_mu_z  # [B, K]

        return log_lik

    def compute_responsibilities(self, z: torch.Tensor):
        """
        CORRECTED E-STEP: Compute cluster responsibilities ξ_{nk}

        Implements Equations (38-40) from Taghia et al. 2014

        Args:
            z: [B, D] L2-normalized latent vectors

        Returns:
            resp: [B, K] responsibilities ξ_{nk} (soft assignments)
            log_resp: [B, K] log responsibilities
        """
        # E[log π_k] under Dirichlet posterior (Eq. 42)
        alpha = torch.exp(self.log_alpha).clamp(min=1e-6)  # [K]
        digamma_alpha = torch.digamma(alpha)
        digamma_sum = torch.digamma(alpha.sum())
        E_log_pi = digamma_alpha - digamma_sum  # [K]

        # E[log p(z|μ_k,κ_k)] with Taylor bounds (uses corrected vmf_log_likelihood)
        log_lik = self.vmf_log_likelihood(z)  # [B, K]

        # ln ρ̃_{nk} = E[ln π_k] + E[log p(z_n|μ_k,κ_k)] (Eq. 38)
        log_rho_tilde = E_log_pi.unsqueeze(0) + log_lik  # [B, K]

        # Normalize responsibilities (Eq. 40): ξ_{nk} = exp(ln ρ̃_{nk}) / Σ_j exp(ln ρ̃_{nj})
        log_resp = log_rho_tilde - torch.logsumexp(log_rho_tilde, dim=1, keepdim=True)
        resp = torch.exp(log_resp)  # [B, K]

        return resp, log_resp

    def update_posterior_parameters(self, X, resp):
        """
        CORRECTED M-STEP: Update all variational posterior parameters

        Implements Equations (42-48) from Taghia et al. 2014 with proper Bessel derivatives

        This is the critical missing piece that ensures proper variational updates!

        Args:
            X: [N, D] normalized data on unit sphere
            resp: [N, K] responsibilities from E-step
        """
        N, D = X.shape
        nu = D / 2.0 - 1.0  # Bessel order

        # Effective sample size per cluster
        N_k = resp.sum(dim=0)  # [K]

        # === 1. Update Dirichlet posterior α_k (Eq. 42) ===
        alpha_new = self.alpha_0_vec + N_k
        self.log_alpha.data = torch.log(alpha_new.clamp(min=1e-6))

        # === 2. Update vMF posterior for μ_k (Eq. 43-45) ===
        # Eq. (44): s_k = β_{0,k}*m_{0,k} + Σ_n ξ_{nk}*x_n
        weighted_sum = torch.matmul(resp.t(), X)  # [K, D]
        s_k = self.beta_0.unsqueeze(1) * self.m_0 + weighted_sum  # [K, D]

        # β_k = ||s_k||
        beta_new = torch.norm(s_k, p=2, dim=1)  # [K]

        # Eq. (45): μ_k = s_k / β_k
        mu_new = F.normalize(s_k, p=2, dim=1)  # [K, D]

        self.log_beta.data = torch.log(beta_new.clamp(min=1e-6))
        self.means.data = mu_new

        # === 3. Update Gamma posterior for κ_k (Eq. 47-48) ===
        # CRITICAL: Clip lambda_bar to prevent explosion
        lambda_bar = self.lambda_bar.clamp(min=0.1, max=100.0)  # [K]

        # Eq. (47): a_k = a_{0,k} + N_k*(D/2-1) + β_k * ∂/∂β_k ln I_nu(β_k*λ̄_k)
        kappa_bar_beta = (beta_new * lambda_bar).clamp(max=700.0)  # Prevent overflow
        ratio_beta = bessel_ratio_stable(nu, kappa_bar_beta)  # [K]
        deriv_beta = lambda_bar * ratio_beta  # [K]

        # Clip derivative contribution to prevent explosion
        a_new = self.a_0 + N_k * nu + (beta_new * deriv_beta).clamp(max=1000.0)  # [K]

        # Eq. (48): b_k = b_{0,k} + N_k*∂/∂λ̄ ln I_nu(λ̄) + β_{0,k}*∂/∂β_{0,k} ln I_nu(β_{0,k}*λ̄)
        ratio_lambda = bessel_ratio_stable(nu, lambda_bar)  # [K]

        kappa_bar_beta0 = (self.beta_0 * lambda_bar).clamp(max=700.0)  # Prevent overflow
        ratio_beta0 = bessel_ratio_stable(nu, kappa_bar_beta0)  # [K]
        deriv_beta0 = lambda_bar * ratio_beta0  # [K]

        b_new = self.b_0 + N_k * ratio_lambda + self.beta_0 * deriv_beta0  # [K]

        # Clip to reasonable ranges
        self.log_a.data = torch.log(a_new.clamp(min=0.5, max=1000.0))
        self.log_b.data = torch.log(b_new.clamp(min=0.1, max=1000.0))

        # === 4. Update expansion point λ̄_k (Eq. 28) ===
        # λ̄_k = mode of Gamma(a_k, b_k): (a-1)/b if a>1, else a/b
        lambda_bar_new = torch.where(
            a_new > 1.0,
            (a_new - 1.0) / b_new,
            a_new / b_new
        )
        # CRITICAL: Clip to prevent explosion in next iteration
        self.lambda_bar.data = lambda_bar_new.clamp(min=0.1, max=100.0)

    def compute_kl_divergence_vmf(self, resp: torch.Tensor):
        """
        Compute KL divergences for all vMF variational factors
        Following the variational lower bound

        Returns:
            kl_total: scalar KL divergence
        """
        B = resp.shape[0]

        # 1. KL for mixture weights: KL(Dir(α) || Dir(α_0))
        alpha = torch.exp(self.log_alpha).clamp(min=0.01)
        kl_pi = (
            torch.lgamma(alpha.sum()) - torch.lgamma(self.alpha_0_vec.sum()) -
            torch.sum(torch.lgamma(alpha)) + torch.sum(torch.lgamma(self.alpha_0_vec)) +
            torch.sum((alpha - self.alpha_0_vec) *
                     (torch.digamma(alpha) - torch.digamma(alpha.sum())))
        )

        # 2. KL for concentrations: KL(Gamma(a,b) || Gamma(a_0,b_0))
        a = torch.exp(self.log_a).clamp(min=0.1)
        b = torch.exp(self.log_b).clamp(min=0.1)
        kl_kappa = torch.sum(
            (a - self.a_0) * torch.digamma(a) -
            torch.lgamma(a) + torch.lgamma(self.a_0) +
            self.a_0 * (torch.log(b) - torch.log(self.b_0)) +
            a * (self.b_0 - b) / b
        )

        # 3. KL for mean directions: simplified approximation
        # KL between vMF distributions (approximate)
        beta = torch.exp(self.log_beta).clamp(min=0.01)
        mu = F.normalize(self.means, p=2, dim=1)
        kappa = a / b

        # Approximate KL: β*κ*(1 - μ^T*μ_0)
        mu_dot_m0 = torch.sum(mu * self.m_0, dim=1)  # [K]
        kl_mu = torch.sum(beta * kappa * (1.0 - mu_dot_m0))

        # Total KL (normalized by batch size)
        kl_total = (kl_pi + kl_kappa + kl_mu) / B

        return kl_total

    def forward(self, z: torch.Tensor,):

        # Normalize z to unit sphere for vMF
        z_norm = F.normalize(z, p=2, dim=1)

        # Ensure means are normalized
        self.normalize_means()

        # === CORRECTED E-STEP: Compute responsibilities ===
        resp, log_resp = self.compute_responsibilities(z_norm)

        # === CORRECTED M-STEP: Update posterior parameters ===
        # IMPORTANT: M-step runs every N batches to allow gradient accumulation
        # Running too frequently can cause instability
        if not hasattr(self, '_forward_count'):
            self._forward_count = 0

        if self.training:
            self._forward_count += 1
            # Update every 5 forward passes for stability
            if self._forward_count % 5 == 0:
                with torch.no_grad():
                    try:
                        self.update_posterior_parameters(z_norm, resp)
                    except Exception as e:
                        # If update fails, skip this M-step
                        print(f"Warning: M-step failed with error: {e}")
                        pass

        # Compute log probability (for monitoring)
        log_theta_dp = torch.logsumexp(log_resp, dim=1)
        theta_dp = resp

        # Compute KL divergence for clustering
        theta_dp_kl = self.compute_kl_divergence_vmf(resp)

        batch_size = theta_dp.size(0)
        
        # Microbiome
        rho = self.get_beta(batch_size, self.n_output_microbiome, self.bias_d, self.spike_logit, self.slab_lnvar, self.slab_mean, 
                            self.alpha_mu, self.alpha_log_var, self.beta_mu, self.beta_log_var, 
                            self.zeta_a_mu, self.zeta_a_log_var, self.zeta_b_mu, self.zeta_b_log_var)
        rho_kl = self.sparse_kl_loss(self.spike_logit, self.slab_lnvar, self.slab_mean, 
                            self.alpha_mu, self.alpha_log_var, self.beta_mu, self.beta_log_var, 
                            self.zeta_a_mu, self.zeta_a_log_var, self.zeta_b_mu, self.zeta_b_log_var)
        
        # Metabolite
        rho_me = self.get_beta(batch_size, self.n_output_metabolite, self.bias_d_me, self.spike_logit_me, self.slab_lnvar_me, self.slab_mean_me, 
                            self.alpha_mu_me, self.alpha_log_var_me, self.beta_mu_me, self.beta_log_var_me, 
                            self.zeta_a_mu, self.zeta_a_log_var, self.zeta_b_mu, self.zeta_b_log_var)
        rho_kl_me = self.sparse_kl_loss(self.spike_logit_me, self.slab_lnvar_me, self.slab_mean_me, 
                            self.alpha_mu_me, self.alpha_log_var_me, self.beta_mu_me, self.beta_log_var_me, 
                            self.zeta_a_mu, self.zeta_a_log_var, self.zeta_b_mu, self.zeta_b_log_var)
        
        return rho, rho_kl, rho_me, rho_kl_me, log_theta_dp, theta_dp, theta_dp_kl
    
    def get_rho(self):
        rho = self.get_beta(1)  # Batch size 1 for inspection
        return rho
    
    def get_beta(self, batch_size, n_output,
        bias_d: torch.Tensor,
        spike_logit: torch.Tensor,
        slab_lnvar: torch.Tensor,
        slab_mean: torch.Tensor,
        alpha_mu: torch.Tensor,
        alpha_log_var: torch.Tensor,
        beta_mu: torch.Tensor,
        beta_log_var: torch.Tensor,
        zeta_a_mu: torch.Tensor,
        zeta_a_log_var: torch.Tensor,
        zeta_b_mu: torch.Tensor,
        zeta_b_log_var: torch.Tensor
        ):
        """
        Get samples from the variational posterior of beta (weights).
        Applies the Spike-and-Slab Group Horseshoe prior.
        """
        # Get the node inclusion indicators (z)
        z_sample = self.get_z_sample(spike_logit, hard=False)  # Now [n_clusters, n_output]
        
        # Get horseshoe scale parameters
        tau_tilde_squared, _, _, _, _ = self.get_tau_sample(alpha_mu, alpha_log_var, beta_mu, beta_log_var,
                                                            zeta_a_mu, zeta_a_log_var, zeta_b_mu, zeta_b_log_var)  # Now [n_clusters, n_output]
        
        # Get the scale
        tau_tilde = torch.sqrt(tau_tilde_squared)  # Now [n_clusters, n_output]
        
        # Sample weights from variational posterior
        epsilon = torch.randn(self.n_clusters, n_output, device=slab_mean.device)
        weight_sigma = torch.exp(0.5 * slab_lnvar)
        
        # Apply horseshoe prior scaling to the weight standard deviation
        # No need to expand tau_tilde since it already has the right shape
        scaled_sigma = weight_sigma * tau_tilde
        
        # Sample weights using reparameterization trick
        weight_sample = slab_mean + scaled_sigma * epsilon
        
        # Apply sparsity mask (z multiplication for node-wise sparsity)
        # No need to expand z_sample since it already has the right shape
        sparse_weights = weight_sample * z_sample
        
        # Apply bias adjustment
        result = sparse_weights - bias_d
        
        # If we need to repeat for a batch
        if batch_size > 1:
            # Add a small random noise for each batch element to ensure diversity
            batch_noise = torch.randn(batch_size, self.n_clusters, n_output, 
                                     device=result.device) * 0.01 * scaled_sigma
            result = result.unsqueeze(0) + batch_noise
            
        return result.reshape(batch_size, self.n_clusters, n_output) if batch_size > 1 else result
    
    def get_z_sample(self, 
        spike_logit: torch.Tensor,
        hard=False):
        """
        Generate a sample from the variational distribution of z.
        Uses Gumbel-softmax trick for continuous relaxation.
        
        Args:
            hard: If True, uses hard discretization for forward pass
                 but continuous relaxation for backward pass
        
        Returns:
            z_sample: A sample of node inclusion indicators
        """
        # Generate Gumbel noise
        u = torch.rand_like(spike_logit)
        #gumbel_noise = -torch.log(-torch.log(u + self.eps) + self.eps)
        gumbel_noise = torch.log(u + self.eps) - torch.log(1.0 - u + self.eps)
        
        # Apply logits with temperature
        temp = 0.5  # Temperature for Gumbel-softmax
        logits = spike_logit + gumbel_noise
        y_soft = torch.sigmoid(logits / temp)
        
        # Apply hard discretization if needed
        if hard:
            # Create hard sample
            y_hard = torch.zeros_like(y_soft)
            y_hard[y_soft >= 0.5] = 1.0
            
            # Straight-through estimator
            z_sample = y_hard.detach() - y_soft.detach() + y_soft
        else:
            z_sample = y_soft
        
        return z_sample
    
    def get_tau_sample(self,
        alpha_mu: torch.Tensor,
        alpha_log_var: torch.Tensor,
        beta_mu: torch.Tensor,
        beta_log_var: torch.Tensor,
        zeta_a_mu: torch.Tensor,
        zeta_a_log_var: torch.Tensor,
        zeta_b_mu: torch.Tensor,
        zeta_b_log_var: torch.Tensor
                       ):
        """
        Sample from the variational distributions of the Horseshoe parameters.
        Returns tau_tilde which is the regularized Horseshoe local scale parameter.
        """
        # Sample from log-normal distributions for all parameters
        epsilon_alpha = torch.randn_like(alpha_mu)
        epsilon_beta = torch.randn_like(beta_mu)
        epsilon_zeta_a = torch.randn_like(zeta_a_mu)
        epsilon_zeta_b = torch.randn_like(zeta_b_mu)
        
        # Transform to log-normal
        alpha_sigma = torch.exp(0.5 * alpha_log_var)
        beta_sigma = torch.exp(0.5 * beta_log_var)
        zeta_a_sigma = torch.exp(0.5 * zeta_a_log_var)
        zeta_b_sigma = torch.exp(0.5 * zeta_b_log_var)
        
        log_alpha = alpha_mu + alpha_sigma * epsilon_alpha
        log_beta = beta_mu + beta_sigma * epsilon_beta
        log_zeta_a = zeta_a_mu + zeta_a_sigma * epsilon_zeta_a
        log_zeta_b = zeta_b_mu + zeta_b_sigma * epsilon_zeta_b
        
        alpha = torch.exp(log_alpha)
        beta = torch.exp(log_beta)
        zeta_a = torch.exp(log_zeta_a)
        zeta_b = torch.exp(log_zeta_b)
        
        # Compute tau^2 = beta * alpha (local scale parameter)
        tau_squared = beta * alpha  # Now shape [n_clusters, n_output]
        
        # Compute zeta^2 = zeta_a * zeta_b (global scale parameter)
        zeta_squared = zeta_a * zeta_b  # Shape [1]
        
        # Compute regularized tau_tilde^2 = (c_reg^2 * tau^2) / (c_reg^2 + tau^2 * zeta^2)
        c_reg_squared = self.c_reg ** 2
        # Expand zeta_squared for broadcasting
        expanded_zeta_squared = zeta_squared.expand_as(tau_squared)
        denominator = c_reg_squared + tau_squared * expanded_zeta_squared
        tau_tilde_squared = (c_reg_squared * tau_squared) / denominator
        
        return tau_tilde_squared, alpha, beta, zeta_a, zeta_b
    
    def cluster_weights(self):
        """Return current mixture weights π_k for vMF mixture"""
        with torch.no_grad():
            alpha = torch.exp(self.log_alpha)
            pi = alpha / alpha.sum()
        return pi
    
    def soft_max(self, 
                 z: torch.Tensor,
    ):  
        
        return torch.exp(self.log_softmax(z))
    
    def _kl_gamma_lognormal(self, alpha, beta, mu, log_var):
        """
        Approximate KL divergence between Gamma(alpha, beta) and LogNormal(mu, log_var).
        This is an approximation since the exact KL is not analytically tractable.
        """
        # Convert scalar parameters to tensors with proper device
        alpha_tensor = torch.tensor(alpha, device=mu.device)
        beta_tensor = torch.tensor(beta, device=mu.device)
        
        # Expected log(x) under log-normal
        E_log_x = mu
        
        # Expected x under log-normal
        sigma2 = torch.exp(log_var)
        E_x = torch.exp(mu + sigma2/2)
        
        # KL approximation
        kl = (alpha_tensor - 1) * E_log_x - beta_tensor * E_x + \
             alpha_tensor * torch.log(beta_tensor) - torch.lgamma(alpha_tensor) + \
             0.5 * log_var + 0.5 * torch.log(torch.tensor(2 * math.pi, device=mu.device)) + 0.5
        
        return kl.sum()
    
    def _kl_inverse_gamma_lognormal(self, alpha, beta, mu, log_var):
        """
        Approximate KL divergence between InvGamma(alpha, beta) and LogNormal(mu, log_var).
        This is an approximation since the exact KL is not analytically tractable.
        """
        # Convert scalar parameters to tensors with proper device
        alpha_tensor = torch.tensor(alpha, device=mu.device)
        beta_tensor = torch.tensor(beta, device=mu.device)
        
        # Expected log(x) under log-normal
        E_log_x = mu
        
        # Expected 1/x under log-normal
        sigma2 = torch.exp(log_var)
        E_inv_x = torch.exp(-mu + sigma2/2)
        
        # KL approximation
        kl = -(alpha_tensor + 1) * E_log_x - beta_tensor * E_inv_x - \
             alpha_tensor * torch.log(beta_tensor) + torch.lgamma(alpha_tensor) + \
             0.5 * log_var + 0.5 * torch.log(torch.tensor(2 * math.pi, device=mu.device)) + 0.5
        
        return kl.sum()
    
    def sparse_kl_loss(self,
        spike_logit: torch.Tensor,
        slab_lnvar: torch.Tensor,
        slab_mean: torch.Tensor,
        alpha_mu: torch.Tensor,
        alpha_log_var: torch.Tensor,
        beta_mu: torch.Tensor,
        beta_log_var: torch.Tensor,
        zeta_a_mu: torch.Tensor,
        zeta_a_log_var: torch.Tensor,
        zeta_b_mu: torch.Tensor,
        zeta_b_log_var: torch.Tensor
                       ):
        
        """
        Calculate the KL divergence between the variational posterior and the prior
        for the SS-GHS model.
        """
        # KL for node inclusion indicators z
        gamma = torch.sigmoid(spike_logit)
        prior_prob = torch.tensor(self.prior_inclusion_prob, device=gamma.device)
        
        # Add numerical stability
        eps = 1e-10
        gamma_safe = torch.clamp(gamma, eps, 1.0 - eps)
        prior_prob_safe = torch.clamp(prior_prob, eps, 1.0 - eps)
        
        kl_z = gamma_safe * torch.log(gamma_safe / prior_prob_safe) + \
               (1 - gamma_safe) * torch.log((1 - gamma_safe) / (1 - prior_prob_safe))
        kl_z = torch.clamp(kl_z.sum(), 0, 1e5)
        
        # Get horseshoe scale parameters
        tau_tilde_squared, alpha, beta, zeta_a, zeta_b = self.get_tau_sample(alpha_mu, alpha_log_var, beta_mu, beta_log_var,
                                                            zeta_a_mu, zeta_a_log_var, zeta_b_mu, zeta_b_log_var)
        
        # KL for weights under horseshoe prior
        z_sample = self.get_z_sample(spike_logit, hard=False)
        
        # Prior std with scale from Horseshoe
        tau_tilde = torch.sqrt(tau_tilde_squared)  # Now shape [n_clusters, n_output]
        prior_sigma = self.sigma0 * tau_tilde  # Now shape [n_clusters, n_output]
        
        # Standard deviation of the variational distribution
        weight_sigma = torch.exp(0.5 * slab_lnvar)  # [n_clusters, n_output]
        
        # Ensure numerical stability
        prior_sigma = torch.clamp(prior_sigma, min=1e-6, max=1e6)
        weight_sigma = torch.clamp(weight_sigma, min=1e-6, max=1e6)
        
        # No need to expand dimensions for broadcasting since they already match
        # KL divergence between two Gaussians with numerical stability
        weight_var_ratio = torch.clamp(weight_sigma / prior_sigma, min=1e-10, max=1e10)
        mean_square_scaled = torch.clamp(slab_mean**2 / prior_sigma**2, min=0, max=1e6)
        weight_var_scaled = torch.clamp(weight_sigma**2 / prior_sigma**2, min=0, max=1e6)
        
        kl_weights = 0.5 * (
            2 * torch.log(weight_var_ratio) + 
            mean_square_scaled + weight_var_scaled - 1
        )
        
        # Only include KL for nodes that are active (z=1)
        kl_weights = torch.clamp((kl_weights * z_sample).sum(), 0, 1e5)
        
        # KL for Horseshoe variational components
        # KL for alpha ~ G(1/2, 1) vs LogNormal
        kl_alpha = torch.clamp(self._kl_gamma_lognormal(0.5, 1.0, alpha_mu, alpha_log_var), 0, 1e5)
        
        # KL for beta ~ IG(1/2, 1) vs LogNormal
        kl_beta = torch.clamp(self._kl_inverse_gamma_lognormal(0.5, 1.0, beta_mu, beta_log_var), 0, 1e5)
        
        # KL for zeta_a ~ G(1/2, d0^2) vs LogNormal
        kl_zeta_a = torch.clamp(self._kl_gamma_lognormal(0.5, self.d0**2, zeta_a_mu, zeta_a_log_var), 0, 1e5)
        
        # KL for zeta_b ~ IG(1/2, 1) vs LogNormal
        kl_zeta_b = torch.clamp(self._kl_inverse_gamma_lognormal(0.5, 1.0, zeta_b_mu, zeta_b_log_var), 0, 1e5)
        
        # Sum all KL terms
        kl_total = kl_z + kl_weights + kl_alpha + kl_beta + kl_zeta_a + kl_zeta_b
        
        return kl_total 

class DBayesCM(pl.LightningModule):
    
    def __init__(
            self,
            n_genes: int,
            n_metabolite: int,
            n_latent: int = 32,
            n_clusters: int = 10,
            n_layers_encoder_individual: int = 2,
            dim_hidden_encoder: int = 128,
            alpha_0=0.01,
            pip0_rho: float = 0.1,
            kl_weight_beta: float = 1.0,
            learning_rate: float = 1e-3,
            c_reg: float = 1.0,
            d0: float = 1.0, 
            sigma0: float = 1.0,
            n_samples: int = 1000,
            use_log_transform: bool = True,
            use_LayerNorm: bool = False,
            combine_method: str = "add",
        ):
        
        super().__init__()
        
        self.n_input = n_genes
        self.n_input_me = n_metabolite
        self.n_latent = n_latent
        self.n_clusters = n_clusters
        self.alpha_0 = alpha_0
        self.pip0_rho = pip0_rho
        self.kl_weight_beta = kl_weight_beta
        self.c_reg = c_reg
        self.d0 = d0
        self.sigma0 = sigma0
        self.use_log_transform = use_log_transform
        self.use_LayerNorm = use_LayerNorm
        
        self.z_encoder = EncoderDBGCM(
            n_input_microbiome=self.n_input,
            n_input_metabolite=self.n_input_me,
            n_latent=self.n_latent,
            n_hidden=dim_hidden_encoder,
            n_layers=n_layers_encoder_individual,
            use_LayerNorm=self.use_LayerNorm,
            combine_method=combine_method
        )

        # Use vMF mixture decoder with SS-GH for feature sparsity
        self.decoder = DecoderDBGCM_vMF(
            n_input=self.n_latent,
            n_clusters=self.n_clusters,
            n_output_microbiome=self.n_input,
            n_output_metabolite=self.n_input_me,
            pip0=self.pip0_rho,
            alpha_0=self.alpha_0,  # Small Dirichlet prior for automatic pruning
            c_reg=self.c_reg,
            d0=self.d0,
            sigma0=self.sigma0,
        )
        
        
        # Buffers to store epoch-level losses
        self.train_loss_history = []
        self.val_loss_history = []
        self.ari_history = []
        self.train_ari_history = []
        
        self.val_theta_all_epochs = []
        self.train_theta_all_epochs = []
        
        # Microbiome
        self.val_rho_all_epochs = []
        self.val_z_sample_all_epochs = []
        self.train_rho_all_epochs = []     
        self.train_z_sample_all_epochs = []
        
        # Metabolite
        self.val_rho_me_all_epochs = []
        self.val_z_sample_me_all_epochs = []
        self.train_rho_me_all_epochs = []
        self.train_z_sample_me_all_epochs = []

        # -------------------------------------------------------------------
        # vMF GEOMETRY stored PER EPOCH (for visualisation at the best ARI).
        # Just like val_theta_all_epochs, index these at the best-ARI epoch to
        # recover EXACTLY the geometry that produced the highest ARI -- no model
        # re-run, no re-sampling.
        #   encoder    : mu (mean direction), kappa (concentration), z (sample)
        #   clustering : centres (mu_k, unit), cluster_kappa (E[kappa_k])
        # -------------------------------------------------------------------
        self.val_mu_all_epochs = []            # [epoch] -> [N, D]  encoder mean dir
        self.val_kappa_all_epochs = []         # [epoch] -> [N]     encoder concentration
        self.val_z_all_epochs = []             # [epoch] -> [N, D]  on-sphere sample
        self.val_centres_all_epochs = []       # [epoch] -> [K, D]  vMF cluster centres
        self.val_cluster_kappa_all_epochs = [] # [epoch] -> [K]     per-cluster kappa
        self.val_labels_all_epochs = []        # [epoch] -> [N]     true labels (aligned)

        # We'll store per-batch losses each epoch in these, then reset each epoch
        self._train_losses_epoch = []
        self._val_losses_epoch = []
        
        self.learning_rate = learning_rate
        self.n_samples = n_samples
        
    ############################################################################
    #                               Core Methods                               #
    ############################################################################
        
    def dir_llik(self, 
                 xx: torch.Tensor, 
                 aa: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Compute the Dirichlet log-likelihood.
        '''
        reconstruction_loss = None 
        
        term1 = (torch.lgamma(torch.sum(aa, dim=-1)) -
                torch.lgamma(torch.sum(aa + xx, dim=-1))) #[n_batch]
        term2 = torch.sum(torch.where(xx > 0,
                            torch.lgamma(aa + xx) -
                            torch.lgamma(aa),
                            torch.zeros_like(xx)),
                            dim=-1) #[n_batch
        reconstruction_loss = term1 + term2 #[n_batch
        return reconstruction_loss
    
    def inference(self, x: torch.Tensor) -> dict:
        x_ = torch.log(1 + x)
        qz_mu, qz_kappa, z = self.z_encoder(x_)
        return dict(qz_mu=qz_mu, qz_kappa=qz_kappa, z=z)
    
    def generative(self, z) -> dict:

        rho, rho_kl, log_theta_dp, theta_dp, theta_dp_kl  = self.decoder(z)

        return dict(rho = rho, rho_kl = rho_kl, theta_dp = theta_dp)
    
    def get_decoder_outputs(self, x: torch.Tensor, y: torch.Tensor):
        """
        Runs one forward pass from raw input `x`, `y` => (rho, rho_me, theta).

        WARNING: For big datasets, storing these for all samples
        can be huge in memory.
        You may want to only do this on a subset or do mini-batch collection.
        """
        if self.use_log_transform:
            x_ = torch.log1p(x)
            y_ = torch.log1p(y)
        else:
            x_ = x
            y_ = y

        qz_mu, qz_kappa, z = self.z_encoder(x_, y_)
        # vMF decoder normalizes z internally (z is already on unit sphere)
        rho, rho_kl, rho_me, rho_kl_me, log_theta_dp, theta_dp, theta_dp_kl = self.decoder(z)
        
        # If rho is 3D (batch_size, n_clusters, n_output), average across batch dimension
        if rho.dim() == 3:
            rho = rho.mean(dim=0)  # Average to get [n_clusters, n_output]
            
        if rho_me.dim() == 3:
            rho_me = rho_me.mean(dim=0)
            
        return rho, rho_me, theta_dp
    
    def sample_from_posterior_z(
        self, 
        x: torch.Tensor,
        deterministic: bool = True,
        output_softmax_z: bool = True, 
    ):
        """Sample from the posterior z
        """
        inference_out = self.inference(x)
        if deterministic:
            z = inference_out["qz_mu"]
        else:
            z = inference_out["z"]
        if output_softmax_z:
            generative_outputs = self.generative(z)
            z = generative_outputs["theta_dp"]      
        return dict(z=z)
    
    def forward(self, x):
        x_ = torch.log(1 + x)
        qz_mu, qz_kappa, z = self.z_encoder(x_)
        return z
    
    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        
        inference_out = self.inference(x)
        z = inference_out["z"]
        gen_out = self.generative(z)
        theta = gen_out["theta_dp"]
               
        rho = gen_out["rho"]
        log_aa = torch.clamp(torch.mm(theta, rho), -10, 10)
        aa = torch.exp(log_aa)
        
        reconstruction_loss = -self.dir_llik(x, aa)

        return reconstruction_loss
    
    def compute_loss(self, x, y, kl_weight=1.0):
        
        # 1) Preprocessing and normalization
        
        if self.use_log_transform:
            x_ = torch.log1p(x)  # More stable than log(1+x)
            y_ = torch.log1p(y)
        else:
            x_ = x
            y_ = y
            
        x_ = torch.clamp(x_, min=0.0)  # Ensure non-negative
        y_ = torch.clamp(y_, min=0.0)

        # 2) Forward pass through vMF encoder
        qz_mu, qz_kappa, z = self.z_encoder(x_, y_)

        # CRITICAL: vMF decoder normalizes z internally to unit sphere
        # This is different from Gaussian mixture which uses unnormalized z

        # 3) Forward pass through vMF decoder
        rho, rho_kl, rho_me, rho_kl_me, log_theta_dp, theta_dp, theta_dp_kl = self.decoder(z)
        kl_weight_beta = self.kl_weight_beta
        
        # 4) Reconstruction losses
        #    Dirichlet multinominal for microbiome
        # Handle case where rho is 3D tensor (batch_size, n_clusters, n_output)
        if rho.dim() == 3:
            # For each sample in batch, compute mean log_aa across the batch dimension
            batch_size = rho.size(0)
            log_aa_list = []
            
            for i in range(batch_size):
                curr_theta = theta_dp[i:i+1]  # Keep dimension [1, n_clusters]
                curr_rho = rho[i]  # [n_clusters, n_output]
                curr_log_aa = torch.clamp(torch.mm(curr_theta, curr_rho), -10, 10)
                log_aa_list.append(curr_log_aa)
                
            log_aa = torch.cat(log_aa_list, dim=0)
        else:
            # Original case for 2D rho
            log_aa = torch.clamp(torch.mm(theta_dp, rho), -10, 10)
            
        aa = torch.exp(log_aa)
        reconstruction_loss = -self.dir_llik(x_, aa)
        
        #    Dirichlet multinominal for metabolite
        if rho_me.dim() == 3:
            # For each sample in batch, compute mean log_aa across the batch dimension
            batch_size = rho_me.size(0)
            log_aa_list = []
            
            for i in range(batch_size):
                curr_theta = theta_dp[i:i+1]  # Keep dimension [1, n_clusters]
                curr_rho = rho_me[i]  # [n_clusters, n_output]
                curr_log_aa = torch.clamp(torch.mm(curr_theta, curr_rho), -10, 10)
                log_aa_list.append(curr_log_aa)
                
            log_aa_me = torch.cat(log_aa_list, dim=0)
        else:
            # Original case for 2D rho
            log_aa_me = torch.clamp(torch.mm(theta_dp, rho_me), -10, 10)
            
        aa_me = torch.exp(log_aa_me)
        reconstruction_loss_me = -self.dir_llik(y_, aa_me)
        
        # 5) KL divergences
        # vMF: KL( vMF(mu, kappa) || U(S^{D-1}) ) — exact, with analytic gradient w.r.t. kappa
        kl_divergence_z = vmf_kl_divergence(qz_kappa, self.n_latent)   # [B]
        
        kl_divergence_beta = rho_kl
        kl_divergence_beta_me = rho_kl_me
        kl_divergence_theta_dp = theta_dp_kl
        kl_local = kl_divergence_z
        
        # Scale KL divergence terms to balance them with reconstruction loss
        loss = (torch.mean(reconstruction_loss + reconstruction_loss_me + kl_weight * kl_local) 
                + kl_weight_beta * kl_divergence_beta/(10.0 * x.shape[1])
                + kl_weight_beta * kl_divergence_beta_me/y.shape[1]
                + kl_divergence_theta_dp * 0.001)
        
        return loss
    
    ############################################################################
    #                         Training/Validation Steps                        #
    ############################################################################
    
    def on_train_epoch_start(self):
        """Clear the list of per-batch losses for this new epoch."""
        
        self._train_losses_epoch.clear()
        self._train_theta_list = []
        self._train_labels_epoch = []
        
        #Microbiome
        self._train_rho_list = []
        self._train_z_sample_list = []
        
        # Metabolite
        self._train_rho_me_list = []
        self._train_z_sample_me_list = []
    
    def training_step(self, batch, batch_idx):
        """
        Per-batch training step.
        """
        x = batch["Microbiome"]
        y = batch["Metabolite"]
        
        ground_true = batch["Labels"]

        loss = self.compute_loss(x, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # Store in our list so we can average at epoch's end
        self._train_losses_epoch.append(loss.detach())
        
        # Collect data for ARI calculation
        with torch.no_grad():
            rho, rho_me, theta = self.get_decoder_outputs(x, y)
            z_sample = torch.sigmoid(self.decoder.spike_logit)
            z_sample_me = torch.sigmoid(self.decoder.spike_logit_me)
        
        # Store them in temporary lists
        self._train_theta_list.append(theta.detach().cpu())
        self._train_labels_epoch.append(ground_true.detach().cpu())
        
        # Microbiome
        self._train_rho_list.append(rho.detach().cpu())
        self._train_z_sample_list.append(z_sample.detach().cpu())
        
        # Metabolite
        self._train_rho_me_list.append(rho_me.detach().cpu())
        self._train_z_sample_me_list.append(z_sample_me.detach().cpu())
        
        return loss
    
    def on_train_epoch_end(self):
        """Compute the average loss for this epoch and store it."""
        epoch_loss = torch.stack(self._train_losses_epoch).mean().item()
        self.train_loss_history.append(epoch_loss)
        print(f"Epoch {self.current_epoch} Train Loss: {epoch_loss:.4f}")
        
        # Process collected training data for ARI
        if len(self._train_rho_list) > 0:
            
            all_rho = torch.cat(self._train_rho_list, dim=0)         
            all_theta = torch.cat(self._train_theta_list, dim=0)
            all_labels = torch.cat(self._train_labels_epoch, dim=0)
            all_z_sample = torch.stack(self._train_z_sample_list, dim=0)        
            all_rho_me = torch.cat(self._train_rho_me_list, dim=0)
            all_z_sample_me = torch.stack(self._train_z_sample_me_list, dim=0)
            
            # Hard cluster assignment
            pred_labels = torch.argmax(all_theta, dim=1)
            
            # Compute ARI
            all_labels_np = all_labels.numpy()
            pred_labels_np = pred_labels.numpy()
            ari = adjusted_rand_score(all_labels_np, pred_labels_np)
            self.train_ari_history.append(ari)
            self.train_rho_all_epochs.append(all_rho)
            self.train_theta_all_epochs.append(all_theta)
            self.train_z_sample_all_epochs.append(all_z_sample)        
            self.train_rho_me_all_epochs.append(all_rho_me)
            self.train_z_sample_me_all_epochs.append(all_z_sample_me)
            
            print(f"Epoch {self.current_epoch} Train ARI: {ari:.4f}")
        
    def on_validation_epoch_start(self):

        self._val_losses_epoch.clear()
        self._val_rho_list = []
        self._val_theta_list = []
        self._val_labels_epoch = []
        self._val_z_sample_list = []

        self._val_rho_me_list = []
        self._val_z_sample_me_list = []

        # per-batch encoder geometry (mu, kappa, z) for this epoch
        self._val_mu_list = []
        self._val_kappa_list = []
        self._val_z_list = []
    
    def validation_step(self, batch, batch_idx):
        """
        Per-batch validation step.
        """
        
        x = batch["Microbiome"]
        y = batch["Metabolite"]
        
        ground_true = batch["Labels"]

        loss = self.compute_loss(x, y)
        
        # Append batch loss so we can average it later
        self._val_losses_epoch.append(loss.detach())
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        
        # Now get the raw decoder outputs
        with torch.no_grad():
            rho, rho_me, theta = self.get_decoder_outputs(x, y)
            z_sample = torch.sigmoid(self.decoder.spike_logit)
            z_sample_me = torch.sigmoid(self.decoder.spike_logit_me)

            # --- vMF ENCODER geometry for this batch (mu, kappa, z) ---
            # same preprocessing the encoder sees inside get_decoder_outputs.
            # RNG-NEUTRAL: the encoder's vMF sampling consumes randomness, which
            # would shift the training RNG stream and change results vs a run
            # without geometry capture.  Save and restore the RNG state so this
            # extra forward pass consumes ZERO net randomness -> identical results
            # to the original (unmodified validation_step) that produced ARI 0.51.
            if self.use_log_transform:
                x_enc = torch.log1p(x); y_enc = torch.log1p(y)
            else:
                x_enc, y_enc = x, y
            _rng_state = torch.get_rng_state()
            qz_mu, qz_kappa, qz_z = self.z_encoder(x_enc, y_enc)
            torch.set_rng_state(_rng_state)

        # Store them in temporary lists so we can examine them in on_validation_epoch_end
        self._val_rho_list.append(rho.detach().cpu())
        self._val_theta_list.append(theta.detach().cpu())
        self._val_labels_epoch.append(ground_true.detach().cpu())
        self._val_z_sample_list.append(z_sample.detach().cpu())
        self._val_rho_me_list.append(rho_me.detach().cpu())
        self._val_z_sample_me_list.append(z_sample_me.detach().cpu())

        # encoder geometry (mu mean direction, kappa concentration, z sample)
        self._val_mu_list.append(qz_mu.detach().cpu())
        self._val_kappa_list.append(qz_kappa.detach().cpu())
        self._val_z_list.append(qz_z.detach().cpu())

        return loss
    
    def on_validation_epoch_end(self):
        if len(self._val_losses_epoch) > 0:
            # Compute epoch mean
            epoch_loss = torch.stack(self._val_losses_epoch).mean().item()
            self.val_loss_history.append(epoch_loss)
            print(f"Epoch {self.current_epoch} Val Loss: {epoch_loss:.4f}")
            
        # Now handle the stored rho, rho_kl, theta for the entire epoch:
        if len(self._val_rho_list) > 0:
            all_rho = torch.cat(self._val_rho_list, dim=0)         
            all_theta = torch.cat(self._val_theta_list, dim=0)
            all_labels = torch.cat(self._val_labels_epoch, dim=0)
            all_z_sample = torch.stack(self._val_z_sample_list, dim=0)      
            all_rho_me = torch.cat(self._val_rho_me_list, dim=0)
            all_z_sample_me = torch.stack(self._val_z_sample_me_list, dim=0)
            
            # Hard cluster assignment
            pred_labels = torch.argmax(all_theta, dim=1)
            
            # Compute ARI
            all_labels_np = all_labels.numpy()
            pred_labels_np = pred_labels.numpy()
            ari = adjusted_rand_score(all_labels_np, pred_labels_np)
            self.ari_history.append(ari)
            self.val_rho_all_epochs.append(all_rho)
            self.val_theta_all_epochs.append(all_theta)
            self.val_z_sample_all_epochs.append(all_z_sample)
            self.val_rho_me_all_epochs.append(all_rho_me)
            self.val_z_sample_me_all_epochs.append(all_z_sample_me)

            # -----------------------------------------------------------------
            # Store the vMF GEOMETRY for THIS epoch, aligned 1-to-1 with
            # val_theta_all_epochs / ari_history.  Indexing these at the best-ARI
            # epoch gives EXACTLY the geometry behind the highest ARI (the same
            # theta that scored it), ready for visualisation with no re-run.
            # -----------------------------------------------------------------
            all_mu = torch.cat(self._val_mu_list, dim=0)        # [N, D]
            all_kappa = torch.cat(self._val_kappa_list, dim=0)  # [N]
            all_z = torch.cat(self._val_z_list, dim=0)          # [N, D]
            self.val_mu_all_epochs.append(all_mu)
            self.val_kappa_all_epochs.append(all_kappa)
            self.val_z_all_epochs.append(all_z)
            self.val_labels_all_epochs.append(all_labels)       # true labels, same order

            # cluster geometry of the vMF mixture at this epoch
            with torch.no_grad():
                centres = F.normalize(self.decoder.means, p=2, dim=1).detach().cpu()  # [K, D]
                a = torch.exp(self.decoder.log_a).clamp(min=0.1)
                b = torch.exp(self.decoder.log_b).clamp(min=0.1)
                cluster_kappa = (a / b).detach().cpu()                               # [K]  E[kappa_k]
            self.val_centres_all_epochs.append(centres)
            self.val_cluster_kappa_all_epochs.append(cluster_kappa)

            print(f"Epoch {self.current_epoch} ARI: {ari:.4f}")
            
    def collect_decoder_outputs_for_dataset(self, data_loader):
        """
        Run through an entire dataset loader, storing (rho, rho_kl, theta) for each batch.
        Returns concatenated results (watch memory usage).
        """
        self.eval()
        all_rho = []
        all_theta = []
        all_rho_me = []

        with torch.no_grad():
            for batch in data_loader:
                # If your DataLoader yields dict, do:
                x = batch["Microbiome"]
                y = batch["Metabolite"]
                rho, rho_me, theta = self.get_decoder_outputs(x, y)
                #all_rho.append(rho.cpu())
                all_rho.append(rho.cpu().unsqueeze(0))
                all_rho_me.append(rho_me.cpu().unsqueeze(0))
                all_theta.append(theta.cpu())
    
        return (
            torch.cat(all_rho, dim=0),
            torch.cat(all_theta, dim=0),
            torch.cat(all_rho_me, dim=0)
            )

    def configure_optimizers(self):
        """
        Define how to optimize: commonly Adam or SGD.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
class MicrobiomeDataset(Dataset):
    def __init__(self, X, Y, ground_true):
        super().__init__()
        self.X = X  # This is a torch tensor of shape [num_samples, ...]
        self.Y = Y
        self.ground_true = ground_true
        
        # Optionally, you can assert they have matching first dimension
        assert len(self.X) == len(self.Y)
        assert len(self.X) == len(self.ground_true)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Return a dict with the "Microbiome" and Metabolite key
        return {
            "Microbiome": self.X[idx],
            "Metabolite": self.Y[idx],
            "Labels": self.ground_true[idx]
        }





























