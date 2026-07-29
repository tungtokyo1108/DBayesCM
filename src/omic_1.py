#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2026-06-17

@author: tungbioinfo

================================================================================
 ver3 : Hyperspherical (S-VAE) variant of DBayGenCM
================================================================================

This file is a drop-in *ver3* counterpart to
    DBayGenCM_1omics_v2.2_diagonal_SS-GH_sigmod.py   (the Gaussian "N-VAE")

It replaces the Gaussian variational posterior  q(z|x) = N(mu, sigma^2 I)
of the encoder with a **von Mises-Fisher (vMF)** posterior on the unit
hypersphere  S^{m-1},  following

    Davidson, Falorsi, De Cao, Kipf, Tomczak (2018)
    "Hyperspherical Variational Auto-Encoders" (UAI 2018).

Why this should improve the encoder for this model
--------------------------------------------------
* The original model clusters the latent code z with a (truncated) Dirichlet
  Process Gaussian Mixture. A Gaussian prior N(0, I) has "origin gravity":
  it pulls every cluster centre toward 0, which is exactly the failure mode
  the paper describes for clustered data (Sec. 2.2). The vMF posterior with a
  *uniform* prior on the sphere removes this pull, letting clusters spread out
  evenly and become more separable (the paper's Tables 2 & 3 show large gains
  in clusterability / semi-supervised accuracy in low dimensions).
* On the sphere the natural similarity is cosine similarity, which is a more
  meaningful distance in moderate/high dimension than the L2 norm.

What was changed relative to v2.2 (only the encoder side is touched)
--------------------------------------------------------------------
  1. `VonMisesFisher`           : differentiable vMF sample + KL-to-uniform,
                                  Bessel functions via exponentially-scaled
                                  modified Bessel (no scipy needed at runtime).
  2. `HouseholderRotation`      : rotates e_1 -> mu  (Ulrich 1984, Alg. 1).
  3. `EncoderHyperspherical`    : outputs (mu_dir, kappa) and an on-sphere z.
  4. `DecoderDBGCM`             : adds an OPTIONAL mixture-of-vMF latent
                                  clustering (`latent_distribution="vmf"`) so
                                  the DP mixture lives on the same sphere as
                                  the posterior. The Gaussian DP path is kept
                                  for ablation (`latent_distribution="normal"`).
  5. `DBayGenCM`                : KL(vMF || Uniform) replaces KL(N || N);
                                  everything downstream (spike-and-slab +
                                  regularized horseshoe decoder, Dirichlet-
                                  multinomial likelihood) is UNCHANGED so the
                                  comparison against v2.2 is clean.

The decoder (spike-and-slab + regularized horseshoe + Dirichlet-multinomial)
and the training / evaluation scaffolding are kept identical to v2.2 so that
any change in ARI / log-likelihood is attributable to the encoder.

Run side by side with v2.2 to reproduce the N-VAE vs S-VAE comparison of the
paper on your own microbiome data.
"""

import collections
import math
import warnings
from collections.abc import Iterable

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

import lightning.pytorch as pl
from lightning.pytorch.callbacks import RichProgressBar

from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


# =============================================================================
#  Small helpers (shared with v2.2)
# =============================================================================
def _identity(x):
    return x


def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()


def stick_breaking(v: torch.Tensor):
    """Stick-breaking construction for a (truncated) Dirichlet Process."""
    cumprod_v = torch.cumprod(1 - v, dim=0)
    cumprod_v = torch.cat([torch.ones(1, device=v.device), cumprod_v[:-1]], dim=0)
    pi = v * cumprod_v
    return pi


# =============================================================================
#  Modified Bessel functions of the first kind (exponentially scaled)
# =============================================================================
#  We need  I_v(kappa)  and ratios  I_{v}(k) / I_{v-1}(k)  to compute both the
#  vMF normalising constant C_m(kappa) and the KL divergence to the uniform
#  distribution.  For numerical stability we work with the *exponentially
#  scaled* Bessel function  ive(v, k) = I_v(k) * exp(-k)  so that the exp(kappa)
#  in C_m(kappa) cancels and we never overflow for large kappa.
#
#  PyTorch ships `torch.special.i0e` / `i1e` (orders 0 and 1).  For the general
#  half-integer order m/2 - 1 that the vMF needs we combine two differentiable
#  branches, both expressed for the exponentially scaled  ive(v, k) = I_v(k) e^{-k}:
#
#    * small/medium k  : the ascending power series of I_v(k) in log space.
#    * large k OR large order : the UNIFORM (Debye) asymptotic expansion, which
#      stays accurate even when the order v is large relative to k.  This matters
#      a lot here because the vMF order is m/2 - 1, so for a 64-dim latent the
#      order is ~31 and the naive large-argument Hankel series (valid only for
#      k >> v^2) would be wrong.  The uniform expansion is valid for large v and
#      any k, which is exactly the operating regime of a high-dim S-VAE.
#
#  The crossover is chosen so each branch is used only where it is accurate, and
#  the whole thing is differentiable w.r.t. k (no scipy in the training loop).
# =============================================================================
def _log_ive_series(v: float, k: torch.Tensor, n_terms: int = 40) -> torch.Tensor:
    """log( I_v(k) e^{-k} ) via the ascending power series (good for small k).

    The two lgamma terms depend only on the (integer) summation index j and the
    fixed order v -- they are CONSTANTS w.r.t. kappa.  We evaluate them once with
    Python's math.lgamma and ship them as a plain constant tensor (also avoids a
    torch.lgamma call, which some CUDA/Windows builds JIT-compile via nvrtc).
    """
    half_log_k = torch.log(k / 2.0)                                   # [B]
    j = torch.arange(0, n_terms, device=k.device, dtype=k.dtype)      # [n_terms]
    # constant offset  c_j = -lgamma(j+1) - lgamma(v+j+1)
    c = torch.tensor(
        [-math.lgamma(int(jj) + 1.0) - math.lgamma(v + int(jj) + 1.0) for jj in range(n_terms)],
        device=k.device,
        dtype=k.dtype,
    )                                                                # [n_terms]
    log_terms = (2.0 * j + v).unsqueeze(0) * half_log_k.unsqueeze(-1) + c.unsqueeze(0)
    return torch.logsumexp(log_terms, dim=-1) - k


def _log_ive_uniform(v: float, k: torch.Tensor) -> torch.Tensor:
    """
    log( I_v(k) e^{-k} ) via the uniform (Debye) asymptotic expansion
    (DLMF 10.41). Accurate for large order v and/or large argument k:

        I_v(k) ~ 1/sqrt(2 pi) * exp(v eta) / (v^{1/2} (1+t^2)^{1/4}) * U(p)
    with  t = k/v,  eta = sqrt(1+t^2) + log( t / (1 + sqrt(1+t^2)) ),
          p = 1/sqrt(1+t^2),  and the first U-polynomial corrections.
    We subtract k to get the exponentially scaled form.
    """
    v_t = torch.as_tensor(v, device=k.device, dtype=k.dtype)
    t = k / v_t
    s = torch.sqrt(1.0 + t * t)
    eta = s + torch.log(t / (1.0 + s))
    # leading term of log I_v(k)
    log_lead = (
        -0.5 * math.log(2.0 * math.pi)
        + v_t * eta
        - 0.5 * torch.log(v_t)
        - 0.25 * torch.log(1.0 + t * t)
    )
    # U-polynomial correction:  1 + u1(p)/v + u2(p)/v^2 + ...
    p = 1.0 / s
    p2 = p * p
    u1 = (3.0 * p - 5.0 * p * p2) / 24.0
    u2 = (81.0 * p2 - 462.0 * p2 * p2 + 385.0 * p2 * p2 * p2) / 1152.0
    corr = 1.0 + u1 / v_t + u2 / (v_t * v_t)
    corr = corr.clamp(min=1e-12)
    return log_lead + torch.log(corr) - k


def log_ive(v: float, k: torch.Tensor) -> torch.Tensor:
    """
    log of the exponentially scaled modified Bessel function  log( I_v(k) e^{-k} ),
    differentiable in k for k > 0 and real order v >= 0.

    Uses the power series where it converges cheaply (k small relative to the
    order) and the uniform asymptotic expansion otherwise.  The threshold scales
    with the order so high-dim vMFs (large order) are handled correctly.
    """
    k = k.clamp(min=1e-8)
    # The series needs ~k terms to converge; switch to the uniform expansion once
    # k grows past a small multiple of max(v, 1).  This keeps the series in its
    # cheap, accurate regime and hands large-(k or v) cases to the Debye form.
    thresh = max(2.0 * v, 12.0)
    log_series = _log_ive_series(v, k)
    log_unif = _log_ive_uniform(v, k)
    use_unif = (k > thresh).to(k.dtype)
    return use_unif * log_unif + (1.0 - use_unif) * log_series


def ive_ratio(v: float, k: torch.Tensor) -> torch.Tensor:
    """
    Stable ratio  I_v(k) / I_{v-1}(k) = ive(v,k)/ive(v-1,k)  in (0, 1).

    Used for the gradient of the KL term (Eq. 6 of the paper involves these
    ratios) and as the spherical-mean-resultant-length r = I_{m/2}/I_{m/2-1}.
    """
    return torch.exp(log_ive(v, k) - log_ive(v - 1.0, k))


# =============================================================================
#  Spherical k-means  (warm start for the mixture-of-vMF, analogous to the
#  k-means init that scikit-learn's BayesianGaussianMixture uses, but in the
#  correct geometry for a unit-norm latent).
# =============================================================================
#  scikit-learn warm-starts `BayesianGaussianMixture` by running plain k-means
#  on the data and seeding the responsibilities from the result (see
#  sklearn/mixture/_bayesian_mixture.py, init_params="kmeans").  That breaks the
#  symmetry of the variational mixture and avoids the poor local optima you get
#  from a near-uniform random start.
#
#  Our cluster model (`DecoderDBGCM.MixtureOfVMF`) is a mixture of vMFs whose
#  log-likelihood is driven by COSINE similarity kappa_k * (c_k . z) on the unit
#  sphere, NOT L2 distance.  So the correct analogue is *spherical* k-means,
#  which minimises (1 - cos) and yields unit-norm centroids -- exactly the
#  parameterisation of `self.means`.  We additionally recover a per-cluster
#  concentration kappa_k from the mean resultant length r_k via the standard vMF
#  moment estimator r = I_{m/2}(k)/I_{m/2-1}(k), inverted with `ive_ratio`.
# =============================================================================
def _estimate_vmf_kappa(r: torch.Tensor, m: int,
                        kappa_min: float = 1e-2, kappa_max: float = 1e4) -> torch.Tensor:
    """
    Invert the vMF mean-resultant-length relation  r = I_{m/2}(k)/I_{m/2-1}(k)
    to get a moment estimate of kappa for each cluster.

    Banerjee et al. (2005) give the closed-form approximation
        kappa ~ r (m - r^2) / (1 - r^2),
    which we then polish with a few Newton steps on  f(k) = ive_ratio(m/2,k) - r
    (the exact relation), using `ive_ratio` so it stays valid in high dimension.
    """
    r = r.clamp(1e-4, 1.0 - 1e-4)
    order = m / 2.0
    # Banerjee closed-form initial guess
    k = r * (m - r * r) / (1.0 - r * r)
    k = k.clamp(kappa_min, kappa_max)
    # a few Newton refinements on the exact relation ive_ratio(order, k) = r.
    # This root-find is self-contained, so we locally force grad ON even if the
    # caller is inside a @torch.no_grad() block (kmeans_warm_start is).
    with torch.enable_grad():
        for _ in range(5):
            k_var = k.detach().clone().requires_grad_(True)
            f = ive_ratio(order, k_var) - r
            (grad,) = torch.autograd.grad(f.sum(), k_var)
            step = f.detach() / (grad.detach() + 1e-8)
            k = (k - step).clamp(kappa_min, kappa_max)
    return k.detach()


def spherical_kmeans(z: torch.Tensor, n_clusters: int, n_iter: int = 50,
                     n_init: int = 5, seed: int = 0):
    """
    k-means on the unit sphere (a.k.a. spherical k-means / movMF hard-EM).

    z          : [N, m]  points (will be L2-normalised; the latent already is).
    n_clusters : K
    returns
        centres   : [K, m]  unit-norm cluster directions  (-> decoder.means)
        kappa_k   : [K]     per-cluster concentration estimate (-> cluster_log_kappa)
        labels    : [N]     hard assignment
        counts    : [K]     cluster sizes (-> seeds the stick-breaking weights)

    Runs `n_init` restarts (k-means++-on-the-sphere seeding) and keeps the run
    with the highest total cosine objective, mirroring sklearn's n_init.
    """
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
    N, m = z.shape
    K = min(n_clusters, N)
    g = torch.Generator(device=z.device).manual_seed(seed)

    best_obj = -float("inf")
    best = None
    for _ in range(n_init):
        # --- k-means++ seeding in cosine geometry ---
        idx0 = torch.randint(0, N, (1,), generator=g, device=z.device)
        centres = z[idx0]                                   # [1, m]
        for _ in range(1, K):
            sim = z @ centres.t()                           # [N, c]
            d = 1.0 - sim.max(dim=1).values                 # cosine "distance"
            d = d.clamp(min=0.0)
            probs = d / (d.sum() + 1e-12)
            nxt = torch.multinomial(probs, 1, generator=g)
            centres = torch.cat([centres, z[nxt]], dim=0)

        # --- Lloyd iterations in cosine geometry ---
        labels = torch.zeros(N, dtype=torch.long, device=z.device)
        for _ in range(n_iter):
            sim = z @ centres.t()                           # [N, K]
            new_labels = sim.argmax(dim=1)
            if torch.equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            new_centres = torch.zeros_like(centres)
            new_centres.index_add_(0, labels, z)
            norms = new_centres.norm(dim=-1, keepdim=True)
            empty = (norms.squeeze(-1) < 1e-8)
            # re-seed empty clusters at the worst-fit point (sklearn does similar)
            if empty.any():
                worst = (1.0 - sim.max(dim=1).values).argmax()
                new_centres[empty] = z[worst]
                norms = new_centres.norm(dim=-1, keepdim=True)
            centres = new_centres / (norms + 1e-8)

        sim = z @ centres.t()
        obj = sim.max(dim=1).values.sum().item()            # total cosine sim
        if obj > best_obj:
            best_obj = obj
            best = (centres.clone(), labels.clone())

    centres, labels = best
    counts = torch.bincount(labels, minlength=K).to(z.dtype)

    # mean resultant length r_k = ||sum_{i in k} z_i|| / n_k  -> kappa_k
    sums = torch.zeros(K, m, device=z.device, dtype=z.dtype)
    sums.index_add_(0, labels, z)
    r = sums.norm(dim=-1) / counts.clamp(min=1.0)
    kappa_k = _estimate_vmf_kappa(r, m)

    return centres, kappa_k, labels, counts


# =============================================================================
#  Householder rotation:  builds U(mu) with U(mu) e_1 = mu   (Ulrich 1984)
# =============================================================================
def householder_rotate(x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """
    Reflect samples `x` (drawn around the modal vector e_1 = (1, 0, ..., 0))
    so they become distributed around the requested mean direction `mu`.

    A Householder reflection H = I - 2 u u^T / (u^T u) with u = e_1 - mu maps
    e_1 -> mu.  Applying H to x rotates the whole vMF sample.

    x  : [B, m]  samples around e_1
    mu : [B, m]  unit mean directions
    returns [B, m] samples around mu.
    """
    m = mu.size(-1)
    e1 = torch.zeros_like(mu)
    e1[..., 0] = 1.0
    u = e1 - mu
    u = u / (u.norm(dim=-1, keepdim=True) + 1e-8)  # normalise the reflector
    # H x = x - 2 (u . x) u
    return x - 2.0 * (u * x).sum(dim=-1, keepdim=True) * u


# =============================================================================
#  von Mises-Fisher distribution  (sampling + KL to Uniform on the sphere)
# =============================================================================
class VonMisesFisher:
    """
    vMF(mu, kappa) on S^{m-1}.

    * `mu`    : [B, m]  unit mean directions (||mu|| = 1)
    * `kappa` : [B, 1]  concentration (>= 0)

    Implements:
      - rsample()       : reparameterised sample via Ulrich (1984) rejection
                          sampling + Householder rotation (Alg. 1 of the paper).
      - kl_uniform()    : KL( vMF(mu, kappa) || U(S^{m-1}) )  (Eq. 5).
    """

    def __init__(self, mu: torch.Tensor, kappa: torch.Tensor):
        self.mu = mu
        self.kappa = kappa
        self.m = mu.size(-1)
        self.device = mu.device
        self.dtype = mu.dtype

    # ------------------------------------------------------------------ #
    #  Sampling
    # ------------------------------------------------------------------ #
    def _sample_w_rejection(self, shape) -> torch.Tensor:
        """
        Sample the scalar  omega in [-1, 1]  from
            g(omega | kappa, m) ~ exp(kappa * omega) (1 - omega^2)^{(m-3)/2}
        using Ulrich's acceptance-rejection scheme.  Returns [B, 1].

        The proposal is a (rescaled) Beta((m-1)/2, (m-1)/2); the accepted
        omega is reparameterisable w.r.t. kappa (Naesseth et al. 2017), so we
        keep the analytic dependence on kappa through b, x0, c below.
        """
        kappa = self.kappa.squeeze(-1)               # [B]
        m = self.m
        B = kappa.shape[0]

        # Ulrich's constants
        sqrt_term = torch.sqrt(4.0 * kappa ** 2 + (m - 1) ** 2)
        b = (-2.0 * kappa + sqrt_term) / (m - 1)
        b = b.clamp(min=1e-10)
        x0 = (1.0 - b) / (1.0 + b)
        c = kappa * x0 + (m - 1) * torch.log(1.0 - x0 ** 2 + 1e-10)

        w = torch.empty(B, device=self.device, dtype=self.dtype)
        accepted = torch.zeros(B, dtype=torch.bool, device=self.device)

        beta = torch.distributions.Beta(
            torch.full((B,), (m - 1) / 2.0, device=self.device, dtype=self.dtype),
            torch.full((B,), (m - 1) / 2.0, device=self.device, dtype=self.dtype),
        )

        # rejection loop (vectorised over the batch; only the still-rejected
        # rows are resampled each iteration)
        for _ in range(100):
            if accepted.all():
                break
            eps = beta.sample()                       # [B] in (0,1)
            w_prop = (1.0 - (1.0 + b) * eps) / (1.0 - (1.0 - b) * eps)
            t = 2.0 * b / (1.0 - (1.0 - b) * eps)
            t = t.clamp(min=1e-10)
            u = torch.rand(B, device=self.device, dtype=self.dtype)
            accept = (kappa * w_prop + (m - 1) * torch.log(t) - c) >= torch.log(u + 1e-10)
            new = accept & (~accepted)
            w = torch.where(new, w_prop, w)
            accepted = accepted | accept

        # any rows that never accepted (extremely rare) fall back to the mode
        w = torch.where(accepted, w, x0)
        return w.unsqueeze(-1)                         # [B, 1]

    def rsample(self) -> torch.Tensor:
        """Reparameterised vMF sample on S^{m-1}.  Returns [B, m]."""
        m = self.m
        B = self.mu.size(0)

        # 1) sample omega from the tangential density g(omega|kappa, m)
        if m == 3:
            # S^2 special case: g reduces to exp(kappa*omega) on [-1,1]; sample
            # directly by inverting the CDF (no rejection needed).
            kappa = self.kappa                                   # [B,1]
            unif = torch.rand(B, 1, device=self.device, dtype=self.dtype)
            # w = 1 + (1/kappa) * log( u + (1-u) exp(-2 kappa) )
            w = 1.0 + (1.0 / kappa.clamp(min=1e-6)) * torch.log(
                unif + (1.0 - unif) * torch.exp(-2.0 * kappa) + 1e-10
            )
            w = w.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        else:
            w = self._sample_w_rejection(B)                      # [B,1]
            w = w.clamp(-1.0 + 1e-6, 1.0 - 1e-6)

        # 2) sample direction v ~ U(S^{m-2}) in the tangent space of e_1
        v = torch.randn(B, m - 1, device=self.device, dtype=self.dtype)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

        # 3) assemble z' around e_1 = (1, 0, ..., 0)
        #    z' = ( w ; sqrt(1 - w^2) * v )
        scale = torch.sqrt((1.0 - w ** 2).clamp(min=1e-10))      # [B,1]
        z_prime = torch.cat([w, scale * v], dim=-1)              # [B, m]

        # 4) rotate e_1 -> mu via Householder reflection
        z = householder_rotate(z_prime, self.mu)
        # numerical guard: keep it exactly on the sphere
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        return z

    # ------------------------------------------------------------------ #
    #  KL( vMF(mu, kappa) || Uniform(S^{m-1}) )      (paper Eq. 5)
    # ------------------------------------------------------------------ #
    def kl_uniform(self) -> torch.Tensor:
        """
        KL = kappa * I_{m/2}(k)/I_{m/2-1}(k)
             + log C_m(kappa)
             - log( Uniform density )                returns [B]
        where the uniform log-density on S^{m-1} is
            log U = log Gamma(m/2) - log( 2 pi^{m/2} ).
        Everything is expressed through log_ive so it is stable for large kappa.
        """
        m = self.m
        k = self.kappa.squeeze(-1).clamp(min=1e-6)               # [B]
        order = m / 2.0

        # ratio r = I_{m/2}(k) / I_{m/2-1}(k)
        r = ive_ratio(order, k)

        # log C_m(kappa) = (m/2 - 1) log k - (m/2) log(2 pi) - log I_{m/2-1}(k)
        #               = (m/2 - 1) log k - (m/2) log(2 pi) - ( log_ive + k )
        log_iv = log_ive(order - 1.0, k)
        log_C = (
            (order - 1.0) * torch.log(k)
            - order * math.log(2.0 * math.pi)
            - (log_iv + k)
        )

        # log uniform density on the sphere
        log_unif = math.lgamma(m / 2.0) - math.log(2.0) - (m / 2.0) * math.log(math.pi)

        kl_val = k * r + log_C - log_unif
        return kl_val                                            # [B]


# =============================================================================
#  FCLayers  (identical to v2.2)
# =============================================================================
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
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond

    def forward(self, x: torch.Tensor, *cat_list: int):
        one_hot_cat_list = []

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError("nb. categorical args provided doesn't match init. params.")

        for n_cat, cat in zip(self.n_cat_list, cat_list, strict=False):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:
                if cat.size(1) != n_cat:
                    one_hot_cat = nn.functional.one_hot(cat.squeeze(-1), n_cat)
                else:
                    one_hot_cat = cat
                one_hot_cat_list += [one_hot_cat]

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
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


# =============================================================================
#  Hyperspherical encoder  (the core S-VAE change)
# =============================================================================
class EncoderHyperspherical(nn.Module):
    """
    Encoder that outputs a von Mises-Fisher posterior on S^{n_latent - 1}.

    Instead of (mu, sigma^2) it produces:
        * mu_dir : a unit-norm mean *direction*  (L2 normalised)
        * kappa  : a positive concentration (softplus, bounded for stability)
    and a reparameterised sample z living ON the unit hypersphere.

    The latent code z therefore has ||z|| = 1, which is what removes the
    "origin gravity" of the Gaussian VAE and yields the better cluster
    separability reported in the paper.
    """

    def __init__(
        self,
        n_input: int,
        n_latent: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 256,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        kappa_min: float = 1.0,
        kappa_max: float = 1.0e4,
    ):
        super().__init__()

        self.n_latent = n_latent
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max

        self.input_norm = nn.LayerNorm(n_input)

        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        # mean DIRECTION head (will be L2-normalised to lie on the sphere)
        self.z_mean_encoder = nn.Linear(n_hidden, n_latent)
        nn.init.xavier_uniform_(self.z_mean_encoder.weight, gain=0.01)

        # CONCENTRATION head -> single positive scalar kappa per sample
        self.z_kappa_encoder = nn.Linear(n_hidden, 1)
        nn.init.xavier_uniform_(self.z_kappa_encoder.weight, gain=0.01)

    def forward(self, data: torch.Tensor, *cat_list: int):
        data = self.input_norm(data)
        q = self.encoder(data, *cat_list)

        # 1) mean direction on the unit sphere
        mu = self.z_mean_encoder(q)
        mu = mu / (mu.norm(dim=-1, keepdim=True) + 1e-8)          # ||mu|| = 1

        # 2) concentration kappa > 0  (softplus + kappa_min keeps it away from 0
        #    so the posterior stays informative, bounded above so the Bessel
        #    ratios stay numerically safe).  A floor > 1 is the standard remedy
        #    for vMF posterior collapse: it stops the KL from driving kappa -> 0
        #    (the uniform distribution), which would erase all cluster structure.
        kappa = F.softplus(self.z_kappa_encoder(q)) + self.kappa_min
        kappa = kappa.clamp(max=self.kappa_max)                   # [B, 1]

        # 3) reparameterised sample on the sphere
        qz = VonMisesFisher(mu, kappa)
        z = qz.rsample()                                         # [B, m], ||z||=1

        # return signature parallels v2.2's (qz_m, qz_v, z); here the second
        # slot carries kappa instead of a variance so downstream code can log it.
        return mu, kappa, z, qz


# =============================================================================
#  Decoder  (v2.2 spike-and-slab + regularized horseshoe, UNCHANGED)
#  + OPTIONAL mixture-of-vMF latent clustering for the hyperspherical latent.
# =============================================================================
class DecoderDBGCM(nn.Module):
    def __init__(
        self,
        n_clusters: int,
        n_input: int,
        n_output: int,
        pip0=0.1,
        v0=1,
        alpha: float = 10,
        c_reg: float = 1.0,
        d0: float = 1.0,
        latent_distribution: str = "vmf",   # "vmf" (spherical) or "normal"
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_output = n_output
        self.n_input = n_input
        self.latent_distribution = latent_distribution

        # ----- truncated DP stick-breaking weights ------------------------
        self.alpha = nn.Parameter(torch.tensor([alpha]), requires_grad=False)
        self.logits_v = nn.Parameter(torch.zeros(self.n_clusters) - 2.0)

        # ----- cluster centres --------------------------------------------
        # For the vMF mixture the centres are unit directions; for the normal
        # mixture they are free Euclidean means (kept for ablation).
        #self.means = nn.Parameter(torch.randn(self.n_clusters, self.n_input) * 0.1)
        self.means = nn.Parameter(F.normalize(torch.randn(self.n_clusters, self.n_input), dim=-1))
        self.logvars = nn.Parameter(torch.ones(self.n_clusters, self.n_input) * (-1.0))
        # per-cluster concentration for the vMF mixture (log-parameterised)
        #self.cluster_log_kappa = nn.Parameter(
        #    torch.ones(self.n_clusters) * math.log(10.0)
        #)
        self.cluster_log_kappa = nn.Parameter(torch.ones(self.n_clusters) * math.log(10.0) + 0.5 * torch.randn(self.n_clusters))

        # ----- spike-and-slab variational prior (UNCHANGED) ---------------
        self.logit_0 = nn.Parameter(
            torch.logit(torch.ones(1) * pip0, eps=1e-6), requires_grad=False
        )
        self.lnvar_0 = nn.Parameter(torch.log(torch.ones(1) * v0), requires_grad=False)
        self.bias_d = nn.Parameter(torch.zeros(1, self.n_output))
        self.slab_mean = nn.Parameter(torch.randn(self.n_clusters, self.n_output) * 0.01)
        self.slab_lnvar = nn.Parameter(torch.ones(self.n_clusters, self.n_output) * (-1.0))
        self.spike_logit = nn.Parameter(
            torch.zeros(self.n_clusters, self.n_output) * self.logit_0
        )

        # ----- regularized horseshoe (UNCHANGED) --------------------------
        self.beta_lnmean = nn.Parameter(torch.ones(self.n_clusters, self.n_output) * (-2.0))
        self.beta_lnvar = nn.Parameter(torch.ones(self.n_clusters, self.n_output) * (-3.0))
        self.alpha_lnmean = nn.Parameter(torch.ones(self.n_clusters, self.n_output) * (-2.0))
        self.alpha_lnvar = nn.Parameter(torch.ones(self.n_clusters, self.n_output) * (-3.0))

        self.zeta_b_lnmean = nn.Parameter(torch.ones(1) * (-3.0))
        self.zeta_b_lnvar = nn.Parameter(torch.ones(1) * (-4.0))
        self.zeta_a_lnmean = nn.Parameter(torch.ones(1) * (-3.0))
        self.zeta_a_lnvar = nn.Parameter(torch.ones(1) * (-4.0))

        self.c_reg_sq = nn.Parameter(torch.tensor([c_reg ** 2]), requires_grad=False)
        self.d0_sq = nn.Parameter(torch.tensor([d0 ** 2]), requires_grad=False)

        self.log_softmax = nn.LogSoftmax(dim=-1)

    # ---------------------------------------------------------------- #
    def forward(self, z: torch.Tensor):
        if self.latent_distribution == "vmf":
            log_theta_dp, theta_dp, theta_dp_kl = self.MixtureOfVMF(z)
        else:
            log_theta_dp, theta_dp, theta_dp_kl = self.VariationalDirichletProcess(z)
        rho = self.get_beta(self.spike_logit, self.slab_mean, self.slab_lnvar, self.bias_d)
        rho_kl = self.sparse_kl_loss(
            self.logit_0, self.lnvar_0, self.spike_logit, self.slab_mean, self.slab_lnvar
        )
        return rho, rho_kl, log_theta_dp, theta_dp, theta_dp_kl

    def get_rho(self):
        return self.get_beta(self.spike_logit, self.slab_mean, self.slab_lnvar, self.bias_d)

    # ---------------------------------------------------------------- #
    #  Spike-and-slab + regularized horseshoe  (UNCHANGED from v2.2)
    # ---------------------------------------------------------------- #
    def get_beta(self, spike_logit, slab_mean, slab_lnvar, bias_d):
        pip = torch.sigmoid(spike_logit)

        eps_beta = torch.randn_like(self.beta_lnmean)
        eps_alpha = torch.randn_like(self.alpha_lnmean)

        beta_lnmean_clipped = torch.clamp(self.beta_lnmean, -10, 10)
        beta_lnvar_clipped = torch.clamp(self.beta_lnvar, -10, 2)
        alpha_lnmean_clipped = torch.clamp(self.alpha_lnmean, -10, 10)
        alpha_lnvar_clipped = torch.clamp(self.alpha_lnvar, -10, 2)

        beta = torch.exp(beta_lnmean_clipped + torch.sqrt(torch.exp(beta_lnvar_clipped)) * eps_beta)
        alpha = torch.exp(alpha_lnmean_clipped + torch.sqrt(torch.exp(alpha_lnvar_clipped)) * eps_alpha)
        beta = torch.clamp(beta, 1e-6, 1e6)
        alpha = torch.clamp(alpha, 1e-6, 1e6)

        eps_zeta_b = torch.randn_like(self.zeta_b_lnmean)
        eps_zeta_a = torch.randn_like(self.zeta_a_lnmean)

        zeta_b_lnmean_clipped = torch.clamp(self.zeta_b_lnmean, -10, 10)
        zeta_b_lnvar_clipped = torch.clamp(self.zeta_b_lnvar, -10, 2)
        zeta_a_lnmean_clipped = torch.clamp(self.zeta_a_lnmean, -10, 10)
        zeta_a_lnvar_clipped = torch.clamp(self.zeta_a_lnvar, -10, 2)

        zeta_b = torch.exp(zeta_b_lnmean_clipped + torch.sqrt(torch.exp(zeta_b_lnvar_clipped)) * eps_zeta_b)
        zeta_a = torch.exp(zeta_a_lnmean_clipped + torch.sqrt(torch.exp(zeta_a_lnvar_clipped)) * eps_zeta_a)
        zeta_b = torch.clamp(zeta_b, 1e-6, 1e6)
        zeta_a = torch.clamp(zeta_a, 1e-6, 1e6)

        zeta_sq = zeta_a * zeta_b
        tau_sq = beta * alpha

        c_reg_sq = self.c_reg_sq.view(1, 1)
        zeta_sq = zeta_sq.view(1, 1).expand_as(tau_sq)

        denominator = torch.clamp(c_reg_sq + tau_sq * zeta_sq, min=1e-8)
        tau_tilde_sq = (c_reg_sq * tau_sq) / denominator
        tau_tilde_sq = torch.clamp(tau_tilde_sq, 1e-6, 1.0)

        mean = slab_mean * pip
        var = pip * (1 - pip) * torch.square(slab_mean)

        lnvar_clipped = torch.clamp(slab_lnvar, -10, 4)
        horseshoe_var = pip * torch.exp(lnvar_clipped) * tau_tilde_sq
        var = var + horseshoe_var
        var = torch.clamp(var, 1e-8, 1e2)

        eps = torch.randn_like(var)
        return mean + eps * torch.sqrt(var) - bias_d

    # ---------------------------------------------------------------- #
    #  NEW : Mixture of von Mises-Fisher for the hyperspherical latent
    # ---------------------------------------------------------------- #
    def MixtureOfVMF(self, Z: torch.Tensor):
        """
        Soft cluster assignment for z that LIVES ON THE SPHERE, using a
        truncated-DP mixture of vMF components.

        Because q(z|x) is vMF and z is unit-norm, the natural cluster model is
        a mixture of vMFs whose log-likelihood is
            log p(z | c_k, kappa_k) = log C_m(kappa_k) + kappa_k * (c_k . z)
        i.e. driven by COSINE SIMILARITY (c_k . z) rather than L2 distance.
        This is exactly the regime where the paper shows clusters separate
        cleanly on the sphere (Fig. 2b / Fig. 3b).
        """
        # unit-norm cluster centres (directions)
        centres = self.means / (self.means.norm(dim=-1, keepdim=True) + 1e-8)  # [K, m]
        kappa_k = torch.exp(self.cluster_log_kappa).clamp(1e-2, 1e4)           # [K]
        m = self.n_input

        # stick-breaking mixture weights
        v = torch.sigmoid(self.logits_v)
        pi = stick_breaking(v)
        log_pi = torch.log(pi + 1e-12)                                          # [K]

        # log normaliser of each vMF component  log C_m(kappa_k)
        order = m / 2.0
        log_iv = log_ive(order - 1.0, kappa_k)                                  # [K]
        log_C = (
            (order - 1.0) * torch.log(kappa_k)
            - order * math.log(2.0 * math.pi)
            - (log_iv + kappa_k)
        )                                                                       # [K]

        # cosine-similarity log-likelihood   [B, K]
        cos_sim = Z @ centres.t()                                               # [B, K]
        log_lik = log_C.unsqueeze(0) + kappa_k.unsqueeze(0) * cos_sim

        log_pz = log_lik + log_pi.unsqueeze(0)                                  # [B, K]
        log_prob = torch.logsumexp(log_pz, dim=1)                              # [B]
        resp = torch.softmax(log_pz, dim=1)                                     # [B, K]

        # KL of the stick-breaking weights (same structural term as v2.2)
        kl_sticks = torch.sum(
            (1 - v) * (torch.log(1 - v + 1e-10) - torch.log(1 - v.detach() + 1e-10))
            + v * (torch.log(v + 1e-10) - torch.log(v.detach() + 1e-10))
        )
        # mild regulariser pulling component concentrations toward a prior
        kl_kappa = 1e-3 * torch.sum((self.cluster_log_kappa - math.log(10.0)) ** 2)

        vb_dp_kl = kl_sticks + kl_kappa
        return log_prob, resp, vb_dp_kl

    # ---------------------------------------------------------------- #
    #  NEW : spherical-k-means warm start for the mixture-of-vMF
    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def kmeans_warm_start(self, z: torch.Tensor, n_iter: int = 50, n_init: int = 5,
                          seed: int = 0, kappa_cap: float = 200.0):
        """
        Seed the mixture-of-vMF parameters from spherical k-means on encoded z,
        the spherical analogue of scikit-learn's `init_params="kmeans"`.

        Sets, IN PLACE:
          * self.means            <- unit-norm spherical-k-means centroids
          * self.cluster_log_kappa<- log of the moment-estimated per-cluster kappa,
                                     CAPPED at `kappa_cap` (see below)
          * self.logits_v         <- stick-breaking logits ordered by cluster size,
                                     so the truncated-DP weights start matched to
                                     the discovered occupancy.  Ordering the sticks
                                     by population is what breaks the symmetry that
                                     otherwise collapses every point onto one
                                     component (the ver3_1 VB-vMF-MM collapse).

        Why cap kappa
        -------------
        At the seeding point the encoder's mode directions mu are still bunched,
        so within each k-means cluster the mean resultant length r -> 1 and the
        moment estimator kappa = r(m - r^2)/(1 - r^2) blows up (we observed
        kappa ~ 1e4 on CDI).  Seeding the vMF components that sharp makes the
        responsibility softmax near-degenerate: whichever component is marginally
        closest claims essentially everything, and the stick-breaking weights then
        collapse onto that one stick -- re-creating the ver3_1 collapse the warm
        start is meant to PREVENT.  Capping the seeded concentration near the
        posterior scale (kappa_min ~ 50, cap ~ 200) keeps components selective but
        not degenerate; training is then free to sharpen them as structure emerges.

        Only valid for latent_distribution="vmf"; a no-op otherwise.
        """
        if self.latent_distribution != "vmf":
            return None

        z = z.to(self.means.device)
        centres, kappa_k, labels, counts = spherical_kmeans(
            z, self.n_clusters, n_iter=n_iter, n_init=n_init, seed=seed
        )

        # 1) cluster directions  (means is stored un-normalised; MixtureOfVMF
        #    re-normalises, but we store unit vectors so the start is exact)
        self.means.data.copy_(centres)

        # 2) per-cluster concentration from the mean resultant length, capped so
        #    the seeded components are not razor-sharp (see docstring).
        kappa_k = kappa_k.clamp(1e-2, kappa_cap)
        self.cluster_log_kappa.data.copy_(torch.log(kappa_k))

        # 3) seed the stick-breaking logits from the k-means occupancy.
        #    Sort clusters by descending size, then choose v_k so that the
        #    stick-breaking weights pi_k = v_k prod_{j<k}(1 - v_j) reproduce the
        #    empirical proportions p_k.  v_k = p_k / (1 - sum_{j<k} p_j).
        order = torch.argsort(counts, descending=True)
        self.means.data.copy_(self.means.data[order])
        self.cluster_log_kappa.data.copy_(self.cluster_log_kappa.data[order])
        p = (counts[order] + 1e-3)
        p = p / p.sum()
        remaining = torch.cumsum(p, dim=0) - p          # sum_{j<k} p_j
        v = (p / (1.0 - remaining).clamp(min=1e-6)).clamp(1e-4, 1.0 - 1e-4)
        self.logits_v.data.copy_(torch.log(v) - torch.log1p(-v))   # logit(v)

        return dict(centres=centres, kappa_k=kappa_k, counts=counts, labels=labels)

    # ---------------------------------------------------------------- #
    #  Gaussian DP-GMM  (kept for the "normal" ablation; UNCHANGED)
    # ---------------------------------------------------------------- #
    def VariationalDirichletProcess(self, Z: torch.Tensor):
        v = torch.sigmoid(self.logits_v)
        pi = stick_breaking(v)
        log_pi = torch.log(pi + 1e-12)

        Z_exp = Z.unsqueeze(1)
        mu_exp = self.means.unsqueeze(0)
        lv_exp = self.logvars.unsqueeze(0)
        var = torch.exp(lv_exp)
        var = torch.clamp(var, min=1e-6, max=1e6)

        log_det_cov = torch.sum(lv_exp, dim=2)
        diff = Z_exp - mu_exp
        mahalanobis_dist = torch.sum(diff ** 2 / var, dim=2)
        log_lik = -0.5 * (self.n_input * math.log(2 * math.pi) + log_det_cov + mahalanobis_dist)

        log_pz = log_lik + log_pi
        log_prob = torch.logsumexp(log_pz, dim=1)
        resp = torch.softmax(log_pz, dim=1)

        kl_sticks = torch.sum(
            (1 - v) * (torch.log(1 - v + 1e-10) - torch.log(1 - v.detach() + 1e-10))
            + v * (torch.log(v + 1e-10) - torch.log(v.detach() + 1e-10))
        )
        kl_gaussians = 0.5 * torch.sum(
            self.means ** 2 + torch.exp(self.logvars) - self.logvars - 1
        )
        vb_dp_kl = kl_sticks + kl_gaussians
        return log_prob, resp, vb_dp_kl

    def cluster_weights(self):
        with torch.no_grad():
            v = torch.sigmoid(self.logits_v)
            pi = stick_breaking(v)
        return pi

    def soft_max(self, z: torch.Tensor):
        return torch.exp(self.log_softmax(z))

    # ---------------------------------------------------------------- #
    #  Sparse KL  (UNCHANGED from v2.2)
    # ---------------------------------------------------------------- #
    def sparse_kl_loss(self, logit_0, lnvar_0, spike_logit, slab_mean, slab_lnvar):
        pip_hat = torch.sigmoid(spike_logit)
        kl_pip_1 = pip_hat * (spike_logit - logit_0)
        kl_pip = kl_pip_1 - nn.functional.softplus(spike_logit) + nn.functional.softplus(logit_0)

        sq_term = torch.exp(-lnvar_0) * (torch.square(slab_mean) + torch.exp(slab_lnvar))
        kl_g = torch.clamp(-0.5 * (1.0 + slab_lnvar - lnvar_0 - sq_term), min=0.0)

        horseshoe_scale = 1e-4

        kl_beta = self.log_normal_to_inv_gamma_kl(self.beta_lnmean, self.beta_lnvar, 0.5, 1.0)
        kl_alpha = self.log_normal_to_gamma_kl(self.alpha_lnmean, self.alpha_lnvar, 0.5, 1.0)
        kl_zeta_b = self.log_normal_to_inv_gamma_kl(self.zeta_b_lnmean, self.zeta_b_lnvar, 0.5, 1.0)
        kl_zeta_a = self.log_normal_to_gamma_kl(self.zeta_a_lnmean, self.zeta_a_lnvar, 0.5, self.d0_sq.item())

        ss_kl = torch.sum(torch.clamp(kl_pip + pip_hat * kl_g, min=0.0))
        kl_horseshoe = horseshoe_scale * (kl_beta + kl_alpha + kl_zeta_b + kl_zeta_a)
        return ss_kl + kl_horseshoe

    def log_normal_to_inv_gamma_kl(self, mu, log_var, alpha, beta):
        alpha_tensor = torch.tensor(alpha, device=mu.device)
        beta_tensor = torch.tensor(beta, device=mu.device)
        var = torch.exp(torch.clamp(log_var, -20, 20))
        kl = alpha_tensor * torch.log(beta_tensor) - torch.lgamma(alpha_tensor)
        kl = kl - (alpha_tensor + 1) * (mu - 0.5 * var)
        kl = kl + beta_tensor * torch.exp(-mu + 0.5 * var)
        kl = torch.abs(kl)
        return kl.sum() + 1e-8

    def log_normal_to_gamma_kl(self, mu, log_var, alpha, beta):
        alpha_tensor = torch.tensor(alpha, device=mu.device)
        beta_tensor = torch.tensor(beta, device=mu.device)
        var = torch.exp(torch.clamp(log_var, -20, 20))
        mean_lognormal = torch.exp(mu + var / 2)
        kl = -alpha_tensor * torch.log(beta_tensor) + torch.lgamma(alpha_tensor)
        kl = kl + beta_tensor * mean_lognormal
        kl = kl - (alpha_tensor - 1) * mu
        kl = torch.abs(kl)
        return kl.sum() + 1e-8


# =============================================================================
#  LightningModule  (S-VAE)
# =============================================================================
class DBayesCM(pl.LightningModule):
    """
    Hyperspherical (S-VAE) version of DBayGenCM.

    Identical training/eval interface to v2.2; the only modelling differences:
      * encoder  -> EncoderHyperspherical (vMF posterior, z on the sphere)
      * KL term  -> KL(vMF || Uniform)  instead of  KL(N || N)
      * latent clustering -> mixture-of-vMF (when latent_distribution="vmf")
    """

    def __init__(
        self,
        n_genes: int,
        n_latent: int = 32,
        n_clusters: int = 10,
        n_layers_encoder_individual: int = 2,
        dim_hidden_encoder: int = 128,
        pip0_rho: float = 0.1,
        kl_weight_beta: float = 1.0,
        c_reg: float = 1.0,
        d0: float = 1.0,
        learning_rate: float = 1e-3,
        n_samples: int = 1000,
        latent_distribution: str = "vmf",   # "vmf" (S-VAE) or "normal" (ablation)
        kappa_min: float = 1.0,             # floor on the posterior concentration
        kl_anneal_epochs: int = 0,          # >0 linearly warms up the spherical KL
        kl_z_max: float = 1.0,              # final weight on KL(vMF || Uniform)
        kmeans_init: bool = False,          # spherical-k-means warm start (opt-in)
        kmeans_init_epoch: int = 20,        # encoder-warmup epochs before k-means
        kmeans_n_init: int = 5,             # k-means restarts (cf. sklearn n_init)
        kmeans_seed: int = 0,
        kmeans_kappa_cap: float = 200.0,    # cap on seeded cluster concentration
    ):
        super().__init__()

        self.n_input = n_genes
        self.n_latent = n_latent
        self.n_clusters = n_clusters
        self.pip0_rho = pip0_rho
        self.kl_weight_beta = kl_weight_beta
        self.c_reg = c_reg
        self.d0 = d0
        self.latent_distribution = latent_distribution
        self.kappa_min = kappa_min
        self.kl_anneal_epochs = kl_anneal_epochs
        self.kl_z_max = kl_z_max
        # spherical-k-means warm start (opt-in; off by default so the v2.2
        # comparison is byte-for-byte unchanged)
        self.kmeans_init = kmeans_init
        self.kmeans_init_epoch = kmeans_init_epoch
        self.kmeans_n_init = kmeans_n_init
        self.kmeans_seed = kmeans_seed
        self.kmeans_kappa_cap = kmeans_kappa_cap
        self._kmeans_done = False
        # Used to drive KL annealing when training OUTSIDE a Lightning Trainer
        # (inside a Trainer the read-only `current_epoch` property is used).
        self._manual_epoch = None

        self.z_encoder = EncoderHyperspherical(
            n_input=self.n_input,
            n_latent=self.n_latent,
            n_hidden=dim_hidden_encoder,
            n_layers=n_layers_encoder_individual,
            kappa_min=kappa_min,
        )

        self.decoder = DecoderDBGCM(
            n_input=self.n_latent,
            n_clusters=self.n_clusters,
            n_output=self.n_input,
            pip0=self.pip0_rho,
            c_reg=self.c_reg,
            d0=self.d0,
            latent_distribution=self.latent_distribution,
        )

        # history buffers
        self.train_loss_history = []
        self.val_loss_history = []
        self.ari_history = []
        self.train_ari_history = []
        self.val_rho_all_epochs = []
        self.val_theta_all_epochs = []
        self.val_z_sample_all_epochs = []
        self.train_rho_all_epochs = []
        self.train_theta_all_epochs = []
        self.train_z_sample_all_epochs = []
        self.val_kappa_history = []

        # --- vMF hypersphere GEOMETRY per validation epoch (for visualisation),
        # aligned 1-to-1 with ari_history / val_theta_all_epochs. Indexing these
        # at the best-ARI epoch gives exactly the geometry behind the top ARI.
        self.val_mu_all_epochs = []            # [epoch] -> [N, D]  encoder mean direction
        self.val_kappa_all_epochs = []         # [epoch] -> [N]     encoder concentration
        self.val_z_all_epochs = []             # [epoch] -> [N, D]  on-sphere sample
        self.val_centres_all_epochs = []       # [epoch] -> [K, D]  vMF cluster centres
        self.val_cluster_kappa_all_epochs = [] # [epoch] -> [K]     per-cluster kappa
        self.val_labels_all_epochs = []        # [epoch] -> [N]     true labels (aligned)

        self._train_losses_epoch = []
        self._val_losses_epoch = []

        self.learning_rate = learning_rate
        self.n_samples = n_samples

    # ------------------------------------------------------------------ #
    #  Core methods
    # ------------------------------------------------------------------ #
    def dir_llik(self, xx: torch.Tensor, aa: torch.Tensor) -> torch.Tensor:
        """Dirichlet (-multinomial) log-likelihood.  (UNCHANGED)"""
        term1 = torch.lgamma(torch.sum(aa, dim=-1)) - torch.lgamma(torch.sum(aa + xx, dim=-1))
        term2 = torch.sum(
            torch.where(xx > 0, torch.lgamma(aa + xx) - torch.lgamma(aa), torch.zeros_like(xx)),
            dim=-1,
        )
        return term1 + term2

    def inference(self, x: torch.Tensor) -> dict:
        x_ = torch.log(1 + x)
        mu, kappa, z, qz = self.z_encoder(x_)
        # keep keys compatible with v2.2 callers; qz_v slot carries kappa
        return dict(qz_m=mu, qz_v=kappa, z=z, qz=qz)

    def generative(self, z) -> dict:
        rho, rho_kl, log_theta_dp, theta_dp, theta_dp_kl = self.decoder(z)
        return dict(rho=rho, rho_kl=rho_kl, theta_dp=theta_dp)

    def get_decoder_outputs(self, x: torch.Tensor):
        x_ = torch.log(1 + x)
        mu, kappa, z, qz = self.z_encoder(x_)
        rho, rho_kl, log_theta_dp, theta_dp, theta_dp_kl = self.decoder(z)
        return rho, theta_dp

    def sample_from_posterior_z(
        self, x: torch.Tensor, deterministic: bool = True, output_softmax_z: bool = True
    ):
        inference_out = self.inference(x)
        if deterministic:
            # the vMF "mode" is its mean direction mu (already unit-norm)
            z = inference_out["qz_m"]
        else:
            z = inference_out["z"]
        if output_softmax_z:
            generative_outputs = self.generative(z)
            z = generative_outputs["theta_dp"]
        return dict(z=z)

    def forward(self, x):
        x_ = torch.log(1 + x)
        mu, kappa, z, qz = self.z_encoder(x_)
        return z

    def get_reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        inference_out = self.inference(x)
        z = inference_out["z"]
        gen_out = self.generative(z)
        theta = gen_out["theta_dp"]
        rho = gen_out["rho"]
        log_aa = torch.clamp(torch.mm(theta, rho), -10, 10)
        aa = torch.exp(log_aa)
        return -self.dir_llik(x, aa)

    # ------------------------------------------------------------------ #
    #  Loss  (only the latent KL term differs from v2.2)
    # ------------------------------------------------------------------ #
    def compute_loss(self, x, kl_weight=1.0):
        # 1) preprocessing / normalisation
        x_ = torch.log1p(x)
        x_ = torch.clamp(x_, min=0.0)

        # 2) encoder (vMF posterior on the sphere)
        torch.nn.utils.clip_grad_norm_(self.z_encoder.parameters(), max_norm=1.0)
        mu, kappa, z, qz = self.z_encoder(x_)

        # 3) decoder
        rho, rho_kl, log_theta_dp, theta_dp, theta_dp_kl = self.decoder(z)

        kl_weight_beta = self.kl_weight_beta

        # 4) reconstruction (Dirichlet-multinomial)  (UNCHANGED)
        log_aa = torch.clamp(torch.mm(theta_dp, rho), -10, 10)
        aa = torch.exp(log_aa)
        reconstruction_loss = -self.dir_llik(x_, aa)

        # 5) KL divergences
        #    ***  THE S-VAE CHANGE  ***
        #    KL( vMF(mu, kappa) || Uniform(S^{m-1}) ) replaces KL(N || N).
        #    This term depends only on kappa, never on mu (paper Sec. 3.2):
        #    the mean direction is optimised purely through reconstruction,
        #    so clusters are free to spread over the sphere with no origin pull.
        kl_divergence_z = qz.kl_uniform()                      # [B]

        kl_divergence_beta = torch.clamp(rho_kl, min=0.0, max=1e5)
        kl_divergence_theta_dp = torch.clamp(theta_dp_kl, min=0.0, max=1e5)
        kl_local = torch.clamp(kl_divergence_z, min=0.0, max=1e5)

        # KL warm-up: ramp the spherical KL weight from 0 to kl_z_max over the
        # first `kl_anneal_epochs`.  Without this the strong vMF KL collapses the
        # posterior to the uniform (kappa -> floor) before the encoder has learnt
        # any directional structure -- the classic vMF-VAE failure mode.
        if self.kl_anneal_epochs > 0:
            epoch = self._manual_epoch if self._manual_epoch is not None else self.current_epoch
            anneal = min(1.0, (epoch + 1) / float(self.kl_anneal_epochs))
        else:
            anneal = 1.0
        kl_z_weight = kl_weight * self.kl_z_max * anneal

        reconstruction_term = torch.mean(reconstruction_loss)
        kl_z_term = torch.mean(kl_z_weight * kl_local)
        kl_beta_term = kl_weight_beta * kl_divergence_beta / x.shape[1]

        # During the spherical-k-means warmup the cluster term is suppressed so
        # the encoder learns DIRECTIONAL structure on the sphere first; the
        # mixture is only switched on (full weight) once it has been seeded from
        # spherical k-means.  Without the warmup, k-means would be run on a
        # randomly-initialised encoder's noise.  When kmeans_init is off this is
        # a no-op (theta_weight == 0.1 always).
        theta_weight = 0.1
        if self.kmeans_init and not self._kmeans_done:
            theta_weight = 0.0
        kl_theta_term = theta_weight * kl_divergence_theta_dp

        loss = reconstruction_term + kl_z_term + kl_beta_term + kl_theta_term

        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: Loss was NaN or Inf, using substitute loss")
            return torch.tensor(1.0, device=x.device, requires_grad=True)

        return loss

    # ------------------------------------------------------------------ #
    #  Training / validation scaffolding  (mirrors v2.2)
    # ------------------------------------------------------------------ #
    def on_train_epoch_start(self):
        self._train_losses_epoch.clear()
        self._train_rho_list = []
        self._train_theta_list = []
        self._train_labels_epoch = []
        self._train_z_sample_list = []

    def training_step(self, batch, batch_idx):
        x = batch["Microbiome"]
        y = batch["Labels"]

        loss = self.compute_loss(x)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self._train_losses_epoch.append(loss.detach())

        with torch.no_grad():
            rho, theta = self.get_decoder_outputs(x)
            z_sample = torch.sigmoid(self.decoder.spike_logit)

        self._train_rho_list.append(rho.detach().cpu())
        self._train_theta_list.append(theta.detach().cpu())
        self._train_labels_epoch.append(y.detach().cpu())
        self._train_z_sample_list.append(z_sample.detach().cpu())
        return loss

    def on_train_epoch_end(self):
        epoch_loss = torch.stack(self._train_losses_epoch).mean().item()
        self.train_loss_history.append(epoch_loss)
        print(f"Epoch {self.current_epoch} Train Loss: {epoch_loss:.4f}")

        if len(self._train_rho_list) > 0:
            all_rho = torch.cat(self._train_rho_list, dim=0)
            all_theta = torch.cat(self._train_theta_list, dim=0)
            all_labels = torch.cat(self._train_labels_epoch, dim=0)
            all_z_sample = torch.stack(self._train_z_sample_list, dim=0)

            pred_labels = torch.argmax(all_theta, dim=1)
            ari = adjusted_rand_score(all_labels.numpy(), pred_labels.numpy())
            self.train_ari_history.append(ari)
            self.train_rho_all_epochs.append(all_rho)
            self.train_theta_all_epochs.append(all_theta)
            self.train_z_sample_all_epochs.append(all_z_sample)
            print(f"Epoch {self.current_epoch} Train ARI: {ari:.4f}")

        # spherical-k-means warm start: fire once, at the end of the warmup
        # window, after the encoder has learnt some directional structure.
        if (
            self.kmeans_init
            and not self._kmeans_done
            and self.current_epoch + 1 >= self.kmeans_init_epoch
        ):
            self._run_kmeans_warm_start()

    @torch.no_grad()
    def _encode_training_set(self):
        """Encode the whole training set to its on-sphere latent z (mode = mu)."""
        loader = self.trainer.train_dataloader
        was_training = self.training
        self.eval()
        zs = []
        for batch in loader:
            x = batch["Microbiome"].to(self.device)
            inf = self.inference(x)
            zs.append(inf["qz_m"])              # mean direction (unit-norm mode)
        if was_training:
            self.train()
        return torch.cat(zs, dim=0)

    def _run_kmeans_warm_start(self):
        """Seed the mixture-of-vMF from spherical k-means on encoded z (once)."""
        z = self._encode_training_set()
        info = self.decoder.kmeans_warm_start(
            z, n_init=self.kmeans_n_init, seed=self.kmeans_seed,
            kappa_cap=self.kmeans_kappa_cap,
        )
        self._kmeans_done = True
        if info is not None:
            sizes = info["counts"].long().tolist()
            kappas = info["kappa_k"].tolist()
            print(
                f"[k-means init @ epoch {self.current_epoch}] "
                f"seeded {self.n_clusters} vMF centres on the sphere | "
                f"cluster sizes {sizes} | "
                f"kappa range [{min(kappas):.1f}, {max(kappas):.1f}]"
            )

    def on_validation_epoch_start(self):
        self._val_losses_epoch.clear()
        self._val_rho_list = []
        self._val_theta_list = []
        self._val_labels_epoch = []
        self._val_z_sample_list = []
        self._val_kappa_list = []
        self._val_mu_list = []
        self._val_z_list = []

    def validation_step(self, batch, batch_idx):
        x = batch["Microbiome"]
        y = batch["Labels"]

        loss = self.compute_loss(x)
        self._val_losses_epoch.append(loss.detach())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        with torch.no_grad():
            rho, theta = self.get_decoder_outputs(x)
            z_sample = torch.sigmoid(self.decoder.spike_logit)
            inf = self.inference(x)
            # vMF encoder geometry: qz_m = mean direction, qz_v = kappa, z = sample
            self._val_kappa_list.append(inf["qz_v"].detach().cpu())
            self._val_mu_list.append(inf["qz_m"].detach().cpu())
            self._val_z_list.append(inf["z"].detach().cpu())

        self._val_rho_list.append(rho.detach().cpu())
        self._val_theta_list.append(theta.detach().cpu())
        self._val_labels_epoch.append(y.detach().cpu())
        self._val_z_sample_list.append(z_sample.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        if len(self._val_losses_epoch) > 0:
            epoch_loss = torch.stack(self._val_losses_epoch).mean().item()
            self.val_loss_history.append(epoch_loss)
            print(f"Epoch {self.current_epoch} Val Loss: {epoch_loss:.4f}")

        if len(self._val_rho_list) > 0:
            all_rho = torch.cat(self._val_rho_list, dim=0)
            all_theta = torch.cat(self._val_theta_list, dim=0)
            all_labels = torch.cat(self._val_labels_epoch, dim=0)
            all_z_sample = torch.stack(self._val_z_sample_list, dim=0)

            pred_labels = torch.argmax(all_theta, dim=1)
            ari = adjusted_rand_score(all_labels.numpy(), pred_labels.numpy())
            self.ari_history.append(ari)
            # expose val ARI as a Lightning metric so a ModelCheckpoint callback
            # can monitor it and save the best-ARI epoch (not just the last one)
            self.log("val_ari", ari, on_step=False, on_epoch=True, prog_bar=False)
            self.val_rho_all_epochs.append(all_rho)
            self.val_theta_all_epochs.append(all_theta)
            self.val_z_sample_all_epochs.append(all_z_sample)

            # -----------------------------------------------------------------
            # Store the vMF GEOMETRY for THIS epoch, aligned 1-to-1 with
            # val_theta_all_epochs / ari_history. Indexing at the best-ARI epoch
            # yields exactly the geometry behind the highest ARI (the same theta
            # that scored it) -- ready for hypersphere visualisation, no re-run.
            # -----------------------------------------------------------------
            all_mu = torch.cat(self._val_mu_list, dim=0)        # [N, D]
            all_kappa = torch.cat(self._val_kappa_list, dim=0)  # [N]
            all_z = torch.cat(self._val_z_list, dim=0)          # [N, D]
            self.val_mu_all_epochs.append(all_mu)
            self.val_kappa_all_epochs.append(all_kappa)
            self.val_z_all_epochs.append(all_z)
            self.val_labels_all_epochs.append(all_labels)       # true labels, same order

            # cluster geometry of the mixture-of-vMF at this epoch. v3 parameterises
            # the per-cluster concentration directly as cluster_log_kappa (NOT the
            # Gamma a/b used in v2.5), so kappa_k = exp(cluster_log_kappa).
            with torch.no_grad():
                centres = F.normalize(self.decoder.means, p=2, dim=1).detach().cpu()   # [K, D]
                cluster_kappa = torch.exp(self.decoder.cluster_log_kappa).clamp(1e-2, 1e4).detach().cpu()  # [K]
            self.val_centres_all_epochs.append(centres)
            self.val_cluster_kappa_all_epochs.append(cluster_kappa)

            if len(self._val_kappa_list) > 0:
                mean_kappa = all_kappa.mean().item()
                self.val_kappa_history.append(mean_kappa)
                print(f"Epoch {self.current_epoch} ARI: {ari:.4f}  (mean kappa: {mean_kappa:.2f})")
            else:
                print(f"Epoch {self.current_epoch} ARI: {ari:.4f}")

    def collect_decoder_outputs_for_dataset(self, data_loader):
        self.eval()
        all_rho = []
        all_theta = []
        with torch.no_grad():
            for batch in data_loader:
                x = batch["Microbiome"]
                rho, theta = self.get_decoder_outputs(x)
                all_rho.append(rho.cpu().unsqueeze(0))
                all_theta.append(theta.cpu())
        return torch.cat(all_rho, dim=0), torch.cat(all_theta, dim=0)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# =============================================================================
#  Dataset
# =============================================================================
from torch.utils.data import Dataset, DataLoader


class MicrobiomeDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"Microbiome": self.X[idx], "Labels": self.y[idx]}


# =============================================================================
#  Hypersphere-visualisation helpers (permutation null + best-ARI geometry save)
#  Mirrors DBayGenCM_2omics_v2.5. Adapted to 1-omics (MicrobiomeDataset(X, y))
#  and to this model's per-epoch buffers, which are populated during training:
#    val_mu / val_kappa / val_z / val_centres / val_cluster_kappa / val_theta /
#    val_labels  (all *_all_epochs, aligned 1-to-1 with ari_history).
#  Import and call these from your own training/seed-sweep runner.
# =============================================================================
def permutation_null(y_true, y_pred, n_perm, rng):
    """Empirical null for ARI by shuffling the TRUE labels: z-score + p-value."""
    obs = adjusted_rand_score(y_true, y_pred)
    null = np.empty(n_perm, dtype=np.float64)
    yt = np.asarray(y_true).copy()
    for i in range(n_perm):
        rng.shuffle(yt)
        null[i] = adjusted_rand_score(yt, y_pred)
    mu, sd = null.mean(), null.std(ddof=1)
    z = (obs - mu) / (sd + 1e-12)
    p_emp = (1.0 + np.sum(null >= obs)) / (n_perm + 1.0)
    return dict(obs_ari=obs, null=null, null_mean=mu, null_std=sd, z_score=z, p_emp=p_emp)


def save_best_geometry(model, seed, best_ep, best_ari, y_pred, yva, out_dir="."):
    """
    Index every per-epoch vMF buffer at `best_ep` -> the exact highest-ARI state.
    The saved theta is the SAME one that scored best_ari, so ARI recomputed from
    it matches (ari_from_saved_theta). 1-omics hyperspherical S-VAE geometry:
    mu/kappa/z (encoder) + theta/centres/cluster_kappa (mixture-of-vMF) + labels.
    """
    theta_best   = model.val_theta_all_epochs[best_ep]
    mu_best      = model.val_mu_all_epochs[best_ep]
    kappa_best   = model.val_kappa_all_epochs[best_ep]
    z_best       = model.val_z_all_epochs[best_ep]
    centres_best = model.val_centres_all_epochs[best_ep]
    ckappa_best  = model.val_cluster_kappa_all_epochs[best_ep]

    y_true = np.asarray(yva, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    ari_check = adjusted_rand_score(y_true, y_pred)     # should equal best_ari

    path = os.path.join(out_dir, f"best_seed{seed}_geometry.npz")
    np.savez(
        path,
        seed=np.int64(seed), best_epoch=np.int64(best_ep),
        best_ari=np.float64(best_ari), ari_from_saved_theta=np.float64(ari_check),
        n_latent=np.int64(mu_best.shape[1]), n_clusters=np.int64(centres_best.shape[0]),
        # vMF encoder
        mu=mu_best.cpu().numpy(), kappa=kappa_best.cpu().numpy(), z=z_best.cpu().numpy(),
        # mixture-of-vMF clustering
        theta=theta_best.cpu().numpy(), centres=centres_best.cpu().numpy(),
        cluster_kappa=ckappa_best.cpu().numpy(),
        # labels
        y_true=y_true, y_pred=y_pred,
    )
    print(f"   [geometry] seed {seed}: saved best-ARI hypersphere geometry "
          f"(ep {best_ep}, ARI {best_ari:.4f}, ARI-from-saved-theta {ari_check:.4f}) "
          f"-> {path}", flush=True)
    return ari_check


# =============================================================================
#  Quick self-test of the vMF machinery (runs only when executed directly).
#  This does NOT touch your data files; it just checks the new components are
#  numerically sane before you plug them into the full training pipeline.
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 70)
    print(" Self-test: von Mises-Fisher sampling + KL on S^{m-1}")
    print("=" * 70)

    for m in [3, 8, 64]:
        B = 2048
        mu = torch.randn(B, m)
        mu = mu / mu.norm(dim=-1, keepdim=True)
        kappa = torch.full((B, 1), 50.0)

        q = VonMisesFisher(mu, kappa)
        z = q.rsample()

        norm_err = (z.norm(dim=-1) - 1.0).abs().max().item()
        # empirical mean resultant length should match I_{m/2}/I_{m/2-1}(kappa)
        mean_dir = z.mean(dim=0)
        cos_to_mu = (z * mu).sum(-1).mean().item()
        r_theory = ive_ratio(m / 2.0, kappa[0]).item()
        kl_val = q.kl_uniform().mean().item()

        print(
            f"  m={m:3d} | ||z||-1 max={norm_err:.2e} | "
            f"E[z.mu]={cos_to_mu:.3f} (theory r={r_theory:.3f}) | "
            f"KL(vMF||U)={kl_val:.3f}"
        )

    print("\n  Encoder smoke test:")
    enc = EncoderHyperspherical(n_input=120, n_latent=64)
    x = torch.rand(16, 120) * 100
    mu, kappa, z, qz = enc(torch.log1p(x))
    print(f"    mu shape {tuple(mu.shape)} | ||mu||={mu.norm(dim=-1).mean():.3f} "
          f"| kappa range [{kappa.min():.2f}, {kappa.max():.2f}] "
          f"| ||z||={z.norm(dim=-1).mean():.3f}")

    print("\n  Decoder (mixture-of-vMF) smoke test:")
    dec = DecoderDBGCM(n_clusters=10, n_input=64, n_output=120, latent_distribution="vmf")
    rho, rho_kl, log_theta, theta, theta_kl = dec(z)
    print(f"    theta shape {tuple(theta.shape)} (rows sum to {theta.sum(1).mean():.3f}) "
          f"| rho_kl={rho_kl.item():.3f} | theta_kl={theta_kl.item():.3f}")

    print("\n  Full model loss smoke test:")
    model = DBayGenCM(n_genes=120, n_latent=64, n_clusters=10, latent_distribution="vmf")
    loss = model.compute_loss(x)
    print(f"    loss = {loss.item():.4f}")
    loss.backward()
    print("    backward() OK  ->  vMF S-VAE is differentiable end-to-end.")

    print("\n  Spherical k-means warm-start smoke test:")
    # build 4 well-separated vMF blobs on S^63 and check k-means recovers them
    m_lat, K_true = 64, 4
    true_centres = F.normalize(torch.randn(K_true, m_lat), dim=-1)
    blobs = []
    true_lab = []
    for k in range(K_true):
        c = true_centres[k].expand(150, m_lat)
        zc = VonMisesFisher(c, torch.full((150, 1), 80.0)).rsample()
        blobs.append(zc)
        true_lab.append(torch.full((150,), k))
    z_all = torch.cat(blobs, 0)
    true_lab = torch.cat(true_lab, 0)
    centres, kappa_k, labels, counts = spherical_kmeans(z_all, n_clusters=K_true,
                                                        n_init=5, seed=0)
    ari_km = adjusted_rand_score(true_lab.numpy(), labels.numpy())
    print(f"    spherical k-means ARI vs true blobs = {ari_km:.3f} "
          f"| recovered kappa range [{kappa_k.min():.1f}, {kappa_k.max():.1f}] "
          f"(true=80) | counts {counts.long().tolist()}")
    assert ari_km > 0.9, "spherical k-means failed to recover well-separated blobs"

    # exercise the decoder warm start end to end
    dec2 = DecoderDBGCM(n_clusters=K_true, n_input=m_lat, n_output=120,
                        latent_distribution="vmf")
    before = dec2.means.data.clone()
    info = dec2.kmeans_warm_start(z_all, n_init=5, seed=0)
    moved = (dec2.means.data - before).abs().max().item()
    pi = dec2.cluster_weights()
    print(f"    decoder.kmeans_warm_start: means moved (max abs) = {moved:.3f} "
          f"| seeded stick weights pi = {[round(p, 3) for p in pi.tolist()]}")
    assert moved > 0, "warm start did not update the cluster centres"
    print("    warm start OK  ->  means/kappa/sticks seeded from spherical k-means.")
    print("\nAll self-tests passed.")
