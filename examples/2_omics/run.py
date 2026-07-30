#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score
import lightning.pytorch as pl
from lightning.pytorch.strategies.ddp import DDPStrategy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks import RichProgressBar

from omics_2 import (
    DBayesCM, MicrobiomeDataset, save_best_geometry, permutation_null,
)


OUT = os.path.join("sphere")
os.makedirs(OUT, exist_ok=True)

N_SEEDS = 20
MAX_EPOCHS = 5000
N_PERM = 1000
N_LATENT = 64
N_CLUSTERS = 10


def load_data():
    X = pd.read_csv(os.path.join("microbiome.csv"))
    Y = pd.read_csv(os.path.join("miRNA.csv"))
    ground_true = pd.read_csv(os.path.join("ground_true.csv"))
    
    X = X.values.astype(np.float32)
    Y = Y.values.astype(np.float32)
    ground_true = ground_true.values.squeeze()
    
    return X, Y, ground_true


def make_loaders(X, Y, y, seed):
    idx = np.arange(len(y))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    to = lambda a, d: torch.tensor(a, dtype=d)
    trl = DataLoader(MicrobiomeDataset(to(X[tr_idx], torch.float32), to(Y[tr_idx], torch.float32),
                                       to(y[tr_idx], torch.long)),
                     batch_size=32, shuffle=False, drop_last=False)
    val = DataLoader(MicrobiomeDataset(to(X[va_idx], torch.float32), to(Y[va_idx], torch.float32),
                                       to(y[va_idx], torch.long)),
                     batch_size=32, shuffle=False, drop_last=False)
    return trl, val, X[va_idx], y[va_idx]



X, Y, ground_true = load_data()

rows = []
for si, seed in enumerate(range(N_SEEDS)):
    print(f"\n{'='*70}\n[seed {seed}]  ({si+1}/{N_SEEDS})\n{'='*70}")
    
    torch.manual_seed(seed); np.random.seed(seed)
    trl, val, Xva, yva = make_loaders(X, Y, ground_true, seed)
    
    n_input_microbiome = X.shape[1] 
    n_input_metabolite = Y.shape[1] 
    
    model = DBayesCM(n_genes=n_input_microbiome, 
                      n_metabolite=n_input_metabolite, 
                      n_latent = 64,
                      n_clusters = 10,
                      n_layers_encoder_individual = 2,
                      dim_hidden_encoder = 128,
                      learning_rate=1e-3, 
                      n_samples=100,
                      alpha_0=0.01,
                      pip0_rho = 0.1, # Higher sparsity 0.05 or 0.01
                      c_reg = 1.0, # Regularization constant for horseshoe smaller values like 0.1 or 0.01 for stronger shrinkage
                      d0 = 1.0,  # Scale parameter for global scale
                      sigma0 = 1.0,
                      use_log_transform = True,
                      use_LayerNorm = True,
                      combine_method = "concat" # add or concat
                      )

    trainer = pl.Trainer(max_epochs=5000, log_every_n_steps=1, enable_progress_bar=True, callbacks=[RichProgressBar()], 
                         accelerator="gpu", devices=1)
    trainer.fit(model, trl, val)
    ari_hist = np.array(model.ari_history)
    best_ep = int(ari_hist.argmax()) 
    best_ari = float(ari_hist[best_ep])
    best_theta = model.val_theta_all_epochs[best_ep]
    y_pred = torch.argmax(best_theta, dim=1).numpy()
    
    rng = np.random.default_rng(10_000 + seed)
    pn = permutation_null(yva, y_pred, N_PERM, rng)

     # save the best-ARI geometry for visualisation
    ari_saved = save_best_geometry(model, seed, best_ep, best_ari, y_pred, yva)

    row = dict(seed=seed, best_ari=best_ari, best_epoch=best_ep,
               final_ari=float(ari_hist[-1]), null_mean=pn["null_mean"], null_std=pn["null_std"],
               z_score=pn["z_score"], p_emp=pn["p_emp"])
    rows.append(row)
    print(f"[seed {seed}] best ARI={best_ari:.4f} @ep{best_ep} | "
          f"null={pn['null_mean']:+.4f}±{pn['null_std']:.4f} | "
          f"z={pn['z_score']:.2f} p={pn['p_emp']:.2e}")
    pd.DataFrame(rows).to_csv(f"TCGA_per_{seed}.csv", index=False)


