#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the 1-omics hyperspherical S-VAE (v3) on CDI and save the best-ARI vMF
hypersphere geometry per seed -- the CDI analogue of the yachida_sphere run.

Loads CDI the same way as save_best_seed_for_sphere3d.py (DiseaseState 3-class,
matched microbiome/meta indices, 80/20 stratified split per seed), trains on CPU
(torch.lgamma CUDA kernel is broken on this box), then calls the model file's
own save_best_geometry() to write best_seed{N}_geometry.npz containing
mu/kappa/z (encoder) + theta/centres/cluster_kappa (mixture-of-vMF) + labels,
all indexed at the highest-ARI epoch. Also runs a permutation-null test.

Config: N_SEEDS=3, MAX_EPOCHS=1500 (quick pipeline validation).
"""
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

from omic_1 import (
    DBayesCM, MicrobiomeDataset, save_best_geometry, permutation_null,
)


OUT = os.path.join("sphere")
os.makedirs(OUT, exist_ok=True)

N_SEEDS = 3
MAX_EPOCHS = 1500
N_PERM = 1000
N_LATENT = 64
N_CLUSTERS = 10


def load_data():
    mic = pd.read_csv(os.path.join("microbiome.csv"), index_col="index")
    meta = pd.read_csv(os.path.join("ground_truth.csv"), index_col="sample_id")
    matched = sorted(set(mic.index) & set(meta.index))
    mic, meta = mic.loc[matched], meta.loc[matched]
    labels = meta["DiseaseState"].unique()
    lab_map = {l: i for i, l in enumerate(labels)}
    y = meta["DiseaseState"].map(lab_map).values.astype(np.int64)
    X = mic.values.astype(np.float32)
    return X, y, lab_map


def make_loaders(X, y, seed):
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    to = lambda a, d: torch.tensor(a, dtype=d)
    trl = DataLoader(MicrobiomeDataset(to(Xtr, torch.float32), to(ytr, torch.long)),
                     batch_size=32, shuffle=False, drop_last=False)
    val = DataLoader(MicrobiomeDataset(to(Xva, torch.float32), to(yva, torch.long)),
                     batch_size=32, shuffle=False, drop_last=False)
    return trl, val, Xva, yva


def main():
    X, y, lab_map = load_data()
    print(f"X={X.shape}, classes={lab_map}, counts={np.bincount(y).tolist()}")

    rows = []
    for seed in range(N_SEEDS):
        print(f"\n{'='*70}\n[seed {seed}] ({seed+1}/{N_SEEDS})\n{'='*70}")
        torch.manual_seed(seed); np.random.seed(seed)
        trl, val, Xva, yva = make_loaders(X, y, seed)

        model = DBayesCM(
            n_genes=X.shape[1], n_latent=N_LATENT, n_clusters=N_CLUSTERS,
            n_layers_encoder_individual=2, dim_hidden_encoder=128,
            learning_rate=1e-3, n_samples=100,
            latent_distribution="vmf",
        )
        # progress bar disabled: RichProgressBar crashes on this cp932 console
        trainer = pl.Trainer(max_epochs=MAX_EPOCHS, log_every_n_steps=1,
                             enable_progress_bar=False,
                             accelerator="cpu", devices=1,
                             enable_checkpointing=False, logger=False)
        trainer.fit(model, trl, val)

        ari_hist = np.array(model.ari_history)
        best_ep = int(ari_hist.argmax())
        best_ari = float(ari_hist[best_ep])
        best_theta = model.val_theta_all_epochs[best_ep]
        y_pred = torch.argmax(best_theta, dim=1).numpy()

        rng = np.random.default_rng(10_000 + seed)
        pn = permutation_null(yva, y_pred, N_PERM, rng)

        save_best_geometry(model, seed, best_ep, best_ari, y_pred, yva, out_dir=OUT)

        row = dict(seed=seed, best_ari=best_ari, best_epoch=best_ep,
                   final_ari=float(ari_hist[-1]), null_mean=pn["null_mean"],
                   null_std=pn["null_std"], z_score=pn["z_score"], p_emp=pn["p_emp"])
        rows.append(row)
        print(f"[seed {seed}] best ARI={best_ari:.4f} @ep{best_ep} | "
              f"null={pn['null_mean']:+.4f}±{pn['null_std']:.4f} | "
              f"z={pn['z_score']:.2f} p={pn['p_emp']:.2e}")
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"CDI_per_{seed}.csv"), index=False)

    df = pd.DataFrame(rows)
    a = df["best_ari"].values
    print("\n" + "=" * 70)
    print(f"CDI SUMMARY over {len(a)} seeds: ARI {a.mean():.4f} ± "
          f"{a.std(ddof=1) if len(a) > 1 else 0:.4f} "
          f"(max {a.max():.4f} @ seed {int(df.loc[df['best_ari'].idxmax(),'seed'])})")
    print("=" * 70)
    print("Geometry saved to", OUT, "-> render with visualize_cdi_sphere.py")


if __name__ == "__main__":
    main()
