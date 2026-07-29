#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D hypersphere visualisation of the 1-omics S-VAE (v3) on CDI, for EVERY seed
whose best-ARI geometry was saved by run_cdi_sphere.py into Data/cdi_sphere/.

Same style as the yachida_sphere figures: project the S^63 vMF mode directions
to S^2 (top-3 PCA, renormalised), colour LEFT by true DiseaseState and RIGHT by
predicted cluster (Hungarian-mapped to state names, empirical centres computed
IN the S^2 space and coloured to match their cluster).

CDI has THREE classes (ignore-nonCDI / CDI / H).
"""
import warnings
warnings.filterwarnings("ignore")

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

DATA = "sphere"

LABEL_NAMES = ["ignore-nonCDI", "CDI", "H"]

STATE_COLORS = ["#7f7f7f", "#d62728", "#1f77b4"]  # grey / red / blue

sns.set_theme(style="white", context="paper", font_scale=1.2)


def remap_cluster_labels(pred_labels, true_labels):
    up, ut = np.unique(pred_labels), np.unique(true_labels)
    conf = np.zeros((len(up), len(ut)))
    for i, p in enumerate(up):
        for j, t in enumerate(ut):
            conf[i, j] = np.sum((pred_labels == p) & (true_labels == t))
    r, c = linear_sum_assignment(-conf)
    mapping, assigned = {}, set()
    for i, j in zip(r, c):
        mapping[up[i]] = ut[j]; assigned.add(up[i])
    nxt = int(ut.max()) + 1
    for p in sorted(set(up) - assigned):
        mapping[p] = nxt; nxt += 1
    return np.array([mapping[l] for l in pred_labels])


# wireframe unit globe
_u = np.linspace(0, 2 * np.pi, 40); _v = np.linspace(0, np.pi, 20)
gx = np.outer(np.cos(_u), np.sin(_v))
gy = np.outer(np.sin(_u), np.sin(_v))
gz = np.outer(np.ones_like(_u), np.cos(_v))


def draw_globe(ax):
    ax.plot_wireframe(gx, gy, gz, color="0.8", linewidth=0.4, alpha=0.6)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([-1, 0, 1])
    ax.tick_params(labelsize=15, pad=1)


def state_color(named):
    return STATE_COLORS[int(named)] if int(named) < len(STATE_COLORS) else "0.4"


def visualize_seed(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    mu = np.asarray(d["mu"], np.float64)
    y_val = np.asarray(d["y_true"]).flatten().astype(int)
    theta = np.asarray(d["theta"], np.float64)
    pred = theta.argmax(1)
    ari = float(d["best_ari"]); seed = int(d["seed"]); best_epoch = int(d["best_epoch"])
    n_clusters = int(d["n_clusters"])
    uncertainty = 1.0 - theta.max(1)

    ari_check = adjusted_rand_score(y_val, pred)
    assert abs(ari_check - ari) < 1e-6, (ari_check, ari)

    mu = mu / np.linalg.norm(mu, axis=1, keepdims=True)
    pca = PCA(n_components=3, random_state=0).fit(mu)
    P = pca.transform(mu); P = P / np.linalg.norm(P, axis=1, keepdims=True)
    explained = pca.explained_variance_ratio_

    used = np.unique(pred)
    pred_named = remap_cluster_labels(pred, y_val)

    # centres computed DIRECTLY in the S^2 display space (mean of members,
    # renormalised) so each diamond sits inside its own point cloud.
    C = np.zeros((n_clusters, 3))
    for k in used:
        v = P[pred == k].mean(0); n = np.linalg.norm(v)
        C[k] = v / n if n > 0 else v

    fig = plt.figure(figsize=(13, 6))

    # LEFT: true DiseaseState
    axL = fig.add_subplot(1, 2, 1, projection="3d")
    draw_globe(axL)
    for c in range(len(LABEL_NAMES)):
        m = y_val == c
        if m.any():
            axL.scatter(P[m, 0], P[m, 1], P[m, 2], s=30, depthshade=True,
                        color=STATE_COLORS[c], edgecolors="k", linewidths=0.3,
                        label=LABEL_NAMES[c])
    axL.view_init(elev=22, azim=45)
    axL.set_title("Coloured by true DiseaseState", fontsize=11)
    axL.legend(title="Disease state", fontsize=9, title_fontsize=10, loc="upper left")

    # RIGHT: predicted cluster (coloured by mapped state), matched centres
    axR = fig.add_subplot(1, 2, 2, projection="3d")
    draw_globe(axR)
    for k in used:
        m = pred == k
        col = state_color(pred_named[m][0])
        axR.scatter(P[m, 0], P[m, 1], P[m, 2], s=30, depthshade=True,
                    color=col, edgecolors="k", linewidths=0.3)
        axR.scatter(C[k, 0], C[k, 1], C[k, 2], s=170, marker="D",
                    color=col, edgecolors="k", linewidths=1.4, depthshade=False)
    axR.view_init(elev=22, azim=45)
    axR.set_title(f"Coloured by predicted cluster (ARI = {ari:.2f})", fontsize=11)

    fig.suptitle(f"CDI S-VAE (seed {seed}, ARI = {ari:.2f}): S$^{{63}}$ vMF "
                 f"mixture head uses {len(used)}/{n_clusters} clusters",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    base = os.path.join(DATA, f"seed{seed}_cdi_sphere_true_vs_pred")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


    print(f"seed {seed}: ARI={ari:.4f} ep={best_epoch} | clusters used "
          f"{used.tolist()} | PCA var {np.round(explained, 3).tolist()}")


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(DATA, "best_seed*_geometry.npz")),
                   key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p)))))
    if not files:
        raise SystemExit(f"No geometry npz found in {DATA} -- run run_cdi_sphere.py first.")
    for f in files:
        visualize_seed(f)
    print(f"\nSaved CDI sphere figures for {len(files)} seed(s) to", DATA)
