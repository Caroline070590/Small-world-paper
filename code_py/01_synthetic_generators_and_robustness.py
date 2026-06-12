#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto-converted from the Jupyter notebook ``all-funtions.ipynb``.

Code cells are reproduced verbatim. Markdown cells are kept as comments.
Jupyter shell-magic lines (if any) are commented out so the file is valid
Python; they are preserved for reference. Cell boundaries are marked with
``# In[...]`` to match the original notebook ordering.
"""


# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch

# =====================================================
# 0) Global settings
# =====================================================
Q = 12
N = 3000
N_REALIZATIONS = 30

# For small Q, use higher density so clustering/path length are meaningful
TARGET_DENSITY   = 0.50
N_RANDOMIZATIONS = 50
REWIRES_PER_EDGE = 5

# =====================================================
# 1) QG (your QTN / Quantile Graph), GAF, and MTF (QxQ)
# =====================================================

# ---- QG / QTN counts (QxQ) ----
K_VALUES = [1, 2, 3]  # lags

def qg_qtn_counts(signal: np.ndarray, Q: int, k_values=K_VALUES) -> np.ndarray:
    """
    Quantile Graph (your QTN): nodes=quantile bins, edges=count of transitions.
    Returns a QxQ count matrix (directed). We'll symmetrize when we treat it as an undirected graph.
    """
    x = np.asarray(signal, dtype=float)
    n = x.size
    A = np.zeros((Q, Q), dtype=np.int64)
    if n <= 1:
        return A

    ranks = np.argsort(np.argsort(x))
    q_edges = np.linspace(0, n, Q + 1, dtype=int)
    loc = np.clip(np.searchsorted(q_edges, ranks, side="right") - 1, 0, Q - 1)

    for k in k_values:
        k = int(k)
        if k <= 0 or k >= n:
            continue
        for i in range(n - k):
            A[loc[i], loc[i + k]] += 1
    return A

# ---- GAF (QxQ) ----
def gaf_matrix(signal: np.ndarray, Q: int) -> np.ndarray:
    """
    Compute GAF on a signal by first downsampling/interpolating to length Q,
    then applying the GAF transform -> QxQ.
    """
    x = np.asarray(signal, dtype=float)
    xQ = downsample_to_length(x, Q)

    min_val, max_val = float(np.min(xQ)), float(np.max(xQ))
    rng = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    scaled = 2 * (xQ - min_val) / rng - 1
    scaled = np.clip(scaled, -1, 1)

    phi = np.arccos(scaled)
    return np.cos(phi[:, None] + phi[None, :])

# ---- MTF transition matrix P (QxQ), quantile binning ----
BIN_MODE   = "quantile"
SMOOTH_EPS = 1e-3

def _states_from_01(z01: np.ndarray, Q: int, bin_mode: str) -> np.ndarray:
    if bin_mode == "quantile":
        edges = np.quantile(z01, np.linspace(0.0, 1.0, Q + 1))
        edges = np.unique(edges)
        if edges.size < Q + 1:
            edges = np.linspace(0.0, 1.0, Q + 1)
    else:
        edges = np.linspace(0.0, 1.0, Q + 1)
    s = np.digitize(z01, edges[1:-1], right=True)
    return np.clip(s, 0, Q - 1)

def mtf_transition_matrix(signal: np.ndarray, Q: int,
                          bin_mode: str = BIN_MODE,
                          smooth_eps: float = SMOOTH_EPS) -> np.ndarray:
    """
    Returns QxQ transition probability matrix P (Markov transition matrix)
    using quantile bins by default.
    """
    x = np.asarray(signal, dtype=float)
    T = x.size
    if T < 2:
        return np.ones((Q, Q), dtype=float) / Q

    xmin, xmax = float(np.min(x)), float(np.max(x))
    rng = (xmax - xmin) if (xmax - xmin) != 0 else 1.0
    z01 = (x - xmin) / rng

    s = _states_from_01(z01, Q, bin_mode)

    C = np.zeros((Q, Q), dtype=float)
    for t in range(T - 1):
        C[s[t], s[t + 1]] += 1.0

    if smooth_eps and smooth_eps > 0.0:
        C += smooth_eps

    row_sums = C.sum(axis=1, keepdims=True)
    P = np.empty_like(C)
    nz = row_sums.squeeze() > 0
    P[nz]  = C[nz] / row_sums[nz]
    P[~nz] = 1.0 / Q
    return P

# =====================================================
# 2) Synthetic time-series generators (your set)
# =====================================================
def _powerlaw_noise(n, beta, rng):
    x = rng.normal(size=n)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n)
    S = np.ones_like(freqs)
    S[1:] = freqs[1:] ** (-beta / 2.0)
    x_pl = np.fft.irfft(X * S, n=n)
    x_pl -= x_pl.mean()
    x_pl /= (x_pl.std() + 1e-12)
    return x_pl

def _logistic_map(n, r, rng, burn=1000):
    x = rng.uniform(0, 1)
    for _ in range(burn):
        x = r * x * (1.0 - x)
    xs = np.empty(n, dtype=float)
    for i in range(n):
        x = r * x * (1.0 - x)
        xs[i] = x
    xs -= xs.mean()
    xs /= (xs.std() + 1e-12)
    return xs

def generate_ts(kind="iid_normal", n=2000, seed=0, **kwargs):
    rng = np.random.default_rng(seed)

    if kind == "iid_normal":
        mu = kwargs.get("mu", 0.0)
        sigma = kwargs.get("sigma", 1.0)
        x = rng.normal(mu, sigma, size=n)

    elif kind == "ar1":
        phi = kwargs.get("phi", 0.9)
        sigma = kwargs.get("sigma", 1.0)
        x = np.zeros(n)
        eps = rng.normal(0, sigma, size=n)
        for t in range(1, n):
            x[t] = phi * x[t-1] + eps[t]

    elif kind == "sinusoid_noise":
        freq = kwargs.get("freq", 0.02)
        sigma = kwargs.get("sigma", 0.2)
        t = np.arange(n)
        x = np.sin(2 * np.pi * freq * t) + rng.normal(0, sigma, size=n)

    elif kind == "regime_switching":
        phi1 = kwargs.get("phi1", 0.2)
        phi2 = kwargs.get("phi2", 0.9)
        p_switch = kwargs.get("p_switch", 0.01)
        sigma = kwargs.get("sigma", 1.0)
        x = np.zeros(n)
        r = np.zeros(n, dtype=int)
        eps = rng.normal(0, sigma, size=n)
        for t in range(1, n):
            if rng.uniform() < p_switch:
                r[t] = 1 - r[t-1]
            else:
                r[t] = r[t-1]
            phi = phi1 if r[t] == 0 else phi2
            x[t] = phi * x[t-1] + eps[t]

    elif kind == "pink_noise":
        beta = kwargs.get("beta", 1.0)
        x = _powerlaw_noise(n, beta=beta, rng=rng)

    elif kind == "logistic_map":
        r_par = kwargs.get("r", 4.0)
        x = _logistic_map(n, r=r_par, rng=rng)

    elif kind == "fbm":
        H = kwargs.get("H", 0.7)
        beta = 2.0 * H + 1.0
        x = _powerlaw_noise(n, beta=beta, rng=rng)

    else:
        raise ValueError(f"Unknown kind: {kind}")

    return x

# =====================================================
# 3) Utilities: downsample + small-world sigma
# =====================================================
def downsample_to_length(x: np.ndarray, L: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if L <= 1:
        return x[:1].copy()
    n = x.size
    if n == L:
        return x.copy()
    if n < 2:
        return np.full(L, float(x[0]) if n == 1 else 0.0, dtype=float)
    xp = np.linspace(0.0, 1.0, n)
    return np.interp(np.linspace(0.0, 1.0, L), xp, x).astype(float)

def proportional_binary_from_weights(W: np.ndarray, target_density: float) -> np.ndarray:
    n = W.shape[0]
    A = np.abs(W).astype(float).copy()
    np.fill_diagonal(A, 0.0)

    upper = np.triu(A, 1)
    vals = upper[upper > 0]
    if vals.size == 0:
        return np.zeros((n, n), dtype=int)

    m_target = int(round(target_density * n * (n - 1) / 2))
    m_target = max(1, min(m_target, vals.size))
    thresh = np.partition(vals, -m_target)[-m_target]

    B = (A >= thresh).astype(int)
    B = np.triu(B, 1)
    B = B + B.T
    np.fill_diagonal(B, 0)
    return B

def char_path_length_gcc(B: np.ndarray) -> float:
    G = nx.from_numpy_array(B)
    if G.number_of_edges() == 0:
        return np.nan
    comps = list(nx.connected_components(G))
    if not comps:
        return np.nan
    largest = G.subgraph(max(comps, key=len)).copy()
    if largest.number_of_nodes() < 2 or largest.number_of_edges() == 0:
        return np.nan
    return nx.average_shortest_path_length(largest)

def mean_clustering(B: np.ndarray) -> float:
    G = nx.from_numpy_array(B)
    if G.number_of_edges() == 0:
        return np.nan
    return float(np.mean(list(nx.clustering(G).values())))

def null_model_stats(B: np.ndarray, n_rand: int, rewires_per_edge: int, seed: int):
    m = int(B.sum() // 2)
    swaps = max(1, rewires_per_edge * m)
    rng = np.random.default_rng(seed)

    C_list, L_list = [], []
    for _ in range(n_rand):
        G = nx.from_numpy_array(B)
        if G.number_of_edges() == 0:
            continue
        try:
            nx.double_edge_swap(
                G, nswap=swaps, max_tries=10 * swaps,
                seed=int(rng.integers(0, 1_000_000))
            )
        except Exception:
            pass
        Br = nx.to_numpy_array(G, dtype=int)
        C_list.append(mean_clustering(Br))
        L_list.append(char_path_length_gcc(Br))

    C_rand = np.nanmean(C_list) if C_list else np.nan
    L_rand = np.nanmean(L_list) if L_list else np.nan
    return C_rand, L_rand

def small_world_sigma_from_W(W: np.ndarray, seed: int = 0) -> float:
    B = proportional_binary_from_weights(W, TARGET_DENSITY)
    C_obs = mean_clustering(B)
    L_obs = char_path_length_gcc(B)
    C_rand, L_rand = null_model_stats(B, N_RANDOMIZATIONS, REWIRES_PER_EDGE, seed)

    gamma  = C_obs / C_rand if (np.isfinite(C_obs) and np.isfinite(C_rand) and C_rand > 0) else np.nan
    lambd  = L_obs / L_rand if (np.isfinite(L_obs) and np.isfinite(L_rand) and L_rand > 0) else np.nan
    sigma  = gamma / lambd if (np.isfinite(gamma) and np.isfinite(lambd) and lambd != 0) else np.nan
    return sigma

# =====================================================
# 4) Compute σ for QG(QTN), GAF, MTF across variants
# =====================================================
CONFIGS = [
    ("iid",            "iid_normal",       {},                                "i.i.d. Gaussian"),
    ("pink",           "pink_noise",       {"beta": 1.0},                      "Pink noise (1/f)"),
    ("ar_phi0.5",      "ar1",              {"phi": 0.5},                       "AR(1) φ=0.5"),
    ("ar_phi0.9",      "ar1",              {"phi": 0.9},                       "AR(1) φ=0.9"),
    ("sin_low",        "sinusoid_noise",   {"freq": 0.01},                     "Sinusoid f=0.01"),
    ("sin_high",       "sinusoid_noise",   {"freq": 0.05},                     "Sinusoid f=0.05"),
    ("logistic",       "logistic_map",     {"r": 4.0},                         "Logistic map r=4"),
    ("fbm_H0.3",       "fbm",              {"H": 0.3},                         "fBm-like H=0.3"),
    ("fbm_H0.7",       "fbm",              {"H": 0.7},                         "fBm-like H=0.7"),
    ("regime_rare",    "regime_switching", {"phi1":0.2,"phi2":0.95,"p_switch":0.005}, "Regime p=0.005"),
    ("regime_often",   "regime_switching", {"phi1":0.2,"phi2":0.95,"p_switch":0.05},  "Regime p=0.05"),
]

def compute_sigmas_for_all_methods(Q=12, n=3000, n_realizations=30):
    labels = [lbl for *_rest, lbl in CONFIGS]
    sigma_qg, sigma_gaf, sigma_mtf = [], [], []

    for cfg_idx, (cfg_id, kind, params, label) in enumerate(CONFIGS):
        qg_list, gaf_list, mtf_list = [], [], []
        for seed in range(n_realizations):
            x = generate_ts(kind=kind, n=n, seed=seed, **params)

            # QG/QTN
            A = qg_qtn_counts(x, Q=Q, k_values=K_VALUES)
            W_qg = (A + A.T).astype(float)
            s_qg = small_world_sigma_from_W(W_qg, seed=10_000*cfg_idx + seed)
            if np.isfinite(s_qg):
                qg_list.append(s_qg)

            # GAF
            W_gaf = gaf_matrix(x, Q=Q)
            s_gaf = small_world_sigma_from_W(W_gaf, seed=20_000*cfg_idx + seed)
            if np.isfinite(s_gaf):
                gaf_list.append(s_gaf)

            # MTF (transition matrix)
            W_mtf = mtf_transition_matrix(x, Q=Q, bin_mode="quantile", smooth_eps=SMOOTH_EPS)
            s_mtf = small_world_sigma_from_W(W_mtf, seed=30_000*cfg_idx + seed)
            if np.isfinite(s_mtf):
                mtf_list.append(s_mtf)

        sigma_qg.append(qg_list)
        sigma_gaf.append(gaf_list)
        sigma_mtf.append(mtf_list)

    return labels, sigma_qg, sigma_gaf, sigma_mtf

# =====================================================
# 5) Figure A: Example time series + (QG, GAF, MTF) matrices
# =====================================================
def plot_examples_ts_and_matrices(Q=12, n=3000, show_T=400, outfile_prefix="examples_qg_gaf_mtf"):
    examples = [
        ("iid_normal",      {},              "i.i.d. Gaussian"),
        ("pink_noise",      {"beta": 1.0},   "Pink noise"),
        ("ar1",             {"phi": 0.9},    "AR(1) φ=0.9"),
        ("sinusoid_noise",  {"freq": 0.02},  "Sinusoid f=0.02"),
        ("logistic_map",    {"r": 4.0},      "Logistic map r=4"),
        ("fbm",             {"H": 0.7},      "fBm-like H=0.7"),
    ]

    rows = []
    for idx, (kind, params, label) in enumerate(examples):
        x = generate_ts(kind=kind, n=n, seed=idx, **params)

        A = qg_qtn_counts(x, Q=Q, k_values=K_VALUES)
        W_qg = (A + A.T).astype(float)

        W_gaf = gaf_matrix(x, Q=Q)
        W_mtf = mtf_transition_matrix(x, Q=Q, bin_mode="quantile", smooth_eps=SMOOTH_EPS)

        rows.append((x, W_qg, W_gaf, W_mtf, label))

    # Normalize QG counts for display only (not for metric computations)
    qg_max = max(np.max(Wqg) for _x, Wqg, _g, _m, _l in rows)
    if qg_max <= 0:
        qg_max = 1.0

    # common vmax for heatmaps (each method separately to keep contrast)
    gaf_vmax = max(np.max(Wgaf) for _x, _qg, Wgaf, _m, _l in rows)
    mtf_vmax = max(np.max(Wmtf) for _x, _qg, _g, Wmtf, _l in rows)

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 4, figsize=(14, 2.4 * nrows), tight_layout=True)

    for i, (x, Wqg, Wgaf, Wmtf, label) in enumerate(rows):
        # time series
        ax = axes[i, 0]
        t = np.arange(show_T)
        ax.plot(t, x[:show_T])
        ax.set_title(label)
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Value")

        # QG/QTN adjacency (counts, sym)
        ax = axes[i, 1]
        im1 = ax.imshow(Wqg / qg_max, origin="lower", aspect="equal", vmin=0.0, vmax=1.0)
        ax.set_title(f"QG (QTN) adjacency (Q={Q})")
        ax.set_xlabel("Bin j")
        ax.set_ylabel("Bin i")
        fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

        # GAF
        ax = axes[i, 2]
        im2 = ax.imshow(Wgaf, origin="lower", aspect="equal", vmin=-1.0, vmax=max(1.0, gaf_vmax))
        ax.set_title(f"GAF (Q={Q})")
        ax.set_xlabel("i")
        ax.set_ylabel("j")
        fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

        # MTF transition matrix
        ax = axes[i, 3]
        im3 = ax.imshow(Wmtf, origin="lower", aspect="equal", vmin=0.0, vmax=max(1e-12, mtf_vmax))
        ax.set_title(f"MTF transition P (Q={Q})")
        ax.set_xlabel("Next state j")
        ax.set_ylabel("Current state i")
        fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Examples: time series + QG(QTN), GAF, and MTF matrices", y=1.02)
    plt.savefig(f"{outfile_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{outfile_prefix}.pdf", dpi=300, bbox_inches="tight")
    plt.show()

# =====================================================
# 6) Figure B: One plot with boxplots for σ (QG vs GAF vs MTF) per variant
# =====================================================
def plot_grouped_sigma_boxplot(labels, sigma_qg, sigma_gaf, sigma_mtf,
                               outfile_prefix="sigma_grouped_qg_gaf_mtf"):
    n_models = len(labels)
    base = np.arange(n_models)

    # grouped positions
    offset = 0.25
    pos_qg  = base - offset
    pos_gaf = base
    pos_mtf = base + offset

    plt.figure(figsize=(14, 5))

    # Boxplot styling
    def _boxplot(data, positions, facecolor, label):
        bp = plt.boxplot(
            data,
            positions=positions,
            widths=0.22,
            patch_artist=True,
            showmeans=True,
            manage_ticks=False
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(facecolor)
        for median in bp["medians"]:
            median.set_linewidth(1.5)
        return bp

    # user explicitly asked different colours per function
    _boxplot(sigma_qg,  pos_qg,  facecolor="#4C78A8", label="QG/QTN")
    _boxplot(sigma_gaf, pos_gaf, facecolor="#F58518", label="GAF")
    _boxplot(sigma_mtf, pos_mtf, facecolor="#54A24B", label="MTF")

    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.ylabel("Small-world index σ")
    plt.title("Small-world index σ by model variant and representation (QG/QTN vs GAF vs MTF)")

    plt.xticks(base, labels, rotation=25, ha="right")

    legend_handles = [
        Patch(facecolor="#4C78A8", label="QG/QTN"),
        Patch(facecolor="#F58518", label="GAF"),
        Patch(facecolor="#54A24B", label="MTF"),
    ]
    plt.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    plt.savefig(f"{outfile_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{outfile_prefix}.pdf", dpi=300, bbox_inches="tight")
    plt.show()

# =====================================================
# 7) Main
# =====================================================
if __name__ == "__main__":
    # Figure A: example TS + matrices
    plot_examples_ts_and_matrices(Q=Q, n=N, show_T=400)

    # Figure B: grouped boxplot of σ for QG/QTN vs GAF vs MTF
    labels, sig_qg, sig_gaf, sig_mtf = compute_sigmas_for_all_methods(Q=Q, n=N, n_realizations=N_REALIZATIONS)
    plot_grouped_sigma_boxplot(labels, sig_qg, sig_gaf, sig_mtf)

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch

# =====================================================
# 0) Global settings
# =====================================================
Q = 12
N = 3000
N_REALIZATIONS = 30

# For small Q, use higher density so clustering/path length are meaningful
TARGET_DENSITY   = 0.50
N_RANDOMIZATIONS = 50
REWIRES_PER_EDGE = 5

# Keep your representation colors
PALETTE_DATASET = {
    "QTN": "#008080",   # teal
    "GAF": "#A9A9A9",   # darkgray
    "MTF": "#DDA0DD",   # plum
}

# =====================================================
# 1) QG (your QTN / Quantile Graph), GAF, and MTF (QxQ)
# =====================================================

# ---- QG / QTN counts (QxQ) ----
K_VALUES = [1, 2, 3]  # lags

def qg_qtn_counts(signal: np.ndarray, Q: int, k_values=K_VALUES) -> np.ndarray:
    """
    Quantile Graph (your QTN/QG): nodes=quantile bins, edges=count of transitions.
    Returns a QxQ count matrix (directed). We'll symmetrize for undirected graph metrics.
    """
    x = np.asarray(signal, dtype=float)
    n = x.size
    A = np.zeros((Q, Q), dtype=np.int64)
    if n <= 1:
        return A

    ranks = np.argsort(np.argsort(x))
    q_edges = np.linspace(0, n, Q + 1, dtype=int)
    loc = np.clip(np.searchsorted(q_edges, ranks, side="right") - 1, 0, Q - 1)

    for k in k_values:
        k = int(k)
        if k <= 0 or k >= n:
            continue
        for i in range(n - k):
            A[loc[i], loc[i + k]] += 1
    return A

# ---- GAF (QxQ) ----
def gaf_matrix(signal: np.ndarray, Q: int) -> np.ndarray:
    """
    Compute GAF on a signal by first downsampling/interpolating to length Q,
    then applying the GAF transform -> QxQ.
    """
    x = np.asarray(signal, dtype=float)
    xQ = downsample_to_length(x, Q)

    min_val, max_val = float(np.min(xQ)), float(np.max(xQ))
    rng = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    scaled = 2 * (xQ - min_val) / rng - 1
    scaled = np.clip(scaled, -1, 1)

    phi = np.arccos(scaled)
    return np.cos(phi[:, None] + phi[None, :])

# ---- MTF transition matrix P (QxQ), quantile binning ----
BIN_MODE   = "quantile"
SMOOTH_EPS = 1e-3

def _states_from_01(z01: np.ndarray, Q: int, bin_mode: str) -> np.ndarray:
    if bin_mode == "quantile":
        edges = np.quantile(z01, np.linspace(0.0, 1.0, Q + 1))
        edges = np.unique(edges)
        if edges.size < Q + 1:
            edges = np.linspace(0.0, 1.0, Q + 1)
    else:
        edges = np.linspace(0.0, 1.0, Q + 1)
    s = np.digitize(z01, edges[1:-1], right=True)
    return np.clip(s, 0, Q - 1)

def mtf_transition_matrix(signal: np.ndarray, Q: int,
                          bin_mode: str = BIN_MODE,
                          smooth_eps: float = SMOOTH_EPS) -> np.ndarray:
    """
    Returns QxQ transition probability matrix P (Markov transition matrix)
    using quantile bins by default.
    """
    x = np.asarray(signal, dtype=float)
    T = x.size
    if T < 2:
        return np.ones((Q, Q), dtype=float) / Q

    xmin, xmax = float(np.min(x)), float(np.max(x))
    rng = (xmax - xmin) if (xmax - xmin) != 0 else 1.0
    z01 = (x - xmin) / rng

    s = _states_from_01(z01, Q, bin_mode)

    C = np.zeros((Q, Q), dtype=float)
    for t in range(T - 1):
        C[s[t], s[t + 1]] += 1.0

    if smooth_eps and smooth_eps > 0.0:
        C += smooth_eps

    row_sums = C.sum(axis=1, keepdims=True)
    P = np.empty_like(C)
    nz = row_sums.squeeze() > 0
    P[nz]  = C[nz] / row_sums[nz]
    P[~nz] = 1.0 / Q
    return P

# =====================================================
# 2) Synthetic time-series generators (your set)
# =====================================================
def _powerlaw_noise(n, beta, rng):
    x = rng.normal(size=n)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n)
    S = np.ones_like(freqs)
    S[1:] = freqs[1:] ** (-beta / 2.0)
    x_pl = np.fft.irfft(X * S, n=n)
    x_pl -= x_pl.mean()
    x_pl /= (x_pl.std() + 1e-12)
    return x_pl

def _logistic_map(n, r, rng, burn=1000):
    x = rng.uniform(0, 1)
    for _ in range(burn):
        x = r * x * (1.0 - x)
    xs = np.empty(n, dtype=float)
    for i in range(n):
        x = r * x * (1.0 - x)
        xs[i] = x
    xs -= xs.mean()
    xs /= (xs.std() + 1e-12)
    return xs

def generate_ts(kind="iid_normal", n=2000, seed=0, **kwargs):
    rng = np.random.default_rng(seed)

    if kind == "iid_normal":
        mu = kwargs.get("mu", 0.0)
        sigma = kwargs.get("sigma", 1.0)
        x = rng.normal(mu, sigma, size=n)

    elif kind == "ar1":
        phi = kwargs.get("phi", 0.9)
        sigma = kwargs.get("sigma", 1.0)
        x = np.zeros(n)
        eps = rng.normal(0, sigma, size=n)
        for t in range(1, n):
            x[t] = phi * x[t-1] + eps[t]

    elif kind == "sinusoid_noise":
        freq = kwargs.get("freq", 0.02)
        sigma = kwargs.get("sigma", 0.2)
        t = np.arange(n)
        x = np.sin(2 * np.pi * freq * t) + rng.normal(0, sigma, size=n)

    elif kind == "regime_switching":
        phi1 = kwargs.get("phi1", 0.2)
        phi2 = kwargs.get("phi2", 0.9)
        p_switch = kwargs.get("p_switch", 0.01)
        sigma = kwargs.get("sigma", 1.0)
        x = np.zeros(n)
        r = np.zeros(n, dtype=int)
        eps = rng.normal(0, sigma, size=n)
        for t in range(1, n):
            if rng.uniform() < p_switch:
                r[t] = 1 - r[t-1]
            else:
                r[t] = r[t-1]
            phi = phi1 if r[t] == 0 else phi2
            x[t] = phi * x[t-1] + eps[t]

    elif kind == "pink_noise":
        beta = kwargs.get("beta", 1.0)
        x = _powerlaw_noise(n, beta=beta, rng=rng)

    elif kind == "logistic_map":
        r_par = kwargs.get("r", 4.0)
        x = _logistic_map(n, r=r_par, rng=rng)

    elif kind == "fbm":
        H = kwargs.get("H", 0.7)
        beta = 2.0 * H + 1.0
        x = _powerlaw_noise(n, beta=beta, rng=rng)

    else:
        raise ValueError(f"Unknown kind: {kind}")

    return x

# =====================================================
# 3) Utilities: downsample + small-world sigma
# =====================================================
def downsample_to_length(x: np.ndarray, L: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if L <= 1:
        return x[:1].copy()
    n = x.size
    if n == L:
        return x.copy()
    if n < 2:
        return np.full(L, float(x[0]) if n == 1 else 0.0, dtype=float)
    xp = np.linspace(0.0, 1.0, n)
    return np.interp(np.linspace(0.0, 1.0, L), xp, x).astype(float)

def proportional_binary_from_weights(W: np.ndarray, target_density: float) -> np.ndarray:
    n = W.shape[0]
    A = np.abs(W).astype(float).copy()
    np.fill_diagonal(A, 0.0)

    upper = np.triu(A, 1)
    vals = upper[upper > 0]
    if vals.size == 0:
        return np.zeros((n, n), dtype=int)

    m_target = int(round(target_density * n * (n - 1) / 2))
    m_target = max(1, min(m_target, vals.size))
    thresh = np.partition(vals, -m_target)[-m_target]

    B = (A >= thresh).astype(int)
    B = np.triu(B, 1)
    B = B + B.T
    np.fill_diagonal(B, 0)
    return B

def char_path_length_gcc(B: np.ndarray) -> float:
    G = nx.from_numpy_array(B)
    if G.number_of_edges() == 0:
        return np.nan
    comps = list(nx.connected_components(G))
    if not comps:
        return np.nan
    largest = G.subgraph(max(comps, key=len)).copy()
    if largest.number_of_nodes() < 2 or largest.number_of_edges() == 0:
        return np.nan
    return nx.average_shortest_path_length(largest)

def mean_clustering(B: np.ndarray) -> float:
    G = nx.from_numpy_array(B)
    if G.number_of_edges() == 0:
        return np.nan
    return float(np.mean(list(nx.clustering(G).values())))

def null_model_stats(B: np.ndarray, n_rand: int, rewires_per_edge: int, seed: int):
    m = int(B.sum() // 2)
    swaps = max(1, rewires_per_edge * m)
    rng = np.random.default_rng(seed)

    C_list, L_list = [], []
    for _ in range(n_rand):
        G = nx.from_numpy_array(B)
        if G.number_of_edges() == 0:
            continue
        try:
            nx.double_edge_swap(
                G, nswap=swaps, max_tries=10 * swaps,
                seed=int(rng.integers(0, 1_000_000))
            )
        except Exception:
            pass
        Br = nx.to_numpy_array(G, dtype=int)
        C_list.append(mean_clustering(Br))
        L_list.append(char_path_length_gcc(Br))

    C_rand = np.nanmean(C_list) if C_list else np.nan
    L_rand = np.nanmean(L_list) if L_list else np.nan
    return C_rand, L_rand

def small_world_sigma_from_W(W: np.ndarray, seed: int = 0) -> float:
    B = proportional_binary_from_weights(W, TARGET_DENSITY)
    C_obs = mean_clustering(B)
    L_obs = char_path_length_gcc(B)
    C_rand, L_rand = null_model_stats(B, N_RANDOMIZATIONS, REWIRES_PER_EDGE, seed)

    gamma  = C_obs / C_rand if (np.isfinite(C_obs) and np.isfinite(C_rand) and C_rand > 0) else np.nan
    lambd  = L_obs / L_rand if (np.isfinite(L_obs) and np.isfinite(L_rand) and L_rand > 0) else np.nan
    sigma  = gamma / lambd if (np.isfinite(gamma) and np.isfinite(lambd) and lambd != 0) else np.nan
    return sigma

# =====================================================
# 4) Compute σ for QTN(QG), GAF, MTF across variants
# =====================================================
CONFIGS = [
    ("iid",            "iid_normal",       {},                                "i.i.d. Gaussian"),
    ("pink",           "pink_noise",       {"beta": 1.0},                      "Pink noise (1/f)"),
    ("ar_phi0.5",      "ar1",              {"phi": 0.5},                       "AR(1) φ=0.5"),
    ("ar_phi0.9",      "ar1",              {"phi": 0.9},                       "AR(1) φ=0.9"),
    ("sin_low",        "sinusoid_noise",   {"freq": 0.01},                     "Sinusoid f=0.01"),
    ("sin_high",       "sinusoid_noise",   {"freq": 0.05},                     "Sinusoid f=0.05"),
    ("logistic",       "logistic_map",     {"r": 4.0},                         "Logistic map r=4"),
    ("fbm_H0.3",       "fbm",              {"H": 0.3},                         "fBm-like H=0.3"),
    ("fbm_H0.7",       "fbm",              {"H": 0.7},                         "fBm-like H=0.7"),
    ("regime_rare",    "regime_switching", {"phi1":0.2,"phi2":0.95,"p_switch":0.005}, "Regime p=0.005"),
    ("regime_often",   "regime_switching", {"phi1":0.2,"phi2":0.95,"p_switch":0.05},  "Regime p=0.05"),
]

def compute_sigmas_for_all_methods(Q=12, n=3000, n_realizations=30):
    labels = [lbl for *_rest, lbl in CONFIGS]
    sigma_qtn, sigma_gaf, sigma_mtf = [], [], []

    for cfg_idx, (_cfg_id, kind, params, _label) in enumerate(CONFIGS):
        qtn_list, gaf_list, mtf_list = [], [], []
        for seed in range(n_realizations):
            x = generate_ts(kind=kind, n=n, seed=seed, **params)

            # QTN / QG
            A = qg_qtn_counts(x, Q=Q, k_values=K_VALUES)
            W_qtn = (A + A.T).astype(float)
            s_qtn = small_world_sigma_from_W(W_qtn, seed=10_000*cfg_idx + seed)
            if np.isfinite(s_qtn):
                qtn_list.append(s_qtn)

            # GAF
            W_gaf = gaf_matrix(x, Q=Q)
            s_gaf = small_world_sigma_from_W(W_gaf, seed=20_000*cfg_idx + seed)
            if np.isfinite(s_gaf):
                gaf_list.append(s_gaf)

            # MTF
            W_mtf = mtf_transition_matrix(x, Q=Q, bin_mode="quantile", smooth_eps=SMOOTH_EPS)
            s_mtf = small_world_sigma_from_W(W_mtf, seed=30_000*cfg_idx + seed)
            if np.isfinite(s_mtf):
                mtf_list.append(s_mtf)

        sigma_qtn.append(qtn_list)
        sigma_gaf.append(gaf_list)
        sigma_mtf.append(mtf_list)

    return labels, sigma_qtn, sigma_gaf, sigma_mtf

# =====================================================
# 5) Plot 1: Example time series + (QTN/QG, GAF, MTF) matrices
# =====================================================
def plot_examples_ts_and_matrices(Q=12, n=3000, show_T=400, outfile_prefix="examples_qtn_gaf_mtf"):
    examples = [
        ("iid_normal",      {},              "i.i.d. Gaussian"),
        ("pink_noise",      {"beta": 1.0},   "Pink noise"),
        ("ar1",             {"phi": 0.9},    "AR(1) φ=0.9"),
        ("sinusoid_noise",  {"freq": 0.02},  "Sinusoid f=0.02"),
        ("logistic_map",    {"r": 4.0},      "Logistic map r=4"),
        ("fbm",             {"H": 0.7},      "fBm-like H=0.7"),
    ]

    rows = []
    for idx, (kind, params, label) in enumerate(examples):
        x = generate_ts(kind=kind, n=n, seed=idx, **params)

        A = qg_qtn_counts(x, Q=Q, k_values=K_VALUES)
        W_qtn = (A + A.T).astype(float)

        W_gaf = gaf_matrix(x, Q=Q)
        W_mtf = mtf_transition_matrix(x, Q=Q, bin_mode="quantile", smooth_eps=SMOOTH_EPS)

        rows.append((x, W_qtn, W_gaf, W_mtf, label))

    # Normalize QTN counts for display only
    qtn_max = max(np.max(Wqtn) for _x, Wqtn, _g, _m, _l in rows)
    if qtn_max <= 0:
        qtn_max = 1.0

    gaf_vmax = max(np.max(Wgaf) for _x, _q, Wgaf, _m, _l in rows)
    mtf_vmax = max(np.max(Wmtf) for _x, _q, _g, Wmtf, _l in rows)

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 4, figsize=(14, 2.4 * nrows), tight_layout=True)

    for i, (x, Wqtn, Wgaf, Wmtf, label) in enumerate(rows):
        ax_ts = axes[i, 0]
        t = np.arange(show_T)
        ax_ts.plot(t, x[:show_T])
        ax_ts.set_title(label)
        ax_ts.set_xlabel("Time (samples)")
        ax_ts.set_ylabel("Value")

        ax_qtn = axes[i, 1]
        im1 = ax_qtn.imshow(Wqtn / qtn_max, origin="lower", aspect="equal", vmin=0.0, vmax=1.0)
        ax_qtn.set_title(f"QTN/QG (Q={Q})")
        ax_qtn.set_xlabel("Bin j")
        ax_qtn.set_ylabel("Bin i")
        fig.colorbar(im1, ax=ax_qtn, fraction=0.046, pad=0.04)

        ax_gaf = axes[i, 2]
        im2 = ax_gaf.imshow(Wgaf, origin="lower", aspect="equal", vmin=-1.0, vmax=max(1.0, gaf_vmax))
        ax_gaf.set_title(f"GAF (Q={Q})")
        ax_gaf.set_xlabel("i")
        ax_gaf.set_ylabel("j")
        fig.colorbar(im2, ax=ax_gaf, fraction=0.046, pad=0.04)

        ax_mtf = axes[i, 3]
        im3 = ax_mtf.imshow(Wmtf, origin="lower", aspect="equal", vmin=0.0, vmax=max(1e-12, mtf_vmax))
        ax_mtf.set_title(f"MTF (Q={Q})")
        ax_mtf.set_xlabel("Next state j")
        ax_mtf.set_ylabel("Current state i")
        fig.colorbar(im3, ax=ax_mtf, fraction=0.046, pad=0.04)

    fig.suptitle("Examples: time series + QTN(QG), GAF, and MTF matrices", y=1.02)
    plt.savefig(f"{outfile_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{outfile_prefix}.pdf", dpi=300, bbox_inches="tight")
    plt.show()

# =====================================================
# 6) Plot 2: Grouped boxplot of σ (QTN vs GAF vs MTF)
# =====================================================
def plot_grouped_sigma_boxplot(labels, sigma_qtn, sigma_gaf, sigma_mtf,
                               outfile_prefix="sigma_grouped_qtn_gaf_mtf"):
    n_models = len(labels)
    base = np.arange(n_models)

    offset = 0.25
    pos_qtn = base - offset
    pos_gaf = base
    pos_mtf = base + offset

    plt.figure(figsize=(14, 5))

    def _boxplot(data, positions, facecolor):
        bp = plt.boxplot(
            data, positions=positions, widths=0.22,
            patch_artist=True, showmeans=True, manage_ticks=False
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(facecolor)
        for median in bp["medians"]:
            median.set_linewidth(1.5)
        return bp

    _boxplot(sigma_qtn, pos_qtn, PALETTE_DATASET["QTN"])
    _boxplot(sigma_gaf, pos_gaf, PALETTE_DATASET["GAF"])
    _boxplot(sigma_mtf, pos_mtf, PALETTE_DATASET["MTF"])

    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.ylabel("Small-world index σ")
    plt.title("Small-world index σ by model variant and representation")
    plt.xticks(base, labels, rotation=25, ha="right")

    legend_handles = [
        Patch(facecolor=PALETTE_DATASET["QTN"], label="QTN (QG)"),
        Patch(facecolor=PALETTE_DATASET["GAF"], label="GAF"),
        Patch(facecolor=PALETTE_DATASET["MTF"], label="MTF"),
    ]
    plt.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    plt.savefig(f"{outfile_prefix}.png", dpi=1000, bbox_inches="tight")
    plt.savefig(f"{outfile_prefix}.pdf", dpi=1000, bbox_inches="tight")
    plt.show()

# =====================================================
# 7) Plot 3: Autocorrelation vs σ (colored by representation)
# =====================================================
def autocorr_lag1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation (Pearson)."""
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return np.nan
    x = x - np.mean(x)
    v = np.var(x)
    if v == 0:
        return np.nan
    return float(np.dot(x[:-1], x[1:]) / ((x.size - 1) * v))

def compute_points_autocorr_vs_sigma(Q=12, n=3000, n_realizations=30) -> pd.DataFrame:
    """
    Build one tidy table of points:
      model_label, method, sigma, acf1
    """
    rows = []
    for cfg_idx, (_cfg_id, kind, params, label) in enumerate(CONFIGS):
        for seed in range(n_realizations):
            x = generate_ts(kind=kind, n=n, seed=seed, **params)
            ac1 = autocorr_lag1(x)
            if not np.isfinite(ac1):
                continue

            # QTN
            A = qg_qtn_counts(x, Q=Q, k_values=K_VALUES)
            W_qtn = (A + A.T).astype(float)
            s_qtn = small_world_sigma_from_W(W_qtn, seed=10_000*cfg_idx + seed)
            if np.isfinite(s_qtn):
                rows.append({"model_label": label, "method": "QTN", "sigma": s_qtn, "acf1": ac1})

            # GAF
            W_gaf = gaf_matrix(x, Q=Q)
            s_gaf = small_world_sigma_from_W(W_gaf, seed=20_000*cfg_idx + seed)
            if np.isfinite(s_gaf):
                rows.append({"model_label": label, "method": "GAF", "sigma": s_gaf, "acf1": ac1})

            # MTF
            W_mtf = mtf_transition_matrix(x, Q=Q, bin_mode="quantile", smooth_eps=SMOOTH_EPS)
            s_mtf = small_world_sigma_from_W(W_mtf, seed=30_000*cfg_idx + seed)
            if np.isfinite(s_mtf):
                rows.append({"model_label": label, "method": "MTF", "sigma": s_mtf, "acf1": ac1})

    return pd.DataFrame(rows)

def plot_autocorr_vs_sigma(df_points: pd.DataFrame,
                           outfile_prefix="acf1_vs_sigma"):
    plt.figure(figsize=(7.5, 5))

    for method, sub in df_points.groupby("method"):
        plt.scatter(
            sub["sigma"], sub["acf1"],
            s=18, alpha=0.70,
            color=PALETTE_DATASET.get(method, None),
            label=method
        )

    plt.axvline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("Small-world index σ")
    plt.ylabel("Autocorrelation (lag-1)")
    plt.title("Lag-1 autocorrelation vs small-world index σ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outfile_prefix}.png", dpi=1000, bbox_inches="tight")
    plt.savefig(f"{outfile_prefix}.pdf", dpi=1000, bbox_inches="tight")
    plt.show()

# =====================================================
# 8) Main
# =====================================================
if __name__ == "__main__":
    # Plot 1
    plot_examples_ts_and_matrices(Q=Q, n=N, show_T=400)

    # Plot 2
    labels, sig_qtn, sig_gaf, sig_mtf = compute_sigmas_for_all_methods(Q=Q, n=N, n_realizations=N_REALIZATIONS)
    plot_grouped_sigma_boxplot(labels, sig_qtn, sig_gaf, sig_mtf)

    # Plot 3
    df_points = compute_points_autocorr_vs_sigma(Q=Q, n=N, n_realizations=N_REALIZATIONS)
    plot_autocorr_vs_sigma(df_points)

# In[3]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch

# =====================================================
# 0) Global settings
# =====================================================
Q = 12
N = 3000
N_REALIZATIONS = 30

TARGET_DENSITY   = 0.50
N_RANDOMIZATIONS = 50
REWIRES_PER_EDGE = 5

PALETTE_DATASET = {
    "QTN": "#008080",   # teal
    "GAF": "#A9A9A9",   # darkgray
    "MTF": "#DDA0DD",   # plum
}

# Logistic map sweep (edit as you like)
R_VALUES = [
    3.50, 3.55, 3.57, 3.58, 3.59,
    3.60, 3.62, 3.65, 3.70, 3.75,
    3.80, 3.83, 3.85, 3.90, 3.95, 4.00
]

# For the example figure (Plot 1): choose a few r’s to display
R_EXAMPLES = [3.55, 3.60, 3.70, 3.83, 3.90, 4.00]

# QTN/QG lags
K_VALUES = [1, 2, 3]

# MTF settings
BIN_MODE   = "quantile"
SMOOTH_EPS = 1e-3

# =====================================================
# 1) Utilities
# =====================================================
def downsample_to_length(x: np.ndarray, L: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if L <= 1:
        return x[:1].copy()
    n = x.size
    if n == L:
        return x.copy()
    if n < 2:
        return np.full(L, float(x[0]) if n == 1 else 0.0, dtype=float)
    xp = np.linspace(0.0, 1.0, n)
    return np.interp(np.linspace(0.0, 1.0, L), xp, x).astype(float)

def autocorr_lag1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return np.nan
    x = x - np.mean(x)
    v = np.var(x)
    if v == 0:
        return np.nan
    return float(np.dot(x[:-1], x[1:]) / ((x.size - 1) * v))

# =====================================================
# 2) Logistic map generator
# =====================================================
def logistic_map_series(n: int, r: float, seed: int = 0, burn: int = 1000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0)

    for _ in range(burn):
        x = r * x * (1.0 - x)

    xs = np.empty(n, dtype=float)
    for i in range(n):
        x = r * x * (1.0 - x)
        xs[i] = x

    # standardize
    xs = xs - xs.mean()
    xs = xs / (xs.std() + 1e-12)
    return xs

# =====================================================
# 3) QTN(QG), GAF, MTF (QxQ)
# =====================================================
def qg_qtn_counts(signal: np.ndarray, Q: int, k_values=K_VALUES) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    n = x.size
    A = np.zeros((Q, Q), dtype=np.int64)
    if n <= 1:
        return A

    ranks = np.argsort(np.argsort(x))
    q_edges = np.linspace(0, n, Q + 1, dtype=int)
    loc = np.clip(np.searchsorted(q_edges, ranks, side="right") - 1, 0, Q - 1)

    for k in k_values:
        k = int(k)
        if k <= 0 or k >= n:
            continue
        for i in range(n - k):
            A[loc[i], loc[i + k]] += 1
    return A

def gaf_matrix(signal: np.ndarray, Q: int) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    xQ = downsample_to_length(x, Q)

    min_val, max_val = float(np.min(xQ)), float(np.max(xQ))
    rng = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    scaled = 2 * (xQ - min_val) / rng - 1
    scaled = np.clip(scaled, -1, 1)

    phi = np.arccos(scaled)
    return np.cos(phi[:, None] + phi[None, :])

def _states_from_01(z01: np.ndarray, Q: int, bin_mode: str) -> np.ndarray:
    if bin_mode == "quantile":
        edges = np.quantile(z01, np.linspace(0.0, 1.0, Q + 1))
        edges = np.unique(edges)
        if edges.size < Q + 1:
            edges = np.linspace(0.0, 1.0, Q + 1)
    else:
        edges = np.linspace(0.0, 1.0, Q + 1)
    s = np.digitize(z01, edges[1:-1], right=True)
    return np.clip(s, 0, Q - 1)

def mtf_transition_matrix(signal: np.ndarray, Q: int,
                          bin_mode: str = BIN_MODE,
                          smooth_eps: float = SMOOTH_EPS) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    T = x.size
    if T < 2:
        return np.ones((Q, Q), dtype=float) / Q

    xmin, xmax = float(np.min(x)), float(np.max(x))
    rng = (xmax - xmin) if (xmax - xmin) != 0 else 1.0
    z01 = (x - xmin) / rng

    s = _states_from_01(z01, Q, bin_mode)

    C = np.zeros((Q, Q), dtype=float)
    for t in range(T - 1):
        C[s[t], s[t + 1]] += 1.0

    if smooth_eps and smooth_eps > 0.0:
        C += smooth_eps

    row_sums = C.sum(axis=1, keepdims=True)
    P = np.empty_like(C)
    nz = row_sums.squeeze() > 0
    P[nz]  = C[nz] / row_sums[nz]
    P[~nz] = 1.0 / Q
    return P

# =====================================================
# 4) Small-world sigma on thresholded binary graph
# =====================================================
def proportional_binary_from_weights(W: np.ndarray, target_density: float) -> np.ndarray:
    n = W.shape[0]
    A = np.abs(W).astype(float).copy()
    np.fill_diagonal(A, 0.0)

    upper = np.triu(A, 1)
    vals = upper[upper > 0]
    if vals.size == 0:
        return np.zeros((n, n), dtype=int)

    m_target = int(round(target_density * n * (n - 1) / 2))
    m_target = max(1, min(m_target, vals.size))
    thresh = np.partition(vals, -m_target)[-m_target]

    B = (A >= thresh).astype(int)
    B = np.triu(B, 1)
    B = B + B.T
    np.fill_diagonal(B, 0)
    return B

def char_path_length_gcc(B: np.ndarray) -> float:
    G = nx.from_numpy_array(B)
    if G.number_of_edges() == 0:
        return np.nan
    comps = list(nx.connected_components(G))
    if not comps:
        return np.nan
    largest = G.subgraph(max(comps, key=len)).copy()
    if largest.number_of_nodes() < 2 or largest.number_of_edges() == 0:
        return np.nan
    return nx.average_shortest_path_length(largest)

def mean_clustering(B: np.ndarray) -> float:
    G = nx.from_numpy_array(B)
    if G.number_of_edges() == 0:
        return np.nan
    return float(np.mean(list(nx.clustering(G).values())))

def null_model_stats(B: np.ndarray, n_rand: int, rewires_per_edge: int, seed: int):
    m = int(B.sum() // 2)
    swaps = max(1, rewires_per_edge * m)
    rng = np.random.default_rng(seed)

    C_list, L_list = [], []
    for _ in range(n_rand):
        G = nx.from_numpy_array(B)
        if G.number_of_edges() == 0:
            continue
        try:
            nx.double_edge_swap(
                G, nswap=swaps, max_tries=10 * swaps,
                seed=int(rng.integers(0, 1_000_000))
            )
        except Exception:
            pass
        Br = nx.to_numpy_array(G, dtype=int)
        C_list.append(mean_clustering(Br))
        L_list.append(char_path_length_gcc(Br))

    C_rand = np.nanmean(C_list) if C_list else np.nan
    L_rand = np.nanmean(L_list) if L_list else np.nan
    return C_rand, L_rand

def small_world_sigma_from_W(W: np.ndarray, seed: int = 0) -> float:
    B = proportional_binary_from_weights(W, TARGET_DENSITY)
    C_obs = mean_clustering(B)
    L_obs = char_path_length_gcc(B)
    C_rand, L_rand = null_model_stats(B, N_RANDOMIZATIONS, REWIRES_PER_EDGE, seed)

    gamma  = C_obs / C_rand if (np.isfinite(C_obs) and np.isfinite(C_rand) and C_rand > 0) else np.nan
    lambd  = L_obs / L_rand if (np.isfinite(L_obs) and np.isfinite(L_rand) and L_rand > 0) else np.nan
    sigma  = gamma / lambd if (np.isfinite(gamma) and np.isfinite(lambd) and lambd != 0) else np.nan
    return sigma

# =====================================================
# 5) Compute results for logistic sweep
# =====================================================
def compute_logistic_sweep(Q: int, n: int, r_values, n_realizations: int) -> pd.DataFrame:
    rows = []
    for r_idx, r in enumerate(r_values):
        for seed in range(n_realizations):
            x = logistic_map_series(n=n, r=float(r), seed=seed, burn=1000)
            ac1 = autocorr_lag1(x)

            # QTN
            A = qg_qtn_counts(x, Q=Q, k_values=K_VALUES)
            W_qtn = (A + A.T).astype(float)
            s_qtn = small_world_sigma_from_W(W_qtn, seed=10_000*r_idx + seed)
            rows.append({"r": float(r), "method": "QTN", "sigma": s_qtn, "acf1": ac1})

            # GAF
            W_gaf = gaf_matrix(x, Q=Q)
            s_gaf = small_world_sigma_from_W(W_gaf, seed=20_000*r_idx + seed)
            rows.append({"r": float(r), "method": "GAF", "sigma": s_gaf, "acf1": ac1})

            # MTF
            W_mtf = mtf_transition_matrix(x, Q=Q, bin_mode="quantile", smooth_eps=SMOOTH_EPS)
            s_mtf = small_world_sigma_from_W(W_mtf, seed=30_000*r_idx + seed)
            rows.append({"r": float(r), "method": "MTF", "sigma": s_mtf, "acf1": ac1})

    df = pd.DataFrame(rows)
    df = df[np.isfinite(df["sigma"]) & np.isfinite(df["acf1"])].copy()
    return df

# =====================================================
# 6) Plot 1: Examples (TS + QTN/GAF/MTF matrices) for selected r’s
# =====================================================
def plot_examples_logistic(Q: int, n: int, r_examples, outfile_prefix="logistic_examples_qtn_gaf_mtf", show_T=400):
    rows = []
    for idx, r in enumerate(r_examples):
        x = logistic_map_series(n=n, r=float(r), seed=idx, burn=1000)

        A = qg_qtn_counts(x, Q=Q, k_values=K_VALUES)
        W_qtn = (A + A.T).astype(float)
        W_gaf = gaf_matrix(x, Q=Q)
        W_mtf = mtf_transition_matrix(x, Q=Q, bin_mode="quantile", smooth_eps=SMOOTH_EPS)

        rows.append((x, W_qtn, W_gaf, W_mtf, r))

    qtn_max = max(np.max(Wqtn) for _x, Wqtn, _g, _m, _r in rows)
    if qtn_max <= 0:
        qtn_max = 1.0
    gaf_vmax = max(np.max(Wgaf) for _x, _q, Wgaf, _m, _r in rows)
    mtf_vmax = max(np.max(Wmtf) for _x, _q, _g, Wmtf, _r in rows)

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 4, figsize=(14, 2.4 * nrows), tight_layout=True)

    for i, (x, Wqtn, Wgaf, Wmtf, r) in enumerate(rows):
        ax_ts = axes[i, 0]
        t = np.arange(show_T)
        ax_ts.plot(t, x[:show_T])
        ax_ts.set_title(f"Logistic map r={r:.2f}")
        ax_ts.set_xlabel("Time (samples)")
        ax_ts.set_ylabel("Value")

        ax_qtn = axes[i, 1]
        im1 = ax_qtn.imshow(Wqtn / qtn_max, origin="lower", aspect="equal", vmin=0.0, vmax=1.0)
        ax_qtn.set_title(f"QTN/QG (Q={Q})")
        ax_qtn.set_xlabel("Bin j")
        ax_qtn.set_ylabel("Bin i")
        fig.colorbar(im1, ax=ax_qtn, fraction=0.046, pad=0.04)

        ax_gaf = axes[i, 2]
        im2 = ax_gaf.imshow(Wgaf, origin="lower", aspect="equal", vmin=-1.0, vmax=max(1.0, gaf_vmax))
        ax_gaf.set_title(f"GAF (Q={Q})")
        ax_gaf.set_xlabel("i")
        ax_gaf.set_ylabel("j")
        fig.colorbar(im2, ax=ax_gaf, fraction=0.046, pad=0.04)

        ax_mtf = axes[i, 3]
        im3 = ax_mtf.imshow(Wmtf, origin="lower", aspect="equal", vmin=0.0, vmax=max(1e-12, mtf_vmax))
        ax_mtf.set_title(f"MTF (Q={Q})")
        ax_mtf.set_xlabel("Next state j")
        ax_mtf.set_ylabel("Current state i")
        fig.colorbar(im3, ax=ax_mtf, fraction=0.046, pad=0.04)

    fig.suptitle("Logistic map examples: time series + QTN(QG), GAF, and MTF matrices", y=1.02)
    plt.savefig(f"{outfile_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{outfile_prefix}.pdf", dpi=300, bbox_inches="tight")
    plt.show()

# =====================================================
# 7) Plot 2: Grouped σ boxplot across r values (colors = methods)
# =====================================================
def plot_sigma_boxplot_by_r(df: pd.DataFrame, r_values, outfile_prefix="logistic_sigma_by_r"):
    base = np.arange(len(r_values))
    offset = 0.25

    pos_qtn = base - offset
    pos_gaf = base
    pos_mtf = base + offset

    data_qtn = [df[(df["r"] == r) & (df["method"] == "QTN")]["sigma"].tolist() for r in r_values]
    data_gaf = [df[(df["r"] == r) & (df["method"] == "GAF")]["sigma"].tolist() for r in r_values]
    data_mtf = [df[(df["r"] == r) & (df["method"] == "MTF")]["sigma"].tolist() for r in r_values]

    plt.figure(figsize=(14, 5))

    def _boxplot(data, positions, facecolor):
        bp = plt.boxplot(
            data, positions=positions, widths=0.22,
            patch_artist=True, showmeans=True, manage_ticks=False
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(facecolor)
        for median in bp["medians"]:
            median.set_linewidth(1.5)
        return bp

    _boxplot(data_qtn, pos_qtn, PALETTE_DATASET["QTN"])
    _boxplot(data_gaf, pos_gaf, PALETTE_DATASET["GAF"])
    _boxplot(data_mtf, pos_mtf, PALETTE_DATASET["MTF"])

    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.ylabel("Small-world index σ")
    plt.title("Logistic map: σ across r values (QTN vs GAF vs MTF)")

    xt = [f"{r:.2f}" for r in r_values]
    plt.xticks(base, xt, rotation=45, ha="right")

    legend_handles = [
        Patch(facecolor=PALETTE_DATASET["QTN"], label="QTN (QG)"),
        Patch(facecolor=PALETTE_DATASET["GAF"], label="GAF"),
        Patch(facecolor=PALETTE_DATASET["MTF"], label="MTF"),
    ]
    plt.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    plt.savefig(f"{outfile_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{outfile_prefix}.pdf", dpi=300, bbox_inches="tight")
    plt.show()

# =====================================================
# 8) Plot 3: ACF(1) vs σ scatter (colors = methods)
# =====================================================
def plot_acf1_vs_sigma(df: pd.DataFrame, outfile_prefix="logistic_acf1_vs_sigma"):
    plt.figure(figsize=(7.5, 5))

    for method in ["QTN", "GAF", "MTF"]:
        sub = df[df["method"] == method]
        plt.scatter(
            sub["sigma"], sub["acf1"],
            s=18, alpha=0.70,
            color=PALETTE_DATASET[method],
            label=method
        )

    plt.axvline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("Small-world index σ")
    plt.ylabel("Autocorrelation (lag-1)")
    plt.title("Logistic map: lag-1 autocorrelation vs σ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outfile_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{outfile_prefix}.pdf", dpi=300, bbox_inches="tight")
    plt.show()

# =====================================================
# 9) Main
# =====================================================
if __name__ == "__main__":
    # Compute sweep
    df = compute_logistic_sweep(Q=Q, n=N, r_values=R_VALUES, n_realizations=N_REALIZATIONS)

    # Plot 1: examples
    plot_examples_logistic(Q=Q, n=N, r_examples=R_EXAMPLES, outfile_prefix="logistic_examples_qtn_gaf_mtf", show_T=400)

    # Plot 2: sigma boxplot by r
    plot_sigma_boxplot_by_r(df, r_values=R_VALUES, outfile_prefix="logistic_sigma_by_r")

    # Plot 3: ACF(1) vs sigma
    plot_acf1_vs_sigma(df, outfile_prefix="logistic_acf1_vs_sigma")

    # Optional: save the raw results table
    df.to_csv("logistic_sweep_sigma_acf1.csv", index=False)
    print("Saved: logistic_sweep_sigma_acf1.csv", df.shape)

# In[4]:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Keep your palette
PALETTE_DATASET = {
    "QTN": "#008080",   # teal
    "GAF": "#A9A9A9",   # darkgray
    "MTF": "#DDA0DD",   # plum
}

# -----------------------------
# Robustness settings
# -----------------------------
Q_LIST = list(range(6, 41, 2))  # 6,8,...,40  (change if you want)
ROBUST_MODELS = CONFIGS         # reuse your models list
ROBUST_N_REALIZATIONS = N_REALIZATIONS  # reuse

def _sigma_for_method_and_Q(x, method, Q, cfg_idx, seed):
    """
    Compute sigma for one signal x, for a given method and Q.
    """
    if method == "QTN":
        A = qg_qtn_counts(x, Q=Q, k_values=K_VALUES)
        W = (A + A.T).astype(float)
        return small_world_sigma_from_W(W, seed=10_000*cfg_idx + seed + 1_000_000*Q)

    if method == "GAF":
        W = gaf_matrix(x, Q=Q)
        return small_world_sigma_from_W(W, seed=20_000*cfg_idx + seed + 1_000_000*Q)

    if method == "MTF":
        W = mtf_transition_matrix(x, Q=Q, bin_mode="quantile", smooth_eps=SMOOTH_EPS)
        return small_world_sigma_from_W(W, seed=30_000*cfg_idx + seed + 1_000_000*Q)

    raise ValueError("method must be one of: QTN, GAF, MTF")

def compute_sigma_vs_Q(n=N, q_list=Q_LIST, n_realizations=ROBUST_N_REALIZATIONS):
    """
    Returns a nested dict:
      results[model_label][method] = array shape (n_realizations, len(q_list))
    """
    methods = ["QTN", "GAF", "MTF"]
    results = {}

    for cfg_idx, (cfg_id, kind, params, label) in enumerate(ROBUST_MODELS):
        model_out = {m: np.full((n_realizations, len(q_list)), np.nan, float) for m in methods}

        for r_i in range(n_realizations):
            x = generate_ts(kind=kind, n=n, seed=r_i, **params)

            for q_j, Qv in enumerate(q_list):
                for m in methods:
                    s = _sigma_for_method_and_Q(x, m, Qv, cfg_idx, r_i)
                    if np.isfinite(s):
                        model_out[m][r_i, q_j] = s

        results[label] = model_out

    return results

def plot_sigma_vs_Q_curves(results, q_list=Q_LIST, outfile="robust_sigma_vs_Q.png"):
    """
    One figure: for each model (row), plot median sigma(Q) with IQR band for each method.
    """
    methods = ["QTN", "GAF", "MTF"]
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 2.4*n_models), sharex=True, tight_layout=True)
    if n_models == 1:
        axes = [axes]

    for ax, (model_label, model_out) in zip(axes, results.items()):
        for m in methods:
            A = model_out[m]  # (R, Q)
            # drop realizations with too many NaNs
            med = np.nanmedian(A, axis=0)
            q25 = np.nanpercentile(A, 25, axis=0)
            q75 = np.nanpercentile(A, 75, axis=0)

            ax.plot(q_list, med, label=m, color=PALETTE_DATASET[m], linewidth=2)
            ax.fill_between(q_list, q25, q75, color=PALETTE_DATASET[m], alpha=0.15)

        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_ylabel("σ")
        ax.set_title(model_label)

    axes[-1].set_xlabel("Quantile number Q")
    axes[0].legend(loc="upper right")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.savefig(outfile.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.show()

def plot_Q_robustness_CV(results, q_list=Q_LIST, outfile="robust_cv_boxplot.png"):
    """
    Robustness score per realization: CV over Q for each method.
    Then boxplot CVs (lower is better).
    """
    methods = ["QTN", "GAF", "MTF"]
    model_labels = list(results.keys())

    # collect CVs per model & method
    cv_data = {m: [] for m in methods}
    group_ticks = []
    for model_label in model_labels:
        group_ticks.append(model_label)
        for m in methods:
            A = results[model_label][m]  # (R, Q)
            cvs = []
            for r in range(A.shape[0]):
                y = A[r, :]
                y = y[np.isfinite(y)]
                if y.size < max(5, int(0.5*len(q_list))):
                    continue
                mu = np.mean(y)
                sd = np.std(y, ddof=1) if y.size > 1 else np.nan
                if np.isfinite(mu) and np.isfinite(sd) and abs(mu) > 1e-12:
                    cvs.append(sd / abs(mu))
            cv_data[m].append(cvs)

    # grouped boxplot positions
    n_models = len(model_labels)
    base = np.arange(n_models)
    offset = 0.25
    pos = {"QTN": base - offset, "GAF": base, "MTF": base + offset}

    plt.figure(figsize=(14, 5))

    def _boxplot(data, positions, facecolor):
        bp = plt.boxplot(
            data,
            positions=positions,
            widths=0.22,
            patch_artist=True,
            showmeans=True,
            manage_ticks=False
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(facecolor)
        return bp

    _boxplot(cv_data["QTN"], pos["QTN"], PALETTE_DATASET["QTN"])
    _boxplot(cv_data["GAF"], pos["GAF"], PALETTE_DATASET["GAF"])
    _boxplot(cv_data["MTF"], pos["MTF"], PALETTE_DATASET["MTF"])

    plt.xticks(base, model_labels, rotation=25, ha="right")
    plt.ylabel("CV across Q  (std_Q / |mean_Q|)")
    plt.title("Robustness to quantile number Q (lower CV = more robust)")

    legend_handles = [
        Patch(facecolor=PALETTE_DATASET["QTN"], label="QTN"),
        Patch(facecolor=PALETTE_DATASET["GAF"], label="GAF"),
        Patch(facecolor=PALETTE_DATASET["MTF"], label="MTF"),
    ]
    plt.legend(handles=legend_handles, loc="upper right")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.savefig(outfile.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.show()

def plot_sigma_Q_trend(results, q_list=Q_LIST, outfile="robust_trend_spearman.png"):
    """
    Optional: show Spearman rho between sigma(Q) and Q across Q, per realization, boxplot per model & method.
    (If sigma strongly depends on Q, that's less robust.)
    """
    methods = ["QTN", "GAF", "MTF"]
    model_labels = list(results.keys())
    Qv = np.asarray(q_list, float)

    def spearman_rho(a, b):
        # minimal spearman: rank then Pearson
        ar = np.argsort(np.argsort(a))
        br = np.argsort(np.argsort(b))
        ar = ar.astype(float)
        br = br.astype(float)
        ar -= ar.mean(); br -= br.mean()
        denom = (np.sqrt((ar**2).sum()) * np.sqrt((br**2).sum()))
        return float((ar*br).sum() / denom) if denom > 0 else np.nan

    rho_data = {m: [] for m in methods}
    for model_label in model_labels:
        for m in methods:
            A = results[model_label][m]  # (R, Q)
            rhos = []
            for r in range(A.shape[0]):
                y = A[r, :]
                mask = np.isfinite(y)
                if mask.sum() < max(5, int(0.5*len(q_list))):
                    continue
                rhos.append(spearman_rho(Qv[mask], y[mask]))
            rho_data[m].append(rhos)

    n_models = len(model_labels)
    base = np.arange(n_models)
    offset = 0.25
    pos = {"QTN": base - offset, "GAF": base, "MTF": base + offset}

    plt.figure(figsize=(14, 5))

    def _boxplot(data, positions, facecolor):
        bp = plt.boxplot(
            data,
            positions=positions,
            widths=0.22,
            patch_artist=True,
            showmeans=True,
            manage_ticks=False
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(facecolor)
        return bp

    _boxplot(rho_data["QTN"], pos["QTN"], PALETTE_DATASET["QTN"])
    _boxplot(rho_data["GAF"], pos["GAF"], PALETTE_DATASET["GAF"])
    _boxplot(rho_data["MTF"], pos["MTF"], PALETTE_DATASET["MTF"])

    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xticks(base, model_labels, rotation=25, ha="right")
    plt.ylabel("Spearman ρ(σ(Q), Q)")
    plt.title("Trend of σ with Q (ρ near 0 = robust to Q)")

    legend_handles = [
        Patch(facecolor=PALETTE_DATASET["QTN"], label="QTN"),
        Patch(facecolor=PALETTE_DATASET["GAF"], label="GAF"),
        Patch(facecolor=PALETTE_DATASET["MTF"], label="MTF"),
    ]
    plt.legend(handles=legend_handles, loc="upper right")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.savefig(outfile.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.show()
    
    


# In[5]:

if __name__ == "__main__":
    # your existing plots
    plot_examples_ts_and_matrices(Q=Q, n=N, show_T=400)
    labels, sig_qg, sig_gaf, sig_mtf = compute_sigmas_for_all_methods(Q=Q, n=N, n_realizations=N_REALIZATIONS)
    plot_grouped_sigma_boxplot(labels, sig_qg, sig_gaf, sig_mtf)

    # NEW: robustness to Q
    results_Q = compute_sigma_vs_Q(n=N, q_list=Q_LIST, n_realizations=N_REALIZATIONS)

    plot_sigma_vs_Q_curves(results_Q, q_list=Q_LIST, outfile="robust_sigma_vs_Q.png")
    plot_Q_robustness_CV(results_Q, q_list=Q_LIST, outfile="robust_cv_boxplot.png")
    plot_sigma_Q_trend(results_Q, q_list=Q_LIST, outfile="robust_trend_spearman.png")

# In[8]:

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ============================================================
# Palette + style (match your attached plots)
# ============================================================
PALETTE = {
    "QTN": "#0B7A77",   # teal
    "GAF": "#7A7A7A",   # gray
    "MTF": "#C77BCB",   # magenta/purple
}
BAND_ALPHA = 0.18
LINE_ALPHA = 1.0

def _mean_ci(x, ci=0.95):
    """
    Mean and CI via normal approx on bootstrap distribution (no assumption on x).
    Returns mean, lo, hi.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    mu = float(np.mean(x))
    if x.size == 1:
        return mu, mu, mu
    # basic bootstrap for CI of mean
    B = 500
    rng = np.random.default_rng(0)
    boots = np.empty(B, dtype=float)
    for b in range(B):
        boots[b] = np.mean(rng.choice(x, size=x.size, replace=True))
    lo = float(np.quantile(boots, (1-ci)/2))
    hi = float(np.quantile(boots, 1-(1-ci)/2))
    return mu, lo, hi

def _bootstrap_slope(Qs, y_means, B=800, seed=0):
    """
    Bootstrap slope of y ~ a + b*Q using resampling over points (Qs, y_means).
    Returns slope_mean, slope_ci_lo, slope_ci_hi.
    """
    Qs = np.asarray(Qs, dtype=float)
    y  = np.asarray(y_means, dtype=float)
    ok = np.isfinite(Qs) & np.isfinite(y)
    Qs, y = Qs[ok], y[ok]
    if Qs.size < 2:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    slopes = np.empty(B, dtype=float)
    n = Qs.size
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        qb, yb = Qs[idx], y[idx]
        # simple least squares slope
        qb_mean = qb.mean()
        denom = np.sum((qb - qb_mean)**2)
        slopes[b] = np.sum((qb - qb_mean)*(yb - yb.mean())) / denom if denom > 0 else np.nan

    slopes = slopes[np.isfinite(slopes)]
    if slopes.size == 0:
        return np.nan, np.nan, np.nan
    mu = float(np.mean(slopes))
    lo = float(np.quantile(slopes, 0.025))
    hi = float(np.quantile(slopes, 0.975))
    return mu, lo, hi


# ============================================================
# 1) Run σ across Q for each config/method (collect raw values)
# ============================================================
def compute_sigma_vs_Q(
    Q_values,
    CONFIGS,
    n=3000,
    n_realizations=30,
    seed_offset=0,
):
    """
    Returns:
      results[cfg_label][method] = dict with:
         Qs, mean, lo, hi, raw_by_Q (list of arrays), and also mean_per_Q for trend tests
    """
    results = {}

    for cfg_idx, (cfg_id, kind, params, cfg_label) in enumerate(CONFIGS):
        results[cfg_label] = {}

        raw = {m: [] for m in ["QTN", "GAF", "MTF"]}

        for qi, Q in enumerate(Q_values):
            sig_qtn, sig_gaf, sig_mtf = [], [], []

            for s in range(n_realizations):
                seed = seed_offset + 1_000_000*cfg_idx + 10_000*qi + s
                x = generate_ts(kind=kind, n=n, seed=seed, **params)

                # QTN/QG
                A = qg_qtn_counts(x, Q=int(Q), k_values=K_VALUES)
                W_qtn = (A + A.T).astype(float)
                v = small_world_sigma_from_W(W_qtn, seed=seed + 1)
                if np.isfinite(v):
                    sig_qtn.append(v)

                # GAF
                W_gaf = gaf_matrix(x, Q=int(Q))
                v = small_world_sigma_from_W(W_gaf, seed=seed + 2)
                if np.isfinite(v):
                    sig_gaf.append(v)

                # MTF
                W_mtf = mtf_transition_matrix(x, Q=int(Q), bin_mode="quantile", smooth_eps=SMOOTH_EPS)
                v = small_world_sigma_from_W(W_mtf, seed=seed + 3)
                if np.isfinite(v):
                    sig_mtf.append(v)

            raw["QTN"].append(np.array(sig_qtn, dtype=float))
            raw["GAF"].append(np.array(sig_gaf, dtype=float))
            raw["MTF"].append(np.array(sig_mtf, dtype=float))

        # summarize with mean + CI bands per Q (this gives the "shadow")
        for method in ["QTN", "GAF", "MTF"]:
            means, lo, hi = [], [], []
            for arr in raw[method]:
                mu, l, h = _mean_ci(arr, ci=0.95)
                means.append(mu); lo.append(l); hi.append(h)

            results[cfg_label][method] = {
                "Qs": np.array(Q_values, dtype=int),
                "mean": np.array(means, dtype=float),
                "lo": np.array(lo, dtype=float),
                "hi": np.array(hi, dtype=float),
                "raw_by_Q": raw[method],  # list of arrays
            }

    return results


# ============================================================
# 2) Plot σ vs Q (same style as your attached figure)
# ============================================================
def plot_sigma_vs_Q_panels(results, outfile="robust_sigma_vs_Q_with_CI"):
    cfgs = list(results.keys())
    n_panels = len(cfgs)

    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 1.8*n_panels), sharex=True, tight_layout=True)
    if n_panels == 1:
        axes = [axes]

    for ax, cfg_label in zip(axes, cfgs):
        for method in ["QTN", "GAF", "MTF"]:
            d = results[cfg_label][method]
            Qs, mu, lo, hi = d["Qs"], d["mean"], d["lo"], d["hi"]

            ax.plot(Qs, mu, color=PALETTE[method], alpha=LINE_ALPHA, label=method)
            ax.fill_between(Qs, lo, hi, color=PALETTE[method], alpha=BAND_ALPHA, linewidth=0)

        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_ylabel("σ")
        ax.set_title(cfg_label)

    axes[-1].set_xlabel("Q (number of quantile bins / states)")
    axes[0].legend(loc="upper right")
    plt.savefig(outfile + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(outfile + ".pdf", dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# 3) Robustness metric: slope of σ vs Q (with CI)
# ============================================================
def compute_slope_stats(results):
    """
    For each config/method, slope is fitted on mean(σ|Q).
    Returns slope[cfg][method] = (slope_mean, slope_ci_lo, slope_ci_hi)
    """
    slopes = {}
    for cfg_label in results:
        slopes[cfg_label] = {}
        for method in ["QTN", "GAF", "MTF"]:
            Qs = results[cfg_label][method]["Qs"]
            mu = results[cfg_label][method]["mean"]
            s_mu, s_lo, s_hi = _bootstrap_slope(Qs, mu, B=800, seed=1)
            slopes[cfg_label][method] = (s_mu, s_lo, s_hi)
    return slopes

def plot_slope_summary(slopes, outfile="robust_slope_vs_Q"):
    cfgs = list(slopes.keys())
    x = np.arange(len(cfgs))

    fig, ax = plt.subplots(figsize=(10, 4.5), tight_layout=True)

    offsets = {"QTN": -0.25, "GAF": 0.0, "MTF": 0.25}
    for method in ["QTN", "GAF", "MTF"]:
        y  = np.array([slopes[c][method][0] for c in cfgs], dtype=float)
        lo = np.array([slopes[c][method][1] for c in cfgs], dtype=float)
        hi = np.array([slopes[c][method][2] for c in cfgs], dtype=float)

        xpos = x + offsets[method]
        ax.plot(xpos, y, marker="o", linewidth=1.8, color=PALETTE[method], label=method)
        ax.fill_between(xpos, lo, hi, color=PALETTE[method], alpha=BAND_ALPHA, linewidth=0)

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(cfgs, rotation=30, ha="right")
    ax.set_ylabel("Slope of mean σ vs Q")
    ax.set_title("Robustness metric: trend (slope) of σ across Q (95% bootstrap CI)")
    ax.legend(loc="upper right")

    plt.savefig(outfile + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(outfile + ".pdf", dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# 4) Statistical test: Spearman trend test (Q-invariance)
# ============================================================
def compute_spearman_tests(results):
    """
    Spearman correlation between Q and mean σ(Q) per config/method.
    Returns stats[cfg][method] = (rho, pvalue)
    """
    stats = {}
    for cfg_label in results:
        stats[cfg_label] = {}
        for method in ["QTN", "GAF", "MTF"]:
            Qs = results[cfg_label][method]["Qs"].astype(float)
            mu = results[cfg_label][method]["mean"].astype(float)
            ok = np.isfinite(Qs) & np.isfinite(mu)
            if np.sum(ok) < 3:
                stats[cfg_label][method] = (np.nan, np.nan)
            else:
                rho, p = spearmanr(Qs[ok], mu[ok])
                stats[cfg_label][method] = (float(rho), float(p))
    return stats

def plot_spearman_pvalues(stats, outfile="robust_spearman_pvalues"):
    """
    Plot -log10(p) per config with shaded style (bands not meaningful here),
    and include rho via marker direction.
    """
    cfgs = list(stats.keys())
    x = np.arange(len(cfgs))
    fig, ax = plt.subplots(figsize=(10, 4.5), tight_layout=True)

    offsets = {"QTN": -0.25, "GAF": 0.0, "MTF": 0.25}
    for method in ["QTN", "GAF", "MTF"]:
        pvals = np.array([stats[c][method][1] for c in cfgs], dtype=float)
        rhos  = np.array([stats[c][method][0] for c in cfgs], dtype=float)

        y = -np.log10(np.clip(pvals, 1e-300, 1.0))
        xpos = x + offsets[method]

        ax.plot(xpos, y, marker="o", linewidth=1.8, color=PALETTE[method], label=method)

        # encode sign of rho in marker (triangle up/down) for quick reading
        for xi, yi, rho in zip(xpos, y, rhos):
            if np.isfinite(rho):
                mk = "^" if rho > 0 else "v"
                ax.scatter([xi], [yi], marker=mk, s=55, color=PALETTE[method])

    # conventional threshold p=0.05
    ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(cfgs, rotation=30, ha="right")
    ax.set_ylabel(r"$-\log_{10}(p)$  (Spearman trend test)")
    ax.set_title("Q-invariance test: Spearman trend of mean σ across Q (markers show sign of ρ)")
    ax.legend(loc="upper right")

    plt.savefig(outfile + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(outfile + ".pdf", dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# 5) Single summary robustness score per method (two options)
# ============================================================
def compute_robustness_scores(results):
    """
    Produces two scalar robustness scores per config/method:

    A) trend_score = |Spearman rho|  (0 = perfectly Q-invariant trend-wise)
    B) dispersion_score = CV across Q of mean σ(Q):
         std_Q(meanσ) / |mean_Q(meanσ)|
    """
    spearman_stats = compute_spearman_tests(results)
    scores = {}

    for cfg_label in results:
        scores[cfg_label] = {}
        for method in ["QTN", "GAF", "MTF"]:
            rho, p = spearman_stats[cfg_label][method]
            muQ = results[cfg_label][method]["mean"]
            muQ = muQ[np.isfinite(muQ)]
            if muQ.size == 0:
                disp = np.nan
            else:
                disp = float(np.std(muQ, ddof=1) / max(abs(np.mean(muQ)), 1e-8)) if muQ.size > 1 else 0.0

            scores[cfg_label][method] = {
                "trend_score": abs(rho) if np.isfinite(rho) else np.nan,
                "dispersion_score": disp,
                "rho": rho, "p": p
            }
    return scores

def plot_robustness_score_boxplots(scores, which="dispersion_score", outfile="robust_score_boxplot"):
    """
    Boxplots across configs for each method.
    which: 'dispersion_score' or 'trend_score'
    """
    assert which in {"dispersion_score", "trend_score"}

    methods = ["QTN", "GAF", "MTF"]
    data = []
    for m in methods:
        vals = []
        for cfg in scores:
            v = scores[cfg][m][which]
            if np.isfinite(v):
                vals.append(v)
        data.append(np.array(vals, dtype=float))

    fig, ax = plt.subplots(figsize=(7.2, 4.6), tight_layout=True)
    bp = ax.boxplot(data, patch_artist=True, showmeans=True)

    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(PALETTE[m])
        patch.set_alpha(0.35)
    for median in bp["medians"]:
        median.set_linewidth(1.5)

    ax.set_xticklabels(methods)
    ax.set_ylabel(which.replace("_", " "))
    ax.set_title(f"Single robustness score per method across generators: {which.replace('_',' ')}")

    plt.savefig(outfile + f"_{which}.png", dpi=600, bbox_inches="tight")
    plt.savefig(outfile + f"_{which}.pdf", dpi=600, bbox_inches="tight")
    plt.show()


# ============================================================
# 6) Example usage (run this after your base functions exist)
# ============================================================
if __name__ == "__main__":
    # Choose the Q sweep you used in the attached figure
    Q_values = np.arange(6, 33, 2)  # example: 6..32 step 2

    # CONFIGS must be of the form:
    # (cfg_id, kind, params_dict, label)
    # using the same CONFIGS list you already defined earlier.

    results = compute_sigma_vs_Q(
        Q_values=Q_values,
        CONFIGS=CONFIGS,
        n=3000,
        n_realizations=30,
        seed_offset=0
    )

    # (A) σ vs Q panels (same palette + shaded CI bands)
    plot_sigma_vs_Q_panels(results, outfile="robust_sigma_vs_Q")

    # (B) Slope robustness metric
    slopes = compute_slope_stats(results)
    plot_slope_summary(slopes, outfile="robust_slope_vs_Q")

    # (C) Spearman Q-invariance test (-log10 p)
    stats = compute_spearman_tests(results)
    plot_spearman_pvalues(stats, outfile="robust_spearman_trend_test")

    # (D) Single robustness score per method (boxplots)
    scores = compute_robustness_scores(results)
    plot_robustness_score_boxplots(scores, which="dispersion_score", outfile="robust_score_boxplot")
    plot_robustness_score_boxplots(scores, which="trend_score", outfile="robust_score_boxplot")

# In[ ]:
