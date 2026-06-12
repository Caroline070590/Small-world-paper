"""Logistic-map analysis across dynamical regimes (Fig. 4, Fig. S1):
sigma vs control parameter r across periodic / period-doubling / chaotic /
fully developed chaos.

Provenance: extracted verbatim from the notebook ``logistic-maps.ipynb``.
Figure-generating / analysis script; run top-to-bottom after setting paths.
"""

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
    "QTN": "#2A9D8F",   # teal
    "GAF": "#B0B0B0",   # soft gray
    "MTF": "#C994C7",   # soft plum
}

# Light background bands for dynamical regimes
REGIME_BANDS = [
    {"name": "Periodic", "xmin": 2.80, "xmax": 3.00, "color": "#E8F5E9"},
    {"name": "Period-doubling", "xmin": 3.00, "xmax": 3.57, "color": "#FFF8E1"},
    {"name": "Chaotic", "xmin": 3.57, "xmax": 3.80, "color": "#FBE9E7"},
    {"name": "Fully developed chaos", "xmin": 3.80, "xmax": 4.00, "color": "#F3E5F5"},
]

# Extended sweep across all regimes
R_VALUES = np.round(np.concatenate([
    np.arange(2.80, 3.01, 0.05),   # periodic
    np.arange(3.05, 3.58, 0.05),   # period-doubling
    np.arange(3.58, 3.81, 0.03),   # chaotic
    np.arange(3.82, 4.01, 0.03),   # fully developed chaos
]), 2)

R_VALUES = sorted(np.unique(R_VALUES.tolist()))

# Examples for matrix visualization
R_EXAMPLES = [2.90, 3.20, 3.50, 3.60, 3.83, 4.00]

# QTN/QG lags
K_VALUES = [1, 2, 3]

# MTF settings
BIN_MODE   = "quantile"
SMOOTH_EPS = 1e-3

# Typography for publication-quality plots
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

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

def classify_regime(r: float) -> str:
    if 2.80 <= r <= 3.00:
        return "Periodic"
    elif 3.00 < r <= 3.57:
        return "Period-doubling"
    elif 3.57 < r < 3.80:
        return "Chaotic"
    elif 3.80 <= r <= 4.00:
        return "Fully developed chaos"
    return "Outside range"

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
# 4) Small-world sigma
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

    gamma = C_obs / C_rand if (np.isfinite(C_obs) and np.isfinite(C_rand) and C_rand > 0) else np.nan
    lambd = L_obs / L_rand if (np.isfinite(L_obs) and np.isfinite(L_rand) and L_rand > 0) else np.nan
    sigma = gamma / lambd if (np.isfinite(gamma) and np.isfinite(lambd) and lambd != 0) else np.nan
    return sigma

# =====================================================
# 5) Compute results
# =====================================================
def compute_logistic_sweep(Q: int, n: int, r_values, n_realizations: int) -> pd.DataFrame:
    rows = []
    for r_idx, r in enumerate(r_values):
        for seed in range(n_realizations):
            x = logistic_map_series(n=n, r=float(r), seed=seed, burn=1000)
            ac1 = autocorr_lag1(x)
            regime = classify_regime(float(r))

            A = qg_qtn_counts(x, Q=Q, k_values=K_VALUES)
            W_qtn = (A + A.T).astype(float)
            s_qtn = small_world_sigma_from_W(W_qtn, seed=10_000 * r_idx + seed)
            rows.append({"r": float(r), "regime": regime, "method": "QTN", "sigma": s_qtn, "acf1": ac1})

            W_gaf = gaf_matrix(x, Q=Q)
            s_gaf = small_world_sigma_from_W(W_gaf, seed=20_000 * r_idx + seed)
            rows.append({"r": float(r), "regime": regime, "method": "GAF", "sigma": s_gaf, "acf1": ac1})

            W_mtf = mtf_transition_matrix(x, Q=Q, bin_mode="quantile", smooth_eps=SMOOTH_EPS)
            s_mtf = small_world_sigma_from_W(W_mtf, seed=30_000 * r_idx + seed)
            rows.append({"r": float(r), "regime": regime, "method": "MTF", "sigma": s_mtf, "acf1": ac1})

    df = pd.DataFrame(rows)
    df = df[np.isfinite(df["sigma"]) & np.isfinite(df["acf1"])].copy()
    return df

# =====================================================
# 6) Plot: publication-style sigma boxplot with regime bands
# =====================================================
def plot_sigma_boxplot_by_r(df: pd.DataFrame, r_values, outfile_prefix="logistic_sigma_by_r_regimes"):
    r_values = sorted(r_values)
    pos_map = {r: i for i, r in enumerate(r_values)}
    base = np.arange(len(r_values))
    offset = 0.25

    pos_qtn = base - offset
    pos_gaf = base
    pos_mtf = base + offset

    data_qtn = [df[(df["r"] == r) & (df["method"] == "QTN")]["sigma"].tolist() for r in r_values]
    data_gaf = [df[(df["r"] == r) & (df["method"] == "GAF")]["sigma"].tolist() for r in r_values]
    data_mtf = [df[(df["r"] == r) & (df["method"] == "MTF")]["sigma"].tolist() for r in r_values]

    fig, ax = plt.subplots(figsize=(16, 6))

    # Background dynamical-regime bands
    for band in REGIME_BANDS:
        in_band = [r for r in r_values if band["xmin"] <= r <= band["xmax"]]
        if not in_band:
            continue
        xmin = pos_map[min(in_band)] - 0.6
        xmax = pos_map[max(in_band)] + 0.6
        ax.axvspan(xmin, xmax, color=band["color"], alpha=0.65, zorder=0)
        ax.text((xmin + xmax) / 2, 0.98, band["name"],
                ha="center", va="top", fontsize=11,
                transform=ax.get_xaxis_transform())

    def _boxplot(data, positions, facecolor):
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.22,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker="^", markerfacecolor="#E76F51", markeredgecolor="#E76F51", markersize=6),
            medianprops=dict(color="#F4A261", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker='o', markersize=4, markerfacecolor='white', markeredgecolor='black', alpha=0.8),
            manage_ticks=False
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(facecolor)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.2)
        return bp

    _boxplot(data_qtn, pos_qtn, PALETTE_DATASET["QTN"])
    _boxplot(data_gaf, pos_gaf, PALETTE_DATASET["GAF"])
    _boxplot(data_mtf, pos_mtf, PALETTE_DATASET["MTF"])

    ax.axhline(1.0, linestyle="--", linewidth=1.4, color="#3A3A3A", alpha=0.9)
    ax.set_ylabel("Small-world index $\\sigma$")
    ax.set_xlabel("Logistic map parameter $r$")
    ax.set_title("Small-world topology across logistic-map regimes")

    ax.set_xticks(base)
    ax.set_xticklabels([f"{r:.2f}" for r in r_values], rotation=45, ha="right")

    legend_handles = [
        Patch(facecolor=PALETTE_DATASET["QTN"], edgecolor="black", label="QTN (QG)"),
        Patch(facecolor=PALETTE_DATASET["GAF"], edgecolor="black", label="GAF"),
        Patch(facecolor=PALETTE_DATASET["MTF"], edgecolor="black", label="MTF"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    ax.grid(axis="y", linestyle=":", alpha=0.35)
    plt.tight_layout()

    fig.savefig(f"{outfile_prefix}.png", dpi=1000, bbox_inches="tight")
    fig.savefig(f"{outfile_prefix}.pdf", dpi=1000, bbox_inches="tight")
    plt.show()

# =====================================================
# 7) Export summaries
# =====================================================
def export_summary_tables(df: pd.DataFrame, outfile_prefix="logistic_sigma_summary"):
    summary = (
        df.groupby(["r", "regime", "method"], as_index=False)
          .agg(
              sigma_mean=("sigma", "mean"),
              sigma_median=("sigma", "median"),
              sigma_std=("sigma", "std"),
              sigma_sem=("sigma", lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
              sigma_min=("sigma", "min"),
              sigma_max=("sigma", "max"),
              acf1_mean=("acf1", "mean"),
          )
    )

    regime_summary = (
        df.groupby(["regime", "method"], as_index=False)
          .agg(
              sigma_mean=("sigma", "mean"),
              sigma_median=("sigma", "median"),
              sigma_std=("sigma", "std"),
              sigma_sem=("sigma", lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
              n=("sigma", "size"),
          )
    )

    summary.to_csv(f"{outfile_prefix}_by_r.csv", index=False)
    regime_summary.to_csv(f"{outfile_prefix}_by_regime.csv", index=False)
    return summary, regime_summary

# =====================================================
# 8) Main
# =====================================================
if __name__ == "__main__":
    df = compute_logistic_sweep(Q=Q, n=N, r_values=R_VALUES, n_realizations=N_REALIZATIONS)
    df.to_csv("logistic_sweep_sigma_acf1_full.csv", index=False)

    summary_r, summary_regime = export_summary_tables(df, outfile_prefix="logistic_sigma_summary")

    plot_sigma_boxplot_by_r(df, r_values=R_VALUES, outfile_prefix="logistic_sigma_by_r_regimes")

    print("Saved:")
    print("- logistic_sweep_sigma_acf1_full.csv")
    print("- logistic_sigma_summary_by_r.csv")
    print("- logistic_sigma_summary_by_regime.csv")
    print("- logistic_sigma_by_r_regimes.png")
    print("- logistic_sigma_by_r_regimes.pdf")

# %% ---- next notebook cell ----

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

TARGET_DENSITY = 0.50
N_RANDOMIZATIONS = 50
REWIRES_PER_EDGE = 5

PALETTE_DATASET = {
    "QTN": "#2A9D8F",   # teal
    "GAF": "#B0B0B0",   # soft gray
    "MTF": "#C994C7",   # soft plum
}

# Light background bands for dynamical regimes
REGIME_BANDS = [
    {"name": "Periodic", "xmin": 2.80, "xmax": 3.00, "color": "#E8F5E9"},
    {"name": "Period-doubling", "xmin": 3.00, "xmax": 3.57, "color": "#FFF8E1"},
    {"name": "Chaotic", "xmin": 3.57, "xmax": 3.80, "color": "#FBE9E7"},
    {"name": "Fully developed chaos", "xmin": 3.80, "xmax": 4.00, "color": "#F3E5F5"},
]

# Extended sweep across all regimes
R_VALUES = np.round(np.concatenate([
    np.arange(2.80, 3.01, 0.05),   # periodic
    np.arange(3.05, 3.58, 0.05),   # period-doubling
    np.arange(3.58, 3.81, 0.03),   # chaotic
    np.arange(3.82, 4.01, 0.03),   # fully developed chaos
]), 2)
R_VALUES = sorted(np.unique(R_VALUES.tolist()))

# Examples for matrix visualization
R_EXAMPLES = [2.90, 3.20, 3.50, 3.60, 3.83, 4.00]

# QTN/QG lags
K_VALUES = [1, 2, 3]

# MTF settings
BIN_MODE = "quantile"
SMOOTH_EPS = 1e-3

# Typography for publication-quality plots
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

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

def classify_regime(r: float) -> str:
    if 2.80 <= r <= 3.00:
        return "Periodic"
    elif 3.00 < r <= 3.57:
        return "Period-doubling"
    elif 3.57 < r < 3.80:
        return "Chaotic"
    elif 3.80 <= r <= 4.00:
        return "Fully developed chaos"
    return "Outside range"

def regime_color(regime: str) -> str:
    for band in REGIME_BANDS:
        if band["name"] == regime:
            return band["color"]
    return "#FFFFFF"

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
    P[nz] = C[nz] / row_sums[nz]
    P[~nz] = 1.0 / Q
    return P

# =====================================================
# 4) Small-world sigma
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

    gamma = C_obs / C_rand if (np.isfinite(C_obs) and np.isfinite(C_rand) and C_rand > 0) else np.nan
    lambd = L_obs / L_rand if (np.isfinite(L_obs) and np.isfinite(L_rand) and L_rand > 0) else np.nan
    sigma = gamma / lambd if (np.isfinite(gamma) and np.isfinite(lambd) and lambd != 0) else np.nan
    return sigma

# =====================================================
# 5) Compute results
# =====================================================
def compute_logistic_sweep(Q: int, n: int, r_values, n_realizations: int) -> pd.DataFrame:
    rows = []
    for r_idx, r in enumerate(r_values):
        for seed in range(n_realizations):
            x = logistic_map_series(n=n, r=float(r), seed=seed, burn=1000)
            ac1 = autocorr_lag1(x)
            regime = classify_regime(float(r))

            A = qg_qtn_counts(x, Q=Q, k_values=K_VALUES)
            W_qtn = (A + A.T).astype(float)
            s_qtn = small_world_sigma_from_W(W_qtn, seed=10_000 * r_idx + seed)
            rows.append({"r": float(r), "regime": regime, "method": "QTN", "sigma": s_qtn, "acf1": ac1})

            W_gaf = gaf_matrix(x, Q=Q)
            s_gaf = small_world_sigma_from_W(W_gaf, seed=20_000 * r_idx + seed)
            rows.append({"r": float(r), "regime": regime, "method": "GAF", "sigma": s_gaf, "acf1": ac1})

            W_mtf = mtf_transition_matrix(x, Q=Q, bin_mode="quantile", smooth_eps=SMOOTH_EPS)
            s_mtf = small_world_sigma_from_W(W_mtf, seed=30_000 * r_idx + seed)
            rows.append({"r": float(r), "regime": regime, "method": "MTF", "sigma": s_mtf, "acf1": ac1})

    df = pd.DataFrame(rows)
    df = df[np.isfinite(df["sigma"]) & np.isfinite(df["acf1"])].copy()
    return df

# =====================================================
# 6) Plot 1: examples with regime labels
# =====================================================
def plot_examples_logistic_with_regimes(
    Q: int,
    n: int,
    r_examples,
    outfile_prefix="logistic_examples_qtn_gaf_mtf_regimes",
    show_T=400
):
    rows = []
    for idx, r in enumerate(r_examples):
        x = logistic_map_series(n=n, r=float(r), seed=idx, burn=1000)

        A = qg_qtn_counts(x, Q=Q, k_values=K_VALUES)
        W_qtn = (A + A.T).astype(float)
        W_gaf = gaf_matrix(x, Q=Q)
        W_mtf = mtf_transition_matrix(x, Q=Q, bin_mode="quantile", smooth_eps=SMOOTH_EPS)

        regime = classify_regime(float(r))
        rows.append((x, W_qtn, W_gaf, W_mtf, r, regime))

    qtn_max = max(np.max(Wqtn) for _x, Wqtn, _g, _m, _r, _reg in rows)
    if qtn_max <= 0:
        qtn_max = 1.0
    gaf_vmax = max(np.max(Wgaf) for _x, _q, Wgaf, _m, _r, _reg in rows)
    mtf_vmax = max(np.max(Wmtf) for _x, _q, _g, Wmtf, _r, _reg in rows)

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 4, figsize=(15, 2.8 * nrows), tight_layout=True)

    if nrows == 1:
        axes = np.array([axes])

    for i, (x, Wqtn, Wgaf, Wmtf, r, regime) in enumerate(rows):
        bg = regime_color(regime)

        # ---- time series
        ax_ts = axes[i, 0]
        ax_ts.set_facecolor(bg)
        t = np.arange(show_T)
        ax_ts.plot(t, x[:show_T], linewidth=1.2)
        ax_ts.set_title(f"r = {r:.2f} | {regime}")
        ax_ts.set_xlabel("Time")
        ax_ts.set_ylabel("Value")

        # ---- QTN
        ax_qtn = axes[i, 1]
        ax_qtn.set_facecolor(bg)
        im1 = ax_qtn.imshow(
            Wqtn / qtn_max,
            origin="lower",
            aspect="equal",
            vmin=0.0,
            vmax=1.0
        )
        ax_qtn.set_title(f"QTN/QG (Q={Q})")
        ax_qtn.set_xlabel("Bin j")
        ax_qtn.set_ylabel("Bin i")
        fig.colorbar(im1, ax=ax_qtn, fraction=0.046, pad=0.04)

        # ---- GAF
        ax_gaf = axes[i, 2]
        ax_gaf.set_facecolor(bg)
        im2 = ax_gaf.imshow(
            Wgaf,
            origin="lower",
            aspect="equal",
            vmin=-1.0,
            vmax=max(1.0, gaf_vmax)
        )
        ax_gaf.set_title(f"GAF (Q={Q})")
        ax_gaf.set_xlabel("i")
        ax_gaf.set_ylabel("j")
        fig.colorbar(im2, ax=ax_gaf, fraction=0.046, pad=0.04)

        # ---- MTF
        ax_mtf = axes[i, 3]
        ax_mtf.set_facecolor(bg)
        im3 = ax_mtf.imshow(
            Wmtf,
            origin="lower",
            aspect="equal",
            vmin=0.0,
            vmax=max(1e-12, mtf_vmax)
        )
        ax_mtf.set_title(f"MTF (Q={Q})")
        ax_mtf.set_xlabel("Next state j")
        ax_mtf.set_ylabel("Current state i")
        fig.colorbar(im3, ax=ax_mtf, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Logistic map examples across dynamical regimes: time series and induced network representations",
        y=1.02
    )
    plt.savefig(f"{outfile_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{outfile_prefix}.pdf", dpi=300, bbox_inches="tight")
    plt.show()

# =====================================================
# 7) Plot 2: publication-style sigma boxplot with regime bands
# =====================================================
def plot_sigma_boxplot_by_r(df: pd.DataFrame, r_values, outfile_prefix="logistic_sigma_by_r_regimes"):
    r_values = sorted(r_values)
    pos_map = {r: i for i, r in enumerate(r_values)}
    base = np.arange(len(r_values))
    offset = 0.25

    pos_qtn = base - offset
    pos_gaf = base
    pos_mtf = base + offset

    data_qtn = [df[(df["r"] == r) & (df["method"] == "QTN")]["sigma"].tolist() for r in r_values]
    data_gaf = [df[(df["r"] == r) & (df["method"] == "GAF")]["sigma"].tolist() for r in r_values]
    data_mtf = [df[(df["r"] == r) & (df["method"] == "MTF")]["sigma"].tolist() for r in r_values]

    fig, ax = plt.subplots(figsize=(16, 6))

    # Background dynamical-regime bands
    for band in REGIME_BANDS:
        in_band = [r for r in r_values if band["xmin"] <= r <= band["xmax"]]
        if not in_band:
            continue
        xmin = pos_map[min(in_band)] - 0.6
        xmax = pos_map[max(in_band)] + 0.6
        ax.axvspan(xmin, xmax, color=band["color"], alpha=0.65, zorder=0)
        ax.text((xmin + xmax) / 2, 0.98, band["name"],
                ha="center", va="top", fontsize=11,
                transform=ax.get_xaxis_transform())

    def _boxplot(data, positions, facecolor):
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.22,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker="^", markerfacecolor="#E76F51", markeredgecolor="#E76F51", markersize=6),
            medianprops=dict(color="#F4A261", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker='o', markersize=4, markerfacecolor='white', markeredgecolor='black', alpha=0.8),
            manage_ticks=False
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(facecolor)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.2)
        return bp

    _boxplot(data_qtn, pos_qtn, PALETTE_DATASET["QTN"])
    _boxplot(data_gaf, pos_gaf, PALETTE_DATASET["GAF"])
    _boxplot(data_mtf, pos_mtf, PALETTE_DATASET["MTF"])

    ax.axhline(1.0, linestyle="--", linewidth=1.4, color="#3A3A3A", alpha=0.9)
    ax.set_ylabel("Small-world index $\\sigma$")
    ax.set_xlabel("Logistic map parameter $r$")
    ax.set_title("Small-world topology across logistic-map regimes")

    ax.set_xticks(base)
    ax.set_xticklabels([f"{r:.2f}" for r in r_values], rotation=45, ha="right")

    legend_handles = [
        Patch(facecolor=PALETTE_DATASET["QTN"], edgecolor="black", label="QTN (QG)"),
        Patch(facecolor=PALETTE_DATASET["GAF"], edgecolor="black", label="GAF"),
        Patch(facecolor=PALETTE_DATASET["MTF"], edgecolor="black", label="MTF"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    ax.grid(axis="y", linestyle=":", alpha=0.35)
    plt.tight_layout()

    fig.savefig(f"{outfile_prefix}.png", dpi=1000, bbox_inches="tight")
    fig.savefig(f"{outfile_prefix}.pdf", dpi=1000, bbox_inches="tight")
    plt.show()

# =====================================================
# 8) Plot 3: ACF(1) vs sigma
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
# 9) Export summaries
# =====================================================
def export_summary_tables(df: pd.DataFrame, outfile_prefix="logistic_sigma_summary"):
    summary = (
        df.groupby(["r", "regime", "method"], as_index=False)
          .agg(
              sigma_mean=("sigma", "mean"),
              sigma_median=("sigma", "median"),
              sigma_std=("sigma", "std"),
              sigma_sem=("sigma", lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
              sigma_min=("sigma", "min"),
              sigma_max=("sigma", "max"),
              acf1_mean=("acf1", "mean"),
          )
    )

    regime_summary = (
        df.groupby(["regime", "method"], as_index=False)
          .agg(
              sigma_mean=("sigma", "mean"),
              sigma_median=("sigma", "median"),
              sigma_std=("sigma", "std"),
              sigma_sem=("sigma", lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
              n=("sigma", "size"),
          )
    )

    summary.to_csv(f"{outfile_prefix}_by_r.csv", index=False)
    regime_summary.to_csv(f"{outfile_prefix}_by_regime.csv", index=False)
    return summary, regime_summary

# =====================================================
# 10) Main
# =====================================================
if __name__ == "__main__":
    df = compute_logistic_sweep(Q=Q, n=N, r_values=R_VALUES, n_realizations=N_REALIZATIONS)
    df.to_csv("logistic_sweep_sigma_acf1_full.csv", index=False)

    summary_r, summary_regime = export_summary_tables(df, outfile_prefix="logistic_sigma_summary")

    plot_examples_logistic_with_regimes(
        Q=Q,
        n=N,
        r_examples=R_EXAMPLES,
        outfile_prefix="logistic_examples_qtn_gaf_mtf_regimes",
        show_T=400
    )

    plot_sigma_boxplot_by_r(df, r_values=R_VALUES, outfile_prefix="logistic_sigma_by_r_regimes")
    plot_acf1_vs_sigma(df, outfile_prefix="logistic_acf1_vs_sigma")

    print("Saved:")
    print("- logistic_sweep_sigma_acf1_full.csv")
    print("- logistic_sigma_summary_by_r.csv")
    print("- logistic_sigma_summary_by_regime.csv")
    print("- logistic_examples_qtn_gaf_mtf_regimes.png")
    print("- logistic_examples_qtn_gaf_mtf_regimes.pdf")
    print("- logistic_sigma_by_r_regimes.png")
    print("- logistic_sigma_by_r_regimes.pdf")
    print("- logistic_acf1_vs_sigma.png")
    print("- logistic_acf1_vs_sigma.pdf")
 

# %% ---- next notebook cell ----


