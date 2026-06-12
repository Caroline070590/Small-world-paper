"""Computes the ABIDE/TD real-vs-surrogate metrics (per-subject and group-summary
CSVs). The output of this script is consumed by
abide_surrogate_validation.py to produce Fig. 8.

Provenance: extracted verbatim from the notebook ``Untitled1.ipynb``.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import zscore
from tqdm.auto import tqdm


# ============================================================
# CONFIG
# ============================================================
DATA_DIR = Path("matrices-bold-time-series/TD")
OUT_DIR = Path("abide_td_surrogate_qg_gaf_mtf")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 1234
N_SURROGATES = 100

# Representation settings
K_VALUES = [1, 2, 3]
TARGET_DENSITY = 0.10

# Small-world null for each graph
N_RANDOMIZATIONS = 20
REWIRINGS_PER_EDGE = 5

METHODS = ["QG", "GAF", "MTF"]

OUT_SUBJECT = OUT_DIR / "td_subject_surrogate_qg_gaf_mtf.csv"
OUT_SUMMARY = OUT_DIR / "td_group_summary_surrogate_qg_gaf_mtf.csv"
OUT_SKIPPED = OUT_DIR / "td_skipped_subjects.csv"


# ============================================================
# Helpers
# ============================================================
def compute_Q_from_T(T: int) -> int:
    return int(round(2 * (T ** (1 / 3))))


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


# ============================================================
# Surrogates
# ============================================================
def phase_randomize_1d(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if np.any(~np.isfinite(x)) or np.std(x) == 0:
        return x.copy()

    n = len(x)
    xf = np.fft.rfft(x)
    amp = np.abs(xf)
    phase = np.angle(xf)

    if len(phase) > 2:
        phase[1:-1] = rng.uniform(0, 2 * np.pi, size=len(phase) - 2)

    xf_new = amp * np.exp(1j * phase)
    xs = np.fft.irfft(xf_new, n=n)
    return np.real(xs)


def multivariate_phase_surrogate(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    Xs = np.zeros_like(X, dtype=float)
    for j in range(X.shape[1]):
        Xs[:, j] = phase_randomize_1d(X[:, j], rng)
    return Xs


# ============================================================
# QG / GAF / MTF
# ============================================================
def calculate_quantile_graph_varying_k(signal_1d: np.ndarray, Q: int, k_values) -> np.ndarray:
    A = np.zeros((Q, Q), dtype=np.int64)
    n = int(signal_1d.size)
    if n <= 1 or Q <= 1:
        return A

    ranks = np.argsort(np.argsort(signal_1d))
    q_edges = np.linspace(0, n, Q + 1, dtype=int)
    loc = np.clip(np.searchsorted(q_edges, ranks, side="right") - 1, 0, Q - 1)

    for k in k_values:
        k = int(k)
        if k <= 0 or k >= n:
            continue
        src = loc[:-k]
        dst = loc[k:]
        np.add.at(A, (src, dst), 1)
    return A


def calculate_gaf_from_lengthQ(signal_Q: np.ndarray) -> np.ndarray:
    x = np.asarray(signal_Q, dtype=float)
    min_val, max_val = float(np.min(x)), float(np.max(x))
    rng = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    scaled = 2 * (x - min_val) / rng - 1
    scaled = np.clip(scaled, -1, 1)
    phi = np.arccos(scaled)
    return np.cos(phi[:, None] + phi[None, :])


def calculate_mtf_from_lengthQ(signal_Q: np.ndarray, Q: int) -> np.ndarray:
    x = np.asarray(signal_Q, dtype=float)
    min_val, max_val = float(np.min(x)), float(np.max(x))
    rng = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    norm_signal = (x - min_val) / rng

    bins = np.linspace(0, 1, Q + 1)
    symbols = np.digitize(norm_signal, bins) - 1
    symbols = np.clip(symbols, 0, Q - 1)

    trans_mat = np.zeros((Q, Q), dtype=float)
    for i in range(Q - 1):
        trans_mat[symbols[i], symbols[i + 1]] += 1.0

    row_sums = trans_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    trans_mat /= row_sums

    mtf = np.zeros((Q, Q), dtype=float)
    for i in range(Q):
        mtf[i, :] = trans_mat[symbols[i], symbols]
    return mtf


# ============================================================
# Small-world machinery
# ============================================================
def proportional_binary_from_weights(W: np.ndarray, target_density: float, use_abs: bool) -> np.ndarray:
    n = W.shape[0]
    A = W.astype(float).copy()
    np.fill_diagonal(A, 0.0)

    M = np.abs(A) if use_abs else A
    upper_vals = np.triu(M, 1)
    vals = upper_vals[upper_vals > 0]

    if vals.size == 0:
        return np.zeros((n, n), dtype=int)

    m_target = int(round(target_density * n * (n - 1) / 2))
    m_target = max(1, min(m_target, vals.size))
    thresh = np.partition(vals, -m_target)[-m_target]

    B = (M >= thresh).astype(int)
    B = np.triu(B, 1)
    B = B + B.T
    np.fill_diagonal(B, 0)
    return B


def gcc_char_path_length_binary(B: np.ndarray) -> float:
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


def degree_preserving_randomize_binary(B: np.ndarray, swaps: int, seed: int) -> np.ndarray:
    G = nx.from_numpy_array(B)
    if G.number_of_edges() == 0:
        return B.copy()
    try:
        nx.double_edge_swap(G, nswap=max(1, swaps), max_tries=10 * max(1, swaps), seed=seed)
    except Exception:
        pass
    return nx.to_numpy_array(G, dtype=int)


def ring_lattice(n: int, m: int) -> np.ndarray:
    if n < 3 or m <= 0:
        return np.zeros((n, n), dtype=int)

    r = max(1, int(round(m / n)))
    B = np.zeros((n, n), dtype=int)
    for i in range(n):
        for t in range(1, r + 1):
            j1 = (i + t) % n
            j2 = (i - t) % n
            B[i, j1] = B[j1, i] = 1
            B[i, j2] = B[j2, i] = 1

    current_m = int(B.sum() // 2)
    if current_m > m:
        G = nx.from_numpy_array(B)
        rng = np.random.default_rng(0)
        edges = list(G.edges())
        rng.shuffle(edges)
        to_remove = current_m - m
        removed = 0
        for u, v in edges:
            if removed >= to_remove:
                break
            G.remove_edge(u, v)
            if not nx.is_connected(G):
                G.add_edge(u, v)
                continue
            removed += 1
        B = nx.to_numpy_array(G, dtype=int)

    np.fill_diagonal(B, 0)
    return B


def null_model_stats(B: np.ndarray, n_rand: int, rewires_per_edge: int, seed: int):
    m = int(B.sum() // 2)
    swaps = max(1, rewires_per_edge * m)

    Cr, Lr = [], []
    rng = np.random.default_rng(seed)

    for _ in range(n_rand):
        Br = degree_preserving_randomize_binary(B, swaps, int(rng.integers(0, 1_000_000)))
        C = nx.transitivity(nx.from_numpy_array(Br))
        L = gcc_char_path_length_binary(Br)
        if C > 0 and not np.isnan(L):
            Cr.append(C)
            Lr.append(L)

    def _ms(arr):
        if len(arr) == 0:
            return np.nan, np.nan
        if len(arr) == 1:
            return float(np.mean(arr)), 0.0
        return float(np.mean(arr)), float(np.std(arr, ddof=1))

    Cmu, Csd = _ms(Cr)
    Lmu, Lsd = _ms(Lr)
    return Cmu, Csd, Lmu, Lsd


def small_world_omega_phi(B: np.ndarray, C: float, L: float, Crand: float, Lrand: float):
    n = B.shape[0]
    m = int(B.sum() // 2)
    if n < 3 or m == 0 or np.isnan(C) or np.isnan(L) or np.isnan(Crand) or np.isnan(Lrand):
        return np.nan, np.nan

    Blatt = ring_lattice(n, m)
    Clatt = nx.transitivity(nx.from_numpy_array(Blatt))
    Llatt = gcc_char_path_length_binary(Blatt)

    if Clatt == 0 or np.isnan(Llatt):
        return np.nan, np.nan

    omega = (Lrand / L) - (C / Clatt)

    denomC = (Crand - Clatt)
    denomL = (Llatt - Lrand)
    if denomC == 0 or denomL == 0:
        return float(omega), np.nan

    dC = (Crand - C) / denomC
    dL = (L - Lrand) / denomL
    dC = float(np.clip(dC, 0.0, 1.0))
    dL = float(np.clip(dL, 0.0, 1.0))
    phi = 1.0 - np.sqrt((dC**2 + dL**2) / 2.0)
    return float(omega), float(phi)


def compute_smallworld_metrics_from_W(W: np.ndarray, seed: int, use_abs_for_threshold: bool) -> dict:
    n = W.shape[0]
    np.fill_diagonal(W, 0.0)

    B = proportional_binary_from_weights(W, TARGET_DENSITY, use_abs=use_abs_for_threshold)
    G = nx.from_numpy_array(B)

    C_obs = nx.transitivity(G) if n > 1 else np.nan
    L_obs = gcc_char_path_length_binary(B)

    Cmu, Csd, Lmu, Lsd = null_model_stats(B, N_RANDOMIZATIONS, REWIRINGS_PER_EDGE, seed)

    gamma = (C_obs / Cmu) if (not np.isnan(Cmu) and Cmu > 0) else np.nan
    lambd = (L_obs / Lmu) if (not np.isnan(Lmu) and Lmu > 0) else np.nan
    sigma = (gamma / lambd) if (not np.isnan(gamma) and not np.isnan(lambd) and lambd != 0) else np.nan

    zC = ((C_obs - Cmu) / Csd) if (not np.isnan(Csd) and Csd > 0) else np.nan
    zL = ((L_obs - Lmu) / Lsd) if (not np.isnan(Lsd) and Lsd > 0) else np.nan

    omega, phi = small_world_omega_phi(B, C_obs, L_obs, Cmu, Lmu)

    return {
        "n_nodes": int(n),
        "density": float(np.sum(B) / (n * (n - 1))) if n > 1 else np.nan,
        "transitivity": float(C_obs) if not np.isnan(C_obs) else np.nan,
        "char_path_len_gcc": float(L_obs) if not np.isnan(L_obs) else np.nan,
        "gamma_C_over_Crand": float(gamma) if not np.isnan(gamma) else np.nan,
        "lambda_L_over_Lrand": float(lambd) if not np.isnan(lambd) else np.nan,
        "sigma_small_world": float(sigma) if not np.isnan(sigma) else np.nan,
        "zC": float(zC) if not np.isnan(zC) else np.nan,
        "zL": float(zL) if not np.isnan(zL) else np.nan,
        "omega": float(omega) if not np.isnan(omega) else np.nan,
        "phi": float(phi) if not np.isnan(phi) else np.nan,
        "global_efficiency": float(nx.global_efficiency(G)),
    }


# ============================================================
# Per subject / per method
# ============================================================
def build_method_matrix(x: np.ndarray, method: str, Q: int) -> np.ndarray:
    if method == "QG":
        A = calculate_quantile_graph_varying_k(x, Q=Q, k_values=K_VALUES)
        return (A + A.T).astype(float)
    elif method == "GAF":
        xQ = downsample_to_length(x, Q)
        return calculate_gaf_from_lengthQ(xQ)
    elif method == "MTF":
        xQ = downsample_to_length(x, Q)
        return calculate_mtf_from_lengthQ(xQ, Q=Q)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_subject_method_metrics(X: np.ndarray, method: str, seed_base: int) -> dict:
    T = X.shape[0]
    Q = compute_Q_from_T(T)

    per_roi_metrics = []
    for roi in range(X.shape[1]):
        x = X[:, roi].astype(float)
        x = x[np.isfinite(x)]
        if x.size < 30:
            continue

        W = build_method_matrix(x, method=method, Q=Q)

        use_abs = True if method == "GAF" else False
        metrics = compute_smallworld_metrics_from_W(
            W,
            seed=seed_base + 100 * roi,
            use_abs_for_threshold=use_abs
        )
        per_roi_metrics.append(metrics)

    if not per_roi_metrics:
        raise ValueError(f"No valid ROI metrics for method {method}")

    dfm = pd.DataFrame(per_roi_metrics)
    out = dfm.mean(numeric_only=True).to_dict()
    out["Q_used"] = Q
    out["n_rois_used"] = len(dfm)
    return out


def empirical_p_greater(obs, surr):
    surr = np.asarray(surr, dtype=float)
    surr = surr[np.isfinite(surr)]
    if len(surr) == 0 or not np.isfinite(obs):
        return np.nan
    return (np.sum(surr >= obs) + 1) / (len(surr) + 1)


def zscore_against_surrogates(obs, surr):
    surr = np.asarray(surr, dtype=float)
    surr = surr[np.isfinite(surr)]
    if len(surr) < 2 or not np.isfinite(obs):
        return np.nan
    sd = np.std(surr, ddof=1)
    if sd == 0:
        return np.nan
    return (obs - np.mean(surr)) / sd


# ============================================================
# MAIN
# ============================================================
rng = np.random.default_rng(RNG_SEED)

files = sorted(DATA_DIR.glob("*.csv"))
results = []
skipped = []

metric_names = [
    "transitivity",
    "char_path_len_gcc",
    "gamma_C_over_Crand",
    "lambda_L_over_Lrand",
    "sigma_small_world",
    "omega",
    "phi",
    "global_efficiency",
]

for i, fpath in enumerate(tqdm(files, desc="TD subjects")):
    subject_id = fpath.stem

    try:
        X = pd.read_csv(fpath).values

        if X.ndim != 2 or X.shape[1] != 122:
            skipped.append([subject_id, str(fpath), "bad_shape", str(X.shape)])
            continue

        X = zscore(X, axis=0, ddof=1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        for method in METHODS:
            obs = compute_subject_method_metrics(
                X, method=method, seed_base=RNG_SEED + 100000 * i + 1000
            )

            surr_rows = []
            for s in range(N_SURROGATES):
                Xs = multivariate_phase_surrogate(X, rng)
                Xs = zscore(Xs, axis=0, ddof=1)
                Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

                sm = compute_subject_method_metrics(
                    Xs, method=method, seed_base=RNG_SEED + 100000 * i + 5000 + s
                )
                surr_rows.append(sm)

            surr_df = pd.DataFrame(surr_rows)

            row = {
                "subject_id": subject_id,
                "group": "TD",
                "method": method,
                "n_timepoints": X.shape[0],
                "n_rois": X.shape[1],
                "Q_used": obs.get("Q_used", np.nan),
                "n_rois_used": obs.get("n_rois_used", np.nan),
            }

            for m in metric_names:
                obs_val = obs.get(m, np.nan)
                surr_vals = surr_df[m].values
                row[f"obs_{m}"] = obs_val
                row[f"surr_mean_{m}"] = np.nanmean(surr_vals)
                row[f"surr_std_{m}"] = np.nanstd(surr_vals, ddof=1)
                row[f"p_empirical_{m}"] = empirical_p_greater(obs_val, surr_vals)
                row[f"zscore_vs_surr_{m}"] = zscore_against_surrogates(obs_val, surr_vals)

            results.append(row)

    except Exception as e:
        skipped.append([subject_id, str(fpath), "error", str(e)])

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_SUBJECT, index=False)

if skipped:
    pd.DataFrame(skipped, columns=["subject_id", "file", "reason", "details"]).to_csv(OUT_SKIPPED, index=False)

summary = (
    results_df.groupby("method")
    .agg(
        n_subjects=("subject_id", "count"),
        mean_obs_sigma=("obs_sigma_small_world", "mean"),
        mean_surr_sigma=("surr_mean_sigma_small_world", "mean"),
        mean_z_sigma=("zscore_vs_surr_sigma_small_world", "mean"),
        frac_p_lt_0_05_sigma=("p_empirical_sigma_small_world", lambda x: np.mean(x < 0.05)),
        mean_obs_phi=("obs_phi", "mean"),
        mean_surr_phi=("surr_mean_phi", "mean"),
        mean_z_phi=("zscore_vs_surr_phi", "mean"),
        frac_p_lt_0_05_phi=("p_empirical_phi", lambda x: np.mean(x < 0.05)),
        mean_obs_transitivity=("obs_transitivity", "mean"),
        mean_surr_transitivity=("surr_mean_transitivity", "mean"),
        mean_z_transitivity=("zscore_vs_surr_transitivity", "mean"),
    )
    .reset_index()
)

summary.to_csv(OUT_SUMMARY, index=False)

print("Saved:", OUT_SUBJECT)
print("Saved:", OUT_SUMMARY)
print("Saved:", OUT_SKIPPED)

# %% ---- next notebook cell ----

import os
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import zscore
from tqdm.auto import tqdm
from joblib import Parallel, delayed

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = Path("matrices-bold-time-series/TD")
OUT_DIR = Path("abide_td_surrogate_qg_gaf_mtf")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 1234
N_SURROGATES = 100

# Parallel
N_JOBS = 4          # use all cores; change to e.g. 8 if needed
BACKEND = "loky"     # process-based, safer for networkx/numpy
BATCH_SIZE = 1

# Representation settings
K_VALUES = [1, 2, 3]
TARGET_DENSITY = 0.10

# Small-world null for each graph
N_RANDOMIZATIONS = 20
REWIRINGS_PER_EDGE = 5

METHODS = ["QG", "GAF", "MTF"]

OUT_SUBJECT = OUT_DIR / "td_subject_surrogate_qg_gaf_mtf.csv"
OUT_SUMMARY = OUT_DIR / "td_group_summary_surrogate_qg_gaf_mtf.csv"
OUT_SKIPPED = OUT_DIR / "td_skipped_subjects.csv"


# ============================================================
# Helpers
# ============================================================
def compute_Q_from_T(T: int) -> int:
    return int(round(2 * (T ** (1 / 3))))


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


# ============================================================
# Surrogates
# ============================================================
def phase_randomize_1d(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if np.any(~np.isfinite(x)) or np.std(x) == 0:
        return x.copy()

    n = len(x)
    xf = np.fft.rfft(x)
    amp = np.abs(xf)
    phase = np.angle(xf)

    if len(phase) > 2:
        phase[1:-1] = rng.uniform(0, 2 * np.pi, size=len(phase) - 2)

    xf_new = amp * np.exp(1j * phase)
    xs = np.fft.irfft(xf_new, n=n)
    return np.real(xs)


def multivariate_phase_surrogate(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    Xs = np.zeros_like(X, dtype=float)
    for j in range(X.shape[1]):
        Xs[:, j] = phase_randomize_1d(X[:, j], rng)
    return Xs


# ============================================================
# QG / GAF / MTF
# ============================================================
def calculate_quantile_graph_varying_k(signal_1d: np.ndarray, Q: int, k_values) -> np.ndarray:
    A = np.zeros((Q, Q), dtype=np.int64)
    n = int(signal_1d.size)
    if n <= 1 or Q <= 1:
        return A

    ranks = np.argsort(np.argsort(signal_1d))
    q_edges = np.linspace(0, n, Q + 1, dtype=int)
    loc = np.clip(np.searchsorted(q_edges, ranks, side="right") - 1, 0, Q - 1)

    for k in k_values:
        k = int(k)
        if k <= 0 or k >= n:
            continue
        src = loc[:-k]
        dst = loc[k:]
        np.add.at(A, (src, dst), 1)
    return A


def calculate_gaf_from_lengthQ(signal_Q: np.ndarray) -> np.ndarray:
    x = np.asarray(signal_Q, dtype=float)
    min_val, max_val = float(np.min(x)), float(np.max(x))
    rngv = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    scaled = 2 * (x - min_val) / rngv - 1
    scaled = np.clip(scaled, -1, 1)
    phi = np.arccos(scaled)
    return np.cos(phi[:, None] + phi[None, :])


def calculate_mtf_from_lengthQ(signal_Q: np.ndarray, Q: int) -> np.ndarray:
    x = np.asarray(signal_Q, dtype=float)
    min_val, max_val = float(np.min(x)), float(np.max(x))
    rngv = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    norm_signal = (x - min_val) / rngv

    bins = np.linspace(0, 1, Q + 1)
    symbols = np.digitize(norm_signal, bins) - 1
    symbols = np.clip(symbols, 0, Q - 1)

    trans_mat = np.zeros((Q, Q), dtype=float)
    for i in range(Q - 1):
        trans_mat[symbols[i], symbols[i + 1]] += 1.0

    row_sums = trans_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    trans_mat /= row_sums

    mtf = np.zeros((Q, Q), dtype=float)
    for i in range(Q):
        mtf[i, :] = trans_mat[symbols[i], symbols]
    return mtf


# ============================================================
# Small-world machinery
# ============================================================
def proportional_binary_from_weights(W: np.ndarray, target_density: float, use_abs: bool) -> np.ndarray:
    n = W.shape[0]
    A = W.astype(float).copy()
    np.fill_diagonal(A, 0.0)

    M = np.abs(A) if use_abs else A
    upper_vals = np.triu(M, 1)
    vals = upper_vals[upper_vals > 0]

    if vals.size == 0:
        return np.zeros((n, n), dtype=int)

    m_target = int(round(target_density * n * (n - 1) / 2))
    m_target = max(1, min(m_target, vals.size))
    thresh = np.partition(vals, -m_target)[-m_target]

    B = (M >= thresh).astype(int)
    B = np.triu(B, 1)
    B = B + B.T
    np.fill_diagonal(B, 0)
    return B


def gcc_char_path_length_binary(B: np.ndarray) -> float:
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


def degree_preserving_randomize_binary(B: np.ndarray, swaps: int, seed: int) -> np.ndarray:
    G = nx.from_numpy_array(B)
    if G.number_of_edges() == 0:
        return B.copy()
    try:
        nx.double_edge_swap(G, nswap=max(1, swaps), max_tries=10 * max(1, swaps), seed=seed)
    except Exception:
        pass
    return nx.to_numpy_array(G, dtype=int)


def ring_lattice(n: int, m: int) -> np.ndarray:
    if n < 3 or m <= 0:
        return np.zeros((n, n), dtype=int)

    r = max(1, int(round(m / n)))
    B = np.zeros((n, n), dtype=int)
    for i in range(n):
        for t in range(1, r + 1):
            j1 = (i + t) % n
            j2 = (i - t) % n
            B[i, j1] = B[j1, i] = 1
            B[i, j2] = B[j2, i] = 1

    current_m = int(B.sum() // 2)
    if current_m > m:
        G = nx.from_numpy_array(B)
        rng = np.random.default_rng(0)
        edges = list(G.edges())
        rng.shuffle(edges)
        to_remove = current_m - m
        removed = 0
        for u, v in edges:
            if removed >= to_remove:
                break
            G.remove_edge(u, v)
            if not nx.is_connected(G):
                G.add_edge(u, v)
                continue
            removed += 1
        B = nx.to_numpy_array(G, dtype=int)

    np.fill_diagonal(B, 0)
    return B


def null_model_stats(B: np.ndarray, n_rand: int, rewires_per_edge: int, seed: int):
    m = int(B.sum() // 2)
    swaps = max(1, rewires_per_edge * m)

    Cr, Lr = [], []
    rng = np.random.default_rng(seed)

    for _ in range(n_rand):
        Br = degree_preserving_randomize_binary(B, swaps, int(rng.integers(0, 1_000_000)))
        C = nx.transitivity(nx.from_numpy_array(Br))
        L = gcc_char_path_length_binary(Br)
        if C > 0 and not np.isnan(L):
            Cr.append(C)
            Lr.append(L)

    def _ms(arr):
        if len(arr) == 0:
            return np.nan, np.nan
        if len(arr) == 1:
            return float(np.mean(arr)), 0.0
        return float(np.mean(arr)), float(np.std(arr, ddof=1))

    Cmu, Csd = _ms(Cr)
    Lmu, Lsd = _ms(Lr)
    return Cmu, Csd, Lmu, Lsd


def small_world_omega_phi(B: np.ndarray, C: float, L: float, Crand: float, Lrand: float):
    n = B.shape[0]
    m = int(B.sum() // 2)
    if n < 3 or m == 0 or np.isnan(C) or np.isnan(L) or np.isnan(Crand) or np.isnan(Lrand):
        return np.nan, np.nan

    Blatt = ring_lattice(n, m)
    Clatt = nx.transitivity(nx.from_numpy_array(Blatt))
    Llatt = gcc_char_path_length_binary(Blatt)

    if Clatt == 0 or np.isnan(Llatt):
        return np.nan, np.nan

    omega = (Lrand / L) - (C / Clatt)

    denomC = (Crand - Clatt)
    denomL = (Llatt - Lrand)
    if denomC == 0 or denomL == 0:
        return float(omega), np.nan

    dC = (Crand - C) / denomC
    dL = (L - Lrand) / denomL
    dC = float(np.clip(dC, 0.0, 1.0))
    dL = float(np.clip(dL, 0.0, 1.0))
    phi = 1.0 - np.sqrt((dC**2 + dL**2) / 2.0)
    return float(omega), float(phi)


def compute_smallworld_metrics_from_W(W: np.ndarray, seed: int, use_abs_for_threshold: bool) -> dict:
    n = W.shape[0]
    np.fill_diagonal(W, 0.0)

    B = proportional_binary_from_weights(W, TARGET_DENSITY, use_abs=use_abs_for_threshold)
    G = nx.from_numpy_array(B)

    C_obs = nx.transitivity(G) if n > 1 else np.nan
    L_obs = gcc_char_path_length_binary(B)

    Cmu, Csd, Lmu, Lsd = null_model_stats(B, N_RANDOMIZATIONS, REWIRINGS_PER_EDGE, seed)

    gamma = (C_obs / Cmu) if (not np.isnan(Cmu) and Cmu > 0) else np.nan
    lambd = (L_obs / Lmu) if (not np.isnan(Lmu) and Lmu > 0) else np.nan
    sigma = (gamma / lambd) if (not np.isnan(gamma) and not np.isnan(lambd) and lambd != 0) else np.nan

    zC = ((C_obs - Cmu) / Csd) if (not np.isnan(Csd) and Csd > 0) else np.nan
    zL = ((L_obs - Lmu) / Lsd) if (not np.isnan(Lsd) and Lsd > 0) else np.nan

    omega, phi = small_world_omega_phi(B, C_obs, L_obs, Cmu, Lmu)

    return {
        "n_nodes": int(n),
        "density": float(np.sum(B) / (n * (n - 1))) if n > 1 else np.nan,
        "transitivity": float(C_obs) if not np.isnan(C_obs) else np.nan,
        "char_path_len_gcc": float(L_obs) if not np.isnan(L_obs) else np.nan,
        "gamma_C_over_Crand": float(gamma) if not np.isnan(gamma) else np.nan,
        "lambda_L_over_Lrand": float(lambd) if not np.isnan(lambd) else np.nan,
        "sigma_small_world": float(sigma) if not np.isnan(sigma) else np.nan,
        "zC": float(zC) if not np.isnan(zC) else np.nan,
        "zL": float(zL) if not np.isnan(zL) else np.nan,
        "omega": float(omega) if not np.isnan(omega) else np.nan,
        "phi": float(phi) if not np.isnan(phi) else np.nan,
        "global_efficiency": float(nx.global_efficiency(G)),
    }


# ============================================================
# Per subject / per method
# ============================================================
def build_method_matrix(x: np.ndarray, method: str, Q: int) -> np.ndarray:
    if method == "QG":
        A = calculate_quantile_graph_varying_k(x, Q=Q, k_values=K_VALUES)
        return (A + A.T).astype(float)
    elif method == "GAF":
        xQ = downsample_to_length(x, Q)
        return calculate_gaf_from_lengthQ(xQ)
    elif method == "MTF":
        xQ = downsample_to_length(x, Q)
        return calculate_mtf_from_lengthQ(xQ, Q=Q)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_subject_method_metrics(X: np.ndarray, method: str, seed_base: int) -> dict:
    T = X.shape[0]
    Q = compute_Q_from_T(T)

    per_roi_metrics = []
    for roi in range(X.shape[1]):
        x = X[:, roi].astype(float)
        x = x[np.isfinite(x)]
        if x.size < 30:
            continue

        W = build_method_matrix(x, method=method, Q=Q)

        use_abs = True if method == "GAF" else False
        metrics = compute_smallworld_metrics_from_W(
            W,
            seed=seed_base + 100 * roi,
            use_abs_for_threshold=use_abs
        )
        per_roi_metrics.append(metrics)

    if not per_roi_metrics:
        raise ValueError(f"No valid ROI metrics for method {method}")

    dfm = pd.DataFrame(per_roi_metrics)
    out = dfm.mean(numeric_only=True).to_dict()
    out["Q_used"] = Q
    out["n_rois_used"] = len(dfm)
    return out


def empirical_p_greater(obs, surr):
    surr = np.asarray(surr, dtype=float)
    surr = surr[np.isfinite(surr)]
    if len(surr) == 0 or not np.isfinite(obs):
        return np.nan
    return (np.sum(surr >= obs) + 1) / (len(surr) + 1)


def zscore_against_surrogates(obs, surr):
    surr = np.asarray(surr, dtype=float)
    surr = surr[np.isfinite(surr)]
    if len(surr) < 2 or not np.isfinite(obs):
        return np.nan
    sd = np.std(surr, ddof=1)
    if sd == 0:
        return np.nan
    return (obs - np.mean(surr)) / sd


# ============================================================
# Worker
# ============================================================
METRIC_NAMES = [
    "transitivity",
    "char_path_len_gcc",
    "gamma_C_over_Crand",
    "lambda_L_over_Lrand",
    "sigma_small_world",
    "omega",
    "phi",
    "global_efficiency",
]


def process_subject(i: int, fpath: Path):
    subject_id = fpath.stem
    subject_rows = []
    subject_skips = []

    try:
        X = pd.read_csv(fpath).values

        if X.ndim != 2 or X.shape[1] != 122:
            subject_skips.append([subject_id, str(fpath), "bad_shape", str(X.shape)])
            return subject_rows, subject_skips

        X = zscore(X, axis=0, ddof=1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # subject-specific RNG for reproducibility
        base_seed = RNG_SEED + 10_000_000 * i
        rng = np.random.default_rng(base_seed)

        for method_idx, method in enumerate(METHODS):
            obs = compute_subject_method_metrics(
                X,
                method=method,
                seed_base=base_seed + 1000 * (method_idx + 1)
            )

            surr_rows = []
            for s in range(N_SURROGATES):
                Xs = multivariate_phase_surrogate(X, rng)
                Xs = zscore(Xs, axis=0, ddof=1)
                Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

                sm = compute_subject_method_metrics(
                    Xs,
                    method=method,
                    seed_base=base_seed + 100000 + 1000 * (method_idx + 1) + s
                )
                surr_rows.append(sm)

            surr_df = pd.DataFrame(surr_rows)

            row = {
                "subject_id": subject_id,
                "group": "TD",
                "method": method,
                "n_timepoints": X.shape[0],
                "n_rois": X.shape[1],
                "Q_used": obs.get("Q_used", np.nan),
                "n_rois_used": obs.get("n_rois_used", np.nan),
            }

            for m in METRIC_NAMES:
                obs_val = obs.get(m, np.nan)
                surr_vals = surr_df[m].values
                row[f"obs_{m}"] = obs_val
                row[f"surr_mean_{m}"] = np.nanmean(surr_vals)
                row[f"surr_std_{m}"] = np.nanstd(surr_vals, ddof=1)
                row[f"p_empirical_{m}"] = empirical_p_greater(obs_val, surr_vals)
                row[f"zscore_vs_surr_{m}"] = zscore_against_surrogates(obs_val, surr_vals)

            subject_rows.append(row)

    except Exception as e:
        subject_skips.append([subject_id, str(fpath), "error", str(e)])

    return subject_rows, subject_skips


# ============================================================
# MAIN
# ============================================================
files = sorted(DATA_DIR.glob("*.csv"))

parallel_results = Parallel(
    n_jobs=N_JOBS,
    backend=BACKEND,
    batch_size=BATCH_SIZE,
)(
    delayed(process_subject)(i, fpath)
    for i, fpath in tqdm(list(enumerate(files)), desc="Submitting TD subjects")
)

results = []
skipped = []

for rows, skips in parallel_results:
    results.extend(rows)
    skipped.extend(skips)

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_SUBJECT, index=False)

if skipped:
    pd.DataFrame(skipped, columns=["subject_id", "file", "reason", "details"]).to_csv(OUT_SKIPPED, index=False)

if not results_df.empty:
    summary = (
        results_df.groupby("method")
        .agg(
            n_subjects=("subject_id", "count"),
            mean_obs_sigma=("obs_sigma_small_world", "mean"),
            mean_surr_sigma=("surr_mean_sigma_small_world", "mean"),
            mean_z_sigma=("zscore_vs_surr_sigma_small_world", "mean"),
            frac_p_lt_0_05_sigma=("p_empirical_sigma_small_world", lambda x: np.mean(x < 0.05)),
            mean_obs_phi=("obs_phi", "mean"),
            mean_surr_phi=("surr_mean_phi", "mean"),
            mean_z_phi=("zscore_vs_surr_phi", "mean"),
            frac_p_lt_0_05_phi=("p_empirical_phi", lambda x: np.mean(x < 0.05)),
            mean_obs_transitivity=("obs_transitivity", "mean"),
            mean_surr_transitivity=("surr_mean_transitivity", "mean"),
            mean_z_transitivity=("zscore_vs_surr_transitivity", "mean"),
        )
        .reset_index()
    )
else:
    summary = pd.DataFrame()

summary.to_csv(OUT_SUMMARY, index=False)

print("Saved:", OUT_SUBJECT)
print("Saved:", OUT_SUMMARY)
print("Saved:", OUT_SKIPPED)

# %% ---- next notebook cell ----


