"""Canonical size-robust fMRI (BASC-122) processor: reads per-subject T x 122
BOLD time-series CSVs, builds QTN/GAF/MTF weighted networks, and writes the
size-robust metric CSVs used throughout the paper. This is the reference
implementation that smallworld_qtn.network_metrics was extracted from.

Provenance: extracted verbatim from the notebook ``SMALL-world-FINAL.ipynb``.
"""

# pipeline_size_robust_timeseries_only.py
# Compute size-robust graph metrics from time-series via QTN/GAF/MTF
# (NO Spearman processing)

import os, glob, re, warnings
import numpy as np
import pandas as pd
import networkx as nx
from typing import List

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =======================
# USER CONFIG / PATHS
# =======================
#DIR_TIMESERIES = "Bold-time-series-TD/ABIDE-data"   # folder with CSVs, each is T x 122 (time x regions), no header/index
#DIR_TIMESERIES = "Bold-time-series-TD/SCZ-data"
#DIR_TIMESERIES = "Bold-time-series-ASD/ASD"
DIR_TIMESERIES = "ADHD-bold/ADHD"
#OUT_QTN_QTN_CSV = "CSV-Small-world/metrics_QTN_SCZ.csv"
#OUT_QTN_GAF_CSV = "CSV-Small-world/metrics_GAF_SCZ.csv"
#OUT_QTN_MTF_CSV = "CSV-Small-world/metrics_MTF_SCZ.csv"
#OUT_SKIPPED_LOG = "CSV-Small-world/skipped_timeseries_sizeRobust.csv"
#OUT_QTN_QTN_CSV = "CSV-Small-world/metrics_QTN_SCZ-SCZ.csv"
#OUT_QTN_GAF_CSV = "CSV-Small-world/metrics_GAF_SCZ-SCZ.csv"
#OUT_QTN_MTF_CSV = "CSV-Small-world/metrics_MTF_SCZ-SCZ.csv"
#OUT_SKIPPED_LOG = "CSV-Small-world/skipped_timeseries_sizeRobust-SCZ.csv"
OUT_QTN_QTN_CSV = "CSV-Small-world/metrics_QTN_ADHD-ADHD.csv"
OUT_QTN_GAF_CSV = "CSV-Small-world/metrics_GAF_ADHD-ADHD.csv"
OUT_QTN_MTF_CSV = "CSV-Small-world/metrics_MTF_ADHD-ADHD.csv"
OUT_SKIPPED_LOG = "CSV-Small-world/skipped_timeseries_sizeRobust-ADHD.csv"

# Graph / null model settings
TARGET_DENSITY      = 0.10
N_RANDOMIZATIONS    = 20
REWIRINGS_PER_EDGE  = 5
RNG_SEED            = 1234

# QTN lag(s)
K_VALUES = [1, 2, 3]

# Robust CSV loader behavior
FILL_NANS_AS_ZERO = True   # fill unknowns with 0 (set False to fail-hard)

SKIPPED = []  # log skipped files

# =======================
# I/O HELPERS
# =======================
def list_csvs(path: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    if len(files) == 0:
        raise FileNotFoundError(f"No CSV files found in: {path}")
    return files

def load_numeric_matrix_csv(path):
    """
    Robust loader for headerless CSV numeric matrices.
    - drops fully-empty rows/cols
    - coerces to numeric
    - fills NaNs with 0 if FILL_NANS_AS_ZERO else raises
    """
    df = pd.read_csv(path, header=None, dtype=str, engine="python")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    n_nans = int(df.isna().sum().sum())
    if n_nans > 0:
        if FILL_NANS_AS_ZERO:
            print(f"[WARN] {os.path.basename(path)}: {n_nans} non-numeric/empty -> 0.")
            df = df.fillna(0.0)
        else:
            raise ValueError(f"{os.path.basename(path)} contains non-numeric cells.")
    return df.to_numpy(dtype=float)

def load_timeseries_matrix(path: str) -> np.ndarray:
    X = load_numeric_matrix_csv(path)   # T x 122 expected (we don't force)
    if X.ndim != 2:
        raise ValueError(f"{os.path.basename(path)} not 2D: {X.shape}")
    # Drop fully-empty rows (all zeros)
    keep_rows = ~np.all(X == 0, axis=1)
    if not np.all(keep_rows):
        X = X[keep_rows, :]
    return X

# =======================
# TIME-SERIES → QTN / GAF / MTF
# =======================
def calculate_quantile_graph_varying_k(signal, Q, k_values):
    A = np.zeros((Q, Q), dtype=int)
    n = len(signal)
    if n == 0:
        return A
    ranks = np.argsort(np.argsort(signal))
    q = np.linspace(0, n, Q + 1, dtype=int)
    loc = np.clip(np.searchsorted(q, ranks, side='right') - 1, 0, Q - 1)
    for k in k_values:
        if k <= 0:
            continue
        for i in range(n - k):
            A[loc[i], loc[i + k]] += 1
    return A

def calculate_gaf(signal, Q):
    min_val, max_val = float(np.min(signal)), float(np.max(signal))
    rng = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    scaled = 2 * (signal - min_val) / rng - 1
    scaled = np.clip(scaled, -1, 1)
    phi = np.arccos(scaled)
    gaf = np.cos(phi[:, None] + phi[None, :])
    return gaf[:Q, :Q]

def calculate_mtf(signal, Q):
    min_val, max_val = float(np.min(signal)), float(np.max(signal))
    rng = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    norm_signal = (signal - min_val) / rng
    bins = np.linspace(0, 1, Q + 1)
    symbols = np.digitize(norm_signal, bins) - 1
    symbols = np.clip(symbols, 0, Q - 1)
    trans_mat = np.zeros((Q, Q), dtype=float)
    for i in range(len(symbols) - 1):
        trans_mat[symbols[i], symbols[i + 1]] += 1
    row_sums = trans_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    trans_mat /= row_sums
    n = len(symbols)
    mtf = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            mtf[i, j] = trans_mat[symbols[i], symbols[j]]
    return mtf[:Q, :Q]

def compute_Q_from_T(T: int) -> int:
    return int(round(2 * (T ** (1/3))))

# =======================
# GRAPH METRIC HELPERS (size-robust)
# =======================
def proportional_binary_from_weights(W: np.ndarray, target_density: float) -> np.ndarray:
    n = W.shape[0]
    A = W.copy().astype(float)
    np.fill_diagonal(A, 0.0)
    upper_vals = np.abs(np.triu(A, 1))
    vals = upper_vals[upper_vals > 0]
    if vals.size == 0:
        return np.zeros_like(A, dtype=int)
    m_target = int(round(target_density * n * (n - 1) / 2))
    m_target = max(1, min(m_target, vals.size))
    thresh = np.partition(vals, -m_target)[-m_target]
    B = (np.abs(A) >= thresh).astype(int)
    B = np.triu(B, 1); B = B + B.T
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
        nx.double_edge_swap(G, nswap=max(1, swaps), max_tries=10*max(1, swaps), seed=seed)
    except Exception:
        pass
    return nx.to_numpy_array(G, dtype=int)

def global_efficiency_binary(B: np.ndarray) -> float:
    return nx.global_efficiency(nx.from_numpy_array(B))

def normalized_laplacian_eigs(B: np.ndarray) -> np.ndarray:
    G = nx.from_numpy_array(B)
    if G.number_of_nodes() == 0:
        return np.array([])
    Lnorm = nx.normalized_laplacian_matrix(G).toarray()
    w = np.linalg.eigvalsh(Lnorm)
    return np.clip(w, 0.0, 2.0)

def von_neumann_entropy_normalized(B: np.ndarray) -> float:
    n = B.shape[0]
    G = nx.from_numpy_array(B)
    m = G.number_of_edges()
    if n <= 1 or m == 0:
        return np.nan
    L = nx.laplacian_matrix(G).astype(float).toarray()
    rho = L / (2.0 * m)
    evals = np.linalg.eigvalsh(rho)
    evals = np.clip(evals, 0.0, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -np.nansum(evals * np.log(evals + 1e-15))
    return float(h / np.log(n))

def ring_lattice(n: int, m: int) -> np.ndarray:
    if n < 3 or m <= 0:
        return np.zeros((n, n), dtype=int)
    r = max(1, int(round(m / n)))
    B = np.zeros((n, n), dtype=int)
    for i in range(n):
        for t in range(1, r+1):
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
    Cr, Lr, Er = [], [], []
    rng = np.random.default_rng(seed)
    for _ in range(n_rand):
        Br = degree_preserving_randomize_binary(B, swaps, int(rng.integers(0, 1_000_000)))
        C = nx.transitivity(nx.from_numpy_array(Br))
        L = gcc_char_path_length_binary(Br)
        E = global_efficiency_binary(Br)
        if C > 0 and not np.isnan(L):
            Cr.append(C); Lr.append(L)
        Er.append(E)
    def _ms(arr):
        if len(arr) == 0: return (np.nan, np.nan)
        return (float(np.mean(arr)), float(np.std(arr) if len(arr) > 1 else 0.0))
    Cmu, Csd = _ms(Cr)
    Lmu, Lsd = _ms(Lr)
    Emu, Esd = _ms(Er)
    return Cmu, Csd, Lmu, Lsd, Emu, Esd

def small_world_omega_phi(B: np.ndarray, C: float, L: float,
                          Crand: float, Lrand: float) -> tuple:
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
    phi = 1.0 - np.sqrt((dC**2 + dL**2)/2.0)
    return float(omega), float(phi)

def compute_metrics_size_robust(W: np.ndarray,
                                target_density: float,
                                n_rand: int,
                                rewires_per_edge: int,
                                seed: int,
                                use_abs_weights: bool = True) -> dict:
    n = W.shape[0]
    Ww = np.abs(W) if use_abs_weights else W.copy()
    np.fill_diagonal(Ww, 0.0)
    B  = proportional_binary_from_weights(Ww, target_density)

    Gw = nx.from_numpy_array(Ww)
    Gw.remove_edges_from([(u,v) for u,v,w in Gw.edges(data=True) if w.get('weight',0.0)==0.0])
    degrees = np.array([d for _, d in nx.degree(nx.from_numpy_array(B))], dtype=float)
    avg_deg = float(degrees.mean()) if degrees.size else np.nan
    density = float(np.sum(B) / (n*(n-1))) if n > 1 else np.nan
    C_obs   = nx.transitivity(nx.from_numpy_array(B)) if n > 1 else np.nan
    L_obs   = gcc_char_path_length_binary(B)
    E_obs   = global_efficiency_binary(B)
    c_w     = float(np.mean(list(nx.clustering(Gw, weight='weight').values()))) if Gw.number_of_edges()>0 else np.nan
    try:
        assort  = nx.degree_assortativity_coefficient(Gw) if Gw.number_of_edges()>0 else np.nan
    except Exception:
        assort = np.nan

    Cmu, Csd, Lmu, Lsd, Emu, Esd = null_model_stats(B, n_rand, rewires_per_edge, seed)
    gamma  = (C_obs / Cmu) if (Cmu and not np.isnan(Cmu) and Cmu>0) else np.nan
    lambd  = (L_obs / Lmu) if (Lmu and not np.isnan(Lmu) and Lmu>0) else np.nan
    sigma  = (gamma / lambd) if (gamma and lambd and not np.isnan(gamma) and not np.isnan(lambd) and lambd!=0) else np.nan
    zC     = ((C_obs - Cmu) / Csd) if (Csd and not np.isnan(Csd) and Csd>0) else np.nan
    zL     = ((L_obs - Lmu) / Lsd) if (Lsd and not np.isnan(Lsd) and Lsd>0) else np.nan
    Enorm  = (E_obs / Emu) if (Emu and not np.isnan(Emu) and Emu>0) else np.nan
    Hvn    = von_neumann_entropy_normalized(B)
    eigs   = normalized_laplacian_eigs(B)
    lambda2= float(eigs[1]) if eigs.size >= 2 else np.nan
    omega, phi = small_world_omega_phi(B, C_obs, L_obs, Cmu, Lmu)

    return {
        # classic
        "n_nodes": n,
        "density": density,
        "avg_degree": avg_deg,
        "assortativity": assort,
        "transitivity": C_obs,
        "avg_clustering_weighted": c_w,
        "global_efficiency": E_obs,
        "char_path_len_gcc": L_obs,
        # size-robust / normalized
        "gamma_C_over_Crand": gamma,
        "lambda_L_over_Lrand": lambd,
        "sigma_small_world": sigma,
        "zC": zC,
        "zL": zL,
        "E_over_Erand": Enorm,
        "Hvn_norm": Hvn,
        "lambda2_normlap": lambda2,
        "omega": omega,
        "phi": phi,
    }

# =======================
# PIPELINE: time series only
# =======================
def _per_column_metrics_from_signal_matrix(X: np.ndarray, method: str, k_values, seed_base: int) -> dict:
    n_time, n_regions = X.shape
    if n_regions != 122:
        print(f"[WARN] time-series has {n_regions} columns (expected 122). Proceeding.")
    Q = compute_Q_from_T(n_time)

    per_col = []
    for j in range(n_regions):
        sig = X[:, j].astype(float)
        if method == "QTN":
            A = calculate_quantile_graph_varying_k(sig, Q=Q, k_values=k_values)
            W = A + A.T
            use_abs = False
        elif method == "GAF":
            W = calculate_gaf(sig, Q=Q)
            use_abs = True
        elif method == "MTF":
            W = calculate_mtf(sig, Q=Q)
            use_abs = False
        else:
            raise ValueError(f"Unknown method: {method}")

        if np.allclose(W, 0):
            m = {k: np.nan for k in [
                "n_nodes","density","avg_degree","assortativity","transitivity",
                "avg_clustering_weighted","global_efficiency","char_path_len_gcc",
                "gamma_C_over_Crand","lambda_L_over_Lrand","sigma_small_world",
                "zC","zL","E_over_Erand","Hvn_norm","lambda2_normlap","omega","phi"
            ]}
        else:
            m = compute_metrics_size_robust(
                W=W,
                target_density=TARGET_DENSITY,
                n_rand=N_RANDOMIZATIONS,
                rewires_per_edge=REWIRINGS_PER_EDGE,
                seed=seed_base + j,
                use_abs_weights=use_abs
            )
        per_col.append(m)

    dfc = pd.DataFrame(per_col)
    mean_metrics = dfc.mean(numeric_only=True).to_dict()
    mean_metrics["Q_used"] = int(Q)
    return mean_metrics

def process_timeseries_qtn(folder: str, method: str, k_values) -> pd.DataFrame:
    files = list_csvs(folder)
    rows = []
    for idx, f in enumerate(files, 1):
        try:
            X = load_timeseries_matrix(f)
        except Exception as e:
            SKIPPED.append({"file": os.path.basename(f), "dataset": method, "reason": f"{e}"})
            print(f"[SKIP] load {os.path.basename(f)}: {e}")
            continue
        mean_metrics = _per_column_metrics_from_signal_matrix(
            X=X, method=method, k_values=k_values, seed_base=RNG_SEED + 1000*idx
        )
        mean_metrics["patient_id"] = os.path.splitext(os.path.basename(f))[0]
        mean_metrics["n_regions"]  = X.shape[1]
        rows.append(mean_metrics)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("patient_id")

# =======================
# MAIN
# =======================
def main():
    print("Processing Time Series with QTN/GAF/MTF (size-robust)…")
    df_qtn = process_timeseries_qtn(DIR_TIMESERIES, method="QTN", k_values=K_VALUES)
    df_gaf = process_timeseries_qtn(DIR_TIMESERIES, method="GAF", k_values=K_VALUES)
    df_mtf = process_timeseries_qtn(DIR_TIMESERIES, method="MTF", k_values=K_VALUES)

    if not df_qtn.empty:
        df_qtn.to_csv(OUT_QTN_QTN_CSV);  print("  Saved:", OUT_QTN_QTN_CSV, df_qtn.shape)
    else:
        print("  [WARN] No QTN output saved.")
    if not df_gaf.empty:
        df_gaf.to_csv(OUT_QTN_GAF_CSV);  print("  Saved:", OUT_QTN_GAF_CSV, df_gaf.shape)
    else:
        print("  [WARN] No GAF output saved.")
    if not df_mtf.empty:
        df_mtf.to_csv(OUT_QTN_MTF_CSV);  print("  Saved:", OUT_QTN_MTF_CSV, df_mtf.shape)
    else:
        print("  [WARN] No MTF output saved.")

    if SKIPPED:
        pd.DataFrame(SKIPPED).to_csv(OUT_SKIPPED_LOG, index=False)
        print(f"Logged {len(SKIPPED)} skipped files -> {OUT_SKIPPED_LOG}")
    else:
        print("No files were skipped 🎉")

if __name__ == "__main__":
    main()


# %% ---- next notebook cell ----

# smallworld_diagnostics_sizeRobust_QTN_GAF_MTF_scatter.py
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------- file names (no Spearman) ----------
#OUT_QTN_QTN_CSV = "CSV-Small-world/metrics_QTN_ABIDE.csv"
#OUT_QTN_GAF_CSV = "CSV-Small-world/metrics_GAF_ABIDE.csv"
#OUT_QTN_MTF_CSV = "CSV-Small-world/metrics_MTF_ABIDE.csv"
#FILES = {
 #   "QTN": "CSV-Small-world/metrics_QTN_ADHD.csv",
  #  "GAF": "CSV-Small-world/metrics_GAF_ADHD.csv",
   # "MTF": "CSV-Small-world/metrics_MTF_ADHD.csv",
#}
#FILES = {
 #   "QTN": "CSV-Small-world/metrics_QTN_SCZ.csv",
  #  "GAF": "CSV-Small-world/metrics_GAF_SCZ.csv",
   # "MTF": "CSV-Small-world/metrics_MTF_SCZ.csv",
#}
#FILES = {
   # "QTN": "CSV-Small-world/metrics_QTN_SCZ-SCZ.csv",
  #  "GAF": "CSV-Small-world/metrics_GAF_SCZ-SCZ.csv",
 #   "MTF": "CSV-Small-world/metrics_MTF_SCZ-SCZ.csv",
#}

#FILES = {
 #   "QTN": "CSV-Small-world/metrics_QTN_ABIDE-ASD.csv",
  #  "GAF": "CSV-Small-world/metrics_GAF_ABIDE-ASD.csv",
   # "MTF": "CSV-Small-world/metrics_MTF_ABIDE-ASD.csv",
#}
FILES = {
    "QTN": "CSV-Small-world/metrics_QTN_ADHD-ADHD.csv",
    "GAF": "CSV-Small-world/metrics_GAF_ADHD-ADHD.csv",
    "MTF": "CSV-Small-world/metrics_MTF_ADHD-ADHD.csv",
}
# Your palette for the violins
PALETTE_DATASET = {
    "QTN": "#008080",   # teal
    "GAF": "#A9A9A9",   # darkgray
    "MTF": "#DDA0DD",   # plum
}

DATASET_ORDER = ["QTN", "GAF", "MTF"]

# column aliases
SIGMA_ALIASES  = ["small_world_sigma", "sigma_small_world", "sigma", "sigma_sw"]
GAMMA_ALIASES  = ["gamma_C_over_Crand", "gamma c over crand", "gamma_c_over_crand", "gamma"]
LAMBDA_ALIASES = ["lambda_L_over_Lrand", "lambda l over lrand", "lambda_l_over_lrand", "lambda"]

# thresholds
SIGMA_THRESH   = 1.0                 # not plotted here, but kept in summary
#OUT_DIR        = "ABIDE-RESULTS/smallworld_diag_scatter_QTN_GAF_MTF-ABIDE-TD"
#OUT_DIR        = "ADHD-RESULTS-Control/smallworld_diag_scatter_QTN_GAF_MTF-ADHD-TD"
#OUT_DIR        = "SCZ-RESULTS-Control/smallworld_diag_scatter_QTN_GAF_MTF-ADHD-TD"
#OUT_DIR        = "OTHER-RESULTS/SCZ/smallworld_diag_scatter_QTN_GAF_MTF-SCZ-SCZ"
OUT_DIR         = "OTHER-RESULTS/ADHD/smallworld_diag_scatter_QTN_GAF_MTF-ADHD-ADHD"
OUT_PLOT_PNG    = "gamma_lambda_scatter_QTN_GAF_MTF_shaded.png"
OUT_PLOT_PDF    = "gamma_lambda_scatter_QTN_GAF_MTF_shaded.pdf"
# your colors
GREEN_SHADE = "#14b814"  # nice green for the overlay
ALPHA_SHADE = 0.12       # transparency for overlay

sns.set_theme(context="notebook", style="white")
plt.rcParams.update({"axes.grid": False, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})

def _norm(s: str) -> str:
    return s.strip().lower().replace("-", "_").replace(" ", "_")

def _find_col(df: pd.DataFrame, aliases) -> str | None:
    norm_map = {_norm(c): c for c in df.columns}
    for a in aliases:
        key = _norm(a)
        if key in norm_map:
            return norm_map[key]
    return None

def _load_one(path: str, dataset: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[WARN] missing: {dataset} -> {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)

    c_sigma  = _find_col(df, SIGMA_ALIASES)
    c_gamma  = _find_col(df, GAMMA_ALIASES)
    c_lambda = _find_col(df, LAMBDA_ALIASES)

    # compute sigma if not found but gamma/lambda available
    if c_sigma is None and (c_gamma is not None and c_lambda is not None):
        sigma = pd.to_numeric(df[c_gamma], errors="coerce") / \
                pd.to_numeric(df[c_lambda], errors="coerce").replace({0: np.nan})
        df["__sigma__"] = sigma
        c_sigma = "__sigma__"

    if c_sigma is None:
        print(f"[WARN] {dataset}: no sigma column and cannot compute from gamma/lambda. Skipping.")
        return pd.DataFrame()

    out = pd.DataFrame({
        "dataset": dataset,
        "sigma":  pd.to_numeric(df[c_sigma],  errors="coerce"),
        "gamma":  pd.to_numeric(df[c_gamma],  errors="coerce") if c_gamma  else np.nan,
        "lambda": pd.to_numeric(df[c_lambda], errors="coerce") if c_lambda else np.nan,
    })
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["sigma"])
    return out

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    frames = []
    for name, path in FILES.items():
        frames.append(_load_one(path, name))
    data = pd.concat(frames, ignore_index=True)

    if data.empty:
        raise RuntimeError("No data loaded. Check CSV names/columns.")

    # order & colors
    data["dataset"] = pd.Categorical(data["dataset"], categories=DATASET_ORDER, ordered=True)

    # -------- γ vs λ scatter with shaded small-world region (γ>1 & λ<1) --------
    sub = data.dropna(subset=["gamma","lambda"]).copy()
    g = sns.FacetGrid(
        sub, col="dataset", col_order=DATASET_ORDER,
        sharex=False, sharey=False, height=3.6
    )
    g.map_dataframe(
        sns.scatterplot, x="gamma", y="lambda", alpha=0.8, s=30, edgecolor="none"
    )

    for ax, ds in zip(g.axes.flatten(), DATASET_ORDER):
        # color points by dataset
        for coll in ax.collections:
            coll.set_color(PALETTE_DATASET[ds])

        # draw guide lines at γ=1 and λ=1
        ax.axvline(1.0, ls="--", lw=1, color="black")
        ax.axhline(1.0, ls="--", lw=1, color="black")

        # compute current limits and shade the small-world quadrant: x>1, y<1
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        x_left = max(1.0, x_min)
        x_right = x_max
        y_bottom = y_min
        y_top = min(1.0, y_max)

        if (x_right > x_left) and (y_top > y_bottom):
            rect = Rectangle(
                (x_left, y_bottom),
                width=(x_right - x_left),
                height=(y_top - y_bottom),
                facecolor=GREEN_SHADE, edgecolor="none", alpha=ALPHA_SHADE, zorder=0
            )
            ax.add_patch(rect)

        ax.set_xlabel("γ = C/Crand")
        ax.set_ylabel("λ = L/Lrand")
        ax.set_title(ds)

    #g.figure.suptitle("γ vs λ (green region = small-world: γ>1 & λ<1)", y=1.02)
   # out_scatter = os.path.join(OUT_DIR, "gamma_lambda_scatter_QTN_GAF_MTF_shaded.png")
    #out_pdf = os.path.join(OUT_DIR, OUT_PLOT_PDF)
    #plt.tight_layout()
  #  g.figure.savefig(out_scatter, dpi=1000, bbox_inches="tight")
    out_png = os.path.join(OUT_DIR, OUT_PLOT_PNG)
    out_pdf = os.path.join(OUT_DIR, OUT_PLOT_PDF)
    plt.tight_layout()
    plt.savefig(out_png, dpi=1000, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=1000, bbox_inches="tight")
    plt.close()
    print(f"[OK] wrote scatter -> {out_pdf}")

    # optional: a compact numeric summary (no plots)
    summary = (
        data.groupby("dataset", observed=True)
            .agg(n=("sigma","count"),
                 sigma_mean=("sigma","mean"),
                 sigma_median=("sigma","median"),
                 sigma_std=("sigma","std"),
                 sigma_min=("sigma","min"),
                 sigma_max=("sigma","max"))
            .reset_index()
            .sort_values("dataset")
    )
    summary_path = os.path.join(OUT_DIR, "smallworld_summary_QTN_GAF_MTF.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[OK] wrote summary -> {summary_path}")

if __name__ == "__main__":
    main()


# %% ---- next notebook cell ----

# paired_violin_sigma_QTN_GAF_MTF.py
import os, re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- input files ----------
#FILES = {
 #   "QTN": "CSV-Small-world/metrics_QTN_ABIDE.csv",
  #  "GAF": "CSV-Small-world/metrics_GAF_ABIDE.csv",
   # "MTF": "CSV-Small-world/metrics_MTF_ABIDE.csv",
#}
#FILES = {
 #   "QTN": "CSV-Small-world/metrics_QTN_ADHD.csv",
  #  "GAF": "CSV-Small-world/metrics_GAF_ADHD.csv",
   # "MTF": "CSV-Small-world/metrics_MTF_ADHD.csv",
#}
#FILES = {
 #   "QTN": "CSV-Small-world/metrics_QTN_SCZ-SCZ.csv",
  #  "GAF": "CSV-Small-world/metrics_GAF_SCZ-SCZ.csv",
   # "MTF": "CSV-Small-world/metrics_MTF_SCZ-SCZ.csv",
#}

#FILES = {
 #   "QTN": "CSV-Small-world/metrics_QTN_ABIDE-ASD.csv",
  #  "GAF": "CSV-Small-world/metrics_GAF_ABIDE-ASD.csv",
   # "MTF": "CSV-Small-world/metrics_MTF_ABIDE-ASD.csv",
#}
FILES = {
    "QTN": "CSV-Small-world/metrics_QTN_ADHD-ADHD.csv",
    "GAF": "CSV-Small-world/metrics_GAF_ADHD-ADHD.csv",
    "MTF": "CSV-Small-world/metrics_MTF_ADHD-ADHD.csv",
}

DATASET_ORDER = ["QTN", "GAF", "MTF"]

# Your palette for the violins
PALETTE_DATASET = {
    "QTN": "#008080",   # teal
    "GAF": "#A9A9A9",   # darkgray
    "MTF": "#DDA0DD",   # plum
}

# Which column holds sigma? (auto-detect among these)
SIGMA_ALIASES = ["small_world_sigma", "sigma_small_world", "sigma", "sigma_sw"]

# Output
#OUT_DIR         = "ABIDE-RESULTS/paired_sigma_plots-ABIDE-TD"
#OUT_DIR         = "ADHD-RESULTS-Control/paired_sigma_plots-ADHD-TD"
#OUT_DIR         = "SCZ-RESULTS-Control/paired_sigma_plots-SCZ-TD"
OUT_DIR         = "OTHER-RESULTS/ADHD/paired_sigma_plots-ADHD-ADHD"
OUT_PLOT_PNG    = "sigma_violin_box_points_lines.png"
OUT_PLOT_PDF    = "sigma_violin_box_points_lines.pdf"
OUT_WIDE_CSV    = "sigma_wide_matched.csv"
OUT_LONG_CSV    = "sigma_long_matched.csv"

# Plot tuning
SIGMA_THRESH   = 1.0
POINT_SIZE     = 34
LINE_WIDTH     = 1.2
JITTER_SD      = 0.05
RNG_SEED       = 2025

# Style
sns.set_theme(context="notebook", style="white")
plt.rcParams.update({"axes.grid": False, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})

# ------------- helpers -------------
def _norm(s: str) -> str:
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")

def _find_sigma_col(df: pd.DataFrame) -> str | None:
    norm_map = {_norm(c): c for c in df.columns}
    for a in SIGMA_ALIASES:
        if _norm(a) in norm_map: return norm_map[_norm(a)]
    return None

def _normalize_key(s: str) -> str:
    s = str(s).strip().lower()
    m = re.search(r'(?:^|[^a-z0-9])(patient|td|hc|sz)[\s_-]*([0-9]+)\b', s)
    if m: return f"num_{int(m.group(2))}"
    m = re.search(r'([0-9]+)', s)
    if m: return f"num_{int(m.group(1))}"
    return re.sub(r'[^a-z0-9]+', '_', s)

def _load_sigma(path: str, dataset: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[WARN] missing {dataset}: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "patient_id" in df.columns:
        ids = df["patient_id"].astype(str)
    else:
        ids = pd.Series([f"{dataset}_{i}" for i in range(len(df))], name="patient_id")

    c_sigma = _find_sigma_col(df)
    if c_sigma is None:
        print(f"[WARN] {dataset}: no sigma column in {path}. Columns: {list(df.columns)[:10]} …")
        return pd.DataFrame()

    out = pd.DataFrame({
        "patient_id": ids,
        "key": [ _normalize_key(x) for x in ids ],
        dataset: pd.to_numeric(df[c_sigma], errors="coerce")
    })
    # average duplicates by key
    out = out.groupby(["key"], as_index=False)[dataset].mean(numeric_only=True)
    return out

# ------------- main -------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load each dataset's sigma
    tables = []
    for ds, path in FILES.items():
        d = _load_sigma(path, ds)
        if not d.empty:
            tables.append(d)
    if len(tables) < 2:
        raise RuntimeError("Could not load at least two datasets with sigma.")

    # Inner-join on key to keep subjects present in ALL sets
    wide = tables[0]
    for t in tables[1:]:
        wide = wide.merge(t, on="key", how="inner")

    keep_cols = ["key"] + DATASET_ORDER
    wide = wide[keep_cols].dropna()
    if wide.empty:
        raise RuntimeError("No overlapping subjects across QTN/GAF/MTF after matching keys.")

    # Save matched tables
    wide.to_csv(os.path.join(OUT_DIR, OUT_WIDE_CSV), index=False)
    long = wide.melt(id_vars=["key"], value_vars=DATASET_ORDER,
                     var_name="dataset", value_name="sigma")
    long["dataset"] = pd.Categorical(long["dataset"], categories=DATASET_ORDER, ordered=True)
    long.to_csv(os.path.join(OUT_DIR, OUT_LONG_CSV), index=False)

    # ---------- Plot ----------
    rng = np.random.default_rng(RNG_SEED)
    xcats = {ds: i for i, ds in enumerate(DATASET_ORDER)}
    fig, ax = plt.subplots(figsize=(9, 5.4))

    # Violin per dataset with your palette
    sns.violinplot(
        data=long, x="dataset", y="sigma", order=DATASET_ORDER,
        inner=None, cut=0, palette=PALETTE_DATASET, alpha=0.7, linewidth=1, saturation=1, ax=ax
    )

    # Box overlay (white fill)
    sns.boxplot(
        data=long, x="dataset", y="sigma", order=DATASET_ORDER,
        width=0.28, showcaps=True,
        boxprops={"facecolor":"white", "zorder":3},
        showfliers=False, whiskerprops={"linewidth":1}, medianprops={"linewidth":1.5},
        ax=ax
    )

    # Threshold line
    ax.axhline(SIGMA_THRESH, ls="--", lw=1.2, color="black", alpha=0.9)
    ax.text(len(DATASET_ORDER)-0.05, SIGMA_THRESH+0.01, f"σ = {SIGMA_THRESH}",
            ha="right", va="bottom", fontsize=10)

    # Grey, semi-transparent lines/points connecting same subject across methods
    line_color = "#9e9e9e"
    point_face = "#bdbdbd"
    point_edge = "#f2f2f2"

    for _, row in wide.iterrows():
        jitters = rng.normal(0, JITTER_SD, size=len(DATASET_ORDER))
        xs = [xcats[ds] + jitters[i] for i, ds in enumerate(DATASET_ORDER)]
        ys = [row[ds] for ds in DATASET_ORDER]
        # line
        ax.plot(xs, ys, color=line_color, alpha=0.3, lw=LINE_WIDTH, zorder=3.5)
        # points
        ax.scatter(xs, ys, s=POINT_SIZE, facecolor=point_face, edgecolor=point_edge,
                   linewidths=0.5, alpha=0.7, zorder=4)

    ax.set_xlabel("")
    ax.set_ylabel(r"Small-worldness $\sigma$")
   # ax.set_title("Paired σ across methods (QTN / GAF / MTF)\n"
            #     )
    sns.despine(trim=True)

    out_png = os.path.join(OUT_DIR, OUT_PLOT_PNG)
    out_pdf = os.path.join(OUT_DIR, OUT_PLOT_PDF)
    plt.tight_layout()
    plt.savefig(out_png, dpi=1000, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=1000, bbox_inches="tight")
    plt.close()
    print(f"[OK] wrote: {out_png}\n[OK] wrote: {out_pdf}")

if __name__ == "__main__":
    main()

# %% ---- next notebook cell ----


