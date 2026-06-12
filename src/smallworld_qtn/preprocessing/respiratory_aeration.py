"""Respiratory aeration dataset (respiratory-heartrate-dataset/1.0.0) preprocessing.
Conservative QC for processed respiratory signals; band-pass per config; no envelope/smoothing.

Provenance: extracted verbatim from the notebook ``Respiratory-aeration-dataset.ipynb``.
The dataset-specific download, preprocessing, QC and CSV-writing logic is
preserved exactly as used to produce the published results. The shared
representation/metric definitions are duplicated inside this module (as in
the original notebook); the canonical copies live in
``smallworld_qtn.representations`` and ``smallworld_qtn.network_metrics``.

Run as a script (see ``scripts/``) after setting the CONFIG/paths block.
"""

#!/usr/bin/env python3
# respiratory_aeration_stream_to_csv_qtn_gaf_mtf.py
#
# Stream-to-CSV pipeline for the Respiratory Aeration dataset
# -----------------------------------------------------------
# Features:
# - download one CSV at a time
# - preprocess with stronger QC
# - compute QTN / GAF / MTF small-world metrics
# - append results incrementally
# - delete raw downloaded CSV after processing
# - keep only final output CSVs
#
# Outputs:
#   resp_outputs/out_resp_aeration_stream/
#       metrics_QTN_resp_aeration_stream.csv
#       metrics_GAF_resp_aeration_stream.csv
#       metrics_MTF_resp_aeration_stream.csv
#       skipped_resp_aeration_stream.csv
#
# Requirements:
#   pip install requests pandas numpy networkx scipy joblib tqdm

import os
import re
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin

import requests
import numpy as np
import pandas as pd
import networkx as nx

from scipy import signal
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIG
# ============================================================
BASE_URL = "https://physionet.org/files/respiratory-heartrate-dataset/1.0.0/Processed_Dataset/"

DATASET_NAME = "resp_aeration_stream"
OUT_DIR = "resp_outputs/out_resp_aeration_stream"
TMP_DIR = "tmp_resp_aeration"

# Adjust this if you know exact filenames
FILE_PATTERN_REGEX = r"^ProcessedData_Subject\d+\.csv$"

# If you already know subject count, you can hardcode.
# Otherwise, list manually after checking the site.
SUBJECT_FILES = [
    f"ProcessedData_Subject{i}.csv" for i in range(1, 21)
]

DELETE_RAW_AFTER_PROCESS = True
REQUEST_TIMEOUT = 120

# Columns to analyze
SIGNAL_COLS = [
    "PSD Flow [L/s]",
    "EIT Global Aeration",
]

# Optional nominal sampling frequency if known.
# If unknown, keep None.
DEFAULT_FS = None

K_VALUES = [1, 2, 3]
TARGET_DENSITY = 0.10
N_RANDOMIZATIONS = 10
REWIRINGS_PER_EDGE = 5
RNG_SEED = 1234

MAX_SAMPLES = None
START_SAMPLE = 0

N_JOBS = 2
BATCH_SIZE = 4
BACKEND = "loky"

METHODS = ["QTN", "GAF", "MTF"]

TQDM_ENABLED = True
TQDM_LEAVE = True

# -----------------------
# Preprocessing config
# -----------------------
PREPROCESS_MODE = "strong_fast"

# Detrend
DO_DETREND = True

# Bandpass / lowpass assumptions for respiratory-like signals
# If fs is unknown, filtering is skipped safely.
RESP_LOWPASS_HZ = 1.0
RESP_HIGHPASS_HZ = 0.01
BUTTER_ORDER = 4

# QC thresholds
MIN_VALID_SAMPLES = 200
MAX_ABS_Z = 10.0
MAX_FLAT_FRAC = 0.20           # reject if too much flat signal
MIN_STD = 1e-8
MIN_UNIQUE_VALUES = 20

# epoch QC
DO_EPOCH_QC = True
EPOCH_SEC = 30.0
MIN_KEEP_EPOCHS = 3
PTP_THRESHOLD_MULT = 6.0

# Missing data handling
MAX_INTERP_GAP_FRAC = 0.01     # short gaps only


# ============================================================
# Helpers
# ============================================================
def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)


def slice_signals(sig: np.ndarray) -> np.ndarray:
    start = int(START_SAMPLE or 0)
    if MAX_SAMPLES is None:
        return sig[start:, :]
    end = start + int(MAX_SAMPLES)
    return sig[start:end, :]


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


def robust_zscore(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        std = np.nanstd(x)
        scale = std if np.isfinite(std) and std > 0 else 1.0
    z = (x - med) / scale
    return np.clip(z, -MAX_ABS_Z, MAX_ABS_Z)


def subject_id_from_filename(fname: str) -> str:
    m = re.search(r"Subject(\d+)", fname, flags=re.IGNORECASE)
    if m:
        return f"S{int(m.group(1)):02d}"
    return Path(fname).stem


def out_paths(out_dir: str, dataset_name: str):
    return {
        "QTN": os.path.join(out_dir, f"metrics_QTN_{dataset_name}.csv"),
        "GAF": os.path.join(out_dir, f"metrics_GAF_{dataset_name}.csv"),
        "MTF": os.path.join(out_dir, f"metrics_MTF_{dataset_name}.csv"),
        "SKIP": os.path.join(out_dir, f"skipped_{dataset_name}.csv"),
    }


def load_done_ids(csv_path: str, id_col: str = "patient_id") -> set:
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
        if id_col in df.columns:
            return set(df[id_col].astype(str).tolist())
    except Exception:
        return set()
    return set()


def append_rows(csv_path: str, rows: List[dict], id_col: str = "patient_id"):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if id_col not in df.columns:
        raise ValueError(f"append_rows: missing '{id_col}'")
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=header)


def append_skips(csv_path: str, rows: List[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=header)


# ============================================================
# Download / cleanup
# ============================================================
def download_file(url: str, local_path: str):
    if os.path.exists(local_path):
        return
    ensure_out_dir(str(Path(local_path).parent))
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def cleanup_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ============================================================
# Physiological preprocessing
# ============================================================
def interpolate_short_gaps(x: np.ndarray, max_gap_frac: float = 0.01) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    finite = np.isfinite(x)
    n = len(x)

    if finite.sum() == 0:
        return x

    if finite.all():
        return x

    idx = np.arange(n)
    x_interp = x.copy()

    # full interpolation first
    x_interp[~finite] = np.interp(idx[~finite], idx[finite], x[finite])

    # reject if gaps too long
    gap_mask = ~finite
    if gap_mask.any():
        starts = np.where(np.diff(np.r_[False, gap_mask, False].astype(int)) == 1)[0]
        ends = np.where(np.diff(np.r_[False, gap_mask, False].astype(int)) == -1)[0]
        max_gap = max((e - s) for s, e in zip(starts, ends)) if len(starts) else 0
        if max_gap > max_gap_frac * n:
            raise ValueError(f"Gap too long for safe interpolation: {max_gap} samples.")
    return x_interp


def butter_filter(x: np.ndarray, fs: Optional[float], low: Optional[float], high: Optional[float], order: int = 4) -> np.ndarray:
    if fs is None or not np.isfinite(fs) or fs <= 0:
        return x

    nyq = 0.5 * fs

    if low is not None and high is not None:
        if high >= nyq:
            high = nyq * 0.99
        if low <= 0 or low >= high:
            return x
        sos = signal.butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    elif low is not None:
        if low <= 0 or low >= nyq:
            return x
        sos = signal.butter(order, low / nyq, btype="highpass", output="sos")
    elif high is not None:
        if high <= 0 or high >= nyq:
            return x
        sos = signal.butter(order, high / nyq, btype="lowpass", output="sos")
    else:
        return x

    return signal.sosfiltfilt(sos, x)


def flat_fraction(x: np.ndarray) -> float:
    dx = np.diff(x)
    if dx.size == 0:
        return 1.0
    return float(np.mean(np.abs(dx) < 1e-12))


def epoch_qc_keep_mask(x: np.ndarray, fs: Optional[float]) -> np.ndarray:
    if not DO_EPOCH_QC:
        return np.ones(1, dtype=bool)

    if fs is None or not np.isfinite(fs) or fs <= 0:
        return np.ones(1, dtype=bool)

    epoch_len = max(1, int(round(EPOCH_SEC * fs)))
    n_epochs = len(x) // epoch_len
    if n_epochs == 0:
        return np.zeros(0, dtype=bool)

    y = x[: n_epochs * epoch_len]
    epochs = y.reshape(n_epochs, epoch_len)
    ptp = np.ptp(epochs, axis=1)

    med = np.median(ptp)
    mad = np.median(np.abs(ptp - med))
    scale = 1.4826 * mad if mad > 0 else (np.std(ptp) if np.std(ptp) > 0 else 1.0)
    thr = med + PTP_THRESHOLD_MULT * scale
    return ptp <= thr


def preprocess_resp_signal(x: np.ndarray, fs: Optional[float]) -> Tuple[np.ndarray, dict]:
    qc = {}

    x = np.asarray(x, dtype=float)
    finite_before = np.isfinite(x)
    qc["n_total"] = int(x.size)
    qc["n_finite_before"] = int(finite_before.sum())
    qc["missing_frac_before"] = float(1.0 - finite_before.mean()) if x.size else np.nan

    if finite_before.sum() < MIN_VALID_SAMPLES:
        raise ValueError("Too few valid samples before preprocessing.")

    x = interpolate_short_gaps(x, max_gap_frac=MAX_INTERP_GAP_FRAC)

    if DO_DETREND:
        x = signal.detrend(x, type="linear")

    if PREPROCESS_MODE == "strong_fast":
        x = butter_filter(
            x,
            fs=fs,
            low=RESP_HIGHPASS_HZ,
            high=RESP_LOWPASS_HZ,
            order=BUTTER_ORDER,
        )

    if not np.all(np.isfinite(x)):
        raise ValueError("Non-finite values after filtering.")

    qc["std_before_z"] = float(np.std(x))
    qc["n_unique_before_z"] = int(np.unique(np.round(x, 10)).size)
    qc["flat_frac_before_z"] = flat_fraction(x)

    if qc["std_before_z"] <= MIN_STD:
        raise ValueError("Signal variance too small.")
    if qc["n_unique_before_z"] < MIN_UNIQUE_VALUES:
        raise ValueError("Too few unique values.")
    if qc["flat_frac_before_z"] > MAX_FLAT_FRAC:
        raise ValueError("Too much flat signal.")

    keep_mask = epoch_qc_keep_mask(x, fs=fs)
    if keep_mask.size > 0 and fs is not None and np.isfinite(fs) and fs > 0:
        epoch_len = max(1, int(round(EPOCH_SEC * fs)))
        n_epochs = len(x) // epoch_len
        x = x[: n_epochs * epoch_len].reshape(n_epochs, epoch_len)[keep_mask].reshape(-1)
        qc["n_epochs_total"] = int(n_epochs)
        qc["n_epochs_kept"] = int(keep_mask.sum())
        qc["epochs_kept_frac"] = float(keep_mask.mean()) if keep_mask.size else np.nan
        if keep_mask.sum() < MIN_KEEP_EPOCHS:
            raise ValueError(f"Too few clean epochs kept ({keep_mask.sum()}).")
    else:
        qc["n_epochs_total"] = np.nan
        qc["n_epochs_kept"] = np.nan
        qc["epochs_kept_frac"] = np.nan

    x = robust_zscore(x)

    qc["n_final"] = int(x.size)
    qc["std_final"] = float(np.std(x))
    qc["min_final"] = float(np.min(x))
    qc["max_final"] = float(np.max(x))

    if x.size < MIN_VALID_SAMPLES:
        raise ValueError("Too few valid samples after preprocessing.")

    return x, qc


# ============================================================
# Load processed respiratory CSV
# ============================================================
def load_resp_csv_matrix(filepath: str, signal_cols: List[str], default_fs: Optional[float]):
    df = pd.read_csv(filepath)

    cols = [c for c in signal_cols if c in df.columns]
    if not cols:
        raise ValueError(
            f"None of requested columns found in {filepath}. Available: {list(df.columns)}"
        )

    kept = []
    good_cols = []
    qc_rows = []

    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").values.astype(float)
        x_clean, qc = preprocess_resp_signal(x, fs=default_fs)
        kept.append(x_clean)
        good_cols.append(c)
        qc["signal_name"] = c
        qc_rows.append(qc)

    if not kept:
        raise ValueError("No valid respiratory columns after preprocessing.")

    min_len = min(len(x) for x in kept)
    arr = np.column_stack([x[:min_len] for x in kept])

    qc_df = pd.DataFrame(qc_rows)
    return arr, good_cols, default_fs, qc_df


# ============================================================
# QTN / GAF / MTF
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
# Graph / small-world
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
            return (np.nan, np.nan)
        return float(np.mean(arr)), float(np.std(arr) if len(arr) > 1 else 0.0)

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
    C_obs = nx.transitivity(nx.from_numpy_array(B)) if n > 1 else np.nan
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
        "global_efficiency": float(nx.global_efficiency(nx.from_numpy_array(B))),
    }


# ============================================================
# Core
# ============================================================
def file_to_all_method_rows(filepath: str, signal_cols: List[str], seed_base: int):
    sig, col_names, fs, qc_df = load_resp_csv_matrix(filepath, signal_cols=signal_cols, default_fs=DEFAULT_FS)
    sig = slice_signals(sig)

    T = sig.shape[0]
    if T < 50:
        raise ValueError("Too few samples after preprocessing.")
    Q = compute_Q_from_T(T)
    if Q < 3:
        raise ValueError(f"Q too small: {Q}")

    per_method_metrics = {m: [] for m in METHODS}

    for li in range(sig.shape[1]):
        x = sig[:, li].astype(float)
        x = x[np.isfinite(x)]
        if x.size < 50:
            continue

        A_qtn = calculate_quantile_graph_varying_k(x, Q=Q, k_values=K_VALUES)
        W_qtn = (A_qtn + A_qtn.T).astype(float)
        per_method_metrics["QTN"].append(
            compute_smallworld_metrics_from_W(W_qtn, seed=seed_base + 100 * li + 1, use_abs_for_threshold=False)
        )

        xQ = downsample_to_length(x, Q)
        W_gaf = calculate_gaf_from_lengthQ(xQ)
        per_method_metrics["GAF"].append(
            compute_smallworld_metrics_from_W(W_gaf, seed=seed_base + 100 * li + 2, use_abs_for_threshold=True)
        )

        W_mtf = calculate_mtf_from_lengthQ(xQ, Q=Q)
        per_method_metrics["MTF"].append(
            compute_smallworld_metrics_from_W(W_mtf, seed=seed_base + 100 * li + 3, use_abs_for_threshold=False)
        )

    out = {}
    qc_summary = qc_df.mean(numeric_only=True).to_dict()

    for method in METHODS:
        if not per_method_metrics[method]:
            raise ValueError(f"No valid signals produced metrics for {method}.")
        df = pd.DataFrame(per_method_metrics[method])
        avg = df.mean(numeric_only=True).to_dict()
        avg.update({
            "fs_hz": fs if fs is not None else np.nan,
            "n_signals": int(sig.shape[1]),
            "T_used_samples": int(T),
            "Q_used": int(Q),
            "signal_cols": ";".join(col_names),
            "qc_missing_frac_before_mean": qc_summary.get("missing_frac_before", np.nan),
            "qc_std_before_z_mean": qc_summary.get("std_before_z", np.nan),
            "qc_flat_frac_before_z_mean": qc_summary.get("flat_frac_before_z", np.nan),
            "qc_epochs_kept_frac_mean": qc_summary.get("epochs_kept_frac", np.nan),
            "qc_std_final_mean": qc_summary.get("std_final", np.nan),
        })
        out[method] = avg

    return out


def compute_task_subject(dataset_name: str, filename: str, task_i: int):
    patient_id = subject_id_from_filename(filename)
    local_path = os.path.join(TMP_DIR, filename)
    url = urljoin(BASE_URL, filename)

    try:
        download_file(url, local_path)
        rows = file_to_all_method_rows(local_path, SIGNAL_COLS, seed_base=RNG_SEED + 100000 * task_i)
        for method in METHODS:
            rows[method].update({
                "patient_id": patient_id,
                "dataset": dataset_name,
                "source_file": filename,
            })
        return patient_id, rows, None
    except Exception as e:
        skip = {
            "dataset": dataset_name,
            "method": "ALL",
            "record_or_patient": filename,
            "reason": str(e),
        }
        return patient_id, None, skip
    finally:
        if DELETE_RAW_AFTER_PROCESS:
            cleanup_file(local_path)


# ============================================================
# Runner
# ============================================================
def run_dataset():
    ensure_out_dir(OUT_DIR)
    ensure_out_dir(TMP_DIR)
    paths = out_paths(OUT_DIR, DATASET_NAME)

    done = {m: load_done_ids(paths[m], id_col="patient_id") for m in METHODS}
    done_all = done["QTN"].intersection(done["GAF"]).intersection(done["MTF"])

    tasks = []
    task_i = 0
    for fname in SUBJECT_FILES:
        pid = subject_id_from_filename(fname)
        if pid in done_all:
            continue
        task_i += 1
        tasks.append((fname, task_i))

    print(f"[{DATASET_NAME}] total files={len(SUBJECT_FILES)} | pending={len(tasks)} | n_jobs={N_JOBS}")

    buffer_rows = {m: [] for m in METHODS}
    buffer_skips = []

    iterator = range(0, len(tasks), BATCH_SIZE)
    if TQDM_ENABLED:
        iterator = tqdm(iterator, desc=f"{DATASET_NAME} batches", unit="batch", leave=TQDM_LEAVE)

    for start in iterator:
        chunk = tasks[start:start + BATCH_SIZE]
        results = Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=0)(
            delayed(compute_task_subject)(DATASET_NAME, fname, task_i)
            for (fname, task_i) in chunk
        )

        for patient_id, rows, skip in results:
            if rows is not None:
                for method in METHODS:
                    buffer_rows[method].append(rows[method])
            if skip is not None:
                buffer_skips.append(skip)

        for method in METHODS:
            if buffer_rows[method]:
                append_rows(paths[method], buffer_rows[method], id_col="patient_id")
                buffer_rows[method].clear()

        if buffer_skips:
            append_skips(paths["SKIP"], buffer_skips)
            buffer_skips.clear()

    print(f"[{DATASET_NAME}] done. CSVs in {OUT_DIR}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    run_dataset()

# %% ---- next notebook cell ----

#!/usr/bin/env python3
# respiratory_aeration_stream_to_csv_qtn_gaf_mtf_v2.py
#
# Respiratory Aeration (PhysioNet) -> stream-to-CSV (QTN/GAF/MTF small-world)
# -------------------------------------------------------------------------
# What this version fixes / adds:
# - Uses the REAL processed filenames:
#     ProcessedData_Subject01_PEEP.csv
#     ProcessedData_Subject01_PEEP_BH.csv
#     ProcessedData_Subject01_FEM.csv
#   for subjects 01..20  => up to 60 files total.
# - Temporary download of ONE CSV at a time (keeps disk low).
# - Strong-but-fast preprocessing with QC.
# - Incremental append to output CSVs.
# - Deletes raw downloaded CSV after processing (optional).
# - Produces TWO levels of outputs:
#     (A) per-file rows  (subject-trial)  -> 3 metrics CSVs + skipped CSV
#     (B) per-subject aggregated rows     -> 3 metrics CSVs
#
# Outputs (in OUT_DIR):
#   metrics_QTN_resp_aeration_stream.csv
#   metrics_GAF_resp_aeration_stream.csv
#   metrics_MTF_resp_aeration_stream.csv
#   skipped_resp_aeration_stream.csv
#
#   metrics_QTN_resp_aeration_stream_subjectAVG.csv
#   metrics_GAF_resp_aeration_stream_subjectAVG.csv
#   metrics_MTF_resp_aeration_stream_subjectAVG.csv
#
# Requirements:
#   pip install requests pandas numpy networkx scipy joblib tqdm

import os
import re
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin

import requests
import numpy as np
import pandas as pd
import networkx as nx

from scipy import signal
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIG
# ============================================================
BASE_URL = "https://physionet.org/files/respiratory-heartrate-dataset/1.0.0/Processed_Dataset/"

DATASET_NAME = "resp_aeration_stream"
OUT_DIR = "resp_outputs"
TMP_DIR = "tmp_resp_aeration"

# Real processed files are SubjectXX_{PEEP,PEEP_BH,FEM}.csv for XX=01..20
SUBJECTS = list(range(1, 21))
TRIALS = ["PEEP", "PEEP_BH", "FEM"]
SUBJECT_FILES = [
    f"ProcessedData_Subject{s:02d}_{trial}.csv"
    for s in SUBJECTS
    for trial in TRIALS
]

DELETE_RAW_AFTER_PROCESS = True
REQUEST_TIMEOUT = 120

# Columns to analyze (must exist in CSV)
SIGNAL_COLS = [
    "PSD Flow [L/s]",
    "EIT Global Aeration",
]

# Sampling frequency is not clearly specified for processed columns.
# Leave None => filtering will be skipped safely (QC still runs).
DEFAULT_FS = None  # if you later learn fs, set e.g. 10.0 or 25.0

# QTN / graph knobs
K_VALUES = [1, 2, 3]
TARGET_DENSITY = 0.10
N_RANDOMIZATIONS = 10
REWIRINGS_PER_EDGE = 5
RNG_SEED = 1234

MAX_SAMPLES = None
START_SAMPLE = 0

# Parallelization
N_JOBS = 2
BATCH_SIZE = 4
BACKEND = "loky"

METHODS = ["QTN", "GAF", "MTF"]

TQDM_ENABLED = True
TQDM_LEAVE = True

# -----------------------
# Preprocessing config
# -----------------------
PREPROCESS_MODE = "strong_fast"

# Detrend
DO_DETREND = True

# If fs known, bandpass for respiratory-like signals
RESP_LOWPASS_HZ = 1.0
RESP_HIGHPASS_HZ = 0.01
BUTTER_ORDER = 4

# QC thresholds
MIN_VALID_SAMPLES = 200
MAX_ABS_Z = 10.0
MAX_FLAT_FRAC = 0.20
MIN_STD = 1e-8
MIN_UNIQUE_VALUES = 20

# Epoch QC (only if fs known)
DO_EPOCH_QC = True
EPOCH_SEC = 30.0
MIN_KEEP_EPOCHS = 3
PTP_THRESHOLD_MULT = 6.0

# Missing data handling
MAX_INTERP_GAP_FRAC = 0.01  # reject long missing gaps


# ============================================================
# Helpers
# ============================================================
def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)


def slice_signals(sig: np.ndarray) -> np.ndarray:
    start = int(START_SAMPLE or 0)
    if MAX_SAMPLES is None:
        return sig[start:, :]
    end = start + int(MAX_SAMPLES)
    return sig[start:end, :]


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


def robust_zscore(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        std = np.nanstd(x)
        scale = std if np.isfinite(std) and std > 0 else 1.0
    z = (x - med) / scale
    return np.clip(z, -MAX_ABS_Z, MAX_ABS_Z)


def subject_id_from_filename(fname: str) -> str:
    m = re.search(r"Subject(\d+)", fname, flags=re.IGNORECASE)
    if m:
        return f"S{int(m.group(1)):02d}"
    return Path(fname).stem


def subject_trial_id_from_filename(fname: str) -> str:
    m = re.search(r"Subject(\d+)_([A-Za-z0-9_]+)\.csv", fname, flags=re.IGNORECASE)
    if m:
        return f"S{int(m.group(1)):02d}_{m.group(2)}"
    return Path(fname).stem


def trial_from_filename(fname: str) -> str:
    m = re.search(r"Subject\d+_([A-Za-z0-9_]+)\.csv", fname, flags=re.IGNORECASE)
    return m.group(1) if m else "UNKNOWN"


def out_paths(out_dir: str, dataset_name: str):
    return {
        "QTN": os.path.join(out_dir, f"metrics_QTN_{dataset_name}.csv"),
        "GAF": os.path.join(out_dir, f"metrics_GAF_{dataset_name}.csv"),
        "MTF": os.path.join(out_dir, f"metrics_MTF_{dataset_name}.csv"),
        "SKIP": os.path.join(out_dir, f"skipped_{dataset_name}.csv"),

        # subject-aggregated outputs
        "QTN_SUBJ": os.path.join(out_dir, f"metrics_QTN_{dataset_name}_subjectAVG.csv"),
        "GAF_SUBJ": os.path.join(out_dir, f"metrics_GAF_{dataset_name}_subjectAVG.csv"),
        "MTF_SUBJ": os.path.join(out_dir, f"metrics_MTF_{dataset_name}_subjectAVG.csv"),
    }


def load_done_ids(csv_path: str, id_col: str = "subject_trial_id") -> set:
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
        if id_col in df.columns:
            return set(df[id_col].astype(str).tolist())
    except Exception:
        return set()
    return set()


def append_rows(csv_path: str, rows: List[dict], required_col: str):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if required_col not in df.columns:
        raise ValueError(f"append_rows: missing '{required_col}' for {csv_path}")
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=header)


def append_skips(csv_path: str, rows: List[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=header)


# ============================================================
# Download / cleanup
# ============================================================
def download_file(url: str, local_path: str):
    if os.path.exists(local_path):
        return
    ensure_out_dir(str(Path(local_path).parent))
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def cleanup_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ============================================================
# Preprocessing (strong_fast + QC)
# ============================================================
def interpolate_short_gaps(x: np.ndarray, max_gap_frac: float = 0.01) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    finite = np.isfinite(x)
    n = len(x)

    if finite.sum() == 0:
        return x
    if finite.all():
        return x

    idx = np.arange(n)
    x_interp = x.copy()
    x_interp[~finite] = np.interp(idx[~finite], idx[finite], x[finite])

    gap_mask = ~finite
    if gap_mask.any():
        starts = np.where(np.diff(np.r_[False, gap_mask, False].astype(int)) == 1)[0]
        ends = np.where(np.diff(np.r_[False, gap_mask, False].astype(int)) == -1)[0]
        max_gap = max((e - s) for s, e in zip(starts, ends)) if len(starts) else 0
        if max_gap > max_gap_frac * n:
            raise ValueError(f"Gap too long for safe interpolation: {max_gap} samples.")
    return x_interp


def butter_filter(x: np.ndarray, fs: Optional[float], low: Optional[float], high: Optional[float], order: int = 4) -> np.ndarray:
    if fs is None or not np.isfinite(fs) or fs <= 0:
        return x

    nyq = 0.5 * fs
    if low is not None and high is not None:
        if high >= nyq:
            high = nyq * 0.99
        if low <= 0 or low >= high:
            return x
        sos = signal.butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    elif low is not None:
        if low <= 0 or low >= nyq:
            return x
        sos = signal.butter(order, low / nyq, btype="highpass", output="sos")
    elif high is not None:
        if high <= 0 or high >= nyq:
            return x
        sos = signal.butter(order, high / nyq, btype="lowpass", output="sos")
    else:
        return x

    return signal.sosfiltfilt(sos, x)


def flat_fraction(x: np.ndarray) -> float:
    dx = np.diff(x)
    if dx.size == 0:
        return 1.0
    return float(np.mean(np.abs(dx) < 1e-12))


def epoch_qc_keep_mask(x: np.ndarray, fs: Optional[float]) -> np.ndarray:
    if not DO_EPOCH_QC:
        return np.ones(1, dtype=bool)
    if fs is None or not np.isfinite(fs) or fs <= 0:
        return np.ones(1, dtype=bool)

    epoch_len = max(1, int(round(EPOCH_SEC * fs)))
    n_epochs = len(x) // epoch_len
    if n_epochs == 0:
        return np.zeros(0, dtype=bool)

    y = x[: n_epochs * epoch_len]
    epochs = y.reshape(n_epochs, epoch_len)
    ptp = np.ptp(epochs, axis=1)

    med = np.median(ptp)
    mad = np.median(np.abs(ptp - med))
    scale = 1.4826 * mad if mad > 0 else (np.std(ptp) if np.std(ptp) > 0 else 1.0)
    thr = med + PTP_THRESHOLD_MULT * scale
    return ptp <= thr


def preprocess_resp_signal(x: np.ndarray, fs: Optional[float]) -> Tuple[np.ndarray, dict]:
    qc = {}
    x = np.asarray(x, dtype=float)

    finite_before = np.isfinite(x)
    qc["n_total"] = int(x.size)
    qc["n_finite_before"] = int(finite_before.sum())
    qc["missing_frac_before"] = float(1.0 - finite_before.mean()) if x.size else np.nan

    if finite_before.sum() < MIN_VALID_SAMPLES:
        raise ValueError("Too few valid samples before preprocessing.")

    x = interpolate_short_gaps(x, max_gap_frac=MAX_INTERP_GAP_FRAC)

    if DO_DETREND:
        x = signal.detrend(x, type="linear")

    if PREPROCESS_MODE == "strong_fast":
        x = butter_filter(
            x,
            fs=fs,
            low=RESP_HIGHPASS_HZ,
            high=RESP_LOWPASS_HZ,
            order=BUTTER_ORDER,
        )

    if not np.all(np.isfinite(x)):
        raise ValueError("Non-finite values after filtering.")

    qc["std_before_z"] = float(np.std(x))
    qc["n_unique_before_z"] = int(np.unique(np.round(x, 10)).size)
    qc["flat_frac_before_z"] = flat_fraction(x)

    if qc["std_before_z"] <= MIN_STD:
        raise ValueError("Signal variance too small.")
    if qc["n_unique_before_z"] < MIN_UNIQUE_VALUES:
        raise ValueError("Too few unique values.")
    if qc["flat_frac_before_z"] > MAX_FLAT_FRAC:
        raise ValueError("Too much flat signal.")

    keep_mask = epoch_qc_keep_mask(x, fs=fs)
    if keep_mask.size > 0 and fs is not None and np.isfinite(fs) and fs > 0:
        epoch_len = max(1, int(round(EPOCH_SEC * fs)))
        n_epochs = len(x) // epoch_len
        x = x[: n_epochs * epoch_len].reshape(n_epochs, epoch_len)[keep_mask].reshape(-1)
        qc["n_epochs_total"] = int(n_epochs)
        qc["n_epochs_kept"] = int(keep_mask.sum())
        qc["epochs_kept_frac"] = float(keep_mask.mean()) if keep_mask.size else np.nan
        if keep_mask.sum() < MIN_KEEP_EPOCHS:
            raise ValueError(f"Too few clean epochs kept ({keep_mask.sum()}).")
    else:
        qc["n_epochs_total"] = np.nan
        qc["n_epochs_kept"] = np.nan
        qc["epochs_kept_frac"] = np.nan

    x = robust_zscore(x)
    qc["n_final"] = int(x.size)
    qc["std_final"] = float(np.std(x))
    qc["min_final"] = float(np.min(x))
    qc["max_final"] = float(np.max(x))

    if x.size < MIN_VALID_SAMPLES:
        raise ValueError("Too few valid samples after preprocessing.")

    return x, qc


def load_resp_csv_matrix(filepath: str, signal_cols: List[str], default_fs: Optional[float]):
    df = pd.read_csv(filepath)

    cols = [c for c in signal_cols if c in df.columns]
    if not cols:
        raise ValueError(
            f"None of requested columns found in {filepath}. Available columns: {list(df.columns)}"
        )

    kept = []
    good_cols = []
    qc_rows = []

    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").values.astype(float)
        x_clean, qc = preprocess_resp_signal(x, fs=default_fs)
        kept.append(x_clean)
        good_cols.append(c)
        qc["signal_name"] = c
        qc_rows.append(qc)

    min_len = min(len(x) for x in kept)
    arr = np.column_stack([x[:min_len] for x in kept])
    qc_df = pd.DataFrame(qc_rows)
    return arr, good_cols, default_fs, qc_df


# ============================================================
# QTN / GAF / MTF
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
# Graph / small-world metrics
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
            return (np.nan, np.nan)
        return float(np.mean(arr)), float(np.std(arr) if len(arr) > 1 else 0.0)

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
    C_obs = nx.transitivity(nx.from_numpy_array(B)) if n > 1 else np.nan
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
        "global_efficiency": float(nx.global_efficiency(nx.from_numpy_array(B))),
    }


# ============================================================
# Core: preprocess once, compute all methods (QTN/GAF/MTF)
# ============================================================
def file_to_all_method_rows(filepath: str, seed_base: int):
    sig, col_names, fs, qc_df = load_resp_csv_matrix(filepath, signal_cols=SIGNAL_COLS, default_fs=DEFAULT_FS)
    sig = slice_signals(sig)

    T = sig.shape[0]
    if T < 50:
        raise ValueError("Too few samples after preprocessing.")
    Q = compute_Q_from_T(T)
    if Q < 3:
        raise ValueError(f"Q too small: {Q}")

    per_method_metrics = {m: [] for m in METHODS}
    qc_summary = qc_df.mean(numeric_only=True).to_dict()

    for li in range(sig.shape[1]):
        x = sig[:, li].astype(float)
        x = x[np.isfinite(x)]
        if x.size < 50:
            continue

        # QTN
        A_qtn = calculate_quantile_graph_varying_k(x, Q=Q, k_values=K_VALUES)
        W_qtn = (A_qtn + A_qtn.T).astype(float)
        per_method_metrics["QTN"].append(
            compute_smallworld_metrics_from_W(W_qtn, seed=seed_base + 100 * li + 1, use_abs_for_threshold=False)
        )

        # GAF / MTF use x downsampled to Q
        xQ = downsample_to_length(x, Q)

        W_gaf = calculate_gaf_from_lengthQ(xQ)
        per_method_metrics["GAF"].append(
            compute_smallworld_metrics_from_W(W_gaf, seed=seed_base + 100 * li + 2, use_abs_for_threshold=True)
        )

        W_mtf = calculate_mtf_from_lengthQ(xQ, Q=Q)
        per_method_metrics["MTF"].append(
            compute_smallworld_metrics_from_W(W_mtf, seed=seed_base + 100 * li + 3, use_abs_for_threshold=False)
        )

    out = {}
    for method in METHODS:
        if not per_method_metrics[method]:
            raise ValueError(f"No valid signals produced metrics for {method}.")
        dfm = pd.DataFrame(per_method_metrics[method])
        avg = dfm.mean(numeric_only=True).to_dict()

        # attach run metadata + QC summary
        avg.update({
            "fs_hz": fs if fs is not None else np.nan,
            "n_signals": int(sig.shape[1]),
            "T_used_samples": int(T),
            "Q_used": int(Q),
            "signal_cols": ";".join(col_names),

            "qc_missing_frac_before_mean": qc_summary.get("missing_frac_before", np.nan),
            "qc_std_before_z_mean": qc_summary.get("std_before_z", np.nan),
            "qc_flat_frac_before_z_mean": qc_summary.get("flat_frac_before_z", np.nan),
            "qc_epochs_kept_frac_mean": qc_summary.get("epochs_kept_frac", np.nan),
            "qc_std_final_mean": qc_summary.get("std_final", np.nan),
        })
        out[method] = avg

    return out


def compute_task_file(filename: str, task_i: int):
    patient_id = subject_id_from_filename(filename)
    subject_trial_id = subject_trial_id_from_filename(filename)
    trial = trial_from_filename(filename)

    local_path = os.path.join(TMP_DIR, filename)
    url = urljoin(BASE_URL, filename)

    try:
        download_file(url, local_path)
        rows = file_to_all_method_rows(local_path, seed_base=RNG_SEED + 100000 * task_i)

        for method in METHODS:
            rows[method].update({
                "patient_id": patient_id,                 # subject only: S01..S20
                "subject_trial_id": subject_trial_id,     # S01_PEEP etc
                "trial": trial,                           # PEEP / PEEP_BH / FEM
                "dataset": DATASET_NAME,
                "source_file": filename,
            })

        return subject_trial_id, rows, None

    except Exception as e:
        skip = {
            "dataset": DATASET_NAME,
            "method": "ALL",
            "record_or_patient": filename,
            "reason": str(e),
        }
        return subject_trial_id, None, skip

    finally:
        if DELETE_RAW_AFTER_PROCESS:
            cleanup_file(local_path)


# ============================================================
# Subject aggregation (optional but useful)
# ============================================================
def rebuild_subject_level_csvs(paths: dict):
    """
    Reads the per-file CSVs (subject-trial rows) and writes per-subject averaged CSVs.
    Only numeric columns are averaged; non-numeric metadata is kept minimally.
    """
    for method, in_path, out_path in [
        ("QTN", paths["QTN"], paths["QTN_SUBJ"]),
        ("GAF", paths["GAF"], paths["GAF_SUBJ"]),
        ("MTF", paths["MTF"], paths["MTF_SUBJ"]),
    ]:
        if not os.path.exists(in_path):
            continue

        df = pd.read_csv(in_path)

        if "patient_id" not in df.columns:
            continue

        # keep some metadata if present
        meta_cols = [c for c in ["dataset", "signal_cols", "fs_hz"] if c in df.columns]

        # average numeric columns per subject
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        g = df.groupby("patient_id")[num].mean().reset_index()

        # attach basic metadata (first occurrence)
        for c in meta_cols:
            g[c] = df.groupby("patient_id")[c].first().values

        # store number of trials actually used
        g["n_trials_used"] = df.groupby("patient_id")["subject_trial_id"].nunique().values if "subject_trial_id" in df.columns else df.groupby("patient_id").size().values
        g["aggregation"] = "mean_over_trials"

        g.to_csv(out_path, index=False)
        print(f"[OK] wrote subject-avg: {out_path} ({g.shape[0]} subjects) for {method}")


# ============================================================
# Runner
# ============================================================
def run_dataset():
    ensure_out_dir(OUT_DIR)
    ensure_out_dir(TMP_DIR)
    paths = out_paths(OUT_DIR, DATASET_NAME)

    done = {m: load_done_ids(paths[m], id_col="subject_trial_id") for m in METHODS}
    done_all = done["QTN"].intersection(done["GAF"]).intersection(done["MTF"])

    tasks = []
    task_i = 0
    for fname in SUBJECT_FILES:
        stid = subject_trial_id_from_filename(fname)
        if stid in done_all:
            continue
        task_i += 1
        tasks.append((fname, task_i))

    print(f"[{DATASET_NAME}] expected files={len(SUBJECT_FILES)} (<=60) | pending={len(tasks)} | n_jobs={N_JOBS}")

    buffer_rows = {m: [] for m in METHODS}
    buffer_skips = []

    iterator = range(0, len(tasks), BATCH_SIZE)
    if TQDM_ENABLED:
        iterator = tqdm(iterator, desc=f"{DATASET_NAME} batches", unit="batch", leave=TQDM_LEAVE)

    for start in iterator:
        chunk = tasks[start:start + BATCH_SIZE]
        results = Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=0)(
            delayed(compute_task_file)(fname, task_i)
            for (fname, task_i) in chunk
        )

        for subject_trial_id, rows, skip in results:
            if rows is not None:
                for method in METHODS:
                    buffer_rows[method].append(rows[method])
            if skip is not None:
                buffer_skips.append(skip)

        for method in METHODS:
            if buffer_rows[method]:
                append_rows(paths[method], buffer_rows[method], required_col="subject_trial_id")
                buffer_rows[method].clear()

        if buffer_skips:
            append_skips(paths["SKIP"], buffer_skips)
            buffer_skips.clear()

    print(f"[{DATASET_NAME}] per-file CSVs done. Now building subject-level averages...")
    rebuild_subject_level_csvs(paths)
    print(f"[{DATASET_NAME}] done. Outputs in {OUT_DIR}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    run_dataset()

# %% ---- next notebook cell ----

#!/usr/bin/env python3
# respiratory_aeration_stream_to_csv_qtn_gaf_mtf_FIXED.py
#
# Respiratory + heart rate monitoring dataset (aeration study, PhysioNet)
# -> stream-to-CSV (QTN/GAF/MTF small-world)
#
# FIXES vs your mixed runs:
# - Always writes into a dataset-specific folder:
#       resp_outputs/out_resp_aeration_stream/
# - Produces ONE consistent set of outputs:
#       * per-file (subject-trial) CSVs  (3)
#       * per-subject averaged CSVs      (3)
#       * skipped CSV
#       * manifest CSV (what each file is)
# - Automatically discovers real filenames from SHA256SUMS.txt
# - Downloads 1 raw CSV at a time, processes, then deletes it (optional)
# - Preprocessing is “strong-but-fast”; note:
#       If DEFAULT_FS is None, frequency filtering + epoch-QC are skipped (by design).
#
# Requirements:
#   pip install requests pandas numpy networkx scipy joblib tqdm

import os
import re
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from urllib.parse import urljoin

import requests
import numpy as np
import pandas as pd
import networkx as nx
from scipy import signal
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIG
# ============================================================
BASE_URL = "https://physionet.org/files/respiratory-heartrate-dataset/1.0.0/"
PROCESSED_BASE_URL = urljoin(BASE_URL, "Processed_Dataset/")
SHA256_URL = urljoin(BASE_URL, "SHA256SUMS.txt")

DATASET_NAME = "resp_aeration_stream"

# IMPORTANT: dataset-specific output folder (prevents mixing with other datasets)
BASE_OUT_DIR = "resp_outputs"
OUT_DIR = os.path.join(BASE_OUT_DIR, "out_resp_aeration_stream")
TMP_DIR = "tmp_resp_aeration"

DELETE_RAW_AFTER_PROCESS = True
REQUEST_TIMEOUT = 120

# Columns to analyze (must exist in processed CSVs)
SIGNAL_COLS = [
    "PSD Flow [L/s]",
    "EIT Global Aeration",
]

# If you know the sampling rate for these processed columns, set it here.
# If None, frequency filtering + epoch-QC will be skipped safely.
DEFAULT_FS: Optional[float] = None

# QTN / graph knobs
K_VALUES = [1, 2, 3]
TARGET_DENSITY = 0.10
N_RANDOMIZATIONS = 10
REWIRINGS_PER_EDGE = 5
RNG_SEED = 1234

MAX_SAMPLES = None
START_SAMPLE = 0

# Parallelization
N_JOBS = 2
BATCH_SIZE = 4
BACKEND = "loky"

METHODS = ["QTN", "GAF", "MTF"]

TQDM_ENABLED = True
TQDM_LEAVE = True

# -----------------------
# Preprocessing config (strong-but-fast)
# -----------------------
PREPROCESS_MODE = "strong_fast"

DO_DETREND = True

# Respiratory-like band (only applied if fs is known)
RESP_LOWPASS_HZ = 1.0
RESP_HIGHPASS_HZ = 0.01
BUTTER_ORDER = 4

# QC thresholds
MIN_VALID_SAMPLES = 200
MAX_ABS_Z = 10.0
MAX_FLAT_FRAC = 0.20
MIN_STD = 1e-8
MIN_UNIQUE_VALUES = 20

# Epoch QC (only if fs known)
DO_EPOCH_QC = True
EPOCH_SEC = 30.0
MIN_KEEP_EPOCHS = 3
PTP_THRESHOLD_MULT = 6.0

# Missing data handling
MAX_INTERP_GAP_FRAC = 0.01  # reject long missing gaps

# ============================================================
# Helpers
# ============================================================
def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)

def slice_signals(sig: np.ndarray) -> np.ndarray:
    start = int(START_SAMPLE or 0)
    if MAX_SAMPLES is None:
        return sig[start:, :]
    end = start + int(MAX_SAMPLES)
    return sig[start:end, :]

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

def robust_zscore(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        std = np.nanstd(x)
        scale = std if np.isfinite(std) and std > 0 else 1.0
    z = (x - med) / scale
    return np.clip(z, -MAX_ABS_Z, MAX_ABS_Z)

def subject_id_from_filename(fname: str) -> str:
    m = re.search(r"Subject(\d+)", fname, flags=re.IGNORECASE)
    if m:
        return f"S{int(m.group(1)):02d}"
    return Path(fname).stem

def trial_from_filename(fname: str) -> str:
    # ProcessedData_Subject01_PEEP_BH.csv -> PEEP_BH
    m = re.search(r"ProcessedData_Subject\d+_(.+)\.csv$", fname, flags=re.IGNORECASE)
    return m.group(1) if m else "UNKNOWN"

def file_id_from_filename(fname: str) -> str:
    return Path(fname).stem  # unique per processed file

def out_paths(out_dir: str, dataset_name: str) -> Dict[str, str]:
    return {
        "QTN_FILE": os.path.join(out_dir, f"metrics_QTN_{dataset_name}_per_file.csv"),
        "GAF_FILE": os.path.join(out_dir, f"metrics_GAF_{dataset_name}_per_file.csv"),
        "MTF_FILE": os.path.join(out_dir, f"metrics_MTF_{dataset_name}_per_file.csv"),
        "QTN_SUBJ": os.path.join(out_dir, f"metrics_QTN_{dataset_name}_per_subject.csv"),
        "GAF_SUBJ": os.path.join(out_dir, f"metrics_GAF_{dataset_name}_per_subject.csv"),
        "MTF_SUBJ": os.path.join(out_dir, f"metrics_MTF_{dataset_name}_per_subject.csv"),
        "SKIP": os.path.join(out_dir, f"skipped_{dataset_name}.csv"),
        "MANIFEST": os.path.join(out_dir, f"manifest_{dataset_name}.csv"),
    }

def load_done_ids(csv_path: str, id_col: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
        if id_col in df.columns:
            return set(df[id_col].astype(str).tolist())
    except Exception:
        return set()
    return set()

def append_rows(csv_path: str, rows: List[dict], required_col: str):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if required_col not in df.columns:
        raise ValueError(f"append_rows: missing '{required_col}' for {csv_path}")
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=header)

def append_skips(csv_path: str, rows: List[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=header)

# ============================================================
# Discover files from PhysioNet (robust)
# ============================================================
def discover_processed_files_from_sha256() -> List[str]:
    r = requests.get(SHA256_URL, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    text = r.text.splitlines()

    files = []
    pat = re.compile(r"Processed_Dataset/(ProcessedData_Subject\d+_[A-Za-z0-9_]+\.csv)$")
    for line in text:
        m = pat.search(line)
        if m:
            files.append(m.group(1))

    files = sorted(set(files))
    if not files:
        raise RuntimeError("No processed respiratory CSV files discovered from SHA256SUMS.txt")
    return files

# ============================================================
# Download / cleanup
# ============================================================
def download_file(url: str, local_path: str):
    if os.path.exists(local_path):
        return
    ensure_out_dir(str(Path(local_path).parent))
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def cleanup_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

# ============================================================
# Preprocessing (strong_fast + QC)
# ============================================================
def interpolate_short_gaps(x: np.ndarray, max_gap_frac: float = 0.01) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    finite = np.isfinite(x)
    n = len(x)

    if finite.sum() == 0:
        return x
    if finite.all():
        return x

    idx = np.arange(n)
    x_interp = x.copy()
    x_interp[~finite] = np.interp(idx[~finite], idx[finite], x[finite])

    gap_mask = ~finite
    if gap_mask.any():
        starts = np.where(np.diff(np.r_[False, gap_mask, False].astype(int)) == 1)[0]
        ends = np.where(np.diff(np.r_[False, gap_mask, False].astype(int)) == -1)[0]
        max_gap = max((e - s) for s, e in zip(starts, ends)) if len(starts) else 0
        if max_gap > max_gap_frac * n:
            raise ValueError(f"Gap too long for safe interpolation: {max_gap} samples.")
    return x_interp

def butter_filter(
    x: np.ndarray,
    fs: Optional[float],
    low: Optional[float],
    high: Optional[float],
    order: int = 4
) -> np.ndarray:
    if fs is None or not np.isfinite(fs) or fs <= 0:
        return x

    nyq = 0.5 * fs
    if low is not None and high is not None:
        if high >= nyq:
            high = nyq * 0.99
        if low <= 0 or low >= high:
            return x
        sos = signal.butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    elif low is not None:
        if low <= 0 or low >= nyq:
            return x
        sos = signal.butter(order, low / nyq, btype="highpass", output="sos")
    elif high is not None:
        if high <= 0 or high >= nyq:
            return x
        sos = signal.butter(order, high / nyq, btype="lowpass", output="sos")
    else:
        return x

    return signal.sosfiltfilt(sos, x)

def flat_fraction(x: np.ndarray) -> float:
    dx = np.diff(x)
    if dx.size == 0:
        return 1.0
    return float(np.mean(np.abs(dx) < 1e-12))

def epoch_qc_keep_mask(x: np.ndarray, fs: Optional[float]) -> np.ndarray:
    if not DO_EPOCH_QC:
        return np.ones(1, dtype=bool)
    if fs is None or not np.isfinite(fs) or fs <= 0:
        # no fs -> cannot epoch meaningfully -> keep all
        return np.ones(1, dtype=bool)

    epoch_len = max(1, int(round(EPOCH_SEC * fs)))
    n_epochs = len(x) // epoch_len
    if n_epochs == 0:
        return np.zeros(0, dtype=bool)

    y = x[: n_epochs * epoch_len]
    epochs = y.reshape(n_epochs, epoch_len)
    ptp = np.ptp(epochs, axis=1)

    med = np.median(ptp)
    mad = np.median(np.abs(ptp - med))
    scale = 1.4826 * mad if mad > 0 else (np.std(ptp) if np.std(ptp) > 0 else 1.0)
    thr = med + PTP_THRESHOLD_MULT * scale
    return ptp <= thr

def preprocess_resp_signal(x: np.ndarray, fs: Optional[float]) -> Tuple[np.ndarray, dict]:
    qc = {}
    x = np.asarray(x, dtype=float)

    finite_before = np.isfinite(x)
    qc["n_total"] = int(x.size)
    qc["n_finite_before"] = int(finite_before.sum())
    qc["missing_frac_before"] = float(1.0 - finite_before.mean()) if x.size else np.nan

    if finite_before.sum() < MIN_VALID_SAMPLES:
        raise ValueError("Too few valid samples before preprocessing.")

    x = interpolate_short_gaps(x, max_gap_frac=MAX_INTERP_GAP_FRAC)

    if DO_DETREND:
        x = signal.detrend(x, type="linear")

    # NOTE: if fs is None, butter_filter returns x unchanged
    if PREPROCESS_MODE == "strong_fast":
        x = butter_filter(x, fs=fs, low=RESP_HIGHPASS_HZ, high=RESP_LOWPASS_HZ, order=BUTTER_ORDER)

    if not np.all(np.isfinite(x)):
        raise ValueError("Non-finite values after filtering.")

    qc["std_before_z"] = float(np.std(x))
    qc["n_unique_before_z"] = int(np.unique(np.round(x, 10)).size)
    qc["flat_frac_before_z"] = flat_fraction(x)

    if qc["std_before_z"] <= MIN_STD:
        raise ValueError("Signal variance too small.")
    if qc["n_unique_before_z"] < MIN_UNIQUE_VALUES:
        raise ValueError("Too few unique values.")
    if qc["flat_frac_before_z"] > MAX_FLAT_FRAC:
        raise ValueError("Too much flat signal.")

    keep_mask = epoch_qc_keep_mask(x, fs=fs)
    if keep_mask.size > 0 and fs is not None and np.isfinite(fs) and fs > 0:
        epoch_len = max(1, int(round(EPOCH_SEC * fs)))
        n_epochs = len(x) // epoch_len
        x = x[: n_epochs * epoch_len].reshape(n_epochs, epoch_len)[keep_mask].reshape(-1)
        qc["n_epochs_total"] = int(n_epochs)
        qc["n_epochs_kept"] = int(keep_mask.sum())
        qc["epochs_kept_frac"] = float(keep_mask.mean()) if keep_mask.size else np.nan
        if keep_mask.sum() < MIN_KEEP_EPOCHS:
            raise ValueError(f"Too few clean epochs kept ({keep_mask.sum()}).")
    else:
        qc["n_epochs_total"] = np.nan
        qc["n_epochs_kept"] = np.nan
        qc["epochs_kept_frac"] = np.nan

    x = robust_zscore(x)

    qc["n_final"] = int(x.size)
    qc["std_final"] = float(np.std(x))
    qc["min_final"] = float(np.min(x))
    qc["max_final"] = float(np.max(x))

    if x.size < MIN_VALID_SAMPLES:
        raise ValueError("Too few valid samples after preprocessing.")

    return x, qc

def load_resp_csv_matrix(filepath: str, signal_cols: List[str], default_fs: Optional[float]):
    df = pd.read_csv(filepath)

    cols = [c for c in signal_cols if c in df.columns]
    if not cols:
        raise ValueError(f"None of requested columns found in {filepath}. Available: {list(df.columns)}")

    kept = []
    good_cols = []
    qc_rows = []

    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").values.astype(float)
        x_clean, qc = preprocess_resp_signal(x, fs=default_fs)
        kept.append(x_clean)
        good_cols.append(c)
        qc["signal_name"] = c
        qc_rows.append(qc)

    min_len = min(len(x) for x in kept)
    arr = np.column_stack([x[:min_len] for x in kept])
    qc_df = pd.DataFrame(qc_rows)
    return arr, good_cols, default_fs, qc_df

# ============================================================
# QTN / GAF / MTF
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
# Graph / small-world
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
            return (np.nan, np.nan)
        return float(np.mean(arr)), float(np.std(arr) if len(arr) > 1 else 0.0)

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
    C_obs = nx.transitivity(nx.from_numpy_array(B)) if n > 1 else np.nan
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
        "global_efficiency": float(nx.global_efficiency(nx.from_numpy_array(B))),
    }

# ============================================================
# Core (per-file)
# ============================================================
def file_to_all_method_rows(filepath: str, seed_base: int) -> Dict[str, dict]:
    sig, col_names, fs, qc_df = load_resp_csv_matrix(filepath, signal_cols=SIGNAL_COLS, default_fs=DEFAULT_FS)
    sig = slice_signals(sig)

    T = sig.shape[0]
    if T < 50:
        raise ValueError("Too few samples after preprocessing.")
    Q = compute_Q_from_T(T)
    if Q < 3:
        raise ValueError(f"Q too small: {Q}")

    per_method_metrics = {m: [] for m in METHODS}
    qc_summary = qc_df.mean(numeric_only=True).to_dict()

    for li in range(sig.shape[1]):
        x = sig[:, li].astype(float)
        x = x[np.isfinite(x)]
        if x.size < 50:
            continue

        # QTN
        A_qtn = calculate_quantile_graph_varying_k(x, Q=Q, k_values=K_VALUES)
        W_qtn = (A_qtn + A_qtn.T).astype(float)
        per_method_metrics["QTN"].append(
            compute_smallworld_metrics_from_W(W_qtn, seed=seed_base + 100 * li + 1, use_abs_for_threshold=False)
        )

        # GAF / MTF
        xQ = downsample_to_length(x, Q)

        W_gaf = calculate_gaf_from_lengthQ(xQ)
        per_method_metrics["GAF"].append(
            compute_smallworld_metrics_from_W(W_gaf, seed=seed_base + 100 * li + 2, use_abs_for_threshold=True)
        )

        W_mtf = calculate_mtf_from_lengthQ(xQ, Q=Q)
        per_method_metrics["MTF"].append(
            compute_smallworld_metrics_from_W(W_mtf, seed=seed_base + 100 * li + 3, use_abs_for_threshold=False)
        )

    out = {}
    for method in METHODS:
        if not per_method_metrics[method]:
            raise ValueError(f"No valid signals produced metrics for {method}.")
        dfm = pd.DataFrame(per_method_metrics[method])
        avg = dfm.mean(numeric_only=True).to_dict()

        avg.update({
            "dataset": DATASET_NAME,
            "fs_hz_used": float(fs) if fs is not None else np.nan,
            "fs_hz_note": "filtering/epochQC applied only if fs_hz_used is finite" if fs is None else "filtering/epochQC applied",
            "n_signals": int(sig.shape[1]),
            "T_used_samples": int(T),
            "Q_used": int(Q),
            "signal_cols": ";".join(col_names),

            # QC summary
            "qc_missing_frac_before_mean": qc_summary.get("missing_frac_before", np.nan),
            "qc_std_before_z_mean": qc_summary.get("std_before_z", np.nan),
            "qc_flat_frac_before_z_mean": qc_summary.get("flat_frac_before_z", np.nan),
            "qc_epochs_kept_frac_mean": qc_summary.get("epochs_kept_frac", np.nan),
            "qc_std_final_mean": qc_summary.get("std_final", np.nan),
        })
        out[method] = avg

    return out

def compute_task_file(filename: str, task_i: int):
    pid = subject_id_from_filename(filename)
    trial = trial_from_filename(filename)
    fid = file_id_from_filename(filename)

    local_path = os.path.join(TMP_DIR, filename)
    url = urljoin(PROCESSED_BASE_URL, filename)

    try:
        download_file(url, local_path)
        rows = file_to_all_method_rows(local_path, seed_base=RNG_SEED + 100000 * task_i)

        for method in METHODS:
            rows[method].update({
                "file_id": fid,
                "patient_id": pid,
                "trial": trial,
                "source_file": filename,
            })

        return fid, rows, None

    except Exception as e:
        skip = {
            "dataset": DATASET_NAME,
            "method": "ALL",
            "record_or_patient": filename,
            "reason": str(e),
        }
        return fid, None, skip

    finally:
        if DELETE_RAW_AFTER_PROCESS:
            cleanup_file(local_path)

# ============================================================
# Aggregate per-file -> per-subject
# ============================================================
def aggregate_per_subject(file_csv: str, out_csv: str):
    if not os.path.exists(file_csv):
        return

    df = pd.read_csv(file_csv)
    if df.empty or "patient_id" not in df.columns:
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    g = df.groupby("patient_id")[numeric_cols].mean().reset_index()

    # keep minimal metadata (first occurrence)
    for c in ["dataset", "signal_cols", "fs_hz_used"]:
        if c in df.columns:
            g[c] = df.groupby("patient_id")[c].first().values

    # how many trials contributed
    if "trial" in df.columns:
        g["n_trials_aggregated"] = df.groupby("patient_id")["trial"].nunique().values
    g["aggregation"] = "mean_over_trials"

    g.to_csv(out_csv, index=False)

# ============================================================
# Runner
# ============================================================
def run_dataset():
    ensure_out_dir(OUT_DIR)
    ensure_out_dir(TMP_DIR)
    paths = out_paths(OUT_DIR, DATASET_NAME)

    # discover real processed filenames
    files = discover_processed_files_from_sha256()

    # write manifest (always)
    manifest_rows = []
    for fname in files:
        manifest_rows.append({
            "file_name": fname,
            "file_id": file_id_from_filename(fname),
            "patient_id": subject_id_from_filename(fname),
            "trial": trial_from_filename(fname),
        })
    pd.DataFrame(manifest_rows).to_csv(paths["MANIFEST"], index=False)

    # resume logic: use file_id for per-file outputs
    done_qtn = load_done_ids(paths["QTN_FILE"], id_col="file_id")
    done_gaf = load_done_ids(paths["GAF_FILE"], id_col="file_id")
    done_mtf = load_done_ids(paths["MTF_FILE"], id_col="file_id")
    done_all = done_qtn.intersection(done_gaf).intersection(done_mtf)

    tasks = []
    task_i = 0
    for fname in files:
        fid = file_id_from_filename(fname)
        if fid in done_all:
            continue
        task_i += 1
        tasks.append((fname, task_i))

    print(f"[{DATASET_NAME}] processed files discovered={len(files)} | pending={len(tasks)} | n_jobs={N_JOBS}")
    if not tasks:
        # still rebuild subject-level in case per-file changed externally
        aggregate_per_subject(paths["QTN_FILE"], paths["QTN_SUBJ"])
        aggregate_per_subject(paths["GAF_FILE"], paths["GAF_SUBJ"])
        aggregate_per_subject(paths["MTF_FILE"], paths["MTF_SUBJ"])
        print(f"[{DATASET_NAME}] nothing to do. Outputs in {OUT_DIR}")
        return

    buffer_rows = {m: [] for m in METHODS}
    buffer_skips = []

    iterator = range(0, len(tasks), BATCH_SIZE)
    if TQDM_ENABLED:
        iterator = tqdm(iterator, desc=f"{DATASET_NAME} batches", unit="batch", leave=TQDM_LEAVE)

    for start in iterator:
        chunk = tasks[start:start + BATCH_SIZE]
        results = Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=0)(
            delayed(compute_task_file)(fname, task_i)
            for (fname, task_i) in chunk
        )

        for fid, rows, skip in results:
            if rows is not None:
                buffer_rows["QTN"].append(rows["QTN"])
                buffer_rows["GAF"].append(rows["GAF"])
                buffer_rows["MTF"].append(rows["MTF"])
            if skip is not None:
                buffer_skips.append(skip)

        if buffer_rows["QTN"]:
            append_rows(paths["QTN_FILE"], buffer_rows["QTN"], required_col="file_id")
            buffer_rows["QTN"].clear()
        if buffer_rows["GAF"]:
            append_rows(paths["GAF_FILE"], buffer_rows["GAF"], required_col="file_id")
            buffer_rows["GAF"].clear()
        if buffer_rows["MTF"]:
            append_rows(paths["MTF_FILE"], buffer_rows["MTF"], required_col="file_id")
            buffer_rows["MTF"].clear()

        if buffer_skips:
            append_skips(paths["SKIP"], buffer_skips)
            buffer_skips.clear()

    # build subject-level averages
    aggregate_per_subject(paths["QTN_FILE"], paths["QTN_SUBJ"])
    aggregate_per_subject(paths["GAF_FILE"], paths["GAF_SUBJ"])
    aggregate_per_subject(paths["MTF_FILE"], paths["MTF_SUBJ"])

    print(f"[{DATASET_NAME}] done. Outputs in: {OUT_DIR}")

if __name__ == "__main__":
    run_dataset()

# %% ---- next notebook cell ----


