#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto-converted from the Jupyter notebook ``EMG.ipynb``.

Code cells are reproduced verbatim. Markdown cells are kept as comments.
Jupyter shell-magic lines (if any) are commented out so the file is valid
Python; they are preserved for reference. Cell boundaries are marked with
``# In[...]`` to match the original notebook ordering.
"""


# In[19]:

# =========================
# EMG plantar (PhysioNet plantar/1.0.0) -> QTN/GAF/MTF small-world
# Jupyter-friendly (NO argparse) -- UPDATED (AUTO-ORIENTATION FIX)
# =========================

import os, re, warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin

import requests
import numpy as np
import pandas as pd
import networkx as nx
from scipy import signal
from joblib import Parallel, delayed
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIG (DATASET)
# ============================================================
BASE_URL = "https://physionet.org/files/plantar/1.0.0/"
SHA256_URL = urljoin(BASE_URL, "SHA256SUMS.txt")

DATASET_NAME = "emg_plantar"

# Use a fresh output dir to avoid mixing with previous all-skipped run
OUT_DIR = "emg_outputs/out_emg_plantar_v2"
TMP_DIR = "tmp_emg_plantar"

DELETE_RAW_AFTER_PROCESS = True
REQUEST_TIMEOUT = 120

# ============================================================
# FILE DISCOVERY FILTER (plantar/1.0.0)
# Most files: E_001_post1.txt, P_014_pre2.txt, ...
# Exclude Summary-sheet.txt
# ============================================================
FILE_REGEX = re.compile(r"^[EP]_\d{3}_.+\.txt$", flags=re.IGNORECASE)
EXCLUDE_REGEXES = [
    re.compile(r"summary[-_ ]sheet", re.IGNORECASE),
    re.compile(r"readme", re.IGNORECASE),
    re.compile(r"license", re.IGNORECASE),
    re.compile(r"sha256sums", re.IGNORECASE),
    re.compile(r"\.pdf$", re.IGNORECASE),
]

# ============================================================
# PARAMETERS (QTN / SMALL-WORLD)
# ============================================================
K_VALUES = [1, 2, 3]
TARGET_DENSITY = 0.10
N_RANDOMIZATIONS = 10
REWIRINGS_PER_EDGE = 5
RNG_SEED = 1234

MAX_SAMPLES = None
START_SAMPLE = 0

# Parallelization (notebooks: loky usually OK; threading safest)
N_JOBS = 2
BATCH_SIZE = 4
BACKEND = "loky"   # if you see hanging, set to "threading"

METHODS = ["QTN", "GAF", "MTF"]
TQDM_ENABLED = True
TQDM_LEAVE = True

# ============================================================
# EMG PREPROCESSING (more robust + less brittle)
# ============================================================
DO_DETREND = True

# EMG filtering (only if fs known)
EMG_HIGHPASS_HZ = 20.0
EMG_LOWPASS_HZ = 450.0
BUTTER_ORDER = 4

# Notch (50 Hz EU)
DO_NOTCH = True
NOTCH_HZ = 50.0
NOTCH_Q = 30.0

# Optional envelope (off by default)
USE_ENVELOPE = False
ENVELOPE_LOWPASS_HZ = 10.0

# QC thresholds (relaxed for this dataset until structure is confirmed)
MIN_VALID_SAMPLES = 200
MIN_FINITE_FRAC = 0.80          # was 0.95; too strict for some txt quirks
MIN_STD = 1e-10                 # slightly looser
MIN_UNIQUE_VALUES = 10          # slightly looser
MAX_ABS_Z = 10.0
MAX_FLAT_FRAC = 0.50            # much looser; the previous 0.20 was killing many files

# Epoch QC (only if fs known)
DO_EPOCH_QC = True
EPOCH_SEC = 2.0
MIN_KEEP_EPOCHS = 2             # looser than 3
PTP_THRESHOLD_MULT = 10.0       # looser; EMG is bursty

# Missing data handling
MAX_INTERP_GAP_FRAC = 0.05      # allow slightly longer gaps before rejecting


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

def flat_fraction(x: np.ndarray) -> float:
    dx = np.diff(x)
    if dx.size == 0:
        return 1.0
    # tolerate quantized signals by using a small epsilon
    return float(np.mean(np.abs(dx) < 1e-12))

def out_paths(out_dir: str, dataset_name: str):
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

def append_rows(csv_path: str, rows: List[dict], id_col: str):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if id_col not in df.columns:
        raise ValueError(f"append_rows: missing '{id_col}' in {csv_path}")
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=header)

def append_skips(csv_path: str, rows: List[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=header)

def file_id_from_path(rel_path: str) -> str:
    return Path(rel_path).stem

def patient_id_from_path(rel_path: str) -> str:
    stem = Path(rel_path).stem
    m = re.match(r"^([EP]_\d{3})_", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return stem


# ============================================================
# Discover files from SHA256SUMS
# ============================================================
def discover_files_from_sha256() -> List[str]:
    r = requests.get(SHA256_URL, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    files = []
    for line in r.text.splitlines():
        chunks = line.strip().split()
        if len(chunks) < 2:
            continue
        rel_path = chunks[-1].lstrip("./")

        if rel_path.lower().endswith("sha256sums.txt"):
            continue
        if not FILE_REGEX.match(rel_path):
            continue
        if any(rx.search(rel_path) for rx in EXCLUDE_REGEXES):
            continue
        files.append(rel_path)

    files = sorted(set(files))
    if not files:
        raise RuntimeError("No files matched FILE_REGEX from SHA256SUMS.")
    return files


# ============================================================
# Download / cleanup
# ============================================================
def download_file(url: str, local_path: str):
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return
    ensure_out_dir(str(Path(local_path).parent))
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
        r.raise_for_status()
        tmp = local_path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, local_path)

def cleanup_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ============================================================
# Preprocessing (EMG) - robust + orientation-agnostic
# ============================================================
def interpolate_short_gaps(x: np.ndarray, max_gap_frac: float = 0.05) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    n = x.size
    finite = np.isfinite(x)

    if n == 0 or finite.sum() == 0:
        return x
    if finite.all():
        return x

    idx = np.arange(n)
    x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])

    gap_mask = ~finite
    if gap_mask.any():
        starts = np.where(np.diff(np.r_[False, gap_mask, False].astype(int)) == 1)[0]
        ends = np.where(np.diff(np.r_[False, gap_mask, False].astype(int)) == -1)[0]
        max_gap = max((e - s) for s, e in zip(starts, ends)) if len(starts) else 0
        if max_gap > max_gap_frac * n:
            raise ValueError(f"Gap too long for safe interpolation: {max_gap} samples.")
    return x

def butter_sos(fs: float, btype: str, cutoff):
    nyq = 0.5 * fs
    if btype in ("highpass", "lowpass"):
        w = float(cutoff) / nyq
        if not (0 < w < 1):
            return None
        return signal.butter(BUTTER_ORDER, w, btype=btype, output="sos")
    elif btype == "bandpass":
        lo, hi = cutoff
        lo = float(lo); hi = float(hi)
        if hi >= nyq:
            hi = nyq * 0.99
        if not (0 < lo < hi < nyq):
            return None
        return signal.butter(BUTTER_ORDER, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return None

def notch_filter_1d(x: np.ndarray, fs: float, f0: float, Q: float):
    if f0 <= 0 or f0 >= 0.5 * fs:
        return x
    b, a = signal.iirnotch(w0=f0, Q=Q, fs=fs)
    return signal.filtfilt(b, a, x)

def epoch_qc_keep_mask(x: np.ndarray, fs: Optional[float]) -> np.ndarray:
    if (not DO_EPOCH_QC) or fs is None or (not np.isfinite(fs)) or fs <= 0:
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

def preprocess_emg_signal(x: np.ndarray, fs: Optional[float]) -> Tuple[np.ndarray, dict]:
    qc: Dict[str, float] = {}
    x = np.asarray(x, dtype=float)

    if x.size < MIN_VALID_SAMPLES:
        raise ValueError(f"Too few samples in channel (n={x.size}).")

    finite = np.isfinite(x)
    finite_frac = float(finite.mean())
    qc["finite_frac_before"] = finite_frac
    qc["n_total"] = int(x.size)
    qc["n_finite_before"] = int(finite.sum())

    if finite_frac < MIN_FINITE_FRAC:
        raise ValueError(f"Too many NaNs: finite_frac={finite_frac:.3f}")

    x = interpolate_short_gaps(x, max_gap_frac=MAX_INTERP_GAP_FRAC)

    if DO_DETREND:
        x = signal.detrend(x, type="constant")

    # filtering only if fs known
    if fs is not None and np.isfinite(fs) and fs > 0:
        fs = float(fs)

        # high-pass
        sos = butter_sos(fs, "highpass", EMG_HIGHPASS_HZ)
        if sos is not None:
            x = signal.sosfiltfilt(sos, x)

        # notch
        if DO_NOTCH:
            x = notch_filter_1d(x, fs=fs, f0=NOTCH_HZ, Q=NOTCH_Q)

        # low-pass (clip to nyquist)
        lp = EMG_LOWPASS_HZ
        if lp >= 0.5 * fs:
            lp = 0.5 * fs * 0.99
        sos = butter_sos(fs, "lowpass", lp)
        if sos is not None:
            x = signal.sosfiltfilt(sos, x)

        # optional envelope
        if USE_ENVELOPE:
            x = np.abs(x)
            env_lp = ENVELOPE_LOWPASS_HZ
            if env_lp >= 0.5 * fs:
                env_lp = 0.5 * fs * 0.99
            sos = butter_sos(fs, "lowpass", env_lp)
            if sos is not None:
                x = signal.sosfiltfilt(sos, x)

    if not np.all(np.isfinite(x)):
        raise ValueError("Non-finite values after filtering/interp.")

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
        raise ValueError("Too few samples after preprocessing.")
    return x, qc


# ============================================================
# Load plantar txt file -> matrix (T, n_signals)  [AUTO-ORIENTATION FIX]
# ============================================================
def _read_table_any(filepath: str) -> pd.DataFrame:
    # Try whitespace (no header)
    try:
        df = pd.read_csv(filepath, sep=r"\s+", engine="python", header=None)
        if df.shape[1] > 1 or df.shape[0] > 1:
            return df
    except Exception:
        pass

    # whitespace (with header)
    try:
        df = pd.read_csv(filepath, sep=r"\s+", engine="python")
        if df.shape[1] >= 1:
            return df
    except Exception:
        pass

    # CSV fallback
    return pd.read_csv(filepath)

def _numeric_matrix_from_df(df: pd.DataFrame) -> np.ndarray:
    df2 = df.apply(pd.to_numeric, errors="coerce")
    X = df2.values.astype(float)

    # drop all-NaN rows/cols
    if X.ndim != 2:
        X = np.atleast_2d(X)
    row_keep = np.any(np.isfinite(X), axis=1)
    col_keep = np.any(np.isfinite(X), axis=0)
    X = X[row_keep][:, col_keep]
    return X

def _ensure_T_by_channels(X: np.ndarray) -> np.ndarray:
    """
    We want (T, n_channels).
    Many plantar files appear to be 1 row x many columns (samples stored horizontally).
    If rows are small and cols are large -> transpose.
    """
    if X.ndim != 2:
        X = np.atleast_2d(X)

    r, c = X.shape
    if r == 0 or c == 0:
        return X

    # If it's a single row with many columns, that's a 1-channel time series
    if r == 1 and c > 1:
        return X.T  # (T=c, 1)

    # If rows are "too small" but cols are big, likely time is in columns
    if (r < MIN_VALID_SAMPLES and c >= MIN_VALID_SAMPLES) or (r < c and c >= 50):
        return X.T

    # Otherwise assume rows are time
    return X

def load_emg_table_matrix(filepath: str, fs: Optional[float]) -> Tuple[np.ndarray, List[str], Optional[float], pd.DataFrame]:
    df = _read_table_any(filepath)
    if df.empty:
        raise ValueError("Empty table.")

    X = _numeric_matrix_from_df(df)
    X = _ensure_T_by_channels(X)

    if X.size == 0:
        raise ValueError("No numeric data after cleaning.")

    # Now X is (T, n_channels)
    T, n_ch = X.shape

    # Build column names
    cols = [f"ch{j}" for j in range(n_ch)]

    kept = []
    qc_rows = []
    good_cols = []

    for j in range(n_ch):
        x = X[:, j].astype(float)
        x_clean, qc = preprocess_emg_signal(x, fs=fs)
        kept.append(x_clean)
        good_cols.append(cols[j])
        qc["signal_name"] = cols[j]
        qc_rows.append(qc)

    min_len = min(len(x) for x in kept)
    arr = np.column_stack([x[:min_len] for x in kept])  # (T, n_signals)
    qc_df = pd.DataFrame(qc_rows)
    return arr, good_cols, fs, qc_df


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
            Cr.append(C); Lr.append(L)

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

    dC = float(np.clip((Crand - C) / denomC, 0.0, 1.0))
    dL = float(np.clip((L - Lrand) / denomL, 0.0, 1.0))
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
# Core: preprocess once, compute all methods
# ============================================================
def file_to_all_method_rows(local_path: str, fs: Optional[float], seed_base: int) -> Dict[str, dict]:
    sig, col_names, fs_used, qc_df = load_emg_table_matrix(local_path, fs=fs)
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

        xQ = downsample_to_length(x, Q)

        # GAF
        W_gaf = calculate_gaf_from_lengthQ(xQ)
        per_method_metrics["GAF"].append(
            compute_smallworld_metrics_from_W(W_gaf, seed=seed_base + 100 * li + 2, use_abs_for_threshold=True)
        )

        # MTF
        W_mtf = calculate_mtf_from_lengthQ(xQ, Q=Q)
        per_method_metrics["MTF"].append(
            compute_smallworld_metrics_from_W(W_mtf, seed=seed_base + 100 * li + 3, use_abs_for_threshold=False)
        )

    out: Dict[str, dict] = {}
    for method in METHODS:
        if not per_method_metrics[method]:
            raise ValueError(f"No valid signals produced metrics for {method}.")
        dfm = pd.DataFrame(per_method_metrics[method])
        avg = dfm.mean(numeric_only=True).to_dict()
        avg.update({
            "fs_hz": fs_used if fs_used is not None else np.nan,
            "n_signals": int(sig.shape[1]),
            "T_used_samples": int(T),
            "Q_used": int(Q),
            "signal_cols": ";".join(col_names),
            "use_envelope": bool(USE_ENVELOPE),
            "qc_finite_frac_before_mean": qc_summary.get("finite_frac_before", np.nan),
            "qc_std_before_z_mean": qc_summary.get("std_before_z", np.nan),
            "qc_flat_frac_before_z_mean": qc_summary.get("flat_frac_before_z", np.nan),
            "qc_epochs_kept_frac_mean": qc_summary.get("epochs_kept_frac", np.nan),
            "qc_std_final_mean": qc_summary.get("std_final", np.nan),
        })
        out[method] = avg
    return out

def compute_task_file(rel_path: str, fs: Optional[float], task_i: int):
    url = urljoin(BASE_URL, rel_path)
    local_path = os.path.join(TMP_DIR, rel_path)

    file_id = file_id_from_path(rel_path)
    patient_id = patient_id_from_path(rel_path)

    try:
        download_file(url, local_path)
        rows = file_to_all_method_rows(local_path, fs=fs, seed_base=RNG_SEED + 100000 * task_i)

        for method in METHODS:
            rows[method].update({
                "file_id": file_id,
                "patient_id": patient_id,
                "dataset": DATASET_NAME,
                "source_file": rel_path,
            })

        return file_id, rows, None

    except Exception as e:
        skip = {
            "dataset": DATASET_NAME,
            "method": "ALL",
            "record_or_patient": rel_path,
            "reason": str(e),
        }
        return file_id, None, skip

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

    if "dataset" in df.columns:
        g.insert(1, "dataset", df.groupby("patient_id")["dataset"].first().values)

    g["n_files_used"] = df.groupby("patient_id")["file_id"].nunique().values if "file_id" in df.columns else df.groupby("patient_id").size().values
    g["aggregation"] = "mean_over_files"
    g.to_csv(out_csv, index=False)


# ============================================================
# Notebook runner
# ============================================================
def run_emg(mode: str = "manifest", fs: Optional[float] = None):
    """
    mode="manifest": only write manifest (no processing)
    mode="run": process all matching files and export QTN/GAF/MTF (per-file + per-subject)
    fs: sampling rate in Hz. If None, filtering+epochQC are skipped.
    """
    assert mode in ("manifest", "run")

    ensure_out_dir(OUT_DIR)
    ensure_out_dir(TMP_DIR)

    paths = out_paths(OUT_DIR, DATASET_NAME)
    files = discover_files_from_sha256()

    manifest_rows = [{
        "source_file": rel,
        "file_id": file_id_from_path(rel),
        "patient_id": patient_id_from_path(rel),
    } for rel in files]
    pd.DataFrame(manifest_rows).to_csv(paths["MANIFEST"], index=False)
    print(f"[{DATASET_NAME}] wrote manifest: {paths['MANIFEST']}  (n={len(files)})")

    if mode == "manifest":
        print("[MODE] manifest only.")
        return paths

    done_qtn = load_done_ids(paths["QTN_FILE"], "file_id")
    done_gaf = load_done_ids(paths["GAF_FILE"], "file_id")
    done_mtf = load_done_ids(paths["MTF_FILE"], "file_id")
    done_all = done_qtn.intersection(done_gaf).intersection(done_mtf)

    tasks = []
    task_i = 0
    for rel in files:
        fid = file_id_from_path(rel)
        if fid in done_all:
            continue
        task_i += 1
        tasks.append((rel, task_i))

    print(f"[{DATASET_NAME}] fs={fs} | discovered={len(files)} | pending={len(tasks)} | n_jobs={N_JOBS}")
    print(f"[{DATASET_NAME}] OUT_DIR={OUT_DIR} | TMP_DIR={TMP_DIR} | envelope={USE_ENVELOPE}")

    buffer_rows = {m: [] for m in METHODS}
    buffer_skips = []

    iterator = range(0, len(tasks), BATCH_SIZE)
    if TQDM_ENABLED:
        iterator = tqdm(iterator, desc=f"{DATASET_NAME} batches", unit="batch", leave=TQDM_LEAVE)

    for start in iterator:
        chunk = tasks[start:start + BATCH_SIZE]
        results = Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=0)(
            delayed(compute_task_file)(rel, fs, ti) for (rel, ti) in chunk
        )

        for file_id, rows, skip in results:
            if rows is not None:
                buffer_rows["QTN"].append(rows["QTN"])
                buffer_rows["GAF"].append(rows["GAF"])
                buffer_rows["MTF"].append(rows["MTF"])
            if skip is not None:
                buffer_skips.append(skip)

        if buffer_rows["QTN"]:
            append_rows(paths["QTN_FILE"], buffer_rows["QTN"], id_col="file_id")
            buffer_rows["QTN"].clear()
        if buffer_rows["GAF"]:
            append_rows(paths["GAF_FILE"], buffer_rows["GAF"], id_col="file_id")
            buffer_rows["GAF"].clear()
        if buffer_rows["MTF"]:
            append_rows(paths["MTF_FILE"], buffer_rows["MTF"], id_col="file_id")
            buffer_rows["MTF"].clear()

        if buffer_skips:
            append_skips(paths["SKIP"], buffer_skips)
            buffer_skips.clear()

    aggregate_per_subject(paths["QTN_FILE"], paths["QTN_SUBJ"])
    aggregate_per_subject(paths["GAF_FILE"], paths["GAF_SUBJ"])
    aggregate_per_subject(paths["MTF_FILE"], paths["MTF_SUBJ"])

    print(f"[{DATASET_NAME}] done. Outputs in {OUT_DIR}")
    return paths


# =========================
# Notebook usage:
# paths = run_emg(mode="manifest")
# paths = run_emg(mode="run", fs=1000.0)   # choose fs you believe is correct
# =========================

# In[20]:

paths = run_emg(mode="manifest")

# In[21]:

paths = run_emg(mode="run", fs=1000.0)
paths

# In[23]:

# =========================
# EMG plantar (PhysioNet plantar/1.0.0) -> QTN/GAF/MTF small-world
# Jupyter-friendly (NO argparse)
# UPDATED: robust TXT parsing + less brittle flat QC + richer skip logging
# =========================

import os, re, io, warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin

import requests
import numpy as np
import pandas as pd
import networkx as nx
from scipy import signal
from joblib import Parallel, delayed
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIG (DATASET)
# ============================================================
BASE_URL = "https://physionet.org/files/plantar/1.0.0/"
SHA256_URL = urljoin(BASE_URL, "SHA256SUMS.txt")

DATASET_NAME = "emg_plantar"

OUT_DIR = "emg_outputs/out_emg_plantar_v3"
TMP_DIR = "tmp_emg_plantar"

DELETE_RAW_AFTER_PROCESS = True
REQUEST_TIMEOUT = 120

# ============================================================
# FILE DISCOVERY FILTER (plantar/1.0.0)
# ============================================================
FILE_REGEX = re.compile(r"^[EP]_\d{3}_.+\.txt$", flags=re.IGNORECASE)
EXCLUDE_REGEXES = [
    re.compile(r"summary[-_ ]sheet", re.IGNORECASE),
    re.compile(r"readme", re.IGNORECASE),
    re.compile(r"license", re.IGNORECASE),
    re.compile(r"sha256sums", re.IGNORECASE),
    re.compile(r"\.pdf$", re.IGNORECASE),
]

# ============================================================
# PARAMETERS (QTN / SMALL-WORLD)
# ============================================================
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
BACKEND = "loky"   # if notebook hangs, try "threading"

METHODS = ["QTN", "GAF", "MTF"]
TQDM_ENABLED = True
TQDM_LEAVE = True

# ============================================================
# EMG PREPROCESSING (robust defaults)
# ============================================================
DO_DETREND = True

# Filtering (only if fs known)
EMG_HIGHPASS_HZ = 20.0
EMG_LOWPASS_HZ = 450.0
BUTTER_ORDER = 4

# Notch
DO_NOTCH = True
NOTCH_HZ = 50.0
NOTCH_Q = 30.0

# Optional envelope (off by default)
USE_ENVELOPE = False
ENVELOPE_LOWPASS_HZ = 10.0

# QC thresholds (not overly strict)
MIN_VALID_SAMPLES = 200
MIN_FINITE_FRAC = 0.70
MIN_STD = 1e-12
MIN_UNIQUE_VALUES = 5
MAX_ABS_Z = 10.0

# Flat QC: make it optional and scale-aware
USE_FLAT_QC = True
MAX_FLAT_FRAC = 0.98   # allow quantized-ish signals; we just avoid truly constant

# Epoch QC (only if fs known)
DO_EPOCH_QC = True
EPOCH_SEC = 2.0
MIN_KEEP_EPOCHS = 2
PTP_THRESHOLD_MULT = 12.0

# Missing data handling
MAX_INTERP_GAP_FRAC = 0.10

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

def flat_fraction_scale_aware(x: np.ndarray) -> float:
    """
    Scale-aware flatness:
    counts diffs "effectively zero" relative to signal scale.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 1.0
    dx = np.diff(x)
    s = float(np.nanstd(x))
    eps = max(1e-12, 1e-6 * s)   # << key change vs 1e-12 absolute
    return float(np.mean(np.abs(dx) <= eps))

def out_paths(out_dir: str, dataset_name: str):
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

def append_rows(csv_path: str, rows: List[dict], id_col: str):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if id_col not in df.columns:
        raise ValueError(f"append_rows: missing '{id_col}' in {csv_path}")
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=header)

def append_skips(csv_path: str, rows: List[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=header)

def file_id_from_path(rel_path: str) -> str:
    return Path(rel_path).stem

def patient_id_from_path(rel_path: str) -> str:
    stem = Path(rel_path).stem
    m = re.match(r"^([EP]_\d{3})_", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return stem

# ============================================================
# Discover files from SHA256SUMS
# ============================================================
def discover_files_from_sha256() -> List[str]:
    r = requests.get(SHA256_URL, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    files = []
    for line in r.text.splitlines():
        chunks = line.strip().split()
        if len(chunks) < 2:
            continue
        rel_path = chunks[-1].lstrip("./")
        if rel_path.lower().endswith("sha256sums.txt"):
            continue
        if not FILE_REGEX.match(rel_path):
            continue
        if any(rx.search(rel_path) for rx in EXCLUDE_REGEXES):
            continue
        files.append(rel_path)

    files = sorted(set(files))
    if not files:
        raise RuntimeError("No files matched FILE_REGEX from SHA256SUMS.")
    return files

# ============================================================
# Download / cleanup
# ============================================================
def download_file(url: str, local_path: str):
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return
    ensure_out_dir(str(Path(local_path).parent))
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
        r.raise_for_status()
        tmp = local_path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, local_path)

def cleanup_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

# ============================================================
# Robust TXT numeric parsing
# ============================================================
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def read_numeric_matrix_any(filepath: str) -> np.ndarray:
    """
    Goal: return a numeric matrix, even if file has headers / mixed text lines.
    Strategy:
      1) read text, normalize decimal comma, unify separators
      2) try genfromtxt with multiple delimiters
      3) if still empty -> regex-extract numbers line-by-line
    """
    raw = Path(filepath).read_bytes()
    txt = raw.decode("utf-8", errors="ignore")
    # normalize decimal comma only when it looks like "12,34"
    txt = re.sub(r"(\d),(\d)", r"\1.\2", txt)

    # attempt genfromtxt with a few delimiters
    for delim in [None, " ", "\t", ",", ";"]:
        try:
            arr = np.genfromtxt(io.StringIO(txt), delimiter=delim, invalid_raise=False)
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1) if arr.size > 0 else arr.reshape(0, 0)
            # drop all-nan rows/cols
            if arr.size > 0:
                row_keep = np.any(np.isfinite(arr), axis=1)
                col_keep = np.any(np.isfinite(arr), axis=0)
                arr = arr[row_keep][:, col_keep]
            if arr.size > 0:
                return arr
        except Exception:
            pass

    # fallback: regex numeric extraction (handles arbitrary text)
    rows = []
    for line in txt.splitlines():
        nums = _NUM_RE.findall(line)
        if not nums:
            continue
        rows.append([float(x) for x in nums])

    if not rows:
        return np.zeros((0, 0), dtype=float)

    # ragged -> pad with NaN
    m = max(len(r) for r in rows)
    out = np.full((len(rows), m), np.nan, dtype=float)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r

    # drop all-nan rows/cols
    row_keep = np.any(np.isfinite(out), axis=1)
    col_keep = np.any(np.isfinite(out), axis=0)
    out = out[row_keep][:, col_keep]
    return out

def ensure_T_by_channels(X: np.ndarray) -> np.ndarray:
    """
    We want (T, n_channels).
    If it's 1xN or has too-few rows but many cols, transpose.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        X = np.atleast_2d(X)

    r, c = X.shape
    if r == 0 or c == 0:
        return X

    # common: single row of samples -> (T,1)
    if r == 1 and c > 1:
        return X.T

    # heuristic: time dimension should usually be the larger one
    # but keep multi-channel cases (rows >> cols) intact
    if (r < MIN_VALID_SAMPLES and c >= MIN_VALID_SAMPLES) or (r < c and c >= 50):
        return X.T

    return X

# ============================================================
# Preprocessing (EMG)
# ============================================================
def interpolate_short_gaps(x: np.ndarray, max_gap_frac: float = 0.10) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    n = x.size
    finite = np.isfinite(x)

    if n == 0 or finite.sum() == 0:
        return x
    if finite.all():
        return x

    idx = np.arange(n)
    x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])

    # reject too-long missing gap
    gap_mask = ~finite
    if gap_mask.any():
        starts = np.where(np.diff(np.r_[False, gap_mask, False].astype(int)) == 1)[0]
        ends = np.where(np.diff(np.r_[False, gap_mask, False].astype(int)) == -1)[0]
        max_gap = max((e - s) for s, e in zip(starts, ends)) if len(starts) else 0
        if max_gap > max_gap_frac * n:
            raise ValueError(f"Gap too long: {max_gap} samples.")
    return x

def butter_sos(fs: float, btype: str, cutoff):
    nyq = 0.5 * fs
    if btype in ("highpass", "lowpass"):
        w = float(cutoff) / nyq
        if not (0 < w < 1):
            return None
        return signal.butter(BUTTER_ORDER, w, btype=btype, output="sos")
    elif btype == "bandpass":
        lo, hi = cutoff
        hi = min(float(hi), nyq * 0.99)
        lo = float(lo)
        if not (0 < lo < hi < nyq):
            return None
        return signal.butter(BUTTER_ORDER, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return None

def notch_filter_1d(x: np.ndarray, fs: float, f0: float, Q: float):
    if f0 <= 0 or f0 >= 0.5 * fs:
        return x
    b, a = signal.iirnotch(w0=f0, Q=Q, fs=fs)
    return signal.filtfilt(b, a, x)

def epoch_qc_keep_mask(x: np.ndarray, fs: Optional[float]) -> np.ndarray:
    if (not DO_EPOCH_QC) or fs is None or (not np.isfinite(fs)) or fs <= 0:
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

def preprocess_emg_signal(x: np.ndarray, fs: Optional[float]) -> Tuple[np.ndarray, dict]:
    qc: Dict[str, float] = {}
    x = np.asarray(x, dtype=float)

    if x.size < MIN_VALID_SAMPLES:
        raise ValueError(f"Too few samples (n={x.size}).")

    finite = np.isfinite(x)
    finite_frac = float(finite.mean())
    qc["finite_frac_before"] = finite_frac
    qc["n_total"] = int(x.size)
    qc["n_finite_before"] = int(finite.sum())

    if finite_frac < MIN_FINITE_FRAC:
        raise ValueError(f"Too many NaNs: finite_frac={finite_frac:.3f}")

    x = interpolate_short_gaps(x, max_gap_frac=MAX_INTERP_GAP_FRAC)

    if DO_DETREND:
        x = signal.detrend(x, type="constant")

    # filtering only if fs known
    if fs is not None and np.isfinite(fs) and fs > 0:
        fs = float(fs)

        sos = butter_sos(fs, "highpass", EMG_HIGHPASS_HZ)
        if sos is not None:
            x = signal.sosfiltfilt(sos, x)

        if DO_NOTCH:
            x = notch_filter_1d(x, fs=fs, f0=NOTCH_HZ, Q=NOTCH_Q)

        lp = min(EMG_LOWPASS_HZ, 0.5 * fs * 0.99)
        sos = butter_sos(fs, "lowpass", lp)
        if sos is not None:
            x = signal.sosfiltfilt(sos, x)

        if USE_ENVELOPE:
            x = np.abs(x)
            env_lp = min(ENVELOPE_LOWPASS_HZ, 0.5 * fs * 0.99)
            sos = butter_sos(fs, "lowpass", env_lp)
            if sos is not None:
                x = signal.sosfiltfilt(sos, x)

    if not np.all(np.isfinite(x)):
        raise ValueError("Non-finite after interp/filter.")

    qc["std_before_z"] = float(np.std(x))
    qc["n_unique_before_z"] = int(np.unique(np.round(x, 10)).size)
    qc["flat_frac_before_z"] = float(flat_fraction_scale_aware(x))

    if qc["std_before_z"] <= MIN_STD:
        raise ValueError("Variance too small.")
    if qc["n_unique_before_z"] < MIN_UNIQUE_VALUES:
        raise ValueError("Too few unique values.")
    if USE_FLAT_QC and qc["flat_frac_before_z"] > MAX_FLAT_FRAC:
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
        raise ValueError("Too few samples after preprocessing.")
    return x, qc

# ============================================================
# Load plantar txt file -> matrix (T, n_signals)
# ============================================================
def load_emg_table_matrix(filepath: str, fs: Optional[float]) -> Tuple[np.ndarray, List[str], Optional[float], pd.DataFrame]:
    X0 = read_numeric_matrix_any(filepath)
    if X0.size == 0:
        raise ValueError("No numeric data after parsing.")

    X = ensure_T_by_channels(X0)
    if X.size == 0:
        raise ValueError("No numeric data after orientation.")

    # drop all-NaN rows/cols again (after transpose)
    row_keep = np.any(np.isfinite(X), axis=1)
    col_keep = np.any(np.isfinite(X), axis=0)
    X = X[row_keep][:, col_keep]
    if X.size == 0:
        raise ValueError("No numeric data after cleaning.")

    T, n_ch = X.shape
    cols = [f"ch{j}" for j in range(n_ch)]

    kept, qc_rows = [], []
    good_cols = []
    for j in range(n_ch):
        x = X[:, j]
        x_clean, qc = preprocess_emg_signal(x, fs=fs)
        kept.append(x_clean)
        good_cols.append(cols[j])
        qc["signal_name"] = cols[j]
        qc_rows.append(qc)

    min_len = min(len(x) for x in kept)
    arr = np.column_stack([x[:min_len] for x in kept])
    qc_df = pd.DataFrame(qc_rows)
    return arr, good_cols, fs, qc_df

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
            Cr.append(C); Lr.append(L)

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

    dC = float(np.clip((Crand - C) / denomC, 0.0, 1.0))
    dL = float(np.clip((L - Lrand) / denomL, 0.0, 1.0))
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

    return dict(
        n_nodes=int(n),
        density=float(np.sum(B) / (n * (n - 1))) if n > 1 else np.nan,
        transitivity=float(C_obs) if not np.isnan(C_obs) else np.nan,
        char_path_len_gcc=float(L_obs) if not np.isnan(L_obs) else np.nan,
        gamma_C_over_Crand=float(gamma) if not np.isnan(gamma) else np.nan,
        lambda_L_over_Lrand=float(lambd) if not np.isnan(lambd) else np.nan,
        sigma_small_world=float(sigma) if not np.isnan(sigma) else np.nan,
        zC=float(zC) if not np.isnan(zC) else np.nan,
        zL=float(zL) if not np.isnan(zL) else np.nan,
        omega=float(omega) if not np.isnan(omega) else np.nan,
        phi=float(phi) if not np.isnan(phi) else np.nan,
        global_efficiency=float(nx.global_efficiency(nx.from_numpy_array(B))),
    )

# ============================================================
# Core: preprocess once, compute all methods
# ============================================================
def file_to_all_method_rows(local_path: str, fs: Optional[float], seed_base: int) -> Dict[str, dict]:
    sig, col_names, fs_used, qc_df = load_emg_table_matrix(local_path, fs=fs)
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

    out: Dict[str, dict] = {}
    for method in METHODS:
        if not per_method_metrics[method]:
            raise ValueError(f"No valid signals produced metrics for {method}.")
        dfm = pd.DataFrame(per_method_metrics[method])
        avg = dfm.mean(numeric_only=True).to_dict()
        avg.update({
            "fs_hz": float(fs_used) if fs_used is not None else np.nan,
            "n_signals": int(sig.shape[1]),
            "T_used_samples": int(T),
            "Q_used": int(Q),
            "signal_cols": ";".join(col_names),
            "use_envelope": bool(USE_ENVELOPE),
            "use_flat_qc": bool(USE_FLAT_QC),
            "qc_finite_frac_before_mean": qc_summary.get("finite_frac_before", np.nan),
            "qc_std_before_z_mean": qc_summary.get("std_before_z", np.nan),
            "qc_flat_frac_before_z_mean": qc_summary.get("flat_frac_before_z", np.nan),
            "qc_epochs_kept_frac_mean": qc_summary.get("epochs_kept_frac", np.nan),
            "qc_std_final_mean": qc_summary.get("std_final", np.nan),
        })
        out[method] = avg
    return out

def compute_task_file(rel_path: str, fs: Optional[float], task_i: int):
    url = urljoin(BASE_URL, rel_path)
    local_path = os.path.join(TMP_DIR, rel_path)

    file_id = file_id_from_path(rel_path)
    patient_id = patient_id_from_path(rel_path)

    try:
        download_file(url, local_path)
        rows = file_to_all_method_rows(local_path, fs=fs, seed_base=RNG_SEED + 100000 * task_i)

        for method in METHODS:
            rows[method].update({
                "file_id": file_id,
                "patient_id": patient_id,
                "dataset": DATASET_NAME,
                "source_file": rel_path,
            })

        return file_id, rows, None

    except Exception as e:
        # richer debug info (stage not perfect, but still useful)
        skip = {
            "dataset": DATASET_NAME,
            "method": "ALL",
            "record_or_patient": rel_path,
            "file_id": file_id,
            "patient_id": patient_id,
            "reason": str(e),
        }
        return file_id, None, skip

    finally:
        if DELETE_RAW_AFTER_PROCESS:
            cleanup_file(local_path)

def aggregate_per_subject(file_csv: str, out_csv: str):
    if not os.path.exists(file_csv):
        return
    df = pd.read_csv(file_csv)
    if df.empty or "patient_id" not in df.columns:
        return
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    g = df.groupby("patient_id")[numeric_cols].mean().reset_index()
    if "dataset" in df.columns:
        g.insert(1, "dataset", df.groupby("patient_id")["dataset"].first().values)
    g["n_files_used"] = df.groupby("patient_id")["file_id"].nunique().values if "file_id" in df.columns else df.groupby("patient_id").size().values
    g["aggregation"] = "mean_over_files"
    g.to_csv(out_csv, index=False)

def run_emg(mode: str = "manifest", fs: Optional[float] = None):
    """
    mode="manifest": only write manifest
    mode="run": process all matching files and export QTN/GAF/MTF
    fs: sampling rate in Hz. If None, filtering+epochQC are skipped.
    """
    assert mode in ("manifest", "run")

    ensure_out_dir(OUT_DIR)
    ensure_out_dir(TMP_DIR)

    paths = out_paths(OUT_DIR, DATASET_NAME)
    files = discover_files_from_sha256()

    manifest_rows = [{
        "source_file": rel,
        "file_id": file_id_from_path(rel),
        "patient_id": patient_id_from_path(rel),
    } for rel in files]
    pd.DataFrame(manifest_rows).to_csv(paths["MANIFEST"], index=False)
    print(f"[{DATASET_NAME}] wrote manifest: {paths['MANIFEST']}  (n={len(files)})")

    if mode == "manifest":
        print("[MODE] manifest only.")
        return paths

    done_qtn = load_done_ids(paths["QTN_FILE"], "file_id")
    done_gaf = load_done_ids(paths["GAF_FILE"], "file_id")
    done_mtf = load_done_ids(paths["MTF_FILE"], "file_id")
    done_all = done_qtn.intersection(done_gaf).intersection(done_mtf)

    tasks = []
    task_i = 0
    for rel in files:
        fid = file_id_from_path(rel)
        if fid in done_all:
            continue
        task_i += 1
        tasks.append((rel, task_i))

    print(f"[{DATASET_NAME}] fs={fs} | discovered={len(files)} | pending={len(tasks)} | n_jobs={N_JOBS}")
    print(f"[{DATASET_NAME}] OUT_DIR={OUT_DIR} | TMP_DIR={TMP_DIR} | envelope={USE_ENVELOPE} | flat_qc={USE_FLAT_QC}")

    buffer_rows = {m: [] for m in METHODS}
    buffer_skips = []

    iterator = range(0, len(tasks), BATCH_SIZE)
    if TQDM_ENABLED:
        iterator = tqdm(iterator, desc=f"{DATASET_NAME} batches", unit="batch", leave=TQDM_LEAVE)

    for start in iterator:
        chunk = tasks[start:start + BATCH_SIZE]
        results = Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=0)(
            delayed(compute_task_file)(rel, fs, ti) for (rel, ti) in chunk
        )

        for file_id, rows, skip in results:
            if rows is not None:
                buffer_rows["QTN"].append(rows["QTN"])
                buffer_rows["GAF"].append(rows["GAF"])
                buffer_rows["MTF"].append(rows["MTF"])
            if skip is not None:
                buffer_skips.append(skip)

        if buffer_rows["QTN"]:
            append_rows(paths["QTN_FILE"], buffer_rows["QTN"], id_col="file_id")
            buffer_rows["QTN"].clear()
        if buffer_rows["GAF"]:
            append_rows(paths["GAF_FILE"], buffer_rows["GAF"], id_col="file_id")
            buffer_rows["GAF"].clear()
        if buffer_rows["MTF"]:
            append_rows(paths["MTF_FILE"], buffer_rows["MTF"], id_col="file_id")
            buffer_rows["MTF"].clear()

        if buffer_skips:
            append_skips(paths["SKIP"], buffer_skips)
            buffer_skips.clear()

    aggregate_per_subject(paths["QTN_FILE"], paths["QTN_SUBJ"])
    aggregate_per_subject(paths["GAF_FILE"], paths["GAF_SUBJ"])
    aggregate_per_subject(paths["MTF_FILE"], paths["MTF_SUBJ"])

    print(f"[{DATASET_NAME}] done. Outputs in {OUT_DIR}")
    return paths

# ---- notebook usage ----
# paths = run_emg("manifest")
# paths = run_emg("run", fs=1000.0)
USE_FLAT_QC = False
paths = run_emg("run", fs=1000.0)

# In[1]:

# =========================
# EMG plantar (PhysioNet plantar/1.0.0) -> QTN/GAF/MTF small-world
# Jupyter-friendly (NO argparse)
# OPTION A: EMG-only (E_*) files ONLY
# =========================

import os, re, io, warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin

import requests
import numpy as np
import pandas as pd
import networkx as nx
from scipy import signal
from joblib import Parallel, delayed
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIG (DATASET)
# ============================================================
BASE_URL = "https://physionet.org/files/plantar/1.0.0/"
SHA256_URL = urljoin(BASE_URL, "SHA256SUMS.txt")

DATASET_NAME = "emg_plantar_EMGONLY"   # <- make it explicit

OUT_DIR = "emg_outputs/out_emg_plantar_EMGONLY_v1"
TMP_DIR = "tmp_emg_plantar_EMGONLY_v1"

DELETE_RAW_AFTER_PROCESS = True
REQUEST_TIMEOUT = 120

# ============================================================
# FILE DISCOVERY FILTER (EMG ONLY)
# ============================================================
# IMPORTANT: Only E_ files (EMG) are included
FILE_REGEX = re.compile(r"^E_\d{3}_.+\.txt$", flags=re.IGNORECASE)

EXCLUDE_REGEXES = [
    re.compile(r"summary[-_ ]sheet", re.IGNORECASE),
    re.compile(r"readme", re.IGNORECASE),
    re.compile(r"license", re.IGNORECASE),
    re.compile(r"sha256sums", re.IGNORECASE),
    re.compile(r"\.pdf$", re.IGNORECASE),
]

# ============================================================
# PARAMETERS (QTN / SMALL-WORLD)
# ============================================================
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
BACKEND = "loky"   # if notebook hangs, try "threading"

METHODS = ["QTN", "GAF", "MTF"]
TQDM_ENABLED = True
TQDM_LEAVE = True

# ============================================================
# EMG PREPROCESSING
# ============================================================
DO_DETREND = True

# Filtering (only if fs known)
EMG_HIGHPASS_HZ = 20.0
EMG_LOWPASS_HZ = 450.0
BUTTER_ORDER = 4

# Notch
DO_NOTCH = True
NOTCH_HZ = 50.0
NOTCH_Q = 30.0

# Optional envelope (off by default)
USE_ENVELOPE = False
ENVELOPE_LOWPASS_HZ = 10.0

# QC thresholds
MIN_VALID_SAMPLES = 200
MIN_FINITE_FRAC = 0.70
MIN_STD = 1e-12
MIN_UNIQUE_VALUES = 5
MAX_ABS_Z = 10.0

# Flat QC: optional and scale-aware
USE_FLAT_QC = False           # <- you set this in your notebook; keep default False for EMG
MAX_FLAT_FRAC = 0.98

# Epoch QC (only if fs known)
DO_EPOCH_QC = True
EPOCH_SEC = 2.0
MIN_KEEP_EPOCHS = 2
PTP_THRESHOLD_MULT = 12.0

# Missing data handling
MAX_INTERP_GAP_FRAC = 0.10

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

def flat_fraction_scale_aware(x: np.ndarray) -> float:
    """
    Scale-aware flatness:
    counts diffs "effectively zero" relative to signal scale.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 1.0
    dx = np.diff(x)
    s = float(np.nanstd(x))
    eps = max(1e-12, 1e-6 * s)
    return float(np.mean(np.abs(dx) <= eps))

def out_paths(out_dir: str, dataset_name: str):
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

def append_rows(csv_path: str, rows: List[dict], id_col: str):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if id_col not in df.columns:
        raise ValueError(f"append_rows: missing '{id_col}' in {csv_path}")
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=header)

def append_skips(csv_path: str, rows: List[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=header)

def file_id_from_path(rel_path: str) -> str:
    # robust to any folder structure
    return Path(rel_path).name.rsplit(".", 1)[0]

def patient_id_from_path(rel_path: str) -> str:
    # EMG-only: E_###
    stem = Path(rel_path).name.rsplit(".", 1)[0]
    m = re.match(r"^(E_\d{3})_", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return stem

# ============================================================
# Discover files from SHA256SUMS
# ============================================================
def discover_files_from_sha256() -> List[str]:
    r = requests.get(SHA256_URL, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    files = []
    for line in r.text.splitlines():
        chunks = line.strip().split()
        if len(chunks) < 2:
            continue
        rel_path = chunks[-1].lstrip("./")
        if rel_path.lower().endswith("sha256sums.txt"):
            continue
        if any(rx.search(rel_path) for rx in EXCLUDE_REGEXES):
            continue

        name = Path(rel_path).name  # <- match filename, not full path
        if not FILE_REGEX.match(name):
            continue

        files.append(rel_path)

    files = sorted(set(files))
    if not files:
        raise RuntimeError("No files matched FILE_REGEX from SHA256SUMS.")
    return files

# ============================================================
# Download / cleanup
# ============================================================
def download_file(url: str, local_path: str):
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return
    ensure_out_dir(str(Path(local_path).parent))
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
        r.raise_for_status()
        tmp = local_path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, local_path)

def cleanup_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

# ============================================================
# Robust TXT numeric parsing
# ============================================================
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def read_numeric_matrix_any(filepath: str) -> np.ndarray:
    """
    Return a numeric matrix even if file has headers / mixed text lines.
    Strategy:
      1) read text, normalize decimal comma
      2) try genfromtxt with multiple delimiters
      3) fallback to regex numeric extraction line-by-line
    """
    raw = Path(filepath).read_bytes()
    txt = raw.decode("utf-8", errors="ignore")
    txt = re.sub(r"(\d),(\d)", r"\1.\2", txt)

    for delim in [None, " ", "\t", ",", ";"]:
        try:
            arr = np.genfromtxt(io.StringIO(txt), delimiter=delim, invalid_raise=False)
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1) if arr.size > 0 else arr.reshape(0, 0)
            if arr.size > 0:
                row_keep = np.any(np.isfinite(arr), axis=1)
                col_keep = np.any(np.isfinite(arr), axis=0)
                arr = arr[row_keep][:, col_keep]
            if arr.size > 0:
                return arr
        except Exception:
            pass

    rows = []
    for line in txt.splitlines():
        nums = _NUM_RE.findall(line)
        if not nums:
            continue
        rows.append([float(x) for x in nums])

    if not rows:
        return np.zeros((0, 0), dtype=float)

    m = max(len(r) for r in rows)
    out = np.full((len(rows), m), np.nan, dtype=float)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r

    row_keep = np.any(np.isfinite(out), axis=1)
    col_keep = np.any(np.isfinite(out), axis=0)
    out = out[row_keep][:, col_keep]
    return out

def ensure_T_by_channels(X: np.ndarray) -> np.ndarray:
    """
    We want (T, n_channels).
    If it's 1xN or has too-few rows but many cols, transpose.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        X = np.atleast_2d(X)

    r, c = X.shape
    if r == 0 or c == 0:
        return X

    if r == 1 and c > 1:
        return X.T

    if (r < MIN_VALID_SAMPLES and c >= MIN_VALID_SAMPLES) or (r < c and c >= 50):
        return X.T

    return X

# ============================================================
# Preprocessing (EMG)
# ============================================================
def interpolate_short_gaps(x: np.ndarray, max_gap_frac: float = 0.10) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    n = x.size
    finite = np.isfinite(x)

    if n == 0 or finite.sum() == 0:
        return x
    if finite.all():
        return x

    idx = np.arange(n)
    x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])

    gap_mask = ~finite
    if gap_mask.any():
        starts = np.where(np.diff(np.r_[False, gap_mask, False].astype(int)) == 1)[0]
        ends = np.where(np.diff(np.r_[False, gap_mask, False].astype(int)) == -1)[0]
        max_gap = max((e - s) for s, e in zip(starts, ends)) if len(starts) else 0
        if max_gap > max_gap_frac * n:
            raise ValueError(f"Gap too long: {max_gap} samples.")
    return x

def butter_sos(fs: float, btype: str, cutoff):
    nyq = 0.5 * fs
    if btype in ("highpass", "lowpass"):
        w = float(cutoff) / nyq
        if not (0 < w < 1):
            return None
        return signal.butter(BUTTER_ORDER, w, btype=btype, output="sos")
    elif btype == "bandpass":
        lo, hi = cutoff
        hi = min(float(hi), nyq * 0.99)
        lo = float(lo)
        if not (0 < lo < hi < nyq):
            return None
        return signal.butter(BUTTER_ORDER, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return None

def notch_filter_1d(x: np.ndarray, fs: float, f0: float, Q: float):
    if f0 <= 0 or f0 >= 0.5 * fs:
        return x
    b, a = signal.iirnotch(w0=f0, Q=Q, fs=fs)
    return signal.filtfilt(b, a, x)

def epoch_qc_keep_mask(x: np.ndarray, fs: Optional[float]) -> np.ndarray:
    if (not DO_EPOCH_QC) or fs is None or (not np.isfinite(fs)) or fs <= 0:
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

def preprocess_emg_signal(x: np.ndarray, fs: Optional[float]) -> Tuple[np.ndarray, dict]:
    qc: Dict[str, float] = {}
    x = np.asarray(x, dtype=float)

    if x.size < MIN_VALID_SAMPLES:
        raise ValueError(f"Too few samples (n={x.size}).")

    finite = np.isfinite(x)
    finite_frac = float(finite.mean())
    qc["finite_frac_before"] = finite_frac
    qc["n_total"] = int(x.size)
    qc["n_finite_before"] = int(finite.sum())

    if finite_frac < MIN_FINITE_FRAC:
        raise ValueError(f"Too many NaNs: finite_frac={finite_frac:.3f}")

    x = interpolate_short_gaps(x, max_gap_frac=MAX_INTERP_GAP_FRAC)

    if DO_DETREND:
        x = signal.detrend(x, type="constant")

    if fs is not None and np.isfinite(fs) and fs > 0:
        fs = float(fs)

        sos = butter_sos(fs, "highpass", EMG_HIGHPASS_HZ)
        if sos is not None:
            x = signal.sosfiltfilt(sos, x)

        if DO_NOTCH:
            x = notch_filter_1d(x, fs=fs, f0=NOTCH_HZ, Q=NOTCH_Q)

        lp = min(EMG_LOWPASS_HZ, 0.5 * fs * 0.99)
        sos = butter_sos(fs, "lowpass", lp)
        if sos is not None:
            x = signal.sosfiltfilt(sos, x)

        if USE_ENVELOPE:
            x = np.abs(x)
            env_lp = min(ENVELOPE_LOWPASS_HZ, 0.5 * fs * 0.99)
            sos = butter_sos(fs, "lowpass", env_lp)
            if sos is not None:
                x = signal.sosfiltfilt(sos, x)

    if not np.all(np.isfinite(x)):
        raise ValueError("Non-finite after interp/filter.")

    qc["std_before_z"] = float(np.std(x))
    qc["n_unique_before_z"] = int(np.unique(np.round(x, 10)).size)
    qc["flat_frac_before_z"] = float(flat_fraction_scale_aware(x))

    if qc["std_before_z"] <= MIN_STD:
        raise ValueError("Variance too small.")
    if qc["n_unique_before_z"] < MIN_UNIQUE_VALUES:
        raise ValueError("Too few unique values.")
    if USE_FLAT_QC and qc["flat_frac_before_z"] > MAX_FLAT_FRAC:
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
        raise ValueError("Too few samples after preprocessing.")
    return x, qc

# ============================================================
# Load plantar txt file -> matrix (T, n_signals)
# ============================================================
def load_emg_table_matrix(filepath: str, fs: Optional[float]) -> Tuple[np.ndarray, List[str], Optional[float], pd.DataFrame]:
    X0 = read_numeric_matrix_any(filepath)
    if X0.size == 0:
        raise ValueError("No numeric data after parsing.")

    X = ensure_T_by_channels(X0)
    if X.size == 0:
        raise ValueError("No numeric data after orientation.")

    row_keep = np.any(np.isfinite(X), axis=1)
    col_keep = np.any(np.isfinite(X), axis=0)
    X = X[row_keep][:, col_keep]
    if X.size == 0:
        raise ValueError("No numeric data after cleaning.")

    T, n_ch = X.shape
    cols = [f"ch{j}" for j in range(n_ch)]

    kept, qc_rows, good_cols = [], [], []
    for j in range(n_ch):
        x = X[:, j]
        x_clean, qc = preprocess_emg_signal(x, fs=fs)
        kept.append(x_clean)
        good_cols.append(cols[j])
        qc["signal_name"] = cols[j]
        qc_rows.append(qc)

    min_len = min(len(x) for x in kept)
    arr = np.column_stack([x[:min_len] for x in kept])
    qc_df = pd.DataFrame(qc_rows)
    return arr, good_cols, fs, qc_df

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
            Cr.append(C); Lr.append(L)

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

    dC = float(np.clip((Crand - C) / denomC, 0.0, 1.0))
    dL = float(np.clip((L - Lrand) / denomL, 0.0, 1.0))
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

    return dict(
        n_nodes=int(n),
        density=float(np.sum(B) / (n * (n - 1))) if n > 1 else np.nan,
        transitivity=float(C_obs) if not np.isnan(C_obs) else np.nan,
        char_path_len_gcc=float(L_obs) if not np.isnan(L_obs) else np.nan,
        gamma_C_over_Crand=float(gamma) if not np.isnan(gamma) else np.nan,
        lambda_L_over_Lrand=float(lambd) if not np.isnan(lambd) else np.nan,
        sigma_small_world=float(sigma) if not np.isnan(sigma) else np.nan,
        zC=float(zC) if not np.isnan(zC) else np.nan,
        zL=float(zL) if not np.isnan(zL) else np.nan,
        omega=float(omega) if not np.isnan(omega) else np.nan,
        phi=float(phi) if not np.isnan(phi) else np.nan,
        global_efficiency=float(nx.global_efficiency(nx.from_numpy_array(B))),
    )

# ============================================================
# Core: preprocess once, compute all methods
# ============================================================
def file_to_all_method_rows(local_path: str, fs: Optional[float], seed_base: int) -> Dict[str, dict]:
    sig, col_names, fs_used, qc_df = load_emg_table_matrix(local_path, fs=fs)
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

    out: Dict[str, dict] = {}
    for method in METHODS:
        if not per_method_metrics[method]:
            raise ValueError(f"No valid signals produced metrics for {method}.")
        dfm = pd.DataFrame(per_method_metrics[method])
        avg = dfm.mean(numeric_only=True).to_dict()
        avg.update({
            "fs_hz": float(fs_used) if fs_used is not None else np.nan,
            "n_signals": int(sig.shape[1]),
            "T_used_samples": int(T),
            "Q_used": int(Q),
            "signal_cols": ";".join(col_names),
            "use_envelope": bool(USE_ENVELOPE),
            "use_flat_qc": bool(USE_FLAT_QC),
            "qc_finite_frac_before_mean": qc_summary.get("finite_frac_before", np.nan),
            "qc_std_before_z_mean": qc_summary.get("std_before_z", np.nan),
            "qc_flat_frac_before_z_mean": qc_summary.get("flat_frac_before_z", np.nan),
            "qc_epochs_kept_frac_mean": qc_summary.get("epochs_kept_frac", np.nan),
            "qc_std_final_mean": qc_summary.get("std_final", np.nan),
        })
        out[method] = avg
    return out

def compute_task_file(rel_path: str, fs: Optional[float], task_i: int):
    url = urljoin(BASE_URL, rel_path)
    local_path = os.path.join(TMP_DIR, rel_path)

    file_id = file_id_from_path(rel_path)
    patient_id = patient_id_from_path(rel_path)

    try:
        download_file(url, local_path)
        rows = file_to_all_method_rows(local_path, fs=fs, seed_base=RNG_SEED + 100000 * task_i)

        for method in METHODS:
            rows[method].update({
                "file_id": file_id,
                "patient_id": patient_id,
                "dataset": DATASET_NAME,
                "source_file": rel_path,
            })

        return file_id, rows, None

    except Exception as e:
        skip = {
            "dataset": DATASET_NAME,
            "method": "ALL",
            "record_or_patient": rel_path,
            "file_id": file_id,
            "patient_id": patient_id,
            "reason": str(e),
        }
        return file_id, None, skip

    finally:
        if DELETE_RAW_AFTER_PROCESS:
            cleanup_file(local_path)

def aggregate_per_subject(file_csv: str, out_csv: str):
    if not os.path.exists(file_csv):
        return
    df = pd.read_csv(file_csv)
    if df.empty or "patient_id" not in df.columns:
        return
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    g = df.groupby("patient_id")[numeric_cols].mean().reset_index()
    if "dataset" in df.columns:
        g.insert(1, "dataset", df.groupby("patient_id")["dataset"].first().values)
    g["n_files_used"] = df.groupby("patient_id")["file_id"].nunique().values if "file_id" in df.columns else df.groupby("patient_id").size().values
    g["aggregation"] = "mean_over_files"
    g.to_csv(out_csv, index=False)

def run_emg(mode: str = "manifest", fs: Optional[float] = None):
    """
    mode="manifest": only write manifest
    mode="run": process all matching EMG (E_*) files and export QTN/GAF/MTF
    fs: sampling rate in Hz. If None, filtering+epochQC are skipped.
    """
    assert mode in ("manifest", "run")

    ensure_out_dir(OUT_DIR)
    ensure_out_dir(TMP_DIR)

    paths = out_paths(OUT_DIR, DATASET_NAME)
    files = discover_files_from_sha256()

    manifest_rows = [{
        "source_file": rel,
        "file_id": file_id_from_path(rel),
        "patient_id": patient_id_from_path(rel),
    } for rel in files]
    pd.DataFrame(manifest_rows).to_csv(paths["MANIFEST"], index=False)
    print(f"[{DATASET_NAME}] wrote manifest: {paths['MANIFEST']}  (n_files={len(files)} | n_subjects={pd.DataFrame(manifest_rows)['patient_id'].nunique()})")

    if mode == "manifest":
        print("[MODE] manifest only.")
        return paths

    done_qtn = load_done_ids(paths["QTN_FILE"], "file_id")
    done_gaf = load_done_ids(paths["GAF_FILE"], "file_id")
    done_mtf = load_done_ids(paths["MTF_FILE"], "file_id")
    done_all = done_qtn.intersection(done_gaf).intersection(done_mtf)

    tasks = []
    task_i = 0
    for rel in files:
        fid = file_id_from_path(rel)
        if fid in done_all:
            continue
        task_i += 1
        tasks.append((rel, task_i))

    print(f"[{DATASET_NAME}] fs={fs} | discovered={len(files)} | pending={len(tasks)} | n_jobs={N_JOBS}")
    print(f"[{DATASET_NAME}] OUT_DIR={OUT_DIR} | TMP_DIR={TMP_DIR} | envelope={USE_ENVELOPE} | flat_qc={USE_FLAT_QC}")

    buffer_rows = {m: [] for m in METHODS}
    buffer_skips = []

    iterator = range(0, len(tasks), BATCH_SIZE)
    if TQDM_ENABLED:
        iterator = tqdm(iterator, desc=f"{DATASET_NAME} batches", unit="batch", leave=TQDM_LEAVE)

    for start in iterator:
        chunk = tasks[start:start + BATCH_SIZE]
        results = Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=0)(
            delayed(compute_task_file)(rel, fs, ti) for (rel, ti) in chunk
        )

        for file_id, rows, skip in results:
            if rows is not None:
                buffer_rows["QTN"].append(rows["QTN"])
                buffer_rows["GAF"].append(rows["GAF"])
                buffer_rows["MTF"].append(rows["MTF"])
            if skip is not None:
                buffer_skips.append(skip)

        if buffer_rows["QTN"]:
            append_rows(paths["QTN_FILE"], buffer_rows["QTN"], id_col="file_id")
            buffer_rows["QTN"].clear()
        if buffer_rows["GAF"]:
            append_rows(paths["GAF_FILE"], buffer_rows["GAF"], id_col="file_id")
            buffer_rows["GAF"].clear()
        if buffer_rows["MTF"]:
            append_rows(paths["MTF_FILE"], buffer_rows["MTF"], id_col="file_id")
            buffer_rows["MTF"].clear()

        if buffer_skips:
            append_skips(paths["SKIP"], buffer_skips)
            buffer_skips.clear()

    aggregate_per_subject(paths["QTN_FILE"], paths["QTN_SUBJ"])
    aggregate_per_subject(paths["GAF_FILE"], paths["GAF_SUBJ"])
    aggregate_per_subject(paths["MTF_FILE"], paths["MTF_SUBJ"])

    print(f"[{DATASET_NAME}] done. Outputs in {OUT_DIR}")
    return paths

# ---- notebook usage ----
# 1) Optional: only inspect what will run
# paths = run_emg("manifest")

# 2) Run EMG-only processing
paths = run_emg("run", fs=1000.0)
print(paths)

# In[ ]:
