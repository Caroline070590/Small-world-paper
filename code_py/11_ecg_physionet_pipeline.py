#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto-converted from the Jupyter notebook ``ecg-2-data-FINAL.ipynb``.

Code cells are reproduced verbatim. Markdown cells are kept as comments.
Jupyter shell-magic lines (if any) are commented out so the file is valid
Python; they are preserved for reference. Cell boundaries are marked with
``# In[...]`` to match the original notebook ordering.
"""


# In[ ]:

#!/usr/bin/env python3
# ecg_qtn_gaf_mtf_smallworld_fantasia_nsrdb_preprocessed_parallel_resume.py
#
# Features:
# - Fantasia + NSRDB only
# - Explicit ECG preprocessing
# - Parallel with limited workers
# - Resume if CSV already exists
# - Incremental saving in batches
# - TQDM progress bars
#
# Outputs per dataset folder:
#   metrics_QTN_<dataset>.csv
#   metrics_GAF_<dataset>.csv
#   metrics_MTF_<dataset>.csv
#   skipped_<dataset>.csv

import os
import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
import wfdb
import mne

from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =======================
# CONFIG
# =======================
DATASETS = {
    "nsrdb": {
        "pn_dir": "nsrdb",
        "out_dir": "out_nsrdb",
        "mode": "record_is_patient",
    },
    "fantasia": {
        "pn_dir": "fantasia",
        "out_dir": "out_fantasia",
        "mode": "record_is_patient",
    },
}

K_VALUES = [1, 2, 3]
TARGET_DENSITY = 0.10
N_RANDOMIZATIONS = 20
REWIRINGS_PER_EDGE = 5
RNG_SEED = 1234

MAX_SAMPLES = None
START_SAMPLE = 0

# -----------------------
# Parallel / resume
# -----------------------
N_JOBS = 4
BATCH_SIZE = 6
BACKEND = "loky"
METHODS = ["QTN", "GAF", "MTF"]

TQDM_ENABLED = True
TQDM_LEAVE = True

# =======================
# ECG preprocessing
# =======================
TARGET_FS = 250.0
LINE_FREQ = 50.0
ECG_L_FREQ = 0.5
ECG_H_FREQ = 40.0
EPOCH_SEC = 10.0
MIN_KEEP_EPOCHS = 3

# =======================
# Q scaling
# =======================
def compute_Q_from_T(T: int) -> int:
    return int(round(2 * (T ** (1 / 3))))

# =======================
# Time-series helpers
# =======================
def slice_signals(sig: np.ndarray) -> np.ndarray:
    start = int(START_SAMPLE or 0)
    if MAX_SAMPLES is None:
        return sig[start:, :]
    end = start + int(MAX_SAMPLES)
    return sig[start:end, :]

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
    x_new = np.interp(np.linspace(0.0, 1.0, L), xp, x)
    return x_new.astype(float)

def robust_zscore(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        std = np.nanstd(x)
        scale = std if np.isfinite(std) and std > 0 else 1.0
    z = (x - med) / scale
    return np.clip(z, -10, 10)

# =======================
# ECG preprocessing
# =======================
def preprocess_ecg_matrix(sig: np.ndarray, fs: float) -> Tuple[np.ndarray, float]:
    """
    sig: (T, n_leads)
    returns: cleaned_sig (T_clean, n_leads), fs_used
    """
    if sig.ndim != 2:
        raise ValueError("Signal must be 2D with shape (T, n_leads).")

    x = np.asarray(sig, dtype=float).T  # (n_leads, T)
    sfreq = float(fs)

    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Invalid sampling frequency.")

    # Resample if needed
    if sfreq > TARGET_FS:
        down = sfreq / TARGET_FS
        x = mne.filter.resample(x, down=down, npad="auto", axis=1)
        sfreq = float(TARGET_FS)

    x_proc = np.zeros_like(x, dtype=float)

    for i in range(x.shape[0]):
        ch = x[i].astype(float)
        finite_mask = np.isfinite(ch)

        if finite_mask.sum() < 50:
            x_proc[i] = np.nan
            continue

        if not np.all(finite_mask):
            idx = np.arange(ch.size)
            ch = np.interp(idx, idx[finite_mask], ch[finite_mask])

        if sfreq > 2 * LINE_FREQ:
            ch = mne.filter.notch_filter(
                ch,
                Fs=sfreq,
                freqs=[LINE_FREQ],
                method="fir",
                phase="zero",
                verbose=False,
            )

        ch = mne.filter.filter_data(
            ch,
            sfreq=sfreq,
            l_freq=ECG_L_FREQ,
            h_freq=ECG_H_FREQ,
            method="fir",
            phase="zero",
            verbose=False,
        )

        ch = robust_zscore(ch)
        x_proc[i] = ch

    # Artifact rejection by epochs
    epoch_len = max(1, int(round(EPOCH_SEC * sfreq)))
    n_times = x_proc.shape[1]
    n_epochs = n_times // epoch_len

    if n_epochs == 0:
        raise ValueError("Signal too short after preprocessing.")

    x_proc = x_proc[:, : n_epochs * epoch_len]
    epochs = x_proc.reshape(x_proc.shape[0], n_epochs, epoch_len)
    ptp = np.ptp(epochs, axis=2)

    keep_mask = np.ones(n_epochs, dtype=bool)
    for lead in range(ptp.shape[0]):
        vals = ptp[lead]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        thr = med + 6.0 * (1.4826 * mad if mad > 0 else np.std(vals) if np.std(vals) > 0 else 1.0)
        keep_mask &= (ptp[lead] <= thr)

    if keep_mask.sum() < MIN_KEEP_EPOCHS:
        raise ValueError(f"Too few clean epochs kept ({keep_mask.sum()}).")

    cleaned = epochs[:, keep_mask, :].reshape(x_proc.shape[0], -1)
    return cleaned.T, sfreq

# =======================
# QTN / GAF / MTF
# =======================
def calculate_quantile_graph_varying_k(signal: np.ndarray, Q: int, k_values) -> np.ndarray:
    A = np.zeros((Q, Q), dtype=np.int64)
    n = int(signal.size)
    if n <= 1 or Q <= 1:
        return A

    ranks = np.argsort(np.argsort(signal))
    q_edges = np.linspace(0, n, Q + 1, dtype=int)
    loc = np.clip(np.searchsorted(q_edges, ranks, side="right") - 1, 0, Q - 1)

    for k in k_values:
        k = int(k)
        if k <= 0 or k >= n:
            continue
        for i in range(n - k):
            A[loc[i], loc[i + k]] += 1
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
        si = symbols[i]
        mtf[i, :] = trans_mat[si, symbols]
    return mtf

# =======================
# Graph / small-world
# =======================
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
        return (float(np.mean(arr)), float(np.std(arr) if len(arr) > 1 else 0.0))

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

# =======================
# Per record
# =======================
def record_to_patient_row(pn_dir: str, rec: str, method: str, seed_base: int) -> Tuple[dict, dict]:
    r = wfdb.rdrecord(rec, pn_dir=pn_dir)
    sig = slice_signals(r.p_signal)  # (T, n_leads)
    fs = float(getattr(r, "fs", np.nan))
    lead_names = getattr(r, "sig_name", [f"ch{i}" for i in range(sig.shape[1])])

    sig, fs_used = preprocess_ecg_matrix(sig, fs)

    T = sig.shape[0]
    if T < 50:
        raise ValueError("Too few samples after preprocessing.")
    Q = compute_Q_from_T(T)
    if Q < 3:
        raise ValueError(f"Q too small: {Q}")

    per_lead_metrics = []
    for li in range(sig.shape[1]):
        x = sig[:, li].astype(float)
        x = x[np.isfinite(x)]
        if x.size < 50:
            continue

        if method == "QTN":
            A = calculate_quantile_graph_varying_k(x, Q=Q, k_values=K_VALUES)
            W = (A + A.T).astype(float)
            use_abs = False
        elif method == "GAF":
            xQ = downsample_to_length(x, Q)
            W = calculate_gaf_from_lengthQ(xQ)
            use_abs = True
        elif method == "MTF":
            xQ = downsample_to_length(x, Q)
            W = calculate_mtf_from_lengthQ(xQ, Q=Q)
            use_abs = False
        else:
            raise ValueError(f"Unknown method: {method}")

        m = compute_smallworld_metrics_from_W(W, seed=seed_base + li, use_abs_for_threshold=use_abs)
        m["lead"] = lead_names[li] if li < len(lead_names) else f"ch{li}"
        per_lead_metrics.append(m)

    if not per_lead_metrics:
        raise ValueError("No valid leads produced metrics.")

    df = pd.DataFrame(per_lead_metrics).drop(columns=["lead"], errors="ignore")
    avg = df.mean(numeric_only=True).to_dict()

    info = {
        "fs_hz_original": float(fs),
        "fs_hz_used": float(fs_used),
        "n_leads": int(sig.shape[1]),
        "T_used_samples": int(T),
        "Q_used": int(Q),
    }
    return avg, info

# =======================
# IO / resume
# =======================
def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)

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
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=write_header)

def append_skips(csv_path: str, rows: List[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=write_header)

# =======================
# Parallel task
# =======================
def compute_task_record(dataset_name: str, pn_dir: str, rec: str, method: str, task_i: int):
    seed_base = RNG_SEED + 100000 * task_i + (0 if method == "QTN" else (1 if method == "GAF" else 2)) * 1000
    try:
        row, info = record_to_patient_row(
            pn_dir=pn_dir,
            rec=rec,
            method=method,
            seed_base=seed_base,
        )
        row.update({"patient_id": rec, "dataset": dataset_name, **info})
        return rec, method, row, None
    except Exception as e:
        skip = {
            "dataset": dataset_name,
            "method": method,
            "record_or_patient": rec,
            "reason": str(e),
        }
        return rec, method, None, skip

# =======================
# Runner
# =======================
def run_record_is_patient(dataset_name: str, pn_dir: str, out_dir: str):
    ensure_out_dir(out_dir)
    paths = out_paths(out_dir, dataset_name)

    records = sorted(wfdb.get_record_list(pn_dir))
    done = {m: load_done_ids(paths[m], id_col="patient_id") for m in METHODS}

    tasks = []
    task_i = 0
    for rec in records:
        for method in METHODS:
            if rec in done[method]:
                continue
            task_i += 1
            tasks.append((rec, method, task_i))

    print(f"[{dataset_name}] total records={len(records)} | pending tasks={len(tasks)} | n_jobs={N_JOBS}")

    buffer_rows = {m: [] for m in METHODS}
    buffer_skips = []

    iterator = range(0, len(tasks), BATCH_SIZE)
    if TQDM_ENABLED:
        iterator = tqdm(iterator, desc=f"{dataset_name} batches", unit="batch", leave=TQDM_LEAVE)

    for start in iterator:
        chunk = tasks[start:start + BATCH_SIZE]
        results = Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=0)(
            delayed(compute_task_record)(dataset_name, pn_dir, rec, method, task_i)
            for (rec, method, task_i) in chunk
        )

        for rec, method, row, skip in results:
            if row is not None:
                buffer_rows[method].append(row)
            if skip is not None:
                buffer_skips.append(skip)

        for method in METHODS:
            if buffer_rows[method]:
                append_rows(paths[method], buffer_rows[method], id_col="patient_id")
                buffer_rows[method].clear()

        if buffer_skips:
            append_skips(paths["SKIP"], buffer_skips)
            buffer_skips.clear()

    print(f"[{dataset_name}] done. CSVs in {out_dir}")

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    for dataset_name, cfg in DATASETS.items():
        pn_dir = cfg["pn_dir"]
        out_dir = cfg["out_dir"]
        mode = cfg["mode"]

        if mode == "record_is_patient":
            run_record_is_patient(dataset_name, pn_dir, out_dir)
        else:
            raise ValueError(f"Unknown mode: {mode}")

# In[6]:

#!/usr/bin/env python3
# ecg_qtn_gaf_mtf_smallworld_fantasia_nsrdb_fastlight_agegroups.py

import os
import re
import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
import wfdb

from scipy import signal
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =======================
# CONFIG
# =======================
DATASETS = {
    "nsrdb": {
        "pn_dir": "nsrdb",
        "out_dir": "out_nsrdb",
    },
    "fantasia": {
        "pn_dir": "fantasia",
        "out_dir": "out_fantasia",
    },
}

K_VALUES = [1, 2, 3]
TARGET_DENSITY = 0.10
N_RANDOMIZATIONS = 5
REWIRINGS_PER_EDGE = 5
RNG_SEED = 1234

MAX_SAMPLES = None
START_SAMPLE = 0

N_JOBS = 2
BATCH_SIZE = 4
BACKEND = "loky"

METHODS = ["QTN", "GAF", "MTF"]

# -----------------------
# Fast preprocessing
# -----------------------
PREPROCESS_MODE = "light"   # "none", "light"
TARGET_FS = 250.0
ECG_L_FREQ = 0.5
ECG_H_FREQ = 40.0
BUTTER_ORDER = 4

TQDM_ENABLED = True
TQDM_LEAVE = True


# =======================
# Helpers
# =======================
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
    return np.clip(z, -10, 10)

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


# =======================
# Output paths
# =======================
def out_paths_standard(out_dir: str, dataset_name: str):
    return {
        "QTN": os.path.join(out_dir, f"metrics_QTN_{dataset_name}.csv"),
        "GAF": os.path.join(out_dir, f"metrics_GAF_{dataset_name}.csv"),
        "MTF": os.path.join(out_dir, f"metrics_MTF_{dataset_name}.csv"),
        "SKIP": os.path.join(out_dir, f"skipped_{dataset_name}.csv"),
    }

def out_paths_fantasia_grouped(out_dir: str):
    paths = {
        "SKIP": os.path.join(out_dir, "skipped_fantasia.csv")
    }
    for method in METHODS:
        for group in ["all", "young", "old"]:
            paths[f"{method}_{group}"] = os.path.join(
                out_dir, f"metrics_{method}_fantasia_{group}.csv"
            )
    return paths


# =======================
# Faster ECG preprocessing
# =======================
def butter_bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    low_n = low / nyq if low is not None else None
    high_n = high / nyq if high is not None else None

    if low_n is not None and high_n is not None:
        sos = signal.butter(order, [low_n, high_n], btype="bandpass", output="sos")
    elif low_n is not None:
        sos = signal.butter(order, low_n, btype="highpass", output="sos")
    elif high_n is not None:
        sos = signal.butter(order, high_n, btype="lowpass", output="sos")
    else:
        return x

    return signal.sosfiltfilt(sos, x)

def preprocess_ecg_matrix(sig: np.ndarray, fs: float) -> Tuple[np.ndarray, float]:
    """
    sig: (T, n_leads)
    returns: cleaned_sig (T_clean, n_leads), fs_used
    """
    if sig.ndim != 2:
        raise ValueError("Signal must be 2D with shape (T, n_leads).")

    x = np.asarray(sig, dtype=float).copy()
    sfreq = float(fs)

    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Invalid sampling frequency.")

    if sfreq > TARGET_FS:
        n_target = int(round(x.shape[0] * TARGET_FS / sfreq))
        x = signal.resample(x, n_target, axis=0)
        sfreq = float(TARGET_FS)

    if PREPROCESS_MODE == "none":
        for i in range(x.shape[1]):
            ch = x[:, i]
            finite = np.isfinite(ch)
            if finite.sum() < 50:
                x[:, i] = np.nan
                continue
            if not np.all(finite):
                idx = np.arange(ch.size)
                ch = np.interp(idx, idx[finite], ch[finite])
            x[:, i] = robust_zscore(ch)
        return x, sfreq

    if PREPROCESS_MODE == "light":
        out = np.zeros_like(x, dtype=float)
        for i in range(x.shape[1]):
            ch = x[:, i]
            finite = np.isfinite(ch)

            if finite.sum() < 50:
                out[:, i] = np.nan
                continue

            if not np.all(finite):
                idx = np.arange(ch.size)
                ch = np.interp(idx, idx[finite], ch[finite])

            ch = signal.detrend(ch, type="constant")
            ch = butter_bandpass_filter(ch, sfreq, ECG_L_FREQ, ECG_H_FREQ, order=BUTTER_ORDER)
            ch = robust_zscore(ch)
            out[:, i] = ch

        return out, sfreq

    raise ValueError(f"Unknown PREPROCESS_MODE: {PREPROCESS_MODE}")


# =======================
# Fantasia age parsing
# =======================
def detect_fantasia_age_group(rec: str, record_obj=None) -> str:
    """
    Try to infer 'young' or 'old' from:
    1) comments in WFDB header
    2) record name patterns
    3) fallback: unknown
    """
    texts = []

    if record_obj is not None:
        comments = getattr(record_obj, "comments", None)
        if comments is not None:
            texts.extend([str(c).lower() for c in comments])

    texts.append(str(rec).lower())

    joined = " | ".join(texts)

    # comment-based detection
    if re.search(r"\bold\b|\belderly\b", joined):
        return "old"
    if re.search(r"\byoung\b", joined):
        return "young"

    # some PhysioNet Fantasia records are usually named with f1y/f1o style patterns in some tools,
    # so keep a pattern fallback without assuming too much
    if re.search(r"[._-]y\b|\by\d+\b|young", joined):
        return "young"
    if re.search(r"[._-]o\b|\bo\d+\b|old", joined):
        return "old"

    return "unknown"


# =======================
# QTN / GAF / MTF
# =======================
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


# =======================
# Graph / small-world
# =======================
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


# =======================
# Core: preprocess once, compute all methods
# =======================
def record_to_all_method_rows(pn_dir: str, rec: str, dataset_name: str, seed_base: int) -> Dict[str, dict]:
    r = wfdb.rdrecord(rec, pn_dir=pn_dir)
    sig = slice_signals(r.p_signal)
    fs = float(getattr(r, "fs", np.nan))

    sig, fs_used = preprocess_ecg_matrix(sig, fs)

    T = sig.shape[0]
    if T < 50:
        raise ValueError("Too few samples after preprocessing.")
    Q = compute_Q_from_T(T)
    if Q < 3:
        raise ValueError(f"Q too small: {Q}")

    age_group = "na"
    if dataset_name == "fantasia":
        age_group = detect_fantasia_age_group(rec, r)

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
    for method in METHODS:
        if not per_method_metrics[method]:
            raise ValueError(f"No valid leads produced metrics for {method}.")
        df = pd.DataFrame(per_method_metrics[method])
        avg = df.mean(numeric_only=True).to_dict()
        avg.update({
            "patient_id": rec,
            "dataset": dataset_name,
            "age_group": age_group,
            "fs_hz_original": float(fs),
            "fs_hz_used": float(fs_used),
            "n_leads": int(sig.shape[1]),
            "T_used_samples": int(T),
            "Q_used": int(Q),
        })
        out[method] = avg

    return out


# =======================
# Parallel task
# =======================
def compute_task_record_all(dataset_name: str, pn_dir: str, rec: str, task_i: int):
    seed_base = RNG_SEED + 100000 * task_i
    try:
        rows = record_to_all_method_rows(
            pn_dir=pn_dir,
            rec=rec,
            dataset_name=dataset_name,
            seed_base=seed_base
        )
        return rec, rows, None
    except Exception as e:
        skip = {
            "dataset": dataset_name,
            "method": "ALL",
            "record_or_patient": rec,
            "reason": str(e),
        }
        return rec, None, skip


# =======================
# Standard runner (NSRDB)
# =======================
def run_standard_dataset(dataset_name: str, pn_dir: str, out_dir: str):
    ensure_out_dir(out_dir)
    paths = out_paths_standard(out_dir, dataset_name)

    records = sorted(wfdb.get_record_list(pn_dir))
    done = {m: load_done_ids(paths[m], id_col="patient_id") for m in METHODS}
    done_all = done["QTN"].intersection(done["GAF"]).intersection(done["MTF"])

    tasks = []
    task_i = 0
    for rec in records:
        if rec in done_all:
            continue
        task_i += 1
        tasks.append((rec, task_i))

    print(f"[{dataset_name}] total records={len(records)} | pending records={len(tasks)} | n_jobs={N_JOBS}")

    buffer_rows = {m: [] for m in METHODS}
    buffer_skips = []

    iterator = range(0, len(tasks), BATCH_SIZE)
    if TQDM_ENABLED:
        iterator = tqdm(iterator, desc=f"{dataset_name} batches", unit="batch", leave=TQDM_LEAVE)

    for start in iterator:
        chunk = tasks[start:start + BATCH_SIZE]
        results = Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=0)(
            delayed(compute_task_record_all)(dataset_name, pn_dir, rec, task_i)
            for (rec, task_i) in chunk
        )

        for rec, rows, skip in results:
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

    print(f"[{dataset_name}] done. CSVs in {out_dir}")


# =======================
# Fantasia runner with all/young/old exports
# =======================
def run_fantasia_grouped(dataset_name: str, pn_dir: str, out_dir: str):
    ensure_out_dir(out_dir)
    paths = out_paths_fantasia_grouped(out_dir)

    # For resume, only check ALL files.
    done_qtn = load_done_ids(paths["QTN_all"], id_col="patient_id")
    done_gaf = load_done_ids(paths["GAF_all"], id_col="patient_id")
    done_mtf = load_done_ids(paths["MTF_all"], id_col="patient_id")
    done_all = done_qtn.intersection(done_gaf).intersection(done_mtf)

    records = sorted(wfdb.get_record_list(pn_dir))

    tasks = []
    task_i = 0
    for rec in records:
        if rec in done_all:
            continue
        task_i += 1
        tasks.append((rec, task_i))

    print(f"[fantasia] total records={len(records)} | pending records={len(tasks)} | n_jobs={N_JOBS}")

    buffer_rows = {f"{m}_{g}": [] for m in METHODS for g in ["all", "young", "old"]}
    buffer_skips = []

    iterator = range(0, len(tasks), BATCH_SIZE)
    if TQDM_ENABLED:
        iterator = tqdm(iterator, desc="fantasia batches", unit="batch", leave=TQDM_LEAVE)

    for start in iterator:
        chunk = tasks[start:start + BATCH_SIZE]
        results = Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=0)(
            delayed(compute_task_record_all)(dataset_name, pn_dir, rec, task_i)
            for (rec, task_i) in chunk
        )

        for rec, rows, skip in results:
            if rows is not None:
                age_group = rows["QTN"].get("age_group", "unknown")

                for method in METHODS:
                    # always append to ALL
                    buffer_rows[f"{method}_all"].append(rows[method])

                    # append to subgroup if recognized
                    if age_group in {"young", "old"}:
                        buffer_rows[f"{method}_{age_group}"].append(rows[method])

            if skip is not None:
                buffer_skips.append(skip)

        for method in METHODS:
            for group in ["all", "young", "old"]:
                key = f"{method}_{group}"
                if buffer_rows[key]:
                    append_rows(paths[key], buffer_rows[key], id_col="patient_id")
                    buffer_rows[key].clear()

        if buffer_skips:
            append_skips(paths["SKIP"], buffer_skips)
            buffer_skips.clear()

    print(f"[fantasia] done. CSVs in {out_dir}")


# =======================
# MAIN
# =======================
if __name__ == "__main__":
    for dataset_name, cfg in DATASETS.items():
        if dataset_name == "fantasia":
            run_fantasia_grouped(dataset_name, cfg["pn_dir"], cfg["out_dir"])
        else:
            run_standard_dataset(dataset_name, cfg["pn_dir"], cfg["out_dir"])

# In[ ]:
