#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto-converted from the Jupyter notebook ``ECG-3-data.ipynb``.

Code cells are reproduced verbatim. Markdown cells are kept as comments.
Jupyter shell-magic lines (if any) are commented out so the file is valid
Python; they are preserved for reference. Cell boundaries are marked with
``# In[...]`` to match the original notebook ordering.
"""


# In[ ]:

# ecg_qtn_gaf_mtf_smallworld_3datasets_per_patient_4csv.py
# Outputs 4 CSV per dataset folder: metrics_QTN, metrics_GAF, metrics_MTF, skipped log
# 1 row per patient (avg across leads). For PTB controls: avg across records per patient too.

import os
import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import wfdb

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =======================
# CONFIG
# =======================
DATASETS = {
    "nsrdb": {
        "pn_dir": "nsrdb",
        "out_dir": "out_nsrdb",
        "mode": "record_is_patient",  # one record == one patient
    },
    "fantasia": {
        "pn_dir": "fantasia",
        "out_dir": "out_fantasia",
        "mode": "record_is_patient",
    },
    "ptbdb_controls": {
        "pn_dir": "ptbdb",
        "out_dir": "out_ptbdb_controls",
        "mode": "ptb_controls_patient",  # many records per patient###
    },
}

K_VALUES = [1, 2, 3]              # QTN lags
TARGET_DENSITY = 0.10
N_RANDOMIZATIONS = 20
REWIRINGS_PER_EDGE = 5
RNG_SEED = 1234

# Full-length ECG for QTN by default.
# If you later need speed, set MAX_SAMPLES to a number.
MAX_SAMPLES = None
START_SAMPLE = 0

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
    """Resample 1D signal to length L using linear interpolation."""
    x = np.asarray(x, dtype=float)
    if L <= 1:
        return x[:1].copy()
    n = x.size
    if n == L:
        return x.copy()
    if n < 2:
        return np.full(L, float(x[0]) if n == 1 else 0.0, dtype=float)
    xp = np.linspace(0.0, 1.0, n)
    fp = x
    x_new = np.interp(np.linspace(0.0, 1.0, L), xp, fp)
    return x_new.astype(float)

# =======================
# QTN / GAF / MTF
# =======================
def calculate_quantile_graph_varying_k(signal: np.ndarray, Q: int, k_values) -> np.ndarray:
    """QTN count matrix QxQ from full signal."""
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
    """GAF on a signal already resampled to length Q -> returns QxQ."""
    x = np.asarray(signal_Q, dtype=float)
    min_val, max_val = float(np.min(x)), float(np.max(x))
    rng = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    scaled = 2 * (x - min_val) / rng - 1
    scaled = np.clip(scaled, -1, 1)
    phi = np.arccos(scaled)
    return np.cos(phi[:, None] + phi[None, :])

def calculate_mtf_from_lengthQ(signal_Q: np.ndarray, Q: int) -> np.ndarray:
    """MTF on a signal already resampled to length Q -> returns QxQ."""
    x = np.asarray(signal_Q, dtype=float)
    min_val, max_val = float(np.min(x)), float(np.max(x))
    rng = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    norm_signal = (x - min_val) / rng  # in [0,1]

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
        # vectorized fill per row:
        mtf[i, :] = trans_mat[si, symbols]
    return mtf

# =======================
# Graph / small-world metrics (size-robust)
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
            Cr.append(C); Lr.append(L)

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
# Per record: compute metrics per lead then average -> 1 row per record
# =======================
def record_to_patient_row(pn_dir: str, rec: str, method: str, seed_base: int) -> Tuple[dict, dict]:
    r = wfdb.rdrecord(rec, pn_dir=pn_dir)
    sig = slice_signals(r.p_signal)  # (T, n_leads)
    fs = getattr(r, "fs", np.nan)
    lead_names = getattr(r, "sig_name", [f"ch{i}" for i in range(sig.shape[1])])

    T = sig.shape[0]
    if T < 50:
        raise ValueError("Too few samples after slicing.")
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
            # full-length transitions into Q bins
            A = calculate_quantile_graph_varying_k(x, Q=Q, k_values=K_VALUES)
            W = (A + A.T).astype(float)
            use_abs = False
        elif method == "GAF":
            # downsample to length Q, then compute GAF(QxQ)
            xQ = downsample_to_length(x, Q)
            W = calculate_gaf_from_lengthQ(xQ)
            use_abs = True
        elif method == "MTF":
            # downsample to length Q, then compute MTF(QxQ)
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
        "fs_hz": float(fs) if fs is not None else np.nan,
        "n_leads": int(sig.shape[1]),
        "T_used_samples": int(T),
        "Q_used": int(Q),
    }
    return avg, info

# =======================
# PTB controls list (local-friendly: uses wfdb directly if possible; else hard-fail with guidance)
# =======================
def fetch_ptb_controls_list_via_web_is_not_used() -> List[str]:
    """
    We avoid web fetching here.
    If wfdb can't access the CONTROLS list directly, you can manually download CONTROLS
    and point to a local file. To keep this script self-contained, we try a best effort:
    - Attempt to load a local file named 'PTB_CONTROLS.txt' if present.
    """
    local = "PTB_CONTROLS.txt"
    if os.path.exists(local):
        lines = []
        with open(local, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and re.match(r"^patient\d{3}/", s):
                    lines.append(s)
        if lines:
            return lines
    raise RuntimeError(
        "PTB controls list not found.\n"
        "Create a local file 'PTB_CONTROLS.txt' containing lines like 'patient104/s0306lre' "
        "from the PhysioNet PTBDB CONTROLS file, then rerun."
    )

def ptb_patient_id_from_record_path(rec_path: str) -> str:
    return rec_path.split("/")[0]

# =======================
# Runner per dataset -> 4 CSV files
# =======================
def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_dataset_outputs(out_dir: str, dataset_name: str,
                         df_qtn: pd.DataFrame, df_gaf: pd.DataFrame, df_mtf: pd.DataFrame,
                         skipped: pd.DataFrame):
    ensure_out_dir(out_dir)

    qtn_path = os.path.join(out_dir, f"metrics_QTN_{dataset_name}.csv")
    gaf_path = os.path.join(out_dir, f"metrics_GAF_{dataset_name}.csv")
    mtf_path = os.path.join(out_dir, f"metrics_MTF_{dataset_name}.csv")
    skp_path = os.path.join(out_dir, f"skipped_{dataset_name}.csv")

    df_qtn.to_csv(qtn_path)
    df_gaf.to_csv(gaf_path)
    df_mtf.to_csv(mtf_path)
    skipped.to_csv(skp_path, index=False)

    print("Saved:", qtn_path, df_qtn.shape)
    print("Saved:", gaf_path, df_gaf.shape)
    print("Saved:", mtf_path, df_mtf.shape)
    print("Saved:", skp_path, skipped.shape)

def run_record_is_patient(dataset_name: str, pn_dir: str, out_dir: str):
    records = wfdb.get_record_list(pn_dir)

    skipped_rows = []
    outputs = {"QTN": [], "GAF": [], "MTF": []}

    for i, rec in enumerate(sorted(records), start=1):
        for method in ["QTN", "GAF", "MTF"]:
            try:
                row, info = record_to_patient_row(
                    pn_dir=pn_dir, rec=rec, method=method, seed_base=RNG_SEED + 100000 * i + (0 if method=="QTN" else (1 if method=="GAF" else 2))*1000
                )
                row.update({"patient_id": rec, "dataset": dataset_name, **info})
                outputs[method].append(row)
                print(f"[{dataset_name} OK] {rec} {method}")
            except Exception as e:
                skipped_rows.append({"dataset": dataset_name, "method": method, "record_or_patient": rec, "reason": str(e)})
                print(f"[{dataset_name} SKIP] {rec} {method}: {e}")

    def to_df(rows: List[dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("patient_id").sort_index()

    df_qtn = to_df(outputs["QTN"])
    df_gaf = to_df(outputs["GAF"])
    df_mtf = to_df(outputs["MTF"])
    skipped = pd.DataFrame(skipped_rows)

    save_dataset_outputs(out_dir, dataset_name, df_qtn, df_gaf, df_mtf, skipped)

def run_ptb_controls_patient(dataset_name: str, pn_dir: str, out_dir: str):
    # Read controls list from local PTB_CONTROLS.txt (see function for details)
    controls = fetch_ptb_controls_list_via_web_is_not_used()

    skipped_rows = []
    # For each method, we first compute per-record rows (already averaged across leads),
    # then average across records per patient###
    per_record = {"QTN": [], "GAF": [], "MTF": []}

    for i, rec_path in enumerate(controls, start=1):
        patient_id = ptb_patient_id_from_record_path(rec_path)
        for method in ["QTN", "GAF", "MTF"]:
            try:
                row, info = record_to_patient_row(
                    pn_dir=pn_dir, rec=rec_path, method=method, seed_base=RNG_SEED + 200000 * i + (0 if method=="QTN" else (1 if method=="GAF" else 2))*1000
                )
                row.update({
                    "patient_id": patient_id,
                    "record_path": rec_path,
                    "dataset": dataset_name,
                    **info
                })
                per_record[method].append(row)
                print(f"[{dataset_name} OK] {rec_path} {method}")
            except Exception as e:
                skipped_rows.append({"dataset": dataset_name, "method": method, "record_or_patient": rec_path, "reason": str(e)})
                print(f"[{dataset_name} SKIP] {rec_path} {method}: {e}")

    def aggregate_per_patient(rows: List[dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # numeric metrics + also keep a few informative averages
        # (dataset is constant, record_path is not aggregated)
        drop_cols = ["record_path"]
        metric_cols = [c for c in df.columns if c not in drop_cols]
        df2 = df[metric_cols].copy()
        # groupby patient_id and mean numeric columns
        g = df2.groupby("patient_id").mean(numeric_only=True)
        # add dataset column back (non-numeric)
        g.insert(0, "dataset", dataset_name)
        return g.sort_index()

    df_qtn = aggregate_per_patient(per_record["QTN"])
    df_gaf = aggregate_per_patient(per_record["GAF"])
    df_mtf = aggregate_per_patient(per_record["MTF"])
    skipped = pd.DataFrame(skipped_rows)

    save_dataset_outputs(out_dir, dataset_name, df_qtn, df_gaf, df_mtf, skipped)

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
        elif mode == "ptb_controls_patient":
            run_ptb_controls_patient(dataset_name, pn_dir, out_dir)
        else:
            raise ValueError(f"Unknown mode: {mode}")

# In[5]:

#!/usr/bin/env python3
# ecg_qtn_gaf_mtf_smallworld_3datasets_per_patient_4csv_resume_parallel_tqdm_safeio.py
# - Parallel with limited workers (N_JOBS)
# - Resume if CSV already exists
# - Incremental saving (batch append)
# - TQDM progress bars
# - Robust PhysioNet handling:
#     * Prefer local .hea scanning (no internet)
#     * If not local, retry get_record_list + rdrecord with backoff
#     * Cache record lists per dataset in out_dir/records_<dataset>.txt
# - Outputs 4 CSV per dataset folder:
#     metrics_QTN_<dataset>.csv, metrics_GAF_<dataset>.csv, metrics_MTF_<dataset>.csv, skipped_<dataset>.csv
#
# Notes:
# - If you are still reading remotely from PhysioNet, set N_JOBS=1 to reduce connection resets.
# - Best stability: download datasets once (wfdb.dl_database) and run offline.

import os
import re
import time
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
import wfdb

from joblib import Parallel, delayed
from tqdm import tqdm

from requests.exceptions import ConnectionError as RequestsConnectionError
from urllib3.exceptions import ProtocolError

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =======================
# CONFIG
# =======================
DATASETS = {
    "nsrdb": {
        "pn_dir": "nsrdb",
        "out_dir": "out_nsrdb",
        "mode": "record_is_patient",  # one record == one patient
    },
    "fantasia": {
        "pn_dir": "fantasia",
        "out_dir": "out_fantasia",
        "mode": "record_is_patient",
    },
    "ptbdb_controls": {
        "pn_dir": "ptbdb",
        "out_dir": "out_ptbdb_controls",
        "mode": "ptb_controls_patient",  # many records per patient
    },
}

K_VALUES = [1, 2, 3]        # QTN lags
TARGET_DENSITY = 0.10
N_RANDOMIZATIONS = 20
REWIRINGS_PER_EDGE = 5
RNG_SEED = 1234

MAX_SAMPLES = None          # set e.g. 20000 for speed/debug
START_SAMPLE = 0

# -----------------------
# Parallel + resume knobs
# -----------------------
N_JOBS = 4                 # if remote PhysioNet -> set to 1
BATCH_SIZE = 10             # flush to disk every BATCH_SIZE tasks
BACKEND = "loky"
METHODS = ["QTN", "GAF", "MTF"]

# -----------------------
# Progress bars
# -----------------------
TQDM_ENABLED = True
TQDM_LEAVE = True


# =======================
# Robust PhysioNet IO helpers
# =======================
def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_local_records_from_hea(pn_dir: str) -> List[str]:
    """
    If pn_dir is a local folder containing .hea files, return record names
    in the format expected by wfdb.rdrecord(rec, pn_dir=pn_dir).
    """
    root = Path(pn_dir)
    if not root.exists():
        return []
    hea_files = sorted(root.rglob("*.hea"))
    records: List[str] = []
    for hf in hea_files:
        rel = hf.relative_to(root).with_suffix("")  # strip .hea
        records.append(str(rel).replace("\\", "/"))
    return records


def get_record_list_safe(pn_dir: str, cache_file: str, max_retries: int = 10, base_sleep: float = 1.0) -> List[str]:
    """
    Try:
      1) cache file
      2) local .hea scan
      3) wfdb.get_record_list (remote) with retries + backoff
    """
    # 1) Cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                recs = [line.strip() for line in f if line.strip()]
            if recs:
                return recs
        except Exception:
            pass

    # 2) Local scan
    local_recs = list_local_records_from_hea(pn_dir)
    if local_recs:
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write("\n".join(local_recs) + "\n")
        except Exception:
            pass
        return local_recs

    # 3) Remote with retries
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            recs = sorted(wfdb.get_record_list(pn_dir))
            if recs:
                try:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        f.write("\n".join(recs) + "\n")
                except Exception:
                    pass
            return recs
        except (RequestsConnectionError, ProtocolError, ConnectionResetError, OSError) as e:
            last_err = e
            sleep_s = base_sleep * (2 ** (attempt - 1)) + random.random()
            print(f"[WARN] get_record_list failed for '{pn_dir}' (attempt {attempt}/{max_retries}): {e}")
            print(f"       sleeping {sleep_s:.1f}s then retry...")
            time.sleep(sleep_s)

    raise RuntimeError(f"Could not fetch record list for '{pn_dir}' after {max_retries} retries. Last error: {last_err}")


def rdrecord_safe(rec: str, pn_dir: str, max_retries: int = 8, base_sleep: float = 0.8):
    """
    wfdb.rdrecord with retry/backoff (handles PhysioNet connection resets).
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return wfdb.rdrecord(rec, pn_dir=pn_dir)
        except (RequestsConnectionError, ProtocolError, ConnectionResetError, OSError) as e:
            last_err = e
            sleep_s = base_sleep * (2 ** (attempt - 1)) + random.random()
            print(f"[WARN] rdrecord failed for {rec} (attempt {attempt}/{max_retries}): {e}")
            print(f"       sleeping {sleep_s:.1f}s then retry...")
            time.sleep(sleep_s)

    raise RuntimeError(f"Could not read record '{rec}' from '{pn_dir}' after {max_retries} retries. Last error: {last_err}")


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
    norm_signal = (x - min_val) / rng  # [0,1]

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
# Graph / small-world metrics
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
    phi = 1.0 - np.sqrt((dC ** 2 + dL ** 2) / 2.0)
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
# Per record -> avg across leads -> 1 dict
# =======================
def record_to_patient_row(pn_dir: str, rec: str, method: str, seed_base: int) -> Tuple[dict, dict]:
    r = rdrecord_safe(rec, pn_dir=pn_dir)
    sig = slice_signals(r.p_signal)  # (T, n_leads)
    fs = getattr(r, "fs", np.nan)
    lead_names = getattr(r, "sig_name", [f"ch{i}" for i in range(sig.shape[1])])

    T = sig.shape[0]
    if T < 50:
        raise ValueError("Too few samples after slicing.")
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
        per_lead_metrics.append(m)

    if not per_lead_metrics:
        raise ValueError("No valid leads produced metrics.")

    df = pd.DataFrame(per_lead_metrics)
    avg = df.mean(numeric_only=True).to_dict()

    info = {
        "fs_hz": float(fs) if fs is not None else np.nan,
        "n_leads": int(sig.shape[1]),
        "T_used_samples": int(T),
        "Q_used": int(Q),
    }
    return avg, info


# =======================
# PTB controls list (local file)
# =======================
def fetch_ptb_controls_list_local() -> List[str]:
    local = "PTB_CONTROLS.txt"
    if os.path.exists(local):
        lines: List[str] = []
        with open(local, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and re.match(r"^patient\d{3}/", s):
                    lines.append(s)
        if lines:
            return lines
    raise RuntimeError(
        "PTB controls list not found.\n"
        "Create local 'PTB_CONTROLS.txt' with lines like 'patient104/s0306lre' (from PTBDB CONTROLS file)."
    )


def ptb_patient_id_from_record_path(rec_path: str) -> str:
    return rec_path.split("/")[0]


# =======================
# Resume + incremental IO
# =======================
def out_paths(out_dir: str, dataset_name: str):
    return {
        "QTN": os.path.join(out_dir, f"metrics_QTN_{dataset_name}.csv"),
        "GAF": os.path.join(out_dir, f"metrics_GAF_{dataset_name}.csv"),
        "MTF": os.path.join(out_dir, f"metrics_MTF_{dataset_name}.csv"),
        "SKIP": os.path.join(out_dir, f"skipped_{dataset_name}.csv"),
        "RECORDS_CACHE": os.path.join(out_dir, f"records_{dataset_name}.txt"),
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
# Parallel tasks
# =======================
def compute_task_record(dataset_name: str, pn_dir: str, rec: str, method: str, task_i: int) -> Tuple[str, str, Optional[dict], Optional[dict]]:
    seed_base = RNG_SEED + 100000 * task_i + (0 if method == "QTN" else (1 if method == "GAF" else 2)) * 1000
    try:
        row, info = record_to_patient_row(pn_dir=pn_dir, rec=rec, method=method, seed_base=seed_base)
        row.update({"patient_id": rec, "dataset": dataset_name, **info})
        return rec, method, row, None
    except Exception as e:
        skip = {"dataset": dataset_name, "method": method, "record_or_patient": rec, "reason": str(e)}
        return rec, method, None, skip


def compute_task_ptb_record(dataset_name: str, pn_dir: str, rec_path: str, method: str, task_i: int) -> Tuple[str, str, Optional[dict], Optional[dict]]:
    seed_base = RNG_SEED + 200000 * task_i + (0 if method == "QTN" else (1 if method == "GAF" else 2)) * 1000
    patient_id = ptb_patient_id_from_record_path(rec_path)
    try:
        row, info = record_to_patient_row(pn_dir=pn_dir, rec=rec_path, method=method, seed_base=seed_base)
        row.update({"patient_id": patient_id, "record_path": rec_path, "dataset": dataset_name, **info})
        return rec_path, method, row, None
    except Exception as e:
        skip = {"dataset": dataset_name, "method": method, "record_or_patient": rec_path, "reason": str(e)}
        return rec_path, method, None, skip


# =======================
# Runners
# =======================
def run_record_is_patient(dataset_name: str, pn_dir: str, out_dir: str):
    ensure_out_dir(out_dir)
    paths = out_paths(out_dir, dataset_name)

    records = get_record_list_safe(pn_dir, cache_file=paths["RECORDS_CACHE"])
    done = {m: load_done_ids(paths[m], id_col="patient_id") for m in METHODS}

    tasks: List[Tuple[str, str, int]] = []
    task_i = 0
    for rec in records:
        for method in METHODS:
            if rec in done[method]:
                continue
            task_i += 1
            tasks.append((rec, method, task_i))

    print(f"[{dataset_name}] total records={len(records)} | pending tasks={len(tasks)} | n_jobs={N_JOBS}")

    buffer_rows = {m: [] for m in METHODS}
    buffer_skips: List[dict] = []

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

        # flush
        for method in METHODS:
            if buffer_rows[method]:
                append_rows(paths[method], buffer_rows[method], id_col="patient_id")
                buffer_rows[method].clear()
        if buffer_skips:
            append_skips(paths["SKIP"], buffer_skips)
            buffer_skips.clear()

    print(f"[{dataset_name}] done. CSVs in {out_dir}")


def run_ptb_controls_patient(dataset_name: str, pn_dir: str, out_dir: str):
    ensure_out_dir(out_dir)
    paths = out_paths(out_dir, dataset_name)

    controls = fetch_ptb_controls_list_local()
    done_patients = {m: load_done_ids(paths[m], id_col="patient_id") for m in METHODS}

    patient_to_records: Dict[str, List[str]] = {}
    for rec_path in controls:
        pid = ptb_patient_id_from_record_path(rec_path)
        patient_to_records.setdefault(pid, []).append(rec_path)

    pending_patient_ids = {
        m: sorted([pid for pid in patient_to_records.keys() if pid not in done_patients[m]])
        for m in METHODS
    }

    print(
        f"[{dataset_name}] patients={len(patient_to_records)} | "
        f"pending QTN={len(pending_patient_ids['QTN'])}, "
        f"GAF={len(pending_patient_ids['GAF'])}, "
        f"MTF={len(pending_patient_ids['MTF'])} | n_jobs={N_JOBS}"
    )

    skipped_rows_all: List[dict] = []

    for method in METHODS:
        pids = pending_patient_ids[method]
        if not pids:
            continue

        rec_tasks: List[Tuple[str, int]] = []
        task_i = 0
        for pid in pids:
            for rec_path in patient_to_records[pid]:
                task_i += 1
                rec_tasks.append((rec_path, task_i))

        per_record_rows: List[dict] = []
        iterator = range(0, len(rec_tasks), BATCH_SIZE)
        if TQDM_ENABLED:
            iterator = tqdm(iterator, desc=f"{dataset_name}:{method} batches", unit="batch", leave=TQDM_LEAVE)

        for start in iterator:
            chunk = rec_tasks[start:start + BATCH_SIZE]
            results = Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=0)(
                delayed(compute_task_ptb_record)(dataset_name, pn_dir, rec_path, method, task_i)
                for (rec_path, task_i) in chunk
            )
            for rec_path, _, row, skip in results:
                if row is not None:
                    per_record_rows.append(row)
                if skip is not None:
                    skipped_rows_all.append(skip)

        if not per_record_rows:
            continue

        df = pd.DataFrame(per_record_rows)
        drop_cols = ["record_path"]
        metric_cols = [c for c in df.columns if c not in drop_cols]
        df2 = df[metric_cols].copy()

        g = df2.groupby("patient_id").mean(numeric_only=True)
        g.insert(0, "dataset", dataset_name)
        g = g.reset_index()

        append_rows(paths[method], g.to_dict(orient="records"), id_col="patient_id")

    if skipped_rows_all:
        append_skips(paths["SKIP"], skipped_rows_all)

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
        elif mode == "ptb_controls_patient":
            run_ptb_controls_patient(dataset_name, pn_dir, out_dir)
        else:
            raise ValueError(f"Unknown mode: {mode}")

# In[ ]:
