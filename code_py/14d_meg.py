#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto-converted from the Jupyter notebook ``MEG.ipynb``.

Code cells are reproduced verbatim. Markdown cells are kept as comments.
Jupyter shell-magic lines (if any) are commented out so the file is valid
Python; they are preserved for reference. Cell boundaries are marked with
``# In[...]`` to match the original notebook ordering.
"""


# In[1]:

# =========================
# MEG ds000117 (OpenNeuro) -> QTN/GAF/MTF small-world
# Jupyter-friendly (NO argparse)
# Temp download per-file -> delete -> only export CSV
# =========================

import os, re, io, json, shutil, tempfile, warnings, subprocess
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
# CONFIG
# ============================================================
DATASET_ID = "ds000117"
DATASET_TAG = None  # None => auto-detect latest snapshot tag via GraphQL (recommended)

# GraphQL endpoint documented by OpenNeuro
OPENNEURO_GQL = "https://openneuro.org/crn/graphql"

# Output
DATASET_NAME = "ds000117_meg"
OUT_DIR = "meg_outputs/out_ds000117_meg_v1"

# Temporary download behavior
DELETE_RAW_AFTER_PROCESS = True
REQUEST_TIMEOUT = 120

# Process control
N_JOBS = 2
BATCH_SIZE = 2
BACKEND = "loky"

METHODS = ["QTN", "GAF", "MTF"]
TQDM_ENABLED = True
TQDM_LEAVE = True

# ============================================================
# MEG signal extraction parameters (keep it manageable)
# ============================================================
# Use only a limited duration per run to avoid RAM blowups.
# If you want the entire run, set MAX_SECONDS=None (can be heavy).
MAX_SECONDS = 60.0

# Optional resampling (done on extracted segment via scipy, not MNE resample)
TARGET_FS = 250.0   # set None to keep native fs

# Channel selection
PICK_MAG = True
PICK_GRAD = True

# ============================================================
# QTN / SMALL-WORLD parameters
# ============================================================
K_VALUES = [1, 2, 3]
TARGET_DENSITY = 0.10
N_RANDOMIZATIONS = 10
REWIRINGS_PER_EDGE = 5
RNG_SEED = 1234

MAX_ABS_Z = 10.0

# QC thresholds (generic)
MIN_VALID_SAMPLES = 500
MIN_FINITE_FRAC = 0.90
MIN_STD = 1e-12
MIN_UNIQUE_VALUES = 20

# ============================================================
# Helpers (filesystem / csv)
# ============================================================
def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)

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

def aggregate_per_subject(file_csv: str, out_csv: str):
    if not os.path.exists(file_csv):
        return
    df = pd.read_csv(file_csv)
    if df.empty or "subject_id" not in df.columns:
        return
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    g = df.groupby("subject_id")[numeric_cols].mean().reset_index()
    if "dataset" in df.columns:
        g.insert(1, "dataset", df.groupby("subject_id")["dataset"].first().values)
    g["n_files_used"] = df.groupby("subject_id")["file_id"].nunique().values
    g["aggregation"] = "mean_over_files"
    g.to_csv(out_csv, index=False)

# ============================================================
# OpenNeuro GraphQL: find latest tag + list files recursively
# ============================================================
def gql(query: str, variables: Optional[dict] = None) -> dict:
    payload = {"query": query, "variables": variables or {}}
    r = requests.post(OPENNEURO_GQL, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    out = r.json()
    if "errors" in out:
        raise RuntimeError(f"GraphQL errors: {out['errors']}")
    return out["data"]

def get_latest_snapshot_tag(dataset_id: str) -> str:
    # Try a couple of common fields to be robust to schema changes
    q = """
    query ($id: ID!) {
      dataset(id: $id) {
        id
        latestSnapshot { tag }
        snapshots { tag }
      }
    }
    """
    data = gql(q, {"id": dataset_id})
    ds = data.get("dataset") or {}
    if ds.get("latestSnapshot") and ds["latestSnapshot"].get("tag"):
        return ds["latestSnapshot"]["tag"]
    # fallback: take last snapshots tag if present
    snaps = ds.get("snapshots") or []
    tags = [s.get("tag") for s in snaps if s and s.get("tag")]
    if tags:
        return tags[-1]
    raise RuntimeError("Could not determine latest snapshot tag from GraphQL response.")

def snapshot_files(dataset_id: str, tag: str, tree: Optional[str] = None) -> List[dict]:
    q = """
    query ($datasetId: ID!, $tag: String!, $tree: String) {
      snapshot(datasetId: $datasetId, tag: $tag) {
        files(tree: $tree) {
          id
          key
          filename
          size
          directory
          annexed
        }
      }
    }
    """
    data = gql(q, {"datasetId": dataset_id, "tag": tag, "tree": tree})
    snap = data.get("snapshot") or {}
    return snap.get("files") or []

def list_all_files(dataset_id: str, tag: str) -> List[str]:
    # Recursively walk git-tree objects using the "key" of directories (as per docs)
    all_paths = []

    def walk(prefix: str, tree_key: Optional[str]):
        items = snapshot_files(dataset_id, tag, tree=tree_key)
        for it in items:
            name = it["filename"]
            is_dir = bool(it["directory"])
            key = it.get("key")
            if is_dir:
                walk(prefix + name + "/", key)
            else:
                all_paths.append(prefix + name)

    walk(prefix="", tree_key=None)
    return sorted(set(all_paths))

# ============================================================
# Minimal downloader: openneuro-py for ONE file into a temp dir
# ============================================================
def ensure_openneuro_py():
    try:
        subprocess.run(["openneuro-py", "--help"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        # try install
        subprocess.run([os.sys.executable, "-m", "pip", "install", "-U", "openneuro-py"], check=True)

def download_one_openneuro_file(dataset: str, include_path: str, target_dir: str, tag: Optional[str] = None):
    ensure_openneuro_py()
    cmd = ["openneuro-py", "download", "--dataset", dataset, "--include", include_path, "--target_dir", target_dir]
    if tag is not None:
        # openneuro-py historically uses --snapshot or --tag depending on version;
        # we attempt --snapshot first, then --tag fallback.
        tried = []
        for opt in ("--snapshot", "--tag"):
            try:
                subprocess.run(cmd + [opt, tag], check=True)
                return
            except Exception as e:
                tried.append((opt, str(e)))
        raise RuntimeError(f"openneuro-py failed with snapshot/tag options tried={tried}")
    subprocess.run(cmd, check=True)

# ============================================================
# Signal processing: extract MEG time series segment
# ============================================================
def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        std = np.nanstd(x)
        scale = std if np.isfinite(std) and std > 0 else 1.0
    z = (x - med) / scale
    return np.clip(z, -MAX_ABS_Z, MAX_ABS_Z)

def preprocess_generic_signal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size < MIN_VALID_SAMPLES:
        raise ValueError(f"Too few samples (n={x.size}).")
    finite = np.isfinite(x)
    if finite.mean() < MIN_FINITE_FRAC:
        raise ValueError(f"Too many NaNs: finite_frac={finite.mean():.3f}")
    x = x.copy()
    x[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), x[finite])
    x = signal.detrend(x, type="constant")
    if np.std(x) <= MIN_STD:
        raise ValueError("Variance too small.")
    if np.unique(np.round(x, 12)).size < MIN_UNIQUE_VALUES:
        raise ValueError("Too few unique values.")
    return robust_zscore(x)

def resample_1d(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if fs_out is None or fs_in is None or not np.isfinite(fs_in) or not np.isfinite(fs_out) or fs_out <= 0 or fs_in <= 0:
        return x
    if abs(fs_out - fs_in) / fs_in < 1e-6:
        return x
    n_out = int(round(len(x) * (fs_out / fs_in)))
    n_out = max(10, n_out)
    return signal.resample(x, n_out)

def load_meg_fif_segment(fif_path: str) -> Tuple[np.ndarray, float, List[str]]:
    import mne

    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
    fs = float(raw.info["sfreq"])

    picks = []
    if PICK_MAG:
        picks += mne.pick_types(raw.info, meg="mag", eeg=False, stim=False, eog=False, ecg=False, misc=False)
    if PICK_GRAD:
        picks += mne.pick_types(raw.info, meg="grad", eeg=False, stim=False, eog=False, ecg=False, misc=False)
    picks = np.unique(np.array(picks, dtype=int))

    if picks.size == 0:
        raise ValueError("No MEG mag/grad channels found.")

    n_samples = raw.n_times
    if MAX_SECONDS is None:
        start, stop = 0, n_samples
    else:
        stop = min(n_samples, int(round(MAX_SECONDS * fs)))
        start = 0
    if stop - start < MIN_VALID_SAMPLES:
        raise ValueError(f"Too short after MAX_SECONDS slicing: n={stop-start}")

    data = raw.get_data(picks=picks, start=start, stop=stop)  # shape (n_ch, T)
    ch_names = [raw.ch_names[p] for p in picks]

    # preprocess per channel, then stack as (T, n_ch)
    cleaned = []
    kept_names = []
    for i in range(data.shape[0]):
        x = data[i, :]
        if TARGET_FS is not None:
            x = resample_1d(x, fs_in=fs, fs_out=TARGET_FS)
        x = preprocess_generic_signal(x)
        cleaned.append(x)
        kept_names.append(ch_names[i])

    min_len = min(len(x) for x in cleaned)
    X = np.column_stack([x[:min_len] for x in cleaned])  # (T, n_ch)
    fs_used = float(TARGET_FS) if TARGET_FS is not None else fs
    return X, fs_used, kept_names

# ============================================================
# QTN / GAF / MTF
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
# Core: MEG file -> metrics (average over sensors)
# ============================================================
def file_to_all_method_rows_meg(fif_path: str, seed_base: int) -> Dict[str, dict]:
    X, fs_used, ch_names = load_meg_fif_segment(fif_path)  # (T, n_ch)
    T, n_ch = X.shape
    if T < MIN_VALID_SAMPLES:
        raise ValueError("Too few samples after extraction/preproc.")
    Q = compute_Q_from_T(T)
    if Q < 3:
        raise ValueError(f"Q too small: {Q}")

    per_method_metrics = {m: [] for m in METHODS}

    for li in range(n_ch):
        x = X[:, li].astype(float)
        x = x[np.isfinite(x)]
        if x.size < MIN_VALID_SAMPLES:
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
            raise ValueError(f"No valid sensors produced metrics for {method}.")
        dfm = pd.DataFrame(per_method_metrics[method])
        avg = dfm.mean(numeric_only=True).to_dict()
        avg.update({
            "fs_hz": float(fs_used),
            "n_sensors": int(n_ch),
            "T_used_samples": int(T),
            "Q_used": int(Q),
            "sensor_names": ";".join(ch_names[:50]) + (";..." if len(ch_names) > 50 else ""),
            "max_seconds": float(MAX_SECONDS) if MAX_SECONDS is not None else np.nan,
            "target_fs": float(TARGET_FS) if TARGET_FS is not None else np.nan,
            "pick_mag": bool(PICK_MAG),
            "pick_grad": bool(PICK_GRAD),
        })
        out[method] = avg
    return out

# ============================================================
# Identify subject/file IDs from BIDS paths
# ============================================================
def subject_from_bids_path(p: str) -> str:
    m = re.search(r"(sub-[a-zA-Z0-9]+)", p)
    return m.group(1) if m else "unknown"

def file_id_from_bids_path(p: str) -> str:
    return Path(p).name.rsplit(".", 1)[0]  # strip extension

# ============================================================
# One task: download one fif into temp dir -> process -> delete
# ============================================================
def compute_task_meg(bids_path: str, tag: str, task_i: int):
    subject_id = subject_from_bids_path(bids_path)
    file_id = file_id_from_bids_path(bids_path)

    try:
        with tempfile.TemporaryDirectory(prefix="tmp_ds000117_meg_") as td:
            # Download exactly one file
            download_one_openneuro_file(DATASET_ID, bids_path, target_dir=td, tag=tag)

            local_fif = os.path.join(td, bids_path)
            if not os.path.exists(local_fif):
                # sometimes openneuro-py may place under dataset root folder; try to locate
                candidates = list(Path(td).rglob(Path(bids_path).name))
                if not candidates:
                    raise FileNotFoundError(f"Downloaded file not found for {bids_path}")
                local_fif = str(candidates[0])

            rows = file_to_all_method_rows_meg(local_fif, seed_base=RNG_SEED + 100000 * task_i)

            for method in METHODS:
                rows[method].update({
                    "file_id": file_id,
                    "subject_id": subject_id,
                    "dataset": DATASET_NAME,
                    "source_file": bids_path,
                    "snapshot_tag": tag,
                })

        return file_id, rows, None

    except Exception as e:
        skip = {
            "dataset": DATASET_NAME,
            "method": "ALL",
            "record_or_subject": bids_path,
            "file_id": file_id,
            "subject_id": subject_id,
            "reason": str(e),
        }
        return file_id, None, skip

# ============================================================
# Main runner
# ============================================================
def run_meg_ds000117(mode: str = "run"):
    assert mode in ("manifest", "run")
    ensure_out_dir(OUT_DIR)
    paths = out_paths(OUT_DIR, DATASET_NAME)

    # Determine snapshot tag
    tag = DATASET_TAG or get_latest_snapshot_tag(DATASET_ID)
    print(f"[{DATASET_NAME}] Using snapshot tag: {tag}")

    # List all files, filter to MEG FIF runs
    all_files = list_all_files(DATASET_ID, tag)
    meg_fifs = [p for p in all_files if p.endswith("_meg.fif") and "/ses-meg/" in p and "/meg/" in p]
    meg_fifs = sorted(meg_fifs)

    if not meg_fifs:
        raise RuntimeError("No MEG FIF files found with filter *_meg.fif under ses-meg/meg.")

    manifest_rows = [{
        "source_file": p,
        "file_id": file_id_from_bids_path(p),
        "subject_id": subject_from_bids_path(p),
        "snapshot_tag": tag,
    } for p in meg_fifs]
    pd.DataFrame(manifest_rows).to_csv(paths["MANIFEST"], index=False)
    print(f"[{DATASET_NAME}] wrote manifest: {paths['MANIFEST']} (n_files={len(meg_fifs)} | n_subjects={pd.DataFrame(manifest_rows)['subject_id'].nunique()})")

    if mode == "manifest":
        print("[MODE] manifest only.")
        return paths

    done_qtn = load_done_ids(paths["QTN_FILE"], "file_id")
    done_gaf = load_done_ids(paths["GAF_FILE"], "file_id")
    done_mtf = load_done_ids(paths["MTF_FILE"], "file_id")
    done_all = done_qtn.intersection(done_gaf).intersection(done_mtf)

    tasks = []
    task_i = 0
    for p in meg_fifs:
        fid = file_id_from_bids_path(p)
        if fid in done_all:
            continue
        task_i += 1
        tasks.append((p, task_i))

    print(f"[{DATASET_NAME}] pending={len(tasks)} | n_jobs={N_JOBS} | MAX_SECONDS={MAX_SECONDS} | TARGET_FS={TARGET_FS}")

    buffer_rows = {m: [] for m in METHODS}
    buffer_skips = []

    iterator = range(0, len(tasks), BATCH_SIZE)
    if TQDM_ENABLED:
        iterator = tqdm(iterator, desc=f"{DATASET_NAME} batches", unit="batch", leave=TQDM_LEAVE)

    for start in iterator:
        chunk = tasks[start:start + BATCH_SIZE]
        results = Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=0)(
            delayed(compute_task_meg)(bids_path, tag, ti) for (bids_path, ti) in chunk
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
# paths = run_meg_ds000117("manifest")  # just list files
paths = run_meg_ds000117("run")        # download per-file temporarily, compute, export CSVs
print(paths)

# In[6]:

# =========================
# MEG ds000117 (OpenNeuro) -> QTN/GAF/MTF small-world
# Jupyter-friendly (NO argparse)
# TEMP per-file download -> process -> delete -> ONLY export CSVs
#
# IMPORTANT:
# - Your openneuro-py has NO `ls` command.
# - We list candidate files via OpenNeuro GraphQL, then attempt per-file download.
# - If a path cannot be downloaded (metadata mismatch), we skip+log and continue.
#
# NOTE:
# - Progress bar is PER FILE (updates after each file finishes).
# - N_JOBS=1 is intentional because each task spawns a download subprocess.
# =========================

import os, re, tempfile, warnings, subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import requests
import numpy as np
import pandas as pd
import networkx as nx
from scipy import signal
from tqdm.auto import tqdm  # if it doesn't render, switch to: from tqdm.notebook import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIG
# ============================================================
DATASET_ID = "ds000117"
DATASET_TAG = None  # If None: auto-detect latest snapshot tag via GraphQL

OPENNEURO_GQL = "https://openneuro.org/crn/graphql"

DATASET_NAME = "ds000117_meg"
OUT_DIR = "meg_outputs/out_ds000117_meg_v4"

REQUEST_TIMEOUT = 120

# Progress + execution
TQDM_ENABLED = True
TQDM_LEAVE = True
N_JOBS = 1   # <- IMPORTANT for notebook/subprocess downloads

METHODS = ["QTN", "GAF", "MTF"]

# ============================================================
# MEG signal extraction (kept manageable)
# ============================================================
MAX_SECONDS = 60.0         # set None for full run (can be heavy)
TARGET_FS = 250.0          # set None to keep native sampling rate
PICK_MAG = True
PICK_GRAD = True

# ============================================================
# QTN / SMALL-WORLD params
# ============================================================
K_VALUES = [1, 2, 3]
TARGET_DENSITY = 0.10
N_RANDOMIZATIONS = 10
REWIRINGS_PER_EDGE = 5
RNG_SEED = 1234

MAX_ABS_Z = 10.0

# QC thresholds (generic)
MIN_VALID_SAMPLES = 500
MIN_FINITE_FRAC = 0.90
MIN_STD = 1e-12
MIN_UNIQUE_VALUES = 20

# ============================================================
# CSV helpers
# ============================================================
def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)

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

def aggregate_per_subject(file_csv: str, out_csv: str):
    if not os.path.exists(file_csv):
        return
    df = pd.read_csv(file_csv)
    if df.empty or "subject_id" not in df.columns:
        return
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    g = df.groupby("subject_id")[numeric_cols].mean().reset_index()
    if "dataset" in df.columns:
        g.insert(1, "dataset", df.groupby("subject_id")["dataset"].first().values)
    g["n_files_used"] = df.groupby("subject_id")["file_id"].nunique().values
    g["aggregation"] = "mean_over_files"
    g.to_csv(out_csv, index=False)

# ============================================================
# GraphQL listing (candidate list)
# ============================================================
def gql(query: str, variables: Optional[dict] = None) -> dict:
    payload = {"query": query, "variables": variables or {}}
    r = requests.post(OPENNEURO_GQL, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    out = r.json()
    if "errors" in out:
        raise RuntimeError(f"GraphQL errors: {out['errors']}")
    return out["data"]

def get_latest_snapshot_tag(dataset_id: str) -> str:
    q = """
    query ($id: ID!) {
      dataset(id: $id) {
        id
        latestSnapshot { tag }
        snapshots { tag }
      }
    }
    """
    data = gql(q, {"id": dataset_id})
    ds = data.get("dataset") or {}
    if ds.get("latestSnapshot") and ds["latestSnapshot"].get("tag"):
        return ds["latestSnapshot"]["tag"]
    snaps = ds.get("snapshots") or []
    tags = [s.get("tag") for s in snaps if s and s.get("tag")]
    if tags:
        return tags[-1]
    raise RuntimeError("Could not determine latest snapshot tag from GraphQL response.")

def snapshot_files(dataset_id: str, tag: str, tree: Optional[str] = None) -> List[dict]:
    q = """
    query ($datasetId: ID!, $tag: String!, $tree: String) {
      snapshot(datasetId: $datasetId, tag: $tag) {
        files(tree: $tree) {
          id
          key
          filename
          size
          directory
          annexed
        }
      }
    }
    """
    data = gql(q, {"datasetId": dataset_id, "tag": tag, "tree": tree})
    snap = data.get("snapshot") or {}
    return snap.get("files") or []

def list_all_files_recursive(dataset_id: str, tag: str) -> List[str]:
    all_paths = []

    def walk(prefix: str, tree_key: Optional[str]):
        items = snapshot_files(dataset_id, tag, tree=tree_key)
        for it in items:
            name = it["filename"]
            is_dir = bool(it["directory"])
            key = it.get("key")
            if is_dir:
                walk(prefix + name + "/", key)
            else:
                all_paths.append(prefix + name)

    walk(prefix="", tree_key=None)
    return sorted(set(all_paths))

# ============================================================
# openneuro-py download (your version supports download, not ls)
# ============================================================
def ensure_openneuro_py():
    try:
        subprocess.run(["openneuro-py", "--help"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        subprocess.run([os.sys.executable, "-m", "pip", "install", "-U", "openneuro-py"], check=True)

def download_one_openneuro_file(dataset: str, include_path: str, target_dir: str, tag: Optional[str] = None):
    """
    Download ONE file into target_dir.
    We try common snapshot flags; if none supported, we download default snapshot.
    """
    ensure_openneuro_py()

    base = ["openneuro-py", "download", "--dataset", dataset, "--include", include_path, "--target_dir", target_dir]

    if tag is None:
        subprocess.run(base, check=True)
        return

    for opt in ("--snapshot", "--tag"):
        try:
            subprocess.run(base + [opt, tag], check=True)
            return
        except Exception:
            pass

    # fallback: no tag
    subprocess.run(base, check=True)

# ============================================================
# Signal processing utilities
# ============================================================
def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        std = np.nanstd(x)
        scale = std if np.isfinite(std) and std > 0 else 1.0
    z = (x - med) / scale
    return np.clip(z, -MAX_ABS_Z, MAX_ABS_Z)

def preprocess_generic_signal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size < MIN_VALID_SAMPLES:
        raise ValueError(f"Too few samples (n={x.size}).")
    finite = np.isfinite(x)
    if float(finite.mean()) < MIN_FINITE_FRAC:
        raise ValueError(f"Too many NaNs: finite_frac={float(finite.mean()):.3f}")
    x = x.copy()
    if not finite.all():
        idx = np.arange(x.size)
        x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    x = signal.detrend(x, type="constant")
    if float(np.std(x)) <= MIN_STD:
        raise ValueError("Variance too small.")
    if int(np.unique(np.round(x, 12)).size) < MIN_UNIQUE_VALUES:
        raise ValueError("Too few unique values.")
    return robust_zscore(x)

def resample_1d(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if fs_out is None or fs_in is None:
        return x
    if not (np.isfinite(fs_in) and np.isfinite(fs_out)) or fs_in <= 0 or fs_out <= 0:
        return x
    if abs(fs_out - fs_in) / fs_in < 1e-6:
        return x
    n_out = int(round(len(x) * (fs_out / fs_in)))
    n_out = max(10, n_out)
    return signal.resample(x, n_out)

def load_meg_fif_segment(fif_path: str) -> Tuple[np.ndarray, float, List[str]]:
    import mne

    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
    fs = float(raw.info["sfreq"])

    picks = []
    if PICK_MAG:
        picks += mne.pick_types(raw.info, meg="mag", eeg=False, stim=False, eog=False, ecg=False, misc=False)
    if PICK_GRAD:
        picks += mne.pick_types(raw.info, meg="grad", eeg=False, stim=False, eog=False, ecg=False, misc=False)
    picks = np.unique(np.array(picks, dtype=int))

    if picks.size == 0:
        raise ValueError("No MEG mag/grad channels found.")

    n_samples = int(raw.n_times)
    if MAX_SECONDS is None:
        start, stop = 0, n_samples
    else:
        start = 0
        stop = min(n_samples, int(round(MAX_SECONDS * fs)))

    if stop - start < MIN_VALID_SAMPLES:
        raise ValueError(f"Too short after MAX_SECONDS slicing: n={stop-start}")

    data = raw.get_data(picks=picks, start=start, stop=stop)  # (n_ch, T)
    ch_names = [raw.ch_names[p] for p in picks]

    cleaned = []
    kept_names = []
    for i in range(data.shape[0]):
        x = data[i, :]
        if TARGET_FS is not None:
            x = resample_1d(x, fs_in=fs, fs_out=TARGET_FS)
        x = preprocess_generic_signal(x)
        cleaned.append(x)
        kept_names.append(ch_names[i])

    min_len = min(len(x) for x in cleaned)
    X = np.column_stack([x[:min_len] for x in cleaned])  # (T, n_ch)
    fs_used = float(TARGET_FS) if TARGET_FS is not None else fs
    return X, fs_used, kept_names

# ============================================================
# QTN / GAF / MTF
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
    mn, mx = float(np.min(x)), float(np.max(x))
    rng = (mx - mn) if (mx - mn) != 0 else 1.0
    scaled = 2 * (x - mn) / rng - 1
    scaled = np.clip(scaled, -1, 1)
    phi = np.arccos(scaled)
    return np.cos(phi[:, None] + phi[None, :])

def calculate_mtf_from_lengthQ(signal_Q: np.ndarray, Q: int) -> np.ndarray:
    x = np.asarray(signal_Q, dtype=float)
    mn, mx = float(np.min(x)), float(np.max(x))
    rng = (mx - mn) if (mx - mn) != 0 else 1.0
    norm_signal = (x - mn) / rng

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
# File-level: average over sensors -> one row per MEG run
# ============================================================
def file_to_all_method_rows_meg(fif_path: str, seed_base: int) -> Dict[str, dict]:
    X, fs_used, ch_names = load_meg_fif_segment(fif_path)  # (T, n_sensors)
    T, n_ch = X.shape
    Q = compute_Q_from_T(T)
    if Q < 3:
        raise ValueError(f"Q too small: {Q}")

    per_method_metrics = {m: [] for m in METHODS}

    for li in range(n_ch):
        x = X[:, li]
        x = x[np.isfinite(x)]
        if x.size < MIN_VALID_SAMPLES:
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
            raise ValueError(f"No valid sensors produced metrics for {method}.")
        dfm = pd.DataFrame(per_method_metrics[method])
        avg = dfm.mean(numeric_only=True).to_dict()
        avg.update({
            "fs_hz": float(fs_used),
            "n_sensors": int(n_ch),
            "T_used_samples": int(T),
            "Q_used": int(Q),
            "sensor_names": ";".join(ch_names[:50]) + (";..." if len(ch_names) > 50 else ""),
            "max_seconds": float(MAX_SECONDS) if MAX_SECONDS is not None else np.nan,
            "target_fs": float(TARGET_FS) if TARGET_FS is not None else np.nan,
            "pick_mag": bool(PICK_MAG),
            "pick_grad": bool(PICK_GRAD),
        })
        out[method] = avg
    return out

# ============================================================
# BIDS parsing
# ============================================================
def subject_from_bids_path(p: str) -> str:
    m = re.search(r"(sub-[a-zA-Z0-9]+)", p)
    return m.group(1) if m else "unknown"

def file_id_from_bids_path(p: str) -> str:
    name = Path(p).name
    if name.endswith(".fif.gz"):
        return name[:-7]
    if name.endswith(".fif"):
        return name[:-4]
    return name.rsplit(".", 1)[0]

# ============================================================
# One task: try download -> process -> delete
# ============================================================
def compute_task_meg(bids_path: str, tag: Optional[str], task_i: int):
    subject_id = subject_from_bids_path(bids_path)
    file_id = file_id_from_bids_path(bids_path)

    try:
        with tempfile.TemporaryDirectory(prefix="tmp_ds000117_meg_") as td:
            download_one_openneuro_file(DATASET_ID, bids_path, target_dir=td, tag=tag)

            local_expected = os.path.join(td, bids_path)
            if os.path.exists(local_expected):
                local_fif = local_expected
            else:
                target_name = Path(bids_path).name
                candidates = list(Path(td).rglob(target_name))
                if not candidates:
                    raise FileNotFoundError(f"Downloaded file not found for {bids_path}")
                local_fif = str(candidates[0])

            rows = file_to_all_method_rows_meg(local_fif, seed_base=RNG_SEED + 100000 * task_i)

            for method in METHODS:
                rows[method].update({
                    "file_id": file_id,
                    "subject_id": subject_id,
                    "dataset": DATASET_NAME,
                    "source_file": bids_path,
                    "snapshot_tag": tag if tag is not None else "AUTO_LATEST",
                })

        return file_id, rows, None

    except Exception as e:
        skip = {
            "dataset": DATASET_NAME,
            "method": "ALL",
            "record_or_subject": bids_path,
            "file_id": file_id,
            "subject_id": subject_id,
            "reason": str(e),
        }
        return file_id, None, skip

# ============================================================
# Main runner (PER-FILE tqdm)
# ============================================================
def run_meg_ds000117(mode: str = "run"):
    assert mode in ("manifest", "run")

    ensure_out_dir(OUT_DIR)
    paths = out_paths(OUT_DIR, DATASET_NAME)

    tag = DATASET_TAG or get_latest_snapshot_tag(DATASET_ID)
    print(f"[{DATASET_NAME}] GraphQL latest snapshot tag: {tag}")

    all_files = list_all_files_recursive(DATASET_ID, tag)

    meg_fifs = [
        p for p in all_files
        if (not p.startswith("derivatives/"))
        and ("/ses-meg/meg/" in p)
        and (p.endswith("_meg.fif") or p.endswith("_meg.fif.gz"))
    ]
    meg_fifs = sorted(meg_fifs)

    if not meg_fifs:
        raise RuntimeError("No MEG FIF files found after GraphQL filtering.")

    manifest_rows = [{
        "source_file": p,
        "file_id": file_id_from_bids_path(p),
        "subject_id": subject_from_bids_path(p),
        "snapshot_tag_graphql": tag,
    } for p in meg_fifs]
    mf = pd.DataFrame(manifest_rows)
    mf.to_csv(paths["MANIFEST"], index=False)
    print(f"[{DATASET_NAME}] wrote manifest: {paths['MANIFEST']} "
          f"(candidates n_files={len(meg_fifs)} | n_subjects={mf['subject_id'].nunique()})")

    if mode == "manifest":
        print("[MODE] manifest only.")
        return paths

    done_qtn = load_done_ids(paths["QTN_FILE"], "file_id")
    done_gaf = load_done_ids(paths["GAF_FILE"], "file_id")
    done_mtf = load_done_ids(paths["MTF_FILE"], "file_id")
    done_all = done_qtn.intersection(done_gaf).intersection(done_mtf)

    tasks = []
    task_i = 0
    for p in meg_fifs:
        fid = file_id_from_bids_path(p)
        if fid in done_all:
            continue
        task_i += 1
        tasks.append((p, task_i))

    print(f"[{DATASET_NAME}] pending={len(tasks)} | n_jobs={N_JOBS} | MAX_SECONDS={MAX_SECONDS} | TARGET_FS={TARGET_FS}")

    if TQDM_ENABLED:
        task_iter = tqdm(tasks, desc=f"{DATASET_NAME} files", unit="file", leave=TQDM_LEAVE, dynamic_ncols=True)
    else:
        task_iter = tasks

    for (bids_path, ti) in task_iter:
        file_id, rows, skip = compute_task_meg(bids_path, tag, ti)

        if rows is not None:
            append_rows(paths["QTN_FILE"], [rows["QTN"]], id_col="file_id")
            append_rows(paths["GAF_FILE"], [rows["GAF"]], id_col="file_id")
            append_rows(paths["MTF_FILE"], [rows["MTF"]], id_col="file_id")
        if skip is not None:
            append_skips(paths["SKIP"], [skip])

    aggregate_per_subject(paths["QTN_FILE"], paths["QTN_SUBJ"])
    aggregate_per_subject(paths["GAF_FILE"], paths["GAF_SUBJ"])
    aggregate_per_subject(paths["MTF_FILE"], paths["MTF_SUBJ"])

    print(f"[{DATASET_NAME}] done. Outputs in {OUT_DIR}")
    return paths

# ---- notebook usage ----
# paths = run_meg_ds000117("manifest")
paths = run_meg_ds000117("run")
print(paths)

# In[ ]:
