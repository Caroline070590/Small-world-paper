#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto-converted from the Jupyter notebook ``ds000117.ipynb``.

Code cells are reproduced verbatim. Markdown cells are kept as comments.
Jupyter shell-magic lines (if any) are commented out so the file is valid
Python; they are preserved for reference. Cell boundaries are marked with
``# In[...]`` to match the original notebook ordering.
"""


# In[1]:

# =========================
# ds000117 (OpenNeuro) multimodal -> QTN/GAF/MTF small-world
#
# Rewritten to match the actual BIDS organization:
#   outputs/ds000117/
#       meg/
#       eeg/
#       fmri/
#       mri/
#
# INPUT ORGANIZATION USED:
#   MEG : sub-*/ses-meg/meg/*_meg.fif(.gz)
#   EEG : extracted from the SAME FIF files as MEG
#   fMRI: sub-*/ses-mri/func/*_bold.nii(.gz)
#   MRI : sub-*/ses-mri/anat/*_T1w.nii(.gz)
#
# IMPORTANT:
# - openneuro-py is used only for download
# - file listing comes from OpenNeuro GraphQL
# - download tries path variants automatically:
#     .fif <-> .fif.gz
#     .nii <-> .nii.gz
# - if a file still fails, the full downloader output is written to skipped_*.csv
# - files are downloaded to a temporary directory and deleted afterwards
# =========================

import os
import re
import tempfile
import warnings
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable

import requests
import numpy as np
import pandas as pd
import networkx as nx
from scipy import signal
from tqdm.auto import tqdm  # if needed: from tqdm.notebook import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIG
# ============================================================
DATASET_ID = "ds000117"
DATASET_TAG = None  # None => use latest GraphQL snapshot tag
OPENNEURO_GQL = "https://openneuro.org/crn/graphql"
OPENNEURO_PY_BIN = "openneuro-py"
REQUEST_TIMEOUT = 120

# Output root
OUT_ROOT = "outputs/ds000117"
OUT_MEG = os.path.join(OUT_ROOT, "meg")
OUT_EEG = os.path.join(OUT_ROOT, "eeg")
OUT_FMRI = os.path.join(OUT_ROOT, "fmri")
OUT_MRI = os.path.join(OUT_ROOT, "mri")

# Progress
TQDM_ENABLED = True
TQDM_LEAVE = True

# ============================================================
# COMMON QTN / SMALL-WORLD PARAMS
# ============================================================
METHODS = ["QTN", "GAF", "MTF"]

K_VALUES = [1, 2, 3]
TARGET_DENSITY = 0.10
N_RANDOMIZATIONS = 10
REWIRINGS_PER_EDGE = 5
RNG_SEED = 1234
MAX_ABS_Z = 10.0

# Generic QC
MIN_VALID_SAMPLES = 300
MIN_FINITE_FRAC = 0.90
MIN_STD = 1e-12
MIN_UNIQUE_VALUES = 20

# ============================================================
# MODALITY-SPECIFIC CONTROLS
# ============================================================
# MEG / EEG
MAX_SECONDS_MEEG = 60.0
TARGET_FS_MEEG = 250.0
PICK_MEG_MAG = True
PICK_MEG_GRAD = True
PICK_EEG = True

# fMRI
FMRI_MAX_TRS = None
BASC_RESOLUTION = 122

# MRI
MRI_MAX_SLICES = None  # e.g. 256

# ============================================================
# OUTPUT HELPERS
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

def aggregate_per_subject(file_csv: str, out_csv: str, subj_col: str = "subject_id"):
    if not os.path.exists(file_csv):
        return
    df = pd.read_csv(file_csv)
    if df.empty or subj_col not in df.columns:
        return
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    g = df.groupby(subj_col)[numeric_cols].mean().reset_index()
    if "dataset" in df.columns:
        g.insert(1, "dataset", df.groupby(subj_col)["dataset"].first().values)
    if "file_id" in df.columns:
        g["n_files_used"] = df.groupby(subj_col)["file_id"].nunique().values
    g["aggregation"] = "mean_over_files"
    g.to_csv(out_csv, index=False)

# ============================================================
# GRAPHQL LISTING
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
    raise RuntimeError("Could not determine latest snapshot tag.")

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
# BIDS HELPERS
# ============================================================
def subject_from_bids_path(p: str) -> str:
    m = re.search(r"(sub-[a-zA-Z0-9]+)", p)
    return m.group(1) if m else "unknown"

def file_id_from_bids_path(p: str) -> str:
    name = Path(p).name
    for suf in [".nii.gz", ".nii", ".fif.gz", ".fif"]:
        if name.endswith(suf):
            return name[:-len(suf)]
    return name.rsplit(".", 1)[0]

# ============================================================
# DOWNLOAD HELPERS
# ============================================================
def ensure_openneuro_py():
    try:
        subprocess.run(
            [OPENNEURO_PY_BIN, "--help"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        subprocess.run(
            [os.sys.executable, "-m", "pip", "install", "-U", "openneuro-py"],
            check=True,
        )

def run_cmd_capture(cmd: List[str]) -> Tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout

def candidate_path_variants(p: str) -> List[str]:
    out = [p]
    if p.endswith(".fif"):
        out.append(p + ".gz")
    if p.endswith(".fif.gz"):
        out.append(p[:-3])
    if p.endswith(".nii"):
        out.append(p + ".gz")
    if p.endswith(".nii.gz"):
        out.append(p[:-3])

    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq

def download_one_openneuro_file(include_path: str, target_dir: str, tag: Optional[str]) -> Tuple[bool, str, str, str]:
    """
    Returns:
      ok, stdout, cmd_used, include_used
    """
    ensure_openneuro_py()

    last_out = ""
    last_cmd = ""
    for inc in candidate_path_variants(include_path):
        base = [
            OPENNEURO_PY_BIN, "download",
            "--dataset", DATASET_ID,
            "--include", inc,
            "--target_dir", target_dir
        ]

        if tag is not None:
            for opt in ("--snapshot", "--tag"):
                code, out = run_cmd_capture(base + [opt, tag])
                if code == 0:
                    return True, out, " ".join(base + [opt, tag]), inc
                last_out = out
                last_cmd = " ".join(base + [opt, tag])

        code, out = run_cmd_capture(base)
        if code == 0:
            return True, out, " ".join(base), inc
        last_out = out
        last_cmd = " ".join(base)

    return False, last_out, last_cmd, include_path

def locate_downloaded_file(tmp_root: str, used_include_path: str) -> str:
    expected = os.path.join(tmp_root, used_include_path)
    if os.path.exists(expected):
        return expected
    target_name = Path(used_include_path).name
    candidates = list(Path(tmp_root).rglob(target_name))
    if not candidates:
        raise FileNotFoundError(f"Downloaded file not found for {used_include_path}")
    return str(candidates[0])

# ============================================================
# PREPROCESSING / SIGNAL HELPERS
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

def preprocess_1d(x: np.ndarray) -> np.ndarray:
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
# GRAPH / SMALL-WORLD
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

def compute_rows_from_multichannel_timeseries(X: np.ndarray, fs_used: float, seed_base: int, channel_kind: str) -> Dict[str, dict]:
    T, n_ch = X.shape
    if T < MIN_VALID_SAMPLES:
        raise ValueError("Too few samples after extraction.")
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
            raise ValueError(f"No valid channels produced metrics for {method}.")
        dfm = pd.DataFrame(per_method_metrics[method])
        avg = dfm.mean(numeric_only=True).to_dict()
        avg.update({
            "fs_hz": float(fs_used) if fs_used is not None else np.nan,
            "n_channels": int(n_ch),
            "T_used_samples": int(T),
            "Q_used": int(Q),
            "channel_kind": channel_kind,
        })
        out[method] = avg
    return out

# ============================================================
# MODALITY LOADERS
# ============================================================
def load_meeg_from_fif(fif_path: str, want_meg: bool, want_eeg: bool) -> Tuple[np.ndarray, float]:
    import mne

    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
    fs = float(raw.info["sfreq"])

    picks = []
    if want_meg:
        if PICK_MEG_MAG:
            picks += mne.pick_types(raw.info, meg="mag", eeg=False, stim=False, eog=False, ecg=False, misc=False)
        if PICK_MEG_GRAD:
            picks += mne.pick_types(raw.info, meg="grad", eeg=False, stim=False, eog=False, ecg=False, misc=False)
    if want_eeg and PICK_EEG:
        picks += mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, ecg=False, misc=False)

    picks = np.unique(np.array(picks, dtype=int))
    if picks.size == 0:
        raise ValueError("No channels found for requested type(s).")

    n_samples = int(raw.n_times)
    if MAX_SECONDS_MEEG is None:
        start, stop = 0, n_samples
    else:
        start = 0
        stop = min(n_samples, int(round(MAX_SECONDS_MEEG * fs)))

    if stop - start < MIN_VALID_SAMPLES:
        raise ValueError(f"Too short after MAX_SECONDS_MEEG slicing: n={stop-start}")

    data = raw.get_data(picks=picks, start=start, stop=stop)

    cleaned = []
    fs_used = fs
    for i in range(data.shape[0]):
        x = data[i, :]
        if TARGET_FS_MEEG is not None:
            x = resample_1d(x, fs_in=fs, fs_out=TARGET_FS_MEEG)
            fs_used = float(TARGET_FS_MEEG)
        x = preprocess_1d(x)
        cleaned.append(x)

    min_len = min(len(x) for x in cleaned)
    X = np.column_stack([x[:min_len] for x in cleaned])
    return X, fs_used

def load_fmri_roi_timeseries(bold_nii_path: str) -> Tuple[np.ndarray, float]:
    from nilearn.datasets import fetch_atlas_basc_multiscale_2015
    from nilearn.maskers import NiftiMapsMasker
    import nibabel as nib

    atlas = fetch_atlas_basc_multiscale_2015()
    key = f"scale{int(BASC_RESOLUTION)}"
    if key not in atlas:
        raise ValueError(f"BASC atlas missing key {key}.")

    img = nib.load(bold_nii_path)
    hdr = img.header

    tr = np.nan
    try:
        zooms = hdr.get_zooms()
        if len(zooms) >= 4 and np.isfinite(zooms[3]) and zooms[3] > 0:
            tr = float(zooms[3])
    except Exception:
        tr = np.nan
    fs = (1.0 / tr) if (np.isfinite(tr) and tr > 0) else np.nan

    masker = NiftiMapsMasker(
        maps_img=atlas[key],
        standardize=True,
        detrend=True,
        verbose=0,
    )
    X = masker.fit_transform(bold_nii_path)

    if FMRI_MAX_TRS is not None:
        X = X[: int(FMRI_MAX_TRS), :]

    finite_cols = np.all(np.isfinite(X), axis=0)
    X = X[:, finite_cols]
    if X.shape[1] < 2:
        raise ValueError("Too few valid ROIs after masking/QC.")

    out = []
    for j in range(X.shape[1]):
        out.append(preprocess_1d(X[:, j]))
    min_len = min(len(x) for x in out)
    X2 = np.column_stack([x[:min_len] for x in out])
    return X2, fs

def load_mri_slice_profile(t1w_path: str) -> Tuple[np.ndarray, float]:
    import nibabel as nib

    img = nib.load(t1w_path)
    data = img.get_fdata(dtype=np.float32)

    finite = np.isfinite(data)
    if float(finite.mean()) < 0.95:
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    z_means = data.mean(axis=(0, 1))
    if MRI_MAX_SLICES is not None:
        z_means = z_means[: int(MRI_MAX_SLICES)]

    z_means = preprocess_1d(z_means)
    X = z_means.reshape(-1, 1)
    fs = np.nan
    return X, fs

# ============================================================
# MODALITY PROCESSOR
# ============================================================
def process_one_file(modality: str, bids_path: str, snapshot_tag: str, task_i: int):
    subj = subject_from_bids_path(bids_path)
    fid = file_id_from_bids_path(bids_path)

    with tempfile.TemporaryDirectory(prefix=f"tmp_{DATASET_ID}_{modality}_") as td:
        ok, stdout, cmd_used, include_used = download_one_openneuro_file(bids_path, td, tag=snapshot_tag)
        if not ok:
            raise RuntimeError(
                f"download failed\n"
                f"include={include_used}\n"
                f"cmd={cmd_used}\n"
                f"{stdout}"
            )

        local_path = locate_downloaded_file(td, include_used)

        if modality == "meg":
            X, fs = load_meeg_from_fif(local_path, want_meg=True, want_eeg=False)
            rows = compute_rows_from_multichannel_timeseries(
                X, fs_used=fs, seed_base=RNG_SEED + 100000 * task_i + 10, channel_kind="meg"
            )
            dataset_label = "ds000117_meg"

        elif modality == "eeg":
            X, fs = load_meeg_from_fif(local_path, want_meg=False, want_eeg=True)
            rows = compute_rows_from_multichannel_timeseries(
                X, fs_used=fs, seed_base=RNG_SEED + 100000 * task_i + 20, channel_kind="eeg"
            )
            dataset_label = "ds000117_eeg"

        elif modality == "fmri":
            X, fs = load_fmri_roi_timeseries(local_path)
            rows = compute_rows_from_multichannel_timeseries(
                X, fs_used=fs, seed_base=RNG_SEED + 100000 * task_i + 30, channel_kind="fmri_roi"
            )
            dataset_label = "ds000117_fmri"

        elif modality == "mri":
            X, fs = load_mri_slice_profile(local_path)
            rows = compute_rows_from_multichannel_timeseries(
                X, fs_used=fs, seed_base=RNG_SEED + 100000 * task_i + 40, channel_kind="mri_slice_profile"
            )
            dataset_label = "ds000117_mri"

        else:
            raise ValueError(f"Unknown modality: {modality}")

        for m in METHODS:
            rows[m].update({
                "file_id": fid,
                "subject_id": subj,
                "dataset": dataset_label,
                "source_file": bids_path,
                "snapshot_tag": snapshot_tag,
                "include_used": include_used,
            })
            if modality == "fmri":
                rows[m].update({
                    "basc_resolution": int(BASC_RESOLUTION),
                    "fmri_max_trs": FMRI_MAX_TRS if FMRI_MAX_TRS is not None else np.nan,
                })

        return fid, subj, rows

# ============================================================
# FILTERS MATCHING THE REAL BIDS ORGANIZATION
# ============================================================
def meg_filter(p: str) -> bool:
    return (
        p.startswith("sub-")
        and ("/ses-meg/meg/" in p)
        and (p.endswith("_meg.fif") or p.endswith("_meg.fif.gz"))
    )

def eeg_filter(p: str) -> bool:
    # EEG comes from the SAME FIF files as MEG
    return meg_filter(p)

def fmri_filter(p: str) -> bool:
    return (
        p.startswith("sub-")
        and ("/ses-mri/func/" in p)
        and (p.endswith("_bold.nii") or p.endswith("_bold.nii.gz"))
    )

def mri_filter(p: str) -> bool:
    return (
        p.startswith("sub-")
        and ("/ses-mri/anat/" in p)
        and (p.endswith("_T1w.nii") or p.endswith("_T1w.nii.gz"))
    )

# ============================================================
# RUNNER PER MODALITY
# ============================================================
def run_modality(modality: str, out_dir: str, file_filter_fn: Callable[[str], bool]):
    ensure_out_dir(out_dir)
    paths = out_paths(out_dir, f"{DATASET_ID}_{modality}")

    tag = DATASET_TAG or get_latest_snapshot_tag(DATASET_ID)
    print(f"[{modality}] Using snapshot tag: {tag}")

    all_files = list_all_files_recursive(DATASET_ID, tag)
    files = [p for p in all_files if (not p.startswith("derivatives/")) and file_filter_fn(p)]
    files = sorted(files)

    if not files:
        raise RuntimeError(f"[{modality}] No files found after filtering.")

    manifest = pd.DataFrame([{
        "source_file": p,
        "file_id": file_id_from_bids_path(p),
        "subject_id": subject_from_bids_path(p),
        "snapshot_tag_graphql": tag,
    } for p in files])
    manifest.to_csv(paths["MANIFEST"], index=False)
    print(f"[{modality}] candidates: n_files={len(files)} | n_subjects={manifest['subject_id'].nunique()}")

    done_all = (
        load_done_ids(paths["QTN_FILE"], "file_id")
        & load_done_ids(paths["GAF_FILE"], "file_id")
        & load_done_ids(paths["MTF_FILE"], "file_id")
    )

    tasks = []
    ti = 0
    for p in files:
        fid = file_id_from_bids_path(p)
        if fid in done_all:
            continue
        ti += 1
        tasks.append((p, ti))

    iterator = tqdm(
        tasks,
        desc=f"{DATASET_ID} {modality}",
        unit="file",
        leave=TQDM_LEAVE,
        dynamic_ncols=True,
    ) if TQDM_ENABLED else tasks

    for (p, ti) in iterator:
        fid = file_id_from_bids_path(p)
        subj = subject_from_bids_path(p)
        try:
            _, _, rows = process_one_file(modality, p, tag, ti)
            append_rows(paths["QTN_FILE"], [rows["QTN"]], id_col="file_id")
            append_rows(paths["GAF_FILE"], [rows["GAF"]], id_col="file_id")
            append_rows(paths["MTF_FILE"], [rows["MTF"]], id_col="file_id")
        except Exception as e:
            append_skips(paths["SKIP"], [{
                "dataset": f"{DATASET_ID}_{modality}",
                "method": "ALL",
                "record_or_subject": p,
                "file_id": fid,
                "subject_id": subj,
                "reason": str(e),
            }])

    aggregate_per_subject(paths["QTN_FILE"], paths["QTN_SUBJ"], "subject_id")
    aggregate_per_subject(paths["GAF_FILE"], paths["GAF_SUBJ"], "subject_id")
    aggregate_per_subject(paths["MTF_FILE"], paths["MTF_SUBJ"], "subject_id")

    print(f"[{modality}] done. outputs in {out_dir}")
    return paths

# ============================================================
# RUN ALL
# ============================================================
def run_all():
    ensure_out_dir(OUT_ROOT)

    meg_paths = run_modality("meg", OUT_MEG, meg_filter)
    eeg_paths = run_modality("eeg", OUT_EEG, eeg_filter)
    fmri_paths = run_modality("fmri", OUT_FMRI, fmri_filter)
    mri_paths = run_modality("mri", OUT_MRI, mri_filter)

    return {
        "meg": meg_paths,
        "eeg": eeg_paths,
        "fmri": fmri_paths,
        "mri": mri_paths,
    }

# ============================================================
# NOTEBOOK USAGE
# ============================================================
# To run only one modality:
# run_modality("meg", OUT_MEG, meg_filter)
# run_modality("eeg", OUT_EEG, eeg_filter)
# run_modality("fmri", OUT_FMRI, fmri_filter)
# run_modality("mri", OUT_MRI, mri_filter)

paths = run_all()
print(paths)

# In[ ]:
