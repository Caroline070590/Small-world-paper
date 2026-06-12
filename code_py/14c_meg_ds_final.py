#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto-converted from the Jupyter notebook ``MEG-DS-FINAL.ipynb``.

Code cells are reproduced verbatim. Markdown cells are kept as comments.
Jupyter shell-magic lines (if any) are commented out so the file is valid
Python; they are preserved for reference. Cell boundaries are marked with
``# In[...]`` to match the original notebook ordering.
"""


# In[1]:

import os
import re
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
from scipy import signal
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# FORCE THE CORRECT CONDA ENV BINARIES
# ============================================================
ENV_PREFIX = Path.home() / "miniconda3" / "envs" / "datalad_meg"

PYTHON_BIN = str(ENV_PREFIX / "bin" / "python")
DATALAD_BIN = str(ENV_PREFIX / "bin" / "datalad")
GIT_BIN = str(ENV_PREFIX / "bin" / "git")
GIT_ANNEX_BIN = str(ENV_PREFIX / "bin" / "git-annex")

print("PYTHON_BIN =", PYTHON_BIN, os.path.exists(PYTHON_BIN))
print("DATALAD_BIN =", DATALAD_BIN, os.path.exists(DATALAD_BIN))
print("GIT_BIN =", GIT_BIN, os.path.exists(GIT_BIN))
print("GIT_ANNEX_BIN =", GIT_ANNEX_BIN, os.path.exists(GIT_ANNEX_BIN))

# In[2]:

# ============================================================
# CONFIG
# ============================================================
DATASET_ROOT = Path.home() / "ds000117"
DATASET_URL = "https://github.com/OpenNeuroDatasets/ds000117.git"

DATASET_NAME = "ds000117_raw_meg"
OUT_ROOT = "ds000117_raw_meg_preprocessed"
OUT_MEG = os.path.join(OUT_ROOT, "meg")

TQDM_ENABLED = True
TQDM_LEAVE = True

FORCE_REPROCESS = False
SUBJECTS_TO_RUN = None          # e.g. ["sub-01", "sub-02"]
USE_DATALAD_DROP = True
AUTO_CLONE_IF_MISSING = True

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

MIN_VALID_SAMPLES = 200
MIN_FINITE_FRAC = 0.90
MIN_UNIQUE_VALUES = 20
MIN_VALID_CHANNELS = 10

# ============================================================
# MEG PREPROCESSING
# ============================================================
MAX_SECONDS_MEG = 60.0
TARGET_FS_MEG = 250.0

MEG_LOWPASS_HZ = 40.0
MEG_HIGHPASS_HZ = 0.5

DO_NOTCH = True
NOTCH_FREQS = [50.0, 100.0]
NOTCH_Q = 30.0

# In[3]:

def run_cmd_capture(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str]:
    env = os.environ.copy()
    env["PATH"] = f"{ENV_PREFIX / 'bin'}:{env.get('PATH', '')}"

    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    return p.returncode, p.stdout

# In[4]:

_DATALAD_CHECKED = False

def check_datalad_binary():
    global _DATALAD_CHECKED
    if _DATALAD_CHECKED:
        return

    if not os.path.exists(DATALAD_BIN):
        raise FileNotFoundError(f"Datalad not found at: {DATALAD_BIN}")
    if not os.path.exists(GIT_BIN):
        raise FileNotFoundError(f"Git not found at: {GIT_BIN}")
    if not os.path.exists(GIT_ANNEX_BIN):
        raise FileNotFoundError(f"git-annex not found at: {GIT_ANNEX_BIN}")

    code1, out1 = run_cmd_capture([DATALAD_BIN, "--version"])
    code2, out2 = run_cmd_capture([GIT_BIN, "annex", "version"])

    if code1 != 0:
        raise RuntimeError(f"datalad is not working:\n{out1}")
    if code2 != 0:
        raise RuntimeError(f"git-annex is not working:\n{out2}")
    if "8.20200226" in out2:
        raise RuntimeError("Old git-annex still being picked up. PATH injection did not work.")

    print(out1.strip())
    print(out2.splitlines()[0])

    _DATALAD_CHECKED = True


def ensure_dataset_clone():
    check_datalad_binary()

    if DATASET_ROOT.exists():
        print(f"[ok] dataset clone exists: {DATASET_ROOT}")
        return

    if not AUTO_CLONE_IF_MISSING:
        raise FileNotFoundError(f"DATASET_ROOT does not exist: {DATASET_ROOT}")

    DATASET_ROOT.parent.mkdir(parents=True, exist_ok=True)

    print(f"[clone] creating lightweight DataLad clone in: {DATASET_ROOT}")
    code, out = run_cmd_capture(
        [DATALAD_BIN, "clone", DATASET_URL, str(DATASET_ROOT)],
        cwd=DATASET_ROOT.parent
    )
    print(out)
    if code != 0:
        raise RuntimeError(f"datalad clone failed:\n{out}")


def check_datalad():
    check_datalad_binary()
    ensure_dataset_clone()
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"DATASET_ROOT does not exist: {DATASET_ROOT}")


def datalad_get(paths: List[str]):
    check_datalad()
    code, out = run_cmd_capture([DATALAD_BIN, "get"] + paths, cwd=DATASET_ROOT)
    if code != 0:
        raise RuntimeError(f"datalad get failed:\n{out}")
    return out


def datalad_drop(paths: List[str]):
    check_datalad()
    code, out = run_cmd_capture(
        [DATALAD_BIN, "drop", "--reckless", "availability"] + paths,
        cwd=DATASET_ROOT
    )
    if code != 0:
        raise RuntimeError(f"datalad drop failed:\n{out}")
    return out


def list_subjects() -> List[str]:
    check_datalad()
    subjects = [p.name for p in DATASET_ROOT.glob("sub-*") if p.is_dir() and p.name != "sub-emptyroom"]
    return sorted(set(subjects))

# In[5]:

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

# In[6]:

def subject_scans_tsv_path(subject_id: str) -> Path:
    return DATASET_ROOT / subject_id / "ses-meg" / f"{subject_id}_ses-meg_scans.tsv"

def get_subject_scan_filenames(subject_id: str) -> List[str]:
    scans_tsv = subject_scans_tsv_path(subject_id)
    if not scans_tsv.exists():
        raise FileNotFoundError(f"Missing scans TSV: {scans_tsv}")

    df = pd.read_csv(scans_tsv, sep="\t")
    if "filename" not in df.columns:
        raise ValueError(f"'filename' column missing in {scans_tsv}")

    files = []
    for rel in df["filename"].astype(str):
        rel = rel.strip()
        low = rel.lower()
        if low.startswith("meg/") and (low.endswith("_meg.fif") or low.endswith("_meg.fif.gz")):
            files.append(rel)

    return sorted(set(files))

def datalad_get_subject_raw_meg(subject_id: str) -> List[Path]:
    scans_rel = str(Path(subject_id) / "ses-meg" / f"{subject_id}_ses-meg_scans.tsv")
    datalad_get([scans_rel])

    rel_files = get_subject_scan_filenames(subject_id)
    if not rel_files:
        return []

    rel_paths = [str(Path(subject_id) / "ses-meg" / rel) for rel in rel_files]
    print("[get] files:")
    for rp in rel_paths:
        print("   ", rp)

    datalad_get(rel_paths)

    abs_paths = [DATASET_ROOT / rp for rp in rel_paths]
    return [p for p in abs_paths if p.exists()]

def datalad_drop_subject_raw_meg(subject_id: str):
    try:
        rel_files = get_subject_scan_filenames(subject_id)
    except Exception:
        return
    if not rel_files:
        return
    rel_paths = [str(Path(subject_id) / "ses-meg" / rel) for rel in rel_files]
    datalad_drop(rel_paths)

def subject_from_bids_path(p: str) -> str:
    m = re.search(r"(sub-[A-Za-z0-9]+)", str(p))
    return m.group(1) if m else "unknown"

def file_id_from_bids_path(p: str) -> str:
    name = Path(p).name
    for suf in [".fif.gz", ".fif"]:
        if name.endswith(suf):
            return name[:-len(suf)]
    return name.rsplit(".", 1)[0]

# In[7]:

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

    std = float(np.std(x))
    ptp = float(np.ptp(x))
    n_unique = int(np.unique(np.round(x, 18)).size)

    if not np.isfinite(std) or not np.isfinite(ptp):
        raise ValueError("Non-finite variance/range.")
    if std <= 1e-20 and ptp <= 1e-18:
        raise ValueError("Variance too small.")
    if n_unique < MIN_UNIQUE_VALUES:
        raise ValueError("Too few unique values.")

    return robust_zscore(x)

def bandpass_filter_1d(x: np.ndarray, fs: float, low_hz: Optional[float], high_hz: Optional[float]) -> np.ndarray:
    if low_hz is None and high_hz is None:
        return x

    nyq = 0.5 * fs
    low = None if low_hz is None else low_hz / nyq
    high = None if high_hz is None else high_hz / nyq

    if low is not None and high is not None:
        if not (0 < low < high < 1):
            return x
        btype = "band"
        wn = [low, high]
    elif low is not None:
        if not (0 < low < 1):
            return x
        btype = "high"
        wn = low
    elif high is not None:
        if not (0 < high < 1):
            return x
        btype = "low"
        wn = high
    else:
        return x

    b, a = signal.butter(4, wn, btype=btype)
    return signal.filtfilt(b, a, x)

def notch_filter_1d(x: np.ndarray, fs: float, freqs: List[float], q: float = 30.0) -> np.ndarray:
    y = np.asarray(x, dtype=float)
    for f0 in freqs:
        if f0 <= 0 or f0 >= fs / 2:
            continue
        b, a = signal.iirnotch(w0=f0, Q=q, fs=fs)
        y = signal.filtfilt(b, a, y)
    return y

def resample_multichannel(X: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if fs_out is None or fs_in is None:
        return X
    if not np.isfinite(fs_in) or not np.isfinite(fs_out) or fs_in <= 0 or fs_out <= 0:
        return X
    if abs(fs_out - fs_in) / fs_in < 1e-8:
        return X
    n_out = int(round(X.shape[0] * fs_out / fs_in))
    n_out = max(10, n_out)
    return signal.resample(X, n_out, axis=0)

# In[8]:

def load_meg_from_fif(fif_path: str):
    import mne

    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")

    picks = mne.pick_types(
        raw.info,
        meg=True,
        ref_meg=False,
        eeg=False,
        stim=False,
        eog=False,
        ecg=False,
        misc=False,
        exclude="bads",
    )
    picks = np.asarray(picks, dtype=int)

    if picks.size == 0:
        raise ValueError("No MEG data channels found.")

    fs_native = float(raw.info["sfreq"])
    n_samples_total = int(raw.n_times)

    stop = min(n_samples_total, int(round(MAX_SECONDS_MEG * fs_native)))
    start = 0

    if stop - start < MIN_VALID_SAMPLES:
        raise ValueError(f"Too few raw samples after cropping: n={stop - start}")

    data = raw.get_data(picks=picks, start=start, stop=stop)
    X = np.asarray(data, dtype=float).T
    fs_used = fs_native

    for j in range(X.shape[1]):
        x = X[:, j]
        x = bandpass_filter_1d(x, fs=fs_used, low_hz=MEG_HIGHPASS_HZ, high_hz=MEG_LOWPASS_HZ)
        if DO_NOTCH:
            x = notch_filter_1d(x, fs=fs_used, freqs=NOTCH_FREQS, q=NOTCH_Q)
        X[:, j] = x

    if TARGET_FS_MEG is not None and np.isfinite(fs_used) and fs_used > 0:
        X = resample_multichannel(X, fs_in=fs_used, fs_out=TARGET_FS_MEG)
        fs_used = float(TARGET_FS_MEG)

    kept = []
    dropped = 0
    drop_reasons = []

    for j in range(X.shape[1]):
        try:
            xj = preprocess_1d(X[:, j])
            kept.append(xj)
        except Exception as e:
            dropped += 1
            drop_reasons.append(str(e))

    if len(kept) < MIN_VALID_CHANNELS:
        raise ValueError(
            f"Too few valid channels after QC. kept={len(kept)}, dropped={dropped}, "
            f"first_reasons={drop_reasons[:10]}"
        )

    min_len = min(len(x) for x in kept)
    X_clean = np.column_stack([x[:min_len] for x in kept])

    info = {
        "n_channels_before_qc": int(X.shape[1]),
        "n_channels_after_qc": int(X_clean.shape[1]),
        "n_channels_dropped_qc": int(dropped),
        "fs_used": fs_used,
        "shape": X_clean.shape,
    }
    return X_clean, fs_used, info

# In[9]:

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

# In[10]:

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

# In[11]:

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

def process_meg_file(fif_path: str, task_i: int):
    file_id = file_id_from_bids_path(fif_path)
    subject_id = subject_from_bids_path(fif_path)

    X, fs, qc_info = load_meg_from_fif(fif_path)
    rows = compute_rows_from_multichannel_timeseries(
        X,
        fs_used=fs,
        seed_base=RNG_SEED + 100000 * task_i + 10,
        channel_kind="meg_fif_raw"
    )

    for m in METHODS:
        rows[m].update({
            "file_id": file_id,
            "subject_id": subject_id,
            "dataset": DATASET_NAME,
            "source_file": str(fif_path),
            "meg_max_seconds": MAX_SECONDS_MEG if MAX_SECONDS_MEG is not None else np.nan,
            "meg_target_fs": TARGET_FS_MEG if TARGET_FS_MEG is not None else np.nan,
            "meg_bandpass_low": MEG_HIGHPASS_HZ,
            "meg_bandpass_high": MEG_LOWPASS_HZ,
            "meg_notch": ";".join(str(f) for f in NOTCH_FREQS) if DO_NOTCH else "",
            **qc_info,
        })

    return file_id, subject_id, rows

# In[12]:

def run_ds000117_raw_meg_all_subjects():
    ensure_out_dir(OUT_MEG)
    paths = out_paths(OUT_MEG, DATASET_NAME)

    subjects = list_subjects()
    if SUBJECTS_TO_RUN is not None:
        subjects = [s for s in subjects if s in set(SUBJECTS_TO_RUN)]

    if not subjects:
        raise RuntimeError("No subjects found to process.")

    manifest_rows = []
    for s in subjects:
        manifest_rows.append({
            "subject_id": s,
            "local_prefix": str(DATASET_ROOT / s / "ses-meg"),
            "dataset": DATASET_NAME,
        })
    pd.DataFrame(manifest_rows).to_csv(paths["MANIFEST"], index=False)

    print(f"[manifest] written: {paths['MANIFEST']}")
    print(f"[subjects] n={len(subjects)} -> {subjects[:10]}{' ...' if len(subjects) > 10 else ''}")

    done_all = (
        load_done_ids(paths["QTN_FILE"], "file_id")
        & load_done_ids(paths["GAF_FILE"], "file_id")
        & load_done_ids(paths["MTF_FILE"], "file_id")
    ) if not FORCE_REPROCESS else set()

    subj_iter = tqdm(subjects, desc="Subjects", leave=TQDM_LEAVE) if TQDM_ENABLED else subjects
    global_task_i = 0

    for subject_id in subj_iter:
        try:
            print(f"\n[get] {subject_id}")
            local_files = datalad_get_subject_raw_meg(subject_id)

            if not local_files:
                append_skips(paths["SKIP"], [{
                    "dataset": DATASET_NAME,
                    "method": "ALL",
                    "record_or_subject": subject_id,
                    "file_id": "",
                    "subject_id": subject_id,
                    "reason": "No raw .fif or .fif.gz MEG files found after datalad get."
                }])
                continue

            print(f"[process] {subject_id}: {len(local_files)} file(s)")
            file_iter = tqdm(local_files, desc=f"{subject_id} files", leave=False) if TQDM_ENABLED else local_files

            for fif_path in file_iter:
                fid = file_id_from_bids_path(str(fif_path))
                subj = subject_from_bids_path(str(fif_path))

                if (fid in done_all) and not FORCE_REPROCESS:
                    continue

                global_task_i += 1

                try:
                    _, _, rows = process_meg_file(str(fif_path), global_task_i)
                    append_rows(paths["QTN_FILE"], [rows["QTN"]], id_col="file_id")
                    append_rows(paths["GAF_FILE"], [rows["GAF"]], id_col="file_id")
                    append_rows(paths["MTF_FILE"], [rows["MTF"]], id_col="file_id")
                except Exception as e:
                    append_skips(paths["SKIP"], [{
                        "dataset": DATASET_NAME,
                        "method": "ALL",
                        "record_or_subject": str(fif_path),
                        "file_id": fid,
                        "subject_id": subj,
                        "reason": str(e),
                    }])

        except Exception as e:
            append_skips(paths["SKIP"], [{
                "dataset": DATASET_NAME,
                "method": "ALL",
                "record_or_subject": subject_id,
                "file_id": "",
                "subject_id": subject_id,
                "reason": f"Subject-level failure: {str(e)}",
            }])

        finally:
            if USE_DATALAD_DROP:
                try:
                    datalad_drop_subject_raw_meg(subject_id)
                    print(f"[drop] released content for {subject_id}")
                except Exception as e:
                    print(f"[drop-warning] {subject_id}: {e}")

        aggregate_per_subject(paths["QTN_FILE"], paths["QTN_SUBJ"], "subject_id")
        aggregate_per_subject(paths["GAF_FILE"], paths["GAF_SUBJ"], "subject_id")
        aggregate_per_subject(paths["MTF_FILE"], paths["MTF_SUBJ"], "subject_id")

    print(f"\n[done] outputs in {OUT_MEG}")
    return paths

def wipe_all_outputs():
    if os.path.exists(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)
    print(f"Deleted: {OUT_ROOT}")

# In[13]:

def show_subjects():
    subs = list_subjects()
    print(f"n_subjects = {len(subs)}")
    print(subs)

def run_one_subject(subject_id: str, force_reprocess: bool = True):
    global SUBJECTS_TO_RUN, FORCE_REPROCESS
    SUBJECTS_TO_RUN = [subject_id]
    FORCE_REPROCESS = force_reprocess
    return run_ds000117_raw_meg_all_subjects()

def run_subject_range(start: int, stop: int, force_reprocess: bool = False):
    global SUBJECTS_TO_RUN, FORCE_REPROCESS
    subs = list_subjects()
    SUBJECTS_TO_RUN = subs[start:stop]
    FORCE_REPROCESS = force_reprocess
    print("Running:", SUBJECTS_TO_RUN)
    return run_ds000117_raw_meg_all_subjects()

def run_all_subjects(force_reprocess: bool = False):
    global SUBJECTS_TO_RUN, FORCE_REPROCESS
    SUBJECTS_TO_RUN = None
    FORCE_REPROCESS = force_reprocess
    return run_ds000117_raw_meg_all_subjects()

# In[14]:

check_datalad()
print("DATALAD_BIN =", DATALAD_BIN)
print("DATASET_ROOT =", DATASET_ROOT)
print("Exists =", DATASET_ROOT.exists())

show_subjects()

# In[15]:

paths = run_all_subjects(force_reprocess=False)
print(paths)

# In[16]:

paths = out_paths(OUT_MEG, DATASET_NAME)

for k, v in paths.items():
    print(k, "->", v, os.path.exists(v))

qtn_file = pd.read_csv(paths["QTN_FILE"]) if os.path.exists(paths["QTN_FILE"]) else pd.DataFrame()
gaf_file = pd.read_csv(paths["GAF_FILE"]) if os.path.exists(paths["GAF_FILE"]) else pd.DataFrame()
mtf_file = pd.read_csv(paths["MTF_FILE"]) if os.path.exists(paths["MTF_FILE"]) else pd.DataFrame()

qtn_subj = pd.read_csv(paths["QTN_SUBJ"]) if os.path.exists(paths["QTN_SUBJ"]) else pd.DataFrame()
gaf_subj = pd.read_csv(paths["GAF_SUBJ"]) if os.path.exists(paths["GAF_SUBJ"]) else pd.DataFrame()
mtf_subj = pd.read_csv(paths["MTF_SUBJ"]) if os.path.exists(paths["MTF_SUBJ"]) else pd.DataFrame()

print("QTN per file:", qtn_file.shape)
print("GAF per file:", gaf_file.shape)
print("MTF per file:", mtf_file.shape)

print("QTN per subject:", qtn_subj.shape)
print("GAF per subject:", gaf_subj.shape)
print("MTF per subject:", mtf_subj.shape)

display(qtn_file.head())
display(gaf_file.head())
display(mtf_file.head())

# In[ ]:
