#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto-converted from the Jupyter notebook ``sleep-data-CAPS.ipynb``.

Code cells are reproduced verbatim. Markdown cells are kept as comments.
Jupyter shell-magic lines (if any) are commented out so the file is valid
Python; they are preserved for reference. Cell boundaries are marked with
``# In[...]`` to match the original notebook ordering.
"""


# In[1]:

#!/usr/bin/env python3
# cap_sleep_controls_nrem_fast_once_per_subject.py
#
# Faster CAP healthy-control pipeline:
# - download each subject once
# - load EDF once
# - parse annotations once
# - build NREM mask once
# - preprocess each modality once
# - compute QTN/GAF/MTF together
# - append results incrementally
#
# Outputs:
#   sleep_outputs/cap_sleep_controls/EEG/
#   sleep_outputs/cap_sleep_controls/EMG/
#   sleep_outputs/cap_sleep_controls/RESP/
#   sleep_outputs/cap_sleep_controls/ALL_SLEEP/

import os
import re
import warnings
from pathlib import Path
from typing import List, Dict, Tuple
from urllib.parse import urljoin

import requests
import numpy as np
import pandas as pd
import networkx as nx
import mne

from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIG
# ============================================================
BASE_URL = "https://physionet.org/files/capslpdb/1.0.0/"

DATASET_NAME = "cap_sleep_controls"
BASE_OUT_DIR = "sleep_outputs"
TMP_DIR = "tmp_cap_sleep"

CONTROL_IDS = [f"n{i}" for i in range(1, 17)]
MODALITIES = ["EEG", "EMG", "RESP", "ALL_SLEEP"]

SLEEP_SELECTION = "NREM"   # or WHOLE_NIGHT

DELETE_RAW_AFTER_PROCESS = True
REQUEST_TIMEOUT = 120

K_VALUES = [1, 2, 3]
TARGET_DENSITY = 0.10
N_RANDOMIZATIONS = 5       # faster than 20
REWIRINGS_PER_EDGE = 5
RNG_SEED = 1234

MAX_SAMPLES = None         # for testing, set e.g. 200000
START_SAMPLE = 0

N_JOBS = 1                 # safest for EDF
BACKEND = "loky"

METHODS = ["QTN", "GAF", "MTF"]

TARGET_FS = 128.0
LINE_FREQ = 50.0
EPOCH_SEC = 30.0
MIN_KEEP_EPOCHS = 5

PREPROC = {
    "EEG": {"l_freq": 0.5, "h_freq": 40.0, "notch": True},
    "EMG": {"l_freq": 10.0, "h_freq": 45.0, "notch": True},
    "RESP": {"l_freq": 0.01, "h_freq": 1.0, "notch": False},
    "ALL_SLEEP": {"l_freq": None, "h_freq": None, "notch": False},
}

CHANNEL_REGEX = {
    "EEG": r"EEG|Fpz|Pz|C3|C4|F3|F4|O1|O2",
    "EMG": r"EMG|Chin|Submental|Submentalis|Tibial",
    "RESP": r"Resp|Airflow|Nasal|Thor|Abdo|Abdominal|Thoracic|SaO2|SpO2|Flow",
    "ALL_SLEEP": r"EEG|EOG|EMG|Chin|Submental|Submentalis|Tibial|Resp|Airflow|Nasal|Thor|Abdo|Abdominal|Thoracic|SaO2|SpO2|Flow",
}

NREM_EVENTS = {"SLEEP-S1", "SLEEP-S2", "SLEEP-S3", "SLEEP-S4"}


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
    return np.clip(z, -10, 10)

def select_channels(ch_names: List[str], channel_mode: str) -> List[int]:
    pat = re.compile(CHANNEL_REGEX[channel_mode], flags=re.IGNORECASE)
    return [i for i, ch in enumerate(ch_names) if pat.search(str(ch))]

def channel_family(name: str) -> str:
    s = str(name).lower()
    if re.search(r"emg|chin|submental|submentalis|tibial", s):
        return "EMG"
    if re.search(r"resp|airflow|nasal|thor|abdo|abdominal|thoracic|sao2|spo2|flow", s):
        return "RESP"
    return "EEG"


# ============================================================
# Download
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

def subject_files(subject_id: str) -> Dict[str, str]:
    return {
        "edf": f"{subject_id}.edf",
        "txt": f"{subject_id}.txt",
        "st": f"{subject_id}.edf.st",
    }

def fetch_subject(subject_id: str, tmp_dir: str) -> Dict[str, str]:
    files = subject_files(subject_id)
    local = {}
    for key, fname in files.items():
        url = urljoin(BASE_URL, fname)
        local_path = os.path.join(tmp_dir, fname)
        try:
            download_file(url, local_path)
            local[key] = local_path
        except Exception:
            if key == "st":
                continue
            raise
    return local

def cleanup_subject(local_files: Dict[str, str]):
    for p in local_files.values():
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


# ============================================================
# Annotation parsing
# ============================================================
def parse_cap_stage_intervals(txt_path: str) -> List[Tuple[float, float, str]]:
    intervals = []
    current_start = 0.0
    in_table = False

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("Sleep Stage"):
                in_table = True
                continue
            if not in_table:
                continue

            parts = re.split(r"\t+", s)
            if len(parts) < 5:
                continue

            event = parts[3].strip()
            dur_str = parts[4].strip()

            if event not in {"SLEEP-S0", "SLEEP-S1", "SLEEP-S2", "SLEEP-S3", "SLEEP-S4", "SLEEP-REM", "MT"}:
                continue

            try:
                dur = float(dur_str)
            except Exception:
                continue

            intervals.append((current_start, current_start + dur, event))
            current_start += dur

    if not intervals:
        raise ValueError(f"No sleep-stage intervals found in {txt_path}")

    return intervals

def build_sample_mask_from_intervals(n_times: int, sfreq: float, intervals, sleep_selection: str) -> np.ndarray:
    mask = np.zeros(n_times, dtype=bool)

    if sleep_selection == "WHOLE_NIGHT":
        mask[:] = True
        return mask

    if sleep_selection != "NREM":
        raise ValueError(f"Unknown sleep_selection: {sleep_selection}")

    for start_sec, end_sec, event in intervals:
        if event in NREM_EVENTS:
            i0 = max(0, int(round(start_sec * sfreq)))
            i1 = min(n_times, int(round(end_sec * sfreq)))
            if i1 > i0:
                mask[i0:i1] = True

    return mask


# ============================================================
# Preprocessing
# ============================================================
def preprocess_matrix(data: np.ndarray, sfreq: float, names: List[str], mode: str) -> Tuple[np.ndarray, List[str], float]:
    if sfreq > TARGET_FS:
        data = mne.filter.resample(data, down=sfreq / TARGET_FS, npad="auto", axis=1)
        sfreq = float(TARGET_FS)

    out = np.zeros_like(data, dtype=float)

    for i, ch_name in enumerate(names):
        fam = mode if mode != "ALL_SLEEP" else channel_family(ch_name)
        params = PREPROC[fam]

        x = data[i].astype(float)

        if params["notch"] and sfreq > 2 * LINE_FREQ:
            x = mne.filter.notch_filter(
                x, Fs=sfreq, freqs=[LINE_FREQ], method="fir", phase="zero", verbose=False
            )

        if params["l_freq"] is not None or params["h_freq"] is not None:
            x = mne.filter.filter_data(
                x,
                sfreq=sfreq,
                l_freq=params["l_freq"],
                h_freq=params["h_freq"],
                method="fir",
                phase="zero",
                verbose=False,
            )

        x = robust_zscore(x)
        out[i] = x

    epoch_len = max(1, int(round(EPOCH_SEC * sfreq)))
    n_times = out.shape[1]
    n_epochs = n_times // epoch_len
    if n_epochs == 0:
        raise ValueError("Signal too short after preprocessing.")

    out = out[:, : n_epochs * epoch_len]
    epochs = out.reshape(out.shape[0], n_epochs, epoch_len)
    ptp = np.ptp(epochs, axis=2)

    keep_mask = np.ones(n_epochs, dtype=bool)
    for ch in range(ptp.shape[0]):
        med = np.median(ptp[ch])
        mad = np.median(np.abs(ptp[ch] - med))
        thr = med + 6.0 * (1.4826 * mad if mad > 0 else np.std(ptp[ch]) if np.std(ptp[ch]) > 0 else 1.0)
        keep_mask &= (ptp[ch] <= thr)

    if keep_mask.sum() < MIN_KEEP_EPOCHS:
        raise ValueError(f"Too few clean epochs kept ({keep_mask.sum()}).")

    cleaned = epochs[:, keep_mask, :].reshape(out.shape[0], -1)
    return cleaned.T, names, sfreq


def load_subject_modalities_once(edf_path: str, txt_path: str, sleep_selection: str) -> Dict[str, Tuple[np.ndarray, List[str], float]]:
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    ch_names = raw.ch_names
    intervals = parse_cap_stage_intervals(txt_path)

    sfreq_orig = float(raw.info["sfreq"])
    mask = build_sample_mask_from_intervals(raw.n_times, sfreq_orig, intervals, sleep_selection)
    if mask.sum() == 0:
        raise ValueError(f"No samples kept for sleep_selection={sleep_selection} in {edf_path}")

    all_outputs = {}

    for mode in MODALITIES:
        idx = select_channels(ch_names, mode)
        if not idx:
            continue

        data = raw.get_data(picks=idx)
        data = data[:, mask]
        picked_names = [ch_names[i] for i in idx]

        sig, sel_names, fs = preprocess_matrix(data, sfreq_orig, picked_names, mode)
        all_outputs[mode] = (sig, sel_names, fs)

    if not all_outputs:
        raise ValueError("No valid modality channels found.")

    return all_outputs


# ============================================================
# QTN / GAF / MTF
# ============================================================
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
# Compute all methods for one cleaned modality
# ============================================================
def cleaned_matrix_to_all_method_rows(sig: np.ndarray, fs: float, seed_base: int) -> Dict[str, dict]:
    sig = slice_signals(sig)

    T = sig.shape[0]
    if T < 50:
        raise ValueError("Too few samples after slicing.")
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
    for method in METHODS:
        if not per_method_metrics[method]:
            raise ValueError(f"No valid channels produced metrics for {method}.")
        df = pd.DataFrame(per_method_metrics[method])
        avg = df.mean(numeric_only=True).to_dict()
        avg.update({
            "fs_hz": float(fs),
            "n_channels": int(sig.shape[1]),
            "T_used_samples": int(T),
            "Q_used": int(Q),
            "sleep_selection": SLEEP_SELECTION,
        })
        out[method] = avg

    return out


# ============================================================
# IO helpers
# ============================================================
def out_paths(out_dir: str, dataset_name: str, modality: str, sleep_selection: str):
    suffix = f"{dataset_name}_{modality}_{sleep_selection}"
    return {
        "QTN": os.path.join(out_dir, f"metrics_QTN_{suffix}.csv"),
        "GAF": os.path.join(out_dir, f"metrics_GAF_{suffix}.csv"),
        "MTF": os.path.join(out_dir, f"metrics_MTF_{suffix}.csv"),
        "SKIP": os.path.join(out_dir, f"skipped_{suffix}.csv"),
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


# ============================================================
# Compute one subject once
# ============================================================
def compute_subject_once(subject_id: str):
    local_files = {}
    try:
        local_files = fetch_subject(subject_id, TMP_DIR)
        edf_path = local_files["edf"]
        txt_path = local_files["txt"]

        modality_data = load_subject_modalities_once(
            edf_path=edf_path,
            txt_path=txt_path,
            sleep_selection=SLEEP_SELECTION,
        )

        result = {}
        for mi, modality in enumerate(MODALITIES):
            if modality not in modality_data:
                result[modality] = {"rows": None, "skip": {
                    "dataset": DATASET_NAME,
                    "channel_mode": modality,
                    "sleep_selection": SLEEP_SELECTION,
                    "method": "ALL",
                    "record_or_patient": subject_id,
                    "reason": f"No valid channels found for modality {modality}",
                }}
                continue

            sig, names, fs = modality_data[modality]
            rows = cleaned_matrix_to_all_method_rows(
                sig=sig,
                fs=fs,
                seed_base=RNG_SEED + 100000 * (int(subject_id[1:])) + 1000 * mi,
            )
            for method in METHODS:
                rows[method].update({
                    "patient_id": subject_id,
                    "source_file": edf_path,
                    "dataset": DATASET_NAME,
                    "channel_mode": modality,
                })
            result[modality] = {"rows": rows, "skip": None}

        return subject_id, result
    finally:
        if DELETE_RAW_AFTER_PROCESS and local_files:
            cleanup_subject(local_files)


# ============================================================
# Runner
# ============================================================
def run_all_modalities():
    ensure_out_dir(TMP_DIR)

    paths_by_modality = {}
    done_all = {}

    for modality in MODALITIES:
        out_dir = os.path.join(BASE_OUT_DIR, DATASET_NAME, modality)
        ensure_out_dir(out_dir)
        paths_by_modality[modality] = out_paths(out_dir, DATASET_NAME, modality, SLEEP_SELECTION)

        done = {
            m: load_done_ids(paths_by_modality[modality][m], id_col="patient_id")
            for m in METHODS
        }
        done_all[modality] = done["QTN"].intersection(done["GAF"]).intersection(done["MTF"])

    subjects_to_process = []
    for subject_id in CONTROL_IDS:
        still_needed = False
        for modality in MODALITIES:
            if subject_id not in done_all[modality]:
                still_needed = True
                break
        if still_needed:
            subjects_to_process.append(subject_id)

    print(f"[{DATASET_NAME} | {SLEEP_SELECTION}] total subjects={len(CONTROL_IDS)} | pending subjects={len(subjects_to_process)}")

    iterator = subjects_to_process
    if TQDM_ENABLED:
        iterator = tqdm(subjects_to_process, desc="CAP subjects", unit="subject", leave=TQDM_LEAVE)

    for subject_id in iterator:
        try:
            _, result = compute_subject_once(subject_id)

            for modality in MODALITIES:
                paths = paths_by_modality[modality]
                rows = result[modality]["rows"]
                skip = result[modality]["skip"]

                if rows is not None:
                    for method in METHODS:
                        if subject_id not in done_all[modality]:
                            append_rows(paths[method], [rows[method]], id_col="patient_id")

                if skip is not None:
                    append_skips(paths["SKIP"], [skip])

        except Exception as e:
            for modality in MODALITIES:
                paths = paths_by_modality[modality]
                skip = {
                    "dataset": DATASET_NAME,
                    "channel_mode": modality,
                    "sleep_selection": SLEEP_SELECTION,
                    "method": "ALL",
                    "record_or_patient": subject_id,
                    "reason": str(e),
                }
                append_skips(paths["SKIP"], [skip])

    print(f"[{DATASET_NAME} | {SLEEP_SELECTION}] done.")


if __name__ == "__main__":
    run_all_modalities()

# In[4]:

#!/usr/bin/env python3
# cap_sleep_controls_nrem_parallel_subjects_pubfast.py
#
# Parallel CAP healthy-control pipeline (n1-n16), publication-grade preprocessing, faster:
# - Parallel at subject level (joblib)
# - Each worker downloads/processes one subject into per-modality rows (QTN/GAF/MTF)
# - Main process appends CSVs safely (no concurrent file writes)
#
# Key speedups:
# - IIR filtering (bandpass + notch) instead of FIR
# - Load EDF once per subject
# - Parse annotations once per subject
# - Build NREM mask once per subject
# - Preprocess each modality once per subject
# - Compute QTN/GAF/MTF together in one pass per channel
#
# Outputs (per modality folder):
#   sleep_outputs/cap_sleep_controls/EEG/metrics_QTN_cap_sleep_controls_EEG_NREM.csv
#   sleep_outputs/cap_sleep_controls/EEG/metrics_GAF_cap_sleep_controls_EEG_NREM.csv
#   sleep_outputs/cap_sleep_controls/EEG/metrics_MTF_cap_sleep_controls_EEG_NREM.csv
#   sleep_outputs/cap_sleep_controls/EEG/skipped_cap_sleep_controls_EEG_NREM.csv
# ... similarly for EMG, RESP, ALL_SLEEP
#
# Requirements:
#   pip install requests mne numpy pandas networkx joblib tqdm

import os
import re
import shutil
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from urllib.parse import urljoin

import requests
import numpy as np
import pandas as pd
import networkx as nx
import mne

from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIG
# ============================================================
BASE_URL = "https://physionet.org/files/capslpdb/1.0.0/"

DATASET_NAME = "cap_sleep_controls"
BASE_OUT_DIR = "sleep_outputs"
TMP_DIR = "tmp_cap_sleep"

CONTROL_IDS = [f"n{i}" for i in range(1, 17)]   # n1..n16
MODALITIES = ["EEG", "EMG", "RESP", "ALL_SLEEP"]

SLEEP_SELECTION = "NREM"   # "NREM" or "WHOLE_NIGHT"
DELETE_RAW_AFTER_PROCESS = True

REQUEST_TIMEOUT = 180
DOWNLOAD_RETRIES = 5
RETRY_SLEEP_SEC = 2.0

# QTN / GAF / MTF
K_VALUES = [1, 2, 3]
RNG_SEED = 1234

# Small-world (speed/rigor tradeoff)
TARGET_DENSITY = 0.10
N_RANDOMIZATIONS = 5
REWIRINGS_PER_EDGE = 5

# Data slicing (debug)
MAX_SAMPLES = None   # e.g. 200000 for quick tests
START_SAMPLE = 0

# Parallel
N_JOBS = 2
BACKEND = "loky"

METHODS = ["QTN", "GAF", "MTF"]

# Preprocessing / QC
TARGET_FS = 128.0
LINE_FREQ = 50.0
EPOCH_SEC = 30.0
MIN_KEEP_EPOCHS = 5

# Filter config (IIR is MUCH faster than FIR; still publishable if you report it)
FILTER_METHOD = "iir"  # "iir" (fast) or "fir" (slow)
IIR_PARAMS = dict(order=4, ftype="butter")  # stable default

PREPROC = {
    "EEG": {"l_freq": 0.5, "h_freq": 40.0, "notch": True},
    "EMG": {"l_freq": 10.0, "h_freq": 45.0, "notch": True},
    "RESP": {"l_freq": 0.01, "h_freq": 1.0, "notch": False},
    "ALL_SLEEP": {"l_freq": None, "h_freq": None, "notch": False},
}

CHANNEL_REGEX = {
    "EEG": r"EEG|Fpz|Pz|C3|C4|F3|F4|O1|O2",
    "EMG": r"EMG|Chin|Submental|Submentalis|Tibial",
    "RESP": r"Resp|Airflow|Nasal|Thor|Abdo|Abdominal|Thoracic|SaO2|SpO2|Flow",
    "ALL_SLEEP": r"EEG|EOG|EMG|Chin|Submental|Submentalis|Tibial|Resp|Airflow|Nasal|Thor|Abdo|Abdominal|Thoracic|SaO2|SpO2|Flow",
}

NREM_EVENTS = {"SLEEP-S1", "SLEEP-S2", "SLEEP-S3", "SLEEP-S4"}


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
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        std = np.nanstd(x)
        scale = std if np.isfinite(std) and std > 0 else 1.0
    z = (x - med) / scale
    return np.clip(z, -10, 10)

def select_channels(ch_names: List[str], channel_mode: str) -> List[int]:
    pat = re.compile(CHANNEL_REGEX[channel_mode], flags=re.IGNORECASE)
    return [i for i, ch in enumerate(ch_names) if pat.search(str(ch))]

def channel_family(name: str) -> str:
    s = str(name).lower()
    if re.search(r"emg|chin|submental|submentalis|tibial", s):
        return "EMG"
    if re.search(r"resp|airflow|nasal|thor|abdo|abdominal|thoracic|sao2|spo2|flow", s):
        return "RESP"
    return "EEG"


# ============================================================
# Download (retry-safe)
# ============================================================
def download_file(url: str, local_path: str, session: requests.Session):
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return

    ensure_out_dir(str(Path(local_path).parent))

    last_err: Optional[Exception] = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
                r.raise_for_status()
                tmp = local_path + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, local_path)
            return
        except Exception as e:
            last_err = e
            time.sleep(RETRY_SLEEP_SEC * attempt)

    raise RuntimeError(f"Failed to download {url} -> {local_path}. Last error: {last_err}")

def subject_files(subject_id: str) -> Dict[str, str]:
    return {
        "edf": f"{subject_id}.edf",
        "txt": f"{subject_id}.txt",
        "st": f"{subject_id}.edf.st",  # optional
    }

def fetch_subject(subject_id: str, tmp_subject_dir: str) -> Dict[str, str]:
    session = requests.Session()
    files = subject_files(subject_id)
    local: Dict[str, str] = {}

    for key, fname in files.items():
        url = urljoin(BASE_URL, fname)
        local_path = os.path.join(tmp_subject_dir, fname)
        try:
            download_file(url, local_path, session=session)
            local[key] = local_path
        except Exception:
            if key == "st":
                continue
            raise

    return local

def cleanup_subject_dir(tmp_subject_dir: str):
    try:
        if os.path.exists(tmp_subject_dir):
            shutil.rmtree(tmp_subject_dir)
    except Exception:
        pass


# ============================================================
# CAP annotation parsing
# ============================================================
def parse_cap_stage_intervals(txt_path: str) -> List[Tuple[float, float, str]]:
    intervals: List[Tuple[float, float, str]] = []
    current_start = 0.0
    in_table = False

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s.startswith("Sleep Stage"):
                in_table = True
                continue
            if not in_table:
                continue

            parts = re.split(r"\t+", s)
            if len(parts) < 5:
                continue

            event = parts[3].strip()
            dur_str = parts[4].strip()

            if event not in {"SLEEP-S0", "SLEEP-S1", "SLEEP-S2", "SLEEP-S3", "SLEEP-S4", "SLEEP-REM", "MT"}:
                continue

            try:
                dur = float(dur_str)
            except Exception:
                continue

            intervals.append((current_start, current_start + dur, event))
            current_start += dur

    if not intervals:
        raise ValueError(f"No sleep-stage intervals found in {txt_path}")

    return intervals

def build_sample_mask_from_intervals(n_times: int, sfreq: float, intervals, sleep_selection: str) -> np.ndarray:
    mask = np.zeros(n_times, dtype=bool)

    if sleep_selection == "WHOLE_NIGHT":
        mask[:] = True
        return mask

    if sleep_selection != "NREM":
        raise ValueError(f"Unknown sleep_selection: {sleep_selection}")

    for start_sec, end_sec, event in intervals:
        if event in NREM_EVENTS:
            i0 = max(0, int(round(start_sec * sfreq)))
            i1 = min(n_times, int(round(end_sec * sfreq)))
            if i1 > i0:
                mask[i0:i1] = True

    return mask


# ============================================================
# Preprocessing (fast IIR)
# ============================================================
def _notch_1d(x: np.ndarray, sfreq: float) -> np.ndarray:
    # mne notch_filter can do iir and is faster than fir
    return mne.filter.notch_filter(
        x, Fs=sfreq, freqs=[LINE_FREQ],
        method=FILTER_METHOD,
        iir_params=IIR_PARAMS if FILTER_METHOD == "iir" else None,
        phase="zero",
        verbose=False
    )

def _bandpass_1d(x: np.ndarray, sfreq: float, l_freq, h_freq) -> np.ndarray:
    return mne.filter.filter_data(
        x,
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        method=FILTER_METHOD,
        iir_params=IIR_PARAMS if FILTER_METHOD == "iir" else None,
        phase="zero",
        verbose=False,
    )

def preprocess_matrix(
    data_ch_by_t: np.ndarray,  # (n_channels, T)
    sfreq: float,
    names: List[str],
    mode: str
) -> Tuple[np.ndarray, List[str], float, dict]:
    """
    Returns:
      cleaned_T_by_ch: (T_clean, n_channels)
      names
      sfreq_used
      qc dict (epochs kept, etc.)
    """
    sfreq = float(sfreq)

    # Resample once (fast path)
    if sfreq > TARGET_FS:
        data_ch_by_t = mne.filter.resample(data_ch_by_t, down=sfreq / TARGET_FS, npad="auto", axis=1)
        sfreq = float(TARGET_FS)

    out = np.zeros_like(data_ch_by_t, dtype=float)

    for i, ch_name in enumerate(names):
        fam = mode if mode != "ALL_SLEEP" else channel_family(ch_name)
        params = PREPROC[fam]
        x = data_ch_by_t[i].astype(float)

        # Notch only where meaningful
        if params["notch"] and sfreq > 2 * LINE_FREQ:
            x = _notch_1d(x, sfreq)

        if params["l_freq"] is not None or params["h_freq"] is not None:
            x = _bandpass_1d(x, sfreq, params["l_freq"], params["h_freq"])

        out[i] = robust_zscore(x)

    # Epoch-based artifact rejection (publication-friendly + simple)
    epoch_len = max(1, int(round(EPOCH_SEC * sfreq)))
    n_times = out.shape[1]
    n_epochs = n_times // epoch_len
    if n_epochs == 0:
        raise ValueError("Signal too short after preprocessing.")

    out = out[:, : n_epochs * epoch_len]
    epochs = out.reshape(out.shape[0], n_epochs, epoch_len)
    ptp = np.ptp(epochs, axis=2)  # (n_channels, n_epochs)

    keep_mask = np.ones(n_epochs, dtype=bool)
    for ch in range(ptp.shape[0]):
        vals = ptp[ch]
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        scale = 1.4826 * mad if mad > 0 else float(np.std(vals)) if np.std(vals) > 0 else 1.0
        thr = med + 6.0 * scale
        keep_mask &= (vals <= thr)

    n_keep = int(keep_mask.sum())
    if n_keep < MIN_KEEP_EPOCHS:
        raise ValueError(f"Too few clean epochs kept ({n_keep}).")

    cleaned = epochs[:, keep_mask, :].reshape(out.shape[0], -1)  # (n_channels, T_clean)

    qc = dict(
        sfreq_used=float(sfreq),
        epoch_sec=float(EPOCH_SEC),
        n_epochs_total=int(n_epochs),
        n_epochs_kept=int(n_keep),
        pct_epochs_kept=float(100.0 * n_keep / n_epochs),
        n_channels=int(out.shape[0]),
        T_clean_samples=int(cleaned.shape[1]),
    )

    return cleaned.T, names, sfreq, qc  # (T_clean, n_channels)


def load_subject_modalities_once(edf_path: str, txt_path: str, sleep_selection: str) -> Dict[str, Tuple[np.ndarray, List[str], float, dict]]:
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    ch_names = raw.ch_names

    intervals = parse_cap_stage_intervals(txt_path)
    sfreq_orig = float(raw.info["sfreq"])

    mask = build_sample_mask_from_intervals(raw.n_times, sfreq_orig, intervals, sleep_selection)
    if mask.sum() == 0:
        raise ValueError(f"No samples kept for sleep_selection={sleep_selection} in {edf_path}")

    outputs: Dict[str, Tuple[np.ndarray, List[str], float, dict]] = {}

    for mode in MODALITIES:
        idx = select_channels(ch_names, mode)
        if not idx:
            continue

        data = raw.get_data(picks=idx)  # (n_channels, T)
        data = data[:, mask]            # apply NREM mask
        picked_names = [ch_names[i] for i in idx]

        sig_T_by_ch, sel_names, fs_used, qc = preprocess_matrix(data, sfreq_orig, picked_names, mode)
        outputs[mode] = (sig_T_by_ch, sel_names, fs_used, qc)

    if not outputs:
        raise ValueError("No valid modality channels found.")

    return outputs


# ============================================================
# QTN / GAF / MTF representations
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
# Compute QTN/GAF/MTF in one pass over channels (fast)
# ============================================================
def cleaned_matrix_to_all_method_rows(sig_T_by_ch: np.ndarray, fs: float, seed_base: int) -> Dict[str, dict]:
    sig_T_by_ch = slice_signals(sig_T_by_ch)

    T = sig_T_by_ch.shape[0]
    if T < 50:
        raise ValueError("Too few samples after slicing.")

    Q = compute_Q_from_T(T)
    if Q < 3:
        raise ValueError(f"Q too small: {Q}")

    per_method = {m: [] for m in METHODS}

    for li in range(sig_T_by_ch.shape[1]):
        x = sig_T_by_ch[:, li].astype(float)
        x = x[np.isfinite(x)]
        if x.size < 50:
            continue

        # QTN
        A_qtn = calculate_quantile_graph_varying_k(x, Q=Q, k_values=K_VALUES)
        W_qtn = (A_qtn + A_qtn.T).astype(float)
        per_method["QTN"].append(
            compute_smallworld_metrics_from_W(W_qtn, seed=seed_base + 100 * li + 1, use_abs_for_threshold=False)
        )

        # GAF + MTF share xQ
        xQ = downsample_to_length(x, Q)

        W_gaf = calculate_gaf_from_lengthQ(xQ)
        per_method["GAF"].append(
            compute_smallworld_metrics_from_W(W_gaf, seed=seed_base + 100 * li + 2, use_abs_for_threshold=True)
        )

        W_mtf = calculate_mtf_from_lengthQ(xQ, Q=Q)
        per_method["MTF"].append(
            compute_smallworld_metrics_from_W(W_mtf, seed=seed_base + 100 * li + 3, use_abs_for_threshold=False)
        )

    out: Dict[str, dict] = {}
    for method in METHODS:
        if not per_method[method]:
            raise ValueError(f"No valid channels produced metrics for {method}.")
        df = pd.DataFrame(per_method[method])
        avg = df.mean(numeric_only=True).to_dict()
        avg.update({
            "fs_hz": float(fs),
            "n_channels": int(sig_T_by_ch.shape[1]),
            "T_used_samples": int(T),
            "Q_used": int(Q),
            "sleep_selection": SLEEP_SELECTION,
        })
        out[method] = avg

    return out


# ============================================================
# IO helpers (main process only)
# ============================================================
def out_paths(out_dir: str, dataset_name: str, modality: str, sleep_selection: str):
    suffix = f"{dataset_name}_{modality}_{sleep_selection}"
    return {
        "QTN": os.path.join(out_dir, f"metrics_QTN_{suffix}.csv"),
        "GAF": os.path.join(out_dir, f"metrics_GAF_{suffix}.csv"),
        "MTF": os.path.join(out_dir, f"metrics_MTF_{suffix}.csv"),
        "SKIP": os.path.join(out_dir, f"skipped_{suffix}.csv"),
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


# ============================================================
# Worker: compute one subject once (returns rows + qc)
# ============================================================
def compute_subject_once(subject_id: str):
    tmp_subject_dir = os.path.join(TMP_DIR, subject_id)
    ensure_out_dir(tmp_subject_dir)

    local_files: Dict[str, str] = {}
    try:
        local_files = fetch_subject(subject_id, tmp_subject_dir)
        edf_path = local_files["edf"]
        txt_path = local_files["txt"]

        modality_data = load_subject_modalities_once(edf_path, txt_path, sleep_selection=SLEEP_SELECTION)

        sid_num = int(subject_id[1:])
        result = {}

        for mi, modality in enumerate(MODALITIES):
            if modality not in modality_data:
                result[modality] = dict(rows=None, skip={
                    "dataset": DATASET_NAME,
                    "channel_mode": modality,
                    "sleep_selection": SLEEP_SELECTION,
                    "method": "ALL",
                    "record_or_patient": subject_id,
                    "reason": f"No valid channels found for modality {modality}",
                })
                continue

            sig, sel_names, fs, qc = modality_data[modality]

            rows = cleaned_matrix_to_all_method_rows(
                sig_T_by_ch=sig,
                fs=fs,
                seed_base=RNG_SEED + 100000 * sid_num + 1000 * mi,
            )

            # Attach metadata + QC
            for method in METHODS:
                rows[method].update({
                    "patient_id": subject_id,
                    "dataset": DATASET_NAME,
                    "channel_mode": modality,
                    "source_file": os.path.basename(edf_path),
                    "filter_method": FILTER_METHOD,
                    "target_fs": float(TARGET_FS),
                    "epoch_sec": float(EPOCH_SEC),
                    "n_epochs_total": qc["n_epochs_total"],
                    "n_epochs_kept": qc["n_epochs_kept"],
                    "pct_epochs_kept": qc["pct_epochs_kept"],
                    "T_clean_samples": qc["T_clean_samples"],
                })

            result[modality] = dict(rows=rows, skip=None)

        return subject_id, result

    except Exception as e:
        # Return one global failure object; main will write one skip per modality
        return subject_id, {"__fatal__": str(e)}

    finally:
        if DELETE_RAW_AFTER_PROCESS:
            cleanup_subject_dir(tmp_subject_dir)


# ============================================================
# Runner (parallel subjects, main writes)
# ============================================================
def run_all_modalities():
    ensure_out_dir(TMP_DIR)

    paths_by_modality: Dict[str, dict] = {}
    done_all: Dict[str, set] = {}

    for modality in MODALITIES:
        out_dir = os.path.join(BASE_OUT_DIR, DATASET_NAME, modality)
        ensure_out_dir(out_dir)
        paths_by_modality[modality] = out_paths(out_dir, DATASET_NAME, modality, SLEEP_SELECTION)

        done = {m: load_done_ids(paths_by_modality[modality][m], id_col="patient_id") for m in METHODS}
        done_all[modality] = done["QTN"].intersection(done["GAF"]).intersection(done["MTF"])

    subjects_to_process = []
    for subject_id in CONTROL_IDS:
        if any(subject_id not in done_all[mod] for mod in MODALITIES):
            subjects_to_process.append(subject_id)

    print(
        f"[{DATASET_NAME} | {SLEEP_SELECTION}] total subjects={len(CONTROL_IDS)} | "
        f"pending subjects={len(subjects_to_process)} | n_jobs={N_JOBS} | filter={FILTER_METHOD}"
    )

    if not subjects_to_process:
        print("Nothing to do.")
        return

    results = Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=0)(
        delayed(compute_subject_once)(sid)
        for sid in tqdm(subjects_to_process, desc="CAP subjects", unit="subject", leave=True)
    )

    # Main process writes (safe)
    for subject_id, payload in results:
        if "__fatal__" in payload:
            err = payload["__fatal__"]
            for modality in MODALITIES:
                paths = paths_by_modality[modality]
                skip = {
                    "dataset": DATASET_NAME,
                    "channel_mode": modality,
                    "sleep_selection": SLEEP_SELECTION,
                    "method": "ALL",
                    "record_or_patient": subject_id,
                    "reason": err,
                }
                append_skips(paths["SKIP"], [skip])
            continue

        for modality in MODALITIES:
            paths = paths_by_modality[modality]
            pack = payload.get(modality, None)
            if pack is None:
                continue

            rows = pack["rows"]
            skip = pack["skip"]

            if rows is not None and subject_id not in done_all[modality]:
                for method in METHODS:
                    append_rows(paths[method], [rows[method]], id_col="patient_id")

            if skip is not None:
                append_skips(paths["SKIP"], [skip])

    print(f"[{DATASET_NAME} | {SLEEP_SELECTION}] done. Outputs in: {os.path.join(BASE_OUT_DIR, DATASET_NAME)}")


if __name__ == "__main__":
    run_all_modalities()

# In[ ]:
