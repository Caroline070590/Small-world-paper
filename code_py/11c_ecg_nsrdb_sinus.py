#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto-converted from the Jupyter notebook ``ECG-sinus.ipynb``.

Code cells are reproduced verbatim. Markdown cells are kept as comments.
Jupyter shell-magic lines (if any) are commented out so the file is valid
Python; they are preserved for reference. Cell boundaries are marked with
``# In[...]`` to match the original notebook ordering.
"""


# In[1]:

# nsrdb_qtn_smallworld_per_patient.py
# Entire MIT-BIH Normal Sinus Rhythm Database (normal patients)
# For each record: compute QTN per lead -> small-world metrics -> average across leads -> one row per patient

import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import wfdb

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =======================
# CONFIG
# =======================
PN_DIR = "nsrdb"  # PhysioNet directory
OUT_CSV = "nsrdb_QTN_smallworld_per_patient.csv"

# Choose how much data per record (to keep runtime reasonable).
# NSRDB records are long (~24h). If you truly want "entire recording", set MAX_SAMPLES=None,
# but computing QTN on extremely long signals is heavy.
MAX_SAMPLES = 200_000   # e.g., ~26 min at 128 Hz. Set None for full length (not recommended).
START_SAMPLE = 0

# QTN lags
K_VALUES = [1, 2, 3]

# Graph / null model settings
TARGET_DENSITY      = 0.10
N_RANDOMIZATIONS    = 20
REWIRINGS_PER_EDGE  = 5
RNG_SEED            = 1234

# =======================
# QTN / GRAPH FUNCTIONS
# =======================
def compute_Q_from_T(T: int) -> int:
    # same rule you used
    return int(round(2 * (T ** (1/3))))

def calculate_quantile_graph_varying_k(signal: np.ndarray, Q: int, k_values):
    """
    Quantile Transition Network (QTN): bins timepoints into Q quantiles (by rank),
    counts transitions at lags k in k_values.
    Returns QxQ count matrix.
    """
    A = np.zeros((Q, Q), dtype=np.int64)
    n = int(signal.size)
    if n == 0 or Q <= 0:
        return A

    # rank-based quantile binning (robust to outliers)
    ranks = np.argsort(np.argsort(signal))
    q_edges = np.linspace(0, n, Q + 1, dtype=int)
    loc = np.clip(np.searchsorted(q_edges, ranks, side="right") - 1, 0, Q - 1)

    for k in k_values:
        k = int(k)
        if k <= 0 or k >= n:
            continue
        # transitions i -> i+k
        for i in range(n - k):
            A[loc[i], loc[i + k]] += 1
    return A

def proportional_binary_from_weights(W: np.ndarray, target_density: float) -> np.ndarray:
    n = W.shape[0]
    A = W.astype(float).copy()
    np.fill_diagonal(A, 0.0)

    upper_vals = np.abs(np.triu(A, 1))
    vals = upper_vals[upper_vals > 0]
    if vals.size == 0:
        return np.zeros_like(A, dtype=int)

    m_target = int(round(target_density * n * (n - 1) / 2))
    m_target = max(1, min(m_target, vals.size))
    thresh = np.partition(vals, -m_target)[-m_target]

    B = (np.abs(A) >= thresh).astype(int)
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
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(evals * np.log(evals + 1e-15))
    return float(h / np.log(n))

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
        if len(arr) == 0:
            return (np.nan, np.nan)
        return (float(np.mean(arr)), float(np.std(arr) if len(arr) > 1 else 0.0))

    Cmu, Csd = _ms(Cr)
    Lmu, Lsd = _ms(Lr)
    Emu, Esd = _ms(Er)
    return Cmu, Csd, Lmu, Lsd, Emu, Esd

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

def compute_metrics_size_robust(W: np.ndarray, seed: int) -> dict:
    n = W.shape[0]
    np.fill_diagonal(W, 0.0)

    # for QTN we don't use abs (counts are >=0)
    B = proportional_binary_from_weights(W, TARGET_DENSITY)

    degrees = np.array([d for _, d in nx.degree(nx.from_numpy_array(B))], dtype=float)
    avg_deg = float(degrees.mean()) if degrees.size else np.nan
    density = float(np.sum(B) / (n * (n - 1))) if n > 1 else np.nan

    C_obs = nx.transitivity(nx.from_numpy_array(B)) if n > 1 else np.nan
    L_obs = gcc_char_path_length_binary(B)
    E_obs = global_efficiency_binary(B)

    Cmu, Csd, Lmu, Lsd, Emu, Esd = null_model_stats(B, N_RANDOMIZATIONS, REWIRINGS_PER_EDGE, seed)

    gamma = (C_obs / Cmu) if (not np.isnan(Cmu) and Cmu > 0) else np.nan
    lambd = (L_obs / Lmu) if (not np.isnan(Lmu) and Lmu > 0) else np.nan
    sigma = (gamma / lambd) if (not np.isnan(gamma) and not np.isnan(lambd) and lambd != 0) else np.nan

    zC = ((C_obs - Cmu) / Csd) if (not np.isnan(Csd) and Csd > 0) else np.nan
    zL = ((L_obs - Lmu) / Lsd) if (not np.isnan(Lsd) and Lsd > 0) else np.nan

    Enorm = (E_obs / Emu) if (not np.isnan(Emu) and Emu > 0) else np.nan
    Hvn = von_neumann_entropy_normalized(B)
    eigs = normalized_laplacian_eigs(B)
    lambda2 = float(eigs[1]) if eigs.size >= 2 else np.nan

    omega, phi = small_world_omega_phi(B, C_obs, L_obs, Cmu, Lmu)

    return {
        "n_nodes": n,
        "density": density,
        "avg_degree": avg_deg,
        "transitivity": C_obs,
        "global_efficiency": E_obs,
        "char_path_len_gcc": L_obs,
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
# DATASET DRIVER
# =======================
def get_nsrdb_record_list():
    """
    Prefer PhysioNet record listing file via wfdb (if available).
    If it fails, fall back to the canonical NSRDB record IDs commonly used.
    """
    try:
        # wfdb can list records hosted in a PhysioNet directory
        recs = wfdb.get_record_list(PN_DIR)
        return sorted(recs)
    except Exception:
        # fallback (widely-used NSRDB record IDs)
        return sorted([
            "16265","16272","16273","16420","16483","16539","16773","16786","16795",
            "17052","17453","18177","18184","19088","19090","19093","19140","19830"
        ])

def load_record_signals(record_name: str):
    record = wfdb.rdrecord(record_name, pn_dir=PN_DIR)
    sig = record.p_signal  # shape: (T, n_leads)
    fs = getattr(record, "fs", np.nan)
    lead_names = getattr(record, "sig_name", [f"ch{i}" for i in range(sig.shape[1])])
    return sig, fs, lead_names

def slice_signal(sig: np.ndarray):
    if START_SAMPLE is None:
        start = 0
    else:
        start = int(START_SAMPLE)

    if MAX_SAMPLES is None:
        return sig[start:, :]
    else:
        end = start + int(MAX_SAMPLES)
        return sig[start:end, :]

def per_record_patient_metrics(record_name: str, idx: int) -> dict:
    sig, fs, lead_names = load_record_signals(record_name)
    sig = slice_signal(sig)

    # per lead metrics
    lead_metrics = []
    T = sig.shape[0]
    if T < 10:
        raise ValueError("Too few samples after slicing.")

    Q = compute_Q_from_T(T)
    if Q < 3:
        raise ValueError(f"Computed Q too small: Q={Q}")

    for lead_i in range(sig.shape[1]):
        x = sig[:, lead_i].astype(float)
        # drop NaNs if any
        x = x[np.isfinite(x)]
        if x.size < 10:
            continue

        Q_eff = min(Q, x.size)  # just in case
        A = calculate_quantile_graph_varying_k(x, Q=Q_eff, k_values=K_VALUES)
        W = (A + A.T).astype(float)

        if np.allclose(W, 0):
            # all-zero -> metrics NaN
            m = {k: np.nan for k in compute_metrics_size_robust(np.eye(3), seed=0).keys()}
        else:
            m = compute_metrics_size_robust(W, seed=RNG_SEED + 10_000 * idx + lead_i)

        m["lead"] = lead_names[lead_i] if lead_i < len(lead_names) else f"ch{lead_i}"
        lead_metrics.append(m)

    if len(lead_metrics) == 0:
        raise ValueError("No valid leads produced metrics.")

    df = pd.DataFrame(lead_metrics).drop(columns=["lead"])

    # average across electrodes -> one patient value
    avg = df.mean(numeric_only=True).to_dict()

    avg.update({
        "record_id": record_name,
        "fs_hz": float(fs) if fs is not None else np.nan,
        "n_leads": int(sig.shape[1]),
        "T_used_samples": int(sig.shape[0]),
        "Q_used": int(compute_Q_from_T(sig.shape[0])),
    })
    return avg

def main():
    records = get_nsrdb_record_list()
    rows = []
    skipped = []

    for idx, rec in enumerate(records, start=1):
        try:
            row = per_record_patient_metrics(rec, idx)
            rows.append(row)
            print(f"[OK] {rec} (leads={row['n_leads']}, T={row['T_used_samples']}, Q={row['Q_used']})")
        except Exception as e:
            skipped.append({"record_id": rec, "reason": str(e)})
            print(f"[SKIP] {rec}: {e}")

    df_out = pd.DataFrame(rows).set_index("record_id").sort_index()
    df_out.to_csv(OUT_CSV)
    print("Saved:", OUT_CSV, df_out.shape)

    if skipped:
        pd.DataFrame(skipped).to_csv("nsrdb_skipped.csv", index=False)
        print("Also saved: nsrdb_skipped.csv", len(skipped))

if __name__ == "__main__":
    main()

# In[ ]:



# In[ ]:
