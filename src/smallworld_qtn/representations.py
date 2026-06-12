"""Time-series-to-matrix representations: QG/QTN, GAF, and MTF.

These functions are the time-series-to-network mappings described in the paper
(Methods, Section II A). They are extracted verbatim from the project notebooks
so that the packaged code reproduces the published results exactly.

Two GAF/MTF flavours are kept intentionally, because the analysis pipelines use
them in different places and they are NOT interchangeable:

* ``calculate_gaf`` / ``calculate_mtf``
      Compute the field on the full-length signal and then crop to the top-left
      ``Q x Q`` block. Used by the fMRI (BASC-122) pipeline
      (``SMALL-world-FINAL`` notebook).

* ``calculate_gaf_from_lengthQ`` / ``calculate_mtf_from_lengthQ``
      Compute the field on a signal that has already been resampled to length
      ``Q`` (see :func:`downsample_to_length`), returning a ``Q x Q`` matrix
      directly. Used by the ECG, EMG, respiratory, sleep and MEG pipelines.

The QG/QTN construction (multi-lag quantile graph, ``K = {1, 2, 3}``) is shared.

Do not "unify" the two flavours without re-running the affected pipelines: they
produce different matrices and therefore different network metrics.
"""

from __future__ import annotations

import numpy as np

# Temporal lags aggregated by the quantile graph (Methods, Eq. 3).
K_VALUES = [1, 2, 3]


def compute_Q_from_T(T: int) -> int:
    """Number of quantile bins from signal length (Methods, Eq. 1).

    ``Q ~= 2 * T**(1/3)``.
    """
    return int(round(2 * (T ** (1 / 3))))


def downsample_to_length(x: np.ndarray, L: int) -> np.ndarray:
    """Linear interpolation of ``x`` onto ``L`` equally spaced points."""
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


# ---------------------------------------------------------------------------
# Quantile Graph (QG / QTN)
# ---------------------------------------------------------------------------
def calculate_quantile_graph_varying_k(signal, Q, k_values=K_VALUES) -> np.ndarray:
    """Directed ``Q x Q`` transition-count matrix of the quantile graph.

    Each sample is mapped to a quantile-defined state and transitions are
    counted across the lags in ``k_values`` (Methods, Eqs. 2-3). Symmetrise
    ``A + A.T`` before computing undirected graph metrics.
    """
    A = np.zeros((Q, Q), dtype=np.int64)
    n = int(np.asarray(signal).size)
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


# ---------------------------------------------------------------------------
# Gramian Angular Field (GAF)
# ---------------------------------------------------------------------------
def calculate_gaf(signal, Q) -> np.ndarray:
    """GAF on the full signal, cropped to the ``Q x Q`` block (fMRI pipeline)."""
    signal = np.asarray(signal, dtype=float)
    min_val, max_val = float(np.min(signal)), float(np.max(signal))
    rng = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    scaled = 2 * (signal - min_val) / rng - 1
    scaled = np.clip(scaled, -1, 1)
    phi = np.arccos(scaled)
    gaf = np.cos(phi[:, None] + phi[None, :])
    return gaf[:Q, :Q]


def calculate_gaf_from_lengthQ(signal_Q) -> np.ndarray:
    """GAF on a signal already resampled to length ``Q`` -> ``Q x Q``.

    Used by the ECG/EMG/respiratory/sleep/MEG pipelines.
    """
    x = np.asarray(signal_Q, dtype=float)
    min_val, max_val = float(np.min(x)), float(np.max(x))
    rng = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    scaled = 2 * (x - min_val) / rng - 1
    scaled = np.clip(scaled, -1, 1)
    phi = np.arccos(scaled)
    return np.cos(phi[:, None] + phi[None, :])


# ---------------------------------------------------------------------------
# Markov Transition Field (MTF)
# ---------------------------------------------------------------------------
def calculate_mtf(signal, Q) -> np.ndarray:
    """MTF on the full signal, cropped to the ``Q x Q`` block (fMRI pipeline)."""
    signal = np.asarray(signal, dtype=float)
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


def calculate_mtf_from_lengthQ(signal_Q, Q) -> np.ndarray:
    """MTF on a signal already resampled to length ``Q`` -> ``Q x Q``.

    Used by the ECG/EMG/respiratory/sleep/MEG pipelines.
    """
    x = np.asarray(signal_Q, dtype=float)
    min_val, max_val = float(np.min(x)), float(np.max(x))
    rng = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    norm_signal = (x - min_val) / rng  # in [0, 1]

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
