"""Cleaned signals -> representation -> network metrics.

This module wires together :mod:`smallworld_qtn.representations` and
:mod:`smallworld_qtn.network_metrics`. It exposes two helpers that mirror the
two representation flavours used across the project:

* :func:`metrics_for_signal_fmri`
      builds ``W`` with the full-signal GAF/MTF (cropped to ``Q x Q``) used by
      the BASC-122 fMRI pipeline;
* :func:`metrics_for_signal_lengthQ`
      builds ``W`` from a signal first resampled to length ``Q``, as used by the
      ECG/EMG/respiratory/sleep/MEG pipelines.

Both call the same size-robust metric routine. ``Q`` defaults to the empirical
relation ``Q ~= 2 * T**(1/3)`` (Eq. 1) computed from the signal length.

The per-dataset preprocessing modules in :mod:`smallworld_qtn.preprocessing`
own the dataset-specific cleaning and file handling and then delegate the
representation/metrics step here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import representations as rep
from .network_metrics import compute_metrics_size_robust, METRIC_KEYS

# Defaults matching the size-robust biological pipeline.
DEFAULT_TARGET_DENSITY = 0.10
DEFAULT_N_RANDOMIZATIONS = 20
DEFAULT_REWIRES_PER_EDGE = 5


def _nan_metrics() -> dict:
    return {k: np.nan for k in METRIC_KEYS}


def weighted_matrix_for_method(signal, Q, method: str, flavour: str,
                               k_values=rep.K_VALUES):
    """Return ``(W, use_abs_weights)`` for one signal and representation.

    Parameters
    ----------
    method : {"QTN", "GAF", "MTF"}
    flavour : {"fmri", "lengthQ"}
        ``"fmri"`` uses the full-signal GAF/MTF cropped to ``Q x Q``;
        ``"lengthQ"`` resamples the signal to length ``Q`` first.
    """
    method = method.upper()
    if method in ("QTN", "QG"):
        A = rep.calculate_quantile_graph_varying_k(signal, Q=Q, k_values=k_values)
        return (A + A.T).astype(float), False

    if flavour == "fmri":
        if method == "GAF":
            return rep.calculate_gaf(signal, Q=Q), True
        if method == "MTF":
            return rep.calculate_mtf(signal, Q=Q), False
    elif flavour == "lengthQ":
        sigQ = rep.downsample_to_length(np.asarray(signal, dtype=float), Q)
        if method == "GAF":
            return rep.calculate_gaf_from_lengthQ(sigQ), True
        if method == "MTF":
            return rep.calculate_mtf_from_lengthQ(sigQ, Q=Q), False

    raise ValueError(f"Unknown method/flavour: {method}/{flavour}")


def metrics_for_signal(signal, method: str, flavour: str,
                       Q: int | None = None,
                       target_density: float = DEFAULT_TARGET_DENSITY,
                       n_rand: int = DEFAULT_N_RANDOMIZATIONS,
                       rewires_per_edge: int = DEFAULT_REWIRES_PER_EDGE,
                       seed: int = 0,
                       k_values=rep.K_VALUES) -> dict:
    """Full metric set for one 1-D signal under one representation."""
    signal = np.asarray(signal, dtype=float)
    if Q is None:
        Q = rep.compute_Q_from_T(signal.size)
    W, use_abs = weighted_matrix_for_method(signal, Q, method, flavour, k_values)
    if np.allclose(W, 0):
        return _nan_metrics()
    return compute_metrics_size_robust(
        W=W,
        target_density=target_density,
        n_rand=n_rand,
        rewires_per_edge=rewires_per_edge,
        seed=seed,
        use_abs_weights=use_abs,
    )


def metrics_for_signal_fmri(signal, method, **kwargs) -> dict:
    """Convenience wrapper: full-signal GAF/MTF flavour (fMRI pipeline)."""
    return metrics_for_signal(signal, method, flavour="fmri", **kwargs)


def metrics_for_signal_lengthQ(signal, method, **kwargs) -> dict:
    """Convenience wrapper: length-``Q`` GAF/MTF flavour (ECG/EMG/.../MEG)."""
    return metrics_for_signal(signal, method, flavour="lengthQ", **kwargs)


def metrics_for_signal_matrix(X: np.ndarray, method: str, flavour: str,
                              k_values=rep.K_VALUES,
                              target_density: float = DEFAULT_TARGET_DENSITY,
                              n_rand: int = DEFAULT_N_RANDOMIZATIONS,
                              rewires_per_edge: int = DEFAULT_REWIRES_PER_EDGE,
                              seed_base: int = 0) -> dict:
    """Average the metric set over the columns (regions/channels) of ``X``.

    ``X`` is ``time x channels``. ``Q`` is derived once from the number of time
    points. This reproduces the per-subject aggregation used in the paper
    (one feature vector per recording).
    """
    X = np.asarray(X, dtype=float)
    n_time, n_chan = X.shape
    Q = rep.compute_Q_from_T(n_time)

    rows = []
    for j in range(n_chan):
        rows.append(
            metrics_for_signal(
                X[:, j], method=method, flavour=flavour, Q=Q,
                target_density=target_density, n_rand=n_rand,
                rewires_per_edge=rewires_per_edge, seed=seed_base + j,
                k_values=k_values,
            )
        )
    dfc = pd.DataFrame(rows)
    mean_metrics = dfc.mean(numeric_only=True).to_dict()
    mean_metrics["Q_used"] = int(Q)
    return mean_metrics
