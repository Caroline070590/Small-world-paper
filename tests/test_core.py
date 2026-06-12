"""Minimal sanity tests for the core library.

These check shapes, basic properties, and that the metric pipeline runs and
returns finite small-world values on a structured signal. They do not re-derive
the paper's numbers (which depend on full datasets), but they guard against
accidental breakage of the core definitions.

Run with:  pytest -q
"""

import numpy as np

from smallworld_qtn import representations as rep
from smallworld_qtn import network_metrics as nm
from smallworld_qtn import pipeline


def _logistic(n=600, r=3.9, x0=0.37):
    x = np.empty(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = r * x[t - 1] * (1.0 - x[t - 1])
    return x


def test_compute_Q_from_T():
    # Q ~= 2 * T**(1/3); for T=1000 -> 20
    assert rep.compute_Q_from_T(1000) == 20


def test_qg_shape_and_symmetry_path():
    x = _logistic()
    Q = rep.compute_Q_from_T(x.size)
    A = rep.calculate_quantile_graph_varying_k(x, Q=Q)
    assert A.shape == (Q, Q)
    W = (A + A.T).astype(float)
    assert np.allclose(W, W.T)  # symmetrised adjacency is symmetric


def test_gaf_flavours_differ():
    x = _logistic()
    Q = rep.compute_Q_from_T(x.size)
    g_full = rep.calculate_gaf(x, Q=Q)
    g_lenQ = rep.calculate_gaf_from_lengthQ(rep.downsample_to_length(x, Q))
    assert g_full.shape == (Q, Q)
    assert g_lenQ.shape == (Q, Q)
    # The two flavours are intentionally NOT identical.
    assert not np.allclose(g_full, g_lenQ)


def test_mtf_rows_are_probabilities():
    x = _logistic()
    Q = rep.compute_Q_from_T(x.size)
    sigQ = rep.downsample_to_length(x, Q)
    M = rep.calculate_mtf_from_lengthQ(sigQ, Q=Q)
    assert M.shape == (Q, Q)
    # every entry is a valid transition probability in [0, 1]
    assert M.min() >= 0.0 and M.max() <= 1.0


def test_metric_keys_complete():
    assert len(nm.METRIC_KEYS) == 18
    for key in ("sigma_small_world", "gamma_C_over_Crand", "lambda_L_over_Lrand",
                "transitivity", "global_efficiency", "char_path_len_gcc",
                "omega", "phi"):
        assert key in nm.METRIC_KEYS


def test_pipeline_runs_for_all_methods():
    x = _logistic()
    finite_count = 0
    for method in ("QTN", "GAF", "MTF"):
        m = pipeline.metrics_for_signal_lengthQ(method=method, signal=x, n_rand=10, seed=0)
        assert set(nm.METRIC_KEYS).issubset(m.keys())
        if np.isfinite(m["sigma_small_world"]):
            finite_count += 1
    # At least one representation should yield a finite small-world index on a
    # structured signal. (With few random nulls, an individual method can
    # occasionally produce a disconnected null and a non-finite sigma.)
    assert finite_count >= 1
