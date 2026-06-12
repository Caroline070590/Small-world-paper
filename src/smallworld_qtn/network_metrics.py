"""Graph thresholding and size-robust small-world / complex-network metrics.

Extracted verbatim from the canonical ``SMALL-world-FINAL`` notebook (the
size-robust pipeline used for the biological datasets). The metric definitions
match the paper (Methods, Section II B-C):

* small-world index ``sigma = (C/Crand) / (L/Lrand)``  (Eq. 8),
* normalised clustering ``gamma`` and path length ``lambda``  (Eq. 9),
* transitivity ``C``  (Eq. 10) -- used here as the clustering term,
* characteristic path length ``L`` on the giant connected component  (Eq. 11),
* global efficiency ``E``  (Eq. 12),
* the alternative small-world variants ``omega`` and ``phi``.

Important: the clustering term used for ``gamma``/``sigma`` is the network
**transitivity** (``nx.transitivity``), and the random null model is built by
degree-preserving double-edge swaps. This mirrors the published analysis; the
synthetic-figure pipeline (:mod:`smallworld_qtn.synthetic`) uses mean clustering
instead, which is why it is kept separate.
"""

from __future__ import annotations

import numpy as np
import networkx as nx


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------
def proportional_binary_from_weights(W: np.ndarray, target_density: float,
                                     use_abs: bool = True) -> np.ndarray:
    """Threshold a weighted matrix to a binary graph at a fixed density.

    Keeps the strongest ``target_density`` fraction of upper-triangular edges
    (by absolute weight when ``use_abs``) and symmetrises the result.
    """
    n = W.shape[0]
    A = (np.abs(W) if use_abs else W).copy().astype(float)
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


# ---------------------------------------------------------------------------
# Path length, efficiency, spectral helpers
# ---------------------------------------------------------------------------
def gcc_char_path_length_binary(B: np.ndarray) -> float:
    """Characteristic path length on the giant connected component (Eq. 11)."""
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


def global_efficiency_binary(B: np.ndarray) -> float:
    """Global efficiency (Eq. 12)."""
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


# ---------------------------------------------------------------------------
# Null models
# ---------------------------------------------------------------------------
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
    """Connected ring lattice with ``n`` nodes and (approximately) ``m`` edges."""
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
    """Mean/std of transitivity, path length and global efficiency over nulls."""
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
            Cr.append(C)
            Lr.append(L)
        Er.append(E)

    def _ms(arr):
        if len(arr) == 0:
            return (np.nan, np.nan)
        return (float(np.mean(arr)), float(np.std(arr) if len(arr) > 1 else 0.0))

    Cmu, Csd = _ms(Cr)
    Lmu, Lsd = _ms(Lr)
    Emu, Esd = _ms(Er)
    return Cmu, Csd, Lmu, Lsd, Emu, Esd


# ---------------------------------------------------------------------------
# Small-world omega / phi
# ---------------------------------------------------------------------------
def small_world_omega_phi(B: np.ndarray, C: float, L: float,
                          Crand: float, Lrand: float):
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


# ---------------------------------------------------------------------------
# Full metric set
# ---------------------------------------------------------------------------
METRIC_KEYS = [
    "n_nodes", "density", "avg_degree", "assortativity", "transitivity",
    "avg_clustering_weighted", "global_efficiency", "char_path_len_gcc",
    "gamma_C_over_Crand", "lambda_L_over_Lrand", "sigma_small_world",
    "zC", "zL", "E_over_Erand", "Hvn_norm", "lambda2_normlap", "omega", "phi",
]


def compute_metrics_size_robust(W: np.ndarray,
                                target_density: float,
                                n_rand: int,
                                rewires_per_edge: int,
                                seed: int,
                                use_abs_weights: bool = True) -> dict:
    """Size-robust complex-network metrics for a single weighted matrix ``W``.

    Returns a dict with the full metric set listed in :data:`METRIC_KEYS`.
    """
    n = W.shape[0]
    Ww = np.abs(W) if use_abs_weights else W.copy()
    np.fill_diagonal(Ww, 0.0)
    B = proportional_binary_from_weights(Ww, target_density, use_abs=use_abs_weights)

    Gw = nx.from_numpy_array(Ww)
    Gw.remove_edges_from(
        [(u, v) for u, v, w in Gw.edges(data=True) if w.get("weight", 0.0) == 0.0]
    )
    degrees = np.array([d for _, d in nx.degree(nx.from_numpy_array(B))], dtype=float)
    avg_deg = float(degrees.mean()) if degrees.size else np.nan
    density = float(np.sum(B) / (n * (n - 1))) if n > 1 else np.nan
    C_obs = nx.transitivity(nx.from_numpy_array(B)) if n > 1 else np.nan
    L_obs = gcc_char_path_length_binary(B)
    E_obs = global_efficiency_binary(B)
    c_w = (float(np.mean(list(nx.clustering(Gw, weight="weight").values())))
           if Gw.number_of_edges() > 0 else np.nan)
    try:
        assort = nx.degree_assortativity_coefficient(Gw) if Gw.number_of_edges() > 0 else np.nan
    except Exception:
        assort = np.nan

    Cmu, Csd, Lmu, Lsd, Emu, Esd = null_model_stats(B, n_rand, rewires_per_edge, seed)
    gamma = (C_obs / Cmu) if (Cmu and not np.isnan(Cmu) and Cmu > 0) else np.nan
    lambd = (L_obs / Lmu) if (Lmu and not np.isnan(Lmu) and Lmu > 0) else np.nan
    sigma = ((gamma / lambd)
             if (gamma and lambd and not np.isnan(gamma) and not np.isnan(lambd) and lambd != 0)
             else np.nan)
    zC = ((C_obs - Cmu) / Csd) if (Csd and not np.isnan(Csd) and Csd > 0) else np.nan
    zL = ((L_obs - Lmu) / Lsd) if (Lsd and not np.isnan(Lsd) and Lsd > 0) else np.nan
    Enorm = (E_obs / Emu) if (Emu and not np.isnan(Emu) and Emu > 0) else np.nan
    Hvn = von_neumann_entropy_normalized(B)
    eigs = normalized_laplacian_eigs(B)
    lambda2 = float(eigs[1]) if eigs.size >= 2 else np.nan
    omega, phi = small_world_omega_phi(B, C_obs, L_obs, Cmu, Lmu)

    return {
        "n_nodes": n,
        "density": density,
        "avg_degree": avg_deg,
        "assortativity": assort,
        "transitivity": C_obs,
        "avg_clustering_weighted": c_w,
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
