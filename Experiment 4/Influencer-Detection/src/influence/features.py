"""The 20 node features. Every one is local or near-local, so the table for 75,879 nodes builds in seconds."""
from __future__ import annotations

import numpy as np
from numba import njit

from .graph import Graph
from .topology import core_numbers, edge_support, forman_curvature, node_aggregate, truss_numbers

FEATURES = [
    "degree", "in_degree", "out_degree", "reciprocity", "coreness", "trussness", "triangles", "clustering",
    "avg_nbr_degree", "max_nbr_degree", "sum_nbr_coreness", "nbr_degree_hindex", "two_hop_reach",
    "collective_influence_2", "pagerank", "eigenvector", "forman_mean", "forman_min", "forman_std", "frac_negative_forman",
]


@njit(cache=True)
def _two_hop(indptr, nbr, deg):
    """Distinct nodes within distance 2, and CI_2 = (k_i - 1) * sum over nodes at distance exactly 2 of (k_j - 1)."""
    n = len(indptr) - 1
    stamp = np.full(n, -1, np.int64); dist = np.zeros(n, np.int64)
    reach = np.zeros(n, np.int64); ci = np.zeros(n, np.float64)
    for s in range(n):
        if deg[s] == 0:
            continue
        stamp[s] = s; dist[s] = 0
        for p in range(indptr[s], indptr[s + 1]):
            stamp[nbr[p]] = s; dist[nbr[p]] = 1
        cnt = deg[s]; acc = 0.0
        for p in range(indptr[s], indptr[s + 1]):
            u = nbr[p]
            for q in range(indptr[u], indptr[u + 1]):
                w = nbr[q]
                if stamp[w] != s:
                    stamp[w] = s; dist[w] = 2; cnt += 1; acc += deg[w] - 1
        reach[s] = cnt
        ci[s] = (deg[s] - 1) * acc
    return reach, ci


@njit(cache=True)
def _hindex(indptr, nbr, deg):
    n = len(indptr) - 1
    h = np.zeros(n, np.int64)
    for s in range(n):
        d = np.sort(deg[nbr[indptr[s]:indptr[s + 1]]])[::-1]
        k = 0
        while k < len(d) and d[k] >= k + 1:
            k += 1
        h[s] = k
    return h


def pagerank(g: Graph, alpha: float = 0.85, iters: int = 100, tol: float = 1e-12) -> np.ndarray:
    deg = g.deg.astype(float)
    owner = np.repeat(np.arange(g.n), g.deg)
    x = np.full(g.n, 1.0 / g.n)
    for _ in range(iters):
        share = np.divide(x, deg, out=np.zeros(g.n), where=deg > 0)
        new = np.bincount(g.nbr, share[owner], g.n) * alpha
        new += (1 - new.sum()) / g.n                                  # teleport + dangling mass, keeps the sum at 1
        if np.abs(new - x).sum() < tol:
            return new
        x = new
    return x


def eigenvector(g: Graph, iters: int = 200) -> np.ndarray:
    owner = np.repeat(np.arange(g.n), g.deg)
    x = np.ones(g.n) / np.sqrt(g.n)
    for _ in range(iters):
        new = np.bincount(g.nbr, x[owner], g.n) + x                    # (A + I) x : the shift avoids oscillation on bipartite parts
        new /= np.linalg.norm(new)
        if np.abs(new - x).max() < 1e-12:
            break
        x = new
    return x


def build(g: Graph) -> tuple[np.ndarray, dict]:
    deg = g.deg
    sup = edge_support(g)
    core = core_numbers(g)
    truss_e, truss = truss_numbers(g, sup)
    plain, aug = forman_curvature(g, sup)
    f_mean, f_min, f_std = node_aggregate(g, aug)
    neg_frac, _, _ = node_aggregate(g, (aug < 0).astype(float))
    tri = np.zeros(g.n, np.int64); np.add.at(tri, g.eu, sup); np.add.at(tri, g.ev, sup); tri //= 2
    owner = np.repeat(np.arange(g.n), deg)
    nbr_deg = deg[g.nbr]
    avg_nd = np.divide(np.bincount(owner, nbr_deg, g.n), deg, out=np.zeros(g.n), where=deg > 0)
    max_nd = np.zeros(g.n); np.maximum.at(max_nd, owner, nbr_deg)
    sum_nc = np.bincount(owner, core[g.nbr], g.n)
    reach, ci = _two_hop(g.indptr, g.nbr, deg)
    pairs = deg * (deg - 1) / 2
    X = np.column_stack([
        deg, g.in_deg, g.out_deg, np.divide(g.reciprocal, deg, out=np.zeros(g.n), where=deg > 0), core, truss, tri,
        np.divide(tri, pairs, out=np.zeros(g.n), where=pairs > 0), avg_nd, max_nd, sum_nc, _hindex(g.indptr, g.nbr, deg),
        reach, ci, pagerank(g), eigenvector(g), f_mean, f_min, f_std, neg_frac,
    ]).astype(float)
    extra = {"support": sup, "truss_edge": truss_e, "forman_plain": plain, "forman_aug": aug, "core": core, "truss": truss}
    return X, extra
