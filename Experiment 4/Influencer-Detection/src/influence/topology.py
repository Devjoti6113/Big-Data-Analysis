"""Structural measures, compiled with numba so they run on the full graph in seconds.

k-core: Batagelj-Zaversnik bucket peeling, O(n + m).
k-truss: triangle support per edge, then peel edges of least support (bucket queue), O(m^1.5).
Forman-Ricci curvature: plain F = 4 - deg(u) - deg(v); augmented F# = F + 3 * triangles(u, v).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .graph import Graph


@njit(cache=True)
def _support(indptr, nbr, eu, ev):
    sup = np.zeros(len(eu), np.int64)
    for e in range(len(eu)):
        i, j, ie, je = indptr[eu[e]], indptr[ev[e]], indptr[eu[e] + 1], indptr[ev[e] + 1]
        while i < ie and j < je:                                      # sorted-list intersection
            if nbr[i] == nbr[j]:
                sup[e] += 1; i += 1; j += 1
            elif nbr[i] < nbr[j]:
                i += 1
            else:
                j += 1
    return sup


@njit(cache=True)
def _core(indptr, nbr):
    n = len(indptr) - 1
    d = (indptr[1:] - indptr[:-1]).copy()
    md = d.max()
    bins = np.zeros(md + 2, np.int64)
    for v in range(n):
        bins[d[v]] += 1
    start = 0
    for k in range(md + 1):
        c = bins[k]; bins[k] = start; start += c
    pos = np.zeros(n, np.int64); vert = np.zeros(n, np.int64)
    for v in range(n):
        pos[v] = bins[d[v]]; vert[pos[v]] = v; bins[d[v]] += 1
    for k in range(md, 0, -1):
        bins[k] = bins[k - 1]
    bins[0] = 0
    for i in range(n):
        v = vert[i]
        for p in range(indptr[v], indptr[v + 1]):
            u = nbr[p]
            if d[u] > d[v]:
                du = d[u]; pu = pos[u]; pw = bins[du]; w = vert[pw]
                if u != w:
                    pos[u] = pw; vert[pu] = w; pos[w] = pu; vert[pw] = u
                bins[du] += 1; d[u] -= 1
    return d


@njit(cache=True)
def _truss(indptr, nbr, eid, eu, ev, sup0):
    mE = len(eu); sup = sup0.copy(); alive = np.ones(mE, np.bool_); truss = np.zeros(mE, np.int64)
    maxs = sup.max()
    head = np.full(maxs + 1, -1, np.int64); nxt = np.full(mE, -1, np.int64); prv = np.full(mE, -1, np.int64)
    for e in range(mE):
        s = sup[e]; nxt[e] = head[s]
        if head[s] != -1:
            prv[head[s]] = e
        head[s] = e
    level, k, remaining = 0, 2, mE
    while remaining > 0:
        while level <= maxs and head[level] == -1:
            level += 1
        if level > maxs:
            break
        k = max(k, level + 2)
        e = head[level]; head[level] = nxt[e]
        if nxt[e] != -1:
            prv[nxt[e]] = -1
        alive[e] = False; truss[e] = k; remaining -= 1
        u, v = eu[e], ev[e]
        i, j, ie, je = indptr[u], indptr[v], indptr[u + 1], indptr[v + 1]
        while i < ie and j < je:
            if nbr[i] == nbr[j]:
                e1, e2 = eid[i], eid[j]
                if alive[e1] and alive[e2]:
                    for f in (e1, e2):
                        s = sup[f]
                        if s > level:                                  # move one bucket down, never below the current level
                            if prv[f] != -1:
                                nxt[prv[f]] = nxt[f]
                            else:
                                head[s] = nxt[f]
                            if nxt[f] != -1:
                                prv[nxt[f]] = prv[f]
                            s -= 1; sup[f] = s; prv[f] = -1; nxt[f] = head[s]
                            if head[s] != -1:
                                prv[head[s]] = f
                            head[s] = f
                i += 1; j += 1
            elif nbr[i] < nbr[j]:
                i += 1
            else:
                j += 1
    return truss


def edge_support(g: Graph) -> np.ndarray:
    return _support(g.indptr, g.nbr, g.eu, g.ev)


def core_numbers(g: Graph) -> np.ndarray:
    return _core(g.indptr, g.nbr)


def truss_numbers(g: Graph, sup: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Edge trussness, and node trussness = maximum over incident edges (0 for isolated nodes)."""
    te = _truss(g.indptr, g.nbr, g.eid, g.eu, g.ev, sup)
    tn = np.zeros(g.n, np.int64)
    np.maximum.at(tn, g.eu, te)
    np.maximum.at(tn, g.ev, te)
    return te, tn


def forman_curvature(g: Graph, sup: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    deg = g.deg
    plain = 4 - deg[g.eu] - deg[g.ev]
    return plain, plain + 3 * sup


def node_aggregate(g: Graph, edge_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, minimum and standard deviation of an edge quantity over each node's incident edges."""
    vals = edge_values[g.eid].astype(float)                           # one value per CSR slot
    deg = g.deg
    owner = np.repeat(np.arange(g.n), deg)
    s = np.bincount(owner, vals, g.n); s2 = np.bincount(owner, vals * vals, g.n)
    mean = np.divide(s, deg, out=np.zeros(g.n), where=deg > 0)
    var = np.divide(s2, deg, out=np.zeros(g.n), where=deg > 0) - mean ** 2
    mn = np.full(g.n, np.inf); np.minimum.at(mn, owner, vals); mn[deg == 0] = 0
    return mean, mn, np.sqrt(np.maximum(var, 0))
