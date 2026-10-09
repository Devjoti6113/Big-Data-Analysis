"""Loading the network into compressed sparse row (CSR) form.

The raw SNAP file is a directed edge list ("u trusts v"). Structure (cores, trusses, curvature,
spreading) is computed on the undirected simple graph; in- and out-degree are kept as features.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass

import numpy as np


@dataclass
class Graph:
    n: int                 # nodes (ids 0..n-1; ids that never appear have degree 0)
    indptr: np.ndarray     # CSR row pointers, length n+1
    nbr: np.ndarray        # neighbours, sorted within each row
    eu: np.ndarray         # undirected edges, u < v
    ev: np.ndarray
    eid: np.ndarray        # for every CSR slot, the id of its undirected edge
    in_deg: np.ndarray     # from the directed data
    out_deg: np.ndarray
    reciprocal: np.ndarray  # number of neighbours with edges in both directions

    @property
    def deg(self) -> np.ndarray:
        return np.diff(self.indptr)

    @property
    def m(self) -> int:
        return len(self.eu)


def load_snap(path: str) -> tuple[Graph, dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        raw = np.loadtxt(f, dtype=np.int64, comments="#")
    stats = {"directed_edges_in_file": int(len(raw))}
    raw = raw[raw[:, 0] != raw[:, 1]]                              # self-loops
    stats["self_loops_removed"] = stats["directed_edges_in_file"] - len(raw)
    raw = np.unique(raw, axis=0)                                    # duplicate directed edges
    n = int(raw.max()) + 1
    in_deg = np.bincount(raw[:, 1], minlength=n)
    out_deg = np.bincount(raw[:, 0], minlength=n)
    lo, hi = raw.min(1), raw.max(1)
    pair_keys, counts = np.unique(lo * n + hi, return_counts=True)  # count 2 = the pair trusts each other
    und = np.stack([pair_keys // n, pair_keys % n], 1)
    reciprocal = np.zeros(n, np.int64)
    both = und[counts == 2]
    np.add.at(reciprocal, both[:, 0], 1)
    np.add.at(reciprocal, both[:, 1], 1)
    sym = np.vstack([und, und[:, ::-1]])
    sym = sym[np.lexsort((sym[:, 1], sym[:, 0]))]
    indptr = np.zeros(n + 1, np.int64)
    np.add.at(indptr, sym[:, 0] + 1, 1)
    indptr = np.cumsum(indptr)
    nbr = sym[:, 1].copy()
    deg = np.diff(indptr)
    slot_u = np.repeat(np.arange(n), deg)
    a, b = np.minimum(slot_u, nbr), np.maximum(slot_u, nbr)
    eid = np.searchsorted(pair_keys, a * n + b)
    g = Graph(n, indptr, nbr, und[:, 0].copy(), und[:, 1].copy(), eid, in_deg, out_deg, reciprocal)
    stats.update(nodes_with_edges=int((deg > 0).sum()), undirected_edges=int(g.m), reciprocal_pairs=int((counts == 2).sum()))
    return g, stats


def giant_component(g: Graph) -> np.ndarray:
    """Boolean mask of the largest connected component (iterative BFS, no recursion)."""
    label = np.full(g.n, -1, np.int64)
    best, best_size, comp = -1, 0, 0
    for s in np.where(g.deg > 0)[0]:
        if label[s] >= 0:
            continue
        label[s] = comp
        stack, size = [s], 0
        while stack:
            v = stack.pop()
            size += 1
            for w in g.nbr[g.indptr[v]:g.indptr[v + 1]]:
                if label[w] < 0:
                    label[w] = comp
                    stack.append(w)
        if size > best_size:
            best, best_size = comp, size
        comp += 1
    return label == best
