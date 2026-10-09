"""Monte Carlo SIR on the network.

Discrete time. Each infected node gets one chance to infect each susceptible neighbour, with
probability beta, and then recovers (gamma = 1). This is the independent cascade model; the outbreak
from a seed equals the size of its cluster after keeping each edge independently with probability beta.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .graph import Graph


def epidemic_threshold(g: Graph) -> float:
    """beta_c = <k> / (<k^2> - <k>) for an uncorrelated network (heterogeneous mean-field)."""
    d = g.deg[g.deg > 0].astype(float)
    return d.mean() / ((d ** 2).mean() - d.mean())


@njit(cache=True)
def _outbreaks(indptr, nbr, seeds, beta, runs, seed):
    np.random.seed(seed)
    n = len(indptr) - 1
    sizes = np.zeros((len(seeds), runs), np.int32)
    stamp = np.zeros(n, np.int64); tick = 0
    frontier = np.empty(n, np.int64); nxt = np.empty(n, np.int64)
    for si in range(len(seeds)):
        for r in range(runs):
            tick += 1
            frontier[0] = seeds[si]; stamp[seeds[si]] = tick; nf = 1; size = 1
            while nf > 0:
                nn = 0
                for q in range(nf):
                    v = frontier[q]
                    for p in range(indptr[v], indptr[v + 1]):
                        u = nbr[p]
                        if stamp[u] != tick and np.random.random() < beta:
                            stamp[u] = tick; nxt[nn] = u; nn += 1
                size += nn; nf = nn
                for q in range(nn):
                    frontier[q] = nxt[q]
            sizes[si, r] = size
    return sizes


def outbreak_sizes(g: Graph, seeds: np.ndarray, beta: float, runs: int, seed: int = 0) -> np.ndarray:
    """Final outbreak size of every run: array (len(seeds), runs)."""
    return _outbreaks(g.indptr, g.nbr, np.asarray(seeds, np.int64), float(beta), int(runs), int(seed))
