"""Checks against NetworkX on small random graphs, and exact answers on hand-made ones."""
import gzip, os, sys, tempfile
import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from influence import features, graph, sir, topology  # noqa: E402


def from_edges(edges, directed_extra=()):
    path = os.path.join(tempfile.mkdtemp(), "g.txt.gz")
    with gzip.open(path, "wt") as f:
        f.write("# test\n")
        for u, v in list(edges) + list(directed_extra):
            f.write(f"{u}\t{v}\n")
    return graph.load_snap(path)[0]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_core_and_truss_match_networkx(seed):
    G = nx.gnm_random_graph(120, 600, seed=seed)
    g = from_edges(G.edges())
    core = topology.core_numbers(g)
    nxcore = nx.core_number(G)
    assert all(core[v] == nxcore[v] for v in G.nodes if G.degree(v) > 0)
    sup = topology.edge_support(g)
    te, _ = topology.truss_numbers(g, sup)
    for k in range(3, 8):                                   # edge in the k-truss  <=>  trussness >= k
        H = nx.k_truss(G, k)
        mine = {(int(u), int(v)) for u, v, t in zip(g.eu, g.ev, te) if t >= k}
        theirs = {(min(u, v), max(u, v)) for u, v in H.edges()}
        assert mine == theirs, k


def test_triangle_support_and_forman_on_a_hand_graph():
    # a triangle 0-1-2 with a pendant 2-3:   deg = 2, 2, 3, 1
    g = from_edges([(0, 1), (1, 2), (0, 2), (2, 3)])
    sup = topology.edge_support(g)
    plain, aug = topology.forman_curvature(g, sup)
    expect = {(0, 1): (0, 3), (0, 2): (-1, 2), (1, 2): (-1, 2), (2, 3): (0, 0)}
    for u, v, p, a in zip(g.eu, g.ev, plain, aug):
        assert (p, a) == expect[(u, v)]


def test_sir_limits():
    G = nx.disjoint_union(nx.path_graph(5), nx.path_graph(3))
    g = from_edges(G.edges())
    assert (sir.outbreak_sizes(g, np.array([0, 6]), 1.0, 5) == np.array([[5] * 5, [3] * 5])).all()   # beta = 1: whole component
    assert (sir.outbreak_sizes(g, np.array([0, 6]), 0.0, 5) == 1).all()                               # beta = 0: only the seed


def test_sir_matches_percolation_expectation():
    # star with 50 leaves, seeded at the centre: expected outbreak = 1 + 50 * beta
    g = from_edges([(0, i) for i in range(1, 51)])
    m = sir.outbreak_sizes(g, np.array([0]), 0.3, 20000, seed=3).mean()
    assert abs(m - 16.0) < 0.15


def test_features_shape_and_sanity():
    G = nx.gnm_random_graph(200, 900, seed=5)
    g = from_edges(G.edges(), directed_extra=[(v, u) for u, v in list(G.edges())[:100]])
    X, _ = features.build(g)
    assert X.shape == (g.n, 20) and np.isfinite(X).all()
    i = features.FEATURES.index
    assert np.allclose(X[:, i("pagerank")].sum(), 1.0)
    cc = nx.clustering(G)
    assert np.allclose([X[v, i("clustering")] for v in G.nodes], [cc[v] for v in G.nodes])
    two = [len(nx.single_source_shortest_path_length(G, v, cutoff=2)) - 1 for v in G.nodes]
    assert (X[list(G.nodes), i("two_hop_reach")] == two).all()
