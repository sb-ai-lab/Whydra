"""Array convert to causallearn matches the old per-edge ``add_edge`` path."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("causallearn")

from causallearn.graph.Edge import Edge as CLEdge
from causallearn.graph.Endpoint import Endpoint as CLEndpoint
from causallearn.graph.GeneralGraph import GeneralGraph as CLGeneralGraph
from causallearn.graph.GraphNode import GraphNode as CLGraphNode

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whydra.algorithms.graph_core.cl_convert import to_causallearn_graph
from whydra.algorithms.graph_core.endpoints import Endpoint


def _old_convert(G: np.ndarray, names):
    cl_nodes = [CLGraphNode(str(n)) for n in names]
    cl_graph = CLGeneralGraph(cl_nodes)

    def map_end(val):
        if val == 1:
            return CLEndpoint.ARROW
        if val == -1:
            return CLEndpoint.TAIL
        if val == 2:
            return CLEndpoint.CIRCLE
        return CLEndpoint.NULL

    p = G.shape[0]
    for i in range(p):
        for j in range(i + 1, p):
            end_j_val = G[i, j]
            end_i_val = G[j, i]
            if end_j_val != 0 or end_i_val != 0:
                cl_edge = CLEdge(
                    cl_nodes[i], cl_nodes[j], map_end(end_i_val), map_end(end_j_val)
                )
                cl_graph.add_edge(cl_edge)
    return cl_graph


def _random_cpdag(p: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    G = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(i + 1, p):
            kind = rng.integers(0, 5)
            if kind == 0:
                continue
            if kind == 1:
                G[i, j] = G[j, i] = Endpoint.TAIL.value
            elif kind == 2:
                G[i, j] = Endpoint.ARROW.value
                G[j, i] = Endpoint.TAIL.value
            elif kind == 3:
                G[i, j] = Endpoint.TAIL.value
                G[j, i] = Endpoint.ARROW.value
            else:
                G[i, j] = G[j, i] = Endpoint.ARROW.value
    return G


def test_matches_add_edge():
    names = [f"X{i}" for i in range(12)]
    G = _random_cpdag(12, seed=0)
    old = _old_convert(G, names)
    new = to_causallearn_graph(G, names)
    assert np.array_equal(old.graph, new.graph)


def test_dpath_chain():
    """Incremental add_edge dpath matches full closure on a path 0→1→…→n-1."""
    p = 8
    G = np.zeros((p, p), dtype=int)
    for i in range(p - 1):
        G[i, i + 1] = Endpoint.ARROW.value
        G[i + 1, i] = Endpoint.TAIL.value
    names = [f"X{i}" for i in range(p)]
    old = _old_convert(G, names)
    new = to_causallearn_graph(G, names)
    assert np.array_equal(old.graph, new.graph)
    assert np.array_equal(old.dpath, new.dpath)
    for i in range(p):
        for j in range(i, p):
            assert new.dpath[j, i] == 1


def test_directed_is_transpose():
    G = np.zeros((3, 3), dtype=int)
    # 0 → 1
    G[0, 1] = Endpoint.ARROW.value
    G[1, 0] = Endpoint.TAIL.value
    cl = to_causallearn_graph(G, ["A", "B", "C"])
    assert cl.graph[1, 0] == 1
    assert cl.graph[0, 1] == -1
    assert cl.is_parent_of(cl.nodes[0], cl.nodes[1])
    assert cl.is_ancestor_of(cl.nodes[0], cl.nodes[1])


if __name__ == "__main__":
    test_matches_add_edge()
    test_dpath_chain()
    test_directed_is_transpose()
    print("ok")
