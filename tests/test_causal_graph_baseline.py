"""Sequential ``CausalGraph`` setup must match the legacy Edge loop."""

import sys
from pathlib import Path

import numpy as np

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whydra.algorithms.graph_core import CausalGraph, Endpoint
from whydra.algorithms.graph_core.edges import Edge
from whydra.algorithms.graph_core.general_graph import GeneralGraph
from whydra.algorithms.graph_core.nodes import Node


def _edge_loop_graph(p: int):
    names = [f"X{n}" for n in range(p)]
    nodes = [Node(name) for name in names]
    graph = GeneralGraph(nodes)
    for i in range(p):
        for j in range(i + 1, p):
            graph.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))
    return names, graph


def test_sequential_baseline_is_edge_loop():
    p = 25
    names, old = _edge_loop_graph(p)
    cg = CausalGraph(p, names)
    assert cg._G_cuda is None
    assert np.array_equal(cg.G.graph, old.graph)
    assert np.all(np.diag(cg.G.graph) == 0)
    assert np.all(cg.G.graph[np.triu_indices(p, 1)] == Endpoint.TAIL.value)
