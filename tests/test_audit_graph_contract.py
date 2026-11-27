import numpy as np
import pytest

from whydra.algorithms.graph_core import Edge, GeneralGraph, Node
from whydra.algorithms.graph_core import Endpoint as E
from whydra.evaluation import graph_utils as U
from whydra.evaluation.metrics import (
    calculate_metrics,
    dag_adj_to_cpdag_endpoint_matrix,
    is_pag_result,
)


def graph(names, edges):
    g = GeneralGraph([Node(n) for n in names])
    nodes = {n.get_name(): n for n in g.nodes}
    for a, b in edges:
        g.add_directed_edge(nodes[a], nodes[b])
    return g


def test_metrics_align_names_and_reject_disjoint_names():
    edges = [("a", "c"), ("b", "c")]
    truth = graph(["a", "b", "c"], edges)
    estimate = graph(["c", "a", "b"], edges)
    assert calculate_metrics(truth, estimate)["SHD"] == 0
    with pytest.raises(ValueError, match="names"):
        calculate_metrics(truth, graph(["x", "y", "z"], [("x", "z"), ("y", "z")]))


def test_cyclic_dag_truth_rejected():
    with pytest.raises(ValueError, match="cycl"):
        dag_adj_to_cpdag_endpoint_matrix(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))


@pytest.mark.parametrize("a", [E.TAIL, E.ARROW, E.CIRCLE])
@pytest.mark.parametrize("b", [E.TAIL, E.ARROW, E.CIRCLE])
def test_networkx_endpoint_adapter_round_trip(a, b):
    g = GeneralGraph([Node("a"), Node("b")])
    g.add_edge(Edge(*g.nodes, a, b))
    assert g.get_endpoint_at(g.nodes[0], g.nodes[1]) == a
    nx_graph = U.convert_general_graph_to_nx(g)
    assert nx_graph.number_of_edges() > 0
    np.testing.assert_array_equal(U.from_networkx(nx_graph).graph, g.graph)


@pytest.mark.parametrize("a", [E.TAIL, E.ARROW, E.CIRCLE])
@pytest.mark.parametrize("b", [E.TAIL, E.ARROW, E.CIRCLE])
def test_causallearn_endpoint_adapter_round_trip(a, b):
    pytest.importorskip("causallearn")
    g = GeneralGraph([Node("a"), Node("b")])
    g.add_edge(Edge(*g.nodes, a, b))
    external = U.to_causallearn(g)
    np.testing.assert_array_equal(external.graph, g.graph.T)
    np.testing.assert_array_equal(U.from_causallearn(external).graph, g.graph)


def test_declared_pag_without_circles_keeps_type():
    g = graph(["a", "b"], [("a", "b")])
    g.graph_kind = "pag"
    assert is_pag_result(g)


def test_invalid_partial_endpoint_matrix_rejected():
    with pytest.raises(ValueError, match="endpoint"):
        U.adj_matrix_to_graph(np.array([[0, 2], [0, 0]]), ["a", "b"])
