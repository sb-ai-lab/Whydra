"""Test endpoint encodings, node names, and metric edge cases in graph evaluation."""

import numpy as np
import pytest

from whydra.background_knowledge import MatrixEncoding
from whydra.evaluation.graph_utils import (
    adj_matrix_to_graph,
    convert_general_graph_to_nx,
    default_node_names,
    endpoint_lookup,
    get_adj_matrix,
    pag_matrix_to_general_graph,
    result_to_nx,
)
from whydra.evaluation.metrics import (
    calculate_metrics,
    compute_metrics_from_dag_and_result,
    compute_metrics_from_endpoint_matrices,
    convert_pag_to_cpdag_proxy,
    dag_adj_to_cpdag_endpoint_matrix,
    dag_adj_to_general_graph,
    f1_score,
    is_pag_result,
    result_to_endpoint_matrix,
)

NULL, TAIL, ARROW, CIRCLE = 0, -1, 1, 2
NAMES2 = ["A", "B"]


# ============================================================
# Node names: one convention throughout the package
# ============================================================


def test_default_node_names_is_one_based():
    assert default_node_names(3) == ["X1", "X2", "X3"]
    assert default_node_names(0) == []


def test_all_generators_agree_on_node_names():
    """The .npy reference used X1, X2, ... while DAG helpers used X0, X1, ...; comparisons missed matches."""
    A = np.array([[0, 1], [0, 0]])
    from_dag = [n.get_name() for n in dag_adj_to_general_graph(A).get_nodes()]
    assert from_dag == default_node_names(2)


# ============================================================
# Explicit encodings and their validation
# ============================================================


def test_endpoint_lookup_covers_both_encodings():
    std = endpoint_lookup(MatrixEncoding.STANDARD)
    pca = endpoint_lookup(MatrixEncoding.PCALG)
    assert std[-1].name == "TAIL"
    assert std[1].name == "ARROW"
    assert std[2].name == "CIRCLE"
    assert pca[3].name == "TAIL"
    assert pca[2].name == "ARROW"
    assert pca[1].name == "CIRCLE"


@pytest.mark.parametrize(
    ("encoding", "tail", "arrow"),
    [(MatrixEncoding.STANDARD, TAIL, ARROW), (MatrixEncoding.PCALG, 3, 2)],
)
def test_pag_matrix_to_general_graph_reads_the_named_encoding(encoding, tail, arrow):
    m = np.zeros((2, 2), dtype=int)
    m[1, 0], m[0, 1] = tail, arrow  # A -> B
    g = pag_matrix_to_general_graph(m, NAMES2, encoding)
    assert [str(e) for e in g.get_graph_edges()] == ["A --> B"]


def test_wrong_encoding_raises_instead_of_corrupting():
    """Reject a matrix in the wrong encoding before it silently loses tail marks."""
    std = np.array([[0, ARROW], [TAIL, 0]])
    with pytest.raises(ValueError, match="not valid for encoding"):
        pag_matrix_to_general_graph(std, NAMES2, MatrixEncoding.PCALG)


def test_result_to_endpoint_matrix_validates_codes():
    std = np.array([[0, ARROW], [TAIL, 0]])
    assert result_to_endpoint_matrix(std, NAMES2, MatrixEncoding.STANDARD).tolist() == [
        [0, ARROW],
        [TAIL, 0],
    ]
    with pytest.raises(ValueError, match="not valid for encoding"):
        result_to_endpoint_matrix(std, NAMES2, MatrixEncoding.PCALG)


def test_result_to_endpoint_matrix_accepts_graph_objects():
    g = dag_adj_to_general_graph(np.array([[0, 1], [0, 0]]))
    m = result_to_endpoint_matrix(g, default_node_names(2))
    assert m[1, 0] == TAIL
    assert m[0, 1] == ARROW


def test_result_to_endpoint_matrix_rejects_unknown_types():
    with pytest.raises(TypeError):
        result_to_endpoint_matrix("not a graph", NAMES2)


def test_is_pag_result():
    assert is_pag_result(np.zeros((2, 2)))
    assert not is_pag_result(np.zeros(2))
    assert not is_pag_result(dag_adj_to_general_graph(np.zeros((2, 2), dtype=int)))


# ============================================================
# Converting between matrices and graphs
# ============================================================


def test_get_adj_matrix_roundtrip():
    m = np.zeros((3, 3), dtype=int)
    m[1, 0], m[0, 1] = TAIL, ARROW  # X1 -> X2
    m[2, 1], m[1, 2] = CIRCLE, CIRCLE  # X2 o-o X3
    g = adj_matrix_to_graph(m, default_node_names(3))
    assert np.array_equal(get_adj_matrix(g), m)


def test_adj_matrix_to_graph_rounds_floats():
    """Averaging during bagging can produce a floating-point matrix."""
    m = np.array([[0.0, 0.9], [-1.1, 0.0]])
    g = adj_matrix_to_graph(m, NAMES2)
    assert [str(e) for e in g.get_graph_edges()] == ["A --> B"]


def test_convert_general_graph_to_nx_covers_every_edge_type():
    cases = {
        "directed": ((TAIL, ARROW), 1),
        "bidirected": ((ARROW, ARROW), 2),
        "undirected": ((TAIL, TAIL), 2),
        "circle_arrow": ((CIRCLE, ARROW), 1),
        "circle_circle": ((CIRCLE, CIRCLE), 2),
    }
    for label, ((e1, e2), expected_edges) in cases.items():
        m = np.zeros((2, 2), dtype=int)
        m[1, 0], m[0, 1] = e1, e2
        nx_g = convert_general_graph_to_nx(adj_matrix_to_graph(m, NAMES2))
        assert nx_g.number_of_edges() == expected_edges, f"edge type {label} was lost"


def test_result_to_nx_accepts_both_matrix_and_graph():
    m = np.zeros((2, 2), dtype=int)
    m[1, 0], m[0, 1] = TAIL, ARROW  # A -> B in the library's encoding.
    assert result_to_nx(m, NAMES2).number_of_edges() == 1
    # Callers can specify the encoding of a matrix from an external source.
    pcalg = np.zeros((2, 2), dtype=int)
    pcalg[1, 0], pcalg[0, 1] = 3, 2
    assert result_to_nx(pcalg, NAMES2, MatrixEncoding.PCALG).number_of_edges() == 1
    g = dag_adj_to_general_graph(np.array([[0, 1], [0, 0]]))
    assert result_to_nx(g, default_node_names(2)).number_of_edges() == 1


# ============================================================
# Reference graph: DAG to CPDAG
# ============================================================


def test_dag_to_cpdag_leaves_reversible_edge_undirected():
    """A single edge is reversible, so it is undirected in the CPDAG."""
    m = dag_adj_to_cpdag_endpoint_matrix(np.array([[0, 1], [0, 0]]))
    assert m[0, 1] == TAIL
    assert m[1, 0] == TAIL


def test_dag_to_cpdag_keeps_collider_oriented():
    """The unshielded collider A -> C <- B is identifiable and stays directed."""
    A = np.zeros((3, 3), dtype=int)
    A[0, 2] = A[1, 2] = 1
    m = dag_adj_to_cpdag_endpoint_matrix(A)
    assert m[0, 2] == ARROW
    assert m[2, 0] == TAIL
    assert m[1, 2] == ARROW
    assert m[2, 1] == TAIL


def test_pag_to_cpdag_proxy_turns_circles_into_tails():
    m = np.array([[0, CIRCLE], [CIRCLE, 0]])
    assert np.array_equal(
        convert_pag_to_cpdag_proxy(m), np.array([[0, TAIL], [TAIL, 0]])
    )


def test_pag_to_cpdag_proxy_does_not_mutate_the_input():
    m = np.array([[0, CIRCLE], [CIRCLE, 0]])
    before = m.copy()
    convert_pag_to_cpdag_proxy(m)
    assert np.array_equal(m, before)


# ============================================================
# Metrics
# ============================================================


def test_f1_score():
    assert f1_score(1.0, 1.0) == 1.0
    assert f1_score(0.0, 0.0) == 0.0
    assert f1_score(0.5, 0.5) == pytest.approx(0.5)


def test_identical_graphs_score_perfectly():
    m = np.zeros((3, 3), dtype=int)
    m[1, 0], m[0, 1] = TAIL, ARROW
    r = compute_metrics_from_endpoint_matrices(m, m.copy())
    assert r["SHD"] == 0
    assert r["adj_f1"] == 1.0
    assert r["arrow_f1"] == 1.0


def test_shd_counts_each_pair_once():
    a = np.zeros((3, 3), dtype=int)
    b = np.zeros((3, 3), dtype=int)
    b[1, 0], b[0, 1] = TAIL, ARROW  # One extra edge.
    assert compute_metrics_from_endpoint_matrices(a, b)["SHD"] == 1


def test_reversed_edge_costs_one_shd_but_keeps_adjacency():
    a = np.zeros((2, 2), dtype=int)
    a[1, 0], a[0, 1] = TAIL, ARROW
    b = np.zeros((2, 2), dtype=int)
    b[0, 1], b[1, 0] = TAIL, ARROW
    r = compute_metrics_from_endpoint_matrices(a, b)
    assert r["SHD"] == 1
    assert r["adj_f1"] == 1.0
    assert r["arrow_f1"] == 0.0


def test_missing_and_false_edge_shares_are_zero_on_a_perfect_match():
    """Two empty graphs match perfectly; the error rate is zero."""
    z = np.zeros((3, 3), dtype=int)
    r = compute_metrics_from_endpoint_matrices(z, z.copy())
    assert r["missing_edges_share"] == 0.0
    assert r["false_edges_share"] == 0.0


def test_metrics_report_edge_count_and_max_degree():
    m = np.zeros((3, 3), dtype=int)
    m[1, 0], m[0, 1] = TAIL, ARROW
    m[2, 0], m[0, 2] = TAIL, ARROW
    r = compute_metrics_from_endpoint_matrices(m, m.copy())
    assert r["n_edges"] == 2
    assert r["max_degree"] == 2


def test_compute_metrics_from_dag_and_result_handles_pag_flag():
    A = np.zeros((3, 3), dtype=int)
    A[0, 2] = A[1, 2] = 1
    est = dag_adj_to_cpdag_endpoint_matrix(A)
    assert compute_metrics_from_dag_and_result(A, est, is_pag=False)["SHD"] == 0
    # The same graph uses circles in place of tails. The proxy must remove the circles.
    pag = est.copy()
    pag[pag == TAIL] = CIRCLE
    assert compute_metrics_from_dag_and_result(A, pag, is_pag=True)["SHD"] == 0


def test_calculate_metrics_on_general_graphs():
    A = np.zeros((3, 3), dtype=int)
    A[0, 2] = A[1, 2] = 1
    true_g = dag_adj_to_general_graph(A)
    r = calculate_metrics(true_g, true_g)
    assert r["SHD"] == 0
    assert r["adj_f1"] == 1.0


def test_calculate_metrics_chain_is_not_a_collider():
    """Keep a three-node chain free of colliders in the ground-truth CPDAG.

    get_adj_matrix() encodes a tail as -1 and an arrowhead as 1.
    dag_adj_to_cpdag_endpoint_matrix expects binary adjacency. Counting both
    marks as edges can turn X1->X2->X3 into a spurious X1->X2<-X3.
    A fixture with a real collider would hide that mistake.
    """
    A = np.zeros((3, 3), dtype=int)
    A[0, 1] = A[1, 2] = 1
    true_g = dag_adj_to_general_graph(A)

    true_dag_adj = (get_adj_matrix(true_g) == ARROW).astype(int)
    cpdag = dag_adj_to_cpdag_endpoint_matrix(true_dag_adj)
    assert cpdag[0, 1] == TAIL
    assert cpdag[1, 0] == TAIL
    assert cpdag[1, 2] == TAIL
    assert cpdag[2, 1] == TAIL

    # The chain's Markov equivalence class has two undirected edges.
    # This estimate must score as a perfect match. A spurious collider
    # in the ground truth would penalize it for two missing arrowheads.
    est_g = adj_matrix_to_graph(cpdag, default_node_names(3))
    r = calculate_metrics(true_g, est_g)
    assert r["SHD"] == 0
    assert r["adj_f1"] == 1.0
