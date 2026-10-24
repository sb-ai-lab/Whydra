"""Тесты слоя оценки: конвертации графов и метрики.

Здесь сосредоточено то, что в аудите оказалось источником молчаливо неверных
результатов: перепутанная кодировка меток, два соглашения об именах узлов и
краевые случаи метрик. Все проверки формулируются от наблюдаемого поведения,
а не от реализации.
"""

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
#  Имена узлов: одно соглашение на весь пакет
# ============================================================

def test_default_node_names_is_one_based():
    assert default_node_names(3) == ["X1", "X2", "X3"]
    assert default_node_names(0) == []


def test_all_generators_agree_on_node_names():
    """Раньше .npy-эталон давал X1.., а DAG-хелперы X0.. — сопоставление промахивалось."""
    A = np.array([[0, 1], [0, 0]])
    from_dag = [n.get_name() for n in dag_adj_to_general_graph(A).get_nodes()]
    assert from_dag == default_node_names(2)


# ============================================================
#  Кодировки: явные, с проверкой
# ============================================================

def test_endpoint_lookup_covers_both_encodings():
    std = endpoint_lookup(MatrixEncoding.STANDARD)
    pca = endpoint_lookup(MatrixEncoding.PCALG)
    assert std[-1].name == "TAIL" and std[1].name == "ARROW" and std[2].name == "CIRCLE"
    assert pca[3].name == "TAIL" and pca[2].name == "ARROW" and pca[1].name == "CIRCLE"


@pytest.mark.parametrize(
    "encoding,tail,arrow",
    [(MatrixEncoding.STANDARD, TAIL, ARROW), (MatrixEncoding.PCALG, 3, 2)],
)
def test_pag_matrix_to_general_graph_reads_the_named_encoding(encoding, tail, arrow):
    m = np.zeros((2, 2), dtype=int)
    m[1, 0], m[0, 1] = tail, arrow          # A -> B
    g = pag_matrix_to_general_graph(m, NAMES2, encoding)
    assert [str(e) for e in g.get_graph_edges()] == ["A --> B"]


def test_wrong_encoding_raises_instead_of_corrupting():
    """Матрица в другой кодировке обязана падать, а не молча терять хвосты."""
    std = np.array([[0, ARROW], [TAIL, 0]])
    with pytest.raises(ValueError, match="not valid for encoding"):
        pag_matrix_to_general_graph(std, NAMES2, MatrixEncoding.PCALG)


def test_result_to_endpoint_matrix_validates_codes():
    std = np.array([[0, ARROW], [TAIL, 0]])
    assert result_to_endpoint_matrix(std, NAMES2, MatrixEncoding.STANDARD).tolist() == [
        [0, ARROW], [TAIL, 0]
    ]
    with pytest.raises(ValueError, match="not valid for encoding"):
        result_to_endpoint_matrix(std, NAMES2, MatrixEncoding.PCALG)


def test_result_to_endpoint_matrix_accepts_graph_objects():
    g = dag_adj_to_general_graph(np.array([[0, 1], [0, 0]]))
    m = result_to_endpoint_matrix(g, default_node_names(2))
    assert m[1, 0] == TAIL and m[0, 1] == ARROW


def test_result_to_endpoint_matrix_rejects_unknown_types():
    with pytest.raises(TypeError):
        result_to_endpoint_matrix("не граф", NAMES2)


def test_is_pag_result():
    assert is_pag_result(np.zeros((2, 2)))
    assert not is_pag_result(np.zeros(2))
    assert not is_pag_result(dag_adj_to_general_graph(np.zeros((2, 2), dtype=int)))


# ============================================================
#  Матрица <-> граф
# ============================================================

def test_get_adj_matrix_roundtrip():
    m = np.zeros((3, 3), dtype=int)
    m[1, 0], m[0, 1] = TAIL, ARROW          # X1 -> X2
    m[2, 1], m[1, 2] = CIRCLE, CIRCLE       # X2 o-o X3
    g = adj_matrix_to_graph(m, default_node_names(3))
    assert np.array_equal(get_adj_matrix(g), m)


def test_adj_matrix_to_graph_rounds_floats():
    """После усреднения при бэггинге матрица может быть вещественной."""
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
        assert nx_g.number_of_edges() == expected_edges, f"тип ребра {label} потерян"


def test_result_to_nx_accepts_both_matrix_and_graph():
    m = np.zeros((2, 2), dtype=int)
    m[1, 0], m[0, 1] = TAIL, ARROW          # A -> B в кодировке библиотеки
    assert result_to_nx(m, NAMES2).number_of_edges() == 1
    # Матрицу из внешнего источника можно передать с явной кодировкой.
    pcalg = np.zeros((2, 2), dtype=int)
    pcalg[1, 0], pcalg[0, 1] = 3, 2
    assert result_to_nx(pcalg, NAMES2, MatrixEncoding.PCALG).number_of_edges() == 1
    g = dag_adj_to_general_graph(np.array([[0, 1], [0, 0]]))
    assert result_to_nx(g, default_node_names(2)).number_of_edges() == 1


# ============================================================
#  Эталон: DAG -> CPDAG
# ============================================================

def test_dag_to_cpdag_leaves_reversible_edge_undirected():
    """Одно ребро обратимо, поэтому в CPDAG оно неориентированное."""
    m = dag_adj_to_cpdag_endpoint_matrix(np.array([[0, 1], [0, 0]]))
    assert m[0, 1] == TAIL and m[1, 0] == TAIL


def test_dag_to_cpdag_keeps_collider_oriented():
    """A -> C <- B: коллайдер определим по данным и остаётся направленным."""
    A = np.zeros((3, 3), dtype=int)
    A[0, 2] = A[1, 2] = 1
    m = dag_adj_to_cpdag_endpoint_matrix(A)
    assert m[0, 2] == ARROW and m[2, 0] == TAIL
    assert m[1, 2] == ARROW and m[2, 1] == TAIL


def test_pag_to_cpdag_proxy_turns_circles_into_tails():
    m = np.array([[0, CIRCLE], [CIRCLE, 0]])
    assert np.array_equal(convert_pag_to_cpdag_proxy(m), np.array([[0, TAIL], [TAIL, 0]]))


def test_pag_to_cpdag_proxy_does_not_mutate_the_input():
    m = np.array([[0, CIRCLE], [CIRCLE, 0]])
    before = m.copy()
    convert_pag_to_cpdag_proxy(m)
    assert np.array_equal(m, before)


# ============================================================
#  Метрики
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
    assert r["adj_f1"] == 1.0 and r["arrow_f1"] == 1.0


def test_shd_counts_each_pair_once():
    a = np.zeros((3, 3), dtype=int)
    b = np.zeros((3, 3), dtype=int)
    b[1, 0], b[0, 1] = TAIL, ARROW          # одно лишнее ребро
    assert compute_metrics_from_endpoint_matrices(a, b)["SHD"] == 1


def test_reversed_edge_costs_one_shd_but_keeps_adjacency():
    a = np.zeros((2, 2), dtype=int); a[1, 0], a[0, 1] = TAIL, ARROW
    b = np.zeros((2, 2), dtype=int); b[0, 1], b[1, 0] = TAIL, ARROW
    r = compute_metrics_from_endpoint_matrices(a, b)
    assert r["SHD"] == 1
    assert r["adj_f1"] == 1.0
    assert r["arrow_f1"] == 0.0


def test_missing_and_false_edge_shares_are_zero_on_a_perfect_match():
    """Пустой граф против пустого — идеальное совпадение, а не 100% ошибок."""
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
    # Тот же граф, но с кружками вместо хвостов — proxy обязан их снять.
    pag = est.copy()
    pag[pag == TAIL] = CIRCLE
    assert compute_metrics_from_dag_and_result(A, pag, is_pag=True)["SHD"] == 0


def test_calculate_metrics_on_general_graphs():
    A = np.zeros((3, 3), dtype=int)
    A[0, 2] = A[1, 2] = 1
    true_g = dag_adj_to_general_graph(A)
    r = calculate_metrics(true_g, true_g)
    assert r["SHD"] == 0 and r["adj_f1"] == 1.0


def test_calculate_metrics_chain_is_not_a_collider():
    """Regression: get_adj_matrix() returns an endpoint matrix (TAIL=-1,
    ARROW=1), but dag_adj_to_cpdag_endpoint_matrix expects a binary 0/1 DAG
    matrix. Passing the endpoint matrix straight through made every tail
    (-1) look like an edge, so a chain X1->X2->X3 turned into a spurious
    collider X1->X2<-X3 in the ground-truth CPDAG. The v-structure fixture
    above doesn't catch this: it's already a real collider, so the extra
    "parent" changes nothing.
    """
    A = np.zeros((3, 3), dtype=int)
    A[0, 1] = A[1, 2] = 1
    true_g = dag_adj_to_general_graph(A)

    true_dag_adj = (get_adj_matrix(true_g) == ARROW).astype(int)
    cpdag = dag_adj_to_cpdag_endpoint_matrix(true_dag_adj)
    assert cpdag[0, 1] == TAIL and cpdag[1, 0] == TAIL
    assert cpdag[1, 2] == TAIL and cpdag[2, 1] == TAIL

    # End-to-end: an estimate that correctly leaves both edges undirected
    # (the true Markov-equivalence class of a chain) must score as a
    # perfect match against calculate_metrics' ground-truth CPDAG. Before
    # the fix, the ground truth had a phantom collider here, so this
    # correct estimate would have scored as if it missed both arrowheads.
    est_g = adj_matrix_to_graph(cpdag, default_node_names(3))
    r = calculate_metrics(true_g, est_g)
    assert r["SHD"] == 0 and r["adj_f1"] == 1.0
