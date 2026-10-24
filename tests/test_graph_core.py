"""Модульные тесты графового ядра и тестов условной независимости.

Покрывают примитивы, на которых стоят все алгоритмы: Node/Edge/Endpoint,
GeneralGraph, CausalGraph и CIT. Большая часть проверок здесь появилась после
аудита — каждая закрывает найденный дефект, поэтому падение любой из них
означает возврат конкретной проблемы, а не абстрактную регрессию.
"""

import warnings

import numpy as np
import pytest

from whydra.algorithms.graph_core import (
    CIT,
    CONTINUOUS_TESTS,
    DISCRETE_TESTS,
    IMPLEMENTED_TESTS,
    CausalGraph,
    DegenerateColumnWarning,
    Edge,
    Endpoint,
    GeneralGraph,
    Node,
    fisher_z_from_corr,
    is_discrete_test,
    resolve_score_func,
)

NULL, TAIL, ARROW, CIRCLE = 0, -1, 1, 2


# ============================================================
#  Node / Edge / Endpoint
# ============================================================

def test_endpoint_values_match_causal_learn():
    """Кодировка библиотеки обязана совпадать с causal-learn."""
    from causallearn.graph.Endpoint import Endpoint as CLEndpoint

    for name in ("NULL", "TAIL", "ARROW", "CIRCLE"):
        assert getattr(Endpoint, name).value == getattr(CLEndpoint, name).value


def test_node_equality_and_hash_go_by_name():
    assert Node("A") == Node("A")
    assert hash(Node("A")) == hash(Node("A"))
    assert Node("A") != Node("B")
    assert len({Node("A"), Node("A"), Node("B")}) == 2


def test_node_ordering_is_by_name():
    assert sorted([Node("C"), Node("A"), Node("B")], key=str) == [
        Node("A"), Node("B"), Node("C")
    ]


def test_edge_reports_its_endpoints():
    e = Edge(Node("A"), Node("B"), Endpoint.TAIL, Endpoint.ARROW)
    assert e.get_node1() == Node("A") and e.get_node2() == Node("B")
    assert e.get_endpoint1() is Endpoint.TAIL
    assert e.get_endpoint2() is Endpoint.ARROW
    assert "A" in str(e) and "B" in str(e)


# ============================================================
#  GeneralGraph
# ============================================================

@pytest.fixture
def g3():
    return GeneralGraph([Node("A"), Node("B"), Node("C")])


def test_add_and_remove_edge_roundtrip(g3):
    nodes = g3.get_nodes()
    g3.add_edge(Edge(nodes[0], nodes[1], Endpoint.TAIL, Endpoint.ARROW))
    assert g3.is_adjacent_to(nodes[0], nodes[1])
    edge = g3.get_edge(nodes[0], nodes[1])
    assert edge.get_endpoint1() is Endpoint.TAIL
    assert edge.get_endpoint2() is Endpoint.ARROW

    g3.remove_edge(edge)
    assert not g3.is_adjacent_to(nodes[0], nodes[1])
    assert g3.get_edge(nodes[0], nodes[1]) is None


def test_matrix_convention_mark_at_target(g3):
    """m[i, j] — метка у j: A -> B это TAIL в m[1,0] и ARROW в m[0,1]."""
    nodes = g3.get_nodes()
    g3.add_directed_edge(nodes[0], nodes[1])
    assert g3.graph[1, 0] == TAIL
    assert g3.graph[0, 1] == ARROW


def test_has_edge_is_the_single_adjacency_predicate(g3):
    """Односторонняя метка видна ОБОИМ API — раньше они расходились."""
    g3.graph[0, 1] = NULL
    g3.graph[1, 0] = ARROW

    assert g3.has_edge(0, 1)
    assert g3.is_adjacent_to(g3.get_nodes()[0], g3.get_nodes()[1])
    assert len(g3.get_graph_edges()) == 1


def test_undirected_and_bidirected_helpers(g3):
    nodes = g3.get_nodes()
    g3.add_undirected_edge(nodes[0], nodes[1])
    assert g3.graph[0, 1] == TAIL and g3.graph[1, 0] == TAIL

    g3.add_bidirected_edge(nodes[1], nodes[2])
    assert g3.graph[1, 2] == ARROW and g3.graph[2, 1] == ARROW


def test_get_adjacent_nodes(g3):
    nodes = g3.get_nodes()
    g3.add_directed_edge(nodes[0], nodes[1])
    g3.add_directed_edge(nodes[2], nodes[1])
    assert set(map(str, g3.get_adjacent_nodes(nodes[1]))) == {"A", "C"}


# ============================================================
#  CausalGraph: валидация входа
# ============================================================

def test_complete_graph_is_built_undirected():
    cg = CausalGraph(3, ["A", "B", "C"])
    assert len(cg.G.get_graph_edges()) == 3
    assert all(cg.is_undirected(i, j) for i, j in [(0, 1), (0, 2), (1, 2)])


def test_duplicate_node_names_are_rejected():
    """Раньше дубликаты молча схлопывали граф: 3 узла -> 1 ребро вместо 3."""
    with pytest.raises(ValueError, match="unique"):
        CausalGraph(3, ["A", "A", "B"])


def test_node_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        CausalGraph(5, ["A", "B"])


def test_neighbors_and_max_degree():
    cg = CausalGraph(3, ["A", "B", "C"])
    assert cg.max_degree() == 2
    assert sorted(cg.neighbors(0).tolist()) == [1, 2]

    cg.remove_edge(cg.nodes[0], cg.nodes[1])
    assert sorted(cg.neighbors(0).tolist()) == [2]


def test_direction_predicates():
    cg = CausalGraph(2, ["A", "B"])
    assert cg.is_undirected(0, 1)

    cg.remove_edge(cg.nodes[0], cg.nodes[1])
    cg.add_directed_edge(cg.nodes[0], cg.nodes[1])
    assert cg.is_fully_directed(0, 1)
    assert not cg.is_undirected(0, 1)
    assert not cg.is_bidirected(0, 1)

    cg.remove_edge(cg.nodes[0], cg.nodes[1])
    cg.add_bidirected_edge(cg.nodes[0], cg.nodes[1])
    assert cg.is_bidirected(0, 1)


# ============================================================
#  CausalGraph: поиск структур
# ============================================================

def test_find_unshielded_triples_yields_ascending_pairs():
    """Тройки строятся по возрастающим индексам — на этом стоял мёртвый фильтр."""
    cg = CausalGraph(3, ["A", "B", "C"])
    cg.remove_edge(cg.nodes[0], cg.nodes[2])       # A - B - C, A и C не смежны

    triples = cg.find_unshielded_triples()
    assert triples == [(0, 1, 2)]
    assert all(x < z for x, _, z in triples)


def test_find_unshielded_triples_uses_both_sides_of_adjacency():
    cg = CausalGraph(3, ["A", "B", "C"])
    cg.G.graph[0, 2] = NULL                        # односторонняя метка
    cg.G.graph[2, 0] = ARROW
    # A и C всё ещё смежны -> тройки быть не должно
    assert cg.find_unshielded_triples() == []


def test_find_triangles():
    cg = CausalGraph(3, ["A", "B", "C"])
    assert cg.find_triangles() == [(0, 1, 2)]

    cg.remove_edge(cg.nodes[0], cg.nodes[2])
    assert cg.find_triangles() == []


def test_find_kites():
    cg = CausalGraph(4, ["A", "B", "C", "D"])
    cg.remove_edge(cg.nodes[1], cg.nodes[2])       # B и C не смежны
    kites = cg.find_kites()
    assert kites, "воздушный змей не найден"
    for i, j, k, l in kites:
        assert not cg.G.has_edge(j, k)
        assert cg.G.has_edge(i, j) and cg.G.has_edge(i, k) and cg.G.has_edge(i, l)


def test_ci_test_delegates_to_the_configured_test():
    cg = CausalGraph(2, ["A", "B"])
    cg.set_ind_test(lambda x, y, S: 0.42)
    assert cg.ci_test(0, 1, ()) == 0.42


# ============================================================
#  CIT: Fisher-Z
# ============================================================

def _linear_data(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    X[:, 1] += 1.5 * X[:, 0]        # X0 -> X1
    X[:, 2] += 1.5 * X[:, 1]        # X1 -> X2
    return X


def test_fisher_z_detects_dependence_and_independence():
    cit = CIT(_linear_data(), method="fisherz")
    assert cit(0, 1, []) < 0.01, "прямая зависимость не обнаружена"
    assert cit(0, 2, []) < 0.01, "косвенная зависимость не обнаружена"
    assert cit(0, 2, [1]) > 0.05, "условная независимость через X1 не обнаружена"


def test_fisher_z_is_symmetric():
    cit = CIT(_linear_data(), method="fisherz")
    assert cit(0, 2, [1]) == pytest.approx(cit(2, 0, [1]))


def test_fisher_z_matches_causal_learn():
    """Наш Fisher-Z обязан численно совпадать с эталоном."""
    from causallearn.utils.cit import CIT as CLCIT

    X = _linear_data()
    ours, theirs = CIT(X, "fisherz"), CLCIT(X, "fisherz")
    for x, y, S in [(0, 1, []), (0, 2, []), (0, 2, [1]), (1, 2, [0])]:
        assert ours(x, y, S) == pytest.approx(theirs(x, y, S), abs=1e-10)


def test_zero_variance_column_is_reported_and_never_faked_as_dependence():
    """Константа давала p=0.0 («сильнейшая зависимость») и ложную клику.

    Отказывать нельзя: вырожденные столбцы встречаются в реальных наборах, а
    эталонные реализации их переваривают. Поэтому предупреждение по умолчанию,
    а неопределённость остаётся неопределённостью (NaN), а не нулём.
    """
    X = _linear_data()
    X[:, 1] = 7.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cit = CIT(X, method="fisherz")
    assert any(w.category is DegenerateColumnWarning for w in caught)
    assert "[1]" in str(caught[0].message)
    assert np.isnan(cit(0, 1, [])), "неопределённость подменена конечным значением"


def test_strict_degenerate_turns_the_warning_into_an_error():
    X = _linear_data()
    X[:, 1] = 7.0
    with pytest.raises(ValueError, match=r"zero variance"):
        CIT(X, method="fisherz", strict_degenerate=True)


def test_nan_column_is_reported_and_returns_nan():
    X = _linear_data(n=200)
    X[0, 1] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cit = CIT(X, method="fisherz")
    degenerate = [w for w in caught if w.category is DegenerateColumnWarning]
    assert len(degenerate) == 1
    assert "[1]" in str(degenerate[0].message)
    assert "missing values (NaN)" in str(degenerate[0].message)
    assert np.isnan(cit(0, 1, []))


def test_strict_degenerate_rejects_nan_column():
    X = _linear_data(n=200)
    X[0, 1] = np.nan
    with pytest.raises(ValueError, match=r"missing values \(NaN\)"):
        CIT(X, method="fisherz", strict_degenerate=True)


def test_fisher_z_without_degrees_of_freedom_returns_one():
    """n - |S| - 3 <= 0: раньше падало ValueError из math.sqrt."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 8))
    cit = CIT(X, method="fisherz")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = cit(0, 1, [2, 3, 4, 5])
    assert p == 1.0
    assert any(w.category is RuntimeWarning for w in caught)


def test_fisher_z_from_corr_propagates_nan_instead_of_inventing_dependence():
    corr = np.array([[1.0, np.nan], [np.nan, np.nan]])
    assert np.isnan(fisher_z_from_corr(corr, 100, 0, 1, []))


def test_fisher_z_from_corr_is_the_shared_implementation():
    """CIT.fisher_z обязан быть тонкой обёрткой над общей функцией."""
    X = _linear_data()
    cit = CIT(X, method="fisherz")
    assert cit.fisher_z(0, 2, [1]) == fisher_z_from_corr(
        cit.corr_matrix, cit.n_samples, 0, 2, [1]
    )


# ============================================================
#  CIT: chi-square
# ============================================================

def _discrete_data(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 3, size=n)
    b = (a + rng.integers(0, 2, size=n)) % 3
    c = rng.integers(0, 3, size=n)
    return np.column_stack([a, b, c])


def test_chi_square_detects_dependence_and_independence():
    cit = CIT(_discrete_data(), method="chisq")
    assert cit(0, 1, []) < 0.01
    assert cit(0, 2, []) > 0.05


def test_chi_square_matches_causal_learn():
    from causallearn.utils.cit import CIT as CLCIT

    X = _discrete_data()
    ours, theirs = CIT(X, "chisq"), CLCIT(X, "chisq")
    for x, y, S in [(0, 1, []), (0, 2, []), (1, 2, [0])]:
        assert ours(x, y, S) == pytest.approx(theirs(x, y, S), abs=1e-8)


def test_chi_square_degrades_to_a_valid_p_value_when_cells_are_sparse():
    """При разреженных ячейках тест обязан вернуть валидное p, а не NaN."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 4, size=(60, 6))
    cit = CIT(X, method="chisq")
    p = cit(0, 1, [2, 3, 4, 5])
    assert np.isfinite(p) and 0.0 <= p <= 1.0


def test_chi_square_returns_one_when_no_stratum_is_usable():
    """Если ни одна страта не даёт степеней свободы — независимость недоказуема."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 5, size=(20, 8))
    cit = CIT(X, method="chisq")
    assert cit(0, 1, [2, 3, 4, 5, 6, 7]) == 1.0


def test_chi_square_reports_a_single_level_column():
    """Кардинальность 1 даёт df=0 и NaN: предупреждаем, но не ломаем прогон.

    В bnlearn/water таких столбцов шесть, и causal-learn их переваривает —
    жёсткий отказ сделал бы сверку с эталоном невозможной.
    """
    X = _discrete_data()
    X[:, 2] = 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CIT(X, method="chisq")
    assert any(w.category is DegenerateColumnWarning for w in caught)
    assert "[2]" in str(caught[0].message)


def test_chi_square_strict_mode_rejects_a_single_level_column():
    X = _discrete_data()
    X[:, 2] = 1
    with pytest.raises(ValueError, match="single observed level"):
        CIT(X, method="chisq", strict_degenerate=True)


# ============================================================
#  CIT: контракт и служебное
# ============================================================

def test_unimplemented_test_names_raise_with_the_available_list():
    cit = CIT(_linear_data(), method="kci")
    with pytest.raises(NotImplementedError, match="chisq"):
        cit(0, 1, [])


def test_test_name_registries_are_consistent():
    assert IMPLEMENTED_TESTS <= (DISCRETE_TESTS | CONTINUOUS_TESTS)
    assert IMPLEMENTED_TESTS == {"fisherz", "chisq"}
    assert not (DISCRETE_TESTS & CONTINUOUS_TESTS)


def test_is_discrete_test():
    assert is_discrete_test("chisq")
    assert not is_discrete_test("fisherz")


def test_resolve_score_func_follows_the_data_type():
    assert resolve_score_func("auto", "chisq") == "local_score_BDeu"
    assert resolve_score_func("auto", "fisherz") == "local_score_BIC"
    assert resolve_score_func(None, "fisherz") == "local_score_BIC"
    # Явный выбор всегда сильнее автоопределения.
    assert resolve_score_func("my_score", "chisq") == "my_score"


def test_profile_decorator_is_a_noop_without_the_env_var():
    from whydra.algorithms.graph_core.independence_tests import profile

    def fn():
        return 1

    assert profile(fn) is fn
