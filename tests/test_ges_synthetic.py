"""GES на синтетике: инварианты, краевые случаи, сходимость, параметры.

Отличие от tests/test_ges.py — там проверяется восстановление конкретных
канонических CPDAG, здесь свойства, которые обязаны выполняться независимо от
того, угадал алгоритм истинный граф или нет. Такие проверки не зависят от
численной грани и не требуют causal-learn (сверка с эталоном — в
tests/test_vs_causallearn_ges.py).
"""

import numpy as np
import pytest

from whydra.algorithms.score_based.ges import ges_search
from whydra.algorithms.score_based.scores import LocalScoreCache
from whydra.evaluation.graph_utils import get_adj_matrix
from whydra.evaluation.metrics import dag_adj_to_cpdag_endpoint_matrix

NULL, TAIL, ARROW = 0, -1, 1


# ============================================================
#  Хелперы
# ============================================================

def sample_linear_sem(W, n_samples, rng):
    p = W.shape[0]
    noise = rng.normal(size=(n_samples, p))
    out = np.zeros_like(noise)
    remaining = set(range(p))
    order = []
    while remaining:
        for v in sorted(remaining):
            if all(W[u, v] == 0 or u not in remaining for u in range(p)):
                order.append(v)
                remaining.discard(v)
                break
    for v in order:
        out[:, v] = noise[:, v] + out @ W[:, v]
    return out


def random_dag(p, n_edges, rng, low=0.7, high=1.5):
    W = np.zeros((p, p))
    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
    for k in rng.choice(len(pairs), size=min(n_edges, len(pairs)), replace=False):
        a, b = pairs[k]
        W[a, b] = rng.uniform(low, high) * rng.choice([-1, 1])
    return W


def run(data, score_func="local_score_BIC", **kwargs):
    p = data.shape[1]
    graph, score = ges_search(
        data, score_func, [f"X{i + 1}" for i in range(p)], **kwargs
    )
    return get_adj_matrix(graph), score


def edge_count(mat):
    p = mat.shape[0]
    return sum(
        1 for i in range(p) for j in range(i + 1, p) if mat[i, j] or mat[j, i]
    )


def shd(a, b):
    p = a.shape[0]
    return sum(
        1
        for i in range(p)
        for j in range(i + 1, p)
        if (a[i, j], a[j, i]) != (b[i, j], b[j, i])
    )


def dag_from_edges(p, edges, weight=1.1):
    W = np.zeros((p, p))
    for a, b in edges:
        W[a, b] = weight
    return W


# ============================================================
#  Инварианты
# ============================================================

def test_is_deterministic():
    rng = np.random.default_rng(1)
    data = sample_linear_sem(random_dag(7, 9, rng), 2000, rng)
    results = [run(data)[0].tobytes() for _ in range(3)]
    assert len(set(results)) == 1


@pytest.mark.parametrize("seed", range(5))
def test_equivariant_under_variable_permutation(seed):
    """Перенумерация переменных не меняет найденный граф.

    Если бы исход решался порядком перебора кандидатов, этот тест падал бы —
    он же и обосновывает, что побитовая сверка с causal-learn осмысленна.
    """
    rng = np.random.default_rng(100 + seed)
    p = 6
    data = sample_linear_sem(random_dag(p, 8, rng), 2000, rng)

    base = run(data)[0]
    perm = rng.permutation(p)
    restored = run(data[:, perm])[0][np.ix_(np.argsort(perm), np.argsort(perm))]
    assert np.array_equal(base, restored)


@pytest.mark.parametrize("seed", range(4))
def test_invariant_under_scaling_and_shifting(seed):
    """Гауссов BIC инвариантен к линейному преобразованию переменных."""
    rng = np.random.default_rng(200 + seed)
    p = 6
    data = sample_linear_sem(random_dag(p, 8, rng), 2000, rng)
    base = run(data)[0]

    assert np.array_equal(base, run(data * rng.uniform(0.2, 5.0, size=p))[0])
    assert np.array_equal(base, run(data + rng.uniform(-10, 10, size=p))[0])


@pytest.mark.parametrize("seed", range(4))
def test_invariant_under_row_permutation(seed):
    rng = np.random.default_rng(300 + seed)
    data = sample_linear_sem(random_dag(6, 8, rng), 1500, rng)
    shuffled = data[rng.permutation(data.shape[0])]
    assert np.array_equal(run(data)[0], run(shuffled)[0])


@pytest.mark.parametrize("seed", range(5))
def test_never_scores_worse_than_the_empty_graph(seed):
    """Поиск стартует с пустого графа и принимает только улучшения."""
    rng = np.random.default_rng(400 + seed)
    p = 6
    data = sample_linear_sem(random_dag(p, 8, rng), 1500, rng)
    _, score = run(data)

    cache = LocalScoreCache(data, "local_score_BIC")
    empty = sum(cache.score(v, ()) for v in range(p))
    assert score >= empty - 1e-9


# ============================================================
#  Семейства структур
# ============================================================

STRUCTURES = {
    "chain_5": (5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
    "star_out": (5, [(0, i) for i in range(1, 5)]),
    "star_in": (5, [(i, 0) for i in range(1, 5)]),
    "two_colliders": (5, [(0, 2), (1, 2), (2, 4), (3, 4)]),
    "diamond": (4, [(0, 1), (0, 2), (1, 3), (2, 3)]),
    "tree": (7, [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]),
    "disjoint": (6, [(0, 1), (1, 2), (3, 4), (4, 5)]),
    "bipartite": (6, [(0, 3), (0, 4), (1, 4), (1, 5), (2, 3), (2, 5)]),
    "lattice": (6, [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5)]),
}


@pytest.mark.parametrize("name", sorted(STRUCTURES))
def test_recovers_skeleton_on_structure_families(name):
    """Скелет восстанавливается точно при большой выборке.

    Проверяется именно скелет, а не ориентация: часть рёбер обратима внутри
    класса эквивалентности, и требовать от них конкретного направления
    неправильно.
    """
    p, edges = STRUCTURES[name]
    W = dag_from_edges(p, edges)
    data = sample_linear_sem(W, 5000, np.random.default_rng(1))
    mat = run(data)[0]

    expected = {(min(a, b), max(a, b)) for a, b in edges}
    got = {
        (i, j)
        for i in range(p)
        for j in range(i + 1, p)
        if mat[i, j] or mat[j, i]
    }
    assert got == expected


def test_independent_variables_give_an_empty_graph():
    rng = np.random.default_rng(0)
    mat, _ = run(rng.normal(size=(2000, 8)))
    assert edge_count(mat) == 0


def test_complete_graph_stays_fully_connected_and_unoriented():
    """Клика не содержит неэкранированных коллайдеров, значит ориентаций нет."""
    p = 4
    W = dag_from_edges(p, [(i, j) for i in range(p) for j in range(i + 1, p)], 0.8)
    data = sample_linear_sem(W, 5000, np.random.default_rng(1))
    mat = run(data)[0]

    assert edge_count(mat) == p * (p - 1) // 2
    for i in range(p):
        for j in range(i + 1, p):
            assert mat[i, j] == TAIL and mat[j, i] == TAIL


# ============================================================
#  Сходимость
# ============================================================

def test_accuracy_improves_with_sample_size():
    """Больше данных — ближе к истине: консистентность GES в грубой форме.

    Сравниваются края диапазона, а не соседние точки: на соседних шум сам по
    себе может дать неубывание, и тест стал бы хрупким. Суммарное SHD по
    нескольким графам тоже устойчивее отдельного.

    Точного восстановления при конечной выборке не требуется: на графах со
    слабыми коэффициентами ошибка остаётся и у эталонного causal-learn. Там,
    где сигнал заведомо сильный, точность проверяется отдельно — на
    канонических структурах в tests/test_ges.py.
    """
    small, large = [], []
    for seed in range(4):
        rng = np.random.default_rng(600 + seed)
        W = random_dag(6, 7, rng)
        truth = dag_adj_to_cpdag_endpoint_matrix((W != 0).astype(int))
        small.append(shd(run(sample_linear_sem(W, 100, rng))[0], truth))
        large.append(shd(run(sample_linear_sem(W, 20000, rng))[0], truth))

    assert sum(large) < sum(small), f"SHD не убыл: {small} -> {large}"


# ============================================================
#  Краевые случаи
# ============================================================

@pytest.mark.parametrize(
    "name, data_factory",
    [
        ("single_variable", lambda rng: rng.normal(size=(200, 1))),
        ("two_independent", lambda rng: rng.normal(size=(500, 2))),
        ("single_row", lambda rng: rng.normal(size=(1, 4))),
        ("fewer_rows_than_columns", lambda rng: rng.normal(size=(10, 15))),
    ],
)
def test_degenerate_shapes_do_not_crash(name, data_factory):
    mat, score = run(data_factory(np.random.default_rng(0)))
    assert np.isfinite(score)
    assert mat.shape[0] == mat.shape[1]


def test_constant_column_is_left_unconnected():
    rng = np.random.default_rng(0)
    data = sample_linear_sem(dag_from_edges(4, [(0, 1), (1, 2)]), 500, rng)
    data[:, 3] = 0.0
    mat, _ = run(data)
    assert not any(mat[3, j] or mat[j, 3] for j in range(4))


def test_duplicated_column_is_connected_to_its_source():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(500, 4))
    data[:, 3] = data[:, 1]
    mat, _ = run(data)
    assert mat[1, 3] or mat[3, 1]


def test_two_variable_edge_is_found_but_left_unoriented():
    """С двумя переменными направление неопределимо — обе ориентации равны."""
    rng = np.random.default_rng(0)
    data = sample_linear_sem(np.array([[0.0, 1.2], [0.0, 0.0]]), 2000, rng)
    mat, _ = run(data)
    assert mat[0, 1] == TAIL and mat[1, 0] == TAIL


# ============================================================
#  Дискретные данные и BDeu
# ============================================================

def discrete_chain(n, rng, card=3, keep=0.8):
    data = np.zeros((n, 3), dtype=int)
    data[:, 0] = rng.integers(0, card, n)
    for a, b in [(0, 1), (1, 2)]:
        data[:, b] = np.where(
            rng.random(n) < keep, data[:, a], rng.integers(0, card, n)
        )
    return data


def discrete_collider(n, rng, card=3):
    data = np.zeros((n, 3), dtype=int)
    data[:, 0] = rng.integers(0, card, n)
    data[:, 1] = rng.integers(0, card, n)
    total = data[:, 0] + data[:, 1]
    data[:, 2] = np.clip((total >= 2).astype(int) + (total >= 3).astype(int), 0, card - 1)
    flip = rng.random(n) < 0.1
    data[flip, 2] = rng.integers(0, card, flip.sum())
    return data


def test_bdeu_recovers_a_discrete_chain():
    data = discrete_chain(3000, np.random.default_rng(0))
    mat, _ = run(data, "local_score_BDeu")
    assert edge_count(mat) == 2
    assert (mat[0, 1] or mat[1, 0]) and (mat[1, 2] or mat[2, 1])


def test_bdeu_recovers_a_discrete_collider():
    data = discrete_collider(5000, np.random.default_rng(3))
    mat, _ = run(data, "local_score_BDeu")
    expected = dag_adj_to_cpdag_endpoint_matrix(
        np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    )
    assert np.array_equal(mat, expected)


def test_parity_structure_is_invisible_to_greedy_search():
    """Документирует предел метода, а не дефект реализации.

    При X3 = (X1 + X2) mod k переменные попарно независимы, поэтому ни один
    одиночный Insert не даёт прироста и forward не стартует. Так ведёт себя
    любой жадный score-based поиск, включая causal-learn, — проверено.
    """
    rng = np.random.default_rng(3)
    n, card = 5000, 3
    data = np.zeros((n, 3), dtype=int)
    data[:, 0] = rng.integers(0, card, n)
    data[:, 1] = rng.integers(0, card, n)
    data[:, 2] = (data[:, 0] + data[:, 1]) % card
    flip = rng.random(n) < 0.2
    data[flip, 2] = rng.integers(0, card, flip.sum())

    mat, _ = run(data, "local_score_BDeu")
    assert edge_count(mat) == 0


# ============================================================
#  Параметры поиска
# ============================================================

def test_operator_set_cap_of_one_already_matches_the_uncapped_search():
    """Дефолтная отсечка |T|,|H| <= 3 не меняет ответ на обычных графах."""
    rng = np.random.default_rng(11)
    data = sample_linear_sem(random_dag(7, 15, rng), 3000, rng)
    uncapped = run(data, max_operator_set=None)[0]
    for cap in (1, 2, 3):
        assert np.array_equal(run(data, max_operator_set=cap)[0], uncapped)


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_max_parents_is_respected(limit):
    rng = np.random.default_rng(11)
    data = sample_linear_sem(random_dag(7, 15, rng), 3000, rng)
    mat, _ = run(data, max_parents=limit)
    p = mat.shape[0]
    for child in range(p):
        n_parents = sum(
            1 for parent in range(p)
            if mat[parent, child] == ARROW and mat[child, parent] == TAIL
        )
        assert n_parents <= limit


def test_higher_min_delta_stops_the_search_earlier():
    rng = np.random.default_rng(11)
    data = sample_linear_sem(random_dag(7, 12, rng), 2000, rng)
    _, baseline = run(data, min_delta=0.0)
    _, strict = run(data, min_delta=500.0)
    assert strict <= baseline


# ============================================================
#  Интеграция: GES как стартовый граф GFCI
# ============================================================

GFCI_GRAPH = (7, [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (1, 6)])


def run_gfci(data, **kwargs):
    import warnings

    from whydra.algorithms.wrappers import create_standalone_algorithm

    kwargs.setdefault("alpha", 0.05)
    kwargs.setdefault("n_jobs", 1)
    kwargs.setdefault("show_progress", False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = create_standalone_algorithm("StandaloneGFCIStable", **kwargs).run(data)
    graph = result if hasattr(result, "get_graph_edges") else result.G
    return get_adj_matrix(graph)


def gfci_data():
    p, edges = GFCI_GRAPH
    return sample_linear_sem(dag_from_edges(p, edges), 2000, np.random.default_rng(3))


def test_gfci_recovers_the_skeleton_from_the_ges_start():
    p, edges = GFCI_GRAPH
    mat = run_gfci(gfci_data())
    expected = {(min(a, b), max(a, b)) for a, b in edges}
    got = {
        (i, j) for i in range(p) for j in range(i + 1, p) if mat[i, j] or mat[j, i]
    }
    assert got == expected


def test_gfci_is_invariant_to_n_jobs():
    data = gfci_data()
    assert np.array_equal(run_gfci(data, n_jobs=1), run_gfci(data, n_jobs=4))


def test_gfci_is_invariant_to_row_order():
    data = gfci_data()
    shuffled = data[np.random.default_rng(0).permutation(data.shape[0])]
    assert np.array_equal(run_gfci(data), run_gfci(shuffled))


def test_gfci_agrees_with_fci_on_the_skeleton():
    """GES-старт не должен уводить скелет от constraint-based эталона."""
    import warnings

    from whydra.algorithms.wrappers import create_standalone_algorithm

    data = gfci_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fci = create_standalone_algorithm(
            "StandaloneFCIStable", alpha=0.05, n_jobs=1, show_progress=False
        ).run(data)
    fci_mat = get_adj_matrix(fci if hasattr(fci, "get_graph_edges") else fci.G)

    skeleton = lambda m: {
        (i, j)
        for i in range(m.shape[0])
        for j in range(i + 1, m.shape[0])
        if m[i, j] or m[j, i]
    }
    assert skeleton(run_gfci(data)) == skeleton(fci_mat)


def test_gfci_reports_ges_statistics():
    from whydra.algorithms.graph_core import CIT, CausalGraph
    from whydra.algorithms.standalone.gfci_algo import gfci_stable

    p, _ = GFCI_GRAPH
    data = gfci_data()
    cg = CausalGraph(no_of_var=p, node_names=[f"X{i + 1}" for i in range(p)])
    cg.set_ind_test(CIT(data, method="fisherz"))
    gfci_stable(cg, data, alpha=0.05)

    stats = cg.ci_stats
    for key in ("ges_insert_steps", "ges_delete_steps", "ges_cache_hits", "ges_score"):
        assert key in stats, f"ci_stats не содержит {key}"
    assert stats["ges_insert_steps"] > 0
    assert stats["ges_edges"] > 0
