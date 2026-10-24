"""Сверка GES из whydra с оригиналом causal-learn.

Критерий строгий: побитовое совпадение CPDAG. Он выбран не наугад — эталон
``causallearn.ges`` детерминирован и инвариантен к перестановке переменных
(проверено отдельно), то есть тай-брейки при равных score исход не решают, и
две корректные реализации обязаны сойтись к одному классу эквивалентности.

Сверять с *истинным* графом на случайных инстансах нельзя: на конечной выборке
GES промахивается и сам по себе (эталон совпадает с истиной примерно в трети
случаев на 6-8 узлах). Поэтому истина проверяется только на канонических
структурах — в tests/test_ges.py, а здесь эталоном служит causal-learn.

causal-learn не является рантайм-зависимостью: он ставится отдельно через
requirements-comparison.txt, поэтому модуль скипается целиком без него.
"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("causallearn")

from causallearn.search.ScoreBased.GES import ges as cl_ges

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from whydra.algorithms.score_based.ges import ges_search
from whydra.evaluation.graph_utils import get_adj_matrix


def sample_linear_sem(W, n_samples, rng):
    """Данные из линейной SEM; порядок вершин выводится из самой матрицы."""
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


def random_dag(p, n_edges, rng):
    W = np.zeros((p, p))
    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
    for k in rng.choice(len(pairs), size=min(n_edges, len(pairs)), replace=False):
        a, b = pairs[k]
        W[a, b] = rng.uniform(0.6, 1.5) * rng.choice([-1, 1])
    return W


def causallearn_cpdag(data):
    """CPDAG эталона в кодировке пакета (mat[i,j] — метка у j на ребре i-j)."""
    g = cl_ges(data, score_func="local_score_BIC")["G"].graph
    p = g.shape[0]
    mat = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(p):
            if i != j and g[i, j] != 0:
                mat[i, j] = g[j, i]
    return mat


def whydra_cpdag(data):
    p = data.shape[1]
    graph, _ = ges_search(data, "local_score_BIC", [f"X{i + 1}" for i in range(p)])
    return get_adj_matrix(graph)


@pytest.mark.parametrize("seed", range(20))
def test_cpdag_matches_causallearn_on_small_graphs(seed):
    """p<=8: побитовое совпадение с эталоном."""
    rng = np.random.default_rng(1000 + seed)
    p = int(rng.integers(4, 9))
    data = sample_linear_sem(random_dag(p, int(1.3 * p), rng), 3000, rng)

    mine = whydra_cpdag(data)
    reference = causallearn_cpdag(data)
    assert np.array_equal(mine, reference), (
        f"p={p}: расхождений в матрице {int((mine != reference).sum())}"
    )


@pytest.mark.parametrize("seed", range(10))
def test_cpdag_matches_causallearn_on_medium_graphs(seed):
    """p=9..12 — там, где старый DAG hill-climbing расходился с эталоном."""
    rng = np.random.default_rng(3000 + seed)
    p = int(rng.integers(9, 13))
    data = sample_linear_sem(random_dag(p, int(1.4 * p), rng), 3000, rng)

    mine = whydra_cpdag(data)
    reference = causallearn_cpdag(data)
    assert np.array_equal(mine, reference), (
        f"p={p}: расхождений в матрице {int((mine != reference).sum())}"
    )


STRUCTURE_FAMILIES = {
    "chain_8": (8, [(i, i + 1) for i in range(7)]),
    "star_out": (5, [(0, i) for i in range(1, 5)]),
    "star_in": (5, [(i, 0) for i in range(1, 5)]),
    "two_colliders": (5, [(0, 2), (1, 2), (2, 4), (3, 4)]),
    "complete_4": (4, [(i, j) for i in range(4) for j in range(i + 1, 4)]),
    "complete_5": (5, [(i, j) for i in range(5) for j in range(i + 1, 5)]),
    "empty_4": (4, []),
    "tree_7": (7, [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]),
    "disjoint": (6, [(0, 1), (1, 2), (3, 4), (4, 5)]),
    "bipartite": (6, [(0, 3), (0, 4), (1, 4), (1, 5), (2, 3), (2, 5)]),
    "lattice": (6, [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5)]),
    "v_chain": (6, [(0, 2), (1, 2), (2, 3), (3, 5), (4, 5)]),
}


@pytest.mark.parametrize("name", sorted(STRUCTURE_FAMILIES))
@pytest.mark.parametrize("seed", (1, 2))
def test_structure_families_match_causallearn(name, seed):
    """Отдельные семейства графов, а не только случайные инстансы.

    Плотные клики и пустой граф — края, на которых легко разойтись:
    у клики нет ни одной v-структуры, у пустого графа не должно появиться
    ни одного ребра.
    """
    p, edges = STRUCTURE_FAMILIES[name]
    W = np.zeros((p, p))
    for a, b in edges:
        W[a, b] = 1.0

    data = sample_linear_sem(W, 5000, np.random.default_rng(seed))
    assert np.array_equal(whydra_cpdag(data), causallearn_cpdag(data))


@pytest.mark.parametrize("seed", range(4))
def test_discrete_data_matches_causallearn_under_bdeu(seed):
    """Дискретная ветка score тоже должна совпадать с эталоном."""
    rng = np.random.default_rng(seed)
    n, card = 3000, 3
    data = np.zeros((n, 3), dtype=int)
    data[:, 0] = rng.integers(0, card, n)
    for a, b in [(0, 1), (1, 2)]:
        data[:, b] = np.where(
            rng.random(n) < 0.8, data[:, a], rng.integers(0, card, n)
        )

    p = data.shape[1]
    mine, _ = ges_search(
        data, "local_score_BDeu", [f"X{i + 1}" for i in range(p)]
    )
    reference = cl_ges(data, score_func="local_score_BDeu")["G"].graph
    ref = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(p):
            if i != j and reference[i, j] != 0:
                ref[i, j] = reference[j, i]

    assert np.array_equal(get_adj_matrix(mine), ref)


def test_does_not_regress_to_the_old_hill_climbing_quality():
    """Регрессия против конкретного дефекта, из-за которого писался GES.

    На 12 узлах / 19 рёбрах старый поиск (add/delete/reverse по DAG) давал
    6-19 лишних рёбер на инстанс и верно ориентировал меньшую часть стрелок.
    Замеры на пяти сидах: лишних рёбер 16, 19, 6, 13, 7 при верных стрелках
    5/27, 8/36, 10/13, 11/27, 8/14. Порог ниже взят с запасом относительно
    худшего из этих чисел и не подгонялся под текущий результат.
    """
    hill_climbing_worst_extra_edges = 19
    total_extra = 0

    for seed in range(5):
        rng = np.random.default_rng(seed)
        W = random_dag(12, 19, rng)
        data = sample_linear_sem(W, 3000, rng)
        mine = whydra_cpdag(data)

        truth = W != 0
        extra = sum(
            1
            for i in range(12)
            for j in range(i + 1, 12)
            if (mine[i, j] != 0 or mine[j, i] != 0) and not (truth[i, j] or truth[j, i])
        )
        total_extra += extra
        assert extra < hill_climbing_worst_extra_edges, (
            f"seed {seed}: лишних рёбер {extra}, не лучше старого поиска"
        )
        assert np.array_equal(mine, causallearn_cpdag(data))

    assert total_extra <= 25, f"суммарно лишних рёбер {total_extra}"
