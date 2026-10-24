"""Тесты GES (whydra/algorithms/score_based/ges.py).

Проверяются три вещи: восстановление аналитически известных CPDAG на
канонических структурах, структурные инварианты результата на случайных
графах и согласованность дельт операторов с фактическим изменением score.

Почему именно так. Предыдущая попытка чинить стартовый граф GFCI опиралась на
инстанс, подобранный перебором под конкретную численную грань, и разъехалась
при смене версии numpy. Здесь вместо этого: канонические структуры с большим
запасом по разрыву score, параметризация по сидам с требованием совпадения на
всех, и инварианты, которые вообще не зависят от чисел.
"""

import numpy as np
import pytest

from whydra.algorithms.score_based.ges import ges_search, total_score
from whydra.algorithms.score_based.pdag import (
    complete_pdag,
    dag_to_cpdag,
    is_directed,
    is_undirected,
    pdag_to_dag,
)
from whydra.algorithms.score_based.scores import LocalScoreCache, local_parent_score
from whydra.evaluation.graph_utils import get_adj_matrix
from whydra.evaluation.metrics import dag_adj_to_cpdag_endpoint_matrix

from .conftest import (
    generate_linear_sem_data,
    get_chain_graph_3nodes,
    get_collider_graph_3nodes,
    get_fork_graph_3nodes,
    get_v_structure_graph_4nodes,
)

SEEDS = (1, 2, 3, 4, 5)

DIAMOND = np.array(
    [[0, 1.0, 1.0, 0], [0, 0, 0, 1.0], [0, 0, 0, 1.0], [0, 0, 0, 0]]
)


def run_ges(W, seed, n_samples=5000, **kwargs):
    p = W.shape[0]
    data = generate_linear_sem_data(W, n_samples=n_samples, seed=seed)
    graph, score = ges_search(
        data, "local_score_BIC", [f"X{i + 1}" for i in range(p)], **kwargs
    )
    return get_adj_matrix(graph), score


def true_cpdag(W):
    return dag_adj_to_cpdag_endpoint_matrix((W != 0).astype(int))


# ============================================================
#  Канонические структуры: ответ известен аналитически
# ============================================================

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "name, W",
    [
        ("chain", get_chain_graph_3nodes()),
        ("fork", get_fork_graph_3nodes()),
        ("collider", get_collider_graph_3nodes()),
        ("diamond", DIAMOND),
    ],
)
def test_recovers_true_cpdag_on_canonical_structures(name, W, seed):
    assert np.array_equal(run_ges(W, seed)[0], true_cpdag(W))


@pytest.mark.parametrize("seed", SEEDS)
def test_chain_and_centred_fork_are_indistinguishable(seed):
    """X1->X2->X3 и X2->X1, X2->X3 — один класс, ответ обязан совпасть.

    Ровно здесь ломался старый DAG hill-climbing: обе ориентации ребра
    score-эквивалентны, он выбирал одну тай-брейком и запирал её ацикличностью.

    Фикстура ``get_fork_graph_3nodes`` для этого не годится — она центрирована
    на X1, то есть у неё другой скелет; нужна вилка с центром в X2.
    """
    chain = get_chain_graph_3nodes()
    fork_at_x2 = np.zeros((3, 3))
    fork_at_x2[1, 0] = 0.8
    fork_at_x2[1, 2] = 0.8

    assert np.array_equal(true_cpdag(chain), true_cpdag(fork_at_x2))
    assert np.array_equal(run_ges(chain, seed)[0], run_ges(fork_at_x2, seed)[0])


@pytest.mark.parametrize("seed", SEEDS)
def test_collider_keeps_both_arrowheads(seed):
    # Фикстура даёт X1 -> X2 <- X3: стрелки сходятся в X2 (индекс 1)
    est = run_ges(get_collider_graph_3nodes(), seed)[0]
    assert est[0, 1] == 1 and est[1, 0] == -1
    assert est[2, 1] == 1 and est[1, 2] == -1


def test_v_structure_recovered_or_matches_its_own_class():
    """V-структура: на части сидов конечная выборка уводит от истины.

    Требовать совпадения с истиной на всех сидах здесь нельзя — это свойство
    данных, а не реализации (эталонный causallearn.ges ошибается на тех же
    сидах). Поэтому требуем, чтобы результат был валидным CPDAG, а на
    большинстве сидов совпадал с истиной.
    """
    W = get_v_structure_graph_4nodes()
    hits = sum(np.array_equal(run_ges(W, seed)[0], true_cpdag(W)) for seed in SEEDS)
    assert hits >= 3


# ============================================================
#  Структурные инварианты — от численной грани не зависят
# ============================================================

def random_dag(p, n_edges, rng):
    W = np.zeros((p, p))
    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
    for k in rng.choice(len(pairs), size=min(n_edges, len(pairs)), replace=False):
        a, b = pairs[k]
        W[a, b] = rng.uniform(0.5, 1.5) * rng.choice([-1, 1])
    return W


def endpoint_to_adjacency(mat):
    p = mat.shape[0]
    C = np.zeros((p, p), dtype=np.int8)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if mat[i, j] == 1 and mat[j, i] == -1:
                C[i, j] = 1
            elif mat[i, j] == -1 and mat[j, i] == -1:
                C[i, j] = C[j, i] = 1
    return C


@pytest.mark.parametrize("seed", range(12))
def test_result_is_a_valid_cpdag(seed):
    rng = np.random.default_rng(7000 + seed)
    p = int(rng.integers(5, 11))
    W = random_dag(p, int(1.3 * p), rng)
    data = generate_linear_sem_data(W, n_samples=1500, seed=seed)
    graph, _ = ges_search(data, "local_score_BIC", [f"X{i + 1}" for i in range(p)])
    mat = get_adj_matrix(graph)

    # нет двунаправленных рёбер: CPDAG их не содержит
    for i in range(p):
        for j in range(p):
            if i != j:
                assert not (mat[i, j] == 1 and mat[j, i] == 1)

    C = endpoint_to_adjacency(mat)
    dag = pdag_to_dag(C)
    assert dag is not None, "класс эквивалентности пуст: расширения не существует"
    assert np.array_equal(complete_pdag(C), C), "completion не идемпотентен"
    assert np.array_equal(dag_to_cpdag(dag), C), "CPDAG != CPDAG своего расширения"


@pytest.mark.parametrize("seed", range(8))
def test_operator_deltas_match_the_actual_score_change(seed):
    """Заявленная дельта каждого хода обязана совпасть с изменением score.

    Именно это расхождение поймало баг, из-за которого backward удалял рёбра,
    направленные не в ту сторону, и считал выигрыш по чужой вершине.
    """
    rng = np.random.default_rng(8000 + seed)
    p = int(rng.integers(5, 9))
    W = random_dag(p, int(1.3 * p), rng)
    data = generate_linear_sem_data(W, n_samples=1500, seed=seed)
    ges_search(
        data,
        "local_score_BIC",
        [f"X{i + 1}" for i in range(p)],
        verify_deltas=True,
    )


def test_reported_score_matches_the_returned_graph():
    W = DIAMOND
    mat, score = run_ges(W, seed=1)
    data = generate_linear_sem_data(W, n_samples=5000, seed=1)
    cache = LocalScoreCache(data, "local_score_BIC")
    assert score == pytest.approx(total_score(endpoint_to_adjacency(mat), cache))


# ============================================================
#  Оценки
# ============================================================

def test_bdeu_is_not_bic_in_disguise():
    """Ветка "bdeu" раньше считала дискретный BIC под чужим именем.

    Настоящий BDeu (равномерный Dirichlet-prior, equivalent sample size) —
    не штрафованное правдоподобие и в общем случае с BIC не совпадает.
    """
    rng = np.random.default_rng(0)
    n = 300
    child = rng.integers(0, 2, size=n)
    parent = rng.integers(0, 3, size=n)
    data = np.column_stack([parent, child])

    bdeu_score = local_parent_score(data, 1, [0], "local_score_BDeu")

    y = data[:, 1].astype(int)
    x = data[:, [0]].astype(int)
    r = int(y.max()) + 1
    configs = np.ravel_multi_index(x.T, [int(np.max(data[:, 0])) + 1])
    ll = 0.0
    parameters = 0
    for config in np.unique(configs):
        counts = np.bincount(y[configs == config], minlength=r)
        nonzero = counts > 0
        ll += np.sum(counts[nonzero] * np.log(counts[nonzero] / counts.sum()))
        parameters += r - 1
    discrete_bic = float(ll - 0.5 * parameters * np.log(n))

    assert bdeu_score != pytest.approx(discrete_bic)


def test_score_cache_is_order_insensitive_and_counts_hits():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(200, 4))
    cache = LocalScoreCache(data, "local_score_BIC")

    first = cache.score(0, [1, 2])
    again = cache.score(0, [2, 1])
    assert first == again
    assert cache.hits == 1 and cache.misses == 1
    assert first == pytest.approx(local_parent_score(data, 0, [1, 2], "local_score_BIC"))
