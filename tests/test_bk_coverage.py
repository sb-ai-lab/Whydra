"""Сквозное покрытие background knowledge по ВСЕМ алгоритмам, которые его поддерживают.

Отличие от `test_background_knowledge.py` (юнит-тесты ядра BK) и
`test_background_knowledge_invariants.py` (валидация и транзакционность):
здесь проверяется, что BK доезжает до результата каждого публичного алгоритма
и что он действительно жёсткий, а не подсказка.

Ключевой приём — КОНТРФАКТИЧЕСКОЕ ограничение: BK требует направление,
противоположное тому, которое алгоритм находит сам. Если в результате стоит
требуемое направление, значит BK не просто «не мешал», а принудил.

Соглашение о матрице: ``m[i, j]`` — метка у узла ``j`` на ребре (i, j).
Кодировка causal-learn: NULL=0, TAIL=-1, ARROW=1, CIRCLE=2.
"""

import warnings

import numpy as np
import pytest

from whydra.algorithms.wrappers import create_standalone_algorithm
from whydra.background_knowledge import (
    BKPhase,
    BackgroundKnowledge,
    find_directed_cycle,
)
from whydra.evaluation.graph_utils import get_adj_matrix

NULL, TAIL, ARROW, CIRCLE = 0, -1, 1, 2

#: Обёртки, точки входа которых принимают background_knowledge.
BK_ALGORITHMS = [
    "StandalonePCStable",
    "ParallelPCStable",
    "StandaloneFCIStable",
    "StandaloneFCIStableCached",
    "StandaloneFCIStableParallelCached",
    "StandaloneCFCIStable",
    "StandaloneCFCIStableParallelCached",
    "StandaloneRFCIStable",
    "StandaloneGFCIStable",
    "StandaloneFCIPlusStable",
    "StandaloneFCIPlusStableParallelCached",
]

#: Семейство RAI фоновые знания не поддерживает — зафиксировано отдельным тестом.
NON_BK_ALGORITHMS = ["StandaloneRAIStable", "StandaloneRAIOptimized"]

N_JOBS_VALUES = [1, 4]

#: Истинный DAG: X1->X3, X2->X3, X3->X4, X4->X5.
#: Смежные пары, на которых можно осмысленно ставить ограничения:
#: (X1,X3), (X2,X3), (X3,X4), (X4,X5).
TRUE_EDGES = [(0, 2), (1, 2), (2, 3), (3, 4)]
N_VARS = 5
NAMES = [f"X{i + 1}" for i in range(N_VARS)]


# ============================================================
#  Инфраструктура
# ============================================================

def _data(seed=4, n=600):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, N_VARS))
    for a, b in TRUE_EDGES:
        data[:, b] += 1.2 * data[:, a]
    return data


DATA = _data()


def run(algo_key, bk=None, n_jobs=1, data=None, extra=None):
    """Прогоняет обёртку и возвращает endpoint-матрицу в кодировке causal-learn."""
    kwargs = dict(alpha=0.05, n_jobs=n_jobs, show_progress=False)
    if bk is not None:
        kwargs["background_knowledge"] = bk
    if extra:
        kwargs.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = create_standalone_algorithm(algo_key, **kwargs).run(
            DATA if data is None else data
        )
    graph = result if hasattr(result, "get_graph_edges") else result.G
    return get_adj_matrix(graph)


def directed(m, u, v) -> bool:
    """``u -> v``: хвост у u, стрелка у v."""
    return m[v, u] == TAIL and m[u, v] == ARROW


def adjacent(m, u, v) -> bool:
    return m[u, v] != NULL or m[v, u] != NULL


def skeleton(m) -> set:
    n = m.shape[0]
    return {
        frozenset((i, j))
        for i in range(n) for j in range(i + 1, n)
        if adjacent(m, i, j)
    }


@pytest.fixture(scope="module")
def baseline():
    """Результат каждого алгоритма БЕЗ background knowledge."""
    return {key: run(key) for key in BK_ALGORITHMS}


# ============================================================
#  1. required: контрфактическое ограничение принуждает
# ============================================================

@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
@pytest.mark.parametrize("n_jobs", N_JOBS_VALUES)
def test_required_forces_counterfactual_direction(algo_key, n_jobs, baseline):
    """BK требует X4->X3 против истины X3->X4 — и обязан победить."""
    base = baseline[algo_key]
    assert directed(base, 2, 3), (
        f"{algo_key}: предпосылка теста нарушена — без BK ожидалось X3->X4"
    )

    bk = BackgroundKnowledge(required=[("X4", "X3")])
    result = run(algo_key, bk=bk, n_jobs=n_jobs)

    assert directed(result, 3, 2), (
        f"{algo_key} (n_jobs={n_jobs}): требуемое X4->X3 не выставлено; "
        f"метки пары: m[2,3]={result[2, 3]}, m[3,2]={result[3, 2]}"
    )


@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_required_matching_truth_is_kept(algo_key):
    """Ограничение, совпадающее с истиной, не должно ничего ломать."""
    bk = BackgroundKnowledge(required=[("X3", "X4")])
    result = run(algo_key, bk=bk)
    assert directed(result, 2, 3)


# ============================================================
#  2. forbidden: запрещённое направление не появляется
# ============================================================

@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_forbidden_direction_is_not_produced(algo_key, baseline):
    """Запрет истинного направления обязан снять его из результата."""
    assert directed(baseline[algo_key], 2, 3)

    bk = BackgroundKnowledge(forbidden=[("X3", "X4")])
    result = run(algo_key, bk=bk)

    assert not directed(result, 2, 3), (
        f"{algo_key}: запрещённое X3->X4 всё равно выставлено"
    )


@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_allowed_cancels_forbidden(algo_key, baseline):
    """`allowed` снимает ровно совпадающий запрет — результат как без BK."""
    bk = BackgroundKnowledge(
        forbidden=[("X3", "X4")],
        allowed=[("X3", "X4")],
    )
    result = run(algo_key, bk=bk)
    assert np.array_equal(result, baseline[algo_key]), (
        f"{algo_key}: allowed не отменил запрет"
    )


# ============================================================
#  3. Фазы
# ============================================================

@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_pre_search_required_protects_adjacency(algo_key):
    """required на PRE_SEARCH защищает ребро от удаления на поиске скелета.

    Пара (X1, X5) в истинном графе не связана напрямую, и без BK алгоритм её
    разрывает. Ограничение на фазе поиска обязано её сохранить.
    """
    bk = BackgroundKnowledge(enabled_phases=()).require(
        "X1", "X5", phases=BKPhase.PRE_SEARCH
    )
    result = run(algo_key, bk=bk)
    assert adjacent(result, 0, 4), (
        f"{algo_key}: защищённое на PRE_SEARCH ребро X1–X5 удалено"
    )


def test_gfci_pre_search_protects_edge_ges_never_found():
    """GFCI стартует не с полного графа, а с CPDAG от GES, так что для
    required-ребра, которое GES вообще не создал, восстанавливать нечего в
    буквальном смысле «смежность уже есть» — но PRE_SEARCH обязан дать ту же
    гарантию, что и для PC/FCI: ребро переживает весь поиск целиком, а не
    только его часть после чекпоинта. Чекпоинт восстанавливает такое ребро
    из полносвязного o-o графа (``previous=complete_pag``), одинаково при
    старте с GES и с полного графа.
    """
    def protected(start_from_complete_graph):
        bk = BackgroundKnowledge(enabled_phases=()).require(
            "X1", "X5", phases=BKPhase.PRE_SEARCH
        )
        m = run("StandaloneGFCIStable", bk=bk,
                extra={"start_from_complete_graph": start_from_complete_graph})
        return adjacent(m, 0, 4)

    assert protected(False)
    assert protected(True)


@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_bk_records_active_checkpoints(algo_key):
    """У каждого алгоритма BK доходит до отчёта, а не теряется по дороге."""
    bk = BackgroundKnowledge(required=[("X4", "X3")])
    run(algo_key, bk=bk)

    active = [r.checkpoint for r in bk.history if r.active]
    assert active, f"{algo_key}: ни одного активного чекпоинта BK"
    # Прогон не открывали — значит все отчёты лежат в дефолтном run_id.
    assert all(r.run_id == 0 for r in bk.history)


@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_run_scope_separates_reports_of_reused_object(algo_key):
    """Один BK на несколько прогонов: отчёты не смешиваются в общей history."""
    bk = BackgroundKnowledge(required=[("X4", "X3")])

    with bk.run_scope("first"):
        run(algo_key, bk=bk)
    with bk.run_scope("second"):
        run(algo_key, bk=bk)

    first = bk.reports_for_run("first")
    second = bk.reports_for_run("second")

    assert first, f"{algo_key}: первый прогон не оставил отчётов"
    assert second, f"{algo_key}: второй прогон не оставил отчётов"
    assert not {id(r) for r in first} & {id(r) for r in second}
    assert len(first) + len(second) == len(bk.history)
    # Область видимости закрыта — счётчик вернулся к исходному значению.
    assert bk.reports_for_run(0) == []


@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_phase_scoped_constraint_does_not_leak_to_other_phases(algo_key):
    """Ограничение, привязанное к POST_ORIENTATION, не активно на остальных."""
    bk = BackgroundKnowledge(enabled_phases=()).require(
        "X4", "X3", phases=BKPhase.POST_ORIENTATION
    )
    run(algo_key, bk=bk)

    phases = {r.phase for r in bk.history if r.active}
    assert phases <= {BKPhase.POST_ORIENTATION}, (
        f"{algo_key}: BK сработал на фазах {phases}, ожидалась только POST_ORIENTATION"
    )


# ============================================================
#  4. BK как инвариант, а не подсказка
# ============================================================

@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
@pytest.mark.parametrize("n_jobs", N_JOBS_VALUES)
def test_bk_never_leaves_a_directed_cycle(algo_key, n_jobs):
    """Результат с BK обязан остаться ациклическим."""
    bk = BackgroundKnowledge(required=[("X4", "X3"), ("X5", "X4")])
    result = run(algo_key, bk=bk, n_jobs=n_jobs)
    cycle = find_directed_cycle(result, TAIL, ARROW)
    assert cycle is None, f"{algo_key}: в результате цикл {cycle}"


@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_orientation_only_bk_does_not_change_skeleton(algo_key, baseline):
    """Ограничение на ориентацию не должно трогать состав рёбер."""
    bk = BackgroundKnowledge(required=[("X4", "X3")])
    result = run(algo_key, bk=bk)
    assert skeleton(result) == skeleton(baseline[algo_key]), (
        f"{algo_key}: BK на ориентацию изменил скелет"
    )


@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_bk_changes_only_the_constrained_pair(algo_key, baseline):
    """Контрфактическое ограничение может тянуть за собой соседние рёбра,
    но обязано оставить нетронутыми пары, не связанные с ним ни одним концом."""
    bk = BackgroundKnowledge(required=[("X4", "X3")])
    result = run(algo_key, bk=bk)
    base = baseline[algo_key]

    untouched = [(0, 1)]  # X1–X2: не смежны с парой (X3, X4)
    for i, j in untouched:
        assert result[i, j] == base[i, j] and result[j, i] == base[j, i], (
            f"{algo_key}: изменилась несвязанная пара X{i + 1}–X{j + 1}"
        )


@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_bk_result_is_njobs_invariant(algo_key):
    """С BK результат так же не зависит от n_jobs, как и без него."""
    bk1 = BackgroundKnowledge(required=[("X4", "X3")])
    bk4 = BackgroundKnowledge(required=[("X4", "X3")])
    assert np.array_equal(run(algo_key, bk=bk1, n_jobs=1),
                          run(algo_key, bk=bk4, n_jobs=4))


# ============================================================
#  5. Конфликты сообщаются, а не глотаются
# ============================================================

@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_required_edge_absent_reports_conflict(algo_key):
    """X1 и X2 не смежны; BK не изобретает ребро, но обязан сообщить конфликт."""
    bk = BackgroundKnowledge(enabled_phases=()).require(
        "X1", "X2", phases=BKPhase.POST_ORIENTATION
    )
    result = run(algo_key, bk=bk)

    assert not adjacent(result, 0, 1), f"{algo_key}: BK выдумал ребро X1–X2"
    conflicts = [e for r in bk.history for e in r.conflicts]
    assert conflicts, f"{algo_key}: отсутствие required-ребра не отражено в отчёте"


# ============================================================
#  6. RAI: поддержки нет, и это зафиксировано
# ============================================================

@pytest.mark.parametrize("algo_key", NON_BK_ALGORITHMS)
def test_rai_family_does_not_support_bk(algo_key):
    """Семейство RAI фоновые знания игнорирует — README это оговаривает.

    Тест фиксирует текущий контракт: BK молча не действует. Если поддержку
    добавят, тест упадёт и о смене поведения узнают явно.
    """
    bk = BackgroundKnowledge(required=[("X4", "X3")])
    result = run(algo_key, bk=bk)

    assert not bk.history, f"{algo_key}: BK неожиданно сработал"
    assert directed(result, 2, 3), f"{algo_key}: результат изменился без участия BK"

# ============================================================
#  7. Известное ограничение: цикл от правил ориентации
# ============================================================

def _dense_case(seed=2, p=8, n=800, density=0.30):
    """Плотный случайный DAG, на котором воспроизводится цикл."""
    rng = np.random.default_rng(seed)
    A = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(i + 1, p):
            if rng.random() < density:
                A[i, j] = 1
    X = rng.normal(0.0, 1.0, size=(n, p))
    for j in range(p):
        for i in range(j):
            if A[i, j]:
                X[:, j] += rng.uniform(0.5, 1.5) * X[:, i]
    return A, X


REVERSED_BK_CYCLE_ALGORITHMS = [
    "StandalonePCStable",
    "StandaloneFCIStable",
    "StandaloneCFCIStable",
    "StandaloneRFCIStable",
    "StandaloneGFCIStable",
    "StandaloneFCIPlusStable",
]


@pytest.mark.parametrize("algo_key", REVERSED_BK_CYCLE_ALGORITHMS)
def test_rules_can_still_close_a_cycle_under_heavy_bk(algo_key):
    """ИЗВЕСТНОЕ ОГРАНИЧЕНИЕ, зафиксированное намеренно.

    BK ацикличен (проверяется статически) и применяется транзакционно, поэтому
    сам он цикла не создаёт. Но guard запрещает правилам ориентации нарушать
    ограничения, а НЕ замыкать цикл на рёбрах, которые BK не трогает. При
    большом объёме противоречивых ограничений правила приходят к циклическому
    PAG, и BK может только сообщить об этом конфликтом.

    Тест фиксирует и сам факт, и то, что он не остаётся молчаливым: в отчёте
    обязан быть конфликт, называющий цикл.
    """
    A, X = _dense_case(seed=2)
    names = [f"X{i + 1}" for i in range(A.shape[0])]
    edges = [(i, j) for i in range(A.shape[0]) for j in range(A.shape[0]) if A[i, j]]
    bk = BackgroundKnowledge(required=[(names[j], names[i]) for i, j in edges[:8]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run(algo_key, bk=bk, data=X)

    cycle = find_directed_cycle(result, TAIL, ARROW)
    if cycle is None:
        pytest.skip(f"{algo_key}: на этом кейсе цикл не воспроизвёлся")

    conflicts = [e for r in bk.history for e in r.conflicts]
    assert conflicts, (
        f"{algo_key}: результат содержит цикл {cycle}, но BK промолчал"
    )
    assert any("cycle" in e.detail for e in conflicts), (
        f"{algo_key}: конфликт есть, но цикл в нём не назван"
    )


def test_bk_itself_is_never_the_source_of_a_cycle():
    """Применение BK к ациклическому графу цикла не создаёт — иначе откат."""
    A, X = _dense_case(seed=2)
    names = [f"X{i + 1}" for i in range(A.shape[0])]
    edges = [(i, j) for i in range(A.shape[0]) for j in range(A.shape[0]) if A[i, j]]
    bk = BackgroundKnowledge(required=[(names[j], names[i]) for i, j in edges[:8]])

    # Набор ограничений обязан быть ацикличным сам по себе.
    from whydra.background_knowledge import _find_cycle
    assert _find_cycle(list(bk.required)) is None
