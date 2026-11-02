"""
Верификация реализаций PC и FCI из whydra относительно оригиналов causal-learn.

Идея та же, что в ``playground/sber_synth_data/verify_vs_causallearn.py``
(ветки ``whydra``/``sber``), но без хардкода путей к внешним датасетам:
данные генерируются линейной SEM прямо в тесте, поэтому прогон
самодостаточен и детерминирован.

Проверяются три свойства:

1. **Соответствие эталону.** Каждая реализация whydra сравнивается с
   последовательным оригиналом из causal-learn (``cl_pc`` / ``cl_fci``)
   на одних и тех же данных при одинаковых параметрах. Сравнение идёт
   по endpoint-матрице, то есть учитывает и скелет, и ориентацию.

2. **Инвариантность к параллелизму.** Для реализаций с ``n_jobs``
   результат при ``n_jobs=1`` и ``n_jobs=4`` обязан совпадать бит-в-бит:
   распараллеливание не имеет права менять ответ.

3. **Согласие скелетов.** Отдельная, более мягкая проверка: даже там,
   где ориентация расходится, множество рёбер должно совпадать.

Известные расхождения перечислены в ``KNOWN_DEVIATIONS`` и помечаются
``xfail`` — они документируются, а не замалчиваются. Если расхождение
будет устранено, тест перейдёт в XPASS, и таблицу нужно будет обновить.
"""

import io
import os
import sys
import contextlib

import numpy as np
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

pytest.importorskip("causallearn")

from causallearn.search.ConstraintBased.PC import pc as cl_pc
from causallearn.search.ConstraintBased.FCI import fci as cl_fci

from whydra.algorithms.graph_core import CausalGraph, CIT
from whydra.algorithms.wrappers import (
    ParallelPCStable,
    StandaloneCFCIStable,
    StandaloneCFCIStableParallelCached,
    StandaloneFCIPlusStable,
    StandaloneFCIPlusStableParallelCached,
    StandaloneFCIStable,
    StandaloneFCIStableCached,
    StandaloneFCIStableParallelCached,
    StandaloneGFCIStable,
    StandalonePCStable,
    StandaloneRFCIStable,
)
from whydra.algorithms.standalone import (
    standalone_pc_stable,
    parallel_pc_stable,
    pc_algo,
    standalone_fci_stable,
    fci_algo_v2,
)
from whydra.evaluation.metrics import result_to_endpoint_matrix

from .conftest import (
    generate_linear_sem_data,
    get_chain_graph_3nodes,
    get_collider_graph_3nodes,
    get_v_structure_graph_4nodes,
)


# ============================================================
#  Параметры прогона
# ============================================================

ALPHA = 0.05
INDEP_TEST = "fisherz"
N_SAMPLES = 2000
SEED = 42

#: Значения ``n_jobs``, на которых проверяется инвариантность к параллелизму.
N_JOBS_VALUES = [1, 4]


# ============================================================
#  Тестовые графы
# ============================================================

def _diamond_graph_5nodes():
    """X1 -> X2, X1 -> X3, X2 -> X4, X3 -> X4, X4 -> X5."""
    W = np.zeros((5, 5))
    W[0, 1] = 0.7
    W[0, 2] = 0.7
    W[1, 3] = 0.7
    W[2, 3] = 0.7
    W[3, 4] = 0.7
    return W


def _dag_6nodes():
    """Две ветви из общего корня, сходящиеся в X6."""
    W = np.zeros((6, 6))
    for src, dst in [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)]:
        W[src, dst] = 0.6
    return W


def _dag_8nodes():
    """Граф с несколькими v-структурами и пересекающимися путями."""
    W = np.zeros((8, 8))
    for src, dst in [(0, 2), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (4, 7), (6, 7)]:
        W[src, dst] = 0.7
    return W


GRAPHS = {
    "chain_3": get_chain_graph_3nodes,
    "collider_3": get_collider_graph_3nodes,
    "v_structure_4": get_v_structure_graph_4nodes,
    "diamond_5": _diamond_graph_5nodes,
    "dag_6": _dag_6nodes,
    "dag_8": _dag_8nodes,
}


# ============================================================
#  Реализации
# ============================================================

def _make_cg(data, node_names):
    """Инициализирует CausalGraph с настроенным тестом независимости."""
    cg = CausalGraph(data.shape[1], list(node_names))
    cg.set_ind_test(CIT(data, method=INDEP_TEST))
    return cg


def _run_cl_pc(data, node_names, n_jobs):
    """Оригинальный PC из causal-learn (последовательный, stable)."""
    result = cl_pc(
        data, alpha=ALPHA, indep_test=INDEP_TEST, stable=True,
        uc_rule=0, uc_priority=2, verbose=False, show_progress=False,
        node_names=list(node_names),
    )
    return result.G


def _run_pc_sequential(data, node_names, n_jobs):
    """whydra: последовательный PC-stable."""
    return standalone_pc_stable.pc_stable(
        _make_cg(data, node_names), alpha=ALPHA, stable=True,
        uc_rule=0, uc_priority=2, verbose=False, show_progress=False,
    )


def _run_pc_parallel(data, node_names, n_jobs):
    """whydra: распараллеленный PC-stable."""
    return parallel_pc_stable.pc_stable(
        _make_cg(data, node_names), alpha=ALPHA, stable=True,
        uc_rule=0, uc_priority=2, verbose=False, show_progress=False,
        n_jobs=n_jobs,
    )


def _run_pc_algo(data, node_names, n_jobs):
    """whydra: PC-stable, используемый обёрткой StandalonePCStable."""
    return pc_algo.pc_stable(
        _make_cg(data, node_names), alpha=ALPHA, stable=True,
        uc_rule=0, uc_priority=2, verbose=False, show_progress=False,
        n_jobs=n_jobs,
    )


def _run_cl_fci(data, node_names, n_jobs):
    """Оригинальный FCI из causal-learn (последовательный)."""
    graph, _edges = cl_fci(
        data, independence_test_method=INDEP_TEST, alpha=ALPHA,
        verbose=False, show_progress=False,
    )
    return graph


def _run_fci_sequential(data, node_names, n_jobs):
    """whydra: последовательный FCI-stable."""
    return standalone_fci_stable.fci_stable(
        data, alpha=ALPHA, indep_test=INDEP_TEST, verbose=False,
    )


def _run_fci_v2(data, node_names, n_jobs):
    """whydra: FCI-stable v2, используемый обёрткой StandaloneFCIStable."""
    return fci_algo_v2.fci_stable_v2(
        _make_cg(data, node_names), data, alpha=ALPHA,
        indep_test=INDEP_TEST, n_jobs=n_jobs, verbose=False,
    )


def _run_pc_wrapper(data, node_names, n_jobs):
    """whydra: публичная обёртка StandalonePCStable (то, что видят конфиги Hydra)."""
    algo = StandalonePCStable(alpha=ALPHA, indep_test=INDEP_TEST, n_jobs=n_jobs)
    return algo.run(data)


def _run_fci_wrapper(data, node_names, n_jobs):
    """whydra: публичная обёртка StandaloneFCIStable (то, что видят конфиги Hydra)."""
    algo = StandaloneFCIStable(alpha=ALPHA, indep_test=INDEP_TEST, n_jobs=n_jobs)
    return algo.run(data)


#: Эталонные последовательные реализации из causal-learn.
BASELINES = {
    "PC": _run_cl_pc,
    "FCI": _run_cl_fci,
}

def _wrapper_runner(cls):
    """Раннер публичной обёртки: то, что реально видит пользователь API."""

    def run(data, node_names, n_jobs):
        return cls(alpha=ALPHA, indep_test=INDEP_TEST, n_jobs=n_jobs,
                   show_progress=False).run(data)

    run.__doc__ = f"whydra: публичная обёртка {cls.__name__}."
    return run


#: Реализации whydra: ключ -> (функция, поддерживает ли n_jobs).
#:
#: Сюда включены ВСЕ публичные обёртки, а не только PC и FCI. Пробел в покрытии
#: обходится дорого: GFCI не сверялся ни с чем, и рассогласование кодировки в
#: нём не поймал ни один тест — нашлось только ручной проверкой.
#: RAI здесь нет: это другой алгоритм (другой представитель класса эквивалентности
#: на тех же данных), его эталон — референс IntelLabs, см. test_vs_reference_rai.py.
PC_IMPLS = {
    "pc_sequential": (_run_pc_sequential, False),
    "pc_parallel": (_run_pc_parallel, True),
    "pc_algo": (_run_pc_algo, True),
    "pc_wrapper": (_run_pc_wrapper, True),
    "pc_parallel_wrapper": (_wrapper_runner(ParallelPCStable), True),
}

FCI_IMPLS = {
    "fci_sequential": (_run_fci_sequential, False),
    "fci_v2": (_run_fci_v2, True),
    "fci_wrapper": (_run_fci_wrapper, True),
    "fci_cached": (_wrapper_runner(StandaloneFCIStableCached), True),
    "fci_parallel_cached": (_wrapper_runner(StandaloneFCIStableParallelCached), True),
    "cfci": (_wrapper_runner(StandaloneCFCIStable), True),
    "cfci_parallel_cached": (_wrapper_runner(StandaloneCFCIStableParallelCached), True),
    "rfci": (_wrapper_runner(StandaloneRFCIStable), True),
    "gfci": (_wrapper_runner(StandaloneGFCIStable), True),
    "fci_plus": (_wrapper_runner(StandaloneFCIPlusStable), True),
    "fci_plus_parallel_cached": (_wrapper_runner(StandaloneFCIPlusStableParallelCached), True),
}

#: Какую реализацию оборачивает каждая публичная обёртка.
#:
#: Отдельная проверка «обёртка == её реализация» ловит дефекты конвертации
#: результата, невидимые при тестировании реализаций напрямую: именно так
#: обнаружилось, что StandaloneFCIStable отображала PAG-код CIRCLE в TAIL.
WRAPPED_IMPL = {
    "pc_wrapper": "pc_algo",
    "pc_parallel_wrapper": "pc_parallel",
    "fci_wrapper": "fci_v2",
}

IMPLS = {**PC_IMPLS, **FCI_IMPLS}

#: К какому эталону относится каждая реализация.
FAMILY = {key: "PC" for key in PC_IMPLS}
FAMILY.update({key: "FCI" for key in FCI_IMPLS})


# ============================================================
#  Известные расхождения
# ============================================================

#: Расхождения с эталоном causal-learn: (ключ реализации, ключ графа) -> причина.
#:
#: Пусто. Здесь стояли два расхождения FCI v1 на ``dag_8`` — «правила
#: ориентации применяются не полностью, рёбра остаются с CIRCLE-концами».
#: Причина найдена: в ``udag2pag`` (fci_algo.py и standalone_fci_stable.py)
#: ``return pag`` стоял ВНУТРИ цикла ``while not array_equal(old_pag1, pag)``,
#: поэтому цикл правил выполнялся ровно один проход вместо схождения к
#: неподвижной точке. После переноса ``return`` на уровень функции v1
#: совпадает с causal-learn на всём наборе.
_GFCI_IS_HYBRID = (
    "GFCI — гибрид: стартует со score-based CPDAG (GES) и доориентирует его "
    "правилами FCI. Эталон сравнения — чисто constraint-based FCI из "
    "causal-learn, поэтому ориентация на diamond_5 расходится при совпадающем "
    "скелете. Скелет как раз обязан совпадать и совпадает 12/12."
)

KNOWN_DEVIATIONS = {
    ("gfci", "diamond_5"): _GFCI_IS_HYBRID,
}

#: Расхождения в скелете (множестве рёбер). Пусто: на текущем наборе графов
#: все реализации находят те же рёбра, что и causal-learn, расходясь только
#: в ориентации.
KNOWN_SKELETON_DEVIATIONS = {}

#: Расхождения между последовательной и параллельной реализациями одного
#: алгоритма: (ключ параллельной реализации, ключ графа) -> причина.
#:
#: Пусто. Здесь стояли два расхождения v2 против fci_sequential на ``dag_8``,
#: вызванные тем же дефектом обрыва цикла правил в v1 (см. KNOWN_DEVIATIONS).
KNOWN_SEQ_VS_PARALLEL_DEVIATIONS = {}


# ============================================================
#  Инфраструктура прогона
# ============================================================

_RESULT_CACHE = {}
_SUMMARY = []


def _node_names(n_vars):
    return [f"X{i + 1}" for i in range(n_vars)]


@contextlib.contextmanager
def _silenced():
    """Гасит отладочную печать реализаций, чтобы вывод теста оставался читаемым."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def get_data(graph_key):
    """Данные линейной SEM для указанного графа."""
    W = GRAPHS[graph_key]()
    return generate_linear_sem_data(W, n_samples=N_SAMPLES, seed=SEED)


def get_endpoint_matrix(impl_key, graph_key, n_jobs):
    """
    Запускает реализацию и возвращает endpoint-матрицу.

    Результаты кэшируются: одна и та же тройка (реализация, граф, n_jobs)
    переиспользуется разными тестами, а прогоны FCI недёшевы.

    Кодировка endpoint-матрицы (causallearn): TAIL=-1, NULL=0, ARROW=1, CIRCLE=2.
    """
    cache_key = (impl_key, graph_key, n_jobs)
    if cache_key in _RESULT_CACHE:
        return _RESULT_CACHE[cache_key]

    if impl_key in BASELINES:
        runner = BASELINES[impl_key]
    else:
        runner, _supports_n_jobs = IMPLS[impl_key]

    data = get_data(graph_key)
    names = _node_names(data.shape[1])

    with _silenced():
        result = runner(data, names, n_jobs)

    matrix = result_to_endpoint_matrix(result, names)
    _RESULT_CACHE[cache_key] = matrix
    return matrix


def compare(matrix_a, matrix_b):
    """
    Сравнивает две endpoint-матрицы.

    Returns
    -------
    dict
        ``identical`` — полное совпадение (скелет и ориентация);
        ``skeleton_identical`` — совпадение множества рёбер;
        ``n_diff_cells`` — число различающихся ячеек;
        ``diffs`` — список ``(i, j, значение_a, значение_b)``.
    """
    identical = np.array_equal(matrix_a, matrix_b)
    skeleton_identical = np.array_equal(matrix_a != 0, matrix_b != 0)
    diff_positions = np.argwhere(matrix_a != matrix_b)
    diffs = [
        (int(i), int(j), int(matrix_a[i, j]), int(matrix_b[i, j]))
        for i, j in diff_positions
    ]
    return {
        "identical": identical,
        "skeleton_identical": skeleton_identical,
        "n_diff_cells": len(diffs),
        "diffs": diffs,
    }


def format_diffs(diffs, node_names, limit=10, label_a="causal-learn", label_b="whydra"):
    """Человекочитаемое описание расхождений."""
    label = {-1: "TAIL", 0: "NULL", 1: "ARROW", 2: "CIRCLE"}
    lines = []
    for i, j, value_ref, value_test in diffs[:limit]:
        lines.append(
            f"    конец у {node_names[j]} на ребре {node_names[i]}-{node_names[j]}: "
            f"{label_a}={label.get(value_ref, value_ref)}, "
            f"{label_b}={label.get(value_test, value_test)}"
        )
    if len(diffs) > limit:
        lines.append(f"    ... ещё {len(diffs) - limit}")
    return "\n".join(lines)


def _mark_known(request, table, impl_key, graph_key):
    """Помечает тест как xfail, если расхождение уже задокументировано."""
    reason = table.get((impl_key, graph_key))
    if reason is not None:
        request.node.add_marker(
            pytest.mark.xfail(reason=reason, strict=False)
        )


# ============================================================
#  Тесты: соответствие эталону causal-learn
# ============================================================

def _impl_params(impls):
    """Разворачивает реализации в пары (ключ, n_jobs) с учётом поддержки n_jobs."""
    params = []
    for key, (_runner, supports_n_jobs) in impls.items():
        for n_jobs in (N_JOBS_VALUES if supports_n_jobs else [1]):
            params.append(pytest.param(key, n_jobs, id=f"{key}-n_jobs{n_jobs}"))
    return params


class TestPCvsCausalLearn:
    """PC из whydra против оригинального PC из causal-learn."""

    @pytest.mark.parametrize("impl_key,n_jobs", _impl_params(PC_IMPLS))
    @pytest.mark.parametrize("graph_key", list(GRAPHS))
    def test_matrix_identical(self, request, impl_key, n_jobs, graph_key):
        _mark_known(request, KNOWN_DEVIATIONS, impl_key, graph_key)

        reference = get_endpoint_matrix("PC", graph_key, 1)
        actual = get_endpoint_matrix(impl_key, graph_key, n_jobs)
        result = compare(reference, actual)

        _SUMMARY.append((impl_key, n_jobs, graph_key, result))

        names = _node_names(reference.shape[0])
        assert result["identical"], (
            f"{impl_key} (n_jobs={n_jobs}) на {graph_key} расходится с causal-learn "
            f"в {result['n_diff_cells']} ячейках:\n"
            f"{format_diffs(result['diffs'], names)}"
        )

    @pytest.mark.parametrize("impl_key,n_jobs", _impl_params(PC_IMPLS))
    @pytest.mark.parametrize("graph_key", list(GRAPHS))
    def test_skeleton_identical(self, request, impl_key, n_jobs, graph_key):
        _mark_known(request, KNOWN_SKELETON_DEVIATIONS, impl_key, graph_key)

        reference = get_endpoint_matrix("PC", graph_key, 1)
        actual = get_endpoint_matrix(impl_key, graph_key, n_jobs)
        result = compare(reference, actual)

        assert result["skeleton_identical"], (
            f"{impl_key} (n_jobs={n_jobs}) на {graph_key}: скелет не совпадает с causal-learn"
        )


class TestFCIvsCausalLearn:
    """FCI из whydra против оригинального FCI из causal-learn."""

    @pytest.mark.parametrize("impl_key,n_jobs", _impl_params(FCI_IMPLS))
    @pytest.mark.parametrize("graph_key", list(GRAPHS))
    def test_matrix_identical(self, request, impl_key, n_jobs, graph_key):
        _mark_known(request, KNOWN_DEVIATIONS, impl_key, graph_key)

        reference = get_endpoint_matrix("FCI", graph_key, 1)
        actual = get_endpoint_matrix(impl_key, graph_key, n_jobs)
        result = compare(reference, actual)

        _SUMMARY.append((impl_key, n_jobs, graph_key, result))

        names = _node_names(reference.shape[0])
        assert result["identical"], (
            f"{impl_key} (n_jobs={n_jobs}) на {graph_key} расходится с causal-learn "
            f"в {result['n_diff_cells']} ячейках:\n"
            f"{format_diffs(result['diffs'], names)}"
        )

    @pytest.mark.parametrize("impl_key,n_jobs", _impl_params(FCI_IMPLS))
    @pytest.mark.parametrize("graph_key", list(GRAPHS))
    def test_skeleton_identical(self, request, impl_key, n_jobs, graph_key):
        _mark_known(request, KNOWN_SKELETON_DEVIATIONS, impl_key, graph_key)

        reference = get_endpoint_matrix("FCI", graph_key, 1)
        actual = get_endpoint_matrix(impl_key, graph_key, n_jobs)
        result = compare(reference, actual)

        assert result["skeleton_identical"], (
            f"{impl_key} (n_jobs={n_jobs}) на {graph_key}: скелет не совпадает с causal-learn.\n"
            f"Рёбра только у causal-learn: {sorted(map(tuple, np.argwhere((reference != 0) & (actual == 0)).tolist()))}\n"
            f"Рёбра только у whydra: {sorted(map(tuple, np.argwhere((reference == 0) & (actual != 0)).tolist()))}"
        )


# ============================================================
#  Тесты: инвариантность к параллелизму
# ============================================================

_PARALLEL_IMPLS = [key for key, (_runner, supports) in IMPLS.items() if supports]


class TestParallelDeterminism:
    """
    Распараллеливание не должно менять результат.

    Проверка сильнее сравнения с эталоном: даже если реализация в чём-то
    расходится с causal-learn, расхождение обязано быть одинаковым при
    любом числе процессов.
    """

    @pytest.mark.parametrize("impl_key", _PARALLEL_IMPLS)
    @pytest.mark.parametrize("graph_key", list(GRAPHS))
    def test_njobs_invariant(self, impl_key, graph_key):
        baseline = get_endpoint_matrix(impl_key, graph_key, 1)

        for n_jobs in N_JOBS_VALUES[1:]:
            actual = get_endpoint_matrix(impl_key, graph_key, n_jobs)
            result = compare(baseline, actual)
            names = _node_names(baseline.shape[0])
            assert result["identical"], (
                f"{impl_key} на {graph_key}: результат при n_jobs={n_jobs} отличается "
                f"от n_jobs=1 в {result['n_diff_cells']} ячейках:\n"
                f"{format_diffs(result['diffs'], names, label_a='n_jobs=1', label_b=f'n_jobs={n_jobs}')}"
            )


class TestSequentialVsParallelImplementations:
    """
    Последовательная и распараллеленная реализации одного алгоритма
    должны давать одинаковый результат на одних данных.
    """

    @pytest.mark.parametrize("graph_key", list(GRAPHS))
    @pytest.mark.parametrize("parallel_key", ["pc_parallel", "pc_algo", "pc_wrapper"])
    def test_pc_sequential_matches_parallel(self, graph_key, parallel_key):
        sequential = get_endpoint_matrix("pc_sequential", graph_key, 1)
        parallel = get_endpoint_matrix(parallel_key, graph_key, max(N_JOBS_VALUES))
        result = compare(sequential, parallel)
        names = _node_names(sequential.shape[0])
        assert result["identical"], (
            f"pc_sequential и {parallel_key} расходятся на {graph_key} "
            f"в {result['n_diff_cells']} ячейках:\n"
            f"{format_diffs(result['diffs'], names, label_a='pc_sequential', label_b=parallel_key)}"
        )

    @pytest.mark.parametrize("graph_key", list(GRAPHS))
    @pytest.mark.parametrize("parallel_key", ["fci_v2", "fci_wrapper"])
    def test_fci_sequential_matches_parallel(self, request, graph_key, parallel_key):
        _mark_known(request, KNOWN_SEQ_VS_PARALLEL_DEVIATIONS, parallel_key, graph_key)

        sequential = get_endpoint_matrix("fci_sequential", graph_key, 1)
        parallel = get_endpoint_matrix(parallel_key, graph_key, max(N_JOBS_VALUES))
        result = compare(sequential, parallel)
        names = _node_names(sequential.shape[0])
        assert result["identical"], (
            f"fci_sequential и {parallel_key} расходятся на {graph_key} "
            f"в {result['n_diff_cells']} ячейках:\n"
            f"{format_diffs(result['diffs'], names, label_a='fci_sequential', label_b=parallel_key)}"
        )


class TestWrappersMatchImplementations:
    """
    Публичная обёртка обязана возвращать ровно то, что вернула
    обёрнутая реализация.

    Между ними лежит конвертация результата в ``GeneralGraph`` — шаг,
    который тестирование реализаций напрямую не покрывает. Ошибка
    именно на этом шаге (PAG-код CIRCLE отображался в TAIL) однажды
    сводила на нет корректность самого алгоритма.
    """

    @pytest.mark.parametrize("graph_key", list(GRAPHS))
    @pytest.mark.parametrize("wrapper_key", list(WRAPPED_IMPL))
    @pytest.mark.parametrize("n_jobs", N_JOBS_VALUES)
    def test_wrapper_matches_wrapped(self, wrapper_key, graph_key, n_jobs):
        impl_key = WRAPPED_IMPL[wrapper_key]

        wrapped = get_endpoint_matrix(impl_key, graph_key, n_jobs)
        wrapper = get_endpoint_matrix(wrapper_key, graph_key, n_jobs)
        result = compare(wrapped, wrapper)

        names = _node_names(wrapped.shape[0])
        assert result["identical"], (
            f"{wrapper_key} (n_jobs={n_jobs}) на {graph_key} искажает результат "
            f"{impl_key} в {result['n_diff_cells']} ячейках:\n"
            f"{format_diffs(result['diffs'], names, label_a=impl_key, label_b=wrapper_key)}"
        )

    @pytest.mark.parametrize("graph_key", list(GRAPHS))
    @pytest.mark.parametrize("wrapper_key", list(WRAPPED_IMPL))
    def test_wrapper_preserves_circle_endpoints(self, wrapper_key, graph_key):
        """
        Обёртка не имеет права терять CIRCLE-концы.

        Проверка узкая и намеренно дублирует часть предыдущего теста:
        CIRCLE — единственный тип конца, который легко «схлопнуть» в TAIL
        при конвертации, и такая потеря молча меняет смысл результата
        с «ориентация неизвестна» на «стрелки точно нет».
        """
        impl_key = WRAPPED_IMPL[wrapper_key]

        wrapped = get_endpoint_matrix(impl_key, graph_key, 1)
        wrapper = get_endpoint_matrix(wrapper_key, graph_key, 1)

        circle = 2  # Endpoint.CIRCLE
        expected = int((wrapped == circle).sum())
        actual = int((wrapper == circle).sum())

        assert actual == expected, (
            f"{wrapper_key} на {graph_key}: у {impl_key} CIRCLE-концов {expected}, "
            f"после конвертации в обёртке — {actual}"
        )


# ============================================================
#  Сводка
# ============================================================

def test_zz_summary_report(capsys):
    """
    Печатает сводную таблицу сравнения.

    Имя с префиксом ``zz`` — чтобы тест шёл последним и видел уже
    заполненный ``_SUMMARY``. Собственных утверждений не содержит:
    расхождения проверяются тестами выше, здесь только отчёт.
    Смотреть через ``pytest -s -k summary_report``.
    """
    if not _SUMMARY:
        pytest.skip("сводка пуста: тесты сравнения не запускались")

    with capsys.disabled():
        print("\n" + "=" * 78)
        print("СВЕРКА С CAUSAL-LEARN")
        print("=" * 78)
        print(f"{'реализация':<16} {'n_jobs':>6}  {'граф':<14} {'матрица':<9} {'скелет':<8} {'диффы':>6}")
        print("-" * 78)

        for impl_key, n_jobs, graph_key, result in _SUMMARY:
            print(
                f"{impl_key:<16} {n_jobs:>6}  {graph_key:<14} "
                f"{'совпала' if result['identical'] else 'РАЗНАЯ':<9} "
                f"{'совпал' if result['skeleton_identical'] else 'РАЗНЫЙ':<8} "
                f"{result['n_diff_cells']:>6}"
            )

        print("-" * 78)
        for impl_key in IMPLS:
            rows = [row for row in _SUMMARY if row[0] == impl_key]
            if not rows:
                continue
            n_identical = sum(1 for row in rows if row[3]["identical"])
            n_skeleton = sum(1 for row in rows if row[3]["skeleton_identical"])
            print(
                f"  {impl_key:<16} матрица {n_identical}/{len(rows)}, "
                f"скелет {n_skeleton}/{len(rows)}"
            )
        print("=" * 78)
