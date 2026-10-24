"""
Сверка PC и FCI из whydra с оригиналами causal-learn на кейсах bnlearn.

В отличие от ``test_vs_causallearn.py``, где данные синтетические и
непрерывные, здесь берутся реальные дискретные выборки из
``BenchmarkCaseRepository`` и тест независимости ``chisq``.

**Расхождения здесь ожидаемы и намеренно НЕ замаскированы.** Таблицы
известных отклонений, в отличие от синтетического теста, нет: пока не
решено, что с расхождениями делать, они должны быть видны как падения,
а не спрятаны в xfail.

Поэтому файл не участвует в обычном прогоне: без флага ``--run-bnlearn``
все тесты скипаются, и ``pytest`` в CI остаётся зелёным.

Запуск сверки::

    pytest tests/test_vs_causallearn_on_bnlearn.py --run-bnlearn

Либо напрямую, с таблицей расхождений и без pytest::

    python tests/test_vs_causallearn_on_bnlearn.py
    python tests/test_vs_causallearn_on_bnlearn.py --cases asia sachs --max-nodes 40

По замерам на 2026-08-21 скелет совпадает с causal-learn на всех
проверенных кейсах, расходится только ориентация, и расхождения растут
с размером графа — начиная примерно с 11 узлов.
"""

import io
import os
import sys
import time
import argparse
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

from whydra.data.case_repository import BenchmarkCaseRepository
from whydra.algorithms.graph_core import CausalGraph, CIT
from whydra.algorithms.wrappers import StandalonePCStable, StandaloneFCIStable
from whydra.algorithms.standalone import (
    standalone_pc_stable,
    parallel_pc_stable,
    pc_algo,
    fci_algo_v2,
)
from whydra.evaluation.metrics import result_to_endpoint_matrix


# ============================================================
#  Параметры
# ============================================================

ALPHA = 0.05

#: Данные bnlearn дискретные, поэтому тест независимости — хи-квадрат.
INDEP_TEST = "chisq"

BENCHMARK = "bnlearn"

N_JOBS = 4

#: Граф крупнее этого в прогон не берётся: время constraint-based
#: алгоритмов растёт быстро, а barley (48), hailfinder (56), hepar2 (70),
#: win95pts (76) и особенно andes (223) превращают сверку в многочасовую.
DEFAULT_MAX_NODES = 40

#: Ограничение для проверки «обёртка доносит indep_test»: она, в отличие
#: от остальных, гоняет алгоритмы ещё и с fisherz. На дискретных данных
#: это заведомо неподходящий тест независимости — он находит куда более
#: плотный граф, и constraint-based алгоритмы взрываются комбинаторно
#: (water, 32 узла: chisq 0.7-2.6 с против >45 с на fisherz). Смысл
#: проверки от размера графа не зависит, поэтому берутся только мелкие.
INDEP_TEST_COMPARISON_MAX_NODES = 11


# ============================================================
#  Реализации
# ============================================================

def _make_cg(data, node_names, indep_test):
    cg = CausalGraph(data.shape[1], list(node_names))
    cg.set_ind_test(CIT(data, method=indep_test))
    return cg


def _run_cl_pc(data, node_names, indep_test):
    result = cl_pc(
        data, alpha=ALPHA, indep_test=indep_test, stable=True,
        uc_rule=0, uc_priority=2, verbose=False, show_progress=False,
        node_names=list(node_names),
    )
    return result.G


def _run_cl_fci(data, node_names, indep_test):
    graph, _edges = cl_fci(
        data, independence_test_method=indep_test, alpha=ALPHA,
        verbose=False, show_progress=False,
    )
    return graph


def _run_pc_sequential(data, node_names, indep_test):
    return standalone_pc_stable.pc_stable(
        _make_cg(data, node_names, indep_test), alpha=ALPHA, stable=True,
        uc_rule=0, uc_priority=2, verbose=False, show_progress=False,
    )


def _run_pc_parallel(data, node_names, indep_test):
    return parallel_pc_stable.pc_stable(
        _make_cg(data, node_names, indep_test), alpha=ALPHA, stable=True,
        uc_rule=0, uc_priority=2, verbose=False, show_progress=False,
        n_jobs=N_JOBS,
    )


def _run_pc_algo(data, node_names, indep_test):
    return pc_algo.pc_stable(
        _make_cg(data, node_names, indep_test), alpha=ALPHA, stable=True,
        uc_rule=0, uc_priority=2, verbose=False, show_progress=False,
        n_jobs=N_JOBS,
    )


def _run_pc_wrapper(data, node_names, indep_test):
    return StandalonePCStable(alpha=ALPHA, indep_test=indep_test, n_jobs=N_JOBS).run(data)


def _run_fci_v2(data, node_names, indep_test):
    return fci_algo_v2.fci_stable_v2(
        _make_cg(data, node_names, indep_test), data, alpha=ALPHA,
        indep_test=indep_test, n_jobs=N_JOBS, verbose=False,
    )


def _run_fci_wrapper(data, node_names, indep_test):
    return StandaloneFCIStable(alpha=ALPHA, indep_test=indep_test, n_jobs=N_JOBS).run(data)


PC_IMPLS = {
    "pc_sequential": _run_pc_sequential,
    "pc_parallel": _run_pc_parallel,
    "pc_algo": _run_pc_algo,
    "pc_wrapper": _run_pc_wrapper,
}

FCI_IMPLS = {
    "fci_v2": _run_fci_v2,
    "fci_wrapper": _run_fci_wrapper,
}

IMPLS = {**PC_IMPLS, **FCI_IMPLS}

BASELINES = {
    "PC": _run_cl_pc,
    "FCI": _run_cl_fci,
}

FAMILY = {key: "PC" for key in PC_IMPLS}
FAMILY.update({key: "FCI" for key in FCI_IMPLS})


# ============================================================
#  Данные
# ============================================================

_repo = None
_DATA_CACHE = {}
_RESULT_CACHE = {}


def get_repository():
    global _repo
    if _repo is None:
        _repo = BenchmarkCaseRepository()
    return _repo


def list_bnlearn_cases():
    """Имена всех доступных кейсов bnlearn."""
    repo = get_repository()
    return sorted(
        case.split("/")[-1]
        for case in repo.list_available_cases()
        if case.split("/")[0] == BENCHMARK
    )


def get_case_data(case_name):
    """Матрица наблюдений кейса, приведённая к float."""
    if case_name in _DATA_CACHE:
        return _DATA_CACHE[case_name]

    repo = get_repository()
    datasets = repo.get_case_datasets(case_name=case_name, benchmark_name=BENCHMARK)
    raw = next(iter(datasets.values()))
    data = np.asarray(raw.values if hasattr(raw, "values") else raw, dtype=float)

    _DATA_CACHE[case_name] = data
    return data


def select_cases(max_nodes=DEFAULT_MAX_NODES, only=None):
    """
    Кейсы, укладывающиеся в ограничение по числу узлов.

    Returns
    -------
    list[tuple[str, int]]
        Пары ``(имя кейса, число узлов)``, отсортированные по размеру.
    """
    names = list_bnlearn_cases()
    if only:
        missing = sorted(set(only) - set(names))
        if missing:
            raise ValueError(f"нет таких кейсов bnlearn: {missing}; доступны: {names}")
        names = [name for name in names if name in only]

    selected = []
    for name in names:
        n_nodes = get_case_data(name).shape[1]
        if max_nodes is None or n_nodes <= max_nodes:
            selected.append((name, n_nodes))

    return sorted(selected, key=lambda pair: pair[1])


# ============================================================
#  Прогон и сравнение
# ============================================================

@contextlib.contextmanager
def _silenced():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def _node_names(n_vars):
    return [f"X{i + 1}" for i in range(n_vars)]


def get_endpoint_matrix(impl_key, case_name, indep_test=INDEP_TEST):
    """
    Запускает реализацию на кейсе и возвращает endpoint-матрицу.

    Кодировка (causallearn): TAIL=-1, NULL=0, ARROW=1, CIRCLE=2.
    Результаты кэшируются — прогоны на реальных данных дорогие.
    """
    cache_key = (impl_key, case_name, indep_test)
    if cache_key in _RESULT_CACHE:
        return _RESULT_CACHE[cache_key]

    runner = BASELINES[impl_key] if impl_key in BASELINES else IMPLS[impl_key]

    data = get_case_data(case_name)
    names = _node_names(data.shape[1])

    with _silenced():
        result = runner(data, names, indep_test)

    matrix = result_to_endpoint_matrix(result, names)
    _RESULT_CACHE[cache_key] = matrix
    return matrix


def compare(matrix_a, matrix_b):
    """Сравнивает две endpoint-матрицы: скелет, ориентация, состав рёбер."""
    edges_a = matrix_a != 0
    edges_b = matrix_b != 0
    return {
        "identical": np.array_equal(matrix_a, matrix_b),
        "skeleton_identical": np.array_equal(edges_a, edges_b),
        "n_diff_cells": int((matrix_a != matrix_b).sum()),
        "only_in_a": int((edges_a & ~edges_b).sum()) // 2,
        "only_in_b": int((~edges_a & edges_b).sum()) // 2,
        "n_orientation_diffs": int((edges_a & edges_b & (matrix_a != matrix_b)).sum()),
    }


def format_summary(result, label_a="causal-learn", label_b="whydra"):
    """Однострочное описание расхождения."""
    return (
        f"ячеек={result['n_diff_cells']}, "
        f"ориентация={result['n_orientation_diffs']}, "
        f"рёбер только у {label_a}={result['only_in_a']}, "
        f"только у {label_b}={result['only_in_b']}"
    )


# ============================================================
#  Тесты
# ============================================================

def _requires_flag(request):
    if not request.config.getoption("--run-bnlearn"):
        pytest.skip("нужен флаг --run-bnlearn (сверка на реальных данных, долго)")


def _case_params():
    """
    Параметризация по кейсам.

    Список строится во время сборки тестов, поэтому репозиторий читается
    даже без флага. Если данные недоступны, параметризация становится
    пустой, а не роняет сборку.
    """
    try:
        return [pytest.param(name, id=f"{name}-{n}n") for name, n in select_cases()]
    except Exception:
        return []


CASE_PARAMS = _case_params()


class TestPCvsCausalLearnOnBnlearn:
    """PC из whydra против оригинального PC из causal-learn на bnlearn."""

    @pytest.mark.parametrize("impl_key", list(PC_IMPLS))
    @pytest.mark.parametrize("case_name", CASE_PARAMS)
    def test_matrix_identical(self, request, impl_key, case_name):
        _requires_flag(request)

        reference = get_endpoint_matrix("PC", case_name)
        actual = get_endpoint_matrix(impl_key, case_name)
        result = compare(reference, actual)

        assert result["identical"], (
            f"{impl_key} на bnlearn/{case_name} расходится с causal-learn: "
            f"{format_summary(result)}"
        )

    @pytest.mark.parametrize("impl_key", list(PC_IMPLS))
    @pytest.mark.parametrize("case_name", CASE_PARAMS)
    def test_skeleton_identical(self, request, impl_key, case_name):
        _requires_flag(request)

        reference = get_endpoint_matrix("PC", case_name)
        actual = get_endpoint_matrix(impl_key, case_name)
        result = compare(reference, actual)

        assert result["skeleton_identical"], (
            f"{impl_key} на bnlearn/{case_name}: скелет не совпадает с causal-learn "
            f"(рёбер только у causal-learn {result['only_in_a']}, "
            f"только у whydra {result['only_in_b']})"
        )


class TestFCIvsCausalLearnOnBnlearn:
    """FCI из whydra против оригинального FCI из causal-learn на bnlearn."""

    @pytest.mark.parametrize("impl_key", list(FCI_IMPLS))
    @pytest.mark.parametrize("case_name", CASE_PARAMS)
    def test_matrix_identical(self, request, impl_key, case_name):
        _requires_flag(request)

        reference = get_endpoint_matrix("FCI", case_name)
        actual = get_endpoint_matrix(impl_key, case_name)
        result = compare(reference, actual)

        assert result["identical"], (
            f"{impl_key} на bnlearn/{case_name} расходится с causal-learn: "
            f"{format_summary(result)}"
        )

    @pytest.mark.parametrize("impl_key", list(FCI_IMPLS))
    @pytest.mark.parametrize("case_name", CASE_PARAMS)
    def test_skeleton_identical(self, request, impl_key, case_name):
        _requires_flag(request)

        reference = get_endpoint_matrix("FCI", case_name)
        actual = get_endpoint_matrix(impl_key, case_name)
        result = compare(reference, actual)

        assert result["skeleton_identical"], (
            f"{impl_key} на bnlearn/{case_name}: скелет не совпадает с causal-learn "
            f"(рёбер только у causal-learn {result['only_in_a']}, "
            f"только у whydra {result['only_in_b']})"
        )


class TestWrappersOnBnlearn:
    """
    Обёртки должны доносить до алгоритма выбранный тест независимости.

    На дискретных данных это проверяется содержательно: результат с
    ``chisq`` обязан отличаться от результата с ``fisherz``, иначе
    параметр где-то теряется по дороге.
    """

    @pytest.mark.parametrize("wrapper_key,impl_key", [
        ("pc_wrapper", "pc_algo"),
        ("fci_wrapper", "fci_v2"),
    ])
    @pytest.mark.parametrize("case_name", CASE_PARAMS)
    def test_wrapper_matches_wrapped(self, request, wrapper_key, impl_key, case_name):
        _requires_flag(request)

        wrapped = get_endpoint_matrix(impl_key, case_name)
        wrapper = get_endpoint_matrix(wrapper_key, case_name)
        result = compare(wrapped, wrapper)

        assert result["identical"], (
            f"{wrapper_key} на bnlearn/{case_name} искажает результат {impl_key}: "
            f"{format_summary(result, label_a=impl_key, label_b=wrapper_key)}"
        )

    @pytest.mark.parametrize("wrapper_key", ["pc_wrapper", "fci_wrapper"])
    def test_wrapper_honours_indep_test(self, request, wrapper_key):
        """
        Проверка агрегирующая, а не покейсовая, и это принципиально:
        на отдельном кейсе chisq и fisherz могут дать один и тот же граф
        по совершенно естественным причинам (так ведёт себя bnlearn/child),
        и покейсовая проверка ловила бы такие совпадения как дефект.
        Параметр точно теряется лишь тогда, когда результаты совпадают
        на всех кейсах разом.

        Берутся только мелкие графы: см. INDEP_TEST_COMPARISON_MAX_NODES.
        """
        _requires_flag(request)

        cases = [
            name for name, _n in select_cases(max_nodes=INDEP_TEST_COMPARISON_MAX_NODES)
        ]
        differing = []
        for case_name in cases:
            with_chisq = get_endpoint_matrix(wrapper_key, case_name, indep_test="chisq")
            with_fisherz = get_endpoint_matrix(wrapper_key, case_name, indep_test="fisherz")
            if not np.array_equal(with_chisq, with_fisherz):
                differing.append(case_name)

        assert differing, (
            f"{wrapper_key}: результаты с chisq и fisherz совпали на всех "
            f"{len(cases)} кейсах bnlearn — параметр indep_test не доходит "
            f"до алгоритма"
        )


# ============================================================
#  Прямой запуск: таблица расхождений
# ============================================================

def run_report(cases, indep_test=INDEP_TEST):
    """Печатает таблицу сверки и возвращает список строк отчёта."""
    rows = []

    print(f"\nсверка с causal-learn на bnlearn (indep_test={indep_test}, n_jobs={N_JOBS})")
    print("=" * 96)
    print(f"{'кейс':<14}{'узлов':>6}  {'реализация':<16}{'время':>8}  "
          f"{'матрица':<9}{'скелет':<8}{'ориент.':>8}{'рёбра±':>8}")
    print("-" * 96)

    for case_name, n_nodes in cases:
        for family, baseline_key in (("PC", "PC"), ("FCI", "FCI")):
            try:
                reference = get_endpoint_matrix(baseline_key, case_name, indep_test)
            except Exception as exc:
                print(f"{case_name:<14}{n_nodes:>6}  эталон {family} упал: "
                      f"{type(exc).__name__}: {exc}")
                continue

            for impl_key in [k for k, fam in FAMILY.items() if fam == family]:
                started = time.perf_counter()
                try:
                    actual = get_endpoint_matrix(impl_key, case_name, indep_test)
                    elapsed = time.perf_counter() - started
                except Exception as exc:
                    print(f"{case_name:<14}{n_nodes:>6}  {impl_key:<16}"
                          f"{'—':>8}  упал: {type(exc).__name__}: {exc}")
                    continue

                result = compare(reference, actual)
                rows.append((case_name, n_nodes, impl_key, elapsed, result))
                print(
                    f"{case_name:<14}{n_nodes:>6}  {impl_key:<16}{elapsed:>7.1f}s  "
                    f"{'совпала' if result['identical'] else 'РАЗНАЯ':<9}"
                    f"{'совпал' if result['skeleton_identical'] else 'РАЗНЫЙ':<8}"
                    f"{result['n_orientation_diffs']:>8}"
                    f"{result['only_in_a'] + result['only_in_b']:>8}"
                )

    print("-" * 96)
    for impl_key in IMPLS:
        impl_rows = [row for row in rows if row[2] == impl_key]
        if not impl_rows:
            continue
        n_identical = sum(1 for row in impl_rows if row[4]["identical"])
        n_skeleton = sum(1 for row in impl_rows if row[4]["skeleton_identical"])
        print(f"  {impl_key:<16} матрица {n_identical}/{len(impl_rows)}, "
              f"скелет {n_skeleton}/{len(impl_rows)}")
    print("=" * 96)

    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Сверка PC/FCI из whydra с оригиналами causal-learn на кейсах bnlearn.",
    )
    parser.add_argument(
        "--cases", nargs="*", default=None,
        help=f"имена кейсов; по умолчанию все, что укладываются в --max-nodes. "
             f"Доступны: {', '.join(list_bnlearn_cases())}",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=DEFAULT_MAX_NODES,
        help=f"не брать графы крупнее (по умолчанию {DEFAULT_MAX_NODES}; 0 — без ограничения)",
    )
    parser.add_argument(
        "--indep-test", default=INDEP_TEST,
        help=f"тест условной независимости (по умолчанию {INDEP_TEST})",
    )
    args = parser.parse_args(argv)

    max_nodes = None if args.max_nodes == 0 else args.max_nodes
    cases = select_cases(max_nodes=max_nodes, only=args.cases)
    if not cases:
        print("под заданные условия не подошёл ни один кейс")
        return 1

    run_report(cases, indep_test=args.indep_test)
    return 0


if __name__ == "__main__":
    sys.exit(main())
