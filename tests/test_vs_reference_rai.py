"""
Сверка RAI whydra с референсом IntelLabs causality-lab (``LearnStructRAI``).

Эталон — граф, который референс строит на тех же данных с тем же CI-тестом
whydra (``rai_algo._CIOracle``). Golden-рёбра ниже посчитаны референсом один
раз и зашиты в файл; если пакет ``causal_discovery_algs`` установлен, тот же
граф дополнительно сверяется с живым референсом.

Проверяется побитовое равенство endpoint-матриц для
``rai_stable(n_jobs=1)`` и ``rai_stable(n_jobs=4)``: параллельный слой RAI
ускоряет только CI-тесты и не имеет права менять результат.
"""

import warnings

import numpy as np
import pytest

from whydra.algorithms.graph_core import CIT
from whydra.algorithms.standalone import rai_algo

from .conftest import generate_linear_sem_data
from .test_vs_causallearn import GRAPHS, get_data

try:
    from causal_discovery_algs.rai import LearnStructRAI
    from graphical_models import arrow_head_types as Mark
    HAS_REFERENCE = True
except ImportError:  # pragma: no cover — референс не входит в зависимости пакета
    HAS_REFERENCE = False

ALPHA = 0.05
TAIL, ARROW = -1, 1

#: Референс на графах tests/test_vs_causallearn.py (fisherz, n=2000, seed=42).
#: Ребро (i, j, "->") — i → j; (i, j, "--") — i — j.
FISHERZ_GOLDEN = {
    "chain_3": [(0, 1, "--"), (1, 2, "--")],
    "collider_3": [(0, 1, "->"), (2, 1, "->")],
    "v_structure_4": [(0, 1, "->"), (2, 1, "->"), (1, 3, "->")],
    "diamond_5": [(0, 1, "--"), (0, 2, "--"), (1, 3, "->"), (2, 3, "->"), (3, 4, "->")],
    "dag_6": [(0, 1, "--"), (0, 2, "--"), (1, 3, "--"), (2, 4, "--"), (3, 5, "->"), (4, 5, "->")],
    "dag_8": [(0, 2, "->"), (0, 4, "->"), (1, 2, "->"), (1, 5, "--"), (2, 3, "->"), (3, 4, "->"),
              (4, 7, "->"), (5, 6, "--"), (6, 7, "->")],
}

#: Дискретный кейс для chisq: случайный DAG на 10 узлах (см. ``_discrete_data``),
#: n=3000, значения — тертили линейной SEM.
DISCRETE_TRUE_DAG = [(0, 4), (0, 5), (0, 9), (1, 8), (2, 3), (2, 9), (3, 4), (3, 6), (3, 7), (4, 7),
                     (5, 8), (5, 9)]
DISCRETE_N_VARS = 10
CHISQ_GOLDEN = [(0, 4, "--"), (0, 5, "--"), (0, 8, "--"), (1, 8, "--"), (3, 2, "->"), (9, 2, "->"),
                (4, 3, "->"), (6, 3, "->"), (3, 7, "->"), (9, 3, "->"), (4, 6, "--"), (4, 7, "->"),
                (5, 8, "--"), (5, 9, "->"), (6, 9, "->")]

RUN_MODES = ["rai_stable_j1", "rai_stable_j4"]


def _golden_matrix(edges, n_vars):
    """Собрать endpoint-матрицу (TAIL=-1, ARROW=1) из списка golden-рёбер."""
    mat = np.zeros((n_vars, n_vars), dtype=int)
    for i, j, kind in edges:
        if kind == "->":
            mat[i, j] = ARROW
            mat[j, i] = TAIL
        else:
            mat[i, j] = TAIL
            mat[j, i] = TAIL
    return mat


def _discrete_data():
    """Дискретные данные для chisq: тертили линейной SEM по ``DISCRETE_TRUE_DAG``."""
    W = np.zeros((DISCRETE_N_VARS, DISCRETE_N_VARS))
    rng = np.random.default_rng(7)
    for i in range(DISCRETE_N_VARS):
        for j in range(i + 1, DISCRETE_N_VARS):
            if rng.random() < 0.3:
                W[i, j] = rng.uniform(0.5, 1.0) * rng.choice([-1, 1])
    assert [(i, j) for i in range(DISCRETE_N_VARS) for j in range(DISCRETE_N_VARS) if W[i, j] != 0] == DISCRETE_TRUE_DAG
    cont = generate_linear_sem_data(W, n_samples=3000, seed=11)
    return np.stack(
        [np.digitize(cont[:, c], np.quantile(cont[:, c], [0.33, 0.66])) for c in range(DISCRETE_N_VARS)],
        axis=1,
    ).astype(float)


def _run_whydra(mode, data, indep_test):
    """Запустить whydra RAI в одном из режимов и вернуть endpoint-матрицу."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n_jobs = int(mode.rsplit("j", 1)[1])
        res = rai_algo.rai_stable(data, alpha=ALPHA, indep_test=indep_test, n_jobs=n_jobs)
    return np.asarray(res.G.graph)


def _run_reference(data, indep_test):
    """Живой референс IntelLabs с CI-оракулом whydra; endpoint-матрица."""
    n_vars = data.shape[1]
    oracle = rai_algo._CIOracle(CIT(data, method=indep_test), ALPHA)
    rai = LearnStructRAI(nodes_set=set(range(n_vars)), ci_test=oracle)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rai.learn_structure()
    mat = np.zeros((n_vars, n_vars), dtype=int)
    for j in range(n_vars):
        for i in rai.graph._graph[j][Mark.Directed]:
            mat[i, j] = ARROW
            mat[j, i] = TAIL
        for i in rai.graph._graph[j][Mark.Undirected]:
            mat[i, j] = TAIL
            mat[j, i] = TAIL
    return mat


def _cases():
    """Все кейсы: (id, data, indep_test, golden-матрица)."""
    cases = []
    for key in GRAPHS:
        data = get_data(key)
        cases.append((key, data, "fisherz", _golden_matrix(FISHERZ_GOLDEN[key], data.shape[1])))
    cases.append(("discrete_10", _discrete_data(), "chisq", _golden_matrix(CHISQ_GOLDEN, DISCRETE_N_VARS)))
    return cases


CASES = _cases()


@pytest.mark.parametrize("mode", RUN_MODES)
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_matches_reference_golden(mode, case):
    """Endpoint-матрица whydra побитово равна golden-матрице референса."""
    _, data, indep_test, golden = case
    mat = _run_whydra(mode, data, indep_test)
    diff = int((mat != golden).sum())
    assert diff == 0, f"{mode}: {diff} ячеек отличаются от референса"


@pytest.mark.skipif(not HAS_REFERENCE, reason="causal_discovery_algs (IntelLabs causality-lab) не установлен")
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_golden_matches_live_reference(case):
    """Golden-матрицы не устарели: живой референс даёт то же самое."""
    _, data, indep_test, golden = case
    assert np.array_equal(_run_reference(data, indep_test), golden)


def test_parallel_matches_sequential_on_random_dags():
    """n_jobs=3 даёт ту же матрицу, что n_jobs=1, на нескольких случайных DAG (fisherz и chisq)."""
    rng = np.random.default_rng(123)
    for _ in range(4):
        n_vars = int(rng.integers(8, 14))
        W = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if rng.random() < 0.3:
                    W[i, j] = rng.uniform(0.4, 0.9) * rng.choice([-1, 1])
        cont = generate_linear_sem_data(W, n_samples=1500, seed=int(rng.integers(1_000_000)))
        disc = np.stack(
            [np.digitize(cont[:, c], np.quantile(cont[:, c], [0.33, 0.66])) for c in range(n_vars)], axis=1
        ).astype(float)
        for indep_test, data in (("fisherz", cont), ("chisq", disc)):
            seq = _run_whydra("rai_stable_j1", data, indep_test)
            par = _run_whydra("rai_stable_j3", data, indep_test)
            assert np.array_equal(seq, par), f"{indep_test}: n_jobs=3 расходится с n_jobs=1"
