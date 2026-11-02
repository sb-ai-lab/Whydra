"""Регрессии кодировки скелета, классического FCI и ограничения Possible-D-SEP."""

import itertools

import numpy as np
import pytest

from whydra.algorithms.graph_core import CIT, CausalGraph
from whydra.algorithms.standalone import (
    cfci_algo,
    fci_algo_cash,
    fci_algo_paralel_cash,
    fci_algo_v2,
    standalone_fci_stable,
)
from whydra.algorithms.standalone.pag_rules import PAG_CIRCLE, PAG_TAIL
from whydra.algorithms.wrappers import StandaloneFCIStable

ALPHA = 0.05
PDS_EDGES = [(0, 1), (1, 2), (1, 4), (2, 4), (0, 3)]
PDSEP_MODULES = [
    fci_algo_cash, fci_algo_paralel_cash, cfci_algo,
    standalone_fci_stable, fci_algo_v2,
]


def _skeleton(p, edges, encoding):
    """Создаёт скелет по p, списку edges и метке encoding; возвращает матрицу."""
    sk = np.zeros((p, p), dtype=int)
    for x, y in edges:
        sk[x, y] = sk[y, x] = encoding
    return sk


def _sepset(p):
    """Создаёт пустые сепараторы для p узлов; возвращает словарь ordered pairs."""
    return {(x, y): set() for x, y in itertools.permutations(range(p), 2)}


def _oracle(independent, calls):
    """Создаёт CI-тест по словарю independent и журналу calls; возвращает функцию."""

    def indep(
        suffStat,
        x,
        y,
        S,
    ):
        """Записывает x, y, S в журнал; suffStat не нужен; возвращает p-value."""
        calls.append((x, y, tuple(S)))
        pair = (min(x, y), max(x, y))
        return 1.0 if frozenset(S) in independent.get(pair, ()) else 0.0

    return indep


def _cg(data):
    """Создаёт CausalGraph по data с fisherz; возвращает граф с установленным тестом."""
    p = data.shape[1]
    cg = CausalGraph(p, [f"X{i + 1}" for i in range(p)])
    cg.set_ind_test(CIT(data, method="fisherz"))
    return cg


def _gaussian_data():
    """Создаёт небольшую линейную SEM с фиксированным seed; возвращает данные."""
    data = np.random.default_rng(41).normal(size=(500, 5))
    data[:, 2] += data[:, 0] + data[:, 1]
    data[:, 3] += 0.8 * data[:, 2]
    data[:, 4] += 0.7 * data[:, 3]
    return data


def _confounded_data():
    """Создаёт SEM с двумя латентными узлами; возвращает наблюдаемые столбцы."""
    rng = np.random.default_rng(10)
    p = 12
    weights = np.zeros((p, p))
    for src in range(p):
        for dst in range(src + 1, p):
            if rng.random() < 0.3:
                weights[src, dst] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.0)
    data = rng.normal(size=(1000, p))
    for dst in range(p):
        data[:, dst] += data[:, :dst] @ weights[:dst, dst]
    return data[:, 2:]


@pytest.mark.parametrize("module", PDSEP_MODULES, ids=lambda module: module.__name__.split(".")[-1])
def test_pdsep_collider_reachability_and_encoding(module, monkeypatch):
    """Для module проверяет PDS через коллайдер и обе кодировки; monkeypatch ставит spy."""
    captured = {}
    original = fci_algo_v2._possible_dsep

    def capture(
        amat,
        x,
        y,
        adj=None,
    ):
        """Запоминает PDS для amat, x, y, adj; возвращает исходный результат обхода."""
        result = original(amat, x, y, adj=adj)
        captured[(x, y)] = result
        return result

    if module is fci_algo_v2:
        monkeypatch.setattr(module, "_possible_dsep", capture)

    def run(encoding, collider):
        """Запускает pdsep с encoding и collider; возвращает результат, PDS и CI-вызовы."""
        p = 5
        sep = _sepset(p)
        # Подавляем все коллайдеры кроме 0 *-> 1 <-* 2. Треугольник
        # 1–2–4 даёт полунаправленный обход из 1 в 2, нужный PDS v2.
        for x, y, middle in [(0, 4, 1), (1, 3, 0)]:
            sep[(x, y)] = sep[(y, x)] = {middle}
        if not collider:
            sep[(0, 2)] = sep[(2, 0)] = {1}
        calls = []
        captured.clear()
        pdsep = module.pdsep_v2 if module is fci_algo_v2 else module.pdsep
        kwargs = {"n_jobs": 1} if module in (fci_algo_paralel_cash, fci_algo_v2) else {}
        result = pdsep(
            {"sk": _skeleton(p, PDS_EDGES, encoding)}, None, _oracle({}, calls),
            p, sep, ALPHA, np.full((p, p), -np.inf), unfVect=set(), **kwargs,
        )
        pds = captured[(0, 3)] if module is fci_algo_v2 else result["allPdsep"][0]
        return result, list(pds), calls

    tail, tail_pds, tail_calls = run(PAG_TAIL, True)
    circle, circle_pds, circle_calls = run(PAG_CIRCLE, True)
    assert 2 in tail_pds
    assert tail_pds == circle_pds
    assert tail_calls == circle_calls
    assert tail_calls
    assert np.array_equal(tail["G"], circle["G"])
    assert tail["sepset"] == circle["sepset"]
    assert tail["allPdsep"] == circle["allPdsep"]
    assert np.array_equal(tail["pMax"], circle["pMax"])
    assert 2 not in run(PAG_TAIL, False)[1]
    assert 2 not in run(PAG_CIRCLE, False)[1]


@pytest.mark.parametrize("module", [fci_algo_cash, fci_algo_paralel_cash, standalone_fci_stable])
def test_classic_fci_has_no_conservative_step(module, monkeypatch):
    """Проверяет отсутствие консервативного шага в module; monkeypatch ловит вызов."""

    def forbidden(*args, **kwargs):
        """При любых args/kwargs прекращает тест; результата не возвращает."""
        pytest.fail("Классический FCI вызвал pc_cons_intern")

    data = _gaussian_data()
    if module is standalone_fci_stable:
        assert not hasattr(module, "pc_cons_intern")
        result = module.fci_stable(data, alpha=ALPHA, indep_test="fisherz")
    else:
        monkeypatch.setattr(module, "pc_cons_intern", forbidden)
        result = module.fci_stable(_cg(data), data, alpha=ALPHA, indep_test="fisherz", n_jobs=1)
    assert result.shape == (5, 5)


@pytest.mark.parametrize("module", [cfci_algo, fci_algo_paralel_cash])
def test_conservative_triples_support_tail_skeleton(module):
    """Проверяет неоднозначную тройку в module на обеих кодировках; результата нет."""

    def run(encoding):
        """Запускает conservative step с encoding; возвращает словарь графа и CI-вызовы."""
        graph = {"sk": _skeleton(3, [(0, 1), (1, 2)], encoding), "sepset": _sepset(3)}
        calls = []
        indep = _oracle({(0, 2): {frozenset(), frozenset({1})}}, calls)
        kwargs = {"n_jobs": 1} if module is fci_algo_paralel_cash else {}
        result = module.pc_cons_intern(graph, None, ALPHA, indep, maj_rule=False, **kwargs)
        return result, calls

    tail, tail_calls = run(PAG_TAIL)
    circle, circle_calls = run(PAG_CIRCLE)
    assert (0, 1, 2) in tail["unfTriples"]
    assert tail["unfTriples"] == circle["unfTriples"]
    assert tail["sepset"] == circle["sepset"]
    assert tail_calls == circle_calls
    assert tail_calls


def test_fci_v2_m_max_and_wrapper(monkeypatch):
    """С monkeypatch проверяет предел CI-наборов и передачу m_max обёрткой; результата нет."""
    data = _confounded_data()
    original = fci_algo_v2.pdsep_v2
    sizes = []
    limits = []

    def spy(
        skel,
        suffStat,
        indepTest,
        p,
        sepSet,
        alpha,
        pMax,
        **kwargs,
    ):
        """Записывает лимит из kwargs и CI-размеры; остальные args передаёт в pdsep."""
        limits.append(kwargs.get("m_max", float("inf")))

        def indep(
            suff,
            x,
            y,
            S,
        ):
            """Записывает размер S, передаёт suff/x/y в CI-тест; возвращает p-value."""
            sizes.append(len(S))
            return indepTest(suff, x, y, S)

        return original(skel, suffStat, indep, p, sepSet, alpha, pMax, **kwargs)

    monkeypatch.setattr(fci_algo_v2, "pdsep_v2", spy)
    default = fci_algo_v2.fci_stable_v2(_cg(data), data, alpha=ALPHA, n_jobs=1)
    assert sizes and max(sizes) >= 3
    sizes.clear()
    explicit = fci_algo_v2.fci_stable_v2(_cg(data), data, alpha=ALPHA, n_jobs=1, m_max=float("inf"))
    assert np.array_equal(default, explicit)
    assert max(sizes) >= 3
    sizes.clear()
    fci_algo_v2.fci_stable_v2(_cg(data), data, alpha=ALPHA, n_jobs=1, m_max=2)
    assert sizes and max(sizes) <= 2
    assert limits[-1] == 2
    sizes.clear()
    StandaloneFCIStable(alpha=ALPHA, indep_test="fisherz", n_jobs=1, m_max=2).run(data)
    assert limits[-1] == 2
    assert sizes and max(sizes) <= 2
    sizes.clear()
    fci_algo_v2.fci_stable_v2(_cg(data), data, alpha=ALPHA, n_jobs=1, m_max=1)
    assert limits[-1] == 1
    assert sizes == []
