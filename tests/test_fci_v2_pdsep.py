"""Тесты шага Possible-D-SEP в FCI v2 (`fci_algo_v2.pdsep_v2`).

Шаг воспроизводит семантику ``removeByPossibleDsep`` из causal-learn 0.1.4.5:
Possible-D-SEP по ребру (порт Tetrad ``getPossibleDsep``), сепараторы размера
не меньше 2, пропуск подмножеств adj(a)/adj(b), немедленные удаления. Порт PDS
сверяется с самим causal-learn как с оракулом; параллельная версия должна
давать тот же результат, что последовательная.
"""

import itertools

import numpy as np
import pytest

from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.FCI import existsSemiDirectedPath, getPossibleDsep

from whydra.algorithms.standalone.fci_algo_v2 import (
    _exists_semi_directed_path,
    _possible_dsep,
    _side_order,
    pdsep_v2,
)
from whydra.algorithms.standalone.pag_rules import (
    PAG_ARROW,
    PAG_CIRCLE,
    PAG_TAIL,
)

N_SEEDS_SEMI_DIRECTED = 20
N_SEEDS_PDS = 60
P_SEMI_DIRECTED = 9
ALPHA = 0.05
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (2, 4), (4, 5), (3, 5), (0, 3),
]
SKELETON_P = 6


def _to_cl_graph(amat):
    """Переводит whydra-матрицу в граф causal-learn (транспонированная раскладка)."""
    p = amat.shape[0]
    nodes = [GraphNode(f"X{i + 1}") for i in range(p)]
    g = GeneralGraph(nodes)
    g.graph = amat.T.copy()
    return nodes, g


def _random_pag(rng, p, density):
    """Случайный PAG: для каждой пары i<j с вероятностью density — ребро со случайными метками."""
    amat = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(i + 1, p):
            if rng.random() < density:
                amat[i][j] = rng.choice((PAG_TAIL, PAG_ARROW, PAG_CIRCLE))
                amat[j][i] = rng.choice((PAG_TAIL, PAG_ARROW, PAG_CIRCLE))
    return amat


def _make_oracle(independent, calls):
    """Фальшивый CI-тест: 1.0, если S входит в независимые наборы пары, иначе 0.0."""

    def indep(suffStat, x, y, S):
        calls.append((x, y, tuple(S)))
        key = (min(x, y), max(x, y))
        return 1.0 if frozenset(S) in independent.get(key, ()) else 0.0

    return indep


def _skeleton_matrix(value=1):
    """Тестовый скелет на 6 узлах; value — код «ребро есть»."""
    sk = np.zeros((SKELETON_P, SKELETON_P), dtype=int)
    for i, j in SKELETON_EDGES:
        sk[i, j] = value
        sk[j, i] = value
    return sk


def _empty_sepset(p=SKELETON_P):
    """Словарь sepset по всем упорядоченным парам с пустыми множествами."""
    return {(i, j): set() for i, j in itertools.permutations(range(p), 2)}


def _cl_names_to_indices(nodes):
    """Индексы узлов causal-learn по их именам 'X{i+1}'."""
    return [int(n.get_name()[1:]) - 1 for n in nodes]


@pytest.mark.parametrize("seed", range(N_SEEDS_SEMI_DIRECTED))
def test_semi_directed_path_matches_causallearn(seed):
    """_exists_semi_directed_path совпадает с existsSemiDirectedPath из causal-learn."""
    rng = np.random.default_rng(seed)
    amat = _random_pag(rng, P_SEMI_DIRECTED, 0.35)
    nodes, g = _to_cl_graph(amat)
    for a in range(P_SEMI_DIRECTED):
        for b in range(P_SEMI_DIRECTED):
            if a == b:
                continue
            assert _exists_semi_directed_path(amat, a, b) == existsSemiDirectedPath(
                nodes[a], nodes[b], g
            ), f"seed={seed}, pair=({a}, {b})"


@pytest.mark.parametrize("seed", range(N_SEEDS_PDS))
def test_possible_dsep_matches_causallearn(seed, monkeypatch):
    """_possible_dsep совпадает (включая порядок) с getPossibleDsep из causal-learn."""
    # getPossibleDsep обходит Python-множества узлов, а GraphNode хешируется по
    # имени, поэтому порядок обхода (и в редких случаях сам результат) зависит
    # от PYTHONHASHSEED. Хеш по индексу делает обход референса возрастающим по
    # индексам — ровно тем, который использует whydra, — и оракул становится
    # детерминированным.
    monkeypatch.setattr(GraphNode, "__hash__", lambda self: int(self.get_name()[1:]) - 1)
    rng = np.random.default_rng(seed)
    p = 8 + seed % 7
    amat = _random_pag(rng, p, 0.3)
    nodes, g = _to_cl_graph(amat)
    for x in range(p):
        for y in range(p):
            if x == y or amat[x][y] == 0:
                continue
            expected = _cl_names_to_indices(getPossibleDsep(nodes[x], nodes[y], g, -1))
            assert _possible_dsep(amat, x, y) == expected, f"seed={seed}, edge=({x}, {y})"


def test_possible_dsep_excludes_endpoints_and_paths_through_y():
    """PDS не содержит концов ребра и не проходит сквозь y."""
    # Граф A: цепочка 0 o-o 1 o-o 2 o-o 3 плюс ребро 0 o-o 3.
    amat = np.zeros((4, 4), dtype=int)
    for i, j in [(0, 1), (1, 2), (2, 3), (0, 3)]:
        amat[i][j] = PAG_CIRCLE
        amat[j][i] = PAG_CIRCLE
    pds = _possible_dsep(amat, 0, 3)
    assert 0 not in pds
    assert 3 not in pds
    assert 1 in pds

    # Граф B: только 0 o-o 3 и 3 o-o 4 — единственный путь к 4 идёт через y=3.
    amat = np.zeros((5, 5), dtype=int)
    for i, j in [(0, 3), (3, 4)]:
        amat[i][j] = PAG_CIRCLE
        amat[j][i] = PAG_CIRCLE
    assert 4 not in _possible_dsep(amat, 0, 3)


def test_possible_dsep_order_is_reverse_by_default_name():
    """Порядок PDS — обратный лексикографический по именам X1..Xp, как у causal-learn."""
    amat = np.zeros((12, 12), dtype=int)
    for j in (1, 9, 10, 11):
        amat[0][j] = PAG_CIRCLE
        amat[j][0] = PAG_CIRCLE
    assert _possible_dsep(amat, 0, 1) == [11, 10, 9]

    amat[0][2] = PAG_CIRCLE
    amat[2][0] = PAG_CIRCLE
    assert _possible_dsep(amat, 0, 1) == [2, 11, 10, 9]


def test_side_order_reproduces_edge_flip():
    """_side_order переворачивает стороны только при стрелке у i и круге/хвосте у j."""
    amat = np.zeros((2, 2), dtype=int)
    amat[1][0] = PAG_ARROW
    amat[0][1] = PAG_CIRCLE
    assert _side_order(amat, 0, 1) == (1, 0)

    amat = np.zeros((2, 2), dtype=int)
    amat[0][1] = PAG_CIRCLE
    amat[1][0] = PAG_CIRCLE
    assert _side_order(amat, 0, 1) == (0, 1)

    amat = np.zeros((2, 2), dtype=int)
    amat[0][1] = PAG_ARROW
    amat[1][0] = PAG_ARROW
    assert _side_order(amat, 0, 1) == (0, 1)


def test_pdsep_result_independent_of_n_jobs():
    """Результат pdsep_v2 не зависит от n_jobs и от кодировки скелета."""
    skel = _skeleton_matrix()
    # Наборы лежат в PDS соответствующей стороны и не являются подмножествами
    # adj(a)/adj(b), поэтому реально тестируются. Удаление 2–4 меняет PDS
    # ребра 3–4, которое идёт следом: так проверяется ветка пересчёта
    # спекулятивного результата на живом графе.
    independent = {
        (0, 1): {frozenset({3, 4})},
        (2, 4): {frozenset({1, 5})},
    }

    def run(n_jobs, sk_matrix):
        calls = []
        return pdsep_v2(
            {"sk": sk_matrix.copy()},
            None,
            _make_oracle(independent, calls),
            SKELETON_P,
            _empty_sepset(),
            ALPHA,
            np.full((SKELETON_P, SKELETON_P), -np.inf),
            n_jobs=n_jobs,
        )

    r1 = run(1, skel)
    r2 = run(2, skel)

    assert np.array_equal(r1["G"], r2["G"])
    assert r1["sepset"] == r2["sepset"]
    assert np.array_equal(r1["pMax"], r2["pMax"])
    assert r1["G"].sum() < skel.sum()

    for i in range(SKELETON_P):
        for j in range(SKELETON_P):
            if i != j and r1["G"][i, j] == 0 and skel[i, j] != 0:
                assert r1["sepset"][(i, j)] == r1["sepset"][(j, i)]
                assert r1["sepset"][(i, j)]

    for value in (-1, 2):
        r = run(1, _skeleton_matrix(value))
        assert np.array_equal(r["G"], r1["G"])


def test_pdsep_conditioning_sets_have_size_at_least_two_and_skip_adjacent_subsets():
    """Кондиционирующие множества имеют размер >= 2 и не лежат целиком в adj(x) или adj(y)."""
    skel = _skeleton_matrix()
    adj = {i: {j for j in range(SKELETON_P) if skel[i, j] != 0} for i in range(SKELETON_P)}
    calls = []
    res = pdsep_v2(
        {"sk": skel.copy()},
        None,
        _make_oracle({}, calls),
        SKELETON_P,
        _empty_sepset(),
        ALPHA,
        np.full((SKELETON_P, SKELETON_P), -np.inf),
    )
    assert calls
    for x, y, S in calls:
        assert len(S) >= 2
        assert x != y
        assert x not in S
        assert y not in S
        assert not set(S) <= adj[x]
        assert not set(S) <= adj[y]
    assert np.array_equal(res["G"], skel)


def test_pdsep_protected_edges_are_never_tested_or_removed():
    """Защищённые (BK) рёбра не тестируются и не удаляются."""
    skel = _skeleton_matrix()
    calls = []
    res = pdsep_v2(
        {"sk": skel.copy()},
        None,
        lambda suff, x, y, S: (calls.append((x, y, tuple(S))), 1.0)[1],
        SKELETON_P,
        _empty_sepset(),
        ALPHA,
        np.full((SKELETON_P, SKELETON_P), -np.inf),
        protected_edges={(0, 1)},
    )
    G = res["G"]
    assert G[0, 1] != 0 and G[1, 0] != 0
    for x, y, _S in calls:
        assert {x, y} != {0, 1}
