"""Unit-тесты примитивов PDAG/CPDAG (whydra/algorithms/score_based/pdag.py).

Все проверки детерминированы и не используют данные: матрицы выписаны руками,
ожидаемый результат выводится из определений, а не из прогона. Это сознательно —
предыдущая попытка чинить GFCI опиралась на инстанс, подобранный перебором, и
разъехалась при смене версии numpy.

Кодировка: C[x,y]=1,C[y,x]=0 → x→y;  C[x,y]=C[y,x]=1 → x—y;  оба 0 → нет ребра.
"""

import numpy as np
import pytest

from whydra.algorithms.score_based.pdag import (
    complete_pdag,
    dag_to_cpdag,
    exists_semidirected_path,
    is_adjacent,
    is_clique,
    is_directed,
    is_undirected,
    meek_rules,
    na_yx,
    neighbours,
    parents,
    pdag_to_dag,
)


def mat(p, directed=(), undirected=()):
    C = np.zeros((p, p), dtype=np.int8)
    for a, b in directed:
        C[a, b] = 1
    for a, b in undirected:
        C[a, b] = 1
        C[b, a] = 1
    return C


# ============================================================
#  Базовые отношения
# ============================================================

def test_relations_distinguish_directed_from_undirected():
    C = mat(3, directed=[(0, 1)], undirected=[(1, 2)])
    assert is_directed(C, 0, 1) and not is_directed(C, 1, 0)
    assert is_undirected(C, 1, 2) and is_undirected(C, 2, 1)
    assert not is_undirected(C, 0, 1)
    assert is_adjacent(C, 0, 1) and is_adjacent(C, 1, 0)
    assert not is_adjacent(C, 0, 2)


def test_neighbours_parents_and_na_yx():
    # 0 -> 1, 1 -- 2, 2 -- 3, 0 -- 2
    C = mat(4, directed=[(0, 1)], undirected=[(1, 2), (2, 3), (0, 2)])
    assert neighbours(C, 1) == {2}
    assert neighbours(C, 2) == {1, 3, 0}
    assert parents(C, 1) == {0}
    assert parents(C, 2) == set()
    # NA_yx: соседи y=2, смежные с x=1 -> только 1 сам сосед; проверяем через 0
    assert na_yx(C, 2, 0) == {1}


@pytest.mark.parametrize(
    "nodes, expected",
    [((), True), ((0,), True), ((0, 1), True), ((0, 1, 2), True), ((0, 1, 3), False)],
)
def test_is_clique(nodes, expected):
    # треугольник 0-1-2 замкнут, 3 висит на 0
    C = mat(4, undirected=[(0, 1), (1, 2), (0, 2), (0, 3)])
    assert is_clique(C, nodes) is expected


# ============================================================
#  Полу-направленные пути
# ============================================================

def test_semidirected_path_follows_arrows_and_undirected_edges():
    # 0 -> 1 -- 2,  3 -> 2 (против стрелки из 2 в 3 хода нет)
    C = mat(4, directed=[(0, 1), (3, 2)], undirected=[(1, 2)])
    assert exists_semidirected_path(C, 0, 2)
    assert not exists_semidirected_path(C, 2, 3)
    assert not exists_semidirected_path(C, 2, 0)


def test_semidirected_path_respects_block_set():
    C = mat(3, directed=[(0, 1), (1, 2)])
    assert exists_semidirected_path(C, 0, 2)
    assert not exists_semidirected_path(C, 0, 2, block={1})


# ============================================================
#  Правила Мика R1-R4
# ============================================================

def test_meek_r1_orients_away_from_collider_free_triple():
    # 0 -> 1, 1 -- 2, 0 и 2 несмежны  =>  1 -> 2
    C = meek_rules(mat(3, directed=[(0, 1)], undirected=[(1, 2)]))
    assert is_directed(C, 1, 2)


def test_meek_r2_closes_directed_path():
    # 0 -> 1 -> 2, 0 -- 2  =>  0 -> 2
    C = meek_rules(mat(3, directed=[(0, 1), (1, 2)], undirected=[(0, 2)]))
    assert is_directed(C, 0, 2)


def test_meek_r3_orients_from_two_nonadjacent_parents():
    # a=0 -- b=1, 0 -- 2, 0 -- 3, 2 -> 1, 3 -> 1, 2 и 3 несмежны  =>  0 -> 1
    C = meek_rules(
        mat(4, directed=[(2, 1), (3, 1)], undirected=[(0, 1), (0, 2), (0, 3)])
    )
    assert is_directed(C, 0, 1)


def test_meek_r4_orients_through_directed_chain():
    # a=0 -- b=1, 0 -- 2, 2 -> 3, 3 -> 1, 1 и 2 несмежны, 0 смежна 3  =>  0 -> 1
    C = meek_rules(
        mat(4, directed=[(2, 3), (3, 1)], undirected=[(0, 1), (0, 2), (0, 3)])
    )
    assert is_directed(C, 0, 1)


def test_meek_is_idempotent_on_a_completed_cpdag():
    C = dag_to_cpdag(mat(4, directed=[(0, 1), (2, 1), (1, 3)]))
    assert np.array_equal(meek_rules(C), C)


def test_meek_leaves_an_unorientable_chain_alone():
    # 0 -- 1 -- 2 без коллайдеров ориентировать нечем
    C = meek_rules(mat(3, undirected=[(0, 1), (1, 2)]))
    assert is_undirected(C, 0, 1) and is_undirected(C, 1, 2)


# ============================================================
#  Расширение PDAG -> DAG
# ============================================================

def test_pdag_to_dag_returns_an_already_directed_dag_unchanged():
    dag = mat(3, directed=[(0, 1), (1, 2)])
    assert np.array_equal(pdag_to_dag(dag), dag)


def test_pdag_to_dag_extends_a_chain_without_inventing_a_collider():
    """Конкретная ориентация не важна — важно, что расширение согласованное.

    Проверяем свойства, а не матрицу: у результата тот же скелет и то же
    множество неэкранированных коллайдеров (для цепочки — пустое).
    """
    C = mat(3, undirected=[(0, 1), (1, 2)])
    dag = pdag_to_dag(C)
    assert dag is not None

    skeleton = {(0, 1), (1, 2)}
    got = {
        (min(i, j), max(i, j))
        for i in range(3)
        for j in range(3)
        if dag[i, j]
    }
    assert got == skeleton

    colliders = [
        y
        for y in range(3)
        for x in range(3)
        for z in range(3)
        if x < z and dag[x, y] and dag[z, y] and not (dag[x, z] or dag[z, x])
    ]
    assert colliders == []


def test_pdag_to_dag_rejects_an_unextendable_pdag():
    """0 -> 3, 1 -> 2, 2 -- 3: обе ориентации рождают новую v-структуру.

    При 2 -> 3 у 3 появляются несмежные родители 0 и 2, при 3 -> 2 — несмежные
    родители 1 и 3 у 2. Расширение обязано сохранять множество v-структур, так
    что консистентного DAG нет. Направленного цикла здесь нет — проверяется
    именно условие Dor-Tarsi, а не тривиальная ацикличность.
    """
    C = mat(4, directed=[(0, 3), (1, 2)], undirected=[(2, 3)])
    assert pdag_to_dag(C) is None


def test_pdag_to_dag_rejects_a_directed_cycle():
    C = mat(3, directed=[(0, 1), (1, 2), (2, 0)])
    assert pdag_to_dag(C) is None


def test_pdag_to_dag_matches_exhaustive_search_on_all_4_node_pdags():
    """Сверка с полным перебором: все 4^6 конфигураций рёбер на 4 узлах.

    Для каждого PDAG перебираются все ориентации неориентированных рёбер;
    расширение существует, если хоть одна даёт ацикличный граф с тем же
    множеством v-структур. Требуем: pdag_to_dag возвращает None ровно тогда,
    когда расширения нет, а иначе — действительно консистентный DAG.
    """
    import itertools

    p = 4
    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]

    def vstructs_dag(dag):
        out = set()
        for y in range(p):
            pa = [x for x in range(p) if dag[x, y]]
            for i, x in enumerate(pa):
                for z in pa[i + 1:]:
                    if not (dag[x, z] or dag[z, x]):
                        out.add((min(x, z), y, max(x, z)))
        return out

    def vstructs_pdag(C):
        out = set()
        for y in range(p):
            pa = [x for x in range(p) if is_directed(C, x, y)]
            for i, x in enumerate(pa):
                for z in pa[i + 1:]:
                    if not is_adjacent(C, x, z):
                        out.add((min(x, z), y, max(x, z)))
        return out

    def acyclic(dag):
        indeg = dag.sum(0).copy()
        remaining = set(range(p))
        while remaining:
            ready = [v for v in remaining if indeg[v] == 0]
            if not ready:
                return False
            v = ready[0]
            remaining.discard(v)
            for w in range(p):
                if dag[v, w]:
                    indeg[w] -= 1
            indeg[v] = -1
        return True

    for cfg in itertools.product([0, 1, 2, 3], repeat=len(pairs)):
        C = np.zeros((p, p), dtype=np.int8)
        for (i, j), code in zip(pairs, cfg):
            if code == 1:
                C[i, j] = 1
            elif code == 2:
                C[j, i] = 1
            elif code == 3:
                C[i, j] = C[j, i] = 1

        base = np.zeros((p, p), dtype=np.int8)
        for i in range(p):
            for j in range(p):
                if is_directed(C, i, j):
                    base[i, j] = 1
        undirected_pairs = [(i, j) for (i, j), code in zip(pairs, cfg) if code == 3]
        target = vstructs_pdag(C)

        extendable = False
        for bits in itertools.product([0, 1], repeat=len(undirected_pairs)):
            cand = base.copy()
            for (i, j), bit in zip(undirected_pairs, bits):
                if bit:
                    cand[i, j] = 1
                else:
                    cand[j, i] = 1
            if acyclic(cand) and vstructs_dag(cand) == target:
                extendable = True
                break

        got = pdag_to_dag(C)
        if not extendable:
            assert got is None, f"ожидался None для {cfg}"
        else:
            assert got is not None, f"расширение существует, но не найдено: {cfg}"
            assert acyclic(got), f"расширение с циклом: {cfg}"
            assert vstructs_dag(got) == target, f"v-структуры изменились: {cfg}"


def test_pdag_to_dag_preserves_existing_arrowheads():
    C = mat(4, directed=[(0, 1), (2, 1)], undirected=[(1, 3)])
    dag = pdag_to_dag(C)
    assert dag is not None
    assert dag[0, 1] and dag[2, 1]
    assert not dag[1, 0] and not dag[1, 2]


# ============================================================
#  DAG -> CPDAG и completion
# ============================================================

def test_dag_to_cpdag_keeps_collider_and_frees_reversible_edges():
    # Коллайдер 0 -> 2 <- 1 инвариантен, обе стрелки обязаны остаться
    C = dag_to_cpdag(mat(3, directed=[(0, 2), (1, 2)]))
    assert is_directed(C, 0, 2) and is_directed(C, 1, 2)

    # Цепочка 0 -> 1 -> 2 обратима целиком
    C = dag_to_cpdag(mat(3, directed=[(0, 1), (1, 2)]))
    assert is_undirected(C, 0, 1) and is_undirected(C, 1, 2)


def test_chain_and_fork_share_one_cpdag():
    """Цепь и вилка лежат в одном классе эквивалентности."""
    chain = dag_to_cpdag(mat(3, directed=[(0, 1), (1, 2)]))
    fork = dag_to_cpdag(mat(3, directed=[(1, 0), (1, 2)]))
    assert np.array_equal(chain, fork)


def test_complete_pdag_is_idempotent():
    C = dag_to_cpdag(mat(4, directed=[(0, 1), (0, 2), (1, 3), (2, 3)]))
    assert np.array_equal(complete_pdag(C), C)


def test_complete_pdag_reports_failure_on_unextendable_input():
    C = mat(4, directed=[(0, 3), (1, 2)], undirected=[(2, 3)])
    assert complete_pdag(C) is None
