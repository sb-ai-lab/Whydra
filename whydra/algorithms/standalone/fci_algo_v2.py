from __future__ import annotations
"""
FCI Stable Algorithm v2.

Changes from v1:
  1. Bug fixes in orientation rules R1, R2, R5, R10 (see inline comments marked [BUGFIX]).
  2. pdsep parallelized via joblib (the main bottleneck in v1).
  3. New entry point: fci_stable_v2() accepts n_jobs and passes it through to pdsep.
  4. pds_path optimization (biconnected-component restriction) for smaller Possible-D-SEP sets.

Compatible with the same CausalGraph / CIT / SkeletonDiscovery interfaces.
"""

import copy
from itertools import combinations
import numpy as np

from joblib import Parallel, delayed

# Импортируем универсальный класс тестов из соседнего файла
from ..graph_core import CIT
from ...background_knowledge import (
    BKOrientationGuard,
    BKPhase,
    MatrixEncoding,
    apply_bk_checkpoint,
)

# --- Вспомогательные функции ---
from .skeleton import SkeletonDiscovery
from .pag_rules import PAG_ARROW, PAG_CIRCLE, PAG_TAIL, faith_check, minDiscPath, minUncovCircPath, minUncovPdPath, powerset, qreach


# =============================================================================
#  Утилиты (без изменений)
# =============================================================================

def checkTriple(a, b, c, nbrsA, nbrsC, sepsetA, sepsetC, suffStat, alpha, indepTest, maj_rule=True):
    nr_indep = 0
    temp = []

    if len(nbrsA) > 0:
        for s in powerset(nbrsA):
            pval = indepTest(suffStat, a, c, list(s))
            if pval >= alpha:
                nr_indep += 1
                temp.append(b in s)

    if len(nbrsC) > 0:
        for s in powerset(nbrsC):
            pval = indepTest(suffStat, a, c, list(s))
            if pval >= alpha:
                nr_indep += 1
                temp.append(b in s)

    newsepsetA = set(sepsetA) if sepsetA is not None else set()
    newsepsetC = set(sepsetC) if sepsetC is not None else set()

    if len(temp) == 0:
        temp.append(False)

    res = 3
    if maj_rule:
        if sum(temp) / len(temp) < .5:
            res = 1
            if b in newsepsetA: newsepsetA.remove(b)
            if b in newsepsetC: newsepsetC.remove(b)
        elif sum(temp) / len(temp) > .5:
            res = 2
            newsepsetA.add(b)
            newsepsetC.add(b)
    else:
        if sum(temp) / len(temp) == 0:
            res = 1
            if b in newsepsetA: newsepsetA.remove(b)
            if b in newsepsetC: newsepsetC.remove(b)
        elif sum(temp) / len(temp) == 1:
            res = 2
            newsepsetA.add(b)
            newsepsetC.add(b)

    return res, {'sepsetA': newsepsetA, 'sepsetC': newsepsetC}


def pc_cons_intern(graphDict, suffstat, alpha, indepTest, version_unf=(None, None), maj_rule=True,
                   verbose=False):
    sk = graphDict['sk']
    p = sk.shape[0]

    if np.any(sk):
        ind = np.transpose(np.where(sk == PAG_CIRCLE))
        ind = ind[ind[:, 1].argsort()]

        tripleMatrix = []
        for a, b in ind:
            for c in range(p):
                if a < c and sk[a, c] == 0 and sk[b, c] == PAG_CIRCLE:
                    tripleMatrix.append((a, b, c))

        for a, b, c in tripleMatrix:
            nbrsA = np.where(sk[:, a] == PAG_CIRCLE)[0]
            nbrsC = np.where(sk[:, c] == PAG_CIRCLE)[0]

            res, r_abc = checkTriple(a, b, c, nbrsA, nbrsC, graphDict['sepset'][(a, c)],
                                     graphDict['sepset'][(c, a)],
                                     suffstat, alpha, indepTest, maj_rule=maj_rule)
            if res == 3:
                if 'unfTriples' in graphDict.keys():
                    graphDict['unfTriples'].add((a, b, c))
                else:
                    graphDict['unfTriples'] = {(a, b, c)}

            graphDict['sepset'][(a, c)] = r_abc['sepsetA']
            graphDict['sepset'][(c, a)] = r_abc['sepsetC']

    return graphDict


# =============================================================================
#  Функции поиска путей (без изменений)
# =============================================================================

# =============================================================================
#  Biconnected components (для pds_path оптимизации)
# =============================================================================

def _find_biconnected_components(adj_matrix):
    """
    Находит биконнектные компоненты неориентированного скелета.
    Возвращает список множеств рёбер (i, j) для каждой компоненты.
    """
    p = adj_matrix.shape[0]
    # Строим скелет (неориентированный)
    skeleton = ((adj_matrix != 0) | (adj_matrix.T != 0))
    np.fill_diagonal(skeleton, False)

    visited = [False] * p
    disc = [0] * p
    low = [0] * p
    parent = [-1] * p
    timer = [0]
    stack = []  # стек рёбер
    components = []  # список биконнектных компонент (множеств рёбер)

    def _dfs(u):
        children = 0
        visited[u] = True
        disc[u] = low[u] = timer[0]
        timer[0] += 1

        for v in range(p):
            if not skeleton[u, v]:
                continue
            if not visited[v]:
                children += 1
                parent[v] = u
                stack.append((u, v))
                _dfs(v)
                low[u] = min(low[u], low[v])

                # Articulation point check -> extract component
                if (parent[u] == -1 and children > 1) or (parent[u] != -1 and low[v] >= disc[u]):
                    component = set()
                    while stack and stack[-1] != (u, v):
                        edge = stack.pop()
                        component.add((min(edge), max(edge)))
                    if stack:
                        edge = stack.pop()
                        component.add((min(edge), max(edge)))
                    components.append(component)

            elif v != parent[u] and disc[v] < disc[u]:
                stack.append((u, v))
                low[u] = min(low[u], disc[v])

    for i in range(p):
        if not visited[i]:
            _dfs(i)
            # Remaining edges on stack form last component
            if stack:
                component = set()
                while stack:
                    edge = stack.pop()
                    component.add((min(edge), max(edge)))
                components.append(component)

    return components


def _build_edge_to_component_map(components):
    """Строит маппинг (i,j) -> индекс компоненты."""
    edge_map = {}
    for idx, comp in enumerate(components):
        for edge in comp:
            edge_map[edge] = idx
    return edge_map


def _nodes_in_component(component):
    """Возвращает множество узлов, участвующих в компоненте."""
    nodes = set()
    for (i, j) in component:
        nodes.add(i)
        nodes.add(j)
    return nodes


# qreach_path (per-edge вариант Colombo Def. 3.4) удалён: он не вызывался,
# а pdsep_v2 берёт объединение per-edge ограничений по всем соседям x.
# Оптимизация из-за этого слабее документированной — см. комментарий там.


# =============================================================================
#  pdsep v2 — параллельная версия
# =============================================================================

def _pdsep_worker(x, adj_x, allPdsep_x, indepTest, suffStat,
                  alpha, pMax_row, m_max, protected_edges=frozenset()):
    """
    Worker для параллельного pdsep.
    Обрабатывает один узел x: для каждого его соседа y проверяет CI-тесты
    по подмножествам Possible-D-SEP.

    ``protected_edges`` — множество ``frozenset({x, y})``, которые требует
    background knowledge. Такие пары пропускаются целиком: ребро всё равно
    не будет удалено, поэтому CI-тесты по ним — чистая трата времени, а
    записанный sepset для остающейся смежной пары только мешал бы ориентации.

    Возвращает:
      - edge_removals: [(x, y), ...]
      - sepset_updates: [(x, y, S), ...]
      - pmax_updates: [(x, y, pval), ...]
      - tests_count: int
    """
    edge_removals = []
    sepset_updates = []
    pmax_updates = []
    tests_count = 0

    tf1 = [i for i in allPdsep_x if i != x]

    for y in adj_x:
        if frozenset((int(x), int(y))) in protected_edges:
            continue
        tf = [i for i in tf1 if i != y]
        diff_set = [i for i in tf if i not in adj_x]

        if len(diff_set) > 0:
            done = False
            ord_val = 0
            current_pmax = pMax_row[y]

            while not done and ord_val < min(len(tf), m_max):
                ord_val += 1
                if ord_val == 1:
                    for S in diff_set:
                        pval = indepTest(suffStat, x, y, [S])
                        tests_count += 1
                        if pval > current_pmax:
                            current_pmax = pval
                            pmax_updates.append((x, y, pval))
                        if pval >= alpha:
                            edge_removals.append((x, y))
                            sepset_updates.append((x, y, {S}))
                            done = True
                            break
                else:
                    for S in combinations(tf, ord_val):
                        if not set(S).issubset(set(adj_x)):
                            pval = indepTest(suffStat, x, y, list(S))
                            tests_count += 1
                            if pval > current_pmax:
                                current_pmax = pval
                                pmax_updates.append((x, y, pval))
                            if pval >= alpha:
                                edge_removals.append((x, y))
                                sepset_updates.append((x, y, set(S)))
                                done = True
                                break

    return edge_removals, sepset_updates, pmax_updates, tests_count


def pdsep_v2(skel, suffStat, indepTest, p, sepSet, alpha, pMax,
             m_max=float('inf'), pdsep_max=float('inf'), unfVect=None,
             n_jobs=1, verbose=False, use_pds_path=True, protected_edges=()):
    """
    Parallelized Possible-D-SEP step for FCI.

    Key differences from v1:
      1. The Possible-D-SEP sets and the amat snapshot are computed BEFORE
         the main loop (stable approach — no mutations during iteration).
      2. The per-node work is dispatched to parallel workers via joblib.
      3. Results are aggregated after all workers finish: edge removals,
         sepset updates, pMax updates are applied in bulk.
      4. Optional pds_path optimization (biconnected component restriction).
    """
    if unfVect is None:
        unfVect = set()
    protected_edges = {
        frozenset((int(x), int(y))) for x, y in protected_edges
    }

    G = skel['sk'].astype(int)
    amat = np.copy(G)

    # --- Ориентация v-структур на amat (как в v1) ---
    ind = []
    for i in range(len(G)):
        for j in range(len(G[i])):
            if G[i][j] == PAG_CIRCLE:
                ind.append((i, j))
    ind = sorted(ind, key=lambda x: (x[1], x[0]))

    for x, y in ind:
        allZ = [i for i in range(len(amat[y])) if amat[y][i] != 0 and i != x]
        for z in allZ:
            if amat[x][z] == 0 and not (y in sepSet[(x, z)] or y in sepSet[(z, x)]):
                if len(unfVect) == 0:
                    amat[x][y] = amat[z][y] = PAG_ARROW
                else:
                    if (x, y, z) not in unfVect and (z, y, x) not in unfVect:
                        amat[x][y] = amat[z][y] = PAG_ARROW

    # --- Вычисляем Possible-D-SEP для всех узлов (ДО параллельного цикла) ---
    if use_pds_path:
        components = _find_biconnected_components(amat)
        edge_to_comp = _build_edge_to_component_map(components)
        # Для pds_path: храним per-edge PDS, вычислим ниже
        allPdsep_full = [qreach(x, amat) for x in range(p)]
    else:
        allPdsep_full = [qreach(x, amat) for x in range(p)]

    # --- Снимок amat для стабильного подхода ---
    amat_snapshot = np.copy(amat)

    # --- Подготовка данных для каждого worker'а ---
    worker_args = []
    for x in range(p):
        an0 = [amat_snapshot[x][i] != 0 for i in range(p)]
        adj_x = [i for i in range(p) if an0[i]]

        if use_pds_path:
            # Для каждого узла собираем объединение pds_path по всем его соседям
            pds_union = set()
            for y in adj_x:
                edge_key = (min(x, y), max(x, y))
                comp_idx = edge_to_comp.get(edge_key, None)
                if comp_idx is not None:
                    comp_nodes = _nodes_in_component(components[comp_idx])
                    pds_union.update(n for n in allPdsep_full[x] if n in comp_nodes)
                else:
                    pds_union.update(allPdsep_full[x])
            allPdsep_x = sorted(list(pds_union))
        else:
            allPdsep_x = allPdsep_full[x]

        worker_args.append((x, adj_x, allPdsep_x, indepTest,
                            suffStat, alpha, pMax[x].copy(), m_max,
                            protected_edges))

    # --- Параллельное выполнение ---
    if n_jobs == 1:
        results_list = [_pdsep_worker(*args) for args in worker_args]
    else:
        if verbose:
            print(f"  [pdsep_v2] Launching parallel with n_jobs={n_jobs}")
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(_pdsep_worker)(*args) for args in worker_args
        )

    # --- Агрегация результатов ---
    total_tests = 0
    max_ord = 0

    for edge_removals, sepset_updates, pmax_updates, tests_count in results_list:
        total_tests += tests_count

        for (x, y, pval) in pmax_updates:
            if pMax[x, y] < pval:
                pMax[x, y] = pval

        for (x, y) in edge_removals:
            if frozenset((int(x), int(y))) in protected_edges:
                continue
            amat[x][y] = amat[y][x] = 0

        for (x, y, S) in sepset_updates:
            sepSet[(x, y)] = S
            sepSet[(y, x)] = S

    if verbose:
        print(f"  [pdsep_v2] Total CI tests: {total_tests}")

    # Обновляем G на основе amat
    for i in range(p):
        for j in range(p):
            G[i][j] = True if amat[i][j] != 0 else False

    return {'G': G, "sepset": sepSet, "pMax": pMax, "allPdsep": [], "max_ord": max_ord}


# =============================================================================
#  udag2pag v2 — с исправленными багами
# =============================================================================

def udag2pag_v2(pag, sepset, p, unfVect=None, rules=None, orientCollider=True,
                background_knowledge=None, node_names=None):
    """
    Ориентация PAG (R0 + R1..R10) с исправленными правилами R1/R2/R5/R10.

    ``background_knowledge`` применяется в четырёх точках, как и в pag_rules:
      * до R0                                — фаза BETWEEN_ORIENTATION;
      * после ориентации коллайдеров (R0)    — фаза BETWEEN_ORIENTATION;
      * после каждого прохода правил R1..R10 — фаза BETWEEN_ORIENTATION;
      * после сходимости цикла               — фаза POST_ORIENTATION.

    Кроме того, каждая запись метки идёт через ``BKOrientationGuard``: одних
    чекпоинтов недостаточно, потому что R0–R10 успевают переписать конец
    required-ребра, а откатить это чекпоинт может уже не всегда — если
    восстановление направления замыкает цикл, транзакция откатывается целиком
    и ограничение остаётся нарушенным.

    Матрица здесь в кодировке causal-learn (0=нет, -1=tail, 1=arrow, 2=circle),
    поэтому чекпойнтам передаётся ``MatrixEncoding.STANDARD``.
    Если ``background_knowledge is None``, guard пустой и ничего не запрещает,
    все чекпоинты — no-op, и результат побитово совпадает с прежним поведением.
    """
    if rules is None: rules = [True] * 10
    if unfVect is None: unfVect = set()
    if node_names is None:
        node_names = list(range(len(pag)))

    pag = np.array(pag, dtype=int)
    skel_mask = (pag != 0)
    pag[skel_mask] = PAG_CIRCLE

    _bk = (
        background_knowledge.orientation_guard(
            node_names, phase=BKPhase.BETWEEN_ORIENTATION, encoding=MatrixEncoding.STANDARD
        )
        if background_knowledge is not None
        else BKOrientationGuard(None, MatrixEncoding.STANDARD)
    )

    # BK применяется ДО R0: правила должны стартовать с графа, который уже
    # уважает ограничения, иначе R0 расставит стрелки поверх них.
    apply_bk_checkpoint(
        background_knowledge,
        pag,
        node_names,
        phase=BKPhase.BETWEEN_ORIENTATION,
        checkpoint="before_orientation",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="pag",
    )

    before_r0 = pag.copy()
    if orientCollider:
        ind = []
        for i in range(len(pag)):
            for j in range(len(pag[i])):
                if pag[i][j] == PAG_CIRCLE:
                    ind.append((i, j))
        ind = sorted(ind, key=lambda x: (x[1], x[0]))
        for x, y in ind:
            allZ = [i for i in range(len(pag[y])) if pag[y][i] != 0 and i != x]
            for z in allZ:
                if pag[x][z] == 0 and not (y in sepset.get((x, z), []) or y in sepset.get((z, x), [])):
                    if len(unfVect) == 0:
                        _bk.set_mark(pag, x, y, PAG_ARROW)
                        _bk.set_mark(pag, z, y, PAG_ARROW)
                    else:
                        if (x, y, z) not in unfVect and (z, y, x) not in unfVect:
                            _bk.set_mark(pag, x, y, PAG_ARROW)
                            _bk.set_mark(pag, z, y, PAG_ARROW)

    apply_bk_checkpoint(
        background_knowledge,
        pag,
        node_names,
        phase=BKPhase.BETWEEN_ORIENTATION,
        checkpoint="after_colliders",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="pag",
        previous=before_r0 if orientCollider else None,
    )

    old_pag1 = None
    while not np.array_equal(old_pag1, pag):
        old_pag1 = copy.deepcopy(pag)

        # R1
        # [BUGFIX] v1 had inverted unfVect logic: `if len(unfVect) != 0` oriented
        #          unconditionally, while `else` branch checked unfVect (backwards).
        #          Fixed: when unfVect is EMPTY -> orient unconditionally.
        #                 when unfVect is NON-EMPTY -> check membership first.
        if rules[0]:
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == PAG_ARROW and pag[j][i] != 0:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for a, b in ind:
                indC = [i for i in range(len(pag)) if
                        pag[b][i] != 0 and pag[i][b] == PAG_CIRCLE and pag[a][i] == 0 and pag[i][a] == 0 and i != a]
                if len(indC) != 0:
                    if len(unfVect) == 0:
                        # [BUGFIX] was `!= 0` in v1 — inverted
                        for c in indC:
                            _bk.set_mark(pag, b, c, PAG_ARROW)
                            _bk.set_mark(pag, c, b, PAG_TAIL)
                    else:
                        for c in indC:
                            if (a, b, c) not in unfVect and (c, b, a) not in unfVect:
                                _bk.set_mark(pag, b, c, PAG_ARROW)
                                _bk.set_mark(pag, c, b, PAG_TAIL)

        # R2
        # [BUGFIX] v1 had `pag[i][1]` (hardcoded index 1) instead of `pag[i][a]`.
        if rules[1]:
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == PAG_CIRCLE and pag[j][i] != 0:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for a, c in ind:
                indB = [i for i in range(len(pag)) if
                        (pag[a][i] == PAG_ARROW and pag[i][a] == PAG_TAIL and pag[c][i] != 0 and pag[i][c] == PAG_ARROW) or
                        (pag[a][i] == PAG_ARROW and pag[i][a] != 0 and pag[c][i] == PAG_TAIL and pag[i][c] == PAG_ARROW)]
                #                              ^^^^^^ [BUGFIX] was pag[i][1] in v1
                if len(indB) > 0:
                    _bk.set_mark(pag, a, c, PAG_ARROW)

        # R3
        if rules[2]:
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] != 0 and pag[j][i] == PAG_CIRCLE:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for b, d in ind:
                indAC = [i for i in range(len(pag)) if
                         pag[b][i] != 0 and pag[i][b] == PAG_ARROW and pag[i][d] == PAG_CIRCLE and pag[d][i] != 0]
                if len(indAC) >= 2:
                    if len(unfVect) == 0:
                        counter = -1
                        while counter < len(indAC) - 1 and pag[d][b] != PAG_ARROW:
                            counter += 1
                            ii = counter
                            while ii < len(indAC) - 1 and pag[d][b] != PAG_ARROW:
                                ii += 1
                                if pag[indAC[counter]][indAC[ii]] == 0 and pag[indAC[ii]][indAC[counter]] == 0:
                                    _bk.set_mark(pag, d, b, PAG_ARROW)
                    else:
                        for a, c in combinations(indAC, 2):
                            if pag[a][c] == 0 and pag[c][a] == 0 and c != a:
                                if (a, b, c) not in unfVect and (c, b, a) not in unfVect:
                                    _bk.set_mark(pag, d, b, PAG_ARROW)

        # R4
        if rules[3]:
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] != 0 and pag[j][i] == PAG_CIRCLE:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))

            while len(ind) > 0:
                b, c = ind[0]
                ind = ind[1:]
                indA = [i for i in range(len(pag)) if
                        pag[b][i] == PAG_ARROW and pag[i][b] != 0 and pag[c][i] == PAG_TAIL and pag[i][c] == PAG_ARROW]

                while len(indA) > 0 and pag[c][b] == PAG_CIRCLE:
                    a = indA[0]
                    indA = indA[1:]
                    done = False
                    while done == False and pag[a][b] != 0 and pag[a][c] != 0 and pag[b][c] != 0:
                        md_path = minDiscPath(pag, a, b, c)
                        # minDiscPath возвращает [] когда пути нет. Проверка
                        # `== 1` пропускала пустой путь в else, и он падал на
                        # md_path[0]. Тот же дефект был в v1 и в pag_rules.
                        if len(md_path) == 0:
                            done = True
                        else:
                            if b in sepset.get((md_path[0], md_path[-1]), []) or b in sepset.get(
                                    (md_path[-1], md_path[0]), []):
                                _bk.set_mark(pag, b, c, PAG_ARROW)
                                _bk.set_mark(pag, c, b, PAG_TAIL)
                            else:
                                _bk.set_mark(pag, a, b, PAG_ARROW)
                                _bk.set_mark(pag, b, c, PAG_ARROW)
                                _bk.set_mark(pag, c, b, PAG_ARROW)
                            done = True

        # R5
        # [BUGFIX] v1 had `pag[c][d] = pag[c][d] = 3` (no-op) instead of
        #          `pag[c][d] = pag[d][c] = 3`.
        if rules[4]:
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == PAG_CIRCLE and pag[j][i] == PAG_CIRCLE:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            while len(ind) > 0:
                a, b = ind[0]
                ind = ind[1:]
                indC = [i for i in range(len(pag)) if
                        pag[a][i] == PAG_CIRCLE and pag[i][a] == PAG_CIRCLE and pag[b][i] == 0 and pag[i][b] == 0 and i != b]
                indD = [i for i in range(len(pag)) if
                        pag[b][i] == PAG_CIRCLE and pag[i][b] == PAG_CIRCLE and pag[a][i] == 0 and pag[i][a] == 0 and i != a]
                if len(indD) > 0 and len(indC) > 0:
                    counterC = -1
                    while counterC < len(indC) - 1 and pag[a][b] == PAG_CIRCLE:
                        counterC += 1
                        c = indC[counterC]
                        counterD = -1
                        while counterD < len(indD) - 1 and pag[a][b] == PAG_CIRCLE:
                            counterD += 1
                            d = indD[counterD]
                            if pag[c][d] == PAG_CIRCLE and pag[d][c] == PAG_CIRCLE:
                                if len(unfVect) == 0:
                                    _bk.set_mark(pag, a, b, PAG_TAIL)
                                    _bk.set_mark(pag, b, a, PAG_TAIL)
                                    _bk.set_mark(pag, a, c, PAG_TAIL)
                                    _bk.set_mark(pag, c, a, PAG_TAIL)
                                    # [BUGFIX] was pag[c][d] = pag[c][d] = 3
                                    _bk.set_mark(pag, c, d, PAG_TAIL)
                                    _bk.set_mark(pag, d, c, PAG_TAIL)
                                    _bk.set_mark(pag, d, b, PAG_TAIL)
                                    _bk.set_mark(pag, b, d, PAG_TAIL)
                                else:
                                    path2check = [a, c, d, b]
                                    if faith_check(path2check, unfVect, p):
                                        _bk.set_mark(pag, a, b, PAG_TAIL)
                                        _bk.set_mark(pag, b, a, PAG_TAIL)
                                        _bk.set_mark(pag, a, c, PAG_TAIL)
                                        _bk.set_mark(pag, c, a, PAG_TAIL)
                                        # [BUGFIX] was pag[c][d] = pag[c][d] = 3
                                        _bk.set_mark(pag, c, d, PAG_TAIL)
                                        _bk.set_mark(pag, d, c, PAG_TAIL)
                                        _bk.set_mark(pag, d, b, PAG_TAIL)
                                        _bk.set_mark(pag, b, d, PAG_TAIL)
                            else:
                                ucp = minUncovCircPath(p, pag=pag, path=(a, c, d, b), unfVect=unfVect)
                                if len(ucp) > 1:
                                    _bk.set_mark(pag, ucp[0], ucp[-1], PAG_TAIL)
                                    _bk.set_mark(pag, ucp[-1], ucp[0], PAG_TAIL)
                                    for j in range(len(ucp) - 1):
                                        _bk.set_mark(pag, ucp[j], ucp[j + 1], PAG_TAIL)
                                        _bk.set_mark(pag, ucp[j + 1], ucp[j], PAG_TAIL)

        # R6
        if rules[5]:
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] != 0 and pag[j][i] == PAG_CIRCLE:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))

            for b, c in ind:
                if len([i for i in range(len(pag)) if pag[b][i] == PAG_TAIL and pag[i][b] == PAG_TAIL]) > 0:
                    _bk.set_mark(pag, c, b, PAG_TAIL)

        # R7
        if rules[6]:
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] != 0 and pag[j][i] == PAG_CIRCLE:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for b, c in ind:
                indA = [i for i in range(len(pag)) if
                        pag[b][i] == PAG_TAIL and pag[i][b] == PAG_CIRCLE and pag[c][i] == 0 and pag[i][c] == 0 and i != c]
                if len(indA) > 0:
                    if len(unfVect) == 0:
                        _bk.set_mark(pag, c, b, PAG_TAIL)
                    else:
                        for a in indA:
                            if (a, b, c) not in unfVect and (c, b, a) not in unfVect:
                                _bk.set_mark(pag, c, b, PAG_TAIL)

        # R8
        if rules[7]:
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == PAG_ARROW and pag[j][i] == PAG_CIRCLE:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for a, c in ind:
                indB = [i for i in range(len(pag)) if
                        pag[i][a] == PAG_TAIL and (pag[a][i] == PAG_ARROW or pag[a][i] == PAG_CIRCLE) and pag[c][i] == PAG_TAIL and pag[i][c] == PAG_ARROW]
                if len(indB) > 0:
                    _bk.set_mark(pag, c, a, PAG_TAIL)

        # R9
        if rules[8]:
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == PAG_ARROW and pag[j][i] == PAG_CIRCLE:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))

            while len(ind) > 0:
                a, c = ind[0]
                ind = ind[1:]
                indB = [i for i in range(len(pag)) if
                        (pag[a][i] == PAG_ARROW or pag[a][i] == PAG_CIRCLE) and
                        (pag[i][a] == PAG_CIRCLE or pag[i][a] == PAG_TAIL) and
                        (pag[c][i] == 0 and pag[i][c] == 0) and
                        i != c]
                while len(indB) > 0 and pag[c][a] == PAG_CIRCLE:
                    b = indB[0]
                    indB = indB[1:]
                    upd = minUncovPdPath(p, pag, a, b, c, unfVect=unfVect)
                    if len(upd) > 1:
                        _bk.set_mark(pag, c, a, PAG_TAIL)

        # R10
        # [BUGFIX] v1 used stale variable `c` from R9 scope instead of correct
        #          variable `a` (the iteration variable is (a, c) but R10 iterates
        #          over (a, b) pairs with a->b having arrow at a and circle at b).
        #          The entire R10 block was broken. Fixed to match Zhang (2008).
        if rules[9]:
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == PAG_ARROW and pag[j][i] == PAG_CIRCLE:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            while len(ind) > 0:
                a, c = ind[0]
                ind = ind[1:]
                # [BUGFIX] was `pag[c][i]` referencing stale `c` from R9.
                # R10: a *-> c  with  c o-* a  (pag[a][c]==2, pag[c][a]==1)
                # Find b, d such that b -> c <- d (both arrows into c)
                indB_r10 = [i for i in range(p) if pag[i][c] == PAG_ARROW and pag[c][i] == PAG_TAIL]
                if len(indB_r10) >= 2:
                    counterB = -1
                    while counterB < len(indB_r10) - 1 and pag[c][a] == PAG_CIRCLE:
                        counterB += 1
                        b = indB_r10[counterB]
                        indD = [i for i in indB_r10 if i != b]
                        counterD = -1
                        while counterD < len(indD) - 1 and pag[c][a] == PAG_CIRCLE:
                            counterD += 1
                            d = indD[counterD]
                            if (
                                    (pag[a][b] == PAG_CIRCLE or pag[a][b] == PAG_ARROW) and
                                    (pag[b][a] == PAG_CIRCLE or pag[b][a] == PAG_TAIL) and
                                    (pag[a][d] == PAG_CIRCLE or pag[a][d] == PAG_ARROW) and
                                    (pag[d][a] == PAG_CIRCLE or pag[d][a] == PAG_TAIL) and
                                    pag[d][b] == 0 and pag[b][d] == 0
                            ):
                                if len(unfVect) == 0:
                                    _bk.set_mark(pag, c, a, PAG_TAIL)
                                else:
                                    if (b, a, d) not in unfVect and (d, a, b) not in unfVect:
                                        _bk.set_mark(pag, c, a, PAG_TAIL)
                            else:
                                indX = [i for i in range(p) if
                                        (pag[a][i] == PAG_CIRCLE or pag[a][i] == PAG_ARROW) and
                                        (pag[i][a] == PAG_CIRCLE or pag[i][a] == PAG_TAIL) and
                                        i != c]
                                if len(indX) >= 2:
                                    counterX1 = -1
                                    while counterX1 < len(indX) - 1 and pag[c][a] == PAG_CIRCLE:
                                        counterX1 += 1
                                        first_pos = indX[counterX1]
                                        indX2 = [i for i in indX if i != first_pos]
                                        counterX2 = -1
                                        while counterX2 < len(indX2) - 1 and pag[c][a] == PAG_CIRCLE:
                                            counterX2 += 1
                                            sec_pos = indX2[counterX2]
                                            t1 = minUncovPdPath(p, pag, a, first_pos, b, unfVect=unfVect)
                                            if len(t1) > 1:
                                                t2 = minUncovPdPath(p, pag, a, sec_pos, d, unfVect=unfVect)
                                                if len(t2) > 1 and first_pos != sec_pos and pag[first_pos][
                                                    sec_pos] == 0:
                                                    if len(unfVect) == 0:
                                                        _bk.set_mark(pag, c, a, PAG_TAIL)
                                                    elif (first_pos, a, sec_pos) not in unfVect and (
                                                            sec_pos, a, first_pos) not in unfVect:
                                                        _bk.set_mark(pag, c, a, PAG_TAIL)

        apply_bk_checkpoint(
            background_knowledge,
            pag,
            node_names,
            phase=BKPhase.BETWEEN_ORIENTATION,
            checkpoint="after_orientation_rule_pass",
            encoding=MatrixEncoding.STANDARD,
            graph_kind="pag",
            previous=old_pag1,
        )

    apply_bk_checkpoint(
        background_knowledge,
        pag,
        node_names,
        phase=BKPhase.POST_ORIENTATION,
        checkpoint="after_orientation",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="pag",
    )
    return pag


# =============================================================================
#  Entry point: fci_stable_v2
# =============================================================================

def fci_stable_v2(cg, data, alpha=0.05, indep_test='fisherz', n_jobs=1,
                  verbose=False, use_pds_path=True, background_knowledge=None,
                  show_progress: bool = False):
    """
    FCI Stable Algorithm v2.

    Improvements over v1:
      - Bug fixes in orientation rules R1, R2, R5, R10.
      - pdsep step is parallelized via joblib (the main bottleneck).
      - Optional pds_path optimization for smaller Possible-D-SEP sets.

    Parameters
    ----------
    cg : CausalGraph
        Causal graph. The CI test is set inside from ``data``/``indep_test``,
        so all phases (skeleton, pdsep, udag2pag) use the same test.
    data : np.ndarray
        Data matrix (n_rows x n_features).
    alpha : float
        Significance level for independence tests.
    indep_test : str
        Independence test method (e.g. 'fisherz').
    n_jobs : int
        Number of parallel jobs. 1 = sequential, -1 = all cores.
    verbose : bool
        Print progress information.
    use_pds_path : bool
        If True, use biconnected-component restriction on Possible-D-SEP
        (Definition 3.4 from Colombo et al. 2012). Reduces search space.
    background_knowledge : BackgroundKnowledge, optional
        Фоновые знания о направлениях рёбер. Применяются на трёх фазах:
        PRE_SEARCH (до поиска скелета — оттуда же берутся обязательные рёбра,
        которые запрещено удалять на шагах skeleton и pdsep),
        BETWEEN_ORIENTATION (после коллайдеров и после каждого прохода правил)
        и POST_ORIENTATION (после сходимости). ``None`` — полный no-op.

    Returns
    -------
    pag : np.ndarray
        PAG adjacency matrix.
    """
    # CI-тест выставляем здесь, а не полагаемся на вызывающего: иначе скелет
    # считался бы тестом, который поставил вызывающий, а pdsep и udag2pag —
    # тестом из indep_test, и на дискретных данных это разные тесты.
    cit = CIT(data, method=indep_test)
    cg.set_ind_test(cit)

    node_names = [node.name for node in cg.nodes]
    apply_bk_checkpoint(
        background_knowledge,
        cg.G.graph,
        node_names,
        phase=BKPhase.PRE_SEARCH,
        checkpoint="before_skeleton",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="pag",
    )
    protected_edges = ()
    if background_knowledge is not None:
        protected_edges = background_knowledge.required_pairs(
            node_names, phase=BKPhase.PRE_SEARCH
        )

    # Step 1: Skeleton discovery (parallelized in SkeletonDiscovery)
    if verbose:
        print("[FCI v2] Step 1: Skeleton discovery...")
    graphDict = SkeletonDiscovery(
        cg, alpha, n_jobs=n_jobs, verbose=verbose, show_progress=show_progress,
        protected_edges=protected_edges
    ).run_fci()

    n, p = data.shape

    def test_func_wrapper(suffStat, x, y, S):
        return cit(x, y, S)

    suffStat = None

    # Step 2: Conservative v-structure orientation
    if verbose:
        print("[FCI v2] Step 2: Conservative v-structure orientation...")
    pc_cons_intern(graphDict, suffStat, alpha, test_func_wrapper, maj_rule=False)

    # Step 3: Possible-D-SEP (PARALLELIZED)
    if verbose:
        print(f"[FCI v2] Step 3: Possible-D-SEP (n_jobs={n_jobs})...")
    pdsep_res = pdsep_v2(
        graphDict, suffStat, test_func_wrapper, p,
        graphDict['sepset'], alpha, graphDict['pMax'],
        unfVect=graphDict['unfTriples'],
        n_jobs=n_jobs,
        verbose=verbose,
        use_pds_path=use_pds_path,
        protected_edges=protected_edges,
    )

    # Steps 4-5: Orient v-structures + R1-R10 (with bug fixes)
    if verbose:
        print("[FCI v2] Steps 4-5: Orientation rules...")
    pag = udag2pag_v2(pdsep_res['G'], pdsep_res['sepset'], p,
                      unfVect=graphDict['unfTriples'],
                      background_knowledge=background_knowledge,
                      node_names=node_names)

    return pag
