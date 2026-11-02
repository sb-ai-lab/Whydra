from __future__ import annotations
"""
FCI Stable Algorithm v2.

Конвейер: скелет (``SkeletonDiscovery.run_fci``) -> Possible-D-SEP
(``pdsep_v2``, семантика ``removeByPossibleDsep`` из causal-learn 0.1.4.5,
параллельно через joblib) -> ориентация (``udag2pag_v2``: R0 + R1..R10).

Отличия от v1 (``standalone_fci_stable.py``): другой скелет (sepset как
объединение всех найденных сепараторов, как в FAS causal-learn), нет
консервативного шага ``pc_cons_intern``, Possible-D-SEP считается по ребру
(Tetrad ``getPossibleDsep``), а не по узлу (pcalg ``qreach``), рёбра на этом
шаге удаляются немедленно, параллельная часть даёт результат, не зависящий
от ``n_jobs``.

Compatible with the same CausalGraph / CIT / SkeletonDiscovery interfaces.
"""

import copy
import time
from collections import deque
from itertools import combinations
import numpy as np

from joblib import Parallel, delayed

from ..graph_core import CIT
from ...background_knowledge import (
    BKOrientationGuard,
    BKPhase,
    MatrixEncoding,
    apply_bk_checkpoint,
)

from .skeleton import SkeletonDiscovery
from .pag_rules import (
    PAG_ARROW,
    PAG_CIRCLE,
    PAG_TAIL,
    faith_check,
    minDiscPath,
    minUncovCircPath,
    minUncovPdPath,
)


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


# =============================================================================
#  Possible-D-SEP по семантике causal-learn (Tetrad)
# =============================================================================

def _cl_node_name_key(i):
    """Ключ сортировки, воспроизводящий порядок узлов causal-learn.

    causal-learn сортирует Possible-D-SEP по ИМЕНАМ узлов (по умолчанию
    'X1', 'X2', ...), лексикографически в обратном порядке. Ключ намеренно
    строится по позиции, а не по реальным именам колонок пользователя: так
    результат whydra не зависит от того, как названы переменные.

    Args:
        i: индекс узла.

    Returns:
        Строка вида 'X{i+1}'.
    """
    return f"X{i + 1}"


def _adjacency_lists(amat):
    """Списки смежности для всех узлов, по возрастанию индексов.

    Args:
        amat: матрица PAG в раскладке whydra (amat[u][v] — метка у узла v
            на ребре u—v); ненулевое значение означает наличие ребра.

    Returns:
        Список из p списков: adj[i] — индексы узлов, смежных с i.
    """
    p = len(amat)
    return [[j for j in range(p) if amat[i][j] != 0] for i in range(p)]


def _is_def_collider(amat, a, b, c):
    """Проверяет, является ли тройка a — b — c определённым коллайдером.

    Соответствует ``graph.is_def_collider(a, b, c)`` из causal-learn:
    у узла b стоят стрелки со стороны обоих соседей, т.е. a *-> b <-* c.

    Args:
        amat: матрица PAG в раскладке whydra.
        a: первый узел тройки.
        b: центральный узел тройки.
        c: третий узел тройки.

    Returns:
        True, если amat[a][b] == PAG_ARROW и amat[c][b] == PAG_ARROW.
    """
    return amat[a][b] == PAG_ARROW and amat[c][b] == PAG_ARROW


def _semi_directed_reachable(amat, adj, src):
    """Множество узлов, достижимых из src полунаправленным путём.

    Порт ``existsSemiDirectedPath`` из causal-learn 0.1.4.5, но за один
    обход возвращает все цели сразу. Шаг из узла n к смежному m разрешён
    тогда и только тогда, когда метка У УЗЛА n (в раскладке whydra это
    amat[m][n]) есть PAG_TAIL или PAG_CIRCLE. Как и в оригинале, сам src
    не считается достижимым, пока до него не дойдёт путь через цикл.

    Args:
        amat: матрица PAG в раскладке whydra.
        adj: списки смежности (см. _adjacency_lists).
        src: индекс стартового узла.

    Returns:
        Множество индексов достижимых узлов.
    """
    visited = set()
    queue = deque()
    for u in adj[src]:
        if amat[u][src] in (PAG_TAIL, PAG_CIRCLE) and u not in visited:
            visited.add(u)
            queue.append(u)
    while queue:
        t = queue.popleft()
        for u in adj[t]:
            if amat[u][t] in (PAG_TAIL, PAG_CIRCLE) and u not in visited:
                visited.add(u)
                queue.append(u)
    return visited


def _exists_semi_directed_path(amat, src, dst, adj=None):
    """Проверяет существование полунаправленного пути src ~> dst.

    Args:
        amat: матрица PAG в раскладке whydra.
        src: индекс стартового узла.
        dst: индекс целевого узла.
        adj: списки смежности; если None — строятся по amat.

    Returns:
        True, если dst достижим из src полунаправленным путём
        (путь нулевой длины путём не считается).
    """
    if adj is None:
        adj = _adjacency_lists(amat)
    return dst in _semi_directed_reachable(amat, adj, src)


def _possible_dsep(amat, x, y, adj=None):
    """Вычисляет Possible-D-SEP(x, y) по определению Tetrad / causal-learn.

    Порт ``getPossibleDsep`` из causal-learn 0.1.4.5 на раскладку whydra.
    BFS ведётся по упорядоченным парам (a, b), засеянным парами (x, b) для
    b из adj(x) \\ {y}; все такие b попадают в результат безусловно. При
    извлечении пары (a, b) для каждого c из adj(b) \\ {a, x, y} сначала
    записывается previous[c].add(b) и лишь затем проверяется условие
    расширения (коллайдер a *-> b <-* c либо смежность a и c).

    Незасеянный узел b попадает в PDS, если в момент извлечения пары (a, b)
    из очереди в previous[b] есть r (не x и не b) с полунаправленным путём
    r ~> x или r ~> b — ровно как в оригинале, где предикат тоже считается
    по частичной карте previous на момент извлечения. Порядок обхода в
    оригинале задаётся Python-множествами (зависит от PYTHONHASHSEED),
    здесь — возрастанием индексов; на случайных графах результат
    референса от порядка не зависел (см. tests/test_fci_v2_pdsep.py).

    Порядок результата — сортировка по ключу _cl_node_name_key в обратном
    порядке: это в точности ``_dsep.sort(reverse=True)`` оригинала, где
    GraphNode сравнивается по имени 'X{i+1}'.

    Args:
        amat: матрица PAG в раскладке whydra.
        x: индекс первого узла ребра.
        y: индекс второго узла ребра.
        adj: списки смежности; если None — строятся по amat.

    Returns:
        Список индексов узлов Possible-D-SEP(x, y).
    """
    if adj is None:
        adj = _adjacency_lists(amat)
    result = set()
    visited = set()
    queue = deque()
    previous = {}
    reach_cache = {}
    for b in adj[x]:
        if b == y:
            continue
        pair = (x, b)
        queue.append(pair)
        visited.add(pair)
        result.add(b)
    while queue:
        a, b = queue.popleft()
        if b not in result:
            for r in sorted(previous.get(b, ())):
                if r == x or r == b:
                    continue
                reach = reach_cache.get(r)
                if reach is None:
                    reach = _semi_directed_reachable(amat, adj, r)
                    reach_cache[r] = reach
                if x in reach or b in reach:
                    result.add(b)
                    break
        for c in adj[b]:
            if c == a or c == x or c == y:
                continue
            previous.setdefault(c, set()).add(b)
            if _is_def_collider(amat, a, b, c) or amat[a][c] != 0:
                pair = (a, c)
                if pair not in visited:
                    visited.add(pair)
                    queue.append(pair)
    result.discard(x)
    result.discard(y)
    return sorted(result, key=_cl_node_name_key, reverse=True)


def _side_order(amat, i, j):
    """Порядок концов ребра (i, j), i < j, как у Edge в causal-learn.

    Конструктор Edge в causal-learn меняет местами node1 и node2, если метка
    у node1 (узла с меньшим индексом) — ARROW, а метка у node2 — TAIL или
    CIRCLE. В раскладке whydra метка у узла i равна amat[j][i].

    Args:
        amat: матрица PAG в раскладке whydra.
        i: меньший индекс ребра.
        j: больший индекс ребра.

    Returns:
        Кортеж (a, b): порядок концов ребра после возможного переворота.
    """
    if amat[j][i] == PAG_ARROW and amat[i][j] in (PAG_CIRCLE, PAG_TAIL):
        return (j, i)
    return (i, j)


def _restrict_to_component(
    pds,
    a,
    b,
    components,
    edge_to_comp,
):
    """Ограничивает Possible-D-SEP двусвязной компонентой ребра (a, b).

    Используется только при use_pds_path=True (Colombo et al. 2012,
    Def. 3.4): узлы вне двусвязной компоненты ребра (a, b) не лежат ни на
    одном пути между a и b и отбрасываются.

    Args:
        pds: список индексов Possible-D-SEP (порядок сохраняется).
        a: первый узел ребра.
        b: второй узел ребра.
        components: список двусвязных компонент
            (см. _find_biconnected_components).
        edge_to_comp: карта ребро -> индекс компоненты
            (см. _build_edge_to_component_map).

    Returns:
        Список индексов из pds, лежащих в компоненте ребра (a, b); если
        ребро не входит ни в одну компоненту — pds без изменений.
    """
    comp_idx = edge_to_comp.get((min(a, b), max(a, b)))
    if comp_idx is None:
        return list(pds)
    allowed = _nodes_in_component(components[comp_idx])
    return [node for node in pds if node in allowed]


def _edge_context(
    amat,
    a,
    b,
    adj,
    components=None,
    edge_to_comp=None,
):
    """Строит контекст проверки ребра a — b для _test_edge.

    Контекст — это всё, от чего зависит результат CI-перебора по ребру:
    PDS с обеих сторон и списки смежности обоих концов. Совпадение двух
    контекстов гарантирует одинаковый результат _test_edge.

    Args:
        amat: матрица PAG в раскладке whydra.
        a: первый узел ребра (в порядке _side_order).
        b: второй узел ребра.
        adj: списки смежности для amat.
        components: список двусвязных компонент или None
            (use_pds_path=False).
        edge_to_comp: карта ребро -> компонента или None.

    Returns:
        Словарь с ключами 'a', 'b', 'pds_a', 'pds_b', 'adj_a', 'adj_b'.
    """
    pds_a = _possible_dsep(amat, a, b, adj)
    pds_b = _possible_dsep(amat, b, a, adj)
    if components is not None:
        pds_a = _restrict_to_component(pds_a, a, b, components, edge_to_comp)
        pds_b = _restrict_to_component(pds_b, a, b, components, edge_to_comp)
    return {
        'a': a,
        'b': b,
        'pds_a': pds_a,
        'pds_b': pds_b,
        'adj_a': list(adj[a]),
        'adj_b': list(adj[b]),
    }


def _search_separator(
    x,
    y,
    pds,
    adj_x,
    adj_y,
    indepTest,
    suffStat,
    alpha,
    m_max,
    pmax_start,
):
    """Перебирает подмножества Possible-D-SEP в поисках сепаратора x ⊥ y.

    Перебор по возрастанию размера от 2 до min(len(pds), m_max)
    включительно, внутри размера — лексикографически по списку pds
    (itertools.combinations). Подмножества S с S ⊆ adj_x или S ⊆ adj_y
    пропускаются, как в causal-learn: они уже проверены на этапе скелета.
    Порог независимости — pval >= alpha (конвенция whydra; в causal-learn
    строго >, расхождение возможно только при pval == alpha).

    Args:
        x: индекс первого узла.
        y: индекс второго узла.
        pds: упорядоченный список индексов Possible-D-SEP(x, y).
        adj_x: список смежных узлов x.
        adj_y: список смежных узлов y.
        indepTest: функция indepTest(suffStat, x, y, S_list) -> p-value.
        suffStat: достаточная статистика для indepTest.
        alpha: уровень значимости.
        m_max: максимальный размер сепаратора.
        pmax_start: стартовое значение максимума p-value для пары.

    Returns:
        Кортеж (sepset, tests, pmax, max_size): sepset — найденный
        сепаратор (set) или None; tests — число CI-тестов; pmax — максимум
        p-value; max_size — наибольший испытанный размер (0, если тестов
        не было).
    """
    tests = 0
    pmax = pmax_start
    max_size = 0
    set_adj_x = set(adj_x)
    set_adj_y = set(adj_y)
    upper = len(pds) if m_max == float('inf') else min(len(pds), int(m_max))
    for size in range(2, upper + 1):
        for S in combinations(pds, size):
            set_S = set(S)
            if set_S <= set_adj_x or set_S <= set_adj_y:
                continue
            pval = indepTest(suffStat, x, y, list(S))
            tests += 1
            if pval > pmax:
                pmax = pval
            if size > max_size:
                max_size = size
            if pval >= alpha:
                return set_S, tests, pmax, max_size
    return None, tests, pmax, max_size


def _test_edge(
    ctx,
    indepTest,
    suffStat,
    alpha,
    m_max,
    pmax_start,
):
    """Проверяет ребро a — b на удаление по Possible-D-SEP с двух сторон.

    Сначала ищется сепаратор со стороны a (x=a, y=b), затем — только если
    ничего не найдено — со стороны b, как в ``removeByPossibleDsep``.

    Args:
        ctx: контекст ребра (результат _edge_context).
        indepTest: функция независимости.
        suffStat: достаточная статистика.
        alpha: уровень значимости.
        m_max: максимальный размер сепаратора.
        pmax_start: стартовый максимум p-value для пары.

    Returns:
        Словарь с ключами 'removed' (bool), 'sepset' (set | None),
        'tests' (int), 'pmax' (float), 'max_ord' (int).
    """
    a = ctx['a']
    b = ctx['b']
    sepset, tests, pmax, max_ord = _search_separator(
        a, b, ctx['pds_a'], ctx['adj_a'], ctx['adj_b'],
        indepTest, suffStat, alpha, m_max, pmax_start,
    )
    if sepset is None:
        sepset, tests_b, pmax_b, max_ord_b = _search_separator(
            b, a, ctx['pds_b'], ctx['adj_b'], ctx['adj_a'],
            indepTest, suffStat, alpha, m_max, pmax_start,
        )
        tests += tests_b
        pmax = max(pmax, pmax_b)
        max_ord = max(max_ord, max_ord_b)
    return {
        'removed': sepset is not None,
        'sepset': sepset,
        'tests': tests,
        'pmax': pmax,
        'max_ord': max_ord,
    }


def _pdsep_edge_worker(
    amat_snapshot,
    i,
    j,
    indepTest,
    suffStat,
    alpha,
    m_max,
    pmax_start,
    components,
    edge_to_comp,
):
    """Единица работы joblib: спекулятивная проверка одного ребра по снимку.

    Args:
        amat_snapshot: копия матрицы PAG на момент старта фазы удалений.
        i: меньший индекс ребра.
        j: больший индекс ребра.
        indepTest: функция независимости.
        suffStat: достаточная статистика.
        alpha: уровень значимости.
        m_max: максимальный размер сепаратора.
        pmax_start: стартовый максимум p-value для пары (i, j).
        components: двусвязные компоненты или None.
        edge_to_comp: карта ребро -> компонента или None.

    Returns:
        Кортеж (ctx, result): контекст ребра и результат _test_edge.
    """
    adj = _adjacency_lists(amat_snapshot)
    a, b = _side_order(amat_snapshot, i, j)
    ctx = _edge_context(amat_snapshot, a, b, adj, components, edge_to_comp)
    result = _test_edge(ctx, indepTest, suffStat, alpha, m_max, pmax_start)
    return ctx, result


def pdsep_v2(
    skel,
    suffStat,
    indepTest,
    p,
    sepSet,
    alpha,
    pMax,
    m_max=float('inf'),
    pdsep_max=float('inf'),
    unfVect=None,
    n_jobs=1,
    verbose=False,
    use_pds_path=False,
    protected_edges=(),
):
    """Шаг Possible-D-SEP алгоритма FCI с семантикой causal-learn 0.1.4.5.

    Воспроизводит ``removeByPossibleDsep`` (Tetrad-style) из causal-learn:
    скелет приводится к o-o, v-структуры ориентируются по sepset (аналог
    rule0), затем для каждого ребра — в порядке (i, j), i < j, с порядком
    концов как у Edge в causal-learn (см. _side_order) — вычисляется
    Possible-D-SEP по ребру (Tetrad ``getPossibleDsep``), перебираются его
    подмножества размера не меньше 2 (подмножества adj(a) и adj(b)
    пропускаются), и при обнаружении независимости ребро удаляется
    НЕМЕДЛЕННО — следующие рёбра видят уже уменьшенный граф. Сначала
    проверяется сторона a, затем (если ничего не найдено) сторона b.

    Намеренно НЕ воспроизводятся два дефекта референса:
    1. causal-learn записывает сепаратор только в ключ sep_sets[(a, b)];
       здесь пишутся оба ключа (a, b) и (b, a).
    2. DepthChoiceGenerator в causal-learn из-за алиасинга списка
       пропускает первое сочетание каждого размера; здесь перебираются
       все сочетания через itertools.combinations.

    Параллельность: спекулятивная фаза + последовательная проверка. При
    n_jobs != 1 все рёбра проверяются параллельно (joblib) на неизменяемом
    снимке графа. Затем идёт последовательный проход в порядке рёбер: для
    каждого ребра контекст (PDS обеих сторон и списки смежности концов)
    пересчитывается на живом графе; если он совпал со спекулятивным,
    берётся готовый результат, иначе ребро проверяется заново. Метки в
    течение шага не меняются (только удаления), а равенство контекстов
    гарантирует равенство входа _test_edge, поэтому результат (граф,
    sepset, pMax) не зависит от n_jobs и совпадает с последовательным
    выполнением.

    use_pds_path — задокументированное отклонение от causal-learn (по
    умолчанию выключено): Possible-D-SEP ограничивается двусвязной
    компонентой проверяемого ребра (Colombo et al. 2012, Def. 3.4);
    компоненты считаются один раз на снимке до фазы удалений.

    pdsep_max сохранён для совместимости сигнатуры и не используется.

    Args:
        skel: результат SkeletonDiscovery; skel['sk'] — матрица скелета,
            любое ненулевое значение означает ребро.
        suffStat: достаточная статистика для indepTest.
        indepTest: функция indepTest(suffStat, x, y, S) -> p-value.
        p: число переменных.
        sepSet: словарь по всем упорядоченным парам (i, j) -> set.
        alpha: уровень значимости.
        pMax: массив p×p float — максимум p-value по парам (обновляется
            симметрично).
        m_max: максимальный размер сепаратора.
        pdsep_max: не используется (совместимость сигнатуры).
        unfVect: множество троек, которые нельзя ориентировать как
            коллайдеры (в v2 всегда пусто).
        n_jobs: число воркеров joblib; на результат не влияет.
        verbose: печать сводки по шагу.
        use_pds_path: ограничивать PDS двусвязной компонентой ребра.
        protected_edges: пары (x, y), которые нельзя удалять (BK).

    Returns:
        Словарь с ключами 'G' (0/1-матрица смежности), 'sepset', 'pMax',
        'allPdsep' (пустой список, совместимость), 'max_ord' (наибольший
        испытанный размер сепаратора).
    """
    if unfVect is None:
        unfVect = set()
    protected = {frozenset((int(x), int(y))) for x, y in protected_edges}

    amat = np.where(np.asarray(skel['sk']) != 0, PAG_CIRCLE, 0).astype(int)

    # --- Ориентация v-структур (эквивалент rule0 causal-learn) ---
    ind = [(i, j) for i in range(p) for j in range(p) if amat[i][j] != 0]
    ind = sorted(ind, key=lambda e: (e[1], e[0]))
    for x, y in ind:
        allZ = [i for i in range(len(amat[y])) if amat[y][i] != 0 and i != x]
        for z in allZ:
            if amat[x][z] == 0 and not (y in sepSet[(x, z)] or y in sepSet[(z, x)]):
                if len(unfVect) == 0:
                    amat[x][y] = amat[z][y] = PAG_ARROW
                else:
                    if (x, y, z) not in unfVect and (z, y, x) not in unfVect:
                        amat[x][y] = amat[z][y] = PAG_ARROW

    # --- Снимок рёбер в порядке causal-learn; защищённые не трогаем ---
    edges = [
        (i, j)
        for i in range(p)
        for j in range(i + 1, p)
        if amat[i][j] != 0 and frozenset((i, j)) not in protected
    ]

    components = None
    edge_to_comp = None
    if use_pds_path:
        components = _find_biconnected_components(amat)
        edge_to_comp = _build_edge_to_component_map(components)

    # --- Спекулятивная фаза (параллельная, только при n_jobs != 1) ---
    t_spec = time.perf_counter()
    speculative = {}
    if n_jobs != 1 and edges:
        amat_snapshot = np.copy(amat)
        parallel_results = Parallel(n_jobs=n_jobs)(
            delayed(_pdsep_edge_worker)(
                amat_snapshot, i, j, indepTest, suffStat, alpha, m_max,
                max(pMax[i, j], pMax[j, i]), components, edge_to_comp,
            )
            for (i, j) in edges
        )
        speculative = dict(zip(edges, parallel_results))

    # --- Последовательная проверка и применение ---
    t_valid = time.perf_counter()
    total_tests = 0
    max_ord = 0
    n_removed = 0
    n_speculative_hits = 0
    n_recomputed = 0
    adj = _adjacency_lists(amat)

    for (i, j) in edges:
        a, b = _side_order(amat, i, j)
        ctx_live = _edge_context(amat, a, b, adj, components, edge_to_comp)
        if (i, j) in speculative and speculative[(i, j)][0] == ctx_live:
            result = speculative[(i, j)][1]
            n_speculative_hits += 1
        else:
            result = _test_edge(
                ctx_live, indepTest, suffStat, alpha, m_max,
                max(pMax[i, j], pMax[j, i]),
            )
            n_recomputed += 1

        if result['removed']:
            amat[a][b] = 0
            amat[b][a] = 0
            adj[a] = [k for k in adj[a] if k != b]
            adj[b] = [k for k in adj[b] if k != a]
            sepSet[(a, b)] = set(result['sepset'])
            sepSet[(b, a)] = set(result['sepset'])
            n_removed += 1
        pMax[a, b] = max(pMax[a, b], pMax[b, a], result['pmax'])
        pMax[b, a] = pMax[a, b]
        total_tests += result['tests']
        if result['max_ord'] > max_ord:
            max_ord = result['max_ord']

    if verbose:
        t_end = time.perf_counter()
        print(
            f"[FCI v2] pdsep: edges={len(edges)} removed={n_removed} "
            f"ci_tests={total_tests} speculative_hits={n_speculative_hits} "
            f"recomputed={n_recomputed} n_jobs={n_jobs} "
            f"t_speculative={t_valid - t_spec:.1f}s t_validate={t_end - t_valid:.1f}s"
        )

    G = (amat != 0).astype(int)
    return {'G': G, "sepset": sepSet, "pMax": pMax, "allPdsep": [], "max_ord": max_ord}

# =============================================================================
#  R1 + R2 в порядке causal-learn (rulesR1R2cycle)
# =============================================================================

def _arrow_allowed(pag, x, y):
    """Можно ли поставить стрелку у y на ребре x—y (is_arrow_point_allowed без BK).

    causal-learn не переписывает уже поставленный хвост стрелкой; whydra
    раньше в R1 это делал.

    Args:
        pag: матрица PAG в раскладке whydra (pag[u][v] — метка у v).
        x: узел, со стороны которого ставится стрелка.
        y: узел, у которого ставится стрелка.

    Returns:
        True, если метка у y на ребре x—y не TAIL.
    """
    return pag[x][y] != PAG_TAIL


def _rule_r1(pag, bk, a, b, c, unfVect):
    """R1: a *-> b o-* c, a и c не смежны => b -> c.

    Args:
        pag: матрица PAG в раскладке whydra.
        bk: BKOrientationGuard, через который пишутся метки.
        a: узел со стрелкой в b.
        b: центральный узел тройки.
        c: узел с кружком у b.
        unfVect: тройки, которые нельзя ориентировать (в v2 пусто).
    """
    if pag[a][c] != 0 or pag[a][b] != PAG_ARROW or pag[c][b] != PAG_CIRCLE:
        return
    if not _arrow_allowed(pag, b, c):
        return
    if unfVect and ((a, b, c) in unfVect or (c, b, a) in unfVect):
        return
    bk.set_mark(pag, b, c, PAG_ARROW)
    bk.set_mark(pag, c, b, PAG_TAIL)


def _rule_r2(pag, bk, a, b, c):
    """R2: (a -> b *-> c или a *-> b -> c) и a *-o c => a *-> c.

    Args:
        pag: матрица PAG в раскладке whydra.
        bk: BKOrientationGuard, через который пишутся метки.
        a: начало цепочки.
        b: середина цепочки.
        c: узел с кружком на ребре a—c.
    """
    if pag[a][c] != PAG_CIRCLE:
        return
    if pag[a][b] == PAG_ARROW and pag[b][c] == PAG_ARROW and (
            pag[b][a] == PAG_TAIL or pag[c][b] == PAG_TAIL):
        bk.set_mark(pag, a, c, PAG_ARROW)


def _rules_r1_r2_cycle(
    pag,
    bk,
    unfVect,
    apply_r1,
    apply_r2,
):
    """Один проход R1 и R2 в порядке causal-learn ``rulesR1R2cycle``.

    По узлам b по возрастанию индекса, по парам его соседей (a, c), a < c,
    применяются R1(a,b,c), R1(c,b,a), R2(a,b,c), R2(c,b,a). Список соседей
    берётся один раз на узел, как в оригинале.

    Args:
        pag: матрица PAG в раскладке whydra (меняется на месте).
        bk: BKOrientationGuard, через который пишутся метки.
        unfVect: тройки, которые нельзя ориентировать (в v2 пусто).
        apply_r1: применять ли R1.
        apply_r2: применять ли R2.
    """
    p = len(pag)
    for b in range(p):
        adj = [i for i in range(p) if pag[b][i] != 0]
        for ia in range(len(adj)):
            for ic in range(ia + 1, len(adj)):
                a, c = adj[ia], adj[ic]
                if apply_r1:
                    _rule_r1(pag, bk, a, b, c, unfVect)
                    _rule_r1(pag, bk, c, b, a, unfVect)
                if apply_r2:
                    _rule_r2(pag, bk, a, b, c)
                    _rule_r2(pag, bk, c, b, a)


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

        # R1 + R2 — в порядке causal-learn (rulesR1R2cycle): по узлам b и
        # парам его соседей (a, c) чередуются R1(a,b,c), R1(c,b,a), R2(a,b,c),
        # R2(c,b,a). Полный проход R1, затем R2 (порядок pcalg) на данных с
        # противоречивыми тройками даёт другую неподвижную точку: на
        # bnlearn/andes три ребра ориентировались R1 раньше, чем R2 успевал
        # поставить стрелку, и итог расходился с causal-learn.
        _rules_r1_r2_cycle(pag, _bk, unfVect, rules[0], rules[1])

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

def fci_stable_v2(
    cg,
    data,
    alpha=0.05,
    indep_test='fisherz',
    n_jobs=1,
    verbose=False,
    use_pds_path=False,
    m_max=float('inf'),
    background_knowledge=None,
    show_progress: bool = False,
):
    """FCI Stable v2: скелет -> Possible-D-SEP -> ориентация R0 + R1..R10.

    Шаг Possible-D-SEP (``pdsep_v2``) следует семантике
    ``removeByPossibleDsep`` из causal-learn 0.1.4.5: PDS по ребру (Tetrad
    ``getPossibleDsep``), сепараторы размера не меньше 2, подмножества
    adj(a)/adj(b) пропускаются, рёбра удаляются немедленно, порядок сторон
    как у Edge в causal-learn. Ориентация — ``udag2pag_v2``.

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
        Independence test method (e.g. 'fisherz', 'chisq').
    n_jobs : int
        Number of parallel jobs. 1 = sequential, -1 = all cores. Результат
        от ``n_jobs`` не зависит.
    verbose : bool
        Print progress information.
    use_pds_path : bool
        Ограничить Possible-D-SEP двусвязной компонентой проверяемого ребра
        (Colombo et al. 2012, Def. 3.4). Отклонение от causal-learn, поэтому
        по умолчанию выключено.
    m_max : float
        Максимальный размер сепаратора в Possible-D-SEP. Сепараторы
        начинаются с размера 2, поэтому m_max < 2 отключает CI-тесты
        этого шага. По умолчанию inf — без ограничения, результат прежний.
    background_knowledge : BackgroundKnowledge, optional
        Фоновые знания о направлениях рёбер. Применяются на трёх фазах:
        PRE_SEARCH (до поиска скелета — оттуда же берутся обязательные рёбра,
        которые запрещено удалять на шагах skeleton и pdsep),
        BETWEEN_ORIENTATION (после коллайдеров и после каждого прохода правил)
        и POST_ORIENTATION (после сходимости). ``None`` — полный no-op.
    show_progress : bool
        Прогресс-бар этапа скелета.

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

    # Step 2: Possible-D-SEP (semantics of causal-learn, parallel via joblib)
    if verbose:
        print(f"[FCI v2] Step 2: Possible-D-SEP (n_jobs={n_jobs})...")
    pdsep_res = pdsep_v2(
        graphDict, suffStat, test_func_wrapper, p,
        graphDict['sepset'], alpha, graphDict['pMax'],
        unfVect=graphDict['unfTriples'],
        m_max=m_max,
        n_jobs=n_jobs,
        verbose=verbose,
        use_pds_path=use_pds_path,
        protected_edges=protected_edges,
    )

    # Step 3: Orient v-structures + R1-R10
    if verbose:
        print("[FCI v2] Step 3: Orientation rules...")
    pag = udag2pag_v2(pdsep_res['G'], pdsep_res['sepset'], p,
                      unfVect=graphDict['unfTriples'],
                      background_knowledge=background_knowledge,
                      node_names=node_names)

    return pag
