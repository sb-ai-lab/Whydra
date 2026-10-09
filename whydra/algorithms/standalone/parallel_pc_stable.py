import math
import os

import numpy as np
from itertools import combinations, permutations
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from ..graph_core import CausalGraph, Endpoint, Edge
from ...background_knowledge import (
    BKPhase,
    MatrixEncoding,
    apply_bk_checkpoint,
    bk_allows_edge,
    orientation_guard_for,
)
from .profiler import profiler

# Количество чанков на одного воркера: больше чанков — лучше балансировка
# нагрузки (LPT по стоимости CI-тестов), но больше накладных расходов.
CHUNKS_PER_WORKER = 4

# Порог для memmap-ирования больших ndarray в joblib: массивы CI-теста
# выгружаются на диск и шарятся между воркерами вместо копирования.
MAX_NBYTES = "10K"


def _node_cost(graph: np.ndarray, x: int, depth: int) -> int:
    """Number of CI tests node ``x`` would run at the given depth.

    A node with degree ``deg`` tests each of its ``deg`` neighbours against
    all ``comb(deg - 1, depth)`` conditioning subsets of size ``depth``
    (subsets drawn from the other neighbours).

    Args:
        graph: adjacency snapshot (p x p endpoint matrix, 0 = no edge).
        x: node index.
        depth: size of the conditioning sets.

    Returns:
        int: estimated number of CI tests; 0 when the node cannot reach this
        depth (``deg < depth``) or has no neighbours at all.
    """
    deg = int(np.count_nonzero(graph[x, :]))
    if deg < depth or deg == 0:
        return 0
    return deg * math.comb(deg - 1, depth)


def _balanced_node_chunks(
    graph: np.ndarray,
    depth: int,
    n_chunks: int,
) -> list[list[int]]:
    """Deterministic greedy LPT partition of nodes into balanced chunks.

    Nodes are taken as plain Python ints from ``range(p)``. Each node is
    assigned an estimated cost via :func:`_node_cost`; nodes are sorted by
    cost descending (stable sort, so equal costs keep ascending index order)
    and greedily placed on the currently least-loaded chunk (ties resolved
    towards the lowest chunk index). Empty chunks are dropped.

    Args:
        graph: adjacency snapshot (p x p endpoint matrix).
        depth: size of the conditioning sets at this depth.
        n_chunks: maximum number of chunks.

    Returns:
        list[list[int]]: non-empty chunks of node indices, deterministic for
        a given adjacency snapshot.
    """
    p = graph.shape[0]
    costs = [_node_cost(graph, x, depth) for x in range(p)]

    # Стабильная сортировка по убыванию стоимости: при равных стоимостях
    # узлы сохраняют возрастающий порядок индексов.
    order = sorted(range(p), key=lambda x: -costs[x])

    loads = [0] * n_chunks
    chunks: list[list[int]] = [[] for _ in range(n_chunks)]

    for x in order:
        # min по loads: при равной загрузке выбирается чанк с меньшим индексом.
        idx = min(range(n_chunks), key=lambda i: loads[i])
        chunks[idx].append(x)
        loads[idx] += costs[x]

    return [chunk for chunk in chunks if chunk]


def _skeleton_discovery_worker(
    nodes: list[int],
    graph: np.ndarray,
    ci_test,
    depth: int,
    alpha: float,
    stable: bool,
    protected_edges: frozenset = frozenset(),
) -> list[tuple[int, list, list]]:
    """Run the PC(-stable) edge-removal loop for a chunk of nodes.

    Instead of pickling the whole ``CausalGraph`` (node list, full graph
    object, the ever-growing ``p x p`` object array of sepsets, which cannot
    be memmapped), each worker receives only:

    * ``graph`` — a snapshot of the adjacency matrix (a small dense int
      matrix that is cheap to serialize);
    * ``ci_test`` — the CI test object, which is callable and is exactly
      what ``cg.ci_test(x, y, S)`` delegates to.

    The CI object is the *same Python object* on every call, so joblib's
    memmap cache dumps its large ``data``/``data_int`` arrays to disk once
    per run instead of re-pickling them per task.

    Under ``stable=True`` (causal-learn parity with
    ``pc(stable=True, uc_rule=0, uc_priority=2)``) the enumeration of
    conditioning sets S of the current depth is NOT stopped at the first
    separating set: every S is tested and the union of separating-set
    members is recorded once per separated pair, because ``orient_colliders``
    only asks whether y occurs in any separating set.  Under ``stable=False``
    the search stops at the first separating S, as in plain PC.  A pair
    protected by background knowledge also stops at the first separating S:
    its edge is never removed and its sepsets are discarded by the caller,
    so the remaining tests of the depth would be wasted.

    Args:
        nodes: node indices handled by this task, in processing order.
        graph: adjacency snapshot of the current depth.
        ci_test: callable ``(x, y, S) -> p-value``.
        depth: size of the conditioning sets.
        alpha: significance level.
        stable: whether to enumerate all separating sets (PC-stable).
        protected_edges: pairs ``frozenset((x, y))`` that background
            knowledge keeps in the graph.

    Returns:
        list[tuple[int, list, list]]: one entry ``(x, edge_removals_x,
        sepsets_x)`` per node in ``nodes``, in the given order, even when
        both lists are empty. Each separated pair has one ``(x, y, union)``
        record, with a sorted tuple of plain Python int members. The per-node
        CI-test order is unchanged.
    """
    results = []

    for x in nodes:
        edge_removals_local = []
        sepsets_local = []

        Neigh_x = np.where(graph[x, :] != 0)[0]
        if len(Neigh_x) < depth:
            results.append((x, edge_removals_local, sepsets_local))
            continue

        for y in Neigh_x:
            # ВАЖНО: Убрана оптимизация "if x >= y: continue".
            # В stable-режиме мы должны проверить (x, y) даже если (y, x) проверяется в другом потоке,
            # чтобы найти sepset именно для направления x->y (хотя граф неориентирован, sepset привязан к паре).
            # Это обеспечивает симметрию и полноту sepsets.
            Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
            protected = frozenset((int(x), int(y))) in protected_edges

            found = False
            members = set()
            for S in combinations(Neigh_x_noy, depth):
                p_val = ci_test(x, y, S)
                if p_val > alpha:
                    # Нашли независимость -> запоминаем, что нужно удалить ребро (x, y)
                    # и накапливаем объединение индексов разделяющих множеств.
                    if not found:
                        edge_removals_local.append((x, y))
                        found = True
                    members.update(int(v) for v in S)

                    # В causal-learn (stable=True) перебор НЕ прерывается: на данной
                    # глубине тестируются все S и объединяются их разделяющие индексы.
                    # Защищённую BK пару хватает разделить один раз: ребро не
                    # удаляется, sepset отбрасывается при агрегации.
                    if not stable or protected:
                        break

            if found:
                sepsets_local.append((x, y, tuple(sorted(members))))

        results.append((x, edge_removals_local, sepsets_local))

    return results


def skeleton_discovery(
    cg: CausalGraph,
    alpha: float,
    stable: bool,
    verbose: bool,
    show_progress: bool,
    n_jobs: int,
    protected_edges=(),
):
    """
    Этап 1: Поиск скелета (Skeleton Discovery) с параллелизацией.
    """
    no_of_var = len(cg.nodes)
    protected_edges = {
        frozenset((int(x), int(y))) for x, y in protected_edges
    }
    depth = -1

    # Если n_jobs < 0, используем все доступные ядра
    if n_jobs < 0:
        n_jobs = os.cpu_count()
    effective_n_jobs = n_jobs

    pbar = None
    if show_progress:
        pbar = tqdm(total=no_of_var)

    # Оборачиваем весь процесс поиска скелета
    with profiler.time_block("skeleton_discovery_total", tags={"algo": "parallel", "n_jobs": effective_n_jobs}):
        # Один объект Parallel на весь запуск: пул loky-воркеров и memmap-кэш
        # joblib (массивы CI-теста выгружаются на диск один раз) живут
        # между глубинами, а не пересоздаются на каждой итерации.
        with Parallel(
            n_jobs=effective_n_jobs,
            backend="loky",
            max_nbytes=MAX_NBYTES,
            batch_size=1,
        ) as parallel:
            while cg.max_degree() - 1 > depth:
                depth += 1

                if show_progress:
                    pbar.reset()
                    pbar.set_description(f'Depth={depth} (Par), max_degree={cg.max_degree()}')

                # Оборачиваем каждую глубину отдельно
                with profiler.time_block(f"skeleton_discovery_depth_{depth}", tags={"algo": "parallel", "depth": depth}):
                    # --- ПАРАЛЛЕЛЬНЫЙ БЛОК ---
                    # Воркеры получают лёгкий снимок матрицы смежности и
                    # CI-объект, а не весь CausalGraph.
                    graph_snapshot = cg.G.graph.copy()
                    n_chunks = max(1, min(no_of_var, CHUNKS_PER_WORKER * effective_n_jobs))
                    chunks = _balanced_node_chunks(graph_snapshot, depth, n_chunks)

                    results = parallel(
                        delayed(_skeleton_discovery_worker)(
                            chunk, graph_snapshot, cg.test, depth, alpha, stable,
                            frozenset(protected_edges),
                        )
                        for chunk in chunks
                    )

                    # Обновляем прогресс бар сразу на все количество (так как Parallel блокирующий)
                    if show_progress:
                        pbar.update(no_of_var)

                    # --- АГРЕГАЦИЯ РЕЗУЛЬТАТОВ ---
                    # Собираем результаты со всех чанков и выравниваем их по
                    # x по возрастанию — в точности порядок однопоточного
                    # цикла по range(no_of_var).
                    items = [
                        item
                        for chunk_result in results
                        for item in chunk_result
                    ]
                    items.sort(key=lambda item: item[0])

                    edge_removal_epoch = []
                    sepset_updates_epoch = []

                    for _x, edges, sepsets in items:
                        edge_removal_epoch.extend(edges)
                        sepset_updates_epoch.extend(sepsets)

                    # --- ОБНОВЛЕНИЕ ГРАФА (Синхронизация) ---
                    # Удаляем ребра и обновляем sepset.
                    # Для PC-Stable это делается строго после завершения всех проверок на текущей глубине.

                    for (x, y) in edge_removal_epoch:
                        if frozenset((int(x), int(y))) in protected_edges:
                            continue
                        # Удаляем ребро (симметрично)
                        edge = cg.G.get_edge(cg.nodes[x], cg.nodes[y])
                        if edge:
                            cg.G.remove_edge(edge)

                    for (x, y, S) in sepset_updates_epoch:
                        # Защищённое ребро не удаляется, поэтому и разделяющее
                        # множество для него записывать нельзя: ориентация
                        # коллайдеров спрашивает, лежит ли средний узел в sepset
                        # внешней пары, и запись для выжившего ребра молча меняет
                        # ориентацию соседних троек.
                        if frozenset((int(x), int(y))) in protected_edges:
                            continue
                        _append_sepset(cg, x, y, S)

                # Если на текущей глубине ничего не удалили и макс. степень уже меньше глубины, можно выходить раньше
                if not edge_removal_epoch and cg.max_degree() - 1 <= depth:
                    break

    if show_progress:
        pbar.close()
    return cg


def _append_sepset(
    cg,
    x,
    y,
    S,
):
    """Merge separating-set members into one sorted tuple per direction.

    Args:
        cg: causal graph whose separating sets are updated.
        x: first node index.
        y: second node index.
        S: separating-set members to merge as plain Python integers.

    Returns:
        None: updates both symmetric cells in place, preserving empty sets.
    """
    members = {int(v) for v in S}
    for i, j in ((x, y), (y, x)):
        current = cg.sepset[i, j]
        union = members if current is None else members.union(current[0])
        cg.sepset[i, j] = [tuple(sorted(union))]


def orient_colliders(cg: CausalGraph, priority: int = 2):
    """
    Этап 2: Ориентация V-структур (Unshielded Colliders).
    Priority 2: Prioritize Existing Colliders (Conservative).
    """
    triples = cg.find_unshielded_triples()

    for (x, y, z) in triples:
        sepsets = cg.sepset[x, z]
        if sepsets is None: continue

        is_collider = True
        for S in sepsets:
            if y in S:
                is_collider = False
                break

        if is_collider:
            # Priority 2: Не создаем коллидер, если он противоречит существующим ребрам
            conflict_x = cg.is_fully_directed(y, x)
            conflict_z = cg.is_fully_directed(y, z)

            if not conflict_x and not conflict_z:
                _orient_edge(cg, x, y)
                _orient_edge(cg, z, y)


def _orient_edge(cg, source, target):
    if not bk_allows_edge(getattr(cg, "bk_guard", None), cg.G.graph,
                          source, target, Endpoint.TAIL.value, Endpoint.ARROW.value):
        return False
    edge = cg.G.get_edge(cg.nodes[source], cg.nodes[target])
    if edge:
        cg.G.remove_edge(edge)
        cg.G.add_edge(Edge(cg.nodes[source], cg.nodes[target], Endpoint.TAIL, Endpoint.ARROW))
        return True
    return False


def _orient_edge_conflict_aware(cg, source, target):
    """
    Пытается ориентировать source -> target.
    Используется в Meek rules.
    """
    if cg.is_fully_directed(source, target): return False
    if cg.is_bidirected(source, target): return False

    if cg.is_fully_directed(target, source):
        # Conflict -> Bi-directed
        edge = cg.G.get_edge(cg.nodes[source], cg.nodes[target])
        if edge:
            cg.G.remove_edge(edge)
            cg.G.add_edge(Edge(cg.nodes[source], cg.nodes[target], Endpoint.ARROW, Endpoint.ARROW))
            return True
        return False

    if cg.is_undirected(source, target):
        # causal-learn Meek skips an orientation whose target is already an ancestor of the source.
        if cg.is_ancestor_of(target, source): return False
        if not bk_allows_edge(getattr(cg, "bk_guard", None), cg.G.graph,
                              source, target, Endpoint.TAIL.value, Endpoint.ARROW.value):
            return False
        edge = cg.G.get_edge(cg.nodes[source], cg.nodes[target])
        if edge:
            cg.G.remove_edge(edge)
            cg.G.add_edge(Edge(cg.nodes[source], cg.nodes[target], Endpoint.TAIL, Endpoint.ARROW))
            return True
        return False
    return False


def apply_meek_rules(cg: CausalGraph, background_knowledge=None):
    """
    Этап 3: Правила Мика (Meek rules).
    """
    loop = True
    while loop:
        before_pass = cg.G.graph.copy()

        # R1
        triples = cg.find_unshielded_triples()
        for (i, j, k) in triples:
            if cg.is_fully_directed(i, j) and cg.is_undirected(j, k):
                _orient_edge_conflict_aware(cg, j, k)
            elif cg.is_fully_directed(k, j) and cg.is_undirected(j, i):
                _orient_edge_conflict_aware(cg, j, i)

        # R2
        triangles = cg.find_triangles()
        for (i, j, k) in triangles:
            nodes_tri = [i, j, k]
            for a, b, c in permutations(nodes_tri, 3):
                if cg.is_fully_directed(a, b) and cg.is_fully_directed(b, c) and cg.is_undirected(a, c):
                    _orient_edge_conflict_aware(cg, a, c)

        # R3
        kites = cg.find_kites()
        for (i, j, k, l) in kites:
            if cg.is_fully_directed(j, l) and cg.is_fully_directed(k, l) and cg.is_undirected(i, l):
                if cg.is_undirected(i, j) and cg.is_undirected(i, k):
                    _orient_edge_conflict_aware(cg, i, l)

        apply_bk_checkpoint(
            background_knowledge,
            cg.G.graph,
            [node.name for node in cg.nodes],
            phase=BKPhase.BETWEEN_ORIENTATION,
            checkpoint="after_meek_pass",
            encoding=MatrixEncoding.STANDARD,
            graph_kind="cpdag",
            previous=before_pass,
        )
        loop = not np.array_equal(before_pass, cg.G.graph)

    return cg


def convert_to_general_graph(local_cg: CausalGraph):
    """Return the local graph maintained by the algorithm."""
    return local_cg.G


def pc_stable(
        cg: CausalGraph,
        alpha=0.05,
        stable: bool = True,
        uc_rule: int = 0,
        uc_priority: int = 2,
        background_knowledge=None,
        verbose: bool = False,
        show_progress: bool = True,
        n_jobs: int = -1,
        **kwargs
):
    node_names = [node.name for node in cg.nodes]
    apply_bk_checkpoint(
        background_knowledge,
        cg.G.graph,
        node_names,
        phase=BKPhase.PRE_SEARCH,
        checkpoint="before_skeleton",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="cpdag",
    )
    # Guard уровня записи: чекпоинт после фазы направление исправит, но
    # коллайдер, потерянный из-за запрещённой стрелки, уже не вернёт.
    cg.bk_guard = orientation_guard_for(background_knowledge, node_names)
    protected_edges = ()
    if background_knowledge is not None:
        protected_edges = background_knowledge.required_pairs(
            node_names, phase=BKPhase.PRE_SEARCH
        )
    if verbose:
        print(f"Starting Parallel Skeleton Discovery with n_jobs={n_jobs}...")

    with profiler.time_block("full_pc_algorithm", tags={"algo": "parallel", "n_jobs": n_jobs}):
        skeleton_discovery(
            cg,
            alpha,
            stable,
            verbose,
            show_progress,
            n_jobs,
            protected_edges=protected_edges,
        )

        if verbose:
            print("Orienting Colliders...")
        with profiler.time_block("orient_colliders", tags={"algo": "parallel"}):
            orient_colliders(cg, priority=uc_priority)
            apply_bk_checkpoint(
                background_knowledge,
                cg.G.graph,
                node_names,
                phase=BKPhase.BETWEEN_ORIENTATION,
                checkpoint="after_colliders",
                encoding=MatrixEncoding.STANDARD,
                graph_kind="cpdag",
            )

        if verbose:
            print("Applying Meek Rules...")
        with profiler.time_block("apply_meek_rules", tags={"algo": "parallel"}):
            apply_meek_rules(cg, background_knowledge=background_knowledge)

        apply_bk_checkpoint(
            background_knowledge,
            cg.G.graph,
            node_names,
            phase=BKPhase.POST_ORIENTATION,
            checkpoint="after_orientation",
            encoding=MatrixEncoding.STANDARD,
            graph_kind="cpdag",
        )

    class ResultWrapper:
        def __init__(self, g):
            self.G = g

    return ResultWrapper(convert_to_general_graph(cg))
