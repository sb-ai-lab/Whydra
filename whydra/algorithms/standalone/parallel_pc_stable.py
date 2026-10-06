import numpy as np
from itertools import combinations, permutations
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from ..graph_core import CausalGraph, Endpoint, Edge
from .profiler import profiler

from causallearn.graph.GeneralGraph import GeneralGraph as CLGeneralGraph
from causallearn.graph.GraphNode import GraphNode as CLGraphNode
from causallearn.graph.Edge import Edge as CLEdge
from causallearn.graph.Endpoint import Endpoint as CLEndpoint


def _skeleton_discovery_worker(x: int, cg: CausalGraph, depth: int, alpha: float, verbose: bool):
    """
    Функция-воркер, которая выполняется в отдельном процессе.
    Проверяет независимость узла x от его соседей.
    """
    edge_removals_local = []
    sepsets_local = []

    Neigh_x = cg.neighbors(x)
    if len(Neigh_x) < depth:
        return edge_removals_local, sepsets_local

    for y in Neigh_x:
        # ВАЖНО: Убрана оптимизация "if x >= y: continue".
        # В stable-режиме мы должны проверить (x, y) даже если (y, x) проверяется в другом потоке,
        # чтобы найти sepset именно для направления x->y (хотя граф неориентирован, sepset привязан к паре).
        # Это обеспечивает симметрию и полноту sepsets.

        Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))

        for S in combinations(Neigh_x_noy, depth):
            p_val = cg.ci_test(x, y, S)
            if p_val > alpha:
                # Нашли независимость -> запоминаем, что нужно удалить ребро (x, y)
                # и сохранить sepset S
                edge_removals_local.append((x, y))
                sepsets_local.append((x, y, S))

                # В PC мы прекращаем перебор S для данной пары (x, y), как только нашли независимость
                break

    return edge_removals_local, sepsets_local


def skeleton_discovery(cg: CausalGraph, alpha: float, stable: bool, verbose: bool, show_progress: bool, n_jobs: int):
    """
    Этап 1: Поиск скелета (Skeleton Discovery) с параллелизацией.
    """
    no_of_var = len(cg.nodes)
    depth = -1

    # Если n_jobs < 0, используем все доступные ядра
    if n_jobs < 0:
        import os
        n_jobs = os.cpu_count()
    n_jobs=12

    pbar = None
    if show_progress:
        pbar = tqdm(total=no_of_var)

    # Оборачиваем весь процесс поиска скелета
    with profiler.time_block("skeleton_discovery_total", tags={"algo": "parallel", "n_jobs": n_jobs}):
        while cg.max_degree() - 1 > depth:
            depth += 1
            if depth == 2:
                break

            if show_progress:
                pbar.reset()
                pbar.set_description(f'Depth={depth} (Par), max_degree={cg.max_degree()}')

            # Оборачиваем каждую глубину отдельно
            with profiler.time_block(f"skeleton_discovery_depth_{depth}", tags={"algo": "parallel", "depth": depth}):
                # --- ПАРАЛЛЕЛЬНЫЙ БЛОК ---
                # Запускаем воркеры для каждого узла x
                # backend="loky" - стандартный и надежный бэкенд joblib для процессов
                # Важно: передаем копию cg или убеждаемся, что он сериализуем.
                # Joblib использует pickle, CausalGraph должен пиклиться.
                results = Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(_skeleton_discovery_worker)(x, cg, depth, alpha, verbose)
                    for x in range(no_of_var)
                )

                # Обновляем прогресс бар сразу на все количество (так как Parallel блокирующий)
                if show_progress:
                    pbar.update(no_of_var)

                # --- АГРЕГАЦИЯ РЕЗУЛЬТАТОВ ---
                # Собираем результаты со всех процессов
                edge_removal_epoch = []
                sepset_updates_epoch = []

                for edges, sepsets in results:
                    edge_removal_epoch.extend(edges)
                    sepset_updates_epoch.extend(sepsets)

                # --- ОБНОВЛЕНИЕ ГРАФА (Синхронизация) ---
                # Удаляем ребра и обновляем sepset.
                # Для PC-Stable это делается строго после завершения всех проверок на текущей глубине.

                for (x, y) in edge_removal_epoch:
                    # Удаляем ребро (симметрично)
                    edge = cg.G.get_edge(cg.nodes[x], cg.nodes[y])
                    if edge:
                        cg.G.remove_edge(edge)

                for (x, y, S) in sepset_updates_epoch:
                    _append_sepset(cg, x, y, S)

            # Если на текущей глубине ничего не удалили и макс. степень уже меньше глубины, можно выходить раньше
            if not edge_removal_epoch and cg.max_degree() - 1 <= depth:
                break

    if show_progress:
        pbar.close()
    return cg


def _append_sepset(cg, x, y, S):
    if cg.sepset[x, y] is None:
        cg.sepset[x, y] = [S]
    else:
        cg.sepset[x, y].append(S)
    # Симметричность
    if cg.sepset[y, x] is None:
        cg.sepset[y, x] = [S]
    else:
        cg.sepset[y, x].append(S)


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
    edge = cg.G.get_edge(cg.nodes[source], cg.nodes[target])
    if edge:
        cg.G.remove_edge(edge)
        cg.G.add_edge(Edge(cg.nodes[source], cg.nodes[target], Endpoint.TAIL, Endpoint.ARROW))


def _orient_edge_conflict_aware(cg, source, target):
    """
    Пытается ориентировать source -> target.
    Используется в Meek rules.
    """
    if cg.is_fully_directed(source, target): return
    if cg.is_bidirected(source, target): return

    if cg.is_fully_directed(target, source):
        # Conflict -> Bi-directed
        edge = cg.G.get_edge(cg.nodes[source], cg.nodes[target])
        if edge:
            cg.G.remove_edge(edge)
            cg.G.add_edge(Edge(cg.nodes[source], cg.nodes[target], Endpoint.ARROW, Endpoint.ARROW))
        return

    if cg.is_undirected(source, target):
        edge = cg.G.get_edge(cg.nodes[source], cg.nodes[target])
        if edge:
            cg.G.remove_edge(edge)
            cg.G.add_edge(Edge(cg.nodes[source], cg.nodes[target], Endpoint.TAIL, Endpoint.ARROW))
        return


def apply_meek_rules(cg: CausalGraph):
    """
    Этап 3: Правила Мика (Meek rules).
    """
    loop = True
    while loop:
        loop = False

        # R1
        triples = cg.find_unshielded_triples()
        for (i, j, k) in triples:
            if cg.is_fully_directed(i, j) and cg.is_undirected(j, k):
                _orient_edge_conflict_aware(cg, j, k)
                loop = True
            elif cg.is_fully_directed(k, j) and cg.is_undirected(j, i):
                _orient_edge_conflict_aware(cg, j, i)
                loop = True

        # R2
        triangles = cg.find_triangles()
        for (i, j, k) in triangles:
            nodes_tri = [i, j, k]
            for a, b, c in permutations(nodes_tri, 3):
                if cg.is_fully_directed(a, b) and cg.is_fully_directed(b, c) and cg.is_undirected(a, c):
                    _orient_edge_conflict_aware(cg, a, c)
                    loop = True

        # R3
        kites = cg.find_kites()
        for (i, j, k, l) in kites:
            if cg.is_fully_directed(j, l) and cg.is_fully_directed(k, l) and cg.is_undirected(i, l):
                if cg.is_undirected(i, j) and cg.is_undirected(i, k):
                    _orient_edge_conflict_aware(cg, i, l)
                    loop = True

    return cg


def convert_to_causallearn_graph(local_cg: CausalGraph) -> CLGeneralGraph:
    cl_nodes = [CLGraphNode(node.name) for node in local_cg.nodes]
    cl_graph = CLGeneralGraph(cl_nodes)

    for i in range(local_cg.G.num_vars):
        for j in range(i + 1, local_cg.G.num_vars):
            end_j_val = local_cg.G.graph[i, j]
            end_i_val = local_cg.G.graph[j, i]

            if end_j_val != 0 or end_i_val != 0:
                def map_end(val):
                    if val == 1: return CLEndpoint.ARROW
                    if val == -1: return CLEndpoint.TAIL
                    if val == 2: return CLEndpoint.CIRCLE
                    return CLEndpoint.NULL

                cl_edge = CLEdge(cl_nodes[i], cl_nodes[j], map_end(end_i_val), map_end(end_j_val))
                cl_graph.add_edge(cl_edge)

    return cl_graph


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
    if verbose:
        print(f"Starting Parallel Skeleton Discovery with n_jobs={n_jobs}...")

    with profiler.time_block("full_pc_algorithm", tags={"algo": "parallel", "n_jobs": n_jobs}):
        skeleton_discovery(cg, alpha, stable, verbose, show_progress, n_jobs)

        if verbose:
            print("Orienting Colliders...")
        with profiler.time_block("orient_colliders", tags={"algo": "parallel"}):
            orient_colliders(cg, priority=uc_priority)

        if verbose:
            print("Applying Meek Rules...")
        with profiler.time_block("apply_meek_rules", tags={"algo": "parallel"}):
            apply_meek_rules(cg)

    class ResultWrapper:
        def __init__(self, g):
            self.G = g

    final_cl_graph = convert_to_causallearn_graph(cg)
    return ResultWrapper(final_cl_graph)