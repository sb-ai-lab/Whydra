import numpy as np
from itertools import combinations, permutations
from tqdm.auto import tqdm

from ..graph_core import CausalGraph, Edge, Endpoint
from ...background_knowledge import (
    BKPhase,
    MatrixEncoding,
    apply_bk_checkpoint,
    bk_allows_edge,
    orientation_guard_for,
)
from .profiler import profiler
def skeleton_discovery(
    cg: CausalGraph,
    alpha: float,
    stable: bool,
    verbose: bool,
    show_progress: bool,
    protected_edges=(),
):
    """
    Этап 1: Поиск скелета.

    ``protected_edges`` — пары, которые требует background knowledge. Такое
    ребро не удаляется и sepset для него не записывается: пара остаётся
    смежной, и записанный sepset только мешал бы ориентации коллайдеров.

    Args:
        cg: causal graph carrying the CI test.
        alpha: significance level.
        stable: whether to test all conditioning sets at each depth.
        verbose: verbosity flag.
        show_progress: whether to display progress.
        protected_edges: pairs that background knowledge keeps adjacent.

    Returns:
        CausalGraph: input graph with its skeleton and separating unions updated.
    """
    no_of_var = len(cg.nodes)
    protected_edges = {
        frozenset((int(x), int(y))) for x, y in protected_edges
    }
    depth = -1
    pbar = tqdm(total=no_of_var) if show_progress else None

    with profiler.time_block("skeleton_discovery_total", tags={"algo": "sequential"}):
        while cg.max_degree() - 1 > depth:
            depth += 1
            edge_removal = []
            if show_progress:
                pbar.reset()
                pbar.set_description(f'Depth={depth}, max_degree={cg.max_degree()}')

            with profiler.time_block(f"skeleton_discovery_depth_{depth}",
                                     tags={"algo": "sequential", "depth": depth}):
                for x in range(no_of_var):
                    if show_progress: pbar.update()

                    Neigh_x = cg.neighbors(x)
                    if len(Neigh_x) < depth: continue

                    for y in Neigh_x:
                        # ВАЖНО: Не пропускаем проверку, даже если (x, y) уже в edge_removal.
                        # Это нужно для объединения индексов всех разделяющих S в stable режиме.

                        Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))

                        found = False
                        members = set()
                        for S in combinations(Neigh_x_noy, depth):
                            p_val = cg.ci_test(x, y, S)
                            if p_val > alpha:
                                if frozenset((int(x), int(y))) in protected_edges:
                                    break
                                if not stable:
                                    edge = cg.G.get_edge(cg.nodes[x], cg.nodes[y])
                                    if edge: cg.G.remove_edge(edge)
                                    _append_sepset(cg, x, y, S)
                                    break
                                else:
                                    # causal-learn (stable=True) не прерывает перебор:
                                    # на данной глубине тестируются все S, а объединение
                                    # их разделяющих индексов определяет коллайдеры
                                    # в orient_colliders.
                                    if not found:
                                        edge_removal.append((x, y))
                                        edge_removal.append((y, x))
                                        found = True
                                    members.update(int(v) for v in S)

                        if found:
                            _append_sepset(cg, x, y, members)

                if stable:
                    for (x, y) in set(edge_removal):
                        if frozenset((int(x), int(y))) in protected_edges:
                            continue
                        edge = cg.G.get_edge(cg.nodes[x], cg.nodes[y])
                        if edge: cg.G.remove_edge(edge)

            if show_progress: pbar.refresh()

    if show_progress: pbar.close()
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
        # causal-learn Meek skips an orientation whose target is already an ancestor of the source.
        if cg.is_ancestor_of(target, source): return False
        if not bk_allows_edge(getattr(cg, "bk_guard", None), cg.G.graph,
                              source, target, Endpoint.TAIL.value, Endpoint.ARROW.value):
            return False
        edge = cg.G.get_edge(cg.nodes[source], cg.nodes[target])
        if edge:
            cg.G.remove_edge(edge)
            cg.G.add_edge(Edge(cg.nodes[source], cg.nodes[target], Endpoint.TAIL, Endpoint.ARROW))
        return


def apply_meek_rules(cg: CausalGraph, background_knowledge=None):
    """
    Этап 3: Правила Мика (Meek rules).

    Выход из цикла — по фактическому изменению матрицы, а не по флагу
    «условие правила сработало»: иначе применённые между проходами фоновые
    знания не были бы учтены в критерии сходимости.
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

    with profiler.time_block("full_pc_algorithm", tags={"algo": "sequential"}):
        skeleton_discovery(cg, alpha, stable, verbose, show_progress,
                           protected_edges=protected_edges)
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
        def __init__(self, g): self.G = g

    return convert_to_general_graph(cg)
