import math
from itertools import combinations, permutations

import numpy as np

from ...background_knowledge import (
    BKPhase,
    MatrixEncoding,
    apply_bk_checkpoint,
    bk_allows_edge,
    orientation_guard_for,
)
from ..graph_core import CausalGraph, Edge, Endpoint
from ..graph_core._optional_imports import Parallel, delayed
from ..graph_core._optional_imports import tqdm_auto as tqdm
from ..graph_core.conditioning import bounded_size, configure_conditioning
from ..graph_core.validation import effective_n_jobs
from .profiler import profiler

# Chunks per worker. More chunks balance the load better (longest CI jobs
# first, based on estimated cost), but increase overhead.
CHUNKS_PER_WORKER = 4

# joblib memmap threshold for large ndarrays. Workers share CI test arrays
# backed by files on disk, which avoids copying them to each worker.
MAX_NBYTES = "1M"


def _node_cost(graph: np.ndarray, x: int, depth: int) -> int:
    """Estimate how many CI tests node ``x`` would run at the given depth.

    A node with degree ``deg`` tests each of its ``deg`` neighbours against
    all ``comb(deg - 1, depth)`` conditioning subsets of size ``depth``
    (subsets drawn from the other neighbours).

    Parameters
    ----------
    graph : numpy.ndarray
        Adjacency snapshot (p x p endpoint matrix, 0 = no edge).
    x : int
        Node index.
    depth : int
        Size of the conditioning sets.

    Returns
    -------
    int
        Estimated number of CI tests. The estimate is zero when the node has
        at most ``depth`` neighbours.
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

    Parameters
    ----------
    graph : numpy.ndarray
        Adjacency snapshot (p x p endpoint matrix).
    depth : int
        Size of the conditioning sets at this depth.
    n_chunks : int
        Maximum number of chunks.

    Returns
    -------
    list[list[int]]
        Non-empty chunks of node indices, deterministic for
        a given adjacency snapshot.
    """
    p = graph.shape[0]
    costs = [_node_cost(graph, x, depth) for x in range(p)]

    # Sort by decreasing estimated cost. The stable sort keeps nodes with
    # equal costs in ascending index order.
    order = sorted(range(p), key=lambda x: -costs[x])

    loads = [0] * n_chunks
    chunks: list[list[int]] = [[] for _ in range(n_chunks)]

    for x in order:
        # min on loads breaks ties by choosing the chunk with the lowest index.
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

    Along with its assigned nodes and search settings, each worker receives
    a dense integer snapshot of the adjacency matrix and the callable used by
    ``cg.ci_test(x, y, S)``. Passing the snapshot avoids serializing the full
    ``CausalGraph``, including its node list and growing p-by-p object array
    of separating sets, which cannot be memory-mapped.

    Under ``stable=False``
    the search stops at the first separating S, as in plain PC.  A pair
    protected by background knowledge also stops at the first separating S:
    its edge is never removed and its sepsets are discarded by the caller,
    so the remaining tests of the depth would be wasted.

    Parameters
    ----------
    nodes : list of int
        Node indices handled by this task, in processing order.
    graph : numpy.ndarray
        Adjacency snapshot of the current depth.
    ci_test : callable
        Callable ``(x, y, S) -> p-value``.
    depth : int
        Size of the conditioning sets.
    alpha : float
        Significance level.
    stable : bool
        Whether to enumerate all separating sets (PC-stable).
    protected_edges : collection of node pairs
        Pairs ``frozenset((x, y))`` that background
        knowledge keeps in the graph.

    Returns
    -------
    list[tuple[int, list, list]]
        One entry ``(x, edge_removals_x,
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
            # Test both directions even in stable mode. Their neighborhoods
            # can differ, so (y, x) may reveal separating sets missed by (x, y).
            Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
            protected = frozenset((int(x), int(y))) in protected_edges

            found = False
            members = set()
            for S in combinations(Neigh_x_noy, depth):
                p_val = ci_test(x, y, S)
                if p_val > alpha:
                    # On finding independence, mark edge (x, y) for removal
                    # and accumulate the union of the separating set indices.
                    if not found:
                        edge_removals_local.append((x, y))
                        found = True
                    members.update(int(v) for v in S)

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
    skeleton_backend="auto",
):
    """Stage 1: Run skeleton discovery in parallel."""
    no_of_var = len(cg.nodes)
    protected_edges = {frozenset((int(x), int(y))) for x, y in protected_edges}
    from ..graph_core.ci_batch import supports_fisher_batch
    from .skeleton import SkeletonDiscovery, _pc_execution_stats, _record_pc_dispatch

    if skeleton_backend not in ("auto", "threads", "processes"):
        msg = "skeleton backend must be 'auto', 'threads', or 'processes'"
        raise ValueError(msg)
    if (
        stable and not protected_edges and supports_fisher_batch(cg.test)
    ) or skeleton_backend in ("auto", "threads"):
        SkeletonDiscovery(
            cg,
            alpha=alpha,
            stable=stable,
            n_jobs=n_jobs,
            verbose=verbose,
            show_progress=show_progress,
            parallel_backend=skeleton_backend,
            protected_edges=protected_edges,
        ).run()
        return cg
    depth = -1

    jobs = effective_n_jobs(n_jobs)
    _pc_execution_stats(cg, skeleton_backend, jobs)

    pbar = None
    if show_progress:
        pbar = tqdm(total=no_of_var)

    # Time the entire skeleton search.
    with (
        profiler.time_block(
            "skeleton_discovery_total", tags={"algo": "parallel", "n_jobs": jobs}
        ),
        # Reuse one Parallel instance throughout the run. Its loky worker pool and
        # joblib memmap cache persist across depths, so the CI test arrays are
        # written to disk only once.
        Parallel(
            n_jobs=jobs,
            backend="loky",
            max_nbytes=MAX_NBYTES,
            batch_size=1,
        ) as parallel,
    ):
        while bounded_size(cg, cg.max_degree() - 1) > depth:
            depth += 1

            if pbar is not None:
                pbar.reset()
                pbar.set_description(
                    f"Depth={depth} (Par), max_degree={cg.max_degree()}"
                )

            # Time each depth separately.
            with profiler.time_block(
                f"skeleton_discovery_depth_{depth}",
                tags={"algo": "parallel", "depth": depth},
            ):
                # --- Parallel search ---
                # Workers receive a lightweight snapshot of the adjacency matrix and the
                # CI test object instead of the entire CausalGraph.
                graph_snapshot = cg.G.graph.copy()
                n_chunks = max(1, min(no_of_var, CHUNKS_PER_WORKER * jobs))
                chunks = _balanced_node_chunks(graph_snapshot, depth, n_chunks)

                _record_pc_dispatch(
                    cg,
                    skeleton_backend,
                    jobs,
                    "processes" if jobs > 1 else "serial",
                    depth,
                    len(chunks),
                )

                results = parallel(
                    delayed(_skeleton_discovery_worker)(
                        chunk,
                        graph_snapshot,
                        cg.test,
                        depth,
                        alpha,
                        stable,
                        frozenset(protected_edges),
                    )
                    for chunk in chunks
                )

                # Parallel blocks, so advance the progress bar for all completed items at once.
                if pbar is not None:
                    pbar.update(no_of_var)

                # --- Collect results ---
                # Collect results from all chunks, then order them by ascending x
                # to match the single-threaded loop over
                # range(no_of_var).
                items = [item for chunk_result in results for item in chunk_result]
                items.sort(key=lambda item: item[0])

                edge_removal_epoch = []
                sepset_updates_epoch = []

                for _x, edges, sepsets in items:
                    edge_removal_epoch.extend(edges)
                    sepset_updates_epoch.extend(sepsets)

                # --- Update the shared graph ---
                # Remove edges and update the separating sets.
                # PC-Stable does this only after all checks at the current depth are complete.

                for x, y in edge_removal_epoch:
                    if frozenset((int(x), int(y))) in protected_edges:
                        continue
                    # Remove the edge in both directions.
                    edge = cg.G.get_edge(cg.nodes[x], cg.nodes[y])
                    if edge:
                        cg.G.remove_edge(edge)

                for x, y, S in sepset_updates_epoch:
                    # A protected edge stays in the graph, so do not record a separating set
                    # for it. Collider orientation checks whether the middle node belongs
                    # to the separating set of the two outer nodes. Recording a set for a
                    # surviving edge would silently change the orientation of neighboring
                    # triples.
                    if frozenset((int(x), int(y))) in protected_edges:
                        continue
                    _append_sepset(cg, x, y, S)

            # Stop if no edges were removed and no node has enough other neighbors
            # to form a conditioning set at the next depth.
            if not edge_removal_epoch and cg.max_degree() - 1 <= depth:
                break

    if pbar is not None:
        pbar.close()
    return cg


def _append_sepset(
    cg,
    x,
    y,
    S,
):
    """Merge separating-set members into one sorted tuple per direction.

    Parameters
    ----------
    cg : CausalGraph
        Causal graph whose separating sets are updated.
    x : int
        First node index.
    y : int
        Second node index.
    S : iterable of int
        Separating-set members to merge as plain Python integers.

    Returns
    -------
    None
        Updates both symmetric cells in place, preserving empty sets.
    """
    members = {int(v) for v in S}
    for i, j in ((x, y), (y, x)):
        current = cg.sepset[i, j]
        union = members if current is None else members.union(current[0])
        cg.sepset[i, j] = [tuple(sorted(union))]


def orient_colliders(cg: CausalGraph, priority: int = 2):
    """Orient unshielded colliders using stable PC's priority rule."""
    from .standalone_pc_stable import orient_colliders as orient

    return orient(cg, priority=priority)


def _orient_edge(cg, source, target):
    if not bk_allows_edge(
        getattr(cg, "bk_guard", None),
        cg.G.graph,
        source,
        target,
        Endpoint.TAIL.value,
        Endpoint.ARROW.value,
    ):
        return False
    edge = cg.G.get_edge(cg.nodes[source], cg.nodes[target])
    if edge:
        cg.G.remove_edge(edge)
        cg.G.add_edge(
            Edge(cg.nodes[source], cg.nodes[target], Endpoint.TAIL, Endpoint.ARROW)
        )
        return True
    return False


def _orient_edge_conflict_aware(cg, source, target):
    """Try to orient source -> target when applying Meek's rules."""
    if cg.is_fully_directed(source, target):
        return False
    if cg.is_bidirected(source, target):
        return False

    if cg.is_fully_directed(target, source):
        # Conflict -> Bi-directed
        edge = cg.G.get_edge(cg.nodes[source], cg.nodes[target])
        if edge:
            cg.G.remove_edge(edge)
            cg.G.add_edge(
                Edge(cg.nodes[source], cg.nodes[target], Endpoint.ARROW, Endpoint.ARROW)
            )
            return True
        return False

    if cg.is_undirected(source, target):
        if cg.is_ancestor_of(target, source):
            return False
        if not bk_allows_edge(
            getattr(cg, "bk_guard", None),
            cg.G.graph,
            source,
            target,
            Endpoint.TAIL.value,
            Endpoint.ARROW.value,
        ):
            return False
        edge = cg.G.get_edge(cg.nodes[source], cg.nodes[target])
        if edge:
            cg.G.remove_edge(edge)
            cg.G.add_edge(
                Edge(cg.nodes[source], cg.nodes[target], Endpoint.TAIL, Endpoint.ARROW)
            )
            return True
        return False
    return False


def apply_meek_rules(cg: CausalGraph, background_knowledge=None):
    """Stage 3: Apply Meek's orientation rules."""
    loop = True
    while loop:
        before_pass = cg.G.graph.copy()

        # R1
        triples = cg.find_unshielded_triples()
        for i, j, k in triples:
            if cg.is_fully_directed(i, j) and cg.is_undirected(j, k):
                _orient_edge_conflict_aware(cg, j, k)
            elif cg.is_fully_directed(k, j) and cg.is_undirected(j, i):
                _orient_edge_conflict_aware(cg, j, i)

        # R2
        triangles = cg.find_triangles()
        for i, j, k in triangles:
            nodes_tri = [i, j, k]
            for a, b, c in permutations(nodes_tri, 3):
                if (
                    cg.is_fully_directed(a, b)
                    and cg.is_fully_directed(b, c)
                    and cg.is_undirected(a, c)
                ):
                    _orient_edge_conflict_aware(cg, a, c)

        # R3
        kites = cg.find_kites()
        for i, j, k, target in kites:
            if (
                (
                    cg.is_fully_directed(j, target)
                    and cg.is_fully_directed(k, target)
                    and cg.is_undirected(i, target)
                )
                and cg.is_undirected(i, j)
                and cg.is_undirected(i, k)
            ):
                _orient_edge_conflict_aware(cg, i, target)

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
    show_progress: bool = False,
    n_jobs: int = -1,
    max_cond_set_size=None,
    skeleton_backend: str = "auto",
    **kwargs,
):
    """Run stable PC with the requested workers and return the graph in a result wrapper."""
    configure_conditioning(cg, max_cond_set_size)
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
    # Check each write. A later background knowledge checkpoint can correct the
    # direction but cannot recover a collider lost to a forbidden arrowhead.
    cg.bk_guard = orientation_guard_for(background_knowledge, node_names)
    protected_edges = ()
    if background_knowledge is not None:
        protected_edges = background_knowledge.required_pairs(
            node_names, phase=BKPhase.PRE_SEARCH
        )
    if verbose:
        print(f"Starting Parallel Skeleton Discovery with n_jobs={n_jobs}...")

    with profiler.time_block(
        "full_pc_algorithm", tags={"algo": "parallel", "n_jobs": n_jobs}
    ):
        skeleton_discovery(
            cg,
            alpha,
            stable,
            verbose,
            show_progress,
            n_jobs,
            protected_edges=protected_edges,
            skeleton_backend=skeleton_backend,
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
