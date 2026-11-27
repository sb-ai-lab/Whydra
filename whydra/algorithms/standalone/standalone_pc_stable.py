"""Reference PC-stable components: skeleton, colliders and Meek rules."""

import time
from itertools import combinations, permutations
from typing import TYPE_CHECKING

import numpy as np

from ...background_knowledge import (
    BKPhase,
    MatrixEncoding,
    apply_bk_checkpoint,
    bk_allows_edge,
    orientation_guard_for,
)
from ..graph_core import CausalGraph, Edge, Endpoint
from ..graph_core._optional_imports import tqdm_auto as tqdm
from ..graph_core.conditioning import bounded_size, configure_conditioning
from ..graph_core.cpu_graph_convert import causal_graph_to_causallearn
from .profiler import profiler

if TYPE_CHECKING:
    from causallearn.graph.GeneralGraph import GeneralGraph as CLGeneralGraph
from .skeleton import (
    _batch_graph,
    _ci_values,
    _depth_work,
    _marginal_values,
    _record_execution,
)


def skeleton_discovery(
    cg: CausalGraph,
    alpha: float,
    stable: bool,
    verbose: bool,
    show_progress: bool,
    protected_edges=(),
):
    """
    Stage 1: Find the skeleton.

    ``protected_edges`` contains the pairs required by background knowledge.
    Keep those edges and do not record separating sets for them: the nodes
    remain adjacent, so recording a set would interfere with collider orientation.

    Parameters
    ----------
    cg : CausalGraph
        Causal graph carrying the CI test.
    alpha : float
        Significance level.
    stable : bool
        Whether to test all conditioning sets at each depth.
    verbose : bool
        Verbosity flag.
    show_progress : bool
        Whether to display progress.
    protected_edges : collection of node pairs
        Pairs that background knowledge keeps adjacent.

    Returns
    -------
    CausalGraph
        Input graph with its skeleton and separating unions updated.
    """
    no_of_var = len(cg.nodes)
    protected_edges = {frozenset((int(x), int(y))) for x, y in protected_edges}
    depth = -1
    pbar = tqdm(total=no_of_var) if show_progress else None

    with profiler.time_block("skeleton_discovery_total", tags={"algo": "sequential"}):
        while bounded_size(cg, cg.max_degree() - 1) > depth:
            depth += 1
            _record_execution(
                cg, "skeleton_pc", depth, _depth_work(cg.G.graph, depth), 1, 1, "serial"
            )
            edge_removal = []
            if pbar is not None:
                pbar.reset()
                pbar.set_description(f"Depth={depth}, max_degree={cg.max_degree()}")

            with profiler.time_block(
                f"skeleton_discovery_depth_{depth}",
                tags={"algo": "sequential", "depth": depth},
            ):
                for x in range(no_of_var):
                    if pbar is not None:
                        pbar.update()

                    Neigh_x = cg.neighbors(x)
                    if len(Neigh_x) < depth:
                        continue

                    if stable and depth == 0 and _batch_graph(cg):
                        for y, p_val in _marginal_values(cg, x, Neigh_x, alpha):
                            if (
                                p_val > alpha
                                and frozenset((int(x), int(y))) not in protected_edges
                            ):
                                edge_removal.extend(((x, y), (y, x)))
                                _append_sepset(cg, x, y, ())
                        continue

                    for y in Neigh_x:
                        # Test the pair even if (x, y) is already marked in edge_removal.
                        # Stable mode needs the union of the indices from all separating sets S.

                        Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))

                        found = False
                        members = set()
                        protected = frozenset((int(x), int(y))) in protected_edges
                        for S, p_val in _ci_values(
                            cg,
                            x,
                            y,
                            combinations(Neigh_x_noy, depth),
                            alpha,
                            allow_batch=stable and not protected,
                        ):
                            if p_val > alpha:
                                if frozenset((int(x), int(y))) in protected_edges:
                                    break
                                if not stable:
                                    edge = cg.G.get_edge(cg.nodes[x], cg.nodes[y])
                                    if edge:
                                        cg.G.remove_edge(edge)
                                    _append_sepset(cg, x, y, S)
                                    break
                                # orient_colliders uses the union of the separating sets found at this depth.
                                if not found:
                                    edge_removal.append((x, y))
                                    edge_removal.append((y, x))
                                    found = True
                                members.update(int(v) for v in S)

                        if found:
                            _append_sepset(cg, x, y, members)

                if stable:
                    for x, y in set(edge_removal):
                        if frozenset((int(x), int(y))) in protected_edges:
                            continue
                        edge = cg.G.get_edge(cg.nodes[x], cg.nodes[y])
                        if edge:
                            cg.G.remove_edge(edge)

            if pbar is not None:
                pbar.refresh()

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


def _prof_add(prof, prof_n, name, dt, n=1):
    if prof is None:
        return
    prof[name] = prof.get(name, 0.0) + dt
    if prof_n is not None:
        prof_n[name] = prof_n.get(name, 0) + n


def orient_colliders(
    cg: CausalGraph, priority: int = 2, *, prof=None, prof_n=None, extra=None
):
    """Orient unshielded colliders (V-structures) in the second stage.

    Priority 2 conservatively prioritizes existing colliders.
    """
    t0 = time.perf_counter()
    triples = cg.find_unshielded_triples()
    _prof_add(prof, prof_n, "coll_find", time.perf_counter() - t0)

    t0 = time.perf_counter()
    n_hit = 0
    for x, y, z in triples:
        sepsets = cg.sepset[x, z]
        if sepsets is None:
            continue

        is_collider = True
        for S in sepsets:
            if y in S:
                is_collider = False
                break

        if is_collider:
            # Priority 2: Create a collider only if it is consistent with existing edges.
            conflict_x = cg.is_fully_directed(y, x)
            conflict_z = cg.is_fully_directed(y, z)

            if not conflict_x and not conflict_z:
                from warnings import warn

                from ...background_knowledge import find_directed_cycle

                candidate = cg.G.graph.copy()
                guard = getattr(cg, "bk_guard", None)
                allowed = all(
                    bk_allows_edge(guard, candidate, source, y, -1, 1)
                    for source in (x, z)
                )
                candidate[x, y] = candidate[z, y] = 1
                candidate[y, x] = candidate[y, z] = -1
                if allowed and find_directed_cycle(candidate, -1, 1) is None:
                    cg.G.graph[...] = candidate
                    n_hit += 1
                else:
                    warn(
                        f"Collider ({x}, {y}, {z}) rejected atomically: cycle or background-knowledge conflict",
                        RuntimeWarning,
                        stacklevel=2,
                    )
    _prof_add(prof, prof_n, "coll_apply", time.perf_counter() - t0)
    if extra is not None:
        extra["coll_triples"] = extra.get("coll_triples", 0) + len(triples)
        extra["coll_hits"] = extra.get("coll_hits", 0) + n_hit


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


def _orient_edge_conflict_aware(cg, source, target):
    """Try to orient source -> target when applying Meek's rules."""
    if cg.is_fully_directed(source, target):
        return None
    if cg.is_bidirected(source, target):
        return None

    if cg.is_fully_directed(target, source):
        # Conflict -> Bi-directed
        edge = cg.G.get_edge(cg.nodes[source], cg.nodes[target])
        if edge:
            cg.G.remove_edge(edge)
            cg.G.add_edge(
                Edge(cg.nodes[source], cg.nodes[target], Endpoint.ARROW, Endpoint.ARROW)
            )
        return None

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
        return None


def apply_meek_rules(
    cg: CausalGraph, background_knowledge=None, *, prof=None, prof_n=None, extra=None
):
    """Apply Meek rules R1–R3 with the backend selected by ``extra['meek']``.

    The reference Python backend supports background knowledge. The vectorized NumPy,
    torch and CUDA backends do not support it.

    Stop when the matrix no longer changes. Checking only whether a rule matched would
    miss updates made by background knowledge between passes.
    """
    kind = extra.get("meek") if extra is not None else None
    if background_knowledge is not None:
        kind = "python"
    if kind in (None, "python"):
        return apply_meek_rules_python(
            cg, background_knowledge, prof=prof, prof_n=prof_n, extra=extra
        )
    if kind in ("torch", "cuda"):
        from ..graph_core.meek_torch import meek_orient

        t0 = time.perf_counter()
        stats = meek_orient(cg.G.graph, device="cuda")
    else:
        from ..graph_core.meek_numpy import meek_orient

        t0 = time.perf_counter()
        stats = meek_orient(cg.G.graph)
    _prof_add(prof, prof_n, "meek_" + kind, time.perf_counter() - t0)
    if extra is not None:
        extra["meek_iters"] = extra.get("meek_iters", 0) + stats["meek_iters"]
        extra["meek_orients"] = extra.get("meek_orients", 0) + stats["meek_orients"]
        extra["meek_calls"] = extra.get("meek_calls", 0) + 1
        if stats["meek_orients"] == 0:
            extra["meek_fixed"] = extra.get("meek_fixed", 0) + 1
    return cg


def apply_meek_rules_python(
    cg: CausalGraph, background_knowledge=None, *, prof=None, prof_n=None, extra=None
):
    """
    Stage 3: Apply Meek's orientation rules.

    Stop when the matrix actually stops changing. A flag indicating that a
    rule matched would miss changes made by background knowledge between
    passes when testing for convergence.
    """
    loop = True
    n_iters = 0
    n_orients = 0
    while loop:
        before_pass = cg.G.graph.copy()
        n_iters += 1

        t0 = time.perf_counter()
        triples = cg.find_unshielded_triples()
        _prof_add(prof, prof_n, "find_triples", time.perf_counter() - t0)
        t0 = time.perf_counter()
        for i, j, k in triples:
            if cg.is_fully_directed(i, j) and cg.is_undirected(j, k):
                _orient_edge_conflict_aware(cg, j, k)
                n_orients += 1
            elif cg.is_fully_directed(k, j) and cg.is_undirected(j, i):
                _orient_edge_conflict_aware(cg, j, i)
                n_orients += 1
        _prof_add(prof, prof_n, "r1_scan", time.perf_counter() - t0)

        t0 = time.perf_counter()
        triangles = cg.find_triangles()
        _prof_add(prof, prof_n, "find_triangles", time.perf_counter() - t0)
        t0 = time.perf_counter()
        for i, j, k in triangles:
            nodes_tri = [i, j, k]
            for a, b, c in permutations(nodes_tri, 3):
                if (
                    cg.is_fully_directed(a, b)
                    and cg.is_fully_directed(b, c)
                    and cg.is_undirected(a, c)
                ):
                    _orient_edge_conflict_aware(cg, a, c)
                    n_orients += 1
        _prof_add(prof, prof_n, "r2_scan", time.perf_counter() - t0)

        t0 = time.perf_counter()
        kites = cg.find_kites()
        _prof_add(prof, prof_n, "find_kites", time.perf_counter() - t0)
        t0 = time.perf_counter()
        for i, j, k, leaf in kites:
            if (
                (
                    cg.is_fully_directed(j, leaf)
                    and cg.is_fully_directed(k, leaf)
                    and cg.is_undirected(i, leaf)
                )
                and cg.is_undirected(i, j)
                and cg.is_undirected(i, k)
            ):
                _orient_edge_conflict_aware(cg, i, leaf)
                n_orients += 1
        _prof_add(prof, prof_n, "r3_scan", time.perf_counter() - t0)
        if extra is not None:
            extra["n_triples"] = extra.get("n_triples", 0) + len(triples)
            extra["n_triangles"] = extra.get("n_triangles", 0) + len(triangles)
            extra["n_kites"] = extra.get("n_kites", 0) + len(kites)

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

    if extra is not None:
        extra["meek_iters"] = extra.get("meek_iters", 0) + n_iters
        extra["meek_orients"] = extra.get("meek_orients", 0) + n_orients
        extra["meek_calls"] = extra.get("meek_calls", 0) + 1
        if n_orients == 0:
            extra["meek_fixed"] = extra.get("meek_fixed", 0) + 1

    return cg


def convert_to_causallearn_graph(local_cg: CausalGraph) -> "CLGeneralGraph":
    """Return a local Whydra graph in causal-learn's matrix encoding."""
    return causal_graph_to_causallearn(local_cg)


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
    max_cond_set_size=None,
    **kwargs,
):
    """Run sequential PC-stable and return the oriented CPDAG."""
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

    with profiler.time_block("full_pc_algorithm", tags={"algo": "sequential"}):
        skeleton_discovery(
            cg, alpha, stable, verbose, show_progress, protected_edges=protected_edges
        )
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
        def __init__(self, g):
            self.G = g

    return convert_to_general_graph(cg)
