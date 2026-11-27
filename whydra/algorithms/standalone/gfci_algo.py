"""GFCI: score-pattern initialization followed by CI refinement and PAG rules.

Modified R0 (C'/F') follows Ogarrio, Spirtes and Ramsey, PMLR 2016,
https://proceedings.mlr.press/v52/ogarrio16.html . Per-triple PAT decisions
are separate from established CI separators. Missing separators are searched
with CI evidence, and provisional score-pattern marks are reset to circles.

The Gaussian score contract remains local_score_BIC, not Tetrad SEM-BIC.
Reference comparisons must record both scores and resolved options. A single
matching example, or agreement with complete-graph initialization under a CI
oracle, does not establish full GFCI reference parity.
"""

import itertools
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from importlib import import_module
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from ...background_knowledge import BKPhase, MatrixEncoding, apply_bk_checkpoint
from ..graph_core import (
    CIT,
    CausalGraph,
    Edge,
    Endpoint,
    GeneralGraph,
    Node,
    fisher_z_from_corr,
    resolve_score_func,
)
from ..graph_core._optional_imports import Parallel, delayed
from ..graph_core.cache import BoundedCache, ci_cache_for
from ..graph_core.conditioning import configure_conditioning
from ..graph_core.validation import effective_n_jobs
from .ci_adapters import attach_counted_batch, attach_legacy_batch, ordered_ci_values
from .pag_rules import PAG_ARROW, qreach, udag2pag

_SCORE_EXPORTS = {
    "ges_search": ("..score_based.ges", "ges_search"),
    "local_parent_score": ("..score_based.scores", "local_parent_score"),
    "_local_parent_score": ("..score_based.scores", "local_parent_score"),
}


def _score_implementation(name):
    implementation = globals().get(name)
    if implementation is None:
        module_name, attribute = _SCORE_EXPORTS[name]
        implementation = getattr(import_module(module_name, __package__), attribute)
        globals()[name] = implementation
    return implementation


def __getattr__(name):
    if name not in _SCORE_EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return _score_implementation(name)


def __dir__():
    return sorted(set(globals()) | set(_SCORE_EXPORTS))


# =============================================================================
#  CPDAG extraction from the local score-search result
# =============================================================================


def _ges_cpdag_to_matrix(ges_graph, node_names=None):
    """Convert a local GeneralGraph CPDAG from GES to the NumPy PAG used for FCI orientation.

    The library uses causal-learn codes: 0 = NULL (no edge), -1 = TAIL (-),
    1 = ARROW (>), and 2 = CIRCLE (o).

    pag[x, y] is the endpoint at y on the edge between x and y.
    """
    if node_names is None:
        nodes = ges_graph.get_nodes()
        node_names = [n.get_name() for n in nodes]
    else:
        nodes = ges_graph.get_nodes()

    p = len(node_names)
    pag = np.zeros((p, p), dtype=int)

    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            node_i = nodes[i]
            node_j = nodes[j]
            edge = ges_graph.get_edge(node_i, node_j)
            if edge is not None:
                # Use Graph.get_endpoint(node, target) which returns endpoint AT node
                pag[i, j] = _endpoint_code(ges_graph.get_endpoint(node_j, node_i))
            else:
                pag[i, j] = 0

    return pag


def _endpoint_code(endpoint):
    """Return the numeric code for a causal-learn endpoint mark.

    The orientation rules accept the causal-learn Endpoint code directly.
    """
    return endpoint.value if hasattr(endpoint, "value") else int(endpoint)


def _get_cpdag_colliders(cpdag_graph, node_names):
    """
    Extract definite collider triples from a local CPDAG.

    Returns a set of (x, y, z) tuples where y is the collider (x *-> y <-* z).
    Indices are mapped through node_names.
    """
    nodes = cpdag_graph.get_nodes()
    name_to_idx = {n.get_name(): idx for idx, n in enumerate(nodes)}

    colliders = set()
    for y_node in nodes:
        adj_y = cpdag_graph.get_adjacent_nodes(y_node)
        for x_node, z_node in itertools.combinations(adj_y, 2):
            # External CPDAG implementations can reject an endpoint lookup;
            # retain evidence from the other triples in that case.
            with suppress(Exception):
                if (
                    not cpdag_graph.is_adjacent_to(x_node, z_node)
                    and cpdag_graph.get_endpoint(y_node, x_node) == Endpoint.ARROW
                    and cpdag_graph.get_endpoint(y_node, z_node) == Endpoint.ARROW
                ):
                    x = name_to_idx[x_node.get_name()]
                    y = name_to_idx[y_node.get_name()]
                    z = name_to_idx[z_node.get_name()]
                    colliders.add((x, y, z))
                    colliders.add((z, y, x))  # symmetric

    return colliders


# =============================================================================
#  Separating set search
# =============================================================================

# Small searches are faster through the scalar Fisher/cache path, especially
# before the first batch initializes numerical dispatch. Use bounded batches
# only once an individual conditioning order amortizes that overhead.
_GFCI_BATCH_MIN_CONDITIONS = 1024


def _get_sepset(x, y, adj_x, cit_func, alpha, depth, use_max_p):
    """
    Find a separating set for (x, y) within adj_x.

    Parameters
    ----------
    x, y : int
        Node indices.
    adj_x : list of int
        Candidate nodes for separating set (adjacency of x).
    cit_func : callable
        Independence test: cit_func(x, y, S) -> p_value.
    alpha : float
        Significance level.
    depth : int
        Maximum size of separating set to consider (-1 = unlimited).
    use_max_p : bool
        If True, use max-P approach (test all subsets, pick max p-value).
        If False, return the first separating set found.

    Returns
    -------
    set or None
        The separating set if found, None otherwise.
    """
    n_adj = len(adj_x)

    max_depth = n_adj if depth < 0 or depth > n_adj else min(depth, n_adj)

    best_sepset = None
    best_pval = -1.0

    def legacy_test(suff_stat, i, j, conditions):
        return cit_func(i, j, conditions)

    if hasattr(cit_func, "batch"):
        attach_legacy_batch(legacy_test, cit_func)

    for d in range(max_depth + 1):
        conditions = combinations(adj_x, d)
        if math.comb(n_adj, d) >= _GFCI_BATCH_MIN_CONDITIONS:
            # Edge searches already run concurrently. Keep numeric batches
            # within an edge serial, deferring failures until query consumption.
            values = ordered_ci_values(legacy_test, None, x, y, conditions)
        else:
            values = ((S, cit_func(x, y, list(S))) for S in conditions)
        for S, pval in values:
            if use_max_p:
                if pval > best_pval:
                    best_pval = pval
                    if pval >= alpha:
                        best_sepset = set(S)
            elif pval >= alpha:
                return set(S)

    if use_max_p and best_sepset is not None:
        return best_sepset

    return None


def _choose_adjacency_separator(x, y, cit_func, from_x, from_y, *, node_names):
    """Recheck both candidates and break p-value ties by size, then name."""
    if from_x is None:
        return from_y
    if from_y is None:
        return from_x
    score_x = cit_func(x, y, list(from_x))
    score_y = cit_func(x, y, list(from_y))
    if score_x != score_y:
        return from_x if score_x > score_y else from_y
    return min(
        from_x,
        from_y,
        key=lambda s: (len(s), sorted(node_names[i] for i in s)),
    )


def _sepset_subset_of_adjx_or_adjy(
    pag, x, y, cit_func, alpha, depth, use_max_p, *, node_names
):
    """Search the two endpoint neighborhoods in GFCI Algorithm 1, step B."""

    def search(source, target):
        neighbors = np.flatnonzero(pag[source]).tolist()
        pool = [v for v in neighbors if v != target]
        return _get_sepset(source, target, pool, cit_func, alpha, depth, use_max_p)

    from_x = search(x, y)
    from_y = search(y, x)
    return _choose_adjacency_separator(
        x, y, cit_func, from_x, from_y, node_names=node_names
    )


# =============================================================================
#  Utility: get current edges as list of (i, j)
# =============================================================================


def _get_edges(pag):
    """Return list of undirected edges (i, j) with i < j from the PAG."""
    p = pag.shape[0]
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if pag[i, j] != 0 or pag[j, i] != 0:
                edges.append((i, j))
    return edges


# =============================================================================
#  Collider copying from CPDAG
# =============================================================================

# =============================================================================
#  Possible-D-SEP computation (from FCI)
# =============================================================================


def _fill_missing_sepsets_gfci(
    pag,
    sepset_map,
    cpdag_colliders=(),
    *,
    cit_func=None,
    alpha=0.05,
    depth=-1,
    node_names=None,
):
    """Recover CI evidence, never synthetic per-pair collider decisions.

    GFCI Algorithm 1 requires a CI search for a missing separator when a rule
    needs one. Unknown pairs stay absent if finite-sample tests find none.
    The historical cpdag_colliders argument is retained for source compatibility.
    """
    if cit_func is None:
        return
    p = len(pag)
    if node_names is None:
        node_names = [f"X{i + 1}" for i in range(p)]
    for x in range(p):
        for z in range(x + 1, p):
            if pag[x, z] != 0 or pag[z, x] != 0 or (x, z) in sepset_map:
                continue
            sep = _sepset_subset_of_adjx_or_adjy(
                pag, x, z, cit_func, alpha, depth, False, node_names=node_names
            )
            if sep is None:
                sep = _get_sepset(
                    x,
                    z,
                    [v for v in range(p) if v not in (x, z)],
                    cit_func,
                    alpha,
                    depth,
                    False,
                )
            if sep is not None:
                sepset_map[x, z] = sepset_map[z, x] = set(sep)


def _modified_r0_triples(pag, pattern, cpdag_colliders, sepsets):
    """GFCI C'/F': PAT collider OR shielded in PAT and CI separator excludes y."""
    colliders = set(cpdag_colliders)
    result = set()
    for y in range(len(pag)):
        neighbors = np.flatnonzero(pag[y] != 0)
        for x, z in combinations(neighbors, 2):
            x, z = int(x), int(z)
            if pag[x, z] != 0 or pag[z, x] != 0:
                continue
            sep = sepsets.get((x, z), sepsets.get((z, x)))
            if (
                (x, y, z) in colliders
                or (z, y, x) in colliders
                or (pattern[x, z] != 0 and sep is not None and y not in sep)
            ):
                result.add((x, y, z))
    return result


def _possible_dsep(pag, x):
    """
    Compute Possible-D-SEP for node x in the PAG.

    Reuses qreach from the maintained FCI implementation.
    """
    amat = np.copy(pag)
    # qreach expects the amat encoding where non-zero = adjacency
    result = qreach(x, amat)
    return result


# =============================================================================
#  Main GFCI algorithm
# =============================================================================


def _ci_cache_key(i, j, S):
    a, b = (int(i), int(j)) if i <= j else (int(j), int(i))
    return a, b, tuple(sorted(int(v) for v in S))


def _worker_count(n_jobs):
    n_jobs = int(n_jobs)
    return n_jobs if n_jobs > 0 else (os.cpu_count() or 1)


def _resolve_gfci_backend(backend, n_vars, process_min_vars):
    requested = "auto" if backend in (None, "auto") else str(backend)
    if requested == "auto":
        effective = "processes" if int(n_vars) > int(process_min_vars) else "threads"
    elif requested in {"threads", "processes"}:
        effective = requested
    else:
        msg = "GFCI backend must be 'auto', 'threads', or 'processes'"
        raise ValueError(msg)
    return requested, effective


def _chunk_sequence(items, n_chunks):
    if not items:
        return []
    n_chunks = max(1, min(int(n_chunks), len(items)))
    chunk_size = math.ceil(len(items) / n_chunks)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


# Fisher-Z lives in graph_core so every backend computes it identically;
# a local copy here would silently drift from CIT.fisher_z.
_fisher_z_from_corr = fisher_z_from_corr


def _gfci_process_cit(corr_matrix, n_samples, x, y, S, cache, stats):
    key = _ci_cache_key(x, y, S)
    stats["calls"] += 1
    if cache is not None and key in cache:
        stats["hits"] += 1
        return cache[key]
    stats["misses"] += 1
    pval = _fisher_z_from_corr(
        corr_matrix,
        n_samples,
        x,
        y,
        key[2],
    )
    if cache is not None:
        cache[key] = pval
    return pval


def _gfci_adjacency_process_chunk(
    pag,
    edges,
    corr_matrix,
    n_samples,
    alpha,
    depth,
    use_max_p,
    use_ci_cache,
    *,
    node_names,
):
    cache = BoundedCache() if use_ci_cache else None
    stats = {"calls": 0, "hits": 0, "misses": 0}

    def cit_func(x, y, S):
        return _gfci_process_cit(
            corr_matrix,
            n_samples,
            x,
            y,
            S,
            cache,
            stats,
        )

    results = []
    for a, b in edges:
        sepset = _sepset_subset_of_adjx_or_adjy(
            pag,
            a,
            b,
            cit_func,
            alpha,
            depth,
            use_max_p,
            node_names=node_names,
        )
        results.append((a, b, sepset))
    return results, stats


def _pdsep_edge(graph, edge, ci, alpha, depth, use_max_p):
    """Return the earliest side with admissible evidence, including an empty set."""
    x, y = edge
    for anchor in edge:
        candidates = [v for v in _possible_dsep(graph, anchor) if v not in edge]
        found = _get_sepset(x, y, candidates, ci, alpha, depth, use_max_p)
        if found is not None:
            return x, y, found
    return x, y, None


def _gfci_pdsep_process_chunk(
    pag,
    edges,
    corr_matrix,
    n_samples,
    alpha,
    depth,
    use_max_p,
    use_ci_cache,
):
    cache = BoundedCache() if use_ci_cache else None
    stats = {"calls": 0, "hits": 0, "misses": 0}

    def cit_func(x, y, S):
        return _gfci_process_cit(
            corr_matrix,
            n_samples,
            x,
            y,
            S,
            cache,
            stats,
        )

    results = [
        _pdsep_edge(pag, edge, cit_func, alpha, depth, use_max_p) for edge in edges
    ]
    return results, stats


def _parallel_map(executor, fn, items):
    if executor is None:
        return [fn(item) for item in items]
    try:
        return list(executor.map(fn, items))
    except Exception:
        executor.shutdown(wait=False, cancel_futures=True)
        raise


def gfci_stable(
    cg,
    data,
    alpha=0.05,
    indep_test="fisherz",
    score_func="auto",
    use_max_p=True,
    depth=-1,
    n_jobs=1,
    verbose=False,
    start_from_complete_graph=False,
    unfVect=None,
    node_names=None,
    use_ci_cache=True,
    ci_backend=None,
    skeleton_backend=None,
    pdsep_backend=None,
    process_min_vars=150,
    background_knowledge=None,
    ges_max_operator_set=3,
    show_progress: bool = False,
    max_cond_set_size=None,
):
    """
    Run the GFCI (Greedy Fast Causal Inference) algorithm.

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_samples, n_features).
    alpha : float
        Significance level for independence tests.
    indep_test : str
        Independence test method ('fisherz', 'chisq', etc.).
    score_func : str
        Score function for the GES initial graph. 'auto' matches the score to
        the data type `indep_test` assumes (BDeu for discrete tests, Gaussian
        BIC otherwise); an explicit name such as 'local_score_BIC' is used as is.
    use_max_p : bool
        If True, select the highest p-value on each side. Otherwise, take
        the first admissible set on each side. In either mode, if the two
        candidates tie, prefer the smaller set, then the lexicographically
        earlier sorted node names.
    depth : int
        Maximum separating set size (-1 = unlimited). Controls complexity.
    verbose : bool
        Print progress information.
    start_from_complete_graph : bool
        If True, skip GES and start from a complete graph (like FCI).
    unfVect : set or None
        Set of ambiguous triples (underdetermined v-structures).
    node_names : list of str or None
        Stable names in data-column order. If None, uses the names in cg.
    ges_max_operator_set : int or None
        Maximum subset size for GES moves from x to y. Insert chooses from
        the undirected neighbours of y that are not adjacent to x. Delete
        chooses from those adjacent to x. Use None to consider all subset
        sizes, which can be expensive in dense graphs.

    Returns
    -------
    pag : np.ndarray
        PAG adjacency matrix in graph_core encoding.
        pag[x, y] = endpoint at y on edge x-y.
        0=NULL, -1=TAIL, 1=ARROW, 2=CIRCLE.
    """
    configure_conditioning(cg, max_cond_set_size)
    if max_cond_set_size is not None:
        depth = max_cond_set_size if depth < 0 else min(depth, max_cond_set_size)
    cg.G.metadata.update(
        effective_max_cond_set_size=depth if depth >= 0 else None,
        depth_limited=depth >= 0,
    )
    total_t0 = time.perf_counter()
    data = np.asarray(data)
    _n, p = data.shape
    n_jobs_eff = effective_n_jobs(n_jobs)
    skeleton_backend_requested, skeleton_backend_eff = _resolve_gfci_backend(
        skeleton_backend if skeleton_backend is not None else ci_backend,
        p,
        process_min_vars,
    )
    pdsep_backend_requested, pdsep_backend_eff = _resolve_gfci_backend(
        pdsep_backend if pdsep_backend is not None else ci_backend,
        p,
        process_min_vars,
    )
    use_process_skeleton = (
        n_jobs_eff != 1
        and indep_test == "fisherz"
        and skeleton_backend_eff == "processes"
    )
    use_process_pdsep = (
        n_jobs_eff != 1 and indep_test == "fisherz" and pdsep_backend_eff == "processes"
    )

    if unfVect is None:
        unfVect = set()

    if node_names is None:
        node_names = [node.name for node in cg.nodes]

    protected_edges = ()
    if background_knowledge is not None:
        protected_edges = background_knowledge.required_pairs(
            node_names, phase=BKPhase.PRE_SEARCH
        )
    protected_pairs = {frozenset((int(a), int(b))) for a, b in protected_edges}

    score_func = resolve_score_func(score_func, indep_test)
    cit = (
        cg.test
        if isinstance(cg.test, CIT)
        and cg.test.method == indep_test
        and np.array_equal(cg.test.data, data)
        else CIT(data, method=indep_test)
    )
    cit.max_cond_set_size = max_cond_set_size
    ci_cache = ci_cache_for(cit) if use_ci_cache else None
    ci_inflight = {}
    stats_lock = threading.Lock()
    phase = {"name": "adjacency"}
    ci_stats = {
        "total": 0,
        "adjacency": 0,
        "pdsep": 0,
        "n_jobs": int(n_jobs_eff),
        "skeleton_backend": "processes" if use_process_skeleton else "threads",
        "pdsep_backend": "processes" if use_process_pdsep else "threads",
        "skeleton_backend_requested": skeleton_backend_requested,
        "pdsep_backend_requested": pdsep_backend_requested,
        "process_min_vars": int(process_min_vars),
        "n_vars": int(p),
    }
    if use_ci_cache:
        ci_stats["ci_cache_hits"] = 0
        ci_stats["ci_cache_misses"] = 0

    def cit_func(x, y, S):
        key = _ci_cache_key(x, y, S)
        with stats_lock:
            ci_stats["total"] += 1
            ci_stats[phase["name"]] += 1

        if ci_cache is None:
            return cit(x, y, list(key[2]))

        while True:
            with stats_lock:
                cached = ci_cache.get(key)
                if cached is not None:
                    ci_stats["ci_cache_hits"] += 1
                    return cached
                event = ci_inflight.get(key)
                if event is None:
                    event = threading.Event()
                    ci_inflight[key] = event
                    ci_stats["ci_cache_misses"] += 1
                    owner = True
                else:
                    owner = False
            if owner:
                break
            event.wait()

        try:
            pval = cit(x, y, list(key[2]))
            with stats_lock:
                ci_cache[key] = pval
            return pval
        finally:
            with stats_lock:
                ci_inflight.pop(key, None)
                event.set()

    attach_counted_batch(
        cit_func, cit, ci_stats, phase, cache=ci_cache, lock=stats_lock
    )

    # --- Step 1: Get initial CPDAG (GES or complete graph) ---
    ges_t0 = time.perf_counter()
    ges_graph = None
    if start_from_complete_graph:
        if verbose:
            print("[GFCI] Step 1: Starting from complete graph")
        pag = np.full((p, p), Endpoint.CIRCLE.value, dtype=int)  # all edges o-o
        np.fill_diagonal(pag, 0)
        cpdag_colliders = set()
    else:
        if verbose:
            print("[GFCI] Step 1: Running GES for initial CPDAG...")
        ges_graph, score_val = _score_implementation("ges_search")(
            data,
            score_func,
            node_names,
            max_operator_set=ges_max_operator_set,
            verbose=verbose,
            stats=ci_stats,
            n_jobs=n_jobs_eff,
        )

        if verbose:
            print(f"  GES complete. Score: {score_val}")

        pag = _ges_cpdag_to_matrix(ges_graph, node_names)
        cpdag_colliders = _get_cpdag_colliders(ges_graph, node_names)

        if verbose:
            n_edges = len(_get_edges(pag))
            n_colliders = len(cpdag_colliders) // 2
            print(f"  CPDAG: {n_edges} edges, {n_colliders} definite colliders")
    ci_stats["time_ges_sec"] = time.perf_counter() - ges_t0
    ci_stats["ges_edges"] = len(_get_edges(pag))
    ci_stats["ges_colliders"] = int(len(cpdag_colliders) // 2)

    # PRE_SEARCH must give the same guarantee here that it gives PC/FCI: a
    # required adjacency always survives, even if the earlier search phase
    # (here, GES) didn't find it. For PC/FCI this is automatic because the
    # checkpoint runs on the still-fully-connected graph before skeleton
    # discovery starts. GES already produced a sparse CPDAG by this point,
    # so a missing required edge would otherwise be flagged as a conflict
    # instead of restored (``apply_matrix`` only restores via ``previous``).
    # ``previous`` is a fully-connected o-o graph so the restore always has
    # somewhere to draw the edge from.
    complete_pag = np.full((p, p), Endpoint.CIRCLE.value, dtype=int)
    np.fill_diagonal(complete_pag, 0)
    apply_bk_checkpoint(
        background_knowledge,
        pag,
        node_names,
        phase=BKPhase.PRE_SEARCH,
        checkpoint="before_skeleton",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="pag",
        previous=complete_pag,
    )

    # PAT is retained separately for modified R0. Its directed marks must not
    # leak into the provisional PAG used for Possible-D-SEP.
    pag[pag != 0] = Endpoint.CIRCLE.value
    sepset_map = {}
    executor = None
    if n_jobs_eff != 1 and (not use_process_skeleton or not use_process_pdsep):
        executor = ThreadPoolExecutor(max_workers=_worker_count(n_jobs_eff))

    # --- Step 2: Extra edge removal (subset of adj(x) or adj(y)) ---
    if verbose:
        print("[GFCI] Step 2: Extra edge removal (subset of adjacency)...")

    adjacency_t0 = time.perf_counter()
    edges_before = _get_edges(pag)
    adjacency_snapshot = np.array(pag, copy=True)

    def adjacency_worker(edge):
        a, b = edge
        sepset = _sepset_subset_of_adjx_or_adjy(
            adjacency_snapshot,
            a,
            b,
            cit_func,
            alpha,
            depth,
            use_max_p,
            node_names=node_names,
        )
        return a, b, sepset

    if use_process_skeleton:
        chunks = _chunk_sequence(
            edges_before,
            max(_worker_count(n_jobs_eff) * 4, 1),
        )
        chunk_results = Parallel(
            n_jobs=n_jobs_eff,
            backend="loky",
            max_nbytes="10K",
        )(
            delayed(_gfci_adjacency_process_chunk)(
                adjacency_snapshot,
                chunk,
                cit.corr_matrix,
                cit.n_samples,
                alpha,
                depth,
                use_max_p,
                use_ci_cache,
                node_names=node_names,
            )
            for chunk in chunks
        )
        adjacency_results = [
            result for results, _stats in chunk_results for result in results
        ]
        for _results, process_stats in chunk_results:
            ci_stats["total"] += int(process_stats["calls"])
            ci_stats["adjacency"] += int(process_stats["calls"])
            if use_ci_cache:
                ci_stats["ci_cache_hits"] += int(process_stats["hits"])
                ci_stats["ci_cache_misses"] += int(process_stats["misses"])
    else:
        adjacency_results = _parallel_map(executor, adjacency_worker, edges_before)
    for a, b, sepset in adjacency_results:
        if sepset is not None and frozenset((int(a), int(b))) not in protected_pairs:
            pag[a, b] = pag[b, a] = 0
            sepset_map[(a, b)] = sepset
            sepset_map[(b, a)] = sepset
    ci_stats["time_adjacency_sec"] = time.perf_counter() - adjacency_t0
    ci_stats["time_skeleton_sec"] = (
        ci_stats["time_ges_sec"] + ci_stats["time_adjacency_sec"]
    )
    ci_stats["adjacency_edges_scanned"] = len(edges_before)
    ci_stats["adjacency_edges_removed"] = int(len(edges_before) - len(_get_edges(pag)))

    if verbose:
        n_after = len(_get_edges(pag))
        print(f"  Removed {len(edges_before) - n_after} edges ({n_after} remaining)")

    # C': orient from the original score pattern and actual CI evidence,
    # before constructing Possible-D-SEP on the partially oriented graph.
    pattern = (
        _ges_cpdag_to_matrix(ges_graph, node_names)
        if ges_graph is not None
        else np.ones_like(pag)
    )
    _fill_missing_sepsets_gfci(
        pag,
        sepset_map,
        cit_func=cit_func,
        alpha=alpha,
        depth=depth,
        node_names=node_names,
    )
    initial_colliders = _modified_r0_triples(pag, pattern, cpdag_colliders, sepset_map)
    for x, y, z in initial_colliders:
        pag[x, y] = pag[z, y] = PAG_ARROW

    # --- Step 4: Possible-D-SEP removal ---
    if verbose:
        print("[GFCI] Step 4: Possible-D-SEP removal...")

    phase["name"] = "pdsep"
    pdsep_t0 = time.perf_counter()
    edges_before_pdsep = _get_edges(pag)
    pdsep_snapshot = np.array(pag, copy=True)

    def pdsep_worker(edge):
        return _pdsep_edge(pdsep_snapshot, edge, cit_func, alpha, depth, use_max_p)

    if use_process_pdsep:
        chunks = _chunk_sequence(
            edges_before_pdsep,
            max(_worker_count(n_jobs_eff) * 4, 1),
        )
        chunk_results = Parallel(
            n_jobs=n_jobs_eff,
            backend="loky",
            max_nbytes="10K",
        )(
            delayed(_gfci_pdsep_process_chunk)(
                pdsep_snapshot,
                chunk,
                cit.corr_matrix,
                cit.n_samples,
                alpha,
                depth,
                use_max_p,
                use_ci_cache,
            )
            for chunk in chunks
        )
        pdsep_results = [
            result for results, _stats in chunk_results for result in results
        ]
        for _results, process_stats in chunk_results:
            ci_stats["total"] += int(process_stats["calls"])
            ci_stats["pdsep"] += int(process_stats["calls"])
            if use_ci_cache:
                ci_stats["ci_cache_hits"] += int(process_stats["hits"])
                ci_stats["ci_cache_misses"] += int(process_stats["misses"])
    else:
        pdsep_results = _parallel_map(executor, pdsep_worker, edges_before_pdsep)
    for a, b, sepset in pdsep_results:
        if sepset is not None and frozenset((int(a), int(b))) not in protected_pairs:
            pag[a, b] = pag[b, a] = 0
            sepset_map[(a, b)] = sepset
            sepset_map[(b, a)] = sepset
    ci_stats["time_pdsep_sec"] = time.perf_counter() - pdsep_t0
    ci_stats["pdsep_edges_scanned"] = len(edges_before_pdsep)
    ci_stats["pdsep_edges_removed"] = int(
        len(edges_before_pdsep) - len(_get_edges(pag))
    )
    if executor is not None:
        executor.shutdown(wait=True)
        executor = None

    if verbose:
        n_after = len(_get_edges(pag))
        print(
            f"  Removed {len(edges_before_pdsep) - n_after} edges ({n_after} remaining)"
        )

    # E/F': discard provisional orientations, then use modified R0 again.
    _fill_missing_sepsets_gfci(
        pag,
        sepset_map,
        cit_func=cit_func,
        alpha=alpha,
        depth=depth,
        node_names=node_names,
    )
    final_colliders = _modified_r0_triples(pag, pattern, cpdag_colliders, sepset_map)

    # --- Step 6: Full FCI orientation (R0-R10) ---
    if verbose:
        print("[GFCI] Step 6: Full FCI orientation (R0-R10)...")
    orient_t0 = time.perf_counter()
    rule_stats = {}
    gfci_rules = [True] * 10
    pag = udag2pag(
        pag,
        sepset_map,
        p,
        unfVect=unfVect,
        rules=gfci_rules,
        orientCollider=True,
        collider_triples=final_colliders,
        rule_stats=rule_stats,
        background_knowledge=background_knowledge,
        node_names=node_names
        if node_names is not None
        else [node.name for node in cg.nodes],
    )
    ci_stats["time_orient_sec"] = time.perf_counter() - orient_t0
    ci_stats["orientation_rule_updates"] = {
        key: int(value) for key, value in rule_stats.items()
    }

    if verbose:
        _summarize_pag(pag)

    cg.G.graph = pag
    cg.gfci_sepset = sepset_map
    if ci_cache is not None:
        ci_stats["ci_cache_size"] = len(ci_cache)
    ci_stats["skeleton"] = int(ci_stats["adjacency"])
    ci_stats["pdsep_ci_tests"] = int(ci_stats["pdsep"])
    ci_stats["time_total_sec"] = time.perf_counter() - total_t0
    cg.ci_stats = ci_stats
    return pag


def gfci(
    data,
    alpha=0.05,
    indep_test="fisherz",
    score_func="auto",
    use_max_p=True,
    depth=-1,
    n_jobs=1,
    verbose=False,
    start_from_complete_graph=False,
    unfVect=None,
    node_names=None,
    use_ci_cache=True,
    ci_backend=None,
    skeleton_backend=None,
    pdsep_backend=None,
    process_min_vars=150,
    background_knowledge=None,
    max_cond_set_size=None,
):
    """Return the standard graph_core endpoint matrix through the convenience API."""
    data = np.asarray(data)
    if node_names is None:
        node_names = [f"X{i + 1}" for i in range(data.shape[1])]
    cg = CausalGraph(data.shape[1], list(node_names))
    return gfci_stable(
        cg,
        data,
        alpha=alpha,
        indep_test=indep_test,
        score_func=score_func,
        use_max_p=use_max_p,
        depth=depth,
        max_cond_set_size=max_cond_set_size,
        n_jobs=n_jobs,
        verbose=verbose,
        start_from_complete_graph=start_from_complete_graph,
        unfVect=unfVect,
        node_names=node_names,
        use_ci_cache=use_ci_cache,
        ci_backend=ci_backend,
        skeleton_backend=skeleton_backend,
        pdsep_backend=pdsep_backend,
        process_min_vars=process_min_vars,
        background_knowledge=background_knowledge,
    )


def _summarize_pag(pag):
    """Print a summary of the PAG edge types."""
    p = pag.shape[0]
    n_edges = 0
    edge_types = {
        "-->": 0,
        "<--": 0,
        "<->": 0,
        "o->": 0,
        "<-o": 0,
        "o-o": 0,
        "---": 0,
        "?": 0,
    }

    tail = Endpoint.TAIL.value
    arrow = Endpoint.ARROW.value
    circle = Endpoint.CIRCLE.value

    for i in range(p):
        for j in range(i + 1, p):
            ei = pag[j, i]  # endpoint at i
            ej = pag[i, j]  # endpoint at j
            if ei == 0 and ej == 0:
                continue
            n_edges += 1
            if ei == tail and ej == arrow:
                edge_types["-->"] += 1
            elif ei == arrow and ej == tail:
                edge_types["<--"] += 1
            elif ei == arrow and ej == arrow:
                edge_types["<->"] += 1
            elif ei == circle and ej == arrow:
                edge_types["o->"] += 1
            elif ei == arrow and ej == circle:
                edge_types["<-o"] += 1
            elif ei == circle and ej == circle:
                edge_types["o-o"] += 1
            elif ei == tail and ej == tail:
                edge_types["---"] += 1
            else:
                edge_types["?"] += 1

    parts = [f"{k}={v}" for k, v in edge_types.items() if v > 0]
    print(f"  PAG: {n_edges} edges ({', '.join(parts)})")


# =============================================================================
#  Convenience: matrix-to-Graph conversion
# =============================================================================


def pag_to_general_graph(pag, node_names=None):
    """
    Convert a graph_core PAG matrix to the local GeneralGraph.

    Parameters
    ----------
    pag : np.ndarray
        PAG matrix in graph_core encoding:
        0=NULL, -1=TAIL, 1=ARROW, 2=CIRCLE.
    node_names : list of str or None
        Node names.

    Returns
    -------
    GeneralGraph
    """
    p = pag.shape[0]
    if node_names is None:
        node_names = [f"X{i + 1}" for i in range(p)]

    nodes = [Node(name) for name in node_names]
    graph = GeneralGraph(nodes)

    endpoint_map = {
        0: Endpoint.NULL,
        Endpoint.TAIL.value: Endpoint.TAIL,
        Endpoint.ARROW.value: Endpoint.ARROW,
        Endpoint.CIRCLE.value: Endpoint.CIRCLE,
    }

    for i in range(p):
        for j in range(i + 1, p):
            mark_at_i = int(pag[j, i])  # endpoint at i
            mark_at_j = int(pag[i, j])  # endpoint at j
            if mark_at_i != 0 or mark_at_j != 0:
                edge = Edge(
                    nodes[i],
                    nodes[j],
                    endpoint_map.get(mark_at_i, Endpoint.NULL),
                    endpoint_map.get(mark_at_j, Endpoint.NULL),
                )
                graph.add_edge(edge)

    return graph


# These score functions were historically included by ``import *``.
if TYPE_CHECKING:
    from ..score_based.ges import ges_search
    from ..score_based.scores import local_parent_score

__all__ = [
    "CIT",
    "PAG_ARROW",
    "BKPhase",
    "BoundedCache",
    "CausalGraph",
    "Edge",
    "Endpoint",
    "GeneralGraph",
    "MatrixEncoding",
    "Node",
    "Parallel",
    "ThreadPoolExecutor",
    "apply_bk_checkpoint",
    "attach_counted_batch",
    "attach_legacy_batch",
    "ci_cache_for",
    "combinations",
    "configure_conditioning",
    "delayed",
    "effective_n_jobs",
    "fisher_z_from_corr",
    "ges_search",
    "gfci",
    "gfci_stable",
    "import_module",
    "itertools",
    "local_parent_score",
    "math",
    "np",
    "ordered_ci_values",
    "os",
    "pag_to_general_graph",
    "qreach",
    "resolve_score_func",
    "threading",
    "time",
    "udag2pag",
]
