"""Greedy Equivalence Search (Chickering 2002).

A completed partially directed acyclic graph (CPDAG) represents each
Markov equivalence class. This lets the search leave an edge undirected
when both orientations belong to the same class.

Starting with an empty graph, the forward phase considers Insert moves.
The backward phase then considers Delete moves to prune edges. After each
accepted move, the algorithm computes the CPDAG of the new class. The
estimated class depends on the data, the score, and the search limits.
"""

import itertools
import time
from contextlib import ExitStack

import numpy as np

from . import pdag
from . import scores as _score_module
from ._ges_scan import (
    ForwardCandidates,
    GraphView,
    MaskedScores,
    backward_rows,
    best_candidate,
)
from .pdag import (
    complete_pdag,
    cpdag_matrix_to_general_graph,
    exists_semidirected_path,
    is_clique,
    is_undirected,
    na_yx,
    parents,
    pdag_to_dag,
)
from .scores import LocalScoreCache

# Retain the geometry helpers historically exported by this module.
adjacent = pdag.adjacent
is_adjacent = pdag.is_adjacent
neighbours = pdag.neighbours


def _record_subset_pool(size, max_size, *, statistics=None):
    """Record a logical pool even when its candidate row is reused."""
    if max_size is not None and (not isinstance(max_size, int) or max_size < 0):
        msg = "max_operator_set must be None or a nonnegative integer"
        raise ValueError(msg)
    truncated = max_size is not None and size > max_size
    if statistics is not None:
        statistics["truncated_candidate_pools"] = statistics.get(
            "truncated_candidate_pools", 0
        ) + int(truncated)
        statistics["largest_candidate_pool"] = max(
            statistics.get("largest_candidate_pool", 0), size
        )
    if truncated:
        import warnings

        warnings.warn(
            "GES candidate pool truncated by max_operator_set; use None for complete enumeration",
            RuntimeWarning,
            stacklevel=2,
        )


def _subsets(items, max_size, *, statistics=None):
    """All subsets of ``items`` up to ``max_size``, smallest first."""
    items = sorted(items)
    _record_subset_pool(len(items), max_size, statistics=statistics)
    limit = len(items) if max_size is None else min(len(items), max_size)
    for size in range(limit + 1):
        yield from itertools.combinations(items, size)


def _record_unlimited_summary(count, largest, *, statistics=None):
    """Exact diagnostic replay for unlimited, already enumerated pools."""
    if count and statistics is not None:
        statistics.setdefault("truncated_candidate_pools", 0)
        statistics["largest_candidate_pool"] = max(
            statistics.get("largest_candidate_pool", 0), largest
        )


# ---------------------------------------------------------------------------
#  Insert(x, y, T)
# ---------------------------------------------------------------------------


def insert_is_valid(C, x, y, T) -> bool:
    """Check whether ``Insert(x, y, T)`` is valid (Chickering 2002, Theorem 15).

    Form the union of ``T`` and the undirected neighbours of ``y`` adjacent
    to ``x`` (the set ``NA_yx``). The union must form a clique and intersect
    every semi-directed path from ``y`` to ``x``.
    """
    block = na_yx(C, y, x) | set(T)
    if not is_clique(C, block):
        return False
    return not exists_semidirected_path(C, y, x, block=block)


def insert_delta(C, cache, x, y, T) -> float:
    """Return the change in the local score of y for the proposed insertion."""
    base = parents(C, y) | na_yx(C, y, x) | set(T)
    return cache.score(y, base | {x}) - cache.score(y, base)


def apply_insert(C, x, y, T):
    """Add ``x -> y`` and orient every ``t -- y`` as ``t -> y``."""
    C[x, y] = 1
    C[y, x] = 0
    for t in T:
        C[y, t] = 0
    return C


# ---------------------------------------------------------------------------
#  Delete(x, y, H)
# ---------------------------------------------------------------------------


def delete_is_valid(C, x, y, H) -> bool:
    r"""Check whether ``NA_yx \ H`` is a clique, as required for ``Delete(x, y, H)`` (Theorem 17)."""
    return is_clique(C, na_yx(C, y, x) - set(H))


def delete_delta(C, cache, x, y, H) -> float:
    """Return the change in the local score of y for the proposed deletion.

    Let ``S`` contain the parents of ``y`` and the members of ``NA_yx``
    outside ``H``, excluding ``x`` from the combined set. The change is
    ``score(y, S) - score(y, S | {x})``. Including ``x`` explicitly in the
    second score also handles an undirected edge, where ``x`` belongs to
    neither the parents of ``y`` nor ``NA_yx``.
    """
    base = parents(C, y) | (na_yx(C, y, x) - set(H))
    return cache.score(y, base - {x}) - cache.score(y, base | {x})


def apply_delete(C, x, y, H):
    """Drop the ``x``-``y`` edge and orient ``y -- h`` / ``x -- h`` out of it."""
    C[x, y] = 0
    C[y, x] = 0
    for h in H:
        C[h, y] = 0
        if is_undirected(C, x, h):
            C[h, x] = 0
    return C


# ---------------------------------------------------------------------------
#  Phases
# ---------------------------------------------------------------------------


def total_score(C, cache):
    """Score of any DAG in the class ``C`` represents (BIC is score-equivalent)."""
    dag = pdag_to_dag(C)
    if dag is None:
        return None
    return sum(
        cache.score(v, {int(u) for u in np.flatnonzero(dag[:, v])})
        for v in range(C.shape[0])
    )


def _check_delta(before, after, cache, claimed, label):
    """Assert a move changed the total score by exactly its claimed delta.

    A mismatch means the operator's delta formula and its effect on the graph
    disagree — the failure mode that made an early version delete edges
    pointing the wrong way while scoring the move against the wrong node.
    """
    objective_cache = (
        LocalScoreCache(cache._data, cache._score_func)
        if getattr(cache, "_backend", None) == "gram"
        else cache
    )
    after_score = total_score(after, objective_cache)
    before_score = total_score(before, objective_cache)
    if after_score is None or before_score is None:
        msg = f"{label}: a graph has no consistent DAG extension"
        raise AssertionError(msg)
    actual = after_score - before_score
    if abs(actual - claimed) > 1e-6:
        msg = (
            f"{label}: claimed delta {claimed:+.6f} but total score moved {actual:+.6f}"
        )
        raise AssertionError(msg)


def _best_move(candidates, C, validity):
    """First valid candidate in (-delta, x, y, set) order.

    Scoring is cheap thanks to the cache, while the clique and path checks are
    not, so candidates are ranked by delta first and only validated until one
    passes. Validity does not depend on delta, so this picks the same move a
    validate-everything sweep would. The explicit ordering keeps ties
    independent of iteration order and numpy version.
    """
    for delta, x, y, subset in sorted(
        candidates, key=lambda c: (-c[0], c[1], c[2], c[3])
    ):
        if validity(C, x, y, subset):
            return delta, x, y, subset
    return None


def _forward_phase(
    C,
    cache,
    p,
    max_operator_set,
    max_parents,
    min_delta,
    verbose,
    verify_deltas=False,
    enumeration=None,
):
    steps = 0
    scores = MaskedScores(cache, p)
    if cache._backend == "gram" and cache._gram is not None:
        from ._gram_scan import GramCandidates

        scan = GramCandidates(
            cache, p, max_operator_set, max_parents, min_delta, forward=True
        )
    else:
        scan = ForwardCandidates(scores, p, max_operator_set, max_parents, min_delta)
    while True:
        view = GraphView(C)
        if not isinstance(scan, ForwardCandidates):
            scan.prepare(
                view,
                _subsets,
                _record_subset_pool,
                enumeration,
                _record_unlimited_summary,
            )
            move = scan.best(view)
        else:
            rows = scan.rows(
                view,
                _subsets,
                _record_subset_pool,
                enumeration,
                _record_unlimited_summary,
            )
            move = best_candidate(rows, view, forward=True)
        if move is None:
            if enumeration is not None:
                enumeration.update(
                    ges_forward_rebuilt_targets=scan.rebuilt_targets,
                    ges_forward_reused_targets=scan.reused_targets,
                    ges_forward_score_reuses=scores.reuses,
                )
            return C, steps
        delta, x, y, T = move
        if verbose:
            print(f"[GES] Insert({x}, {y}, {set(T) or '{}'}) delta={delta:.6f}")
        before = C.copy() if verify_deltas else None
        completed = complete_pdag(apply_insert(C, x, y, T), affected_nodes=(x, y))
        if completed is None:
            msg = (
                f"Insert({x}, {y}, {T}) passed validation but left a PDAG with no "
                "consistent extension — the validity check is wrong"
            )
            raise AssertionError(msg)
        if verify_deltas:
            _check_delta(before, completed, cache, delta, f"Insert({x}, {y}, {set(T)})")
        C = completed
        steps += 1


def _backward_phase(
    C,
    cache,
    p,
    max_operator_set,
    min_delta,
    verbose,
    verify_deltas=False,
    enumeration=None,
):
    steps = 0
    scores = MaskedScores(cache, p)
    scan = None
    if cache._backend == "gram" and cache._gram is not None:
        from ._gram_scan import GramCandidates

        scan = GramCandidates(
            cache, p, max_operator_set, None, min_delta, forward=False
        )
    while True:
        view = GraphView(C)
        if scan is not None:
            scan.prepare(
                view,
                _subsets,
                _record_subset_pool,
                enumeration,
                _record_unlimited_summary,
            )
            move = scan.best(view)
        else:
            rows = backward_rows(
                view, scores, _subsets, max_operator_set, min_delta, enumeration
            )
            move = best_candidate(rows, view, forward=False)
        if move is None:
            if enumeration is not None:
                enumeration["ges_backward_score_reuses"] = scores.reuses
            return C, steps
        delta, x, y, H = move
        if verbose:
            print(f"[GES] Delete({x}, {y}, {set(H) or '{}'}) delta={delta:.6f}")
        before = C.copy() if verify_deltas else None
        completed = complete_pdag(apply_delete(C, x, y, H), affected_nodes=(x, y))
        if completed is None:
            msg = (
                f"Delete({x}, {y}, {H}) passed validation but left a PDAG with no "
                "consistent extension — the validity check is wrong"
            )
            raise AssertionError(msg)
        if verify_deltas:
            _check_delta(before, completed, cache, delta, f"Delete({x}, {y}, {set(H)})")
        C = completed
        steps += 1


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------


def ges_search(
    data,
    score_func,
    node_names,
    *,
    max_operator_set=3,
    max_parents=None,
    min_delta=0.0,
    verbose=False,
    verify_deltas=False,
    stats=None,
    bic_backend="auto",
    n_jobs=1,
    parallel_backend="auto",
):
    """Run GES and return ``(CPDAG as GeneralGraph, total score)``.

    For an Insert from x to y, choose T from the undirected neighbours of y
    that are not adjacent to x. For a Delete, choose H from those adjacent
    to x. ``max_operator_set`` limits the size of each subset. With ``None``,
    the number of subsets doubles for each additional candidate node.

    Accept a move only if its score gain exceeds ``min_delta``. With the
    default threshold, a tie leaves the graph unchanged. For debugging,
    ``verify_deltas`` checks each accepted move's local score change
    against the change in the score of the whole graph.

    ``bic_backend='gram'`` batches Gaussian scores from centered sufficient
    statistics, then refines competitive moves with the reference scorer.
    ``reference`` retains the scalar path; ``covariance`` retains its existing
    opt-in numerical policy. The default ``auto`` selects Gram for built-in
    BIC with at least five variables and 200 samples, retaining the reference
    path for tiny inputs, BDeu and custom scorers. ``n_jobs`` and an explicit
    ``parallel_backend`` control Gram batches; automatic execution stays
    serial because measured dispatch overhead outweighs the numerical work.
    """
    from ..graph_core.validation import effective_n_jobs

    jobs = effective_n_jobs(n_jobs)
    if parallel_backend not in ("auto", "threads", "processes"):
        msg = "parallel_backend must be 'auto', 'threads' or 'processes'"
        raise ValueError(msg)
    data = np.asarray(data)
    p = data.shape[1]
    requested_bic_backend = bic_backend
    if bic_backend == "auto":
        compatible = (
            _score_module._local_score is _score_module._BUILTIN_LOCAL_SCORE
            and (_forward_phase, _backward_phase) == _BUILTIN_PHASES
        )
        bic_backend = (
            "gram"
            if score_func == "local_score_BIC"
            and p >= 5
            and len(data) >= 200
            and compatible
            else "reference"
        )
    cache = LocalScoreCache(data, score_func, bic_backend=bic_backend)
    C = np.zeros((p, p), dtype=np.int8)
    enumeration = {"truncated_candidate_pools": 0, "largest_candidate_pool": 0}

    t0 = time.perf_counter()
    with ExitStack() as stack:
        cache.configure_execution(jobs, parallel_backend, stack)
        C, insert_steps = _forward_phase(
            C,
            cache,
            p,
            max_operator_set,
            max_parents,
            min_delta,
            verbose,
            verify_deltas,
            enumeration,
        )
        C, delete_steps = _backward_phase(
            C,
            cache,
            p,
            max_operator_set,
            min_delta,
            verbose,
            verify_deltas,
            enumeration,
        )
    elapsed = time.perf_counter() - t0

    # Report the original objective of the selected class even when an opt-in
    # numerical backend was used to rank moves.
    total = total_score(
        C, cache if bic_backend == "reference" else LocalScoreCache(data, score_func)
    )
    if total is None:
        msg = "GES produced a CPDAG with no consistent extension"
        raise AssertionError(msg)

    if stats is not None:
        stats.update(
            ges_insert_steps=insert_steps,
            ges_delete_steps=delete_steps,
            ges_cache_hits=cache.hits,
            ges_cache_misses=cache.misses,
            ges_score=float(total),
            ges_search_sec=elapsed,
            ges_truncated_candidate_pools=enumeration["truncated_candidate_pools"],
            ges_approximate=bool(enumeration["truncated_candidate_pools"]),
            ges_n_jobs=jobs,
            ges_fast_scores=cache.fast_scores,
            ges_fallback_scores=cache.fallback_scores,
            ges_score_refinements=cache.refinements,
            ges_parallel_batches=cache.parallel_batches,
            ges_process_rounds=cache.process_rounds,
            ges_process_batches=cache.process_batches,
            ges_backend="processes"
            if cache.process_rounds
            else "threads"
            if cache.parallel_batches
            else "serial",
        )
        stats.update(
            {key: value for key, value in enumeration.items() if key.startswith("ges_")}
        )
        if cache._covariance is not None:
            stats.update(
                ges_covariance_scores=cache._covariance.fast_scores,
                ges_covariance_fallbacks=cache._covariance.fallbacks,
                ges_factor_hits=cache._covariance.factor_hits,
                ges_factor_misses=cache._covariance.factor_misses,
            )

    graph = cpdag_matrix_to_general_graph(C, node_names)
    graph.graph_kind = "cpdag"
    graph.metadata = {
        "max_operator_set": max_operator_set,
        "operator_enumeration": "complete" if max_operator_set is None else "capped",
        "approximate": bool(enumeration["truncated_candidate_pools"]),
        **enumeration,
    }
    graph.metadata.update(
        bic_backend=bic_backend,
        requested_bic_backend=requested_bic_backend,
        numeric_reference_ranking=(bic_backend != "covariance"),
    )
    return graph, float(total)


_BUILTIN_PHASES = (_forward_phase, _backward_phase)
