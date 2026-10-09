"""Greedy Equivalence Search (Chickering 2002).

Searches the space of Markov-equivalence classes rather than the space of
DAGs: every move is an Insert or Delete applied to a CPDAG, and the class is
re-derived after each accepted move. That is what makes an early arbitrary
orientation impossible — the failure mode of plain DAG hill-climbing, where
two orientations of one edge score identically, the tie is broken at random
and acyclicity then locks the wrong direction in place.

Phases run strictly forward-then-backward: forward reaches a superset of the
true class, backward prunes it back. Interleaving them breaks the argument.
"""

from __future__ import annotations

import itertools
import time

import numpy as np

from .pdag import (
    adjacent,
    complete_pdag,
    cpdag_matrix_to_general_graph,
    exists_semidirected_path,
    is_adjacent,
    is_clique,
    is_undirected,
    na_yx,
    neighbours,
    parents,
    pdag_to_dag,
)
from .scores import LocalScoreCache


def _subsets(items, max_size):
    """All subsets of ``items`` up to ``max_size``, smallest first."""
    items = sorted(items)
    limit = len(items) if max_size is None else min(len(items), max_size)
    for size in range(limit + 1):
        for combo in itertools.combinations(items, size):
            yield combo


# ---------------------------------------------------------------------------
#  Insert(x, y, T)
# ---------------------------------------------------------------------------

def insert_is_valid(C, x, y, T) -> bool:
    """Validity of ``Insert(x, y, T)`` (Chickering 2002, Theorem 15).

    1. ``NA_yx u T`` is a clique;
    2. every semi-directed path from ``y`` to ``x`` contains a node of
       ``NA_yx u T``.
    """
    block = na_yx(C, y, x) | set(T)
    if not is_clique(C, block):
        return False
    return not exists_semidirected_path(C, y, x, block=block)


def insert_delta(C, cache, x, y, T) -> float:
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
    """Validity of ``Delete(x, y, H)`` (Theorem 17): ``NA_yx \\ H`` is a clique."""
    return is_clique(C, na_yx(C, y, x) - set(H))


def delete_delta(C, cache, x, y, H) -> float:
    """``score(y, (Pa(y) u (NA_yx \\ H)) \\ {x}) - score(y, ... u {x})``.

    The second term has to add ``x`` explicitly: when the edge is undirected
    (``x -- y``), ``x`` is in neither ``Pa(y)`` nor ``NA_yx``, so leaving it
    implicit makes the delta identically zero and no undirected edge is ever
    deleted.
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
    actual = total_score(after, cache) - total_score(before, cache)
    if abs(actual - claimed) > 1e-6:
        raise AssertionError(
            f"{label}: claimed delta {claimed:+.6f} but total score moved "
            f"{actual:+.6f}"
        )


def _best_move(candidates, C, validity):
    """First valid candidate in (-delta, x, y, set) order.

    Scoring is cheap thanks to the cache, while the clique and path checks are
    not, so candidates are ranked by delta first and only validated until one
    passes. Validity does not depend on delta, so this picks the same move a
    validate-everything sweep would. The explicit ordering keeps ties
    independent of iteration order and numpy version.
    """
    for delta, x, y, subset in sorted(candidates, key=lambda c: (-c[0], c[1], c[2], c[3])):
        if validity(C, x, y, subset):
            return delta, x, y, subset
    return None


def _forward_phase(
    C, cache, p, max_operator_set, max_parents, min_delta, verbose, verify_deltas=False
):
    steps = 0
    while True:
        candidates = []
        for y in range(p):
            pa_y = parents(C, y)
            ne_y = neighbours(C, y)
            for x in range(p):
                if x == y or is_adjacent(C, x, y):
                    continue
                adj_x = adjacent(C, x)
                na = {t for t in ne_y if t in adj_x}
                pool = ne_y - adj_x - {x}
                for T in _subsets(pool, max_operator_set):
                    if max_parents is not None:
                        if len(pa_y | na | set(T) | {x}) > max_parents:
                            continue
                    delta = insert_delta(C, cache, x, y, T)
                    if delta > min_delta:
                        candidates.append((delta, x, y, tuple(sorted(T))))

        move = _best_move(candidates, C, insert_is_valid)
        if move is None:
            return C, steps
        delta, x, y, T = move
        if verbose:
            print(f"[GES] Insert({x}, {y}, {set(T) or '{}'}) delta={delta:.6f}")
        before = C.copy() if verify_deltas else None
        completed = complete_pdag(apply_insert(C, x, y, T))
        if completed is None:
            raise AssertionError(
                f"Insert({x}, {y}, {T}) passed validation but left a PDAG with no "
                "consistent extension — the validity check is wrong"
            )
        if verify_deltas:
            _check_delta(before, completed, cache, delta, f"Insert({x}, {y}, {set(T)})")
        C = completed
        steps += 1


def _backward_phase(
    C, cache, p, max_operator_set, min_delta, verbose, verify_deltas=False
):
    steps = 0
    while True:
        candidates = []
        for y in range(p):
            # Delete applies to x -> y and x -- y only. Including edges that
            # point the other way (y -> x) would score the move against y
            # while the move actually strips a parent from x.
            for x in parents(C, y) | neighbours(C, y):
                for H in _subsets(na_yx(C, y, x), max_operator_set):
                    delta = delete_delta(C, cache, x, y, H)
                    if delta > min_delta:
                        candidates.append((delta, x, y, tuple(sorted(H))))

        move = _best_move(candidates, C, delete_is_valid)
        if move is None:
            return C, steps
        delta, x, y, H = move
        if verbose:
            print(f"[GES] Delete({x}, {y}, {set(H) or '{}'}) delta={delta:.6f}")
        before = C.copy() if verify_deltas else None
        completed = complete_pdag(apply_delete(C, x, y, H))
        if completed is None:
            raise AssertionError(
                f"Delete({x}, {y}, {H}) passed validation but left a PDAG with no "
                "consistent extension — the validity check is wrong"
            )
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
):
    """Run GES and return ``(CPDAG as GeneralGraph, total score)``.

    ``max_operator_set`` caps ``|T|`` and ``|H|``: enumerating them is
    exponential in ``|NA_yx|``, which only bites on dense neighbourhoods.
    ``min_delta`` is compared strictly, so score-equivalent moves (delta == 0)
    cannot cycle forever. ``verify_deltas`` re-scores the whole graph after
    every move and asserts it matches the operator's delta — a debugging aid,
    quadratically slower.
    """
    data = np.asarray(data)
    p = data.shape[1]
    cache = LocalScoreCache(data, score_func)
    C = np.zeros((p, p), dtype=np.int8)

    t0 = time.perf_counter()
    C, insert_steps = _forward_phase(
        C, cache, p, max_operator_set, max_parents, min_delta, verbose, verify_deltas
    )
    C, delete_steps = _backward_phase(
        C, cache, p, max_operator_set, min_delta, verbose, verify_deltas
    )
    elapsed = time.perf_counter() - t0

    total = total_score(C, cache)
    if total is None:
        raise AssertionError("GES produced a CPDAG with no consistent extension")

    if stats is not None:
        stats.update(
            ges_insert_steps=insert_steps,
            ges_delete_steps=delete_steps,
            ges_cache_hits=cache.hits,
            ges_cache_misses=cache.misses,
            ges_score=float(total),
            ges_search_sec=elapsed,
        )

    return cpdag_matrix_to_general_graph(C, node_names), float(total)
