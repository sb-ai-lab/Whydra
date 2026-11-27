"""Compact stable Fisher skeleton levels, identical locally and in processes."""

from collections import deque
from itertools import combinations
from math import comb

from ..graph_core.ci_batch import fisher_z_batch_from_corr
from ..graph_core.native_fisher import native_level

_DEFAULT_FISHER_BATCH = fisher_z_batch_from_corr


def pair_chunks(adjacency, reference, depth, chunks, cache_enabled=True):
    """Balance unique pair queries, subtracting shared endpoint subsets."""
    import heapq

    if depth == 0:
        pairs = [(a, b) for a, row in enumerate(adjacency) for b in row if a < b]
        count = min(chunks, len(pairs))
        return (
            [pairs[i::count] for i in range(count)],
            len(pairs) * (1 if cache_enabled else 2),
        )
    neighbors = [frozenset(row) for row in reference]
    weighted = []
    for a, row in enumerate(adjacency):
        for b in row:
            if a >= b:
                continue
            na, nb = neighbors[a] - {b}, neighbors[b] - {a}
            cost = comb(len(na), depth) + comb(len(nb), depth)
            if cache_enabled:
                cost -= comb(len(na & nb), depth)
            if cost:
                weighted.append((cost, a, b))
    total = sum(row[0] for row in weighted)
    groups = [[] for _ in range(min(chunks, len(weighted)))]
    loads = [(0, i) for i in range(len(groups))]
    heapq.heapify(loads)
    for cost, a, b in sorted(weighted, reverse=True):
        load, i = heapq.heappop(loads)
        groups[i].append((a, b))
        heapq.heappush(loads, (load + cost, i))
    for group in groups:
        group.sort()
    return groups, total


def fisher_level_chunk(
    pairs,
    reference,
    corr,
    n_samples,
    depth,
    alpha,
    seed,
    cache_enabled,
    outer_enabled,
    retain,
    fci=True,
):
    """Return per-pair evidence, maxima, bounded cache deltas and real work.

    Every pair owns both directions. Shared subsets contribute to both pMax
    entries while their evidence is recorded once in original traversal order.
    No graph or mutable coordinator cache is sent to a worker.
    """
    native = native_level(
        pairs,
        reference,
        corr,
        n_samples,
        depth,
        alpha,
        seed,
        cache_enabled,
        outer_enabled,
        retain,
        fci,
        batch=fisher_z_batch_from_corr,
        native_math=fisher_z_batch_from_corr is _DEFAULT_FISHER_BATCH,
    )
    if native is not None:
        return native
    outcomes = []
    # Reserve slots in traversal order, before missing values are evaluated.
    # Separate tails keep an outer-only promotion from dropping a CIT write.
    pending = object()
    cit_events = deque(maxlen=retain)
    outer_events = deque(maxlen=retain)
    requests, destinations = [], []
    count = computed = cit_hits = outer_hits = cacheable = 0
    outer_cacheable = 0
    errors = []
    numeric = {}
    states = []

    def consume(index, direction, condition, common, value):
        state = states[index]
        if isinstance(value, BaseException):
            errors.append(
                (
                    (pairs[index][direction], pairs[index][1 - direction], condition),
                    value,
                )
            )
            return
        state["pmax"][direction] = max(state["pmax"][direction], value)
        if common and value > state["pmax"][1]:
            state["pmax"][1] = value
        if (value >= alpha) if fci else (value > alpha):
            state["separators"].append((direction, condition))
            state["union"].update(condition)

    def flush():
        nonlocal computed, cacheable, outer_cacheable
        if not requests:
            return
        values = fisher_z_batch_from_corr(
            corr,
            n_samples,
            requests,
            n_jobs=1,
            batch_size=1024,
            return_exceptions=True,
            execution_stats=numeric,
            decision_alpha=alpha,
        )
        computed += len(requests)
        for (index, direction, condition, common, event), value in zip(
            destinations, values, strict=False
        ):
            consume(index, direction, condition, common, value)
            if event is not None:
                event[1] = value
            if cache_enabled and not isinstance(value, BaseException):
                cacheable += 1
                outer_cacheable += int(outer_enabled)
        requests.clear()
        destinations.clear()

    for index, (a, b) in enumerate(pairs):
        states.append(
            {"pmax": [float("-inf"), float("-inf")], "separators": [], "union": set()}
        )
        first = tuple(v for v in reference[a] if v != b) if depth else ()
        second = tuple(v for v in reference[b] if v != a) if depth else ()
        first_set, second_set = frozenset(first), frozenset(second)
        for direction, (x, y, pool) in enumerate(((a, b, first), (b, a, second))):
            for condition in combinations(pool, depth):
                count += 1
                if (
                    cache_enabled
                    and direction
                    and all(v in first_set for v in condition)
                ):
                    if outer_enabled:
                        outer_hits += 1
                    else:
                        cit_hits += 1
                    continue
                common = (
                    cache_enabled
                    and not direction
                    and all(v in second_set for v in condition)
                )
                key = (a, b, condition)
                cached = seed.get(key) if cache_enabled else None
                if cached is not None:
                    value, outer = cached
                    if outer:
                        outer_hits += 1
                    else:
                        cit_hits += 1
                        if outer_enabled:
                            outer_events.append([key, value, True])
                            outer_cacheable += 1
                    consume(index, direction, condition, common, value)
                else:
                    requests.append((x, y, condition))
                    event = None
                    if cache_enabled:
                        event = [key, pending, False]
                        cit_events.append(event)
                        if outer_enabled:
                            outer_events.append(event)
                    destinations.append((index, direction, condition, common, event))
                if len(requests) >= 1024:
                    flush()
                    if errors:
                        break
            if errors:
                break
        if errors:
            break
    flush()
    for pair, state in zip(pairs, states, strict=False):
        separators = [condition for _, condition in sorted(state["separators"])]
        outcomes.append(
            (*pair, state["pmax"], separators, tuple(sorted(state["union"])))
        )
    updates = [
        (key, value)
        for key, value, _ in cit_events
        if value is not pending and not isinstance(value, BaseException)
    ]
    outer_updates = [
        (key, value)
        for key, value, _ in outer_events
        if value is not pending and not isinstance(value, BaseException)
    ]
    promotions = [(key, value) for key, value, promotion in outer_events if promotion]
    return {
        "pairs": outcomes,
        "requests": count,
        "computed": computed,
        "cit_hits": cit_hits,
        "outer_hits": outer_hits,
        "updates": updates,
        "promotions": promotions,
        "outer_updates": outer_updates,
        "outer_discarded": outer_cacheable - len(outer_updates),
        "discarded": cacheable - len(updates),
        "errors": errors,
        "numeric": numeric,
    }
