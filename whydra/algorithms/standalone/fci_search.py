"""Separation evidence for FCI's stable adjacency and Possible-D-SEP phases.

Graph definitions: Colombo et al. (2012), doi:10.1214/11-AOS940.
Stable neighborhoods and conservative triples: Colombo & Maathuis (2014),
https://jmlr.org/papers/v15/colombo14a.html.
"""

from itertools import combinations
from time import perf_counter

import numpy as np

from ..graph_core.conditioning import bounded_size
from .ci_adapters import ordered_ci_values
from .compact_pdsep import CompactPDSEP
from .conservative_triples import classify
from .pag_rules import PAG_ARROW, PAG_CIRCLE
from .pdsep_candidates import PossibleDsepRow


def unshielded_triples(graph, *, callback_order=False):
    """Enumerate unordered endpoint pairs at a center.

    CI callbacks can observe the order of ties within a center. Their public
    schedule uses NumPy's column sort on the row-major sequence of edges.
    """
    incoming_by_center: dict[int, list[int]] = {}
    if callback_order:
        sources, centers = np.nonzero(graph)
        for position in np.argsort(centers):
            incoming_by_center.setdefault(centers[position], []).append(
                sources[position]
            )

    for middle, row in enumerate(graph):
        neighbors = np.flatnonzero(row)
        if callback_order:
            for left in incoming_by_center.get(middle, ()):
                for right in neighbors[neighbors.searchsorted(left, side="right") :]:
                    if not graph[left, right]:
                        yield left, middle, right
        else:
            for first, last in combinations(neighbors, 2):
                if not graph[first, last]:
                    yield first, middle, last


def orient_colliders(marks, separators, ambiguous=(), *, either=False, missing=None):
    """Set heads at unshielded centers supported by the supplied evidence.

    Legacy cached discovery accepts evidence from either stored direction.
    The full matrix and packed v2 backends agree on complete separator maps.
    Packed storage represents an absent separator by an empty tuple.
    """
    accept = any if either else all
    for a, b, c in unshielded_triples(marks):
        if ambiguous and ((a, b, c) in ambiguous or (c, b, a) in ambiguous):
            continue
        left = separators.get((a, c), missing)
        right = separators.get((c, a), missing)
        left_excludes = left is not None and b not in left
        right_excludes = right is not None and b not in right
        excludes = (left_excludes, right_excludes)
        if accept(excludes):
            marks[a, b] = marks[c, b] = PAG_ARROW


def record_triple(graph, triple, result, *, classify_all):
    """Record a triple's classification and directional separating sets in the graph."""
    a, _b, c = triple
    status, evidence = result
    if classify_all:
        graph.setdefault("triple_classification", {})[triple] = status
    if status == 3:
        graph.setdefault("unfTriples", set()).add(triple)
    graph["sepset"].update({(a, c): evidence["sepsetA"], (c, a): evidence["sepsetC"]})


def classify_neighbors(
    a, b, c, left, right, sep_left, sep_right, ci, alpha, majority, evaluate
):
    """Collect one vote per distinct separator, with incomplete-search status."""
    limit = getattr(ci, "max_cond_set_size", None)
    sides = [sorted(set(nodes).difference((a, c))) for nodes in (left, right)]
    conditions = []
    for depth in range(bounded_size(ci, max(map(len, sides))) + 1):
        unique = set(combinations(sides[0], depth))
        unique.update(combinations(sides[1], depth))
        conditions.extend(sorted(unique))

    independent = (
        nodes
        for nodes, value in zip(conditions, evaluate(conditions), strict=False)
        if value >= alpha
    )
    status, evidence = classify(b, independent, sep_left, sep_right, majority)
    if limit is not None and any(len(nodes) > limit for nodes in sides):
        status = 3
    return status, evidence


def _search_pair(source, target, row, cap, ci, suff_stat, alpha, peak, jobs):
    consumed = size = 0
    for size, conditions in row.levels(target, cap):
        results = ordered_ci_values(
            ci,
            suff_stat,
            source,
            target,
            conditions,
            n_jobs=jobs,
        )
        for condition, value in results:
            consumed += 1
            peak = max(peak, value)
            if value >= alpha:
                return condition, peak, consumed, size
    return None, peak, consumed, size


def refine_rows(
    skeleton,
    ci,
    suff_stat,
    separators,
    alpha,
    peaks,
    cap,
    ambiguous,
    reach,
    *,
    protected=(),
    jobs=1,
    plain=False,
):
    """Commit one neighborhood at a time, with reachability fixed before deletions."""
    marked = np.where(skeleton != 0, PAG_CIRCLE, 0)
    orient_colliders(marked, separators, ambiguous, either=not plain)
    reachable = [reach(node, marked) for node in range(len(marked))]
    history: list[list[int]] = []
    forbidden = {frozenset(edge) for edge in protected}
    cap = bounded_size(ci, cap)
    scanned = eligible = tests = last_depth = 0
    with CompactPDSEP(ci, jobs, backend="auto") as runner:
        for source, initial in enumerate(reachable):
            started = perf_counter()
            neighbors = np.flatnonzero(marked[source]).tolist()
            row = PossibleDsepRow(source, initial, neighbors, forbidden)
            history.append(row.history_entry())
            scanned += len(row.targets)
            if not row.outside:
                continue
            eligible += len(row.targets)
            workers = (
                None
                if plain
                else runner.evaluate_plan(source, row, cap, alpha, peaks[source])
            )
            for target in row.targets:
                if workers is not None and target in workers:
                    item = workers[target]
                    runner.publish(item)
                    peaks[source, target] = item["pmax"]
                    if item["error"] is not None:
                        raise item["error"]
                    witness, count, last_depth = (
                        item["separator"],
                        item["consumed"],
                        item["order"],
                    )
                else:
                    witness, peaks[source, target], count, last_depth = _search_pair(
                        source,
                        target,
                        row,
                        cap,
                        ci,
                        suff_stat,
                        alpha,
                        peaks[source, target],
                        jobs,
                    )
                tests += count
                if witness is not None:
                    marked[source, target] = marked[target, source] = 0
                    separators[source, target] = separators[target, source] = set(
                        witness
                    )
            runner.observe_serial(perf_counter() - started)
    result = {
        "G": (marked != 0).astype(int) * (1 if plain else PAG_CIRCLE),
        "sepset": separators,
        "pMax": peaks,
        "allPdsep": history,
        "max_ord": last_depth,
    }
    if not plain:
        result["pdsep_stats"] = dict(
            runner.stats,
            pairs_scanned=scanned,
            pairs_with_diffset=eligible,
            ci_tests=tests,
        )
    return result
