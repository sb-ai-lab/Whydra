"""Compact immutable search snapshots shared by the FCI+ implementations."""

from collections import OrderedDict, deque
from contextlib import nullcontext
from typing import TypedDict

import numpy as np

from ..graph_core import validation as _validation_exports
from ..graph_core.ci_batch import fisher_z_batch_from_corr, supports_fisher_batch
from ..graph_core.conditioning import bounded_size
from ..graph_core.process_execution import plan_process_work, process_map
from .hierarchy_cache import hierarchy_cache

effective_n_jobs = _validation_exports.effective_n_jobs


def hierarchy_closure(cg, seeds, decode, cache=None):
    """Reuse closures across searches; direct callers validate external edits."""
    return (cache if cache is not None else hierarchy_cache(cg, decode)).closure(seeds)


def _reachable_from_parents(node, parents, onward):
    """First edge is directed into the origin; later edges permit a circle."""
    found = parents[node].copy()
    found[node] = False
    frontier = found
    while frontier.any():
        following = np.any(onward[frontier], axis=0)
        following[node] = False
        frontier = following & ~found
        found |= frontier
    return found


def naa_ancestors(node, graph):
    """Find candidate ancestors reachable through a parent of the supplied node."""
    onward = (graph != 0) & (graph != 1)
    return set(
        np.flatnonzero(
            _reachable_from_parents(node, onward & (graph.T == 1), onward)
        ).tolist()
    )


def pdsep_candidates(mat):
    """Find the four-vertex witness in Claassen et al. (2013), Lemma 4."""
    two_heads = (mat == 1) & (mat.T == 1)
    edges = np.argwhere(np.tril(two_heads, -1))
    links = deque()
    if not len(edges):
        return links
    absent = (mat == 0) & (mat.T == 0)
    np.fill_diagonal(absent, False)
    onward = (mat != 0) & (mat != 1)
    parents = onward & (mat.T == 1)
    reach = {}
    for high, low in edges.tolist():
        for endpoint in (high, low):
            if endpoint not in reach:
                reach[endpoint] = _reachable_from_parents(endpoint, parents, onward)
        left = two_heads[high] & reach[low]
        right = two_heads[low] & reach[high]
        left[high] = right[low] = False
        if left.any() and right.any() and absent[np.ix_(left, right)].any():
            links.append((high, low))
    return links


def augmentation_groups(cg, count, decode):
    """Observed separating statements, in Whydra's public CI order."""
    for x in range(count):
        for y in range(x + 1, count):
            entry = cg.sepset[x][y]
            if entry is not None:
                yield x, y, tuple(sorted(decode(entry)))


def augmentation_probes(groups, neighbors, cap):
    """Map an extra conditioning variable to the incident endpoints it certifies."""
    for x, y, separator in groups:
        if len(separator) >= cap:
            continue
        support = {x, y, *separator}
        incident = {}
        for member in sorted(support):
            for other in neighbors[member]:
                if other not in support:
                    incident.setdefault(other, []).append(member)
        for other, sources in sorted(incident.items()):
            query = (x, y, tuple(sorted((*separator, other))))
            yield query, (tuple(sources), other)


def augmentation_inputs(cg, count, decode):
    """Generate augmentation queries and the endpoints each query can orient."""
    graph = cg.G.graph
    neighbors = [np.flatnonzero(row).tolist() for row in graph]
    groups = augmentation_groups(cg, count, decode)
    return augmentation_probes(groups, neighbors, bounded_size(cg, len(graph)))


def set_arrowheads(pag, endpoints):
    """Set arrowheads at existing endpoints and return the number of marks changed."""
    added = 0
    for node, target in endpoints:
        if pag[node, target] not in (0, 1):
            pag[node, target] = 1
            added += 1
    return added


def dependent_endpoints(values, evidence, alpha):
    """Yield endpoint pairs whose augmentation test has a p-value below alpha."""
    for record, (sources, target) in zip(values, evidence, strict=False):
        if record[-1] < alpha:
            yield from ((node, target) for node in sources)


class _AugmentResult(TypedDict):
    requests: int
    computed: int
    outer_hits: int
    cit_hits: int
    error: BaseException | None
    updates: list
    arrows: set


def _augment_chunk(corr, n_samples, descriptors, neighbors, alpha, cap, cached, retain):
    """Enumerate, test and reduce pairs in a worker; never receive a graph/CIT."""
    updates = OrderedDict()
    arrows = set()
    result: _AugmentResult = {
        "requests": 0,
        "computed": 0,
        "outer_hits": 0,
        "cit_hits": 0,
        "error": None,
        "updates": [],
        "arrows": arrows,
    }
    requests, metadata = [], []

    def flush():
        missing, positions, values, origins = [], [], [], []
        for position, key in enumerate(requests):
            hit = cached.get(key)
            if hit is None:
                missing.append(key)
                positions.append(position)
                values.append(None)
                origins.append(2)
            else:
                value, origin = hit
                values.append(value)
                origins.append(origin)
                if origin == 0:
                    result["outer_hits"] += 1
                else:
                    result["cit_hits"] += 1
        numeric = fisher_z_batch_from_corr(
            corr,
            n_samples,
            missing,
            n_jobs=1,
            return_exceptions=True,
            decision_alpha=alpha,
        )
        result["requests"] += len(requests)
        result["computed"] += len(missing)
        for position, value in zip(positions, numeric, strict=False):
            values[position] = value
        for key, value, origin, (nodes, adj) in zip(
            requests, values, origins, metadata, strict=False
        ):
            if isinstance(value, BaseException):
                result["error"] = value
                return False
            if retain:
                updates[key] = (value, origin)
                if len(updates) > retain:
                    updates.popitem(last=False)
            if value < alpha:
                arrows.update((node, adj) for node in nodes)
        requests.clear()
        metadata.clear()
        return True

    for query, effect in augmentation_probes(descriptors, neighbors, cap):
        requests.append(query)
        metadata.append(effect)
        if len(requests) >= 256 and not flush():
            result["updates"] = list(updates.items())
            return result
    if requests:
        flush()
    result["updates"] = list(updates.items())
    return result


def augment_native(cg, alpha, n_jobs, backend, decode, stats=None, min_tasks=256):
    """Return False for custom CI; otherwise commit one numerical snapshot."""
    if not supports_fisher_batch(cg.test) or backend == "threads":
        return False
    context = getattr(cg.test, "_pdsep_batch_context", None)
    if context is None:
        cit, outer, counted_stats, phase, lock = cg.test, None, None, None, None
    else:
        cit, outer, counted_stats, phase, lock = context
    if not hasattr(cit, "corr_matrix"):
        return False
    pag = cg.G.graph.copy()
    neighbors = tuple(frozenset(map(int, np.flatnonzero(row))) for row in pag)
    descriptors = list(augmentation_groups(cg, len(pag), decode))
    cap = bounded_size(cg, len(pag))
    # A cheap upper bound chooses local execution before touching a process pool.
    estimated = sum(
        sum(len(neighbors[node]) for node in (x, y, *sep))
        for x, y, sep in descriptors
        if len(sep) < cap
    )
    plan = plan_process_work(
        estimated,
        n_jobs,
        family="fci_plus_augment",
        backend=backend,
        tasks=len(descriptors),
        min_work=max(1024, min_tasks),
        stats=stats,
    )
    jobs = plan.jobs
    size = max(1, (len(descriptors) + jobs * 2 - 1) // (jobs * 2))
    chunks = [
        descriptors[start : start + size] for start in range(0, len(descriptors), size)
    ]
    cached_by_pair = {}
    with lock if lock is not None else nullcontext():
        with cit._lock:
            for key, value in cit._cache.items():
                cached_by_pair.setdefault(key[:2], {})[key] = (value, 1)
        if outer is not None:
            for key, value in outer.items():
                cached_by_pair.setdefault(key[:2], {})[key] = (value, 0)
    retain = max(cit.cache_size, getattr(outer, "max_entries", 0))

    def arguments():
        for chunk in chunks:
            cached = {}
            for x, y, sep in chunk:
                required = frozenset(sep)
                cached.update(
                    (key, value)
                    for key, value in cached_by_pair.get((x, y), {}).items()
                    if len(key[2]) == len(sep) + 1 and required.issubset(key[2])
                )
            yield (
                cit.corr_matrix,
                cit.n_samples,
                chunk,
                neighbors,
                alpha,
                cap,
                cached,
                retain,
            )

    results = process_map(
        _augment_chunk, arguments(), n_jobs=jobs, plan=plan, shared_arg_indices=(0,)
    )
    arrows = set()
    for result in results:
        with lock if lock is not None else nullcontext():
            if counted_stats is not None and phase is not None:
                counted_stats["total"] += result["requests"]
                counted_stats[phase["name"]] += result["requests"]
                if outer is not None:
                    counted_stats["ci_cache_hits"] += result["outer_hits"]
                    counted_stats["ci_cache_misses"] += (
                        result["requests"] - result["outer_hits"]
                    )
            with cit._lock:
                cit.cache_hits += result["cit_hits"]
                cit.cache_misses += result["computed"]
                for key, (value, origin) in result["updates"]:
                    if origin != 0 and cit.cache_size:
                        cit._cache[key] = value
                        cit._cache.move_to_end(key)
                        if len(cit._cache) > cit.cache_size:
                            cit._cache.popitem(last=False)
                            cit.cache_evictions += 1
                    if outer is not None:
                        outer[key] = value
        if result["error"] is not None:
            raise result["error"]
        arrows.update(result["arrows"])
    added = set_arrowheads(pag, arrows)
    if stats is not None:
        total = sum(result["requests"] for result in results)
        stats["augment_tasks"] = stats.get("augment_tasks", 0) + total
        stats["augment_arrowheads_added"] = (
            stats.get("augment_arrowheads_added", 0) + added
        )
        prefix = phase["name"] if phase is not None else "augment"
        kind = "process" if jobs > 1 else "serial"
        stats[f"{prefix}_{kind}_batches"] = stats.get(
            f"{prefix}_{kind}_batches", 0
        ) + bool(results)
        stats[f"{prefix}_{kind}_tasks"] = stats.get(f"{prefix}_{kind}_tasks", 0) + total
        stats["augment_worker_tasks"] = stats.get("augment_worker_tasks", 0) + len(
            results
        )
    cg.G.graph = pag
    return True
