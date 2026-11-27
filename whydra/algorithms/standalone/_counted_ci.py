"""Explicit bridge from our counted CI closures to bounded serial batches.

Arbitrary decorators and custom tests are deliberately not unwrapped. Parallel
calls retain the original closures, including their in-flight coordination.
"""

import warnings
from contextlib import suppress
from itertools import islice
from typing import TypeGuard

from ..graph_core import CausalGraph
from ..graph_core.independence_tests import _builtin_fisherz, _fisher_batch_results

_GRAPH_CI_TEST = CausalGraph.ci_test


def counted_graph(cg):
    return getattr(
        cg.ci_test, "__func__", None
    ) is _GRAPH_CI_TEST and is_counted_fisherz(cg.test)


def is_counted_fisherz(test) -> TypeGuard["CountedCI"]:
    return type(test) is CountedCI and _builtin_fisherz(test.cit)


def counted_fisherz(call, cit, cache, stats, phase, lock, inflight=None):
    if not _builtin_fisherz(cit):
        return call
    return CountedCI(call, cit, cache, stats, phase, lock, inflight)


class CountedCI:
    def __init__(self, call, cit, cache, stats, phase, lock, inflight):
        self.call, self.cit, self.cache = call, cit, cache
        self.stats, self.phase, self.lock = stats, phase, lock
        self.inflight = inflight
        self.prefetch_stats = {
            "numeric_evaluations": 0,
            "unused_numeric_evaluations": 0,
            "batches": 0,
        }

    @property
    def cache_hits(self):
        return self.cit.cache_hits

    @property
    def cache_misses(self):
        return self.cit.cache_misses

    def __call__(self, x, y, conditions=None):
        return self.call(x, y, conditions)

    def _serial_batch(self, requests):
        output = []
        iterator = iter(requests)
        while chunk := list(islice(iterator, 1024)):
            result = self._chunk(chunk)
            output.extend(
                result
                if result is not None
                else [self.call(x, y, s) for x, y, s in chunk]
            )
        return output

    def _iter_values(self, requests):
        """Prefetch arithmetic, publishing ONLY requests consumed by the search.

        Unlike _serial_batch this supports early stopping and cache eviction.
        Future errors, cache writes, counters and reservations are never exposed.
        No lock remains held across a yield. Cache state is rechecked at replay;
        concurrent owners and evicted snapshot hits use the original closure.
        """
        iterator = iter(requests)
        size = 8
        missing = object()
        while True:
            chunk = list(islice(iterator, size))
            if not chunk:
                return
            size = min(256, size * 2)
            prepared = self._prepare_requests(chunk)
            if prepared is None or not is_counted_fisherz(self):
                for query in chunk:
                    yield query, self.call(*query)
                continue
            with self.lock, self.cit._lock:
                positions = [
                    i
                    for i, q in enumerate(prepared)
                    if (self.cache is None or q[3] not in self.cache)
                    and q[3] not in self.cit._cache
                ]
            values = [missing] * len(chunk)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                numeric = (
                    _fisher_batch_results(
                        self.cit.corr_matrix,
                        self.cit.n_samples,
                        [prepared[i][:3] for i in positions],
                    )
                    if positions
                    else []
                )
            if caught:
                # Do not expose warnings from a query beyond the stopping point.
                for query in chunk:
                    yield query, self.call(*query)
                continue
            for i, value in zip(positions, numeric, strict=False):
                values[i] = value
            self.prefetch_stats["numeric_evaluations"] += len(positions)
            self.prefetch_stats["batches"] += bool(positions)
            consumed_numeric = 0
            try:
                for query, request, value in zip(chunk, prepared, values, strict=False):
                    consumed_numeric += value is not missing
                    result = self._publish_prefetched(request, value, missing)
                    yield query, self.call(*query) if result is missing else result
            finally:
                self.prefetch_stats["unused_numeric_evaluations"] += (
                    len(positions) - consumed_numeric
                )

    def _publish_prefetched(self, request, value, missing):
        cit, cache = self.cit, self.cache
        x, y, s, key = request
        with self.lock, cit._lock:
            if key in cit._pending or (
                self.inflight is not None and key in self.inflight
            ):
                return missing
            outer_hit = cache is not None and key in cache
            if not outer_hit and key not in cit._cache and value is missing:
                return missing  # a snapshot hit was evicted before replay
            self.stats["total"] += 1
            self.stats[self.phase["name"]] += 1
            if outer_hit and cache is not None:
                self.stats["ci_cache_hits"] += 1
                return cache[key]
            if cache is not None:
                self.stats["ci_cache_misses"] += 1
            if key in cit._cache:
                cit.cache_hits += 1
                cit._cache.move_to_end(key)
                value = cit._decision_value(x, y, s, cit._cache[key])
            else:
                cit.cache_misses += 1
                if isinstance(value, Exception):
                    raise value
                owner = cit._pending[key] = object()
                try:
                    value = cit._decision_value(x, y, s, value, owner)
                    cit._store(key, value, (x, y))
                finally:
                    cit._pending.pop(key, None)
                    cit._condition.notify_all()
            if cache is not None:
                cache[key] = value
            return value

    def _prepare_requests(self, chunk):
        """Failed speculation must replay through the ordered scalar calls."""
        with suppress(Exception):
            prepared = [self.cit._request(x, y, s) for x, y, s in chunk]
            if all(
                tuple(sorted(s if s is not None else ())) == request[2]
                for (_, _, s), request in zip(chunk, prepared, strict=False)
            ):
                return prepared
        return None

    def _chunk(self, chunk):
        cit, cache = self.cit, self.cache
        if not is_counted_fisherz(self):
            return None
        prepared = self._prepare_requests(chunk)
        if prepared is None:
            return None
        with self.lock, cit._lock:
            keys = {key for _, _, _, key in prepared}
            if (
                cit._pending
                or self.inflight
                or not cit.cache_size
                or len(cit._cache) + len(keys - cit._cache.keys()) > cit.cache_size
                or (
                    cache is not None
                    and len(cache) + len(keys - cache.keys()) > cache.max_entries
                )
            ):
                return None
            unique = {}
            for x, y, s, key in prepared:
                if (cache is None or key not in cache) and key not in cit._cache:
                    unique.setdefault(key, (x, y, s))
            owners = {key: object() for key in unique}
            cit._pending.update(owners)
            try:
                values = (
                    dict(
                        zip(
                            unique,
                            _fisher_batch_results(
                                cit.corr_matrix, cit.n_samples, list(unique.values())
                            ),
                            strict=False,
                        )
                    )
                    if unique
                    else {}
                )
                output = []
                for x, y, s, key in prepared:
                    self.stats["total"] += 1
                    self.stats[self.phase["name"]] += 1
                    if cache is not None and key in cache:
                        self.stats["ci_cache_hits"] += 1
                        output.append(cache[key])
                        continue
                    if cache is not None:
                        self.stats["ci_cache_misses"] += 1
                    if key in cit._cache:
                        cit.cache_hits += 1
                        cit._cache.move_to_end(key)
                        value = cit._decision_value(x, y, s, cit._cache[key])
                    else:
                        cit.cache_misses += 1
                        value = values[key]
                        if isinstance(value, Exception):
                            raise value
                        value = cit._decision_value(x, y, s, value, owners[key])
                        cit._store(key, value, (x, y))
                    if cache is not None:
                        cache[key] = value
                    output.append(value)
                return output
            finally:
                for key in owners:
                    cit._pending.pop(key, None)
                cit._condition.notify_all()


def graph_batch(cg, requests):
    """Only the library graph method and explicitly registered closures batch."""
    requests = list(requests)
    if counted_graph(cg):
        from ..graph_core.conditioning import conditioning_allowed

        if all(conditioning_allowed(cg, s) for _, _, s in requests):
            return cg.test._serial_batch(requests)
    return [cg.ci_test(x, y, s) for x, y, s in requests]


def iter_ci_values(test, suffstat, x, y, subsets):
    """Original candidate order and successful prefix for CFCI Possible-D-SEP."""
    counted = getattr(test, "_counted_ci", None)
    if is_counted_fisherz(counted):
        values = counted._iter_values((x, y, s) for s in subsets)
        try:
            for (_, _, conditions), value in values:
                yield conditions, value
        finally:
            values.close()
    else:
        for conditions in subsets:
            yield conditions, test(suffstat, x, y, list(conditions))
