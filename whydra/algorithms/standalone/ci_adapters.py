"""Batch adapters for the counted CI callables used by PAG algorithms."""

from contextlib import nullcontext
from itertools import islice

from ..graph_core.cache import SharedCacheView
from ..graph_core.ci_batch import (
    _call_ci,
    _normalize_request,
    cit_batch,
    evaluate_ci_batch,
    supports_fisher_batch,
)
from ..graph_core.validation import effective_n_jobs

_MISSING = object()


def observed_phase_backend(after, before, coarse=None, extra=()):
    """Describe dispatch from snapshots without importing a search algorithm."""
    current = after.get("batches", {})
    previous = before.get("batches", {})
    used = {
        backend
        for backend, count in current.items()
        if count > previous.get(backend, 0)
    }
    if coarse is not None:
        used.update(
            backend
            for backend, count in coarse.get("coarse_batches", {}).items()
            if count
        )
    used.update(extra)
    used.discard("serial")
    return next(iter(used)) if len(used) == 1 else "hybrid" if used else "serial"


def attach_counted_batch(counted, cit, stats, phase, cache=None, lock=None):
    """Expose the underlying Fisher batch without bypassing phase accounting.

    Outer memoization is checked before dispatch. Duplicate misses share one
    computation when that cache is enabled, and results keep request order.
    Custom CI overrides retain their scalar callback and side effects.
    """
    counted._batch_cit = cit
    counted._pdsep_batch_context = (cit, cache, stats, phase, lock)

    def batch(
        requests, *, n_jobs=1, backend="auto", batch_size=256, return_exceptions=False
    ):
        requests = list(requests)
        if not requests:
            return []
        if not supports_fisher_batch(cit):
            values = []
            for x, y, conditions in requests:
                try:
                    values.append(counted(x, y, conditions))
                except Exception as exc:
                    if not return_exceptions:
                        raise
                    values.append(exc)
            return values

        if cache is None:
            # CIT already normalizes, deduplicates and owns in-flight keys.
            # Without an outer cache there is no second memoization layer to
            # visit: keep only this adapter's phase accounting.
            with lock if lock is not None else nullcontext():
                stats["total"] += len(requests)
                stats[phase["name"]] += len(requests)
            return evaluate_ci_batch(
                cit,
                requests,
                n_jobs=n_jobs,
                backend=backend,
                batch_size=batch_size,
                return_exceptions=return_exceptions,
            )

        from ..graph_core.independence_tests import _BUILTIN_CIT_BATCH

        if (
            isinstance(cache, SharedCacheView)
            and cache.owner is getattr(cit._cache, "owner", None)
            and getattr(cit.batch, "__func__", None) is _BUILTIN_CIT_BATCH
        ):
            # The owner checks both historical membership layers in one lookup
            # under CIT's single-flight lock. Accounting remains per adapter.
            outer_counts = {"hits": 0, "misses": 0}
            with lock if lock is not None else nullcontext():
                stats["total"] += len(requests)
                stats[phase["name"]] += len(requests)
            try:
                return cit_batch(
                    cit,
                    requests,
                    n_jobs=n_jobs,
                    backend=backend,
                    batch_size=batch_size,
                    return_exceptions=return_exceptions,
                    _outer_cache=cache,
                    _outer_counts=outer_counts,
                )
            finally:
                with lock if lock is not None else nullcontext():
                    stats["ci_cache_hits"] += outer_counts["hits"]
                    stats["ci_cache_misses"] += outer_counts["misses"]

        results = [_MISSING] * len(requests)
        missing = []
        pending = {}
        cache_enabled = cache is not None and getattr(cache, "max_entries", 1) > 0
        with lock if lock is not None else nullcontext():
            stats["total"] += len(requests)
            stats[phase["name"]] += len(requests)
            for position, (x, y, conditions) in enumerate(requests):
                normalized_request = _call_ci(
                    _normalize_request, cit, (x, y, conditions)
                )
                if isinstance(normalized_request, Exception):
                    results[position] = normalized_request
                    continue
                key, normalized = normalized_request
                value = cache.get(key, _MISSING) if cache is not None else _MISSING
                if value is not _MISSING:
                    stats["ci_cache_hits"] += 1
                    results[position] = value
                elif cache_enabled and key in pending:
                    stats["ci_cache_hits"] += 1
                    pending[key][1].append(position)
                else:
                    if cache is not None:
                        stats["ci_cache_misses"] += 1
                    item = (normalized, [position], key)
                    missing.append(item)
                    if cache_enabled:
                        pending[key] = item

        if getattr(cit.batch, "__func__", None) is _BUILTIN_CIT_BATCH:
            values = cit_batch(
                cit,
                (),
                n_jobs=n_jobs,
                backend=backend,
                batch_size=batch_size,
                return_exceptions=True,
                _validated_requests=[(key, query) for query, _, key in missing],
            )
        else:
            # Instrumented/custom public batch callables keep their callbacks.
            values = evaluate_ci_batch(
                cit,
                [query for query, _, _ in missing],
                n_jobs=n_jobs,
                backend=backend,
                batch_size=batch_size,
                return_exceptions=True,
            )
        with lock if lock is not None else nullcontext():
            for (_, positions, key), value in zip(missing, values, strict=False):
                if cache is not None and not isinstance(value, BaseException):
                    cache[key] = value
                for position in positions:
                    results[position] = value
        if not return_exceptions:
            for value in results:
                if isinstance(value, BaseException):
                    raise value
        return results

    counted.batch = batch
    return counted


class LegacyCITest:
    """Adapt a three-argument CI test to callers that also pass sufficient statistics."""

    def __init__(self, test, max_cond_set_size):
        self.test = test
        self.max_cond_set_size = max_cond_set_size

    def __call__(self, suffStat, x, y, S):
        """Forward the node pair and conditioning set to the wrapped test."""
        return self.test(x, y, S)


def attach_legacy_batch(wrapper, counted):
    """Attach a three-argument batch API to a legacy suffStat CI wrapper."""
    wrapper._batch_cit = counted
    wrapper.batch = counted.batch
    wrapper._pdsep_batch_context = getattr(counted, "_pdsep_batch_context", None)
    return wrapper


def ordered_ci_values(
    indep_test, suff_stat, x, y, condition_sets, *, n_jobs=1, batch_size=256
):
    """Yield ordered results while preserving first-separator early exit.

    Numerical errors in speculative later queries stay as values until that
    query is consumed. A separator found earlier therefore still stops search
    without exposing an error the scalar algorithm would never encounter.
    Generic oracles are evaluated lazily, one request at a time.
    """
    if not supports_fisher_batch(indep_test):
        for conditions in condition_sets:
            yield conditions, indep_test(suff_stat, x, y, list(conditions))
        return

    condition_sets = iter(condition_sets)
    # Test a short prefix first; only dependent pairs grow into enough native
    # chunks to occupy several workers. The numerical scratch bound is separate
    # from this speculative search window.
    maximum_window = batch_size * effective_n_jobs(n_jobs)
    window_size = min(32, maximum_window)
    while True:
        chunk = list(islice(condition_sets, window_size))
        if not chunk:
            return
        values = evaluate_ci_batch(
            indep_test,
            [(x, y, conditions) for conditions in chunk],
            n_jobs=n_jobs,
            batch_size=batch_size,
            return_exceptions=True,
        )
        for conditions, value in zip(chunk, values, strict=False):
            if isinstance(value, BaseException):
                raise value
            yield conditions, value
        window_size = min(maximum_window, window_size * 4)
