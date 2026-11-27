"""Bounded, ordered CI batches with a native Fisher-Z numerical kernel.

Only the built-in, unmodified Fisher test uses this path. Custom tests retain
their scalar invocation order. Process tasks contain numerical inputs only;
the dataset cache stays in the coordinator process.
"""

import os
import threading
from collections import OrderedDict, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from math import erfc, log, sqrt
from time import perf_counter
from typing import TypedDict, cast

import numpy as np

from .conditioning import conditioning_allowed
from .linalg import fisher_inverse
from .validation import effective_n_jobs

# Keep every loky entry point compatible with joblib's reusable executor.
# Changing memmap arguments between algorithm phases tears down its workers.
PROCESS_POOL_OPTIONS = {"backend": "loky", "max_nbytes": "1M"}
_THREAD_EXECUTORS = OrderedDict()
_THREAD_EXECUTORS_LOCK = threading.Lock()


def _trim_thread_executors():
    """Evict only idle pools; callers hold the registry lock."""
    for key, entry in list(_THREAD_EXECUTORS.items()):
        if len(_THREAD_EXECUTORS) <= 4:
            break
        if entry["users"] == 0:
            del _THREAD_EXECUTORS[key]
            entry["executor"].shutdown(wait=False)


@contextmanager
def thread_executor(workers):
    """Lease a reusable pool through submission and completion of a batch.

    Concurrent callers can use more than four configurations temporarily.
    Active leases cannot be evicted; release restores the idle-cache bound.
    """
    key = (os.getpid(), effective_n_jobs(workers))
    with _THREAD_EXECUTORS_LOCK:
        entry = _THREAD_EXECUTORS.pop(key, None)
        if entry is None:
            entry = {
                "executor": ThreadPoolExecutor(
                    max_workers=key[1], thread_name_prefix="whydra-ci"
                ),
                "users": 0,
            }
        entry["users"] += 1
        _THREAD_EXECUTORS[key] = entry
        _trim_thread_executors()
    try:
        yield entry["executor"]
    finally:
        with _THREAD_EXECUTORS_LOCK:
            entry["users"] -= 1
            _trim_thread_executors()


class _BatchStats(TypedDict):
    actual_backend: str
    batches: dict[str, int]
    tasks: dict[str, int]
    queries_by_backend: dict[str, int]
    queries: int
    requested_backends: dict[str, int]
    fallback_reasons: dict[str, int]
    build_seconds: float
    compute_seconds: float
    process_policy: list[dict]


def _empty_execution_stats() -> _BatchStats:
    return {
        "actual_backend": "none",
        "batches": dict.fromkeys(("serial", "threads", "processes"), 0),
        "tasks": dict.fromkeys(("serial", "threads", "processes"), 0),
        "queries_by_backend": dict.fromkeys(("serial", "threads", "processes"), 0),
        "queries": 0,
        "requested_backends": {},
        "fallback_reasons": {},
        "build_seconds": 0.0,
        "compute_seconds": 0.0,
        "process_policy": [],
    }


def _merge_execution_stats(target, source):
    for field, value in _empty_execution_stats().items():
        target.setdefault(field, value)
    for field in (
        "batches",
        "tasks",
        "queries_by_backend",
        "requested_backends",
        "fallback_reasons",
    ):
        for key, count in source[field].items():
            target[field][key] = target[field].get(key, 0) + count
    for field in ("queries", "build_seconds", "compute_seconds"):
        target[field] += source[field]
    target["process_policy"].extend(source.get("process_policy", ()))
    del target["process_policy"][:-64]
    backends = [name for name, count in target["batches"].items() if count]
    target["actual_backend"] = (
        backends[0] if len(backends) == 1 else "mixed" if backends else "none"
    )


def batch_execution_stats(test) -> _BatchStats:
    """Snapshot actual numeric dispatch; coarse search workers report separately."""
    seen = set()
    while id(test) not in seen:
        seen.add(id(test))
        inner = getattr(test, "_batch_cit", None)
        if inner is None or inner is test:
            break
        test = inner
    lock = getattr(test, "_lock", None)
    if lock is not None:
        with lock:
            stats = getattr(test, "_batch_execution_stats", _empty_execution_stats())
            return deepcopy(stats)
    return _empty_execution_stats()


def supports_fisher_batch(test):
    """Check whether the test supports Whydra's built-in Fisher-Z batch path."""
    from . import independence_tests as ci

    if type(test) is ci.CIT:
        return (
            test.method == "fisherz"
            and type(test).__call__ is ci._BUILTIN_CIT_CALL
            and ci.fisher_z_from_corr is ci._BUILTIN_FISHER_KERNEL
            and getattr(test.fisher_z, "__func__", None) is ci._BUILTIN_FISHER_Z
        )
    inner = getattr(test, "_batch_cit", None)
    return inner is not None and inner is not test and supports_fisher_batch(inner)


def configure_fisher_alpha(test, alpha):
    """Configure legacy-compatible boundary decisions before a search phase.

    Existing cache evidence and counters survive configuration. Only low-order
    probabilities near the new decision boundary require a reference solve.
    """
    from .independence_tests import (
        CIT,
        _fisher_low_order_components,
        _fisher_near_decision,
        _fisher_probability,
        _fisher_z_from_corr_reference,
    )

    seen, outer_caches = set(), []
    while id(test) not in seen:
        seen.add(id(test))
        context = getattr(test, "_pdsep_batch_context", None)
        if context is not None and context[1] is not None:
            outer_caches.append(context[1])
        inner = getattr(test, "_batch_cit", None)
        if inner is None or inner is test:
            break
        test = inner
    if not isinstance(test, CIT) or test.method != "fisherz":
        return
    with test._lock:
        if getattr(test, "decision_alpha", None) == alpha:
            return
        if test._inflight:
            msg = "Cannot change Fisher decision alpha during pending CI computations"
            raise ValueError(msg)
        replacements = {}
        if alpha is not None:
            for cache in (test._cache, *outer_caches):
                for key, value in cache.items():
                    if key in replacements or len(key[2]) > 1:
                        continue
                    x, y, conditions = key
                    dof = test.n_samples - len(conditions) - 3
                    if dof <= 0:
                        continue
                    components = _fisher_low_order_components(
                        test.corr_matrix, x, y, conditions
                    )
                    if components is not None:
                        numerator, denominator, bound = components
                        _, r = _fisher_probability(numerator, denominator, dof)
                        if _fisher_near_decision(value, r, dof, bound, alpha):
                            replacements[key] = _fisher_z_from_corr_reference(
                                test.corr_matrix, test.n_samples, x, y, conditions
                            )
        for cache in (test._cache, *outer_caches):
            # Bypass LRU-moving __setitem__ implementations: refining an
            # existing probability is not another observed CI request.
            for key in tuple(cache):
                if key in replacements:
                    replace_value = getattr(cache, "replace_value", None)
                    if replace_value is not None:
                        replace_value(key, replacements[key])
                    elif isinstance(cache, OrderedDict):
                        OrderedDict.__setitem__(cache, key, replacements[key])
                    else:
                        cache[key] = replacements[key]
        test.decision_alpha = alpha


def _call_ci(function, *args, return_exceptions=True, **kwargs):
    """Keep per-request failures in the same position as their inputs."""
    try:
        return function(*args, **kwargs)
    except Exception as exc:
        if not return_exceptions:
            raise
        return exc


def evaluate_ci_batch(
    test, requests, *, n_jobs=1, backend="auto", batch_size=256, return_exceptions=False
):
    """Return conditional-independence results in input order, using batch support if available."""
    batch = getattr(test, "batch", None)
    if batch is not None:
        options = {"n_jobs": n_jobs, "backend": backend, "batch_size": batch_size}
        if return_exceptions:
            options["return_exceptions"] = True
        return batch(requests, **options)
    return [
        _call_ci(test, x, y, conditions, return_exceptions=return_exceptions)
        for x, y, conditions in requests
    ]


def _fisher_low_order_chunk(corr, n_samples, queries, matrices, decision_alpha):
    """Vector gather and closed-form arithmetic, preserving scalar libm order."""
    from .independence_tests import (
        _FISHER_EPS,
        _fisher_z_from_corr_reference,
    )

    order = len(queries[0][2])
    dof = n_samples - order - 3
    a, b, d = matrices[:, 0, 0], matrices[:, 0, 1], matrices[:, 1, 1]
    finite = np.isfinite(matrices).all(axis=(1, 2))
    safe = finite & (a > 0) & (d > 0)
    transpose = np.swapaxes(matrices, 1, 2)
    uncertainty = None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        safe &= (
            np.abs(matrices - transpose)
            <= 8
            * _FISHER_EPS
            * np.maximum(1.0, np.maximum(np.abs(matrices), np.abs(transpose)))
        ).all(axis=(1, 2))
        if order == 0:
            trace = a + d
            trace_power = trace * trace
            determinant = a * d - b * b
            numerator, denominator = b, a * d
        else:
            c, e, f = matrices[:, 0, 2], matrices[:, 1, 2], matrices[:, 2, 2]
            first, second = a * f - c * c, d * f - e * e
            numerator = b * f - c * e
            denominator = first * second
            determinant = a * second - b * numerator + c * (b * e - c * d)
            trace = a + d + f
            trace_power = trace * trace * trace
            safe &= (f > 0) & (first > 0) & (second > 0) & (a * d - b * b > 0)
        safe &= (
            np.isfinite(trace_power)
            & np.isfinite(denominator)
            & (denominator > 0)
            & (determinant > trace_power * 1e-6)
        )
        correlations = np.clip(numerator / np.sqrt(denominator), -0.9999999, 0.9999999)
        if decision_alpha is not None:
            uncertainty = (
                256
                * _FISHER_EPS
                * (trace_power / determinant)
                * max(1.0, sqrt(dof))
                / np.maximum(1e-15, 1 - correlations * correlations)
            )
    # Convert once to Python floats, keeping exactly the scalar libm log/erfc
    # order. Vector operands and boundary checks avoid two Python helpers and
    # several NumPy scalar operations for every cheap low-order query.
    root_dof, root_two = sqrt(dof), sqrt(2.0)
    result: list[float | Exception] = [
        erfc(root_dof * abs(0.5 * log((1 + r) / (1 - r))) / root_two)
        for r in correlations.tolist()
    ]
    if uncertainty is not None and decision_alpha is not None:
        safe &= np.abs(np.asarray(result) - decision_alpha) > uncertainty
    for index in np.flatnonzero(~safe):
        x, y, conditions = queries[index]
        result[index] = _call_ci(
            _fisher_z_from_corr_reference, corr, n_samples, x, y, conditions
        )
    return result


def _fisher_chunk(corr, n_samples, queries, decision_alpha=None):
    """Return floats/errors without allowing one LAPACK error to lose a batch."""
    from . import independence_tests as legacy_ci

    legacy = getattr(legacy_ci, "_fisher_batch_results", None)
    if legacy is not None and legacy is not getattr(
        legacy_ci, "_LEGACY_BATCH_CORE", legacy
    ):
        return legacy(corr, n_samples, queries)
    from .independence_tests import _BUILTIN_FISHER_KERNEL, fisher_z_from_corr

    if not queries:
        return []
    if len(queries[0][2]) <= 1 and fisher_z_from_corr is _BUILTIN_FISHER_KERNEL:
        from .native_fisher import native_batch

        native = native_batch(corr, n_samples, queries, decision_alpha)
        if native is not None:
            return native
    if len(queries) == 1:
        return [
            _call_ci(
                fisher_z_from_corr,
                corr,
                n_samples,
                *queries[0],
                decision_alpha=decision_alpha,
            )
        ]
    if any(
        not isinstance(value, (int, np.integer))
        for x, y, conditions in queries
        for value in (x, y, *conditions)
    ):
        # Scalar np.ix_ rejects float indices, even integral-valued floats.
        # Do not silently truncate them in the native integer gather below.
        return [
            _call_ci(
                fisher_z_from_corr,
                corr,
                n_samples,
                *query,
                decision_alpha=decision_alpha,
            )
            for query in queries
        ]
    order = len(queries[0][2])
    dof = n_samples - order - 3
    if dof <= 0:
        return [
            ValueError(
                f"Fisher-Z requires positive degrees of freedom: n={n_samples}, |S|={order}"
            )
            for _ in queries
        ]
    if order <= 1 and len(queries) <= 8:
        # A speculative prefix often has only a few requests; setting up all
        # vector guards costs more than their inexpensive scalar closed forms.
        return [
            _call_ci(
                fisher_z_from_corr,
                corr,
                n_samples,
                *query,
                decision_alpha=decision_alpha,
            )
            for query in queries
        ]
    indices = np.asarray(
        [(int(x), int(y), *conditions) for x, y, conditions in queries], dtype=np.intp
    )
    matrices = corr[indices[:, :, None], indices[:, None, :]]
    if order <= 1 and matrices.dtype == np.float64:
        return _fisher_low_order_chunk(
            corr, n_samples, queries, matrices, decision_alpha
        )
    result: list[float | Exception] = [float("nan")] * len(queries)
    finite = np.flatnonzero(np.isfinite(matrices).all(axis=(1, 2)))
    if not len(finite):
        return result
    try:
        eigenvalues = np.linalg.eigvalsh(matrices[finite])
        singular = eigenvalues[:, 0] <= eigenvalues[:, -1] * 1e-12
        for index in finite[singular]:
            result[int(index)] = ValueError(
                "Fisher-Z correlation submatrix is singular or ill-conditioned; no pseudoinverse is used"
            )
        valid = finite[~singular]
        if not len(valid):
            return result
        inverses = fisher_inverse(matrices[valid])
        denominator = inverses[:, 0, 0] * inverses[:, 1, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            correlations = np.divide(
                -inverses[:, 0, 1],
                np.sqrt(np.abs(denominator)),
                out=np.full(len(valid), np.nan),
                where=denominator != 0,
            )
        finite_r = np.isfinite(correlations)
        correlations = np.clip(correlations[finite_r], -0.9999999, 0.9999999)
        # Keep the canonical libm log/erfc and operation order. NumPy/SciPy's
        # vector transcendental functions differ by ULPs and can change a CI
        # decision when alpha equals the scalar p-value.
        for index, r in zip(valid[finite_r], correlations, strict=False):
            z = 0.5 * log((1 + r) / (1 - r))
            result[int(index)] = erfc(sqrt(dof) * abs(z) / sqrt(2.0))
    except (ValueError, np.linalg.LinAlgError):
        # Preserve per-query failures and scalar numerical semantics on a
        # problematic LAPACK batch; ordinary full-rank batches stay native.
        for index, (x, y, conditions) in enumerate(queries):
            result[index] = _call_ci(
                fisher_z_from_corr,
                corr,
                n_samples,
                x,
                y,
                conditions,
                decision_alpha=decision_alpha,
            )
    return result


def fisher_z_batch_from_corr(
    corr,
    n_samples,
    queries,
    *,
    n_jobs=1,
    backend="auto",
    batch_size=256,
    return_exceptions=False,
    execution_stats=None,
    decision_alpha=None,
):
    """Compute Fisher-Z results for a batch of requests against a shared correlation matrix."""
    jobs = effective_n_jobs(n_jobs)
    if backend not in ("auto", "threads", "processes"):
        msg = "CI backend must be 'auto', 'threads', or 'processes'"
        raise ValueError(msg)
    if not isinstance(batch_size, int) or batch_size < 1:
        msg = "batch_size must be a positive integer"
        raise ValueError(msg)
    build_started = perf_counter()
    groups = defaultdict(list)
    queries = list(queries)
    for index, query in enumerate(queries):
        x, y, conditions = query
        groups[len(conditions)].append((index, (x, y, tuple(conditions))))
    tasks = []
    for order, group in groups.items():
        # Bound scratch matrices, including inverse/eigenvalue temporaries.
        size = min(batch_size, max(1, 16 * 1024 * 1024 // (24 * (order + 2) ** 2)))
        tasks.extend(
            group[start : start + size] for start in range(0, len(group), size)
        )
    outcomes = [None] * len(queries)
    numerical_work = sum(len(chunk) * (len(chunk[0][1][2]) + 2) ** 3 for chunk in tasks)
    # Auto chooses processes or serial. Explicit threads remain compatible
    # for callers that require a shared address space. Small native matrix
    # batches do not amortize serialization or scheduling.
    parallel = jobs > 1 and len(tasks) > 1 and len(queries) >= 1024
    process_queries = 16384 if backend == "auto" else 2048
    process_eligible = (
        parallel
        and backend in ("auto", "processes")
        and (len(queries) >= process_queries or numerical_work >= 1000000)
    )
    use_threads = (
        parallel
        and backend == "threads"
        and numerical_work >= 200000
        and numerical_work >= len(queries) * 64
    )
    stats = _empty_execution_stats()
    plan = None
    if backend in ("auto", "processes") and queries:
        from .native_fisher import NATIVE_AVAILABLE
        from .process_execution import plan_process_work

        work = numerical_work / 8
        plan = plan_process_work(
            work,
            jobs,
            family=f"fisher_{'cpp' if NATIVE_AVAILABLE else 'python'}_orders"
            + "_".join(map(str, sorted(groups))),
            backend=backend,
            tasks=len(tasks),
            min_work=0 if process_eligible else work + 1,
            stats=stats,
        )
    selected_jobs = plan.jobs if plan is not None else jobs
    use_processes = process_eligible and selected_jobs > 1
    actual_backend = (
        "processes" if use_processes else "threads" if use_threads else "serial"
    )
    stats["requested_backends"][backend] = 1
    stats["queries"] = len(queries)
    if tasks:
        stats["batches"][actual_backend] = 1
        stats["tasks"][actual_backend] = len(tasks)
        stats["queries_by_backend"][actual_backend] = len(queries)
    if actual_backend == "serial":
        reason = (
            "no_misses"
            if not queries
            else "single_worker"
            if jobs == 1
            else "single_task"
            if len(tasks) <= 1
            else "insufficient_work"
        )
        stats["fallback_reasons"][reason] = 1
    stats["build_seconds"] = perf_counter() - build_started
    compute_started = perf_counter()
    if use_processes:
        from .process_execution import process_map

        chunks = process_map(
            _fisher_chunk,
            (
                (corr, n_samples, [query for _, query in chunk], decision_alpha)
                for chunk in tasks
            ),
            n_jobs=selected_jobs,
            plan=plan,
            shared_arg_indices=(0,),
        )
    elif use_threads:
        with thread_executor(min(jobs, len(tasks))) as executor:
            chunks = list(
                executor.map(
                    lambda chunk: _fisher_chunk(
                        corr, n_samples, [query for _, query in chunk], decision_alpha
                    ),
                    tasks,
                )
            )
    else:
        chunks = [
            _fisher_chunk(
                corr, n_samples, [query for _, query in chunk], decision_alpha
            )
            for chunk in tasks
        ]
        if plan is not None:
            plan.model.observe(plan, perf_counter() - compute_started)
    for task, values in zip(tasks, chunks, strict=False):
        for (index, _), value in zip(task, values, strict=False):
            outcomes[index] = value
    stats["compute_seconds"] = perf_counter() - compute_started
    if execution_stats is not None:
        _merge_execution_stats(execution_stats, stats)
    if not return_exceptions:
        for value in outcomes:
            if isinstance(value, BaseException):
                raise value
    return outcomes


def _normalize_request(cit, request):
    x, y, conditions = request
    conditions = tuple(sorted(set(conditions if conditions is not None else ())))
    if not conditioning_allowed(cit, conditions):
        msg = "CI request exceeds max_cond_set_size; no p-value was computed"
        raise ValueError(msg)
    x, y = int(x), int(y)
    if x == y or x in conditions or y in conditions:
        msg = "CI variables and conditioning set must be disjoint"
        raise ValueError(msg)
    if any(v < 0 or v >= cit.data.shape[1] for v in (x, y, *conditions)):
        msg = "CI variable index out of bounds"
        raise ValueError(msg)
    return (min(x, y), max(x, y), conditions), (x, y, conditions)


def cit_batch(
    cit,
    requests,
    *,
    n_jobs=1,
    backend="auto",
    batch_size=256,
    return_exceptions=False,
    _validated_requests=None,
    _outer_cache=None,
    _outer_counts=None,
):
    """Evaluate a batch through the CI tester and reuse its cached results."""
    if not supports_fisher_batch(cit):
        if _validated_requests is not None:
            requests = [normalized for _, normalized in _validated_requests]
        results = []
        for request in requests:
            try:
                value = cit(*request)
            except Exception as exc:
                if not return_exceptions:
                    raise
                value = exc
            results.append(value)
        return results
    requests = (
        list(requests) if _validated_requests is None else list(_validated_requests)
    )
    outcomes: list[float | BaseException | None] = [None] * len(requests)
    owned = []
    owned_keys = {}
    waiting = []
    outer_duplicates, outer_publications = [], []
    outer_pending = {}
    outer_enabled = _outer_cache is not None and _outer_cache.max_entries > 0
    cache_missing = object()
    with cit._lock:
        for index, request in enumerate(requests):
            if _validated_requests is None:
                request = _call_ci(_normalize_request, cit, request)
                if isinstance(request, Exception):
                    outcomes[index] = request
                    continue
            key, normalized = request
            entry = None
            if _outer_cache is not None:
                assert _outer_counts is not None
                # Both facades share one key/value index. Their independent
                # membership and LRU policies still determine the hit layer.
                entry = _outer_cache.owner.entries.get(key)
                if _outer_cache.contains_entry(entry):
                    _outer_counts["hits"] += 1
                    outcomes[index] = entry.value
                    continue
                if outer_enabled and key in outer_pending:
                    _outer_counts["hits"] += 1
                    outer_duplicates.append((index, outer_pending[key]))
                    continue
                _outer_counts["misses"] += 1
                if outer_enabled:
                    outer_pending[key] = index
                outer_publications.append((key, index))
                cached = (
                    entry.value if cit._cache.contains_entry(entry) else cache_missing
                )
            else:
                cached = cit._cache.get(key, cache_missing)
            if cached is not cache_missing:
                cit.cache_hits += 1
                if _outer_cache is not None:
                    cit._cache.touch_entry(entry)
                else:
                    cit._cache.move_to_end(key)
                outcomes[index] = cast("float", cached)
            elif cit.cache_size and key in owned_keys:
                owned[owned_keys[key]][2].append(index)
            elif cit.cache_size and key in cit._inflight:
                pending = cit._inflight[key]
                if pending is None:
                    pending = Future()
                    cit._inflight[key] = pending
                waiting.append((index, key, pending))
            else:
                cit.cache_misses += 1
                owned_keys[key] = len(owned)
                owned.append((key, normalized, [index]))
                if cit.cache_size:
                    cit._inflight[key] = None
    execution_stats = _empty_execution_stats()
    try:
        values = fisher_z_batch_from_corr(
            cit.corr_matrix,
            cit.n_samples,
            [item[1] for item in owned],
            n_jobs=n_jobs,
            backend=backend,
            batch_size=batch_size,
            return_exceptions=True,
            execution_stats=execution_stats,
            decision_alpha=getattr(cit, "decision_alpha", None),
        )
    except BaseException as exc:
        with cit._lock:
            for key, _, _ in owned:
                pending = cit._inflight.pop(key, None)
                if pending is not None:
                    pending.set_exception(exc)
        raise
    # Finish our own work before waiting for other batches: two batches may
    # each own keys the other needs, and waiting first would deadlock.
    with cit._lock:
        if not hasattr(cit, "_batch_execution_stats"):
            cit._batch_execution_stats = _empty_execution_stats()
        _merge_execution_stats(cit._batch_execution_stats, execution_stats)
        for (key, _, indices), value in zip(owned, values, strict=False):
            failed = isinstance(value, BaseException)
            if not failed and cit.cache_size:
                publish = getattr(cit._cache, "set_and_touch", None)
                if publish is not None:
                    publish(key, value)
                else:
                    cit._cache[key] = value
                    cit._cache.move_to_end(key)
                if len(cit._cache) > cit.cache_size:
                    cit._cache.popitem(last=False)
                    cit.cache_evictions += 1
            pending = cit._inflight.pop(key, None) if cit.cache_size else None
            if pending is not None:
                if failed:
                    pending.set_exception(value)
                else:
                    pending.set_result(value)
            if not failed:
                cit.cache_hits += len(indices) - 1
            for index in indices:
                outcomes[index] = value
    for index, key, pending in waiting:
        outcome = _call_ci(pending.result)
        outcomes[index] = outcome
        if not isinstance(outcome, BaseException):
            with cit._lock:
                cit.cache_hits += 1
                touch = getattr(cit._cache, "get_and_touch", None)
                if touch is not None:
                    touch(key)
                elif key in cit._cache:
                    cit._cache.move_to_end(key)
    if _outer_cache is not None:
        # Publish after all computations and waits, in first-miss request order.
        # This includes entries promoted on a CIT cache hit.
        with cit._lock:
            for key, index in outer_publications:
                value = outcomes[index]
                if not isinstance(value, BaseException):
                    _outer_cache[key] = value
        for index, original in outer_duplicates:
            outcomes[index] = outcomes[original]
    if not return_exceptions:
        for value in outcomes:
            if isinstance(value, BaseException):
                raise value
    return outcomes
