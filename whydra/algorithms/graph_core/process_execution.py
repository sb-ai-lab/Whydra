"""Reusable bounded process execution for independent numerical search tasks.

Pools belong to this module, not joblib's global memmapping executor. Different
algorithm phases therefore cannot replace each other's workers by changing a
memmap option. Ordered futures avoid Parallel's polling delay on short levels.
Only compact, immutable inputs and top-level worker functions cross this API.
"""

import os
import threading
from collections import OrderedDict, deque
from collections.abc import Callable
from contextlib import contextmanager
from time import monotonic, perf_counter
from typing import TypeVar

from .execution_policy import PROCESS_COST_MODEL
from .validation import effective_n_jobs
from .worker_data import (
    acquire_array,
    call_with_shared_data,
    clear_shared_data,
    initialize_worker,
    release_array,
    retain_array,
)

_POOLS = OrderedDict()
_LOCK = threading.RLock()
_NATIVE_ENV = dict.fromkeys(
    (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ),
    "1",
)


def plan_process_work(
    work, n_jobs, *, family, backend="auto", tasks=None, min_work: float = 0, stats=None
):
    """Choose a worker count before constructing chunks, using observed costs."""
    with _LOCK:
        states = {
            key[1]: {
                "warm": entry.get("ready", False)
                and monotonic() - entry.get("last_used", 0) < 270,
                "users": entry["users"] or int(entry.get("pending", 0) > 0),
            }
            for key, entry in _POOLS.items()
            if key[0] == os.getpid()
        }
    return PROCESS_COST_MODEL.choose(
        work,
        effective_n_jobs(n_jobs),
        family=family,
        backend=backend,
        tasks=tasks,
        min_work=min_work,
        pool_states=states,
        stats=stats,
    )


def _trim_pools():
    for key, entry in list(_POOLS.items()):
        if len(_POOLS) <= 2:
            break
        if entry["users"] == 0 and not entry.get("pending", 0):
            del _POOLS[key]
            entry["executor"].shutdown(wait=False)


@contextmanager
def _process_pool(n_jobs):
    from joblib.externals.loky import ProcessPoolExecutor
    from joblib.externals.loky.process_executor import BrokenProcessPool

    key = (os.getpid(), effective_n_jobs(n_jobs))
    with _LOCK:
        entry = _POOLS.pop(key, None)
        if entry is None:
            entry = {
                "executor": ProcessPoolExecutor(
                    max_workers=key[1],
                    timeout=300,
                    env=_NATIVE_ENV,
                    initializer=initialize_worker,
                ),
                "users": 0,
                "pending": 0,
                "ready": False,
                "last_used": 0,
            }
        entry["users"] += 1
        _POOLS[key] = entry
        _trim_pools()
    try:
        yield entry["executor"]
    except BrokenProcessPool:
        # A failed executor must not poison subsequent independent fits.
        # Do not retry a task automatically: a custom callback may have effects.
        with _LOCK:
            if _POOLS.get(key) is entry:
                del _POOLS[key]
        entry["executor"].shutdown(wait=False, kill_workers=True)
        raise
    finally:
        with _LOCK:
            entry["users"] -= 1
            entry["ready"] = True
            entry["last_used"] = monotonic()
            _trim_pools()


_Result = TypeVar("_Result")


def process_map(
    function: Callable[..., _Result],
    arguments,
    *,
    n_jobs,
    plan=None,
    shared_arg_indices=(),
) -> list[_Result]:
    """Evaluate ``function(*args)`` in order, with bounded outstanding tasks.

    A one-worker invocation uses the identical worker locally. Search callers
    choose whether a level contains enough work before calling this function.
    """
    jobs = effective_n_jobs(n_jobs)
    if plan is not None and jobs != plan.jobs:
        msg = "Process plan and actual worker count differ"
        raise ValueError(msg)
    started = perf_counter()
    leases = {}
    succeeded = False
    try:
        if jobs == 1:
            results = [function(*args) for args in arguments]
            succeeded = True
            return results
        arguments = iter(arguments)
        first = next(arguments, None)
        if first is None:
            succeeded = True
            return []
        with _process_pool(jobs) as executor:
            with _LOCK:
                pool_entry = next(
                    (
                        entry
                        for entry in _POOLS.values()
                        if entry["executor"] is executor
                    ),
                    None,
                )

            def submit(args):
                args = list(args)
                entries = []
                for index in shared_arg_indices:
                    value = args[index]
                    identity = id(value)
                    if identity not in leases:
                        # Keep the original alive while streaming this map.
                        leases[identity] = (value, acquire_array(value))
                    entry = leases[identity][1]
                    if entry is not None:
                        args[index] = entry["ref"]
                        entries.append(entry)
                entries = tuple(entries)
                for entry in entries:
                    retain_array(entry)
                if pool_entry is not None:
                    with _LOCK:
                        pool_entry["pending"] += 1
                try:
                    future = (
                        executor.submit(call_with_shared_data, function, tuple(args))
                        if entries
                        else executor.submit(function, *args)
                    )
                except BaseException:
                    if pool_entry is not None:
                        with _LOCK:
                            pool_entry["pending"] -= 1
                    for entry in entries:
                        release_array(entry)
                    raise

                def finished(_future):
                    if pool_entry is not None:
                        with _LOCK:
                            pool_entry["pending"] -= 1
                    for entry in entries:
                        release_array(entry)

                future.add_done_callback(finished)
                return future

            pending = deque()
            try:
                pending.append(submit(first))
                for _ in range(jobs * 2 - 1):
                    args = next(arguments, None)
                    if args is None:
                        break
                    pending.append(submit(args))
                results = []
                while pending:
                    results.append(pending.popleft().result())
                    args = next(arguments, None)
                    if args is not None:
                        pending.append(submit(args))
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
        succeeded = True
        return results
    finally:
        if plan is not None:
            plan.record["shared_arrays"] = sum(
                entry is not None for _, entry in leases.values()
            )
            plan.record["shared_bytes"] = sum(
                entry["ref"].nbytes for _, entry in leases.values() if entry is not None
            )
            if succeeded:
                plan.model.observe(plan, perf_counter() - started)
        for _array, entry in leases.values():
            if entry is not None:
                release_array(entry)


def shutdown_process_pools(*, wait=True):
    """Release idle pools, for callers explicitly managing service lifecycle."""
    with _LOCK:
        idle = [
            (key, entry)
            for key, entry in _POOLS.items()
            if entry["users"] == 0 and not entry.get("pending", 0)
        ]
        for key, _ in idle:
            del _POOLS[key]
    for _, entry in idle:
        entry["executor"].shutdown(wait=wait)
    clear_shared_data()
