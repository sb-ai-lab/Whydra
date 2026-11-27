"""Compact process tasks for the ordered PDSEP search used by base FCI/CFCI.

Only searches of different neighbours in one fixed row run concurrently.
Rows commit in their original order, preserving the live adjacency snapshot
used to construct the next row's candidate sets.
"""

import math
from collections import OrderedDict
from contextlib import nullcontext
from itertools import islice
from typing import Any

from ..graph_core.ci_batch import (
    fisher_z_batch_from_corr,
    supports_fisher_batch,
)
from ..graph_core.native_fisher import native_pdsep, native_pdsep_available
from ..graph_core.process_execution import plan_process_work, process_map
from ..graph_core.validation import effective_n_jobs
from .pdsep_candidates import PossibleDsepRow, pdsep_conditions

MIN_ROW_TESTS = 4096
MIN_PAIR_TESTS = 512


def _pair_cost(tf, adjacent, cap):
    inside = sum(node in adjacent for node in tf)
    if cap >= len(tf):
        return (1 << len(tf)) - (1 << inside)
    return sum(
        math.comb(len(tf), order) - (math.comb(inside, order) if order <= inside else 0)
        for order in range(1, cap + 1)
    )


def _compact_pair_search(
    x,
    y,
    tf,
    diff_set,
    adjacent,
    cap,
    alpha,
    initial_pmax,
    cached,
    retain,
    start_order=1,
):
    """Ordered pair search; yield bounded numerical windows to its worker."""
    result = {
        "x": x,
        "y": y,
        "separator": None,
        "pmax": initial_pmax,
        "order": 0,
        "consumed": 0,
        "requests": 0,
        "computed": 0,
        "outer_hits": 0,
        "cit_hits": 0,
        "updates": [],
        "error": None,
        "discarded_cit_insertions": 0,
        "discarded_outer_insertions": 0,
    }
    updates = OrderedDict()
    adjacent = frozenset(adjacent)
    outside = tuple(node for node in tf if node not in adjacent)
    for order in range(start_order, cap + 1):
        result["order"] = order
        candidates = pdsep_conditions(tf, diff_set if order == 1 else outside, order)
        window = 32
        while True:
            conditions = list(islice(candidates, window))
            if not conditions:
                break
            normalized = [
                tuple(sorted(int(node) for node in values)) for values in conditions
            ]
            values: list[float | BaseException | None] = [None] * len(conditions)
            missing, positions = [], []
            for index, nodes in enumerate(normalized):
                hit = cached.get(nodes)
                if hit is None:
                    missing.append((x, y, nodes))
                    positions.append(index)
                else:
                    values[index], origin = hit
                    result["outer_hits" if origin == 0 else "cit_hits"] += 1
            computed = yield missing
            result["requests"] += len(conditions)
            result["computed"] += len(missing)
            for index, value in zip(positions, computed, strict=False):
                values[index] = value
            # Return a bounded tail of reusable values, including speculative
            # numerical work. Error envelopes are never cached.
            if retain:
                for nodes, value in zip(normalized, values, strict=False):
                    hit = cached.get(nodes)
                    if not isinstance(value, BaseException) and (
                        hit is None or hit[1] == 1
                    ):
                        updates[nodes] = value
                        updates.move_to_end(nodes)
                        if len(updates) > retain:
                            discarded, _old_value = updates.popitem(last=False)
                            result["discarded_outer_insertions"] += 1
                            if cached.get(discarded) is None:
                                result["discarded_cit_insertions"] += 1
            for nodes, value in zip(conditions, values, strict=False):
                result["consumed"] += 1
                if isinstance(value, BaseException):
                    result["error"] = value
                    result["updates"] = list(updates.items())
                    return result
                assert value is not None
                result["pmax"] = max(result["pmax"], value)
                if value >= alpha:
                    result["separator"] = tuple(int(node) for node in nodes)
                    result["updates"] = list(updates.items())
                    return result
            window = min(2048, window * 4)
    result["updates"] = list(updates.items())
    return result


def _compact_pair(corr, n_samples, *args, **kwargs):
    """Search one pair with numerical inputs; no graph, CIT, or callable IPC."""
    if not kwargs:
        native = native_pdsep(
            corr, n_samples, [(0, args)], batch=fisher_z_batch_from_corr
        )
        if native is not None:
            return native[0][1]
    search = _compact_pair_search(*args, **kwargs)
    alpha = kwargs.get("alpha", args[6] if len(args) > 6 else None)
    values = None
    while True:
        try:
            missing = search.send(values)
        except StopIteration as stopped:
            return stopped.value
        values = fisher_z_batch_from_corr(
            corr,
            n_samples,
            missing,
            n_jobs=1,
            return_exceptions=True,
            decision_alpha=alpha,
        )


def _compact_pair_chunk(corr, n_samples, descriptors):
    """Batch windows across up to four independent searches in one process.

    Keeping generators in the worker removes coordinator round trips and
    combines short pair prefixes. Each pair still consumes only its own
    ordered prefix, so speculative errors remain local until consumed.
    """
    native = native_pdsep(corr, n_samples, descriptors, batch=fisher_z_batch_from_corr)
    if native is not None:
        return native
    results = []
    remaining = iter(descriptors)
    active = []
    while True:
        while len(active) < 4:
            descriptor = next(remaining, None)
            if descriptor is None:
                break
            index, args = descriptor
            search = _compact_pair_search(*args)
            try:
                queries = next(search)
            except StopIteration as stopped:
                results.append((index, stopped.value))
            else:
                active.append((index, search, queries))
        if not active:
            return results
        requests = [query for _, _, queries in active for query in queries]
        values = fisher_z_batch_from_corr(
            corr,
            n_samples,
            requests,
            n_jobs=1,
            return_exceptions=True,
            decision_alpha=descriptors[0][1][6],
        )
        following = []
        offset = 0
        for index, search, queries in active:
            stop = offset + len(queries)
            try:
                pending = search.send(values[offset:stop])
            except StopIteration as stopped:
                results.append((index, stopped.value))
            else:
                following.append((index, search, pending))
            offset = stop
        active = following


def _weighted_pair_chunks(descriptors, n_jobs):
    """Balance whole pair searches; retain their original commit indices."""
    chunks = [[] for _ in range(min(len(descriptors), max(1, n_jobs * 2)))]
    loads = [0] * len(chunks)
    for cost, index, args in sorted(
        descriptors, key=lambda item: item[0], reverse=True
    ):
        target = min(range(len(chunks)), key=loads.__getitem__)
        chunks[target].append((index, args))
        loads[target] += cost
    return chunks


def _cache_index(cache):
    index = {}
    if cache is not None:
        for (x, y, conditions), value in cache.items():
            index.setdefault((x, y), {})[conditions] = value
    return index


def _forget(index, key):
    pair, conditions = key[:2], key[2]
    values = index.get(pair)
    if values is not None:
        values.pop(conditions, None)
        if not values:
            index.pop(pair, None)


class CompactPDSEP:
    """Keep bounded cache indices and lease the shared process executor."""

    def __init__(self, indep_test, n_jobs, backend="processes"):
        self.jobs = effective_n_jobs(n_jobs)
        self.backend = backend
        self.pending_plan = None
        self.last_jobs = 1
        self.context = getattr(indep_test, "_pdsep_batch_context", None)
        self.enabled = (
            (self.jobs > 1 or native_pdsep_available())
            and self.context is not None
            and supports_fisher_batch(indep_test)
        )
        self.pool = None
        self.cit_index = {}
        self.outer_index = {}
        self._index_stamp = None
        self.stats: dict[str, Any] = {
            "process_rows": 0,
            "process_pairs": 0,
            "worker_ci_computations": 0,
            "worker_cache_hits": 0,
            "worker_speculative_tests": 0,
            "cache_index_rebuilds": 0,
            "process_tasks": 0,
            "serial_rows": 0,
            "native_chunks": 0,
        }

    def observe_serial(self, seconds):
        """Learn from the caller's ordered fallback, without replaying any CI."""
        if self.pending_plan is not None:
            self.pending_plan.model.observe(self.pending_plan, seconds)
            self.pending_plan = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        # The reusable executor belongs to process_execution, not this fit.
        self.pool = None

    def _cache_stamp(self):
        assert self.context is not None
        cit, outer, _stats, _phase, _lock = self.context
        return (
            cit.cache_misses,
            cit.cache_evictions,
            len(cit._cache),
            cit.cache_size,
            getattr(outer, "evictions", 0),
            len(outer) if outer is not None else 0,
            getattr(outer, "max_entries", 0),
            getattr(cit, "decision_alpha", None),
            getattr(getattr(cit._cache, "owner", None), "version", None),
        )

    def _refresh_indices(self):
        assert self.context is not None
        cit, outer, _stats, _phase, lock = self.context
        with lock if lock is not None else nullcontext():
            with cit._lock:
                stamp = self._cache_stamp()
                if self._index_stamp == stamp:
                    return
                self.cit_index = _cache_index(cit._cache)
            self.outer_index = _cache_index(outer)
            self._index_stamp = stamp
            self.stats["cache_index_rebuilds"] += 1

    def _initialize(self):
        self._refresh_indices()
        self.pool = True

    def evaluate_row(self, x, adjacent, possible, m_max, alpha, pmax_row, protected):
        """Build and evaluate a Possible-D-SEP search plan for one source node."""
        row = PossibleDsepRow(x, possible, adjacent, protected)
        return self.evaluate_plan(x, row, m_max, alpha, pmax_row)

    def evaluate_plan(self, x, row, m_max, alpha, pmax_row):
        """Evaluate eligible targets for a row, or return None when batching is unavailable."""
        self.pending_plan = None
        if not self.enabled or not row.outside:
            return None
        tasks = []
        for y in row.targets:
            tf = row.pool(y)
            cap = int(min(len(tf), m_max))
            if cap:
                tasks.append((_pair_cost(tf, row.neighbors, cap), y, tf, cap))
        if (
            len(tasks) < 2
            or sum(task[0] for task in tasks) < MIN_ROW_TESTS
            or max(task[0] for task in tasks) < MIN_PAIR_TESTS
        ):
            return None
        plan = plan_process_work(
            sum(task[0] for task in tasks),
            self.jobs,
            family="pdsep_row",
            backend=self.backend,
            tasks=len(tasks),
            stats=self.stats,
        )
        if plan.jobs == 1 and not native_pdsep_available():
            self.pending_plan = plan
            return None
        if self.pool is None:
            self._initialize()
        else:
            # Smaller intervening rows may use the scalar path and change LRU
            # state. Refresh from the bounded authoritative caches at a barrier.
            self._refresh_indices()
        assert self.context is not None
        cit, outer, _stats, _phase, _lock = self.context
        retain = max(cit.cache_size, getattr(outer, "max_entries", 0))
        calls = []
        for _cost, y, tf, cap in sorted(tasks, reverse=True):
            pair = (min(int(x), y), max(int(x), y))
            candidates = frozenset(tf)
            cached = {
                conditions: (value, 1)
                for conditions, value in self.cit_index.get(pair, {}).items()
                if 0 < len(conditions) <= cap
                and all(node in candidates for node in conditions)
            }
            cached.update(
                {
                    conditions: (value, 0)
                    for conditions, value in self.outer_index.get(pair, {}).items()
                    if 0 < len(conditions) <= cap
                    and all(node in candidates for node in conditions)
                }
            )
            calls.append(
                (
                    cit.corr_matrix,
                    cit.n_samples,
                    int(x),
                    y,
                    tf,
                    row.outside,
                    tuple(row.neighbors),
                    cap,
                    alpha,
                    float(pmax_row[y]),
                    cached,
                    retain,
                )
            )
        results = process_map(
            _compact_pair, calls, n_jobs=plan.jobs, plan=plan, shared_arg_indices=(0,)
        )
        self.last_jobs = plan.jobs
        self.stats["native_chunks"] += sum(
            "native_execution" in result for result in results
        )
        self.stats["process_rows" if plan.jobs > 1 else "serial_rows"] += 1
        if plan.jobs > 1:
            self.stats["process_pairs"] += len(results)
            self.stats["process_tasks"] += len(calls)
        return {result["y"]: result for result in results}

    def evaluate_level(self, pairs, order, alpha, pmax, min_tasks=4096):
        """Evaluate a frozen PDSEP depth without materializing its subsets.

        Only immutable pair descriptors cross the process boundary. The caller
        commits results in its original pair order after this level's barrier.
        """
        self.pending_plan = None
        if not self.enabled or len(pairs) < 2:
            return None
        tasks = []
        for pair_idx, pair in pairs:
            inside = sum(node in pair["adj_x"] for node in pair["tf"])
            cost = (
                len(pair["diff_set"])
                if order == 1
                else math.comb(len(pair["tf"]), order)
                - (math.comb(inside, order) if order <= inside else 0)
            )
            if cost:
                tasks.append((cost, pair_idx, pair))
        if len(tasks) < 2 or sum(task[0] for task in tasks) < max(
            MIN_ROW_TESTS, min_tasks
        ):
            return None
        plan = plan_process_work(
            sum(task[0] for task in tasks),
            self.jobs,
            family=f"pdsep_level:{order}",
            backend=self.backend,
            tasks=len(tasks),
            stats=self.stats,
        )
        if plan.jobs == 1 and not native_pdsep_available():
            self.pending_plan = plan
            return None
        if self.pool is None:
            self._initialize()
        else:
            self._refresh_indices()
        assert self.context is not None
        cit, outer, _stats, _phase, _lock = self.context
        retain = max(cit.cache_size, getattr(outer, "max_entries", 0))
        descriptors = []
        for cost, pair_idx, pair in tasks:
            x, y = pair["x"], pair["y"]
            key = (min(x, y), max(x, y))
            candidates = frozenset(pair["tf"])
            cached = {
                conditions: (value, 1)
                for conditions, value in self.cit_index.get(key, {}).items()
                if len(conditions) == order
                and all(node in candidates for node in conditions)
            }
            cached.update(
                {
                    conditions: (value, 0)
                    for conditions, value in self.outer_index.get(key, {}).items()
                    if len(conditions) == order
                    and all(node in candidates for node in conditions)
                }
            )
            descriptors.append(
                (
                    cost,
                    pair_idx,
                    (
                        x,
                        y,
                        pair["tf"],
                        pair["diff_set"],
                        tuple(pair["adj_x"]),
                        order,
                        alpha,
                        float(pmax[x, y]),
                        cached,
                        retain,
                        order,
                    ),
                )
            )
        chunks = _weighted_pair_chunks(descriptors, plan.jobs)
        results = process_map(
            _compact_pair_chunk,
            ((cit.corr_matrix, cit.n_samples, chunk) for chunk in chunks),
            n_jobs=plan.jobs,
            plan=plan,
            shared_arg_indices=(0,),
        )
        self.last_jobs = plan.jobs
        self.stats["native_chunks"] += sum(
            bool(chunk) and "native_execution" in chunk[0][1] for chunk in results
        )
        self.stats["process_rows" if plan.jobs > 1 else "serial_rows"] += 1
        if plan.jobs > 1:
            self.stats["process_pairs"] += len(descriptors)
            self.stats["process_tasks"] += len(chunks)
        return dict(item for chunk in results for item in chunk)

    def publish(self, result):
        """Apply a worker's cache updates and CI statistics to the parent search."""
        assert self.context is not None
        cit, outer, stats, phase, lock = self.context
        with lock if lock is not None else nullcontext():
            stats["total"] += result["requests"]
            stats[phase["name"]] += result["requests"]
            if outer is not None:
                stats["ci_cache_hits"] += result["outer_hits"]
                stats["ci_cache_misses"] += result["requests"] - result["outer_hits"]
            pair = (min(result["x"], result["y"]), max(result["x"], result["y"]))
            with cit._lock:
                cit.cache_hits += result["cit_hits"]
                cit.cache_misses += result["computed"]
                if cit.cache_size:
                    # The worker returns only its bounded update tail. Its
                    # discarded insertions also displaced cached entries even
                    # though replaying those values in the parent is unnecessary.
                    cit.cache_evictions += result["discarded_cit_insertions"]
                for conditions, value in result["updates"]:
                    key = (*pair, conditions)
                    if cit.cache_size:
                        cit._cache[key] = value
                        cit._cache.move_to_end(key)
                        self.cit_index.setdefault(pair, {})[conditions] = value
                        if len(cit._cache) > cit.cache_size:
                            expired, _value = cit._cache.popitem(last=False)
                            _forget(self.cit_index, expired)
                            cit.cache_evictions += 1
            if outer is not None and outer.max_entries:
                outer.evictions += result["discarded_outer_insertions"]
                for conditions, value in result["updates"]:
                    key = (*pair, conditions)
                    expired = (
                        next(iter(outer))
                        if key not in outer and len(outer) >= outer.max_entries
                        else None
                    )
                    outer[key] = value
                    if expired is not None:
                        _forget(self.outer_index, expired)
                    self.outer_index.setdefault(pair, {})[conditions] = value
            self._index_stamp = self._cache_stamp()
        self.stats["worker_ci_computations"] += result["computed"]
        self.stats["worker_cache_hits"] += result["outer_hits"] + result["cit_hits"]
        self.stats["worker_speculative_tests"] += (
            result["requests"] - result["consumed"]
        )
