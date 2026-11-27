import math
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from itertools import combinations, islice
from typing import TYPE_CHECKING

import numpy as np

from ...background_knowledge import BKPhase, MatrixEncoding, apply_bk_checkpoint
from ..graph_core import CIT, CausalGraph, fisher_z_from_corr
from ..graph_core import cache as _cache_exports
from ..graph_core.cache import ci_cache_for
from ..graph_core.ci_batch import (
    batch_execution_stats,
    evaluate_ci_batch,
    fisher_z_batch_from_corr,
    supports_fisher_batch,
)
from ..graph_core.conditioning import (
    bounded_size,
    conditioning_allowed,
    configure_conditioning,
)
from ..graph_core.endpoints import Endpoint
from ..graph_core.process_execution import process_map
from ..graph_core.validation import effective_n_jobs
from ._counted_ci import counted_fisherz
from .ci_adapters import attach_counted_batch, observed_phase_backend
from .fci_plus_search import (
    augment_native,
    augmentation_inputs,
    dependent_endpoints,
    hierarchy_closure,
    naa_ancestors,
    pdsep_candidates,
    set_arrowheads,
)
from .hierarchy_cache import hierarchy_cache, hierarchy_changed

# The sequential and parallel searches use the same PAG orientation rules.
from .pag_rules import PAG_ARROW, PAG_CIRCLE, PAG_TAIL, udag2pag
from .separators import sepset_to_pair_dict as _sepset_to_pair_dict
from .skeleton import SkeletonDiscovery

BoundedCache = _cache_exports.BoundedCache


def _observed_phase_backend(test, before, coarse=None, extra=()):
    return observed_phase_backend(batch_execution_stats(test), before, coarse, extra)


_DEMO_EXPORTS = {
    "make_collider_data": ".rfci_algo",
    "rfci_stable": ".rfci_algo",
    "fci_stable_v2": ".fci_algo_v2",
}


def __getattr__(name):
    module_name = _DEMO_EXPORTS.get(name)
    if module_name is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    value = getattr(import_module(module_name, __package__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_DEMO_EXPORTS))


TAIL = Endpoint.TAIL.value  # -1
ARROW = Endpoint.ARROW.value  # 1
CIRCLE = Endpoint.CIRCLE.value  # 2
NULL = Endpoint.NULL.value  # 0


def _ci_cache_key(i: int, j: int, S):
    a, b = (int(i), int(j)) if i <= j else (int(j), int(i))
    if S is None:
        cond = ()
    elif isinstance(S, (set, frozenset, list, tuple, np.ndarray)):
        cond = tuple(sorted(int(x) for x in S))
    else:
        cond = (int(S),)
    return a, b, cond


def _resolve_fci_plus_backend(backend, n_vars, process_min_vars):
    requested = "auto" if backend in (None, "auto") else str(backend)
    if requested == "auto":
        effective = "processes"
    elif requested in {"threads", "processes"}:
        effective = requested
    else:
        msg = "FCI+ backend must be 'auto', 'threads', or 'processes'"
        raise ValueError(msg)
    return requested, effective


def _sepset_vars(seps):
    vars_set = set()
    for x in seps or []:
        if isinstance(x, (tuple, list, set, np.ndarray)):
            vars_set.update(x)
        else:
            vars_set.add(x)
    return vars_set


def _has_sepset(seps):
    return seps is not None


def _edge_exists(mat, i, j):
    return mat[i][j] != NULL or mat[j][i] != NULL


def _is_bidirected(mat, i, j):
    return mat[i][j] == ARROW and mat[j][i] == ARROW


def _naa_ancestors(x, mat):
    return naa_ancestors(x, mat)


def _get_pdseps(mat):
    return pdsep_candidates(mat)


def _chunk_sequence(items, n_chunks):
    if not items:
        return []
    n_chunks = max(1, min(int(n_chunks), len(items)))
    chunk_size = math.ceil(len(items) / n_chunks)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _thread_worker_count(n_jobs):
    n_jobs = int(n_jobs)
    return n_jobs if n_jobs > 0 else (os.cpu_count() or 1)


def _get_thread_executor(cg, n_jobs):
    executor = getattr(cg, "_fci_plus_thread_executor", None)
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=_thread_worker_count(n_jobs))
        cg._fci_plus_thread_executor = executor
    return executor


def _record_parallel_batch(cg, backend, n_tasks):
    stats = getattr(cg, "_fci_plus_parallel_stats", None)
    phase = getattr(cg, "_fci_plus_parallel_phase", None)
    if stats is None or phase is None:
        return
    prefix = phase["name"]
    stats[f"{prefix}_{backend}_batches"] = (
        int(stats.get(f"{prefix}_{backend}_batches", 0)) + 1
    )
    stats[f"{prefix}_{backend}_tasks"] = int(
        stats.get(f"{prefix}_{backend}_tasks", 0)
    ) + int(n_tasks)


def _keyed_ci_thread_chunk(cg, tasks):
    return [(key, cg.ci_test(a, b, list(cond))) for key, a, b, cond in tasks]


def _ci_thread_chunk(cg, tasks):
    return [(x, y, S, cg.ci_test(x, y, list(S))) for x, y, S in tasks]


# Fisher-Z lives in graph_core so every backend computes it identically;
# a local copy here would silently drift from CIT.fisher_z.
_fisher_z_from_corr = fisher_z_from_corr


def _fisher_z_task_chunk(corr_matrix, n_samples, tasks, alpha=None):
    values = fisher_z_batch_from_corr(
        corr_matrix, n_samples, tasks, decision_alpha=alpha
    )
    return [(x, y, S, value) for (x, y, S), value in zip(tasks, values, strict=False)]


def _fisher_z_keyed_task_chunk(corr_matrix, n_samples, tasks, alpha=None):
    values = fisher_z_batch_from_corr(
        corr_matrix,
        n_samples,
        [(x, y, S) for _, x, y, S in tasks],
        decision_alpha=alpha,
    )
    return [(key, value) for (key, _, _, _), value in zip(tasks, values, strict=False)]


def _augment_fisher_z_chunk(corr_matrix, n_samples, alpha, tasks):
    pvals = []
    arrow_updates = []
    values = fisher_z_batch_from_corr(
        corr_matrix,
        n_samples,
        [(x, y, S) for _, x, y, S, _ in tasks],
        decision_alpha=alpha,
    )
    for (key, _x, _y, _S, metas), pval in zip(tasks, values, strict=False):
        pvals.append((key, pval))
        if pval < alpha:
            for del_nodes, adj in metas:
                for node in del_nodes:
                    arrow_updates.append((int(node), int(adj)))
    return pvals, arrow_updates


def _run_augment_process_tasks(
    tasks,
    task_meta,
    alpha,
    n_jobs,
    process_ci_info,
    ci_cache=None,
    record_process_tests=None,
    record_cache_hits=None,
):
    n_jobs_eff = 1 if n_jobs in (None, 0, 1) else int(n_jobs)
    arrow_updates = []
    missing_by_key = {}

    for (x, y, S), (del_nodes, adj) in zip(tasks, task_meta, strict=False):
        key = _ci_cache_key(x, y, S)
        cached = ci_cache.get(key) if ci_cache is not None else None
        if cached is not None:
            if cached < alpha:
                for node in del_nodes:
                    arrow_updates.append((int(node), int(adj)))
            continue

        if key not in missing_by_key:
            a, b, cond = key
            missing_by_key[key] = {
                "task": (key, a, b, cond),
                "metas": [],
            }
        missing_by_key[key]["metas"].append((del_nodes, adj))

    miss_tasks = [(*item["task"], item["metas"]) for item in missing_by_key.values()]
    chunks = _chunk_sequence(miss_tasks, max(n_jobs_eff * 4, n_jobs_eff))
    chunk_results = (
        process_map(
            _augment_fisher_z_chunk,
            (
                (
                    process_ci_info["corr_matrix"],
                    process_ci_info["n_samples"],
                    alpha,
                    chunk,
                )
                for chunk in chunks
            ),
            n_jobs=n_jobs_eff,
        )
        if chunks
        else []
    )

    for pvals, chunk_arrow_updates in chunk_results:
        if ci_cache is not None:
            for key, pval in pvals:
                ci_cache[key] = pval
        arrow_updates.extend(chunk_arrow_updates)

    if record_process_tests is not None:
        record_process_tests(len(miss_tasks))
    if record_cache_hits is not None:
        record_cache_hits(len(tasks) - len(miss_tasks))

    return arrow_updates


def _ci_batch(
    cg,
    tasks,
    n_jobs,
    parallel_backend="processes",
    process_ci_info=None,
    record_process_tests=None,
    record_cache_hits=None,
    ci_cache=None,
):
    tasks = list(tasks)
    tasks = [(int(x), int(y), tuple(int(v) for v in S)) for x, y, S in tasks]
    if not tasks:
        return []

    if supports_fisher_batch(cg.test):
        values = evaluate_ci_batch(
            cg.test, tasks, n_jobs=n_jobs, backend=parallel_backend
        )
        _record_parallel_batch(cg, "batch", len(tasks))
        return [
            (x, y, list(S), value)
            for (x, y, S), value in zip(tasks, values, strict=False)
        ]

    n_jobs_eff = 1 if n_jobs in (None, 0, 1) else int(n_jobs)
    process_min_tasks = int(getattr(cg, "_fci_plus_process_min_tasks", 256))
    if ci_cache is not None:
        cached_results: dict[int, tuple[int, int, list[int], float]] = {}
        missing_by_key = {}
        for idx, (x, y, S) in enumerate(tasks):
            key = _ci_cache_key(x, y, S)
            cached = ci_cache.get(key)
            if cached is not None:
                cached_results[idx] = (x, y, list(S), cached)
                continue

            if key not in missing_by_key:
                a, b, cond = key
                missing_by_key[key] = {
                    "task": (key, a, b, cond),
                    "positions": [],
                }
            missing_by_key[key]["positions"].append(idx)

        miss_tasks = [item["task"] for item in missing_by_key.values()]
        use_process_batch = (
            parallel_backend == "processes"
            and process_ci_info is not None
            and process_ci_info.get("method") == "fisherz"
            and n_jobs_eff != 1
            and len(miss_tasks) >= process_min_tasks
        )
        if use_process_batch and process_ci_info is not None:
            chunks = _chunk_sequence(miss_tasks, max(n_jobs_eff * 4, n_jobs_eff))
            chunk_results = (
                process_map(
                    _fisher_z_keyed_task_chunk,
                    (
                        (
                            process_ci_info["corr_matrix"],
                            process_ci_info["n_samples"],
                            chunk,
                            process_ci_info.get("decision_alpha"),
                        )
                        for chunk in chunks
                    ),
                    n_jobs=n_jobs_eff,
                )
                if chunks
                else []
            )
            computed = [item for chunk in chunk_results for item in chunk]
            _record_parallel_batch(cg, "process", len(miss_tasks))
            if record_process_tests is not None:
                record_process_tests(len(miss_tasks))
        elif n_jobs_eff == 1 or parallel_backend != "threads":
            computed = [
                (key, cg.ci_test(a, b, list(cond))) for key, a, b, cond in miss_tasks
            ]
        else:
            executor = _get_thread_executor(cg, n_jobs_eff)
            chunks = _chunk_sequence(
                miss_tasks,
                max(_thread_worker_count(n_jobs_eff) * 2, 1),
            )
            futures = [
                executor.submit(_keyed_ci_thread_chunk, cg, chunk) for chunk in chunks
            ]
            computed = [item for future in futures for item in future.result()]
            _record_parallel_batch(cg, "thread", len(miss_tasks))

        for key, pval in computed:
            ci_cache[key] = pval
            for idx in missing_by_key[key]["positions"]:
                x, y, S = tasks[idx]
                cached_results[idx] = (x, y, list(S), pval)

        if record_cache_hits is not None:
            record_cache_hits(len(tasks) - len(miss_tasks))
        return [cached_results[i] for i in range(len(tasks))]

    if n_jobs in (None, 0, 1):
        return [(x, y, list(S), cg.ci_test(x, y, list(S))) for x, y, S in tasks]

    use_process_batch = (
        parallel_backend == "processes"
        and process_ci_info is not None
        and process_ci_info.get("method") == "fisherz"
        and len(tasks) >= process_min_tasks
    )
    if use_process_batch and process_ci_info is not None:
        chunks = _chunk_sequence(tasks, max(n_jobs_eff * 4, n_jobs_eff))
        chunk_results = process_map(
            _fisher_z_task_chunk,
            (
                (
                    process_ci_info["corr_matrix"],
                    process_ci_info["n_samples"],
                    chunk,
                    process_ci_info.get("decision_alpha"),
                )
                for chunk in chunks
            ),
            n_jobs=n_jobs_eff,
        )
        results = [item for chunk in chunk_results for item in chunk]
        _record_parallel_batch(cg, "process", len(results))
        if record_process_tests is not None:
            record_process_tests(len(results))
        return [(x, y, list(S), pval) for x, y, S, pval in results]

    if parallel_backend != "threads":
        return [(x, y, list(S), cg.ci_test(x, y, list(S))) for x, y, S in tasks]
    executor = _get_thread_executor(cg, n_jobs_eff)
    chunks = _chunk_sequence(
        tasks,
        max(_thread_worker_count(n_jobs_eff) * 2, 1),
    )
    futures = [executor.submit(_ci_thread_chunk, cg, chunk) for chunk in chunks]
    computed = [item for future in futures for item in future.result()]
    _record_parallel_batch(cg, "thread", len(tasks))
    return [(x, y, list(S), pval) for x, y, S, pval in computed]


def AugmentGraph(
    cg: CausalGraph,
    p: int,
    alpha: float,
    n_jobs: int = 1,
    parallel_backend: str = "processes",
    process_ci_info=None,
    record_process_tests=None,
    record_cache_hits=None,
    ci_cache=None,
    stats=None,
):
    """Orient additional arrowheads using batched tests of augmented separating sets."""
    if augment_native(
        cg,
        alpha,
        n_jobs,
        parallel_backend,
        _sepset_vars,
        stats,
        int(getattr(cg, "_fci_plus_process_min_tasks", 256)),
    ):
        return cg.G.graph
    pag = cg.G.graph.copy()
    probes = list(augmentation_inputs(cg, p, _sepset_vars))
    tasks = [query for query, _ in probes]
    task_meta = [effect for _, effect in probes]

    if stats is not None:
        stats["augment_tasks"] = int(stats.get("augment_tasks", 0)) + len(tasks)

    if (
        n_jobs not in (None, 0, 1)
        and parallel_backend == "processes"
        and process_ci_info is not None
        and process_ci_info.get("method") == "fisherz"
        and len(tasks) >= int(getattr(cg, "_fci_plus_process_min_tasks", 256))
    ):
        arrow_updates = _run_augment_process_tasks(
            tasks,
            task_meta,
            alpha,
            n_jobs,
            process_ci_info,
            ci_cache=ci_cache,
            record_process_tests=record_process_tests,
            record_cache_hits=record_cache_hits,
        )
        _record_parallel_batch(cg, "process", len(tasks))
        arrowheads_added = set_arrowheads(pag, arrow_updates)
    else:
        pvals = _ci_batch(
            cg,
            tasks,
            n_jobs,
            parallel_backend=parallel_backend,
            process_ci_info=process_ci_info,
            record_process_tests=record_process_tests,
            record_cache_hits=record_cache_hits,
            ci_cache=ci_cache,
        )
        arrowheads_added = set_arrowheads(
            pag, dependent_endpoints(pvals, task_meta, alpha)
        )

    if stats is not None:
        stats["augment_arrowheads_added"] = (
            int(stats.get("augment_arrowheads_added", 0)) + arrowheads_added
        )

    cg.G.graph = pag
    return pag


def _dsep_search(cg: CausalGraph, p: int) -> deque:
    return _get_pdseps(cg.G.graph.copy())


def _get_hie(cg, S, cache=None):
    return hierarchy_closure(cg, S, _sepset_vars, cache)


def _ordered_dsep_ci(cg, x, y, candidates, n_jobs, backend="processes", **options):
    """Bound speculation and defer numerical errors until their query is used."""
    native = supports_fisher_batch(cg.test)
    maximum = max(32, min(2048, 256 * effective_n_jobs(n_jobs)))
    size = 32 if native else 1
    candidates = iter(candidates)
    while True:
        window = list(islice(candidates, size))
        if not window:
            return
        tasks = [(x, y, nodes) for nodes in window]
        if native:
            values = evaluate_ci_batch(
                cg.test, tasks, n_jobs=n_jobs, backend=backend, return_exceptions=True
            )
            _record_parallel_batch(cg, "batch", len(tasks))
        else:
            values = [
                item[3]
                for item in _ci_batch(
                    cg, tasks, n_jobs, parallel_backend=backend, **options
                )
            ]
        for nodes, value in zip(window, values, strict=False):
            if isinstance(value, BaseException):
                raise value
            yield nodes, value
        if native:
            size = min(maximum, size * 4)


def _min_dsep(cg: CausalGraph, S, i, j, alpha=0.05, n_jobs=1, backend="processes"):
    if not conditioning_allowed(cg, set(S)):
        msg = "Separator witness exceeds max_cond_set_size"
        raise ValueError(msg)
    S = tuple(sorted(set(S)))
    if not S:
        S_list = []
        cg.sepset[i][j] = cg.sepset[j][i] = [tuple(sorted(S_list))]
        hierarchy_changed(cg, ((i, j), (j, i)))
        return S_list

    S_list = list(S)
    for size in range(1, bounded_size(cg, len(S)) + 1):
        for subset, value in _ordered_dsep_ci(
            cg, i, j, combinations(S, size), n_jobs, backend
        ):
            if value >= alpha:
                S_list = list(subset)
                cg.sepset[i][j] = cg.sepset[j][i] = [tuple(sorted(S_list))]
                hierarchy_changed(cg, ((i, j), (j, i)))
                return S_list

    cg.sepset[i][j] = cg.sepset[j][i] = [tuple(sorted(S_list))]
    hierarchy_changed(cg, ((i, j), (j, i)))

    return S_list


def _dsep(
    cg,
    alpha,
    x,
    y,
    base_x,
    base_y,
    k,
    p,
    n_jobs=1,
    parallel_backend="processes",
    process_ci_info=None,
    record_process_tests=None,
    record_cache_hits=None,
    ci_cache=None,
    stats=None,
    protected_edges=(),
):
    protected_pairs = {frozenset((int(a), int(b))) for a, b in protected_edges}
    if frozenset((int(x), int(y))) in protected_pairs:
        return False, deque()
    max_n = bounded_size(cg, min(k, len(base_x)))
    max_m = bounded_size(cg, min(k, len(base_y)))
    hierarchy = hierarchy_cache(cg, _sepset_vars)
    for n in range(1, max_n + 1):
        for m in range(1, max_m + 1):

            def candidates(n, m):
                for z_x in combinations(base_x, n):
                    for z_y in combinations(base_y, m):
                        seed = frozenset((x, y, *z_x, *z_y))
                        nodes = tuple(sorted(_get_hie(cg, seed, hierarchy) - {x, y}))
                        if conditioning_allowed(cg, nodes):
                            if stats is not None:
                                stats["dsep_candidate_sets"] = (
                                    stats.get("dsep_candidate_sets", 0) + 1
                                )
                            yield nodes

            for Z_sorted, p_value in _ordered_dsep_ci(
                cg,
                x,
                y,
                candidates(n, m),
                n_jobs,
                parallel_backend,
                process_ci_info=process_ci_info,
                record_process_tests=record_process_tests,
                record_cache_hits=record_cache_hits,
                ci_cache=ci_cache,
            ):
                if p_value >= alpha:
                    Z = _min_dsep(
                        cg,
                        Z_sorted,
                        x,
                        y,
                        alpha=alpha,
                        n_jobs=n_jobs,
                        backend=parallel_backend,
                    )
                    cg.sepset[x][y] = [tuple(sorted(Z))]
                    cg.sepset[y][x] = [tuple(sorted(Z))]
                    hierarchy_changed(cg, ((x, y), (y, x)))
                    # Found a valid D-SEP set: remove the edge X-Y to ensure progress.
                    cg.G.graph[x][y] = NULL
                    cg.G.graph[y][x] = NULL
                    hierarchy_changed(cg, graph=True)
                    AugmentGraph(
                        cg,
                        p,
                        alpha,
                        n_jobs=n_jobs,
                        parallel_backend=parallel_backend,
                        process_ci_info=process_ci_info,
                        record_process_tests=record_process_tests,
                        record_cache_hits=record_cache_hits,
                        ci_cache=ci_cache,
                        stats=stats,
                    )
                    if stats is not None:
                        stats["dsep_links_found"] = (
                            int(stats.get("dsep_links_found", 0)) + 1
                        )
                    return True, _dsep_search(cg, p)

    return False, deque()


def fci_plus_stable(
    cg: CausalGraph,
    data: np.ndarray,
    alpha: float,
    indep_test="fisherz",
    k=None,
    n_jobs: int = 1,
    verbose: bool = False,
    use_ci_cache=True,
    skeleton_backend=None,
    augment_backend=None,
    dsep_backend=None,
    ci_backend=None,
    skeleton_n_jobs=None,
    augment_n_jobs=None,
    dsep_n_jobs=None,
    process_min_tasks=256,
    process_min_vars=150,
    background_knowledge=None,
    show_progress: bool = False,
    max_cond_set_size=None,
) -> np.ndarray:
    """Run stable FCI+ with cached CI results and return its endpoint matrix."""
    # print(data)
    configure_conditioning(cg, max_cond_set_size)
    total_t0 = time.perf_counter()
    _n, p = data.shape
    n_jobs_eff = effective_n_jobs(n_jobs)
    skeleton_jobs_eff = (
        n_jobs_eff if skeleton_n_jobs in (None, 0) else int(skeleton_n_jobs)
    )
    augment_jobs_eff = (
        n_jobs_eff if augment_n_jobs in (None, 0) else int(augment_n_jobs)
    )
    dsep_jobs_eff = n_jobs_eff if dsep_n_jobs in (None, 0) else int(dsep_n_jobs)
    skeleton_backend_requested, skeleton_backend_eff = _resolve_fci_plus_backend(
        skeleton_backend,
        p,
        process_min_vars,
    )
    augment_backend_requested, augment_backend_eff = _resolve_fci_plus_backend(
        augment_backend if augment_backend is not None else ci_backend,
        p,
        process_min_vars,
    )
    dsep_backend_requested, dsep_backend_eff = _resolve_fci_plus_backend(
        dsep_backend if dsep_backend is not None else ci_backend,
        p,
        process_min_vars,
    )
    use_process_skeleton = (
        skeleton_jobs_eff != 1
        and indep_test == "fisherz"
        and skeleton_backend_eff == "processes"
    )
    use_process_augment = (
        augment_jobs_eff != 1
        and indep_test == "fisherz"
        and augment_backend_eff == "processes"
    )
    use_process_dsep = (
        dsep_jobs_eff != 1
        and indep_test == "fisherz"
        and dsep_backend_eff == "processes"
    )
    cit = (
        cg.test
        if isinstance(cg.test, CIT)
        and cg.test.method == indep_test
        and np.array_equal(cg.test.data, data)
        else CIT(data, method=indep_test)
    )
    cit.max_cond_set_size = max_cond_set_size
    native = supports_fisher_batch(cit)
    use_process_skeleton = use_process_skeleton and native
    use_process_augment = use_process_augment and native
    use_process_dsep = use_process_dsep and native
    process_ci_info = (
        {
            "method": indep_test,
            "corr_matrix": cit.corr_matrix,
            "n_samples": cit.n_samples,
            "decision_alpha": alpha,
        }
        if (use_process_skeleton or use_process_augment or use_process_dsep)
        else None
    )
    ci_stats = {
        "total": 0,
        "skeleton": 0,
        "augment": 0,
        "dsep": 0,
        "skeleton_backend": "serial",
        "augment_backend": "serial",
        "ci_backend": "serial",
        "dsep_backend": "serial",
        "skeleton_n_jobs": int(skeleton_jobs_eff),
        "augment_n_jobs": int(augment_jobs_eff),
        "dsep_n_jobs": int(dsep_jobs_eff),
        "process_min_tasks": int(process_min_tasks),
        "skeleton_backend_requested": skeleton_backend_requested,
        "augment_backend_requested": augment_backend_requested,
        "dsep_backend_requested": dsep_backend_requested,
        "process_min_vars": int(process_min_vars),
        "n_vars": int(p),
        "n_jobs": int(n_jobs_eff),
    }
    if use_ci_cache:
        ci_stats["ci_cache_hits"] = 0
        ci_stats["ci_cache_misses"] = 0
    phase = {"name": "skeleton"}
    ci_cache = ci_cache_for(cit) if use_ci_cache else None
    ci_inflight = {}
    stats_lock = threading.Lock()
    cg._fci_plus_process_min_tasks = int(process_min_tasks)
    cg._fci_plus_parallel_stats = ci_stats
    cg._fci_plus_parallel_phase = phase

    def scalar_cit(i, j, S):
        with stats_lock:
            ci_stats["total"] += 1
            ci_stats[phase["name"]] += 1
        if ci_cache is None:
            return cit(i, j, S)

        key = _ci_cache_key(i, j, S)
        while True:
            with stats_lock:
                cached = ci_cache.get(key)
                if cached is not None:
                    ci_stats["ci_cache_hits"] += 1
                    return cached
                event = ci_inflight.get(key)
                if event is None:
                    event = threading.Event()
                    ci_inflight[key] = event
                    ci_stats["ci_cache_misses"] += 1
                    owner = True
                else:
                    owner = False
            if owner:
                break
            event.wait()

        try:
            pval = cit(i, j, list(key[2]))
            with stats_lock:
                ci_cache[key] = pval
            return pval
        finally:
            with stats_lock:
                ci_inflight.pop(key, None)
                event.set()

    def record_process_tests(n_tests):
        with stats_lock:
            ci_stats["total"] += int(n_tests)
            ci_stats[phase["name"]] += int(n_tests)
            if "ci_cache_misses" in ci_stats:
                ci_stats["ci_cache_misses"] += int(n_tests)

    def record_cache_hits(n_tests):
        if n_tests <= 0:
            return
        with stats_lock:
            ci_stats["total"] += int(n_tests)
            ci_stats[phase["name"]] += int(n_tests)
            if "ci_cache_hits" in ci_stats:
                ci_stats["ci_cache_hits"] += int(n_tests)

    counted_cit = counted_fisherz(
        scalar_cit, cit, ci_cache, ci_stats, phase, stats_lock
    )
    attach_counted_batch(counted_cit, cit, ci_stats, phase, ci_cache, stats_lock)
    cg.set_ind_test(cit if use_process_skeleton else counted_cit)
    if not k:
        k = p

    node_names = [node.name for node in cg.nodes]
    apply_bk_checkpoint(
        background_knowledge,
        cg.G.graph,
        node_names,
        phase=BKPhase.PRE_SEARCH,
        checkpoint="before_skeleton",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="pag",
    )
    protected_edges = ()
    if background_knowledge is not None:
        protected_edges = background_knowledge.required_pairs(
            node_names, phase=BKPhase.PRE_SEARCH
        )

    skeleton_t0 = time.perf_counter()
    skeleton_before = batch_execution_stats(cit)
    skeleton = SkeletonDiscovery(
        cg,
        alpha,
        n_jobs=skeleton_jobs_eff if native or skeleton_backend_eff == "threads" else 1,
        verbose=verbose,
        show_progress=show_progress,
        parallel_backend=skeleton_backend_requested,
        process_ci_info=process_ci_info if use_process_skeleton else None,
        protected_edges=protected_edges,
    )
    skeleton.run_fci()
    ci_stats["skeleton_backend"] = _observed_phase_backend(
        cit, skeleton_before, getattr(cg, "fci_skeleton_execution_stats", None)
    )
    ci_stats["time_skeleton_sec"] = time.perf_counter() - skeleton_t0
    if use_process_skeleton:
        skeleton_ci_tests = int(skeleton.ci_tests_performed)
        ci_stats["total"] += skeleton_ci_tests
        ci_stats["skeleton"] += skeleton_ci_tests
        if "ci_cache_misses" in ci_stats:
            ci_stats["ci_cache_misses"] += skeleton_ci_tests
        cg.set_ind_test(counted_cit)
    # Explicit o-o initialization right after skeleton.
    # cg.G.graph = to_circle_adjacency(cg.G.graph, MatrixEncoding.STANDARD)

    # print(graphDict)
    # pag = cg.G.graph
    # for i in range(pag.shape[0]):
    #     for j in range(pag.shape[1]):
    #         if pag[i][j]: pag[i][j] = CIRCLE

    # cg.G.graph = pag
    phase["name"] = "augment"
    phase_backends = {}
    augment_before = batch_execution_stats(cit)
    augment_t0 = time.perf_counter()
    AugmentGraph(
        cg,
        p,
        alpha,
        n_jobs=augment_jobs_eff,
        parallel_backend=augment_backend_requested,
        process_ci_info=process_ci_info if use_process_augment else None,
        record_process_tests=record_process_tests if use_process_augment else None,
        record_cache_hits=record_cache_hits,
        ci_cache=ci_cache,
        stats=ci_stats,
    )
    ci_stats["time_augment_sec"] = time.perf_counter() - augment_t0
    phase_backends["augment"] = _observed_phase_backend(cit, augment_before)
    dsep_pos = _dsep_search(cg, p)
    ci_stats["dsep_links_initial"] = len(dsep_pos)

    phase["name"] = "dsep"
    dsep_before = batch_execution_stats(cit)
    dsep_t0 = time.perf_counter()
    while len(dsep_pos) > 0:
        x, y = dsep_pos.popleft()
        ci_stats["dsep_links_scanned"] = int(ci_stats.get("dsep_links_scanned", 0)) + 1
        base_x = set(cg.neighbors(x)) - {y}
        base_y = set(cg.neighbors(y)) - {x}
        found_dsep, updated_dsep_pos = _dsep(
            cg,
            alpha,
            x,
            y,
            base_x,
            base_y,
            k,
            p,
            n_jobs=dsep_jobs_eff,
            parallel_backend=dsep_backend_requested,
            process_ci_info=process_ci_info if use_process_dsep else None,
            record_process_tests=record_process_tests if use_process_dsep else None,
            record_cache_hits=record_cache_hits,
            ci_cache=ci_cache,
            stats=ci_stats,
            protected_edges=protected_edges,
        )
        if found_dsep:
            dsep_pos = updated_dsep_pos
            ci_stats["dsep_links_after_update"] = len(dsep_pos)
    ci_stats["time_dsep_sec"] = time.perf_counter() - dsep_t0
    phase_backends["dsep"] = _observed_phase_backend(cit, dsep_before)

    pag = cg.G.graph.copy()

    for i in range(p):
        for j in range(p):
            if pag[i][j] == TAIL:
                pag[i][j] = PAG_TAIL
            elif pag[i][j] == ARROW:
                pag[i][j] = PAG_ARROW
            elif pag[i][j] == CIRCLE:
                pag[i][j] = PAG_CIRCLE

    orient_t0 = time.perf_counter()
    sepset_pair = _sepset_to_pair_dict(cg.sepset, p)
    pag = udag2pag(
        pag,
        sepset_pair,
        p,
        orientCollider=True,
        background_knowledge=background_knowledge,
        node_names=[node.name for node in cg.nodes],
        # CFCI and FCI+ leave the collider unoriented when the pair has no
        # recorded separating sets. Keep that behavior here.
        require_recorded_sepset=True,
    )
    # pag = udag2pag(pag, sepset_pair, p, orientCollider=True, rules=[False for _ in range(10)])

    # No conversion is needed because udag2pag and FCI+ both use
    # causal-learn's encoding.
    cg.G.graph = pag
    ci_stats["time_orient_sec"] = time.perf_counter() - orient_t0
    if ci_cache is not None:
        ci_stats["ci_cache_size"] = len(ci_cache)

    for phase_name in ("augment", "dsep"):
        process_tasks = int(ci_stats.get(f"{phase_name}_process_tasks", 0))
        thread_tasks = int(ci_stats.get(f"{phase_name}_thread_tasks", 0))
        process_tasks |= phase_backends[phase_name] in ("processes", "hybrid")
        thread_tasks |= phase_backends[phase_name] in ("threads", "hybrid")
        if process_tasks and thread_tasks:
            effective_backend = "hybrid"
        elif process_tasks:
            effective_backend = "processes"
        elif thread_tasks:
            effective_backend = "threads"
        else:
            effective_backend = "serial"
        ci_stats[f"{phase_name}_effective_backend"] = effective_backend
        ci_stats[f"{phase_name}_backend"] = effective_backend
    observed = {ci_stats[f"{name}_backend"] for name in ("augment", "dsep")} - {
        "serial"
    }
    ci_stats["ci_backend"] = (
        next(iter(observed))
        if len(observed) == 1
        else "hybrid"
        if observed
        else "serial"
    )

    thread_executor = getattr(cg, "_fci_plus_thread_executor", None)
    if thread_executor is not None:
        thread_executor.shutdown(wait=True)
    for attr in (
        "_fci_plus_thread_executor",
        "_fci_plus_process_min_tasks",
        "_fci_plus_parallel_stats",
        "_fci_plus_parallel_phase",
    ):
        if hasattr(cg, attr):
            delattr(cg, attr)
    ci_stats["time_total_sec"] = time.perf_counter() - total_t0
    cg.ci_stats = ci_stats
    return pag


# Keep historical star imports of the demo aliases explicit requests too.
if TYPE_CHECKING:
    from .fci_algo_v2 import fci_stable_v2
    from .rfci_algo import make_collider_data, rfci_stable

__all__ = [
    "ARROW",
    "CIRCLE",
    "CIT",
    "NULL",
    "PAG_ARROW",
    "PAG_CIRCLE",
    "PAG_TAIL",
    "TAIL",
    "AugmentGraph",
    "BKPhase",
    "BoundedCache",
    "CausalGraph",
    "Endpoint",
    "MatrixEncoding",
    "SkeletonDiscovery",
    "ThreadPoolExecutor",
    "apply_bk_checkpoint",
    "attach_counted_batch",
    "augment_native",
    "augmentation_inputs",
    "batch_execution_stats",
    "bounded_size",
    "ci_cache_for",
    "combinations",
    "conditioning_allowed",
    "configure_conditioning",
    "counted_fisherz",
    "dependent_endpoints",
    "deque",
    "effective_n_jobs",
    "evaluate_ci_batch",
    "fci_plus_stable",
    "fci_stable_v2",
    "fisher_z_batch_from_corr",
    "fisher_z_from_corr",
    "hierarchy_cache",
    "hierarchy_changed",
    "hierarchy_closure",
    "import_module",
    "islice",
    "make_collider_data",
    "math",
    "naa_ancestors",
    "np",
    "observed_phase_backend",
    "os",
    "pdsep_candidates",
    "process_map",
    "rfci_stable",
    "set_arrowheads",
    "supports_fisher_batch",
    "threading",
    "time",
    "udag2pag",
]
