import math
import threading
import time
from collections import OrderedDict
from contextlib import ExitStack
from heapq import merge
from itertools import combinations, islice

import numpy as np

from ...background_knowledge import BKPhase, MatrixEncoding, apply_bk_checkpoint
from ..graph_core import CIT, fisher_z_from_corr

# Import the shared test class from the neighboring module.
from ..graph_core import cache as _cache_exports
from ..graph_core._optional_imports import Parallel, delayed
from ..graph_core.cache import ci_cache_for
from ..graph_core.ci_batch import (
    PROCESS_POOL_OPTIONS,
    batch_execution_stats,
    evaluate_ci_batch,
    fisher_z_batch_from_corr,
    supports_fisher_batch,
)
from ..graph_core.conditioning import (
    bounded_size,
    configure_conditioning,
)
from ..graph_core.process_execution import plan_process_work, process_map
from ..graph_core.validation import effective_n_jobs
from . import pag_rules as _pag_rules_exports
from .ci_adapters import LegacyCITest, attach_counted_batch, attach_legacy_batch
from .compact_pdsep import CompactPDSEP, _weighted_pair_chunks
from .fci_search import (
    classify_neighbors,
    orient_colliders,
    record_triple,
    unshielded_triples,
)
from .pag_init import to_circle_adjacency
from .pag_rules import PAG_CIRCLE, qreach, udag2pag
from .pdsep_candidates import PossibleDsepRow, pdsep_conditions

# --- Helper functions ---
from .skeleton import SkeletonDiscovery

BoundedCache = _cache_exports.BoundedCache
powerset = _pag_rules_exports.powerset

MIN_TRIPLE_PROCESS_TESTS = 4096


def _ci_cache_key(i: int, j: int, S):
    """Canonical cache key for CI test symmetry in (i, j, S)."""
    a, b = (int(i), int(j)) if i <= j else (int(j), int(i))
    if S is None:
        cond = ()
    elif isinstance(S, (set, frozenset, list, tuple, np.ndarray)):
        cond = tuple(sorted(int(x) for x in S))
    else:
        cond = (int(S),)
    return a, b, cond


def _normalize_cond_set(S):
    if S is None:
        return ()
    if isinstance(S, (set, frozenset)):
        return tuple(sorted(int(v) for v in S))
    if isinstance(S, np.ndarray):
        return tuple(int(v) for v in S.tolist())
    if isinstance(S, (list, tuple)):
        return tuple(int(v) for v in S)
    return (int(S),)


def _chunk_sequence(items, n_chunks):
    n_chunks = max(1, min(int(n_chunks), len(items)))
    chunk_size = math.ceil(len(items) / n_chunks)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


# Fisher-Z lives in graph_core so every backend computes it identically;
# a local copy here would silently drift from CIT.fisher_z.
_fisher_z_from_corr = fisher_z_from_corr


def _fisher_z_pvals_chunk(corr_matrix, n_samples, x, y, cond_sets):
    return fisher_z_batch_from_corr(
        corr_matrix, n_samples, [(x, y, S) for S in cond_sets]
    )


def _parallel_indep_eval(
    indepTest,
    suffStat,
    x,
    y,
    cond_sets,
    n_jobs=1,
    parallel_backend="auto",
    process_ci_info=None,
):
    cond_sets = [_normalize_cond_set(S) for S in cond_sets]
    if not cond_sets:
        return []
    if supports_fisher_batch(indepTest):
        return evaluate_ci_batch(
            indepTest,
            [(x, y, S) for S in cond_sets],
            n_jobs=n_jobs,
            backend=parallel_backend,
        )
    if n_jobs in (None, 0, 1):
        return [indepTest(suffStat, x, y, list(S)) for S in cond_sets]
    if (
        parallel_backend == "processes"
        and process_ci_info is not None
        and process_ci_info.get("method") == "fisherz"
    ):
        chunks = _chunk_sequence(cond_sets, int(n_jobs))
        chunk_results = Parallel(
            n_jobs=int(n_jobs),
            **PROCESS_POOL_OPTIONS,
        )(
            delayed(_fisher_z_pvals_chunk)(
                process_ci_info["corr_matrix"],
                process_ci_info["n_samples"],
                x,
                y,
                chunk,
            )
            for chunk in chunks
        )
        return [pval for chunk in chunk_results for pval in chunk]
    # User callbacks retain call order and parent-process side effects.
    return [indepTest(suffStat, x, y, list(S)) for S in cond_sets]


def checkTriple(
    a,
    b,
    c,
    nbrsA,
    nbrsC,
    sepsetA,
    sepsetC,
    suffStat,
    alpha,
    indepTest,
    maj_rule=True,
    n_jobs=1,
):
    """Classify an unshielded triple by evaluating candidate separating sets."""

    def evaluate(conditions):
        return _parallel_indep_eval(
            indepTest, suffStat, a, c, conditions, n_jobs=n_jobs
        )

    return classify_neighbors(
        a, b, c, nbrsA, nbrsC, sepsetA, sepsetC, indepTest, alpha, maj_rule, evaluate
    )


def _check_triple_task(args):
    return checkTriple(*args, n_jobs=1)


_BUILTIN_CHECK_TRIPLE = checkTriple


def _triple_conditions(left, right, cap):
    """Enumerate the union of both subset spaces without materializing it."""
    for order in range(min(max(len(left), len(right)), cap) + 1):
        previous = None
        for nodes in merge(combinations(left, order), combinations(right, order)):
            if nodes != previous:
                yield nodes
                previous = nodes


def _triple_pair_worker(
    corr, n_samples, x, y, left, right, cap, alpha, known, cached, retain
):
    """Reduce all separators to sufficient evidence for every middle vertex."""
    result = {
        "x": x,
        "y": y,
        "requests": 0,
        "consumed": 0,
        "computed": 0,
        "outer_hits": 0,
        "cit_hits": 0,
        "updates": [],
        "error": None,
        "discarded_cit_insertions": 0,
        "discarded_outer_insertions": 0,
        "count": 0,
        "members": {},
        "chosen": None,
        "known_seen": set(),
    }
    updates = OrderedDict()
    candidates = iter(_triple_conditions(left, right, cap))
    while True:
        conditions = list(islice(candidates, 4096))
        if not conditions:
            break
        values, missing, positions = [None] * len(conditions), [], []
        for index, nodes in enumerate(conditions):
            hit = cached.get(nodes)
            if hit is None:
                missing.append((x, y, nodes))
                positions.append(index)
            else:
                values[index], origin = hit
                result["outer_hits" if origin == 0 else "cit_hits"] += 1
        computed = fisher_z_batch_from_corr(
            corr,
            n_samples,
            missing,
            n_jobs=1,
            return_exceptions=True,
            decision_alpha=alpha,
        )
        result["computed"] += len(missing)
        result["requests"] += len(conditions)
        for position, value in zip(positions, computed, strict=False):
            values[position] = value
        for nodes, value in zip(conditions, values, strict=False):
            result["consumed"] += 1
            if isinstance(value, BaseException):
                result["error"] = value
                result["updates"] = list(updates.items())
                return result
            hit = cached.get(nodes)
            if retain and (hit is None or hit[1] == 1):
                updates[nodes] = value
                if len(updates) > retain:
                    discarded, _ = updates.popitem(last=False)
                    result["discarded_outer_insertions"] += 1
                    if discarded not in cached:
                        result["discarded_cit_insertions"] += 1
            if value >= alpha:
                result["count"] += 1
                if result["chosen"] is None:
                    result["chosen"] = nodes
                for node in nodes:
                    result["members"][node] = result["members"].get(node, 0) + 1
                if nodes in known:
                    result["known_seen"].add(nodes)
    result["updates"] = list(updates.items())
    return result


def _triple_pair_chunk(corr, n_samples, descriptors):
    return [
        (index, _triple_pair_worker(corr, n_samples, *args))
        for index, args in descriptors
    ]


def _classify_triple_summary(b, result, sepset_a, sepset_c, majority, truncated):
    count, present = result["count"], result["members"].get(int(b), 0)
    known = {tuple(sorted(s)) for s in (sepset_a, sepset_c) if s is not None}
    added = known - result["known_seen"]
    count += len(added)
    present += sum(b in nodes for nodes in added)
    chosen = [nodes for nodes in [result["chosen"], *known] if nodes is not None]
    if not count:
        return 3, {"sepsetA": sepset_a, "sepsetC": sepset_c}
    status = 3
    if not truncated:
        if (majority and 2 * present < count) or (not majority and present == 0):
            status = 1
        elif (majority and 2 * present > count) or (not majority and present == count):
            status = 2
    selected = set(min(chosen, key=lambda s: (len(s), s)))
    return status, {
        "sepsetA": set(sepset_a) if sepset_a is not None else selected,
        "sepsetC": set(sepset_c) if sepset_c is not None else set(selected),
    }


def _grouped_triple_results(triple_args, indep_test, n_jobs, parallel_backend):
    """Share one pair's evidence across triples; keep custom callbacks outside."""
    context = getattr(indep_test, "_pdsep_batch_context", None)
    if (
        not supports_fisher_batch(indep_test)
        or context is None
        or checkTriple is not _BUILTIN_CHECK_TRIPLE
    ):
        return None
    publisher = CompactPDSEP(indep_test, n_jobs)
    publisher._refresh_indices()
    cit, outer, _stats, _phase, _lock = context
    retain = max(cit.cache_size, getattr(outer, "max_entries", 0))
    descriptors, indices, truncated = [], {}, {}
    for args in triple_args:
        a, _b, c, nbrs_a, nbrs_c, known_a, known_c, _suff, alpha, _test, _majority = (
            args
        )
        key = (int(a), int(c))
        if key in indices:
            continue
        index = indices[key] = len(indices)
        left = tuple(sorted(set(map(int, nbrs_a)) - {int(a), int(c)}))
        right = tuple(sorted(set(map(int, nbrs_c)) - {int(a), int(c)}))
        cap = int(bounded_size(indep_test, max(len(left), len(right))))
        truncated[index] = max(len(left), len(right)) > cap
        known = {tuple(sorted(s)) for s in (known_a, known_c) if s is not None}
        cached = {
            nodes: (value, 1)
            for nodes, value in publisher.cit_index.get(key, {}).items()
        }
        cached.update(
            {
                nodes: (value, 0)
                for nodes, value in publisher.outer_index.get(key, {}).items()
            }
        )
        common = len(set(left) & set(right))
        cost = sum(
            (math.comb(len(left), size) if size <= len(left) else 0)
            + (math.comb(len(right), size) if size <= len(right) else 0)
            - (math.comb(common, size) if size <= common else 0)
            for size in range(min(cap, max(len(left), len(right))) + 1)
        )
        descriptors.append(
            (
                cost,
                index,
                (int(a), int(c), left, right, cap, alpha, known, cached, retain),
            )
        )
    plan = plan_process_work(
        sum(item[0] for item in descriptors),
        n_jobs,
        family="conservative_triples",
        backend=parallel_backend,
        tasks=len(descriptors),
        min_work=MIN_TRIPLE_PROCESS_TESTS,
        stats=_stats,
    )
    jobs = plan.jobs
    use_processes = jobs > 1
    chunks = _weighted_pair_chunks(descriptors, jobs)
    parts = process_map(
        _triple_pair_chunk,
        ((cit.corr_matrix, cit.n_samples, chunk) for chunk in chunks),
        n_jobs=jobs,
        plan=plan,
        shared_arg_indices=(0,),
    )
    results = dict(item for part in parts for item in part)
    published, classified = set(), []
    for args in triple_args:
        a, b, c, _na, _nc, known_a, known_c, _suff, _alpha, _test, majority = args
        index = indices[(int(a), int(c))]
        result = results[index]
        if index not in published:
            publisher.publish(result)
            published.add(index)
        if result["error"] is not None:
            raise result["error"]
        classified.append(
            _classify_triple_summary(
                b, result, known_a, known_c, majority, truncated[index]
            )
        )
    return classified, "processes" if use_processes else "serial", len(descriptors)


def pc_cons_intern(
    graphDict,
    suffstat,
    alpha,
    indepTest,
    version_unf=(None, None),
    maj_rule=True,
    verbose=False,
    n_jobs=1,
    parallel_backend="auto",
):
    """Record triple classifications and separating sets in the skeleton's graph dictionary."""
    matrix = graphDict["sk"]
    if not np.any(matrix):
        return graphDict
    triples = list(unshielded_triples(matrix, callback_order=True))
    neighbors = [np.flatnonzero(column) for column in matrix.T]
    separators = graphDict["sepset"]
    requests = [
        (
            a,
            b,
            c,
            neighbors[a],
            neighbors[c],
            separators.get((a, c)),
            separators.get((c, a)),
            suffstat,
            alpha,
            indepTest,
            maj_rule,
        )
        for a, b, c in triples
    ]
    batches = _grouped_triple_results(requests, indepTest, n_jobs, parallel_backend)
    if batches is None:
        results, backend, tasks = (
            map(_check_triple_task, requests),
            "serial",
            len(requests),
        )
    else:
        results, backend, tasks = batches
    for triple, result in zip(triples, results, strict=False):
        record_triple(graphDict, triple, result, classify_all=True)
    graphDict["triple_stats"] = {
        "triples": len(triples),
        "pair_tasks": int(tasks),
        "backend": backend,
        "n_jobs": int(n_jobs or 1),
    }
    return graphDict


def _pdsep_fisherz_task_chunk(corr_matrix, n_samples, tasks):
    values = fisher_z_batch_from_corr(
        corr_matrix,
        n_samples,
        [(x, y, S) for _, _, x, y, S in tasks],
        return_exceptions=True,
    )
    return [
        (pair_idx, cond_idx, value)
        for (pair_idx, cond_idx, _, _, _), value in zip(tasks, values, strict=False)
    ]


def _pdsep_indep_task_chunk(indepTest, suffStat, tasks):
    return [
        (pair_idx, cond_idx, indepTest(suffStat, x, y, list(S)))
        for pair_idx, cond_idx, x, y, S in tasks
    ]


def _run_pdsep_ci_tasks(
    indepTest,
    suffStat,
    tasks,
    n_jobs,
    parallel_backend,
    process_ci_info,
    parallel_executor=None,
):
    if not tasks:
        return []
    if supports_fisher_batch(indepTest):
        values = evaluate_ci_batch(
            indepTest,
            [(x, y, S) for _, _, x, y, S in tasks],
            n_jobs=n_jobs,
            backend=parallel_backend,
            return_exceptions=True,
        )
        return [
            (pair_idx, cond_idx, value)
            for (pair_idx, cond_idx, _, _, _), value in zip(tasks, values, strict=False)
        ]

    if n_jobs in (None, 0, 1):
        return [
            (pair_idx, cond_idx, indepTest(suffStat, x, y, list(S)))
            for pair_idx, cond_idx, x, y, S in tasks
        ]

    n_jobs_eff = effective_n_jobs(n_jobs)
    chunks = _chunk_sequence(tasks, max(n_jobs_eff * 4, n_jobs_eff))

    if (
        parallel_backend == "processes"
        and process_ci_info is not None
        and process_ci_info.get("method") == "fisherz"
    ):
        executor = parallel_executor or Parallel(
            n_jobs=n_jobs_eff,
            **PROCESS_POOL_OPTIONS,
        )
        chunk_results = executor(
            delayed(_pdsep_fisherz_task_chunk)(
                process_ci_info["corr_matrix"],
                process_ci_info["n_samples"],
                chunk,
            )
            for chunk in chunks
        )
    else:
        chunk_results = [
            _pdsep_indep_task_chunk(indepTest, suffStat, chunk) for chunk in chunks
        ]

    return [item for chunk in chunk_results for item in chunk]


def pdsep(
    skel,
    suffStat,
    indepTest,
    p,
    sepSet,
    alpha,
    pMax,
    m_max=float("inf"),
    pdsep_max=float("inf"),
    unfVect=None,
    n_jobs=1,
    parallel_backend="auto",
    process_ci_info=None,
    process_min_tasks=256,
    protected_edges=(),
):
    """Search Possible-D-SEP sets to refine the skeleton and record separation evidence."""
    m_max = bounded_size(indepTest, m_max)
    G = skel["sk"].astype(int)
    # Background knowledge protects these pairs from Possible-D-SEP removal.
    protected_edges = {frozenset((int(x), int(y))) for x, y in protected_edges}
    max_ord_reached = 0
    # run_fci can return a TAIL/TAIL skeleton. Adjacency is independent of the endpoint mark.
    amat = np.where(G != 0, PAG_CIRCLE, 0)
    pdsep_pairs_scanned = 0
    pdsep_pairs_with_diffset = 0
    pdsep_ci_tests = 0
    task_build_time = 0.0
    ci_eval_time = 0.0
    apply_time = 0.0
    rounds = 0
    process_rounds = 0
    thread_rounds = 0
    process_ci_tests = 0
    thread_ci_tests = 0

    orient_colliders(amat, sepSet, unfVect, either=True)
    history = [[] for _ in range(p)]
    pair_states = []
    for x, marked_row in enumerate(amat):
        neighbors = np.flatnonzero(marked_row).tolist()
        row = PossibleDsepRow(x, qreach(x, amat), neighbors, protected_edges)
        history[x] = row.history_entry()
        pdsep_pairs_scanned += len(row.targets)
        if not row.outside:
            continue
        pdsep_pairs_with_diffset += len(row.targets)
        for y in row.targets:
            pool = row.pool(y)
            cap = int(min(len(pool), m_max))
            if cap:
                pair_states.append(
                    {
                        "x": x,
                        "y": y,
                        "tf": pool,
                        "diff_set": row.outside,
                        "adj_x": row.neighbors,
                        "max_ord": cap,
                        "active": True,
                    }
                )

    global_max_ord = max((pair["max_ord"] for pair in pair_states), default=0)

    ci_consumed = 0
    tasks_built = 0
    before_numeric = batch_execution_stats(indepTest)
    # Explicit legacy numerical descriptors authorize direct workers. Counted
    # wrappers retain their callback contract, including custom CIT overrides.
    legacy_process = (
        effective_n_jobs(n_jobs) > 1
        and parallel_backend == "processes"
        and process_ci_info is not None
        and process_ci_info.get("method") == "fisherz"
        and not supports_fisher_batch(indepTest)
        and getattr(indepTest, "_batch_cit", None) is None
        and getattr(indepTest, "_pdsep_batch_context", None) is None
    )
    legacy_executor = None
    # A level retains the initial candidate snapshots and commits in pair order.
    # Coarse workers enumerate only one depth, preserving the original barrier.
    with (
        ExitStack() as executors,
        CompactPDSEP(
            indepTest,
            n_jobs if parallel_backend in ("auto", "processes") else 1,
            backend=parallel_backend,
        ) as compact,
    ):
        for ord in range(1, global_max_ord + 1):
            rounds += 1
            build_t0 = time.perf_counter()
            eligible = []
            for pair_idx, pair in enumerate(pair_states):
                if not pair["active"] or ord > pair["max_ord"]:
                    continue
                x, y = pair["x"], pair["y"]
                if amat[x, y] == 0 and amat[y, x] == 0:
                    pair["active"] = False
                    continue
                eligible.append((pair_idx, pair))
            task_build_time += time.perf_counter() - build_t0
            if not eligible:
                continue

            level_t0 = ci_t0 = time.perf_counter()
            coarse = compact.evaluate_level(
                eligible, ord, alpha, pMax, min_tasks=int(process_min_tasks)
            )
            ci_eval_time += time.perf_counter() - ci_t0
            if coarse is not None:
                process_rounds += int(compact.last_jobs > 1)
                for pair_idx, pair in eligible:
                    result = coarse.get(pair_idx)
                    if result is None:
                        continue
                    x, y = pair["x"], pair["y"]
                    skipped = amat[x, y] == 0 and amat[y, x] == 0
                    # Workers may have searched the opposite direction of a pair
                    # removed earlier at commit. Its result (including errors) is
                    # unconsumed speculation, but its real work is still counted.
                    published = dict(result, consumed=0) if skipped else result
                    compact.publish(published)
                    completed = result["requests"]
                    pdsep_ci_tests += completed
                    process_ci_tests += (
                        result["computed"] if compact.last_jobs > 1 else 0
                    )
                    tasks_built += 1
                    if skipped:
                        pair["active"] = False
                        continue
                    ci_consumed += result["consumed"]
                    if result["error"] is not None:
                        raise result["error"]
                    apply_t0 = time.perf_counter()
                    pMax[x, y] = result["pmax"]
                    if result["separator"] is not None:
                        amat[x, y] = amat[y, x] = 0
                        sepSet[x, y] = sepSet[y, x] = set(result["separator"])
                        pair["active"] = False
                    apply_time += time.perf_counter() - apply_t0
            else:
                # Conditions are generated in bounded windows instead of one
                # combinatorial list. Generic callbacks remain scalar and lazy.
                # Numerical exceptions become values until their query is consumed.
                for pair_idx, pair in eligible:
                    x, y = pair["x"], pair["y"]
                    if amat[x, y] == 0 and amat[y, x] == 0:
                        pair["active"] = False
                        continue
                    candidates = pdsep_conditions(pair["tf"], pair["diff_set"], ord)
                    builtin = supports_fisher_batch(indepTest)
                    window = 32 if builtin or legacy_process else 1
                    while pair["active"]:
                        build_t0 = time.perf_counter()
                        conditions = list(islice(candidates, window))
                        if not conditions:
                            task_build_time += time.perf_counter() - build_t0
                            break
                        tasks = [
                            (pair_idx, index, x, y, nodes)
                            for index, nodes in enumerate(conditions)
                        ]
                        tasks_built += len(tasks)
                        task_build_time += time.perf_counter() - build_t0
                        ci_t0 = time.perf_counter()
                        direct_process = legacy_process and len(tasks) >= int(
                            process_min_tasks
                        )
                        if direct_process and legacy_executor is None:
                            legacy_executor = executors.enter_context(
                                Parallel(
                                    n_jobs=effective_n_jobs(n_jobs),
                                    **PROCESS_POOL_OPTIONS,
                                )
                            )
                        values = _run_pdsep_ci_tasks(
                            indepTest,
                            suffStat,
                            tasks,
                            n_jobs=n_jobs if builtin or direct_process else 1,
                            parallel_backend=parallel_backend,
                            process_ci_info=process_ci_info,
                            parallel_executor=legacy_executor
                            if direct_process
                            else None,
                        )
                        if direct_process:
                            process_rounds += 1
                            process_ci_tests += len(values)
                        ci_eval_time += time.perf_counter() - ci_t0
                        pdsep_ci_tests += len(values)
                        apply_t0 = time.perf_counter()
                        for _pair_idx, cond_idx, value in values:
                            ci_consumed += 1
                            if isinstance(value, BaseException):
                                raise value
                            pMax[x, y] = max(pMax[x, y], value)
                            if value >= alpha:
                                amat[x, y] = amat[y, x] = 0
                                sepSet[x, y] = sepSet[y, x] = set(conditions[cond_idx])
                                pair["active"] = False
                                break
                        apply_time += time.perf_counter() - apply_t0
                        window = (
                            min(2048, window * 4) if builtin or legacy_process else 1
                        )
            compact.observe_serial(time.perf_counter() - level_t0)
            max_ord_reached = ord
            if not any(
                pair["active"] and ord < pair["max_ord"] for pair in pair_states
            ):
                break
        compact_stats = dict(compact.stats)

    after_numeric = batch_execution_stats(indepTest)
    numeric_batches = {
        backend: after_numeric.get("batches", {}).get(backend, 0)
        - before_numeric.get("batches", {}).get(backend, 0)
        for backend in ("serial", "threads", "processes")
    }
    numeric_queries = {
        backend: after_numeric.get("queries_by_backend", {}).get(backend, 0)
        - before_numeric.get("queries_by_backend", {}).get(backend, 0)
        for backend in ("serial", "threads", "processes")
    }
    process_ci_tests += numeric_queries["processes"]
    thread_ci_tests += numeric_queries["threads"]
    actual = {backend for backend, count in numeric_batches.items() if count}
    if process_rounds:
        actual.add("processes")
    if compact_stats.get("serial_rows"):
        actual.add("serial")
    if not supports_fisher_batch(indepTest) and pdsep_ci_tests > process_ci_tests:
        actual.add("serial")
    effective_backend = (
        next(iter(actual)) if len(actual) == 1 else "hybrid" if actual else "serial"
    )
    fallback_reason = ""
    if effective_backend == "serial":
        if not pdsep_ci_tests:
            fallback_reason = "no_work"
        elif effective_n_jobs(n_jobs) == 1:
            fallback_reason = "single_worker"
        elif not supports_fisher_batch(indepTest):
            fallback_reason = "custom_ci_requires_ordered_scalar"
        elif any(
            plan.get("selected_jobs") == 1
            and plan.get("reason") == "predicted_serial_faster"
            for plan in compact_stats.get("process_policy", ())
        ):
            fallback_reason = "adaptive_cost_model"
        elif compact_stats.get("serial_rows"):
            fallback_reason = "serial_native_kernel"
        elif not sum(numeric_queries.values()):
            fallback_reason = "cache_hits_only"
        else:
            fallback_reason = "below_parallel_work_threshold"
    # These counters describe real dispatch, rather than a requested executor.
    thread_rounds = numeric_batches["threads"]
    process_rounds += numeric_batches["processes"]
    G = np.where(amat != 0, PAG_CIRCLE, 0)

    return {
        "G": G,
        "sepset": sepSet,
        "pMax": pMax,
        "allPdsep": history,
        "max_ord": max_ord_reached,
        "pdsep_stats": {
            "pairs_scanned": int(pdsep_pairs_scanned),
            "pairs_with_diffset": int(pdsep_pairs_with_diffset),
            "ci_tests": int(pdsep_ci_tests),
            "backend": effective_backend,
            "requested_backend": parallel_backend,
            "actual_backend": effective_backend,
            "fallback_reason": fallback_reason,
            "numeric_batches": numeric_batches,
            "numeric_queries": numeric_queries,
            "ci_consumed": int(ci_consumed),
            "ci_speculative": int(pdsep_ci_tests - ci_consumed),
            **compact_stats,
            "rounds": int(rounds),
            "process_rounds": int(process_rounds),
            "thread_rounds": int(thread_rounds),
            "process_ci_tests": int(process_ci_tests),
            "thread_ci_tests": int(thread_ci_tests),
            "process_min_tasks": int(process_min_tasks),
            "tasks_built": int(tasks_built),
            "task_build_time": float(task_build_time),
            "ci_eval_time": float(ci_eval_time),
            "apply_time": float(apply_time),
        },
    }


def fci_stable(
    cg,
    data,
    alpha=0.05,
    indep_test="fisherz",
    n_jobs=1,
    verbose=False,
    use_ci_cache=True,
    skeleton_backend=None,
    pdsep_backend=None,
    background_knowledge=None,
    show_progress: bool = False,
    max_cond_set_size=None,
):
    """Run stable FCI with cached and batched CI tests and return its endpoint matrix."""
    # SkeletonDiscovery uses cg.ci_test(...), so CI test must be set before run_fci().
    configure_conditioning(cg, max_cond_set_size)
    total_t0 = time.perf_counter()
    n_jobs_eff = effective_n_jobs(n_jobs)
    skeleton_backend_eff = skeleton_backend or "auto"
    pdsep_backend_eff = pdsep_backend or "auto"
    for backend in (skeleton_backend_eff, pdsep_backend_eff):
        if backend not in ("auto", "threads", "processes"):
            msg = "FCI backend must be 'auto', 'threads', or 'processes'"
            raise ValueError(msg)
    use_process_skeleton = (
        n_jobs_eff != 1
        and indep_test == "fisherz"
        and skeleton_backend_eff in ("auto", "processes")
    )
    use_process_pdsep = (
        n_jobs_eff != 1
        and indep_test == "fisherz"
        and pdsep_backend_eff in ("auto", "processes")
    )
    cit = (
        cg.test
        if isinstance(cg.test, CIT)
        and cg.test.method == indep_test
        and np.array_equal(cg.test.data, data)
        else CIT(data, method=indep_test)
    )
    cit.max_cond_set_size = max_cond_set_size
    # A user override of CIT remains the source of p-values on every backend.
    # Raw Fisher workers are valid only for the unchanged built-in test.
    native_fisher = supports_fisher_batch(cit)
    use_process_skeleton = use_process_skeleton and native_fisher
    use_process_pdsep = use_process_pdsep and native_fisher
    process_ci_info = (
        {
            "method": indep_test,
            "corr_matrix": cit.corr_matrix,
            "n_samples": cit.n_samples,
        }
        if (use_process_skeleton or use_process_pdsep)
        else None
    )
    ci_stats = {
        "total": 0,
        "skeleton": 0,
        "pdsep": 0,
        "skeleton_backend": "serial",
        "skeleton_requested_backend": skeleton_backend_eff,
        "pdsep_backend": "serial",
        "pdsep_requested_backend": pdsep_backend_eff,
    }
    if use_ci_cache:
        ci_stats["ci_cache_hits"] = 0
        ci_stats["ci_cache_misses"] = 0
    phase = {"name": "skeleton"}
    ci_cache = ci_cache_for(cit) if use_ci_cache else None
    stats_lock = threading.Lock()

    def counted_cit(i, j, S):
        with stats_lock:
            ci_stats["total"] += 1
            ci_stats[phase["name"]] += 1
        if ci_cache is None:
            return cit(i, j, S)

        key = _ci_cache_key(i, j, S)
        with stats_lock:
            cached = ci_cache.get(key)
            if cached is not None:
                ci_stats["ci_cache_hits"] += 1
                return cached
            ci_stats["ci_cache_misses"] += 1

        pval = cit(i, j, list(key[2]))
        with stats_lock:
            ci_cache[key] = pval
        return pval

    attach_counted_batch(counted_cit, cit, ci_stats, phase, ci_cache, stats_lock)
    cg.set_ind_test(cit if use_process_skeleton else counted_cit)

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

    # Step 1: Skeleton discovery

    skeleton_t0 = time.perf_counter()
    skeleton = SkeletonDiscovery(
        cg,
        alpha,
        n_jobs=n_jobs_eff if native_fisher else 1,
        verbose=verbose,
        show_progress=show_progress,
        parallel_backend=skeleton_backend_eff if native_fisher else "auto",
        process_ci_info=process_ci_info if use_process_skeleton else None,
        protected_edges=protected_edges,
    )
    skeleton_numeric_before = batch_execution_stats(cit)
    graphDict = skeleton.run_fci()
    skeleton_numeric_after = batch_execution_stats(cit)
    skeleton_actual = {
        backend
        for backend, count in skeleton_numeric_after.get("batches", {}).items()
        if count > skeleton_numeric_before.get("batches", {}).get(backend, 0)
    }
    if not supports_fisher_batch(cg.test) and skeleton.ci_tests_performed:
        skeleton_actual.add(
            "processes"
            if use_process_skeleton
            else "threads"
            if skeleton.n_jobs > 1
            else "serial"
        )
    skeleton_actual.update(
        name
        for name, count in getattr(cg, "fci_skeleton_execution_stats", {})
        .get("coarse_batches", {})
        .items()
        if count
    )
    ci_stats["skeleton_backend"] = (
        next(iter(skeleton_actual))
        if len(skeleton_actual) == 1
        else "hybrid"
        if skeleton_actual
        else "serial"
    )
    ci_stats["time_skeleton_sec"] = time.perf_counter() - skeleton_t0
    if use_process_skeleton:
        skeleton_ci_tests = int(skeleton.ci_tests_performed)
        ci_stats["total"] += skeleton_ci_tests
        ci_stats["skeleton"] += skeleton_ci_tests
        if "ci_cache_misses" in ci_stats:
            ci_stats["ci_cache_misses"] += skeleton_ci_tests
        cg.set_ind_test(counted_cit)
    # Explicitly represent post-skeleton graph as o-o for FCI phases.
    graphDict["sk"] = to_circle_adjacency(graphDict["sk"], MatrixEncoding.STANDARD)

    _n, p = data.shape

    # Adapt the test to the (suffStat, x, y, S) signature that FCI expects.
    test_func_wrapper = LegacyCITest(counted_cit, max_cond_set_size)

    attach_legacy_batch(test_func_wrapper, counted_cit)
    # CIT holds the data. The FCI test adapter ignores suffStat.
    suffStat = None

    # Classical FCI: do not apply conservative triple classification here.

    phase["name"] = "pdsep"
    pdsep_t0 = time.perf_counter()
    pdsep_res = pdsep(
        graphDict,
        suffStat,
        test_func_wrapper,
        p,
        graphDict["sepset"],
        alpha,
        graphDict["pMax"],
        unfVect=graphDict["unfTriples"],
        n_jobs=n_jobs_eff,
        parallel_backend=pdsep_backend_eff,
        process_ci_info=process_ci_info if use_process_pdsep else None,
        protected_edges=protected_edges,
    )
    ci_stats["time_pdsep_sec"] = time.perf_counter() - pdsep_t0

    rule_stats = {}
    orient_t0 = time.perf_counter()
    pag = udag2pag(
        pdsep_res["G"],
        pdsep_res["sepset"],
        p,
        unfVect=graphDict["unfTriples"],
        rule_stats=rule_stats,
        background_knowledge=background_knowledge,
        node_names=node_names,
    )
    ci_stats["time_orient_sec"] = time.perf_counter() - orient_t0
    pdsep_stats = pdsep_res.get("pdsep_stats", {})
    ci_stats.setdefault("process_policy", []).extend(
        pdsep_stats.get("process_policy", ())
    )
    del ci_stats["process_policy"][:-64]
    ci_stats["pdsep_pairs_scanned"] = int(pdsep_stats.get("pairs_scanned", 0))
    ci_stats["pdsep_pairs_with_diffset"] = int(pdsep_stats.get("pairs_with_diffset", 0))
    ci_stats["pdsep_ci_tests"] = int(pdsep_stats.get("ci_tests", 0))
    ci_stats["pdsep_backend"] = pdsep_stats.get("backend", ci_stats["pdsep_backend"])
    ci_stats["pdsep_rounds"] = int(pdsep_stats.get("rounds", 0))
    ci_stats["pdsep_actual_backend"] = pdsep_stats.get("actual_backend", "serial")
    ci_stats["pdsep_fallback_reason"] = pdsep_stats.get("fallback_reason", "")
    for key in (
        "ci_consumed",
        "ci_speculative",
        "process_rows",
        "process_pairs",
        "worker_ci_computations",
        "worker_cache_hits",
        "worker_speculative_tests",
        "process_rounds",
        "thread_rounds",
        "process_ci_tests",
        "thread_ci_tests",
    ):
        ci_stats[f"pdsep_{key}"] = int(pdsep_stats.get(key, 0))
    ci_stats["pdsep_tasks_built"] = int(pdsep_stats.get("tasks_built", 0))
    ci_stats["time_pdsep_task_build_sec"] = float(
        pdsep_stats.get("task_build_time", 0.0)
    )
    ci_stats["time_pdsep_ci_eval_sec"] = float(pdsep_stats.get("ci_eval_time", 0.0))
    ci_stats["time_pdsep_apply_sec"] = float(pdsep_stats.get("apply_time", 0.0))
    if use_process_pdsep and not supports_fisher_batch(test_func_wrapper):
        # Built-in batches already pass through counted_cit in the parent.
        # Only legacy direct numerical workers bypass that accounting.
        direct_tests = int(pdsep_stats.get("process_ci_tests", 0))
        ci_stats["total"] += direct_tests
        ci_stats["pdsep"] += direct_tests
        if "ci_cache_misses" in ci_stats:
            ci_stats["ci_cache_misses"] += direct_tests
    for k, v in rule_stats.items():
        ci_stats[f"orient_{k.lower()}_changes"] = int(v)
    if ci_cache is not None:
        ci_stats["ci_cache_size"] = len(ci_cache)
    ci_stats["time_total_sec"] = time.perf_counter() - total_t0
    cg.ci_stats = ci_stats

    return pag
