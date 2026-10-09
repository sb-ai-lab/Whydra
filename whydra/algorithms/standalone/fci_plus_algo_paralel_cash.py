from itertools import combinations
import numpy as np
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from collections import deque

from ..graph_core import CIT, CausalGraph, fisher_z_from_corr
from ..graph_core.endpoints import Endpoint
from ...background_knowledge import BKPhase, MatrixEncoding, apply_bk_checkpoint

from .skeleton import SkeletonDiscovery

#импорт функции для тестовой проверки работы кода
from .rfci_algo import make_collider_data
from .pag_rules import PAG_ARROW, PAG_CIRCLE, PAG_TAIL, udag2pag


# ============================================================
#  Локальные копии вспомогательных функций FCI.
#  Перенесены из прежнего fci_algo.py.
#  Держим их здесь, чтобы файл не зависел от fci_algo.py:
#  там лежит эталонная реализация FCI из ветки dev, без
#  background_knowledge / protected_edges / rule_stats.
# ============================================================

TAIL = Endpoint.TAIL.value     # -1
ARROW = Endpoint.ARROW.value   # 1
CIRCLE = Endpoint.CIRCLE.value # 2
NULL = Endpoint.NULL.value    # 0


def _ci_cache_key(i: int, j: int, S):
    a, b = (int(i), int(j)) if i <= j else (int(j), int(i))
    if S is None:
        cond = ()
    elif isinstance(S, (set, frozenset)):
        cond = tuple(sorted(int(x) for x in S))
    elif isinstance(S, (list, tuple, np.ndarray)):
        cond = tuple(sorted(int(x) for x in S))
    else:
        cond = (int(S),)
    return a, b, cond


def _resolve_fci_plus_backend(backend, n_vars, process_min_vars):
    requested = "auto" if backend in (None, "auto") else str(backend)
    if requested == "auto":
        effective = "processes" if int(n_vars) > int(process_min_vars) else "threads"
    elif requested in {"threads", "processes"}:
        effective = requested
    else:
        raise ValueError("FCI+ backend must be 'auto', 'threads', or 'processes'")
    return requested, effective


def _sepset_vars(seps):
    vars_set = set()
    for x in seps or []:
        if isinstance(x, tuple):
            vars_set.update(x)
        elif isinstance(x, (list, set, np.ndarray)):
            vars_set.update(x)
        else:
            vars_set.add(x)
    return vars_set


def _has_sepset(seps):
    return seps is not None


def _sepset_to_pair_dict(sepset, p: int):
    """
    Приводит cg.sepset к виду, который ждёт udag2pag: {(x, z): множество узлов}.

    ``cg.sepset[i][j]`` хранит СПИСОК всех разделяющих множеств, найденных для
    пары — [(1,2,3), (4,5), ...]. Раньше здесь бралось их объединение, и это
    ломало ориентацию коллайдеров: правило R0 проверяет ``y not in sepset[(x,z)]``,
    а в объединение y попадал, если встретился хотя бы в одном множестве.
    В результате коллайдеры не ориентировались и на их месте оставались кружки.

    Берём одно множество — наименьшее по размеру. Для правила R0 годится любое
    корректное разделяющее множество, а наименьшее совпадает с тем, что нашла
    фаза скелета первой (поиск идёт по возрастанию глубины), и не зависит от
    порядка вставки.
    """
    out = {}
    for i in range(p):
        for j in range(p):
            s = sepset[i][j]
            if not s:
                out[(i, j)] = set()
                continue
            candidates = [
                set(S) if hasattr(S, "__iter__") else {int(S)}
                for S in s
            ]
            out[(i, j)] = min(candidates, key=len) if candidates else set()
    return out


def _edge_exists(mat, i, j):
    return mat[i][j] != NULL or mat[j][i] != NULL


def _is_bidirected(mat, i, j):
    return mat[i][j] == ARROW and mat[j][i] == ARROW


def _naa_ancestors(x, mat):
    p = mat.shape[0]
    queue = deque([int(x)])
    seen = {int(x)}
    out = []

    while queue:
        t = queue.popleft()
        for j in range(p):
            if j in seen:
                continue
            if t == x:
                legal = mat[j][t] == ARROW and mat[t][j] not in (NULL, ARROW)
            else:
                legal = mat[t][j] not in (NULL, ARROW)
            if legal:
                seen.add(j)
                queue.append(j)
                out.append(j)

    return set(out)


def _get_pdseps(mat):
    p = mat.shape[0]
    naaa = [None for _ in range(p)]
    links = []

    for i in range(1, p):
        naaa_i = _naa_ancestors(i, mat)
        naaa[i] = naaa_i
        for j in range(i):
            if not _is_bidirected(mat, i, j):
                continue

            if naaa[j] is None:
                naaa[j] = _naa_ancestors(j, mat)

            u_nodes = {
                k for k in range(p)
                if k not in (i, j) and _is_bidirected(mat, i, k)
            }
            v_nodes = {
                k for k in range(p)
                if k not in (i, j) and _is_bidirected(mat, j, k)
            }
            # TODO(fci-plus-pdsep): этот клон содержит тот же известный дефект,
            # что и последовательный fci_plus_algo._get_pdseps. После
            # AugmentGraph большинство рёбер двунаправленные, поэтому фильтр
            # Lemma 4 обнуляет кандидатов и Possible-D-SEP фактически не
            # запускается. На 10 синтетических графах (p=8..13, n=600,
            # fisherz, alpha=0.05): 65 кандидатов до фильтра и 0 после него;
            # очередь _dsep_search всегда пуста.
            #
            # Простое удаление фильтра не является исправлением: сверка с
            # causal-learn не улучшилась и на части seeds стала хуже. Нужна
            # корректная дефиниция NAA-предка для аугментированного графа из
            # Claassen et al.; до этого сохраняем текущее поведение.
            u_nodes &= naaa[j]   # U (bidirected with i) must have NAA path to j (Lemma 4)
            v_nodes &= naaa_i    # V (bidirected with j) must have NAA path to i (Lemma 4)

            found = any(
                u != v and not _edge_exists(mat, u, v)
                for u in u_nodes
                for v in v_nodes
            )
            if found:
                links.append((i, j))

    return deque(links)


def _chunk_sequence(items, n_chunks):
    if not items:
        return []
    n_chunks = max(1, min(int(n_chunks), len(items)))
    chunk_size = math.ceil(len(items) / n_chunks)
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


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
    stats[f"{prefix}_{backend}_batches"] = int(
        stats.get(f"{prefix}_{backend}_batches", 0)
    ) + 1
    stats[f"{prefix}_{backend}_tasks"] = int(
        stats.get(f"{prefix}_{backend}_tasks", 0)
    ) + int(n_tasks)


def _keyed_ci_thread_chunk(cg, tasks):
    return [
        (key, cg.ci_test(a, b, list(cond)))
        for key, a, b, cond in tasks
    ]


def _ci_thread_chunk(cg, tasks):
    return [
        (x, y, S, cg.ci_test(x, y, list(S)))
        for x, y, S in tasks
    ]


# Fisher-Z lives in graph_core so every backend computes it identically;
# a local copy here would silently drift from CIT.fisher_z.
_fisher_z_from_corr = fisher_z_from_corr


def _fisher_z_task_chunk(corr_matrix, n_samples, tasks):
    return [
        (x, y, S, _fisher_z_from_corr(corr_matrix, n_samples, x, y, S))
        for x, y, S in tasks
    ]


def _fisher_z_keyed_task_chunk(corr_matrix, n_samples, tasks):
    return [
        (key, _fisher_z_from_corr(corr_matrix, n_samples, x, y, S))
        for key, x, y, S in tasks
    ]


def _augment_fisher_z_chunk(corr_matrix, n_samples, alpha, tasks):
    pvals = []
    arrow_updates = []
    for key, x, y, S, metas in tasks:
        pval = _fisher_z_from_corr(corr_matrix, n_samples, x, y, S)
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

    for (x, y, S), (del_nodes, adj) in zip(tasks, task_meta):
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

    miss_tasks = [
        (*item["task"], item["metas"])
        for item in missing_by_key.values()
    ]
    chunks = _chunk_sequence(miss_tasks, max(n_jobs_eff * 4, n_jobs_eff))
    chunk_results = Parallel(
        n_jobs=n_jobs_eff,
        backend="loky",
        max_nbytes="10K",
    )(
        delayed(_augment_fisher_z_chunk)(
            process_ci_info["corr_matrix"],
            process_ci_info["n_samples"],
            alpha,
            chunk,
        )
        for chunk in chunks
    ) if chunks else []

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
    parallel_backend="threads",
    process_ci_info=None,
    record_process_tests=None,
    record_cache_hits=None,
    ci_cache=None,
):
    tasks = list(tasks)
    tasks = [(int(x), int(y), tuple(int(v) for v in S)) for x, y, S in tasks]
    if not tasks:
        return []

    n_jobs_eff = 1 if n_jobs in (None, 0, 1) else int(n_jobs)
    process_min_tasks = int(getattr(cg, "_fci_plus_process_min_tasks", 256))
    if ci_cache is not None:
        results = [None] * len(tasks)
        missing_by_key = {}
        for idx, (x, y, S) in enumerate(tasks):
            key = _ci_cache_key(x, y, S)
            cached = ci_cache.get(key)
            if cached is not None:
                results[idx] = (x, y, list(S), cached)
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
        if use_process_batch:
            chunks = _chunk_sequence(miss_tasks, max(n_jobs_eff * 4, n_jobs_eff))
            chunk_results = Parallel(
                n_jobs=n_jobs_eff,
                backend="loky",
                max_nbytes="10K",
            )(
                delayed(_fisher_z_keyed_task_chunk)(
                    process_ci_info["corr_matrix"],
                    process_ci_info["n_samples"],
                    chunk,
                )
                for chunk in chunks
            ) if chunks else []
            computed = [item for chunk in chunk_results for item in chunk]
            _record_parallel_batch(cg, "process", len(miss_tasks))
            if record_process_tests is not None:
                record_process_tests(len(miss_tasks))
        elif n_jobs_eff == 1:
            computed = [
                (key, cg.ci_test(a, b, list(cond)))
                for key, a, b, cond in miss_tasks
            ]
        else:
            executor = _get_thread_executor(cg, n_jobs_eff)
            chunks = _chunk_sequence(
                miss_tasks,
                max(_thread_worker_count(n_jobs_eff) * 2, 1),
            )
            futures = [
                executor.submit(_keyed_ci_thread_chunk, cg, chunk)
                for chunk in chunks
            ]
            computed = [
                item
                for future in futures
                for item in future.result()
            ]
            _record_parallel_batch(cg, "thread", len(miss_tasks))

        for key, pval in computed:
            ci_cache[key] = pval
            for idx in missing_by_key[key]["positions"]:
                x, y, S = tasks[idx]
                results[idx] = (x, y, list(S), pval)

        if record_cache_hits is not None:
            record_cache_hits(len(tasks) - len(miss_tasks))
        return results

    if n_jobs in (None, 0, 1):
        return [(x, y, list(S), cg.ci_test(x, y, list(S))) for x, y, S in tasks]

    use_process_batch = (
        parallel_backend == "processes"
        and process_ci_info is not None
        and process_ci_info.get("method") == "fisherz"
        and len(tasks) >= process_min_tasks
    )
    if use_process_batch:
        chunks = _chunk_sequence(tasks, max(n_jobs_eff * 4, n_jobs_eff))
        chunk_results = Parallel(
            n_jobs=n_jobs_eff,
            backend="loky",
            max_nbytes="10K",
        )(
            delayed(_fisher_z_task_chunk)(
                process_ci_info["corr_matrix"],
                process_ci_info["n_samples"],
                chunk,
            )
            for chunk in chunks
        )
        results = [item for chunk in chunk_results for item in chunk]
        _record_parallel_batch(cg, "process", len(results))
        if record_process_tests is not None:
            record_process_tests(len(results))
        return [(x, y, list(S), pval) for x, y, S, pval in results]

    executor = _get_thread_executor(cg, n_jobs_eff)
    chunks = _chunk_sequence(
        tasks,
        max(_thread_worker_count(n_jobs_eff) * 2, 1),
    )
    futures = [
        executor.submit(_ci_thread_chunk, cg, chunk)
        for chunk in chunks
    ]
    computed = [
        item
        for future in futures
        for item in future.result()
    ]
    _record_parallel_batch(cg, "thread", len(tasks))
    return [(x, y, list(S), pval) for x, y, S, pval in computed]



def AugmentGraph(
    cg: CausalGraph,
    p: int,
    alpha: float,
    n_jobs: int = 1,
    parallel_backend: str = "threads",
    process_ci_info=None,
    record_process_tests=None,
    record_cache_hits=None,
    ci_cache=None,
    stats=None,
):
    pag = cg.G.graph.copy()
    sepset = cg.sepset
    tasks = []
    task_meta = []

    for i in range(p):
        for j in range(i + 1, p):
            sep_entry = sepset[i][j]
            if not _has_sepset(sep_entry):
                continue
            sep = _sepset_vars(sep_entry)
            adjacent = set(np.where(pag[i, :] != 0)[0]) | set(np.where(pag[j, :] != 0)[0])
            for sep_node in sep:
                adjacent |= set(np.where(pag[sep_node, :] != 0)[0])

            del_nodes = {i, j} | sep
            adjacent -= del_nodes

            for adj in sorted(adjacent):
                S = tuple(sorted(sep | {adj}))
                tasks.append((i, j, S))
                task_meta.append((tuple(sorted(del_nodes)), adj))

    if stats is not None:
        stats["augment_tasks"] = int(stats.get("augment_tasks", 0)) + len(tasks)

    arrowheads_added = 0
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
        for node, adj in sorted(set(arrow_updates)):
            if pag[node][adj] != NULL and pag[node][adj] != ARROW:
                pag[node][adj] = ARROW
                arrowheads_added += 1
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
        for (_, _, _S, p_value), (del_nodes, adj) in zip(pvals, task_meta):
            if p_value < alpha:
                for node in del_nodes:
                    if pag[node][adj] != NULL and pag[node][adj] != ARROW:
                        pag[node][adj] = ARROW
                        arrowheads_added += 1

    if stats is not None:
        stats["augment_arrowheads_added"] = int(stats.get("augment_arrowheads_added", 0)) + arrowheads_added

    cg.G.graph = pag
    return pag

def _dsep_search(cg: CausalGraph, p: int) -> deque:
    return _get_pdseps(cg.G.graph.copy())


def _get_hie(cg, S):
    H = set(S)
    old_H = None

    while old_H != H:
        old_H = H.copy()
        H_list = list(H)
        for a in H_list:
            for b in H_list:
                if a == b:
                    continue
                sep = cg.sepset[a][b]
                sep_vars = _sepset_vars(sep)
                if sep_vars:
                    H |= set(sep_vars)

    return H


def _min_dsep(cg: CausalGraph, S, i, j, alpha=0.05):
    S = tuple(sorted(set(S)))
    if not S:
        S_list = []
        cg.sepset[i][j] = cg.sepset[j][i] = S_list
        return S_list

    S_list = list(S)
    for size in range(1, len(S) + 1):
        for subset in combinations(S, size):
            if cg.ci_test(i, j, list(subset)) >= alpha:
                S_list = list(subset)
                cg.sepset[i][j] = cg.sepset[j][i] = S_list
                return S_list

    cg.sepset[i][j] = cg.sepset[j][i] = S_list

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
    parallel_backend="threads",
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
    max_n = min(k, len(base_x))
    max_m = min(k, len(base_y))
    for n in range(1, max_n + 1):
        for m in range(1, max_m + 1):
            z_sets = []
            tasks = []
            for z_x in combinations(base_x, n):
                for z_y in combinations(base_y, m):
                    Z = _get_hie(cg, {x, y} | set(z_x) | set(z_y)) - {x, y}
                    Z_sorted = tuple(sorted(Z))
                    z_sets.append(Z_sorted)
                    tasks.append((x, y, Z_sorted))

            if stats is not None:
                stats["dsep_candidate_sets"] = int(stats.get("dsep_candidate_sets", 0)) + len(tasks)

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
            for (_, _, _S, p_value), Z_sorted in zip(pvals, z_sets):
                if p_value >= alpha:
                    Z = _min_dsep(cg, Z_sorted, x, y, alpha=alpha)
                    cg.sepset[x][y] = list(Z)
                    cg.sepset[y][x] = list(Z)
                    # Found a valid D-SEP set: remove the edge X-Y to ensure progress.
                    cg.G.graph[x][y] = NULL
                    cg.G.graph[y][x] = NULL
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
                        stats["dsep_links_found"] = int(stats.get("dsep_links_found", 0)) + 1
                    return True, _dsep_search(cg, p)

    return False, deque()


def fci_plus_stable(
    cg: CausalGraph,
    data: np.ndarray,
    alpha: float,
    indep_test='fisherz',
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
) -> np.ndarray:

    # print(data)
    total_t0 = time.perf_counter()
    n, p = data.shape
    n_jobs_eff = 1 if n_jobs in (None, 0) else int(n_jobs)
    skeleton_jobs_eff = n_jobs_eff if skeleton_n_jobs in (None, 0) else int(skeleton_n_jobs)
    augment_jobs_eff = n_jobs_eff if augment_n_jobs in (None, 0) else int(augment_n_jobs)
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
    cit = CIT(data, method=indep_test)
    process_ci_info = (
        {
            "method": indep_test,
            "corr_matrix": cit.corr_matrix,
            "n_samples": cit.n_samples,
        }
        if (use_process_skeleton or use_process_augment or use_process_dsep)
        else None
    )
    ci_stats = {
        "total": 0,
        "skeleton": 0,
        "augment": 0,
        "dsep": 0,
        "skeleton_backend": "processes" if use_process_skeleton else "threads",
        "augment_backend": "processes" if use_process_augment else "threads",
        "ci_backend": "processes" if (use_process_augment or use_process_dsep) else "threads",
        "dsep_backend": "processes" if use_process_dsep else "threads",
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
    ci_cache = {} if use_ci_cache else None
    ci_inflight = {}
    stats_lock = threading.Lock()
    cg._fci_plus_process_min_tasks = int(process_min_tasks)
    cg._fci_plus_parallel_stats = ci_stats
    cg._fci_plus_parallel_phase = phase

    def counted_cit(i, j, S):
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
    skeleton = SkeletonDiscovery(
        cg,
        alpha,
        n_jobs=skeleton_jobs_eff,
        verbose=verbose,
        show_progress=show_progress,
        parallel_backend="processes" if use_process_skeleton else "threads",
        process_ci_info=process_ci_info if use_process_skeleton else None,
        protected_edges=protected_edges,
    )
    graphDict = skeleton.run_fci()
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
    augment_t0 = time.perf_counter()
    AugmentGraph(
        cg,
        p,
        alpha,
        n_jobs=augment_jobs_eff,
        parallel_backend="processes" if use_process_augment else "threads",
        process_ci_info=process_ci_info if use_process_augment else None,
        record_process_tests=record_process_tests if use_process_augment else None,
        record_cache_hits=record_cache_hits,
        ci_cache=ci_cache,
        stats=ci_stats,
    )
    ci_stats["time_augment_sec"] = time.perf_counter() - augment_t0
    dsep_pos = _dsep_search(cg, p)
    ci_stats["dsep_links_initial"] = len(dsep_pos)

    phase["name"] = "dsep"
    dsep_t0 = time.perf_counter()
    while len(dsep_pos)>0:
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
            parallel_backend="processes" if use_process_dsep else "threads",
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
        # CFCI/FCI+ не ориентируют коллайдер, если для пары не записано
        # ни одного разделяющего множества — сохраняем прежнее поведение.
        require_recorded_sepset=True,
    )
    # pag = udag2pag(pag, sepset_pair, p, orientCollider=True, rules=[False for _ in range(10)])

    # Перекодировка не нужна: udag2pag и FCI+ работают в одной
    # кодировке causal-learn.
    cg.G.graph = pag
    ci_stats["time_orient_sec"] = time.perf_counter() - orient_t0
    if ci_cache is not None:
        ci_stats["ci_cache_size"] = int(len(ci_cache))

    for phase_name in ("augment", "dsep"):
        process_tasks = int(ci_stats.get(f"{phase_name}_process_tasks", 0))
        thread_tasks = int(ci_stats.get(f"{phase_name}_thread_tasks", 0))
        if process_tasks and thread_tasks:
            effective_backend = "hybrid"
        elif process_tasks:
            effective_backend = "processes"
        elif thread_tasks:
            effective_backend = "threads"
        else:
            effective_backend = "serial"
        ci_stats[f"{phase_name}_effective_backend"] = effective_backend

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


from .rfci_algo import rfci_stable
from .fci_algo_v2 import fci_stable_v2

if __name__ == "__main__":
    df = make_collider_data()
    cg = CausalGraph(no_of_var=7, node_names=['X', 'Y', 'Z', 'W', 'T', 'U', 'V'])
    pag1 = fci_plus_stable(cg, df, 0.05, indep_test='fisherz', n_jobs=1, verbose=True)

    cg_rfci = CausalGraph(no_of_var=7, node_names=['X', 'Y', 'Z', 'W', 'T', 'U', 'V'])
    pag2 = rfci_stable(cg_rfci, df, 0.05, indep_test='fisherz', n_jobs=1, verbose=True)

    cg = CausalGraph(no_of_var=7, node_names=['X','Y','Z','W','T','U','V'])
    pag3 = fci_stable_v2(cg, df, alpha=0.05, indep_test='fisherz', n_jobs=1, verbose=True)

    # for i in range(pag1.shape[0]):
    #     print(pag1[i])
    #     print(pag2[i])
    #     print(pag3[i])
    #     print('\n'*3)
