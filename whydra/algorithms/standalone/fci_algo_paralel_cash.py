from itertools import combinations
import numpy as np
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed

# Импортируем универсальный класс тестов из соседнего файла
from ..graph_core import CIT, fisher_z_from_corr
from ...background_knowledge import BKPhase, MatrixEncoding, apply_bk_checkpoint

# --- Вспомогательные функции ---
from .skeleton import SkeletonDiscovery
from .pag_init import to_circle_adjacency
from .pag_rules import PAG_ARROW, PAG_CIRCLE, powerset, qreach, udag2pag


def _ci_cache_key(i: int, j: int, S):
    """Canonical cache key for CI test symmetry in (i, j, S)."""
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
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


# Fisher-Z lives in graph_core so every backend computes it identically;
# a local copy here would silently drift from CIT.fisher_z.
_fisher_z_from_corr = fisher_z_from_corr


def _fisher_z_pvals_chunk(corr_matrix, n_samples, x, y, cond_sets):
    return [
        _fisher_z_from_corr(corr_matrix, n_samples, x, y, S)
        for S in cond_sets
    ]


def _parallel_indep_eval(
    indepTest,
    suffStat,
    x,
    y,
    cond_sets,
    n_jobs=1,
    parallel_backend="threads",
    process_ci_info=None,
):
    cond_sets = [_normalize_cond_set(S) for S in cond_sets]
    if not cond_sets:
        return []
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
            backend="loky",
            max_nbytes="10K",
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
    return Parallel(n_jobs=int(n_jobs), prefer="threads", require="sharedmem")(
        delayed(indepTest)(suffStat, x, y, list(S)) for S in cond_sets
    )


def checkTriple(a, b, c, nbrsA, nbrsC, sepsetA, sepsetC, suffStat, alpha, indepTest, maj_rule=True, n_jobs=1):
    nr_indep = 0
    temp = []

    if len(nbrsA) > 0:
        cond_sets = list(powerset(nbrsA))
        pvals = _parallel_indep_eval(indepTest, suffStat, a, c, cond_sets, n_jobs=n_jobs)
        for s, pval in zip(cond_sets, pvals):
            if pval >= alpha:
                nr_indep += 1
                temp.append(b in s)

    if len(nbrsC) > 0:
        cond_sets = list(powerset(nbrsC))
        pvals = _parallel_indep_eval(indepTest, suffStat, a, c, cond_sets, n_jobs=n_jobs)
        for s, pval in zip(cond_sets, pvals):
            if pval >= alpha:
                nr_indep += 1
                temp.append(b in s)

    newsepsetA = set(sepsetA) if sepsetA is not None else set()
    newsepsetC = set(sepsetC) if sepsetC is not None else set()

    if len(temp) == 0:
        temp.append(False)

    res = 3
    if maj_rule:
        if sum(temp) / len(temp) < .5:
            res = 1
            if b in newsepsetA: newsepsetA.remove(b)
            if b in newsepsetC: newsepsetC.remove(b)
        elif sum(temp) / len(temp) > .5:
            res = 2
            newsepsetA.add(b)
            newsepsetC.add(b)
        else:
            pass
    else:
        if sum(temp) / len(temp) == 0:
            res = 1
            if b in newsepsetA: newsepsetA.remove(b)
            if b in newsepsetC: newsepsetC.remove(b)
        elif sum(temp) / len(temp) == 1:
            res = 2
            newsepsetA.add(b)
            newsepsetC.add(b)
        else:
            pass

    return res, {'sepsetA': newsepsetA, 'sepsetC': newsepsetC}


def _check_triple_task(args):
    return checkTriple(*args, n_jobs=1)


def pc_cons_intern(graphDict, suffstat, alpha, indepTest, version_unf=(None, None), maj_rule=True,
                   verbose=False, n_jobs=1):
    sk = graphDict['sk']
    p = sk.shape[0]

    if np.any(sk):
        # Сортировка индексов как в pcalg для воспроизводимости
        ind = np.transpose(np.where(sk == PAG_CIRCLE))
        ind = ind[ind[:, 1].argsort()]

        tripleMatrix = []
        for a, b in ind:
            for c in range(p):
                if a < c and sk[a, c] == 0 and sk[b, c] == PAG_CIRCLE:
                    tripleMatrix.append((a, b, c))

        triple_args = []
        for a, b, c in tripleMatrix:
            nbrsA = np.where(sk[:, a] == PAG_CIRCLE)[0]
            nbrsC = np.where(sk[:, c] == PAG_CIRCLE)[0]
            triple_args.append((
                a,
                b,
                c,
                nbrsA,
                nbrsC,
                graphDict['sepset'][(a, c)],
                graphDict['sepset'][(c, a)],
                suffstat,
                alpha,
                indepTest,
                maj_rule,
            ))

        if n_jobs not in (None, 0, 1) and len(triple_args) > 1:
            max_workers = int(n_jobs) if int(n_jobs) > 0 else (os.cpu_count() or 1)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                triple_results = list(executor.map(_check_triple_task, triple_args))
            triple_backend = "threads"
        else:
            triple_results = [
                _check_triple_task(args)
                for args in triple_args
            ]
            triple_backend = "serial"

        for (a, b, c), (res, _) in zip(tripleMatrix, triple_results):
            if res == 3:
                if 'unfTriples' in graphDict.keys():
                    graphDict['unfTriples'].add((a, b, c))
                else:
                    graphDict['unfTriples'] = {(a, b, c)}

            sepsetA = set(graphDict['sepset'][(a, c)])
            sepsetC = set(graphDict['sepset'][(c, a)])
            if res == 1:
                sepsetA.discard(b)
                sepsetC.discard(b)
            elif res == 2:
                sepsetA.add(b)
                sepsetC.add(b)
            graphDict['sepset'][(a, c)] = sepsetA
            graphDict['sepset'][(c, a)] = sepsetC

        graphDict["triple_stats"] = {
            "triples": int(len(tripleMatrix)),
            "backend": triple_backend,
            "n_jobs": int(n_jobs) if n_jobs not in (None, 0) else 1,
        }

    return graphDict


# def skeleton(suffStat, indepTest, alpha, labels, method,
#              fixedGaps, fixedEdges,
#              NAdelete, m_max, numCores, verbose):
#     sepset = {}
#     for i in itertools.permutations([i for i in range(len(labels))], 2):
#         sepset[i] = set()
#
#     G = np.ones((len(labels), len(labels)), dtype=int)
#     pMax = np.full((len(labels), len(labels)), -np.inf)
#
#     for i in range(len(labels)):
#         pMax[i, i] = 1
#         G[i, i] = 0
#
#     done = False
#     ord = 0
#     n_edgetests = {}
#
#     while done != True and np.any(G) and ord <= m_max:
#         ord1 = ord + 1
#         n_edgetests[ord1] = 0
#         done = True
#
#         ind = np.transpose(np.where(G == 1))
#
#         # STABLE SKELETON: Snapshot
#         G1 = np.copy(G)
#
#         for x, y in ind:
#             if G[y, x] == 1:
#                 nbrs = np.where(G1[:, x] == 1)[0]
#                 nbrs = nbrs[nbrs != y]
#
#                 if len(nbrs) >= ord:
#                     if len(nbrs) > ord:
#                         done = False
#
#                     for nbrs_S in set(itertools.combinations(nbrs, ord)):
#                         n_edgetests[ord1] = n_edgetests[ord1] + 1
#                         pval = indepTest(suffStat, x, y, list(nbrs_S))
#                         if pMax[x, y] < pval:
#                             pMax[x, y] = pval
#                         if pval >= alpha:
#                             G[x, y] = G[y, x] = 0
#                             sepset[(x, y)] = list(nbrs_S)
#                             sepset[(y, x)] = list(nbrs_S)
#                             break
#         ord += 1
#
#     for i in range(0, len(labels) - 1):
#         for j in range(1, len(labels)):
#             pMax[i, j] = pMax[j, i] = max(pMax[i, j], pMax[j, i])
#
#     return {'sk': G, 'pMax': pMax, 'sepset': sepset, "unfTriples": set(), "max_ord": ord - 1}


# --- Функции для поиска путей (из pcalg) ---

# _pdsep_legacy удалён: не вызывался и уже разошёлся с рабочим pdsep —
# в нём не было protected_edges, то есть откат на него снял бы защиту
# обязательных рёбер BK на шаге Possible-D-SEP.


def _pdsep_fisherz_task_chunk(corr_matrix, n_samples, tasks):
    return [
        (pair_idx, cond_idx, _fisher_z_from_corr(corr_matrix, n_samples, x, y, S))
        for pair_idx, cond_idx, x, y, S in tasks
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

    if n_jobs in (None, 0, 1):
        return [
            (pair_idx, cond_idx, indepTest(suffStat, x, y, list(S)))
            for pair_idx, cond_idx, x, y, S in tasks
        ]

    n_jobs_eff = int(n_jobs)
    chunks = _chunk_sequence(tasks, max(n_jobs_eff * 4, n_jobs_eff))

    if (
        parallel_backend == "processes"
        and process_ci_info is not None
        and process_ci_info.get("method") == "fisherz"
    ):
        executor = parallel_executor or Parallel(
            n_jobs=n_jobs_eff,
            backend="loky",
            max_nbytes="10K",
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
        owns_executor = parallel_executor is None
        executor = parallel_executor or ThreadPoolExecutor(max_workers=n_jobs_eff)
        try:
            futures = [
                executor.submit(_pdsep_indep_task_chunk, indepTest, suffStat, chunk)
                for chunk in chunks
            ]
            chunk_results = [future.result() for future in futures]
        finally:
            if owns_executor:
                executor.shutdown(wait=True)

    return [item for chunk in chunk_results for item in chunk]


def pdsep(skel, suffStat, indepTest, p, sepSet, alpha, pMax, m_max=float('inf'), pdsep_max=float('inf'),
          unfVect=None, n_jobs=1, parallel_backend="threads", process_ci_info=None,
          process_min_tasks=256, protected_edges=()):
    G = skel['sk'].astype(int)
    protected_edges = {
        frozenset((int(x), int(y))) for x, y in protected_edges
    }
    n_edgetest = [0 for i in range(1000)]
    max_ord_reached = 0
    allPdsep_tmp = [set() for i in range(p)]
    amat = np.copy(G)
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

    ind = []
    for i in range(len(G)):
        for j in range(len(G[i])):
            if G[i][j] == PAG_CIRCLE:
                ind.append((i, j))
    ind = sorted(ind, key=lambda x: (x[1], x[0]))

    for x, y in ind:
        allZ = [i for i in range(len(amat[y])) if amat[y][i] != 0 and i != x]
        for z in allZ:
            if amat[x][z] == 0 and not (y in sepSet[(x, z)] or y in sepSet[(z, x)]):
                if len(unfVect) == 0:
                    amat[x][y] = amat[z][y] = PAG_ARROW
                else:
                    if (x, y, z) not in unfVect and (z, y, x) not in unfVect:
                        amat[x][y] = amat[z][y] = PAG_ARROW

    allPdsep = [qreach(x, amat) for x in range(p)]
    allPdsep_tmp = [[] for i in range(p)]
    pair_states = []

    for x in range(p):
        an0 = [True if amat[x][i] != 0 else False for i in range(len(amat))]
        tf1 = [i for i in allPdsep[x] if i != x]
        adj_x = [i for i in range(len(an0)) if an0[i] == True]

        for y in adj_x:
            if frozenset((int(x), int(y))) in protected_edges:
                continue
            pdsep_pairs_scanned += 1
            tf = [i for i in tf1 if i != y]
            diff_set = [i for i in tf if i not in adj_x]
            allPdsep_tmp[x] = tf + [y]

            if len(diff_set) > 0:
                pdsep_pairs_with_diffset += 1
                max_pair_ord = len(tf) if math.isinf(m_max) else min(len(tf), int(m_max))
                if max_pair_ord > 0:
                    pair_states.append({
                        "x": int(x),
                        "y": int(y),
                        "tf": tuple(int(v) for v in tf),
                        "diff_set": tuple(int(v) for v in diff_set),
                        "adj_x": frozenset(int(v) for v in adj_x),
                        "max_ord": int(max_pair_ord),
                        "active": True,
                    })

    global_max_ord = max((pair["max_ord"] for pair in pair_states), default=0)

    thread_executor = None
    process_executor = None
    if n_jobs not in (None, 0, 1):
        thread_workers = int(n_jobs) if int(n_jobs) > 0 else (os.cpu_count() or 1)
        thread_executor = ThreadPoolExecutor(max_workers=thread_workers)

    for ord in range(1, global_max_ord + 1):
        rounds += 1
        build_t0 = time.perf_counter()
        tasks = []
        task_cond_sets = {}

        for pair_idx, pair in enumerate(pair_states):
            if not pair["active"] or ord > pair["max_ord"]:
                continue

            x = pair["x"]
            y = pair["y"]
            if amat[x][y] == 0 and amat[y][x] == 0:
                pair["active"] = False
                continue

            if ord == 1:
                cond_sets = [(S,) for S in pair["diff_set"]]
            else:
                adj_x = pair["adj_x"]
                cond_sets = [
                    tuple(S)
                    for S in combinations(pair["tf"], ord)
                    if not set(S).issubset(adj_x)
                ]

            if not cond_sets:
                continue

            task_cond_sets[pair_idx] = cond_sets
            for cond_idx, S in enumerate(cond_sets):
                tasks.append((pair_idx, cond_idx, x, y, tuple(int(v) for v in S)))

        task_build_time += time.perf_counter() - build_t0
        if not tasks:
            continue

        ci_t0 = time.perf_counter()
        round_backend = parallel_backend
        if (
            round_backend == "processes"
            and n_jobs not in (None, 0, 1)
            and len(tasks) < int(process_min_tasks)
        ):
            round_backend = "threads"

        if n_jobs not in (None, 0, 1):
            if round_backend == "processes":
                process_rounds += 1
                if process_executor is None:
                    process_executor = Parallel(
                        n_jobs=int(n_jobs),
                        backend="loky",
                        max_nbytes="10K",
                    )
                    process_executor.__enter__()
            else:
                thread_rounds += 1

        task_results = _run_pdsep_ci_tasks(
            indepTest,
            suffStat,
            tasks,
            n_jobs=n_jobs,
            parallel_backend=round_backend,
            process_ci_info=process_ci_info,
            parallel_executor=(
                process_executor if round_backend == "processes" else thread_executor
            ),
        )
        ci_eval_time += time.perf_counter() - ci_t0

        pdsep_ci_tests += len(task_results)
        if n_jobs not in (None, 0, 1):
            if round_backend == "processes":
                process_ci_tests += len(task_results)
            else:
                thread_ci_tests += len(task_results)
        if ord + 1 >= len(n_edgetest):
            n_edgetest.extend([0] * (ord + 2 - len(n_edgetest)))
        n_edgetest[ord + 1] += len(task_results)

        grouped_results = {}
        for pair_idx, cond_idx, pval in task_results:
            grouped_results.setdefault(pair_idx, []).append((cond_idx, pval))

        apply_t0 = time.perf_counter()
        for pair_idx in sorted(grouped_results):
            pair = pair_states[pair_idx]
            if not pair["active"]:
                continue

            x = pair["x"]
            y = pair["y"]
            if amat[x][y] == 0 and amat[y][x] == 0:
                pair["active"] = False
                continue

            cond_sets = task_cond_sets[pair_idx]
            for cond_idx, pval in sorted(grouped_results[pair_idx], key=lambda item: item[0]):
                if pval > pMax[x, y]:
                    pMax[x, y] = pval
                if pval >= alpha:
                    S = cond_sets[cond_idx]
                    amat[x][y] = amat[y][x] = 0
                    sepSet[(x, y)] = sepSet[(y, x)] = set(S)
                    pair["active"] = False
                    break

        apply_time += time.perf_counter() - apply_t0
        max_ord_reached = ord

        if not any(pair["active"] and ord < pair["max_ord"] for pair in pair_states):
            break

    if process_executor is not None:
        process_executor.__exit__(None, None, None)
    if thread_executor is not None:
        thread_executor.shutdown(wait=True)

    for i in range(len(amat)):
        for j in range(len(amat[i])):
            if amat[i][j] == 0:
                G[i][j] = False
            else:
                G[i][j] = PAG_CIRCLE

    if process_rounds and thread_rounds:
        effective_backend = "hybrid"
    elif process_rounds:
        effective_backend = "processes"
    elif thread_rounds:
        effective_backend = "threads"
    else:
        effective_backend = "serial"

    return {
        'G': G,
        "sepset": sepSet,
        "pMax": pMax,
        "allPdsep": allPdsep_tmp,
        "max_ord": max_ord_reached,
        "pdsep_stats": {
            "pairs_scanned": int(pdsep_pairs_scanned),
            "pairs_with_diffset": int(pdsep_pairs_with_diffset),
            "ci_tests": int(pdsep_ci_tests),
            "backend": effective_backend,
            "requested_backend": parallel_backend,
            "rounds": int(rounds),
            "process_rounds": int(process_rounds),
            "thread_rounds": int(thread_rounds),
            "process_ci_tests": int(process_ci_tests),
            "thread_ci_tests": int(thread_ci_tests),
            "process_min_tasks": int(process_min_tasks),
            "tasks_built": int(pdsep_ci_tests),
            "task_build_time": float(task_build_time),
            "ci_eval_time": float(ci_eval_time),
            "apply_time": float(apply_time),
        },
    }


def fci_stable(
    cg,
    data,
    alpha=0.05,
    indep_test='fisherz',
    n_jobs=1,
    verbose=False,
    use_ci_cache=True,
    skeleton_backend=None,
    pdsep_backend=None,
    background_knowledge=None,
    show_progress: bool = False):
    # SkeletonDiscovery uses cg.ci_test(...), so CI test must be set before run_fci().
    total_t0 = time.perf_counter()
    n_jobs_eff = 1 if n_jobs in (None, 0) else int(n_jobs)
    skeleton_backend_eff = skeleton_backend or "threads"
    pdsep_backend_eff = pdsep_backend or "threads"
    use_process_skeleton = (
        n_jobs_eff != 1
        and indep_test == "fisherz"
        and skeleton_backend_eff == "processes"
    )
    use_process_pdsep = (
        n_jobs_eff != 1
        and indep_test == "fisherz"
        and pdsep_backend_eff == "processes"
    )
    cit = CIT(data, method=indep_test)
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
        "skeleton_backend": "processes" if use_process_skeleton else "threads",
        "pdsep_backend": "processes" if use_process_pdsep else "threads",
    }
    if use_ci_cache:
        ci_stats["ci_cache_hits"] = 0
        ci_stats["ci_cache_misses"] = 0
    phase = {"name": "skeleton"}
    ci_cache = {} if use_ci_cache else None
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
    # print(f'going skeleton')
    # print(f'cg.G.graph 0 {cg.G.graph}')

    # ini_file: str = 'results/cg_matrix_ini.csv'
    # np.savetxt(ini_file, cg.G.graph, delimiter=',')

    skeleton_t0 = time.perf_counter()
    skeleton = SkeletonDiscovery(
        cg,
        alpha,
        n_jobs=n_jobs_eff,
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
    # Explicitly represent post-skeleton graph as o-o for FCI phases.
    graphDict['sk'] = to_circle_adjacency(graphDict['sk'], MatrixEncoding.STANDARD)

    # skeleton_file: str = 'results/cg_matrix_skeleton.csv'
    # np.savetxt(skeleton_file, cg.G.graph, delimiter=',')


    n, p = data.shape

    # Обертка для совместимости с функциями FCI, которые ожидают (suffStat, x, y, S)
    def test_func_wrapper(suffStat, x, y, S):
        return counted_cit(x, y, S)

    # suffStat больше не нужен, так как CIT хранит данные внутри себя
    suffStat = None

    # graphDict = skeleton(suffStat, test_func_wrapper, alpha, labels=[str(i) for i in range(p)], method="stable",
    #                      fixedGaps=None, fixedEdges=None, NAdelete=True, m_max=float('inf'),
    #                      numCores=1, verbose=verbose)

    # print(f'graphDict skeleton pdsep {graphDict}')
    # print(f'graphDict skeleton pMax {graphDict["pMax"]}')
    # print(f'graphDict skeleton alpha {alpha}')
    # print(f'graphDict skeleton test_func_wrapper {test_func_wrapper}')

    # Classical FCI: do not apply conservative triple classification here.
    # print(f'graphDict before pdsep {graphDict}')
    # print(f'graphDict before suffStat {suffStat}')
    # print(f'graphDict before alpha {alpha}')
    # print(f'graphDict before test_func_wrapper {test_func_wrapper}')

    phase["name"] = "pdsep"
    pdsep_t0 = time.perf_counter()
    pdsep_res = pdsep(
        graphDict,
        suffStat,
        test_func_wrapper,
        p,
        graphDict['sepset'],
        alpha,
        graphDict['pMax'],
        unfVect=graphDict['unfTriples'],
        n_jobs=n_jobs_eff,
        parallel_backend="processes" if use_process_pdsep else "threads",
        process_ci_info=process_ci_info if use_process_pdsep else None,
        protected_edges=protected_edges,
    )
    ci_stats["time_pdsep_sec"] = time.perf_counter() - pdsep_t0
    # print(f'graphDict after pdsep {graphDict}')
    # print(f'graphDict after suffStat {suffStat}')
    # print(f'graphDict after alpha {alpha}')
    # print(f'graphDict after test_func_wrapper {test_func_wrapper}')

    rule_stats = {}
    orient_t0 = time.perf_counter()
    pag = udag2pag(
        pdsep_res['G'],
        pdsep_res['sepset'],
        p,
        unfVect=graphDict['unfTriples'],
        rule_stats=rule_stats,
        background_knowledge=background_knowledge,
        node_names=node_names,
    )
    ci_stats["time_orient_sec"] = time.perf_counter() - orient_t0
    pdsep_stats = pdsep_res.get("pdsep_stats", {})
    ci_stats["pdsep_pairs_scanned"] = int(pdsep_stats.get("pairs_scanned", 0))
    ci_stats["pdsep_pairs_with_diffset"] = int(pdsep_stats.get("pairs_with_diffset", 0))
    ci_stats["pdsep_ci_tests"] = int(pdsep_stats.get("ci_tests", 0))
    ci_stats["pdsep_backend"] = pdsep_stats.get("backend", ci_stats["pdsep_backend"])
    ci_stats["pdsep_rounds"] = int(pdsep_stats.get("rounds", 0))
    ci_stats["pdsep_tasks_built"] = int(pdsep_stats.get("tasks_built", 0))
    ci_stats["time_pdsep_task_build_sec"] = float(pdsep_stats.get("task_build_time", 0.0))
    ci_stats["time_pdsep_ci_eval_sec"] = float(pdsep_stats.get("ci_eval_time", 0.0))
    ci_stats["time_pdsep_apply_sec"] = float(pdsep_stats.get("apply_time", 0.0))
    if use_process_pdsep:
        ci_stats["total"] += ci_stats["pdsep_ci_tests"]
        ci_stats["pdsep"] += ci_stats["pdsep_ci_tests"]
        if "ci_cache_misses" in ci_stats:
            ci_stats["ci_cache_misses"] += ci_stats["pdsep_ci_tests"]
    for k, v in rule_stats.items():
        ci_stats[f"orient_{k.lower()}_changes"] = int(v)
    if ci_cache is not None:
        ci_stats["ci_cache_size"] = int(len(ci_cache))
    ci_stats["time_total_sec"] = time.perf_counter() - total_t0
    cg.ci_stats = ci_stats

    return pag
