"""
GFCI (Greedy Fast Causal Inference) Algorithm.

A hybrid causal discovery algorithm that combines score-based search with
constraint-based FCI to handle latent variables.

Стартовый CPDAG здесь строит GES (Chickering 2002) из ``score_based.ges``.
В статье про GFCI на этом месте стоит FGES (Ramsey et al.) — это оптимизированная
реализация того же поиска (кэш оценок, приоритетная очередь, инкрементальный
пересчёт), а не другой алгоритм, поэтому результат от замены не зависит.
Проверено на Tetrad: его FGES с ``faithfulnessAssumed`` и без даёт один и тот же
граф на 10/10 кейсах, и совпадение с нашим GES остаётся 4/10 в обоих случаях —
то есть расхождение с Tetrad идёт от score-функции (см. пункт 1 ниже), а не от
FGES-эвристик. По скорости наш GES на p=50 укладывается в ~30 с против ~200 с у
Tetrad, так что оптимизации FGES пока не нужны.

Чем эта реализация отличается от эталонного GFCI из Tetrad
----------------------------------------------------------
Эталон GFCI существует только в Tetrad (в causal-learn его нет — там есть
отдельно FCI и отдельно GES). Сверка с ним живёт в
``local_checks/gfci_vs_tetrad.py``; на 23 латентных кейсах совпадение точное в
9 из 23. Ниже — установленные причины расхождения, чтобы следующий читатель не
искал их заново.

1. **Score-функция — главная причина.** Наш гауссов BIC совпадает с
   ``local_score_BIC`` из causal-learn, но не с ``sem-bic`` из Tetrad: на одном
   и том же графе наша формула даёт −821.18, а Tetrad сообщает −972.30.
   Перебор 24 комбинаций (число параметров |Pa|, |Pa|+1, |Pa|+2; оценка
   дисперсии rss/n, rss/(n−1), rss/(n−k), rss/(n−k−1); множитель n или n−1) их
   число не воспроизводит — ближайшая промахивается на 69. То есть формулы
   расходятся структурно, а не на константу penaltyDiscount, и алгоритмы
   оптимизируют разные функции.

2. **Реализация поиска.** Наш ``score_based.ges`` воспроизводит Chickering GES
   и совпадает с ``causallearn.ges`` побитово (10/10 на causally sufficient
   данных), тогда как Tetrad FGES совпадает с causal-learn лишь 4/10 — то есть
   расходится с ним примерно так же, как с нами. Дело не в самом FGES: его
   эвристика ``faithfulnessAssumed`` на этих данных ничего не меняет (10/10
   одинаковых графов), так что переход на FGES расхождение не убрал бы.

3. **Правило R4 (discriminating paths).** У нас коллайдер определяется по
   записанному sepset (подход pcalg, см. ``pag_rules.udag2pag``), у
   Tetrad/causal-learn — новым CI-тестом на данных. На одинаковых скелетах
   результат совпадает (12/12), так что на практике это не расходится.

4. **Заполнение пропущенных sepset'ов** — см. ``_fill_missing_sepsets_gfci``:
   наша структурная эвристика, аналога в Tetrad нет.

5. **Разные дефолты Tetrad**, которые сверка не выравнивает: ``maxDegree=4``,
   ``completeRuleSetUsed=no``, ``alpha=0.01``. Проверено, что расхождение они
   не объясняют: снятие maxDegree и включение полного набора правил меняют
   результат в пределах шума (36 против 37 различающихся ячеек).

Проверено и исключено как причина: локаль JVM (влияет только на печать),
формат записи чисел в CSV, значение penaltyDiscount (перебор 0.5-2.0).

По качеству на причинно достаточных данных наш стартовый CPDAG ближе к истине:
суммарное SHD 26 против 44 у Tetrad FGES на 10 графах.

Открытые вопросы по этим расхождениям заведены в коде:
``TODO(ges-tetrad-sembic)`` в ``score_based/scores.py`` — воспроизвести формулу
Tetrad, чтобы сверка перестала мерить сумму двух разниц; и
``TODO(gfci-missing-sepsets)`` в ``_fill_missing_sepsets_gfci`` — выбрать между
структурным и CI-вариантом восстановления пропущенных sepset'ов.
"""

from __future__ import annotations

import itertools
from itertools import combinations
import numpy as np
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed

from ..graph_core import (
    CIT, CausalGraph, Edge, Endpoint, GeneralGraph, Node,
    resolve_score_func, fisher_z_from_corr,
)
from ...background_knowledge import BKPhase, MatrixEncoding, apply_bk_checkpoint
from ..score_based import ges_search
from ..score_based.scores import local_parent_score

from .pag_rules import PAG_ARROW, qreach, udag2pag

# Kept under its old private name: the score itself moved to
# ``score_based.scores`` so that GES can use it without importing this module.
_local_parent_score = local_parent_score


# =============================================================================
#  CPDAG extraction from the local score-search result
# =============================================================================

def _ges_cpdag_to_matrix(ges_graph, node_names=None):
    """
    Convert a local GeneralGraph (CPDAG from GES) to the internal
    numpy PAG matrix encoding used by the FCI orientation routines.

    Кодировка — causal-learn, та же, что во всей библиотеке:
      0 = NULL (нет ребра), -1 = TAIL (-), 1 = ARROW (>), 2 = CIRCLE (o)

    pag[x, y] = endpoint at y on the edge between x and y.
    """
    if node_names is None:
        nodes = ges_graph.get_nodes()
        node_names = [n.get_name() for n in nodes]
    else:
        nodes = ges_graph.get_nodes()

    p = len(node_names)
    pag = np.zeros((p, p), dtype=int)

    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            node_i = nodes[i]
            node_j = nodes[j]
            edge = ges_graph.get_edge(node_i, node_j)
            if edge is not None:
                # Use Graph.get_endpoint(node, target) which returns endpoint AT node
                pag[i, j] = _endpoint_code(
                    ges_graph.get_endpoint(node_j, node_i))
            else:
                pag[i, j] = 0

    return pag


def _endpoint_code(endpoint):
    """Числовой код метки causal-learn.

    Раньше здесь была перекодировка в pcalg — она была нужна, пока правила
    ориентации считали в другой кодировке. Теперь кодировка в библиотеке одна,
    и значение Endpoint используется напрямую.
    """
    return endpoint.value if hasattr(endpoint, 'value') else int(endpoint)


def _get_cpdag_colliders(cpdag_graph, node_names):
    """
    Extract definite collider triples from a local CPDAG.

    Returns a set of (x, y, z) tuples where y is the collider (x *-> y <-* z).
    Indices are mapped through node_names.
    """
    nodes = cpdag_graph.get_nodes()
    name_to_idx = {n.get_name(): idx for idx, n in enumerate(nodes)}

    colliders = set()
    for y_node in nodes:
        adj_y = cpdag_graph.get_adjacent_nodes(y_node)
        for x_node, z_node in itertools.combinations(adj_y, 2):
            try:
                if (
                    not cpdag_graph.is_adjacent_to(x_node, z_node)
                    and cpdag_graph.get_endpoint(y_node, x_node) == Endpoint.ARROW
                    and cpdag_graph.get_endpoint(y_node, z_node) == Endpoint.ARROW
                ):
                    x = name_to_idx[x_node.get_name()]
                    y = name_to_idx[y_node.get_name()]
                    z = name_to_idx[z_node.get_name()]
                    colliders.add((x, y, z))
                    colliders.add((z, y, x))  # symmetric
            except Exception:
                continue

    return colliders


# =============================================================================
#  Separating set search
# =============================================================================

def _get_sepset(x, y, adj_x, cit_func, alpha, depth, use_max_p):
    """
    Find a separating set for (x, y) within adj_x.

    Parameters
    ----------
    x, y : int
        Node indices.
    adj_x : list of int
        Candidate nodes for separating set (adjacency of x).
    cit_func : callable
        Independence test: cit_func(x, y, S) -> p_value.
    alpha : float
        Significance level.
    depth : int
        Maximum size of separating set to consider (-1 = unlimited).
    use_max_p : bool
        If True, use max-P approach (test all subsets, pick max p-value).
        If False, return the first separating set found.

    Returns
    -------
    set or None
        The separating set if found, None otherwise.
    """
    n_adj = len(adj_x)

    if depth < 0 or depth > n_adj:
        max_depth = n_adj
    else:
        max_depth = min(depth, n_adj)

    best_sepset = None
    best_pval = -1.0

    for d in range(0, max_depth + 1):
        for S in combinations(adj_x, d):
            pval = cit_func(x, y, list(S))

            if use_max_p:
                if pval > best_pval:
                    best_pval = pval
                    if pval >= alpha:
                        best_sepset = set(S)
            else:
                if pval >= alpha:
                    return set(S)

    if use_max_p and best_sepset is not None:
        return best_sepset

    return None


def _sepset_subset_of_adjx_or_adjy(pag, x, y, cit_func, alpha, depth, use_max_p):
    """
    GFCI extra edge removal step: find a separating set within adj(x) or adj(y).

    Corresponds to sepsetSubsetOfAdjxOrAdjy in the Java reference.

    Returns (sepset, pval) or (None, None).
    """
    p = pag.shape[0]

    adj_x = [i for i in range(p) if pag[x, i] != 0 and i != y]
    adj_y = [i for i in range(p) if pag[y, i] != 0 and i != x]

    # Test subsets of adj(x)
    sepset_x = _get_sepset(x, y, adj_x, cit_func, alpha, depth, use_max_p)
    # Test subsets of adj(y)
    sepset_y = _get_sepset(y, x, adj_y, cit_func, alpha, depth, use_max_p)

    if sepset_x is None and sepset_y is None:
        return None

    if sepset_x is not None and sepset_y is None:
        return sepset_x

    if sepset_x is None:
        return sepset_y

    # Both found: pick the one with higher p-value
    pval_x = cit_func(x, y, list(sepset_x))
    pval_y = cit_func(x, y, list(sepset_y))
    return sepset_x if pval_x > pval_y else sepset_y


# =============================================================================
#  Utility: get current edges as list of (i, j)
# =============================================================================

def _get_edges(pag):
    """Return list of undirected edges (i, j) with i < j from the PAG."""
    p = pag.shape[0]
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if pag[i, j] != 0 or pag[j, i] != 0:
                edges.append((i, j))
    return edges


# =============================================================================
#  Collider copying from CPDAG
# =============================================================================

# =============================================================================
#  Possible-D-SEP computation (from FCI)
# =============================================================================

def _fill_missing_sepsets_gfci(pag, sepset_map, cpdag_colliders):
    """Fill sepsets for non-adjacent pairs appearing in unshielded triples.

    GES skeleton phase only CI-tests pairs adjacent in the FGES CPDAG, so pairs
    absent from FGES never get a sepset recorded.  udag2pag R0 does
    ``sepset.get((x,z), [])`` → empty list → y is never in it → y is always
    oriented as collider → false v-structures.

    Fix (structural, not CI-based): for every unshielded triple x-y-z where
    sepset(x,z) is missing, use the GES CPDAG collider information:
      - if (x,y,z) is a CPDAG collider → y is a collider → sepset(x,z) = {}
      - otherwise y is a non-collider → sepset(x,z) = {y}

    This matches the selected GFCI orientation policy and avoids false-positive CI results
    that appear with finite samples when testing non-adjacent pairs.

    Замер на причинно достаточных данных (10 графов, n=2000), где двунаправленных
    рёбер быть не должно вовсе: с эвристикой 8 ложных ``<->``, без неё — 17, при
    этом SHD до истины не растёт (64 против 65). То есть она работает ровно на то,
    для чего написана.
    """
    # TODO(gfci-missing-sepsets): пропущенные разделяющие множества
    # восстанавливаются структурно — из коллайдеров стартового CPDAG, — тогда как
    # в статье GFCI (Ogarrio, Spirtes, Ramsey 2016) они берутся из CI-тестов.
    # Аналога этой эвристики в Tetrad нет, так что расхождение с его GFCI частично
    # идёт отсюда, и какой вариант вернее — не проверено.
    #
    # Замер (10 графов, n=2000, причинная достаточность, двунаправленных рёбер
    # быть не должно): с эвристикой 8 ложных <->, без неё 17; SHD до истины 64
    # против 65. То есть отключение делает хуже, и оставить как было нельзя.
    #
    # Почему вообще понадобилась: GES не тестирует несмежные пары, поэтому для
    # них sepset не записан; udag2pag R0 читает пропуск как пустое множество,
    # «y не входит в пустое множество» истинно — и ориентируется ложный коллайдер.
    #
    # Нужно сравнить со вторым вариантом — явно протестировать несмежные пары
    # из неэкранированных троек CI-тестом — и выбрать по замеру. У структурного
    # варианта плюс в том, что он не зависит от ложноположительных тестов на
    # конечной выборке, минус — опирается на корректность стартового CPDAG.
    p = pag.shape[0]
    cpdag_colliders_set = set(cpdag_colliders)

    def _adj(v):
        return [u for u in range(p) if pag[v, u] != 0 or pag[u, v] != 0]

    checked = set()
    for y in range(p):
        adj_y = _adj(y)
        for xi in range(len(adj_y)):
            x = adj_y[xi]
            for z in adj_y[xi + 1:]:
                if pag[x, z] != 0 or pag[z, x] != 0:
                    continue  # shielded triple — skip
                key = (min(x, z), max(x, z))
                if key in checked:
                    continue
                if (x, z) in sepset_map or (z, x) in sepset_map:
                    continue  # already have a sepset
                checked.add(key)

                if (x, y, z) in cpdag_colliders_set or (z, y, x) in cpdag_colliders_set:
                    final = set()          # collider at y → y not in sepset
                else:
                    final = {y}            # non-collider at y → y in sepset
                sepset_map[(x, z)] = final
                sepset_map[(z, x)] = final


def _possible_dsep(pag, x):
    """
    Compute Possible-D-SEP for node x in the PAG.

    Reuses qreach from the maintained FCI implementation.
    """
    amat = np.copy(pag)
    # qreach expects the amat encoding where non-zero = adjacency
    result = qreach(x, amat)
    return result


# =============================================================================
#  Main GFCI algorithm
# =============================================================================

def _ci_cache_key(i, j, S):
    a, b = (int(i), int(j)) if i <= j else (int(j), int(i))
    return a, b, tuple(sorted(int(v) for v in S))


def _worker_count(n_jobs):
    n_jobs = int(n_jobs)
    return n_jobs if n_jobs > 0 else (os.cpu_count() or 1)


def _resolve_gfci_backend(backend, n_vars, process_min_vars):
    requested = "auto" if backend in (None, "auto") else str(backend)
    if requested == "auto":
        effective = "processes" if int(n_vars) > int(process_min_vars) else "threads"
    elif requested in {"threads", "processes"}:
        effective = requested
    else:
        raise ValueError("GFCI backend must be 'auto', 'threads', or 'processes'")
    return requested, effective


def _chunk_sequence(items, n_chunks):
    if not items:
        return []
    n_chunks = max(1, min(int(n_chunks), len(items)))
    chunk_size = math.ceil(len(items) / n_chunks)
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


# Fisher-Z lives in graph_core so every backend computes it identically;
# a local copy here would silently drift from CIT.fisher_z.
_fisher_z_from_corr = fisher_z_from_corr


def _gfci_process_cit(corr_matrix, n_samples, x, y, S, cache, stats):
    key = _ci_cache_key(x, y, S)
    stats["calls"] += 1
    if cache is not None and key in cache:
        stats["hits"] += 1
        return cache[key]
    stats["misses"] += 1
    pval = _fisher_z_from_corr(
        corr_matrix,
        n_samples,
        x,
        y,
        key[2],
    )
    if cache is not None:
        cache[key] = pval
    return pval


def _gfci_adjacency_process_chunk(
    pag,
    edges,
    corr_matrix,
    n_samples,
    alpha,
    depth,
    use_max_p,
    use_ci_cache,
):
    cache = {} if use_ci_cache else None
    stats = {"calls": 0, "hits": 0, "misses": 0}

    def cit_func(x, y, S):
        return _gfci_process_cit(
            corr_matrix,
            n_samples,
            x,
            y,
            S,
            cache,
            stats,
        )

    results = []
    for a, b in edges:
        sepset = _sepset_subset_of_adjx_or_adjy(
            pag,
            a,
            b,
            cit_func,
            alpha,
            depth,
            use_max_p,
        )
        results.append((a, b, sepset))
    return results, stats


def _gfci_pdsep_process_chunk(
    pag,
    edges,
    corr_matrix,
    n_samples,
    alpha,
    depth,
    use_max_p,
    use_ci_cache,
):
    cache = {} if use_ci_cache else None
    stats = {"calls": 0, "hits": 0, "misses": 0}

    def cit_func(x, y, S):
        return _gfci_process_cit(
            corr_matrix,
            n_samples,
            x,
            y,
            S,
            cache,
            stats,
        )

    results = []
    for a, b in edges:
        pds_a = [v for v in _possible_dsep(pag, a) if v not in (a, b)]
        sepset = _get_sepset(
            a,
            b,
            pds_a,
            cit_func,
            alpha,
            depth,
            use_max_p,
        )
        if sepset is None:
            pds_b = [v for v in _possible_dsep(pag, b) if v not in (a, b)]
            sepset = _get_sepset(
                a,
                b,
                pds_b,
                cit_func,
                alpha,
                depth,
                use_max_p,
            )
        results.append((a, b, sepset))
    return results, stats


def _parallel_map(executor, fn, items):
    if executor is None:
        return [fn(item) for item in items]
    try:
        return list(executor.map(fn, items))
    except Exception:
        executor.shutdown(wait=False, cancel_futures=True)
        raise


def gfci_stable(
    cg,
    data,
    alpha=0.05,
    indep_test='fisherz',
    score_func='auto',
    use_max_p=True,
    depth=-1,
    n_jobs=1,
    verbose=False,
    start_from_complete_graph=False,
    unfVect=None,
    node_names=None,
    use_ci_cache=True,
    ci_backend=None,
    skeleton_backend=None,
    pdsep_backend=None,
    process_min_vars=150,
    background_knowledge=None,
    ges_max_operator_set=3,
    show_progress: bool = False):
    """
    Run the GFCI (Greedy Fast Causal Inference) algorithm.

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_samples, n_features).
    alpha : float
        Significance level for independence tests.
    indep_test : str
        Independence test method ('fisherz', 'chisq', etc.).
    score_func : str
        Score function for the GES initial graph. 'auto' matches the score to
        the data type `indep_test` assumes (BDeu for discrete tests, Gaussian
        BIC otherwise); an explicit name such as 'local_score_BIC' is used as is.
    use_max_p : bool
        If True, use max-P approach for sepset selection (more stable).
    depth : int
        Maximum separating set size (-1 = unlimited). Controls complexity.
    verbose : bool
        Print progress information.
    start_from_complete_graph : bool
        If True, skip GES and start from a complete graph (like FCI).
    unfVect : set or None
        Set of ambiguous triples (underdetermined v-structures).
    node_names : list of str or None
        Node names. If None, uses X1, X2, ...
    ges_max_operator_set : int
        Caps |T| and |H| in the GES Insert/Delete operators. Enumerating those
        subsets is exponential in the size of NA_yx, so the cap only matters on
        dense neighbourhoods.

    Returns
    -------
    pag : np.ndarray
        PAG adjacency matrix in graph_core encoding.
        pag[x, y] = endpoint at y on edge x-y.
        0=NULL, -1=TAIL, 1=ARROW, 2=CIRCLE.
    """
    total_t0 = time.perf_counter()
    data = np.asarray(data)
    n, p = data.shape
    n_jobs_eff = 1 if n_jobs in (None, 0) else int(n_jobs)
    skeleton_backend_requested, skeleton_backend_eff = _resolve_gfci_backend(
        skeleton_backend if skeleton_backend is not None else ci_backend,
        p,
        process_min_vars,
    )
    pdsep_backend_requested, pdsep_backend_eff = _resolve_gfci_backend(
        pdsep_backend if pdsep_backend is not None else ci_backend,
        p,
        process_min_vars,
    )
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

    if unfVect is None:
        unfVect = set()

    if node_names is None:
        node_names = [node.name for node in cg.nodes]

    protected_edges = ()
    if background_knowledge is not None:
        protected_edges = background_knowledge.required_pairs(
            node_names, phase=BKPhase.PRE_SEARCH
        )
    protected_pairs = {frozenset((int(a), int(b))) for a, b in protected_edges}

    score_func = resolve_score_func(score_func, indep_test)
    cit = CIT(data, method=indep_test)
    ci_cache = {} if use_ci_cache else None
    ci_inflight = {}
    stats_lock = threading.Lock()
    phase = {"name": "adjacency"}
    ci_stats = {
        "total": 0,
        "adjacency": 0,
        "pdsep": 0,

        "n_jobs": int(n_jobs_eff),
        "skeleton_backend": "processes" if use_process_skeleton else "threads",
        "pdsep_backend": "processes" if use_process_pdsep else "threads",
        "skeleton_backend_requested": skeleton_backend_requested,
        "pdsep_backend_requested": pdsep_backend_requested,
        "process_min_vars": int(process_min_vars),
        "n_vars": int(p),
    }
    if use_ci_cache:
        ci_stats["ci_cache_hits"] = 0
        ci_stats["ci_cache_misses"] = 0

    def cit_func(x, y, S):
        key = _ci_cache_key(x, y, S)
        with stats_lock:
            ci_stats["total"] += 1
            ci_stats[phase["name"]] += 1

        if ci_cache is None:
            return cit(x, y, list(key[2]))

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
            pval = cit(x, y, list(key[2]))
            with stats_lock:
                ci_cache[key] = pval
            return pval
        finally:
            with stats_lock:
                ci_inflight.pop(key, None)
                event.set()

    # --- Step 1: Get initial CPDAG (GES or complete graph) ---
    ges_t0 = time.perf_counter()
    if start_from_complete_graph:
        if verbose:
            print("[GFCI] Step 1: Starting from complete graph")
        pag = np.full((p, p), Endpoint.CIRCLE.value, dtype=int)  # all edges o-o
        np.fill_diagonal(pag, 0)
        cpdag_colliders = set()
    else:
        if verbose:
            print("[GFCI] Step 1: Running GES for initial CPDAG...")
        ges_graph, score_val = ges_search(
            data,
            score_func,
            node_names,
            max_operator_set=ges_max_operator_set,
            verbose=verbose,
            stats=ci_stats,
        )

        if verbose:
            print(f"  GES complete. Score: {score_val}")

        pag = _ges_cpdag_to_matrix(ges_graph, node_names)
        cpdag_colliders = _get_cpdag_colliders(ges_graph, node_names)

        if verbose:
            n_edges = len(_get_edges(pag))
            n_colliders = len(cpdag_colliders) // 2
            print(f"  CPDAG: {n_edges} edges, {n_colliders} definite colliders")
    ci_stats["time_ges_sec"] = time.perf_counter() - ges_t0
    ci_stats["ges_edges"] = int(len(_get_edges(pag)))
    ci_stats["ges_colliders"] = int(len(cpdag_colliders) // 2)

    # PRE_SEARCH must give the same guarantee here that it gives PC/FCI: a
    # required adjacency always survives, even if the earlier search phase
    # (here, GES) didn't find it. For PC/FCI this is automatic because the
    # checkpoint runs on the still-fully-connected graph before skeleton
    # discovery starts. GES already produced a sparse CPDAG by this point,
    # so a missing required edge would otherwise be flagged as a conflict
    # instead of restored (``apply_matrix`` only restores via ``previous``).
    # ``previous`` is a fully-connected o-o graph so the restore always has
    # somewhere to draw the edge from.
    complete_pag = np.full((p, p), Endpoint.CIRCLE.value, dtype=int)
    np.fill_diagonal(complete_pag, 0)
    apply_bk_checkpoint(
        background_knowledge,
        pag,
        node_names,
        phase=BKPhase.PRE_SEARCH,
        checkpoint="before_skeleton",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="pag",
        previous=complete_pag,
    )

    sepset_map = {}
    executor = None
    if (
        n_jobs_eff != 1
        and (not use_process_skeleton or not use_process_pdsep)
    ):
        executor = ThreadPoolExecutor(max_workers=_worker_count(n_jobs_eff))

    # --- Step 2: Extra edge removal (subset of adj(x) or adj(y)) ---
    if verbose:
        print("[GFCI] Step 2: Extra edge removal (subset of adjacency)...")

    adjacency_t0 = time.perf_counter()
    edges_before = _get_edges(pag)
    adjacency_snapshot = np.array(pag, copy=True)

    def adjacency_worker(edge):
        a, b = edge
        sepset = _sepset_subset_of_adjx_or_adjy(
            adjacency_snapshot,
            a,
            b,
            cit_func,
            alpha,
            depth,
            use_max_p,
        )
        return a, b, sepset

    if use_process_skeleton:
        chunks = _chunk_sequence(
            edges_before,
            max(_worker_count(n_jobs_eff) * 4, 1),
        )
        chunk_results = Parallel(
            n_jobs=n_jobs_eff,
            backend="loky",
            max_nbytes="10K",
        )(
            delayed(_gfci_adjacency_process_chunk)(
                adjacency_snapshot,
                chunk,
                cit.corr_matrix,
                cit.n_samples,
                alpha,
                depth,
                use_max_p,
                use_ci_cache,
            )
            for chunk in chunks
        )
        adjacency_results = [
            result
            for results, _stats in chunk_results
            for result in results
        ]
        for _results, process_stats in chunk_results:
            ci_stats["total"] += int(process_stats["calls"])
            ci_stats["adjacency"] += int(process_stats["calls"])
            if use_ci_cache:
                ci_stats["ci_cache_hits"] += int(process_stats["hits"])
                ci_stats["ci_cache_misses"] += int(process_stats["misses"])
    else:
        adjacency_results = _parallel_map(executor, adjacency_worker, edges_before)
    for a, b, sepset in adjacency_results:
        if sepset is not None and frozenset((int(a), int(b))) not in protected_pairs:
            pag[a, b] = pag[b, a] = 0
            sepset_map[(a, b)] = sepset
            sepset_map[(b, a)] = sepset
    ci_stats["time_adjacency_sec"] = time.perf_counter() - adjacency_t0
    ci_stats["time_skeleton_sec"] = (
        ci_stats["time_ges_sec"] + ci_stats["time_adjacency_sec"]
    )
    ci_stats["adjacency_edges_scanned"] = int(len(edges_before))
    ci_stats["adjacency_edges_removed"] = int(
        len(edges_before) - len(_get_edges(pag))
    )

    if verbose:
        n_after = len(_get_edges(pag))
        print(f"  Removed {len(edges_before) - n_after} edges ({n_after} remaining)")

    # --- Step 3: Incorporate CPDAG colliders into sepset_map ---
    # The FCI orientation pass discovers colliders by checking if the middle
    # node is NOT in the sepset for the outer pair. We ensure CPDAG colliders
    # satisfy this condition so the orientation rules will pick them up naturally.
    if verbose:
        print("[GFCI] Step 3: Incorporating CPDAG colliders into sepset structure...")
    for (x, y, z) in cpdag_colliders:
        # For definite collider x *-> y <-* z, x and z should NOT be adjacent
        # in the working PAG. Also, y should NOT be in sepset(x, z).
        if (x, z) in sepset_map and y in sepset_map[(x, z)]:
            # Remove y from sepset to ensure collider is discovered
            sepset_map[(x, z)] = sepset_map[(x, z)] - {y}
            sepset_map[(z, x)] = sepset_map[(x, z)]

    # --- Step 4: Possible-D-SEP removal ---
    if verbose:
        print("[GFCI] Step 4: Possible-D-SEP removal...")

    phase["name"] = "pdsep"
    pdsep_t0 = time.perf_counter()
    edges_before_pdsep = _get_edges(pag)
    pdsep_snapshot = np.array(pag, copy=True)

    def pdsep_worker(edge):
        a, b = edge
        pds_a = _possible_dsep(pdsep_snapshot, a)
        pds_a = [n for n in pds_a if n != a and n != b]
        sepset = _get_sepset(a, b, pds_a, cit_func, alpha, depth, use_max_p)
        if sepset is not None:
            return a, b, sepset

        pds_b = _possible_dsep(pdsep_snapshot, b)
        pds_b = [n for n in pds_b if n != a and n != b]
        sepset = _get_sepset(a, b, pds_b, cit_func, alpha, depth, use_max_p)
        return a, b, sepset

    if use_process_pdsep:
        chunks = _chunk_sequence(
            edges_before_pdsep,
            max(_worker_count(n_jobs_eff) * 4, 1),
        )
        chunk_results = Parallel(
            n_jobs=n_jobs_eff,
            backend="loky",
            max_nbytes="10K",
        )(
            delayed(_gfci_pdsep_process_chunk)(
                pdsep_snapshot,
                chunk,
                cit.corr_matrix,
                cit.n_samples,
                alpha,
                depth,
                use_max_p,
                use_ci_cache,
            )
            for chunk in chunks
        )
        pdsep_results = [
            result
            for results, _stats in chunk_results
            for result in results
        ]
        for _results, process_stats in chunk_results:
            ci_stats["total"] += int(process_stats["calls"])
            ci_stats["pdsep"] += int(process_stats["calls"])
            if use_ci_cache:
                ci_stats["ci_cache_hits"] += int(process_stats["hits"])
                ci_stats["ci_cache_misses"] += int(process_stats["misses"])
    else:
        pdsep_results = _parallel_map(executor, pdsep_worker, edges_before_pdsep)
    for a, b, sepset in pdsep_results:
        if sepset is not None and frozenset((int(a), int(b))) not in protected_pairs:
            pag[a, b] = pag[b, a] = 0
            sepset_map[(a, b)] = sepset
            sepset_map[(b, a)] = sepset
    ci_stats["time_pdsep_sec"] = time.perf_counter() - pdsep_t0
    ci_stats["pdsep_edges_scanned"] = int(len(edges_before_pdsep))
    ci_stats["pdsep_edges_removed"] = int(
        len(edges_before_pdsep) - len(_get_edges(pag))
    )
    if executor is not None:
        executor.shutdown(wait=True)
        executor = None

    if verbose:
        n_after = len(_get_edges(pag))
        print(f"  Removed {len(edges_before_pdsep) - n_after} edges ({n_after} remaining)")

    # --- Step 5: Re-check sepset_map for CPDAG collider consistency ---
    for (x, y, z) in cpdag_colliders:
        if (x, z) in sepset_map and y in sepset_map[(x, z)]:
            sepset_map[(x, z)] = sepset_map[(x, z)] - {y}
            sepset_map[(z, x)] = sepset_map[(x, z)]

    # --- Step 5.5: Fill missing sepsets for all unshielded triples ---
    # Pairs non-adjacent in GES were never CI-tested, so their sepsets are
    # absent from sepset_map.  udag2pag R0 treats a missing entry as an empty
    # set, making every intermediate node look like a collider → false
    # v-structures.  Use GES CPDAG structure (not CI tests) to determine
    # whether y is a collider under the selected GFCI orientation policy.
    if verbose:
        print("[GFCI] Step 5.5: Filling missing sepsets for unshielded triples...")
    _fill_missing_sepsets_gfci(pag, sepset_map, cpdag_colliders)

    # --- Step 6: Full FCI orientation (R0-R10) ---
    if verbose:
        print("[GFCI] Step 6: Full FCI orientation (R0-R10)...")
    orient_t0 = time.perf_counter()
    rule_stats = {}
    gfci_rules = [True] * 10
    pag = udag2pag(
        pag,
        sepset_map,
        p,
        unfVect=unfVect,
        rules=gfci_rules,
        orientCollider=True,
        rule_stats=rule_stats,
        background_knowledge=background_knowledge,
        node_names=node_names if node_names is not None else [node.name for node in cg.nodes],
    )
    ci_stats["time_orient_sec"] = time.perf_counter() - orient_t0
    ci_stats["orientation_rule_updates"] = {
        key: int(value) for key, value in rule_stats.items()
    }

    if verbose:
        _summarize_pag(pag)

    cg.G.graph = pag
    cg.gfci_sepset = sepset_map
    if ci_cache is not None:
        ci_stats["ci_cache_size"] = int(len(ci_cache))
    ci_stats["skeleton"] = int(ci_stats["adjacency"])
    ci_stats["pdsep_ci_tests"] = int(ci_stats["pdsep"])
    ci_stats["time_total_sec"] = time.perf_counter() - total_t0
    cg.ci_stats = ci_stats
    return pag


def gfci(data, alpha=0.05, indep_test='fisherz', score_func='auto',
         use_max_p=True, depth=-1, n_jobs=1, verbose=False,
         start_from_complete_graph=False, unfVect=None, node_names=None,
         use_ci_cache=True, ci_backend=None, skeleton_backend=None,
         pdsep_backend=None, process_min_vars=150, background_knowledge=None):
    """Convenience API returning the standard graph_core endpoint matrix."""
    data = np.asarray(data)
    if node_names is None:
        node_names = [f"X{i + 1}" for i in range(data.shape[1])]
    cg = CausalGraph(data.shape[1], list(node_names))
    return gfci_stable(
        cg,
        data,
        alpha=alpha,
        indep_test=indep_test,
        score_func=score_func,
        use_max_p=use_max_p,
        depth=depth,
        n_jobs=n_jobs,
        verbose=verbose,
        start_from_complete_graph=start_from_complete_graph,
        unfVect=unfVect,
        node_names=node_names,
        use_ci_cache=use_ci_cache,
        ci_backend=ci_backend,
        skeleton_backend=skeleton_backend,
        pdsep_backend=pdsep_backend,
        process_min_vars=process_min_vars,
        background_knowledge=background_knowledge,
    )


def _summarize_pag(pag):
    """Print a summary of the PAG edge types."""
    p = pag.shape[0]
    n_edges = 0
    edge_types = {'-->': 0, '<--': 0, '<->': 0, 'o->': 0, '<-o': 0, 'o-o': 0, '---': 0, '?': 0}

    tail = Endpoint.TAIL.value
    arrow = Endpoint.ARROW.value
    circle = Endpoint.CIRCLE.value

    for i in range(p):
        for j in range(i + 1, p):
            ei = pag[j, i]  # endpoint at i
            ej = pag[i, j]  # endpoint at j
            if ei == 0 and ej == 0:
                continue
            n_edges += 1
            if ei == tail and ej == arrow:
                edge_types['-->'] += 1
            elif ei == arrow and ej == tail:
                edge_types['<--'] += 1
            elif ei == arrow and ej == arrow:
                edge_types['<->'] += 1
            elif ei == circle and ej == arrow:
                edge_types['o->'] += 1
            elif ei == arrow and ej == circle:
                edge_types['<-o'] += 1
            elif ei == circle and ej == circle:
                edge_types['o-o'] += 1
            elif ei == tail and ej == tail:
                edge_types['---'] += 1
            else:
                edge_types['?'] += 1

    parts = [f"{k}={v}" for k, v in edge_types.items() if v > 0]
    print(f"  PAG: {n_edges} edges ({', '.join(parts)})")


# =============================================================================
#  Convenience: matrix-to-Graph conversion
# =============================================================================

def pag_to_general_graph(pag, node_names=None):
    """
    Convert a graph_core PAG matrix to the local GeneralGraph.

    Parameters
    ----------
    pag : np.ndarray
        PAG matrix in graph_core encoding:
        0=NULL, -1=TAIL, 1=ARROW, 2=CIRCLE.
    node_names : list of str or None
        Node names.

    Returns
    -------
    GeneralGraph
    """
    p = pag.shape[0]
    if node_names is None:
        node_names = [f"X{i + 1}" for i in range(p)]

    nodes = [Node(name) for name in node_names]
    graph = GeneralGraph(nodes)

    endpoint_map = {
        0: Endpoint.NULL,
        Endpoint.TAIL.value: Endpoint.TAIL,
        Endpoint.ARROW.value: Endpoint.ARROW,
        Endpoint.CIRCLE.value: Endpoint.CIRCLE,
    }

    for i in range(p):
        for j in range(i + 1, p):
            mark_at_i = int(pag[j, i])  # endpoint at i
            mark_at_j = int(pag[i, j])  # endpoint at j
            if mark_at_i != 0 or mark_at_j != 0:
                edge = Edge(nodes[i], nodes[j],
                            endpoint_map.get(mark_at_i, Endpoint.NULL),
                            endpoint_map.get(mark_at_j, Endpoint.NULL))
                graph.add_edge(edge)

    return graph
