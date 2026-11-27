import threading
import time

import numpy as np

from ...background_knowledge import BKPhase, MatrixEncoding, apply_bk_checkpoint
from ..graph_core import CIT
from ..graph_core import cache as _cache_exports
from ..graph_core.cache import ci_cache_for
from ..graph_core.ci_batch import batch_execution_stats, supports_fisher_batch
from ..graph_core.conditioning import configure_conditioning
from ..graph_core.validation import effective_n_jobs
from ._counted_ci import counted_fisherz
from .ci_adapters import LegacyCITest, attach_counted_batch, attach_legacy_batch
from .fci_algo_paralel_cash import pc_cons_intern, pdsep, udag2pag
from .pag_init import to_circle_adjacency
from .skeleton import SkeletonDiscovery

BoundedCache = _cache_exports.BoundedCache


def _ci_cache_key(i: int, j: int, S):
    a, b = (int(i), int(j)) if i <= j else (int(j), int(i))
    if S is None:
        cond = ()
    elif isinstance(S, (set, frozenset, list, tuple, np.ndarray)):
        cond = tuple(sorted(int(x) for x in S))
    else:
        cond = (int(S),)
    return a, b, cond


def _resolve_cfci_backend(backend, n_vars, process_min_vars):
    requested = "auto" if backend in (None, "auto") else str(backend)
    if requested == "auto":
        # Work estimates in skeleton/PDSEP decide whether a process is useful.
        # The legacy variable-count knob remains accepted for API compatibility.
        effective = "processes"
    elif requested in {"threads", "processes"}:
        effective = requested
    else:
        msg = "CFCI backend must be 'auto', 'threads', or 'processes'"
        raise ValueError(msg)
    return requested, effective


def cfci_stable(
    cg,
    data,
    alpha=0.05,
    indep_test="fisherz",
    maj_rule=False,
    n_jobs=1,
    verbose=False,
    use_ci_cache=True,
    skeleton_backend=None,
    pdsep_backend=None,
    pdsep_process_min_tasks=256,
    process_min_vars=150,
    background_knowledge=None,
    show_progress: bool = False,
    max_cond_set_size=None,
):
    """Run stable conservative FCI with cached CI results and return its endpoint matrix."""
    configure_conditioning(cg, max_cond_set_size)
    total_t0 = time.perf_counter()
    _, p = data.shape
    n_jobs_eff = effective_n_jobs(n_jobs)
    skeleton_backend_requested, skeleton_backend_eff = _resolve_cfci_backend(
        skeleton_backend,
        p,
        process_min_vars,
    )
    pdsep_backend_requested, pdsep_backend_eff = _resolve_cfci_backend(
        pdsep_backend,
        p,
        process_min_vars,
    )
    use_process_skeleton = (
        n_jobs_eff != 1
        and indep_test == "fisherz"
        and skeleton_backend_eff == "processes"
    )
    use_process_pdsep = (
        n_jobs_eff != 1 and indep_test == "fisherz" and pdsep_backend_eff == "processes"
    )
    cit = (
        cg.test
        if isinstance(cg.test, CIT)
        and cg.test.method == indep_test
        and np.array_equal(cg.test.data, data)
        else CIT(data, method=indep_test)
    )
    cit.max_cond_set_size = max_cond_set_size
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
        "triple_check": 0,
        "pdsep": 0,
        "skeleton_backend": "processes" if use_process_skeleton else "threads",
        "triple_backend": "serial",
        "pdsep_backend": "processes" if use_process_pdsep else "threads",
        "skeleton_backend_requested": skeleton_backend_requested,
        "pdsep_backend_requested": pdsep_backend_requested,
        "process_min_vars": int(process_min_vars),
        "n_vars": int(p),
        "n_jobs": int(n_jobs_eff),
    }
    if use_ci_cache:
        ci_stats["ci_cache_hits"] = 0
        ci_stats["ci_cache_misses"] = 0
    phase = {"name": "skeleton"}
    ci_cache = ci_cache_for(cit) if use_ci_cache else None
    stats_lock = threading.Lock()

    def scalar_cit(i, j, S):
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

    counted_cit = counted_fisherz(
        scalar_cit, cit, ci_cache, ci_stats, phase, stats_lock
    )
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

    skeleton_t0 = time.perf_counter()
    skeleton = SkeletonDiscovery(
        cg,
        alpha,
        n_jobs=n_jobs_eff if native_fisher else 1,
        verbose=verbose,
        show_progress=show_progress,
        parallel_backend=skeleton_backend_requested if native_fisher else "auto",
        process_ci_info=process_ci_info if use_process_skeleton else None,
        protected_edges=protected_edges,
    )
    numeric_before = batch_execution_stats(cit)
    graphDict = skeleton.run_fci()
    numeric_after = batch_execution_stats(cit)
    observed = {
        name
        for name, count in getattr(cg, "fci_skeleton_execution_stats", {})
        .get("coarse_batches", {})
        .items()
        if count
    }
    observed.update(
        name
        for name, count in numeric_after.get("batches", {}).items()
        if count > numeric_before.get("batches", {}).get(name, 0)
    )
    ci_stats["skeleton_backend"] = (
        next(iter(observed))
        if len(observed) == 1
        else "hybrid"
        if observed
        else "serial"
    )
    graphDict["sepset"] = {key: set(value) for key, value in cg.separators.items()}
    ci_stats["time_skeleton_sec"] = time.perf_counter() - skeleton_t0
    if use_process_skeleton:
        skeleton_ci_tests = int(skeleton.ci_tests_performed)
        ci_stats["total"] += skeleton_ci_tests
        ci_stats["skeleton"] += skeleton_ci_tests
        if "ci_cache_misses" in ci_stats:
            ci_stats["ci_cache_misses"] += skeleton_ci_tests
        cg.set_ind_test(counted_cit)
    # Explicitly represent post-skeleton graph as o-o for CFCI phases.
    graphDict["sk"] = to_circle_adjacency(graphDict["sk"], MatrixEncoding.STANDARD)

    test_func_wrapper = LegacyCITest(counted_cit, max_cond_set_size)

    attach_legacy_batch(test_func_wrapper, counted_cit)
    suffStat = None

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
        unfVect=set(),
        n_jobs=n_jobs_eff,
        parallel_backend=pdsep_backend_requested if native_fisher else "auto",
        process_ci_info=process_ci_info if use_process_pdsep else None,
        process_min_tasks=pdsep_process_min_tasks,
        protected_edges=protected_edges,
    )
    ci_stats["time_pdsep_sec"] = time.perf_counter() - pdsep_t0

    # Recompute conservative triples after
    # Possible-D-Sep has updated both the graph and the separation sets.
    phase["name"] = "triple_check"
    triple_t0 = time.perf_counter()
    post_pdsep_graph = {
        "sk": to_circle_adjacency(pdsep_res["G"], MatrixEncoding.STANDARD),
        "sepset": pdsep_res["sepset"],
        "pMax": pdsep_res["pMax"],
        "unfTriples": set(),
    }
    pc_cons_intern(
        post_pdsep_graph,
        suffStat,
        alpha,
        test_func_wrapper,
        version_unf=(1, 1),
        maj_rule=maj_rule,
        verbose=verbose,
        n_jobs=n_jobs_eff,
        parallel_backend=pdsep_backend_requested,
    )
    ci_stats["time_triple_check_sec"] = time.perf_counter() - triple_t0
    ci_stats["ambiguous_triples"] = len(post_pdsep_graph["unfTriples"])
    triple_stats = post_pdsep_graph.get("triple_stats", {})
    ci_stats["triple_backend"] = triple_stats.get("backend", ci_stats["triple_backend"])
    ci_stats["triple_tasks"] = int(triple_stats.get("triples", 0))
    ci_stats["triple_pair_tasks"] = int(triple_stats.get("pair_tasks", 0))
    ci_stats["triple_process_tasks"] = (
        ci_stats["triple_pair_tasks"]
        if ci_stats["triple_backend"] == "processes"
        else 0
    )
    ci_stats["triple_n_jobs"] = int(triple_stats.get("n_jobs", n_jobs_eff))

    orient_t0 = time.perf_counter()
    pag = udag2pag(
        pdsep_res["G"],
        post_pdsep_graph["sepset"],
        p,
        unfVect=post_pdsep_graph["unfTriples"],
        background_knowledge=background_knowledge,
        node_names=[node.name for node in cg.nodes],
        # CFCI and FCI+ leave the collider unoriented when the pair has no
        # recorded separating sets. Keep that behavior here.
        require_recorded_sepset=True,
        collider_triples={
            t
            for t, status in post_pdsep_graph.get("triple_classification", {}).items()
            if status == 1
        },
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
    ci_stats["pdsep_requested_backend"] = pdsep_stats.get(
        "requested_backend", pdsep_backend_eff
    )
    ci_stats["pdsep_rounds"] = int(pdsep_stats.get("rounds", 0))
    ci_stats["pdsep_process_rounds"] = int(pdsep_stats.get("process_rounds", 0))
    ci_stats["pdsep_thread_rounds"] = int(pdsep_stats.get("thread_rounds", 0))
    ci_stats["pdsep_process_ci_tests"] = int(pdsep_stats.get("process_ci_tests", 0))
    ci_stats["pdsep_thread_ci_tests"] = int(pdsep_stats.get("thread_ci_tests", 0))
    ci_stats["pdsep_process_min_tasks"] = int(
        pdsep_stats.get("process_min_tasks", pdsep_process_min_tasks)
    )
    ci_stats["pdsep_tasks_built"] = int(pdsep_stats.get("tasks_built", 0))
    ci_stats["time_pdsep_task_build_sec"] = float(
        pdsep_stats.get("task_build_time", 0.0)
    )
    ci_stats["time_pdsep_ci_eval_sec"] = float(pdsep_stats.get("ci_eval_time", 0.0))
    ci_stats["time_pdsep_apply_sec"] = float(pdsep_stats.get("apply_time", 0.0))
    if ci_stats["pdsep_process_ci_tests"] and not supports_fisher_batch(
        test_func_wrapper
    ):
        ci_stats["total"] += ci_stats["pdsep_process_ci_tests"]
        ci_stats["pdsep"] += ci_stats["pdsep_process_ci_tests"]
        if "ci_cache_misses" in ci_stats:
            ci_stats["ci_cache_misses"] += ci_stats["pdsep_process_ci_tests"]
    if ci_cache is not None:
        ci_stats["ci_cache_size"] = len(ci_cache)
    ci_stats["time_total_sec"] = time.perf_counter() - total_t0
    cg.ci_stats = ci_stats
    return np.array(pag, dtype=int)
