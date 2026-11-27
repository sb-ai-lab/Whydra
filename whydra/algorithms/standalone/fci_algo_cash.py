import threading
from typing import Any

import numpy as np

from ...background_knowledge import BKPhase, MatrixEncoding, apply_bk_checkpoint
from ..graph_core import CIT

# Import the shared test class from the neighboring module.
from ..graph_core import cache as _cache_exports
from ..graph_core.cache import ci_cache_for
from ..graph_core.conditioning import configure_conditioning
from . import pag_rules as _pag_rules_exports
from .ci_adapters import LegacyCITest, attach_counted_batch, attach_legacy_batch
from .fci_search import refine_rows
from .pag_init import to_circle_adjacency
from .pag_rules import qreach, udag2pag

# --- Helper functions ---
from .skeleton import SkeletonDiscovery

BoundedCache = _cache_exports.BoundedCache
powerset = _pag_rules_exports.powerset


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
    protected_edges=(),
    n_jobs=1,
):
    """Refine the skeleton using Possible-D-SEP conditioning sets."""
    return refine_rows(
        skel["sk"],
        indepTest,
        suffStat,
        sepSet,
        alpha,
        pMax,
        m_max,
        unfVect,
        qreach,
        protected=protected_edges,
        jobs=n_jobs,
    )


def fci_stable(
    cg,
    data,
    alpha=0.05,
    indep_test="fisherz",
    n_jobs=1,
    verbose=False,
    use_ci_cache=True,
    background_knowledge=None,
    show_progress: bool = False,
    max_cond_set_size=None,
):
    """Run stable FCI with cached CI results and return its endpoint matrix."""
    # SkeletonDiscovery uses cg.ci_test(...), so CI test must be set before run_fci().
    configure_conditioning(cg, max_cond_set_size)
    cit = (
        cg.test
        if isinstance(cg.test, CIT)
        and cg.test.method == indep_test
        and np.array_equal(cg.test.data, data)
        else CIT(data, method=indep_test)
    )
    cit.max_cond_set_size = max_cond_set_size
    ci_stats: dict[str, Any] = {"total": 0, "skeleton": 0, "pdsep": 0}
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
            if key in ci_cache:
                ci_stats["ci_cache_hits"] += 1
                return ci_cache[key]
            ci_stats["ci_cache_misses"] += 1
        pval = cit(i, j, list(key[2]))
        with stats_lock:
            ci_cache[key] = pval
        return pval

    attach_counted_batch(counted_cit, cit, ci_stats, phase, ci_cache, stats_lock)
    cg.set_ind_test(counted_cit)

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

    graphDict = SkeletonDiscovery(
        cg,
        alpha,
        n_jobs=n_jobs,
        verbose=verbose,
        show_progress=show_progress,
        protected_edges=protected_edges,
    ).run_fci()
    graphDict["sepset"] = {key: set(value) for key, value in cg.separators.items()}
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
    pdsep_res = pdsep(
        graphDict,
        suffStat,
        test_func_wrapper,
        p,
        graphDict["sepset"],
        alpha,
        graphDict["pMax"],
        unfVect=graphDict["unfTriples"],
        protected_edges=protected_edges,
        n_jobs=n_jobs,
    )

    rule_stats = {}
    pag = udag2pag(
        pdsep_res["G"],
        pdsep_res["sepset"],
        p,
        unfVect=graphDict["unfTriples"],
        rule_stats=rule_stats,
        background_knowledge=background_knowledge,
        node_names=node_names,
    )
    pdsep_stats = pdsep_res.get("pdsep_stats", {})
    ci_stats.setdefault("process_policy", []).extend(
        pdsep_stats.get("process_policy", ())
    )
    del ci_stats["process_policy"][:-64]
    ci_stats["pdsep_pairs_scanned"] = int(pdsep_stats.get("pairs_scanned", 0))
    ci_stats["pdsep_pairs_with_diffset"] = int(pdsep_stats.get("pairs_with_diffset", 0))
    ci_stats["pdsep_ci_tests"] = int(pdsep_stats.get("ci_tests", 0))
    for name in (
        "process_rows",
        "process_pairs",
        "worker_ci_computations",
        "worker_cache_hits",
        "worker_speculative_tests",
    ):
        ci_stats[f"pdsep_{name}"] = int(pdsep_stats.get(name, 0))
    for k, v in rule_stats.items():
        ci_stats[f"orient_{k.lower()}_changes"] = int(v)
    if ci_cache is not None:
        ci_stats["ci_cache_size"] = len(ci_cache)
    cg.ci_stats = ci_stats

    return pag
