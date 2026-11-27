import threading
from typing import Any

import numpy as np

from ...background_knowledge import BKPhase, MatrixEncoding, apply_bk_checkpoint
from ..graph_core import CIT
from ..graph_core import cache as _cache_exports
from ..graph_core.cache import ci_cache_for
from ..graph_core.ci_batch import evaluate_ci_batch, supports_fisher_batch
from ..graph_core.conditioning import (
    configure_conditioning,
)
from . import pag_rules as _pag_rules_exports
from ._counted_ci import counted_fisherz
from .ci_adapters import LegacyCITest, attach_counted_batch, attach_legacy_batch
from .fci_search import (
    classify_neighbors,
    record_triple,
    refine_rows,
    unshielded_triples,
)
from .pag_init import to_circle_adjacency
from .pag_rules import qreach, udag2pag
from .skeleton import SkeletonDiscovery

BoundedCache = _cache_exports.BoundedCache
powerset = _pag_rules_exports.powerset


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
    """Classify an unshielded triple using its neighbors and separating sets."""

    def evaluate(conditions):
        if hasattr(indepTest, "batch"):
            return evaluate_ci_batch(
                indepTest, [(a, c, s) for s in conditions], n_jobs=n_jobs
            )
        return (indepTest(suffStat, a, c, list(s)) for s in conditions)

    return classify_neighbors(
        a, b, c, nbrsA, nbrsC, sepsetA, sepsetC, indepTest, alpha, maj_rule, evaluate
    )


_BUILTIN_CHECK_TRIPLE = checkTriple


def pc_cons_intern(
    graphDict,
    suffstat,
    alpha,
    indepTest,
    version_unf=(None, None),
    maj_rule=True,
    verbose=False,
    n_jobs=1,
):
    """Classify unshielded triples and record separation evidence in the graph."""
    if supports_fisher_batch(indepTest) and checkTriple is _BUILTIN_CHECK_TRIPLE:
        # The native route shares pair evidence across middle vertices. Custom
        # callbacks retain this module's sequential evaluation/commit order.
        from .fci_algo_paralel_cash import pc_cons_intern as grouped_triples

        return grouped_triples(
            graphDict,
            suffstat,
            alpha,
            indepTest,
            version_unf,
            maj_rule,
            verbose,
            n_jobs,
        )

    matrix = graphDict["sk"]
    neighbors = [np.flatnonzero(column) for column in matrix.T]
    separators = graphDict["sepset"]
    for triple in unshielded_triples(matrix, callback_order=True):
        a, b, c = triple
        result = checkTriple(
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
            maj_rule=maj_rule,
            n_jobs=n_jobs,
        )
        record_triple(graphDict, triple, result, classify_all=True)
    return graphDict


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


def _ci_cache_key(i: int, j: int, S):
    a, b = (int(i), int(j)) if i <= j else (int(j), int(i))
    if S is None:
        cond = ()
    elif isinstance(S, (set, frozenset, list, tuple, np.ndarray)):
        cond = tuple(sorted(int(x) for x in S))
    else:
        cond = (int(S),)
    return a, b, cond


def cfci_stable(
    cg,
    data,
    alpha=0.05,
    indep_test="fisherz",
    maj_rule=False,
    n_jobs=1,
    verbose=False,
    use_ci_cache=True,
    background_knowledge=None,
    show_progress: bool = False,
    max_cond_set_size=None,
):
    """Run stable conservative FCI and return its endpoint matrix."""
    configure_conditioning(cg, max_cond_set_size)
    cit = (
        cg.test
        if isinstance(cg.test, CIT)
        and cg.test.method == indep_test
        and np.array_equal(cg.test.data, data)
        else CIT(data, method=indep_test)
    )
    cit.max_cond_set_size = max_cond_set_size
    ci_stats: dict[str, Any] = {
        "total": 0,
        "skeleton": 0,
        "triple_check": 0,
        "pdsep": 0,
    }
    # As in cfci_algo_paralel_cash, the lock lets parallel workers share
    # the CI cache. Expose hit/miss counts only when the cache is enabled.
    ci_cache = ci_cache_for(cit) if use_ci_cache else None
    if ci_cache is not None:
        ci_stats["ci_cache_hits"] = 0
        ci_stats["ci_cache_misses"] = 0
    cache_lock = threading.Lock()
    phase = {"name": "skeleton"}

    def scalar_cit(i, j, S):
        with cache_lock:
            ci_stats["total"] += 1
            ci_stats[phase["name"]] += 1
        if ci_cache is None:
            return cit(i, j, S)

        key = _ci_cache_key(i, j, S)
        with cache_lock:
            if key in ci_cache:
                ci_stats["ci_cache_hits"] += 1
                return ci_cache[key]
            ci_stats["ci_cache_misses"] += 1

        pval = cit(i, j, list(key[2]))
        with cache_lock:
            ci_cache[key] = pval
        return pval

    counted_cit = counted_fisherz(
        scalar_cit, cit, ci_cache, ci_stats, phase, cache_lock
    )
    attach_counted_batch(counted_cit, cit, ci_stats, phase, ci_cache, cache_lock)
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

    n_jobs_eff = 1 if n_jobs in (None, 0) else int(n_jobs)
    graphDict = SkeletonDiscovery(
        cg,
        alpha,
        n_jobs=n_jobs_eff,
        verbose=verbose,
        show_progress=show_progress,
        protected_edges=protected_edges,
    ).run_fci()
    graphDict["sepset"] = {key: set(value) for key, value in cg.separators.items()}
    graphDict["sk"] = to_circle_adjacency(graphDict["sk"], MatrixEncoding.STANDARD)
    _, p = data.shape

    test_func_wrapper = LegacyCITest(counted_cit, max_cond_set_size)

    attach_legacy_batch(test_func_wrapper, counted_cit)
    suffStat = None

    phase["name"] = "pdsep"
    pdsep_res = pdsep(
        graphDict,
        suffStat,
        test_func_wrapper,
        p,
        graphDict["sepset"],
        alpha,
        graphDict["pMax"],
        unfVect=set(),
        protected_edges=protected_edges,
        n_jobs=n_jobs_eff,
    )

    # Conservative FCI classifies triples on the final skeleton and sepsets.
    phase["name"] = "triple_check"
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
    )

    rule_stats = {}
    pag = udag2pag(
        pdsep_res["G"],
        post_pdsep_graph["sepset"],
        p,
        unfVect=post_pdsep_graph["unfTriples"],
        rule_stats=rule_stats,
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
    ci_stats["ambiguous_triples"] = len(post_pdsep_graph["unfTriples"])
    triple_stats = post_pdsep_graph.get("triple_stats", {})
    ci_stats["triple_backend"] = triple_stats.get("backend", "serial")
    ci_stats["triple_pair_tasks"] = int(triple_stats.get("pair_tasks", 0))
    ci_stats["triple_process_tasks"] = (
        ci_stats["triple_pair_tasks"]
        if ci_stats["triple_backend"] == "processes"
        else 0
    )
    for rule, changes in rule_stats.items():
        ci_stats[f"orient_{rule.lower()}_changes"] = int(changes)
    if ci_cache is not None:
        ci_stats["ci_cache_size"] = len(ci_cache)
    cg.ci_stats = ci_stats
    return np.array(pag, dtype=int)
