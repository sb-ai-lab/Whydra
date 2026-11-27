import threading
from collections import deque
from itertools import combinations, groupby

import numpy as np

from ...background_knowledge import BKPhase, MatrixEncoding, apply_bk_checkpoint
from ..graph_core import CIT, CausalGraph
from ..graph_core import cache as _cache_exports
from ..graph_core.cache import ci_cache_for
from ..graph_core.ci_batch import evaluate_ci_batch, supports_fisher_batch
from ..graph_core.conditioning import (
    bounded_size,
    conditioning_allowed,
    configure_conditioning,
)
from ..graph_core.endpoints import Endpoint
from ..graph_core.validation import effective_n_jobs

# FCI+ uses the shared PAG orientation rules.
from . import pag_rules as _pag_rules_exports
from ._counted_ci import counted_fisherz
from .ci_adapters import attach_counted_batch, attach_legacy_batch, ordered_ci_values
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
from .pag_rules import udag2pag
from .separators import sepset_to_pair_dict as _sepset_to_pair_dict
from .skeleton import SkeletonDiscovery

BoundedCache = _cache_exports.BoundedCache
PAG_ARROW = _pag_rules_exports.PAG_ARROW
PAG_CIRCLE = _pag_rules_exports.PAG_CIRCLE
PAG_TAIL = _pag_rules_exports.PAG_TAIL

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


def _ci_batch(cg, tasks, n_jobs):
    tasks = list(tasks)
    if supports_fisher_batch(cg.test):
        pvals = evaluate_ci_batch(cg.test, tasks, n_jobs=n_jobs, backend="processes")
        return [(x, y, S, pval) for (x, y, S), pval in zip(tasks, pvals, strict=False)]
    return [(x, y, S, cg.ci_test(x, y, S)) for x, y, S in tasks]


def AugmentGraph(cg: CausalGraph, p: int, alpha: float, n_jobs: int = 1):
    """Orient additional arrowheads using CI evidence from augmented separating sets."""
    if augment_native(cg, alpha, n_jobs, "auto", _sepset_vars):
        return cg.G.graph
    pag = cg.G.graph.copy()
    # Custom CI callbacks can fail or retain state. Keep the serial pair order.
    probes = augmentation_inputs(cg, p, _sepset_vars)
    for _, pair in groupby(probes, key=lambda item: item[0][:2]):
        queries, evidence = zip(*pair, strict=False)
        values = _ci_batch(cg, queries, n_jobs)
        set_arrowheads(pag, dependent_endpoints(values, evidence, alpha))
    cg.G.graph = pag
    return pag


# Lemma 4 restricts the search to candidates with the required bidirected pattern.
def _dsep_search(cg: CausalGraph, p: int) -> deque:
    return _get_pdseps(cg.G.graph.copy())


def _get_hie(cg, S, cache=None):
    return hierarchy_closure(cg, S, _sepset_vars, cache)


def _ordered_graph_ci(cg, x, y, condition_sets, n_jobs):
    """Keep graph-level guards while exposing a batch-capable CI callable."""

    def legacy(_suff_stat, a, b, conditions):
        return cg.ci_test(a, b, conditions)

    if hasattr(cg.test, "batch"):
        attach_legacy_batch(legacy, cg.test)
    return ordered_ci_values(legacy, None, x, y, condition_sets, n_jobs=n_jobs)


# Choose a valid separator of minimum cardinality, as in the parallel search.
def _min_dsep(cg: CausalGraph, S, i, j, alpha=0.05, n_jobs=1):
    if not conditioning_allowed(cg, set(S)):
        msg = "Separator witness exceeds max_cond_set_size"
        raise ValueError(msg)

    # cg.sepset stores a list of separating sets (tuples), following
    # SkeletonDiscovery._append_sepset. _sepset_to_pair_dict expects this format.
    # For example, [(3, 5)] records one two-node separator. A flat [3, 5]
    # can be read as two one-node separators by callers expecting this shape.
    def _store(nodes):
        cg.sepset[i][j] = cg.sepset[j][i] = [tuple(nodes)]
        hierarchy_changed(cg, ((i, j), (j, i)))
        return list(nodes)

    S = tuple(sorted(set(S)))
    if not S:
        return _store(())

    for size in range(1, bounded_size(cg, len(S)) + 1):
        for subset, pvalue in _ordered_graph_ci(
            cg, i, j, combinations(S, size), n_jobs
        ):
            if pvalue >= alpha:
                return _store(subset)

    return _store(S)


# Algorithm 2, lines 8–9, starts both n and m at one.
def _dsep(cg, alpha, x, y, base_x, base_y, k, p, n_jobs=1, protected_edges=()):
    protected_pairs = {frozenset((int(a), int(b))) for a, b in protected_edges}
    if frozenset((int(x), int(y))) in protected_pairs:
        return False, deque()
    max_n = bounded_size(cg, min(k, len(base_x)))
    max_m = bounded_size(cg, min(k, len(base_y)))
    hierarchy = hierarchy_cache(cg, _sepset_vars)
    for n in range(1, max_n + 1):
        for m in range(1, max_m + 1):

            def candidates(n, m):
                # Hierarchical closure only reads the current graph/sepsets.
                # They change after the first separator, when consumption stops.
                for z_x in combinations(base_x, n):
                    for z_y in combinations(base_y, m):
                        seed = frozenset((x, y, *z_x, *z_y))
                        Z = frozenset(_get_hie(cg, seed, hierarchy) - {x, y})
                        if conditioning_allowed(cg, Z):
                            yield Z

            for Z, p_value in _ordered_graph_ci(cg, x, y, candidates(n, m), n_jobs):
                if p_value >= alpha:
                    Z = _min_dsep(cg, Z, x, y, alpha=alpha, n_jobs=n_jobs)
                    # _min_dsep already records the minimum separator.
                    cg.G.graph[x][y] = NULL
                    cg.G.graph[y][x] = NULL
                    hierarchy_changed(cg, graph=True)
                    AugmentGraph(cg, p, alpha, n_jobs=n_jobs)
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
    background_knowledge=None,
    show_progress: bool = False,
    max_cond_set_size=None,
) -> np.ndarray:
    """Run stable FCI+ and return its endpoint matrix."""
    configure_conditioning(cg, max_cond_set_size)
    n_jobs_eff = effective_n_jobs(n_jobs)
    cit = (
        cg.test
        if isinstance(cg.test, CIT)
        and cg.test.method == indep_test
        and np.array_equal(cg.test.data, data)
        else CIT(data, method=indep_test)
    )
    cit.max_cond_set_size = max_cond_set_size
    ci_stats = {"total": 0, "skeleton": 0, "augment": 0, "dsep": 0}
    use_cache = bool(use_ci_cache)
    if use_cache:
        ci_stats["ci_cache_hits"] = 0
        ci_stats["ci_cache_misses"] = 0
    phase = {"name": "skeleton"}
    ci_cache = ci_cache_for(cit) if use_cache else None
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
    cg.set_ind_test(counted_cit)
    _n, p = data.shape
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

    native = supports_fisher_batch(cg.test)
    SkeletonDiscovery(
        cg,
        alpha,
        n_jobs=n_jobs_eff if native else 1,
        verbose=verbose,
        show_progress=show_progress,
        parallel_backend="auto",
        process_ci_info={
            "method": "fisherz",
            "corr_matrix": cit.corr_matrix,
            "n_samples": cit.n_samples,
        }
        if native
        else None,
        protected_edges=protected_edges,
    ).run_fci()

    phase["name"] = "augment"
    AugmentGraph(cg, p, alpha, n_jobs=n_jobs_eff)
    dsep_pos = _dsep_search(cg, p)

    phase["name"] = "dsep"
    while len(dsep_pos) > 0:
        x, y = dsep_pos.popleft()
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
            n_jobs=n_jobs_eff,
            protected_edges=protected_edges,
        )
        if found_dsep:
            dsep_pos = updated_dsep_pos

    pag = cg.G.graph.copy()

    # The graph and the PAG orientation rules use the same Endpoint codes.

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

    # No conversion is needed because udag2pag and FCI+ both use
    # causal-learn's encoding.
    cg.G.graph = pag
    if ci_cache is not None:
        ci_stats["ci_cache_size"] = len(ci_cache)
    cg.ci_stats = ci_stats
    return pag
