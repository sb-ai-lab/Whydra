"""Compact cache tails preserve the order of misses and outer promotions."""

import numpy as np

from whydra.algorithms.graph_core import CIT, CausalGraph
from whydra.algorithms.graph_core.cache import BoundedCache
from whydra.algorithms.standalone.fisher_skeleton import fisher_level_chunk
from whydra.algorithms.standalone.skeleton import SkeletonDiscovery


def test_late_cit_hit_retains_outer_promotion_after_earlier_pending_miss():
    missed, promoted = (0, 1, (2,)), (0, 1, (3,))
    result = fisher_level_chunk(
        [(0, 1)],
        [(1, 2, 3), (0,), (0,), (0,)],
        np.eye(4),
        100,
        1,
        0.05,
        {promoted: (0.9, False)},
        True,
        True,
        1,
    )
    assert result["updates"] == [(missed, 1.0)]
    assert result["outer_updates"] == [(promoted, 0.9)]
    assert result["discarded"] == 0
    assert result["outer_discarded"] == 1
    assert result["computed"] == 1
    assert result["cit_hits"] == 1


def test_coordinator_publishes_separate_cit_and_outer_tails():
    data = np.random.default_rng(196).normal(size=(100, 4))
    cit = CIT(data, cache_size=1)
    cit.corr_matrix = np.eye(4)
    outer = BoundedCache(1)
    stats = {"total": 0, "skeleton": 0, "ci_cache_hits": 0, "ci_cache_misses": 0}

    def counted(x, y, conditions):
        return cit(x, y, conditions)

    counted._batch_cit = cit
    counted._pdsep_batch_context = (cit, outer, stats, {"name": "skeleton"}, None)
    graph = CausalGraph(4, [str(i) for i in range(4)])
    graph.set_ind_test(counted)
    graph.G.graph[:] = 0
    graph.G.graph[0, 1] = graph.G.graph[1, 0] = -1
    skeleton = SkeletonDiscovery(graph, n_jobs=1)
    skeleton.G1[0, 2:] = skeleton.G1[2:, 0] = -1
    graph.pMax = np.full((4, 4), -np.inf)
    promoted, missed = (0, 1, (3,)), (0, 1, (2,))
    cit._cache[promoted] = 0.9
    skeleton._process_depth_fci_native(1, None, {})
    assert list(cit._cache.items()) == [(missed, 1.0)]
    assert list(outer.items()) == [(promoted, 0.9)]
    assert outer.evictions == 1
    assert cit.cache_evictions == 1
