"""Stable process levels preserve full separator evidence and directed maxima."""

import numpy as np
import pytest

from tests._pdsep_cases import reachability_cases
from whydra.algorithms.graph_core import CIT, CausalGraph
from whydra.algorithms.standalone.skeleton import SkeletonDiscovery


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("capacity", [0, 2, 1000])
def test_compact_fci_level_matches_ordered_batch(capacity, jobs):
    data = np.random.default_rng(1801).normal(size=(300, 12))
    data[:, 1] += 0.6 * data[:, 0]

    def make(n_jobs):
        cg = CausalGraph(12, [str(i) for i in range(12)])
        cg.set_ind_test(CIT(data, cache_size=capacity))
        return SkeletonDiscovery(
            cg, n_jobs=n_jobs, parallel_backend="processes", protected_edges=[(0, 2)]
        )

    actual, expected = make(jobs), make(1)
    actual.cg.pMax = np.full((12, 12), -np.inf)
    expected.cg.pMax = np.full((12, 12), -np.inf)
    removals, separators = actual._process_depth_fci_native(1, None, {})
    reference_removals, reference_separators = expected._process_depth_batch(
        1, None, {}
    )
    assert set(removals) == set(reference_removals)
    assert separators == reference_separators
    assert actual.cg.separators == expected.cg.separators
    assert actual.cg.separator_members == expected.cg.separator_members
    np.testing.assert_array_equal(actual.cg.sepset, expected.cg.sepset)
    # A bounded reference cache can recompute the reverse orientation after
    # eviction; inverse-based p-values can differ by a few ULPs. The graph and
    # full separator evidence above must still match exactly.
    np.testing.assert_allclose(actual.cg.pMax, expected.cg.pMax, rtol=1e-12, atol=1e-13)
    assert actual.ci_tests_performed == expected.ci_tests_performed
    assert len(actual.cg.test._cache) <= capacity
    assert (0, 2) not in separators
    assert (
        actual.cg.fci_skeleton_execution_stats["coarse_batches"][
            "processes" if jobs > 1 else "serial"
        ]
        == 1
    )


def test_seeded_cit_hit_promotes_outer_cache_without_another_computation():
    from whydra.algorithms.standalone.fisher_skeleton import fisher_level_chunk

    corr = np.eye(3)
    key = (0, 1, (2,))
    result = fisher_level_chunk(
        [(0, 1)],
        [(1, 2), (0, 2), (0, 1)],
        corr,
        100,
        1,
        0.05,
        {key: (0.9, False)},
        True,
        True,
        8,
    )
    assert result["requests"] == 2
    assert result["computed"] == 0
    assert result["cit_hits"] == result["outer_hits"] == 1
    assert result["promotions"] == [(key, 0.9)]


@pytest.mark.parametrize("case", reachability_cases(), ids=lambda case: case.name)
def test_reachability_on_marked_graph(case):
    from whydra.algorithms.standalone.fci_algo_v2 import qreach as legacy_qreach
    from whydra.algorithms.standalone.fci_search import orient_colliders
    from whydra.algorithms.standalone.pag_rules import PAG_ARROW, PAG_CIRCLE, qreach

    graph = np.where(
        case.skeleton == 2, PAG_ARROW, np.where(case.skeleton, PAG_CIRCLE, 0)
    )
    orient_colliders(graph, case.separators, case.ambiguous, missing=())
    expected_marks = np.where(
        case.marked == 2, PAG_ARROW, np.where(case.marked, PAG_CIRCLE, 0)
    )
    np.testing.assert_array_equal(graph, expected_marks)
    for source, expected in enumerate(case.reachable):
        assert qreach(source, graph) == list(expected)
        assert legacy_qreach(source, case.marked) == list(expected)


def test_reachability_across_machine_word_boundary():
    from whydra.algorithms.standalone.pag_rules import PAG_ARROW, PAG_CIRCLE, qreach

    marked = np.zeros((67, 67), dtype=int)
    marked[0, 33] = marked[65, 33] = PAG_ARROW
    marked[33, 0] = marked[33, 65] = PAG_CIRCLE
    assert qreach(0, marked) == [33, 65]
    assert qreach(65, marked) == [0, 33]
    assert qreach(66, marked) == []


def _worker_identity(value):
    import os

    if value == "fail":
        raise ValueError("intentional worker failure")
    return os.getpid(), value


def test_shared_process_pool_reuses_workers_and_recovers_after_failure():
    import os

    from whydra.algorithms.graph_core.process_execution import process_map

    first = process_map(_worker_identity, [(i,) for i in range(6)], n_jobs=2)
    second = process_map(_worker_identity, [(i,) for i in range(6)], n_jobs=2)
    assert [value for _, value in first] == list(range(6))
    assert [value for _, value in second] == list(range(6))
    assert all(pid != os.getpid() for pid, _ in first + second)
    assert {pid for pid, _ in first} & {pid for pid, _ in second}
    with pytest.raises(ValueError, match="intentional worker failure"):
        process_map(_worker_identity, [("fail",)], n_jobs=2)
    assert process_map(_worker_identity, [(42,)], n_jobs=2)[0][1] == 42
