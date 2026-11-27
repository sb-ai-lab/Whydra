"""Coarse PC jobs preserve unions and report actual cache work."""

from itertools import combinations

import numpy as np
import pytest

from whydra.algorithms.graph_core import CIT, CausalGraph
from whydra.algorithms.standalone.skeleton import (
    SkeletonDiscovery,
    _pc_fisher_edge_chunk,
    _pc_weighted_edge_chunks,
)


def _data():
    data = np.random.default_rng(913).normal(size=(500, 6))
    data[:, 3] = data[:, 2]
    data[:, 0] += data[:, 2]
    data[:, 1] += data[:, 2]
    return data


@pytest.mark.parametrize(("capacity", "seeded"), [(0, False), (1, False), (100, True)])
def test_compact_worker_matches_directional_scalar_union(capacity, seeded):
    data = _data()
    cit = CIT(data, cache_size=capacity)
    neighbors = ((1, 2, 3, 4), (0, 2, 3, 5), (), (), (), ())
    seed = {(0, 1, (2,)): cit(0, 1, (2,))} if seeded else {}
    expected = set()
    found = False
    scalar = CIT(data, cache_size=capacity)
    for x, y, pool in ((0, 1, (2, 3, 4)), (1, 0, (2, 3, 5))):
        for condition in combinations(pool, 1):
            if scalar(x, y, condition) > 0.05:
                found = True
                expected.update(condition)
    reductions, misses, hits, updates, errors, discarded = _pc_fisher_edge_chunk(
        [(0, 1)], neighbors, cit.corr_matrix, cit.n_samples, 1, 0.05, capacity, seed
    )
    assert errors == []
    assert reductions == ([(0, 1, tuple(sorted(expected)))] if found else [])
    assert misses == (6 if not capacity else 3 if seeded else 4)
    assert hits == 6 - misses
    assert len(updates) <= capacity
    assert discarded == (misses - len(updates) if capacity else 0)


def test_compact_coordinator_preserves_bounded_cache_counts_and_unions():
    data = np.random.default_rng(914).normal(size=(300, 6))

    def discovery(jobs):
        cg = CausalGraph(6, [str(index) for index in range(6)])
        cg.set_ind_test(CIT(data, cache_size=2))
        cg.test(0, 1)
        return SkeletonDiscovery(cg, n_jobs=jobs)

    parallel, scalar = discovery(2), discovery(1)
    actual = parallel._process_depth_pc_processes(0, None)
    expected = scalar._process_depth_batch(0, None)
    assert set(actual) == set(expected)
    np.testing.assert_array_equal(parallel.cg.sepset, scalar.cg.sepset)
    assert parallel.cg.test.cache_misses == scalar.cg.test.cache_misses == 15
    assert parallel.cg.test.cache_hits == scalar.cg.test.cache_hits == 16
    assert parallel.cg.test.cache_evictions == 13
    assert len(parallel.cg.test._cache) == 2
    execution = parallel.cg.pc_execution_stats
    assert execution["requested_backend"] == "auto"
    selected = execution["process_policy"][-1]["selected_jobs"]
    actual_backend = "processes" if selected > 1 else "serial"
    assert execution["coarse_batches"][actual_backend] == 1
    assert execution["coarse_batches"]["threads"] == 0
    assert execution["cache_seed_entries"] == 1
    assert (
        0
        < execution["cache_updates_returned"]
        <= 2 * execution["coarse_tasks"][actual_backend]
    )
    assert execution["cache_publication_seconds"] >= 0


def test_compact_phase_defers_to_single_flight_for_active_ci():
    cg = CausalGraph(6, [str(index) for index in range(6)])
    cg.set_ind_test(CIT(np.random.default_rng(915).normal(size=(100, 6))))
    cg.test._inflight[(0, 1, ())] = None
    assert SkeletonDiscovery(cg, n_jobs=2)._process_depth_pc_processes(0, None) is None
    assert cg.test.cache_misses == cg.test.cache_hits == 0


def test_compact_chunks_cover_each_active_unordered_pair_once():
    neighbors = ((1, 2, 3), (0, 2), (0, 1, 3), (0, 2), ())
    chunks = _pc_weighted_edge_chunks(neighbors, 1, 4)
    pairs = [pair for chunk in chunks for pair in chunk]
    assert sorted(pairs) == [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]
    assert all(chunk == sorted(chunk) for chunk in chunks)


def test_compact_path_keeps_cit_depth_guard_before_computation(monkeypatch):
    from whydra.algorithms.standalone import skeleton

    cg = CausalGraph(12, [str(index) for index in range(12)])
    cg.set_ind_test(
        CIT(np.random.default_rng(916).normal(size=(100, 12)), max_cond_set_size=1)
    )

    def unexpected_worker(*args, **kwargs):
        raise AssertionError("CI depth validation must run before compact dispatch")

    monkeypatch.setattr(skeleton, "_pc_fisher_edge_chunk", unexpected_worker)
    with pytest.raises(ValueError, match="max_cond_set_size"):
        SkeletonDiscovery(cg, n_jobs=2)._process_depth(2, None)
    assert cg.test.cache_misses == cg.test.cache_hits == 0


def test_compact_path_leaves_mismatched_graph_indices_to_cit_validation():
    cg = CausalGraph(12, [str(index) for index in range(12)])
    cg.set_ind_test(CIT(np.random.default_rng(917).normal(size=(100, 8))))
    assert SkeletonDiscovery(cg, n_jobs=2)._process_depth_pc_processes(0, None) is None


def test_compact_path_leaves_invalid_degrees_of_freedom_to_cit_validation():
    cg = CausalGraph(12, [str(index) for index in range(12)])
    cg.set_ind_test(CIT(np.random.default_rng(919).normal(size=(4, 12))))
    assert SkeletonDiscovery(cg, n_jobs=2)._process_depth_pc_processes(2, None) is None


def test_explicit_thread_backend_never_dispatches_compact_processes(monkeypatch):
    cg = CausalGraph(65, [str(index) for index in range(65)])
    cg.set_ind_test(CIT(np.random.default_rng(920).normal(size=(100, 65))))
    discovery = SkeletonDiscovery(cg, n_jobs=2, parallel_backend="threads")

    def unexpected_process(*args, **kwargs):
        raise AssertionError("Explicit threads must not dispatch process jobs")

    monkeypatch.setattr(discovery, "_process_depth_pc_processes", unexpected_process)
    monkeypatch.setattr(discovery, "_process_depth_batch", lambda depth, pbar: [])
    assert discovery._process_depth(0, None) == []


@pytest.mark.parametrize("algorithm_name", ["StandalonePCStable", "ParallelPCStable"])
def test_public_pc_forwards_explicit_threads_without_process_jobs(
    monkeypatch, algorithm_name
):
    from whydra.algorithms import wrappers
    from whydra.algorithms.standalone import parallel_pc_stable, skeleton

    def unexpected_process(*args, **kwargs):
        raise AssertionError(
            "Explicit PC threads must never construct a process executor"
        )

    monkeypatch.setattr(skeleton, "Parallel", unexpected_process)
    monkeypatch.setattr(parallel_pc_stable, "Parallel", unexpected_process)
    data = np.random.default_rng(921).normal(size=(300, 12))
    algorithm = getattr(wrappers, algorithm_name)
    graph = algorithm(n_jobs=2, skeleton_backend="threads").run(data)
    reference = wrappers.StandalonePCStable(n_jobs=1).run(data)
    np.testing.assert_array_equal(graph.graph, reference.graph)
    assert graph.metadata["pc_execution"]["requested_backend"] == "threads"
    assert graph.metadata["pc_execution"]["coarse_tasks"]["processes"] == 0


def test_generic_pc_records_actual_thread_work_and_protected_sepsets():
    cg = CausalGraph(4, [str(index) for index in range(4)])
    cg.set_ind_test(lambda x, y, conditions: 1.0)
    discovery = SkeletonDiscovery(
        cg, n_jobs=2, parallel_backend="threads", protected_edges=[(0, 1)]
    )
    discovery.run()
    stats = cg.pc_execution_stats
    assert stats["coarse_batches"]["threads"] > 0
    assert stats["coarse_tasks"]["processes"] == 0
    assert cg.G.graph[0, 1] != 0
    assert cg.sepset[0, 1] is cg.sepset[1, 0] is None
    assert cg.sepset[0, 2] == [()]


def test_pc_rejects_unknown_backend_before_work():
    cg = CausalGraph(3, ["a", "b", "c"])
    with pytest.raises(ValueError, match="skeleton backend"):
        SkeletonDiscovery(cg, parallel_backend="typo")


@pytest.mark.parametrize("entrypoint", ["skeleton_discovery", "pc_stable"])
@pytest.mark.parametrize("backend_options", [{}, {"skeleton_backend": "auto"}])
def test_parallel_pc_auto_keeps_custom_ci_in_parent(
    monkeypatch, entrypoint, backend_options
):
    import os

    from whydra.algorithms.standalone import parallel_pc_stable, skeleton

    def unexpected_executor(*args, **kwargs):
        raise AssertionError("Automatic custom PC must not create an executor")

    monkeypatch.setattr(parallel_pc_stable, "Parallel", unexpected_executor)
    monkeypatch.setattr(skeleton, "Parallel", unexpected_executor)
    monkeypatch.setattr(skeleton, "ThreadPoolExecutor", unexpected_executor)
    calls = []

    def custom_ci(x, y, conditions):
        calls.append((os.getpid(), x, y, tuple(conditions)))
        return 1.0

    def graph():
        cg = CausalGraph(4, [str(index) for index in range(4)])
        cg.set_ind_test(custom_ci)
        return cg

    reference = graph()
    SkeletonDiscovery(reference, n_jobs=1).run()
    expected_calls = calls.copy()
    calls.clear()
    actual = graph()
    if entrypoint == "skeleton_discovery":
        parallel_pc_stable.skeleton_discovery(
            actual, 0.05, True, False, False, 2, **backend_options
        )
    else:
        parallel_pc_stable.pc_stable(actual, n_jobs=2, **backend_options)
    assert calls == expected_calls
    assert len(calls) == 12
    assert {call[0] for call in calls} == {os.getpid()}
    np.testing.assert_array_equal(actual.G.graph, reference.G.graph)
    assert actual.pc_execution_stats["requested_backend"] == "auto"
    assert actual.pc_execution_stats["coarse_tasks"]["processes"] == 0


@pytest.mark.parametrize("backend_options", [{}, {"skeleton_backend": "auto"}])
def test_parallel_pc_auto_keeps_categorical_ci_serial(monkeypatch, backend_options):
    from whydra.algorithms.graph_core import process_execution
    from whydra.algorithms.standalone import parallel_pc_stable, skeleton
    from whydra.algorithms.wrappers import ParallelPCStable, StandalonePCStable

    def unexpected_executor(*args, **kwargs):
        raise AssertionError("Automatic categorical PC must not dispatch process jobs")

    monkeypatch.setattr(parallel_pc_stable, "Parallel", unexpected_executor)
    monkeypatch.setattr(skeleton, "Parallel", unexpected_executor)
    monkeypatch.setattr(skeleton, "ThreadPoolExecutor", unexpected_executor)
    monkeypatch.setattr(process_execution, "process_map", unexpected_executor)
    data = np.random.default_rng(922).integers(0, 3, size=(300, 6))
    actual = ParallelPCStable(indep_test="chisq", n_jobs=2, **backend_options).run(data)
    reference = StandalonePCStable(indep_test="chisq", n_jobs=1).run(data)
    np.testing.assert_array_equal(actual.graph, reference.graph)
    assert actual.metadata["pc_execution"]["requested_backend"] == "auto"
    assert actual.metadata["pc_execution"]["coarse_tasks"]["processes"] == 0


@pytest.mark.parametrize("capacity", [0, 2])
def test_full_pc_compact_activation_matches_fast_sequential_graph(capacity):
    from whydra.algorithms.wrappers import StandalonePCStable

    data = np.random.default_rng(918).normal(size=(500, 65))
    for column in range(1, data.shape[1]):
        data[:, column] += 0.7 * data[:, column - 1]
    sequential = StandalonePCStable(n_jobs=1, ci_cache_size=capacity).run(data)
    parallel = StandalonePCStable(n_jobs=4, ci_cache_size=capacity).run(data)
    np.testing.assert_array_equal(parallel.graph, sequential.graph)
    assert 0 in parallel.metadata["pc_process_batches"]["depths"]
    if not capacity:
        assert (
            parallel.metadata["ci_computations"]
            == sequential.metadata["ci_computations"]
        )
