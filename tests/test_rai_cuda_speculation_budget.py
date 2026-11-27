"""Limits on speculative CI requests in CUDA RAI."""

from itertools import combinations
from math import comb
from types import SimpleNamespace

import numpy as np
import pytest

from whydra.algorithms.graph_core import cpu_cuda_rai_bridge as cuda_rai
from whydra.algorithms.standalone.rai_algo import _PDAG


def _complete_graph(size, cls=_PDAG):
    graph = cls(set(range(size)))
    graph.create_complete_graph()
    return graph


@pytest.mark.parametrize(("n", "k"), [(0, 0), (3, 4), (7, 0), (7, 2), (100, 50)])
def test_combination_counter_stops_at_the_remaining_budget(n, k):
    expected = comb(n, k) if k <= n else 0
    for limit in (0, 1, 10, 100):
        assert cuda_rai._bounded_combinations(n, k, limit) == min(expected, limit + 1)


def test_budget_accumulates_both_pools_and_all_connected_pairs():
    graph = _complete_graph(5)
    pairs = [(0, 1), (2, 3)]
    # Each endpoint has three potential parents; each pool contributes C(3,2).
    assert cuda_rai._speculation_within_budget(graph, pairs, 2, 12)
    assert not cuda_rai._speculation_within_budget(graph, pairs, 2, 11)
    assert cuda_rai._speculation_within_budget(graph, pairs, 0, 2)
    assert not cuda_rai._speculation_within_budget(graph, pairs, 0, 1)


def test_budget_uses_live_graph_when_native_replay_left_bitsets_stale():
    class StaleAdjacencyGraph(_PDAG):
        def is_connected(self, a, b):
            return True

    graph = _complete_graph(4, StaleAdjacencyGraph)
    _PDAG.delete_edge(graph, 0, 1)  # Native replay similarly bypasses _FastPDAG.
    assert graph.is_connected(0, 1)
    assert not _PDAG.is_connected(graph, 0, 1)
    assert cuda_rai._speculation_within_budget(graph, [(0, 1)], 0, 0)


@pytest.mark.parametrize("method", ["_speculate", "_speculate_fast"])
def test_oversized_phase_returns_no_memo_before_native_or_cuda_work(
    method, monkeypatch
):
    learner = object.__new__(cuda_rai.RAIStableCudaLearner)
    learner.graph = _complete_graph(100)

    def unexpected_work(*args, **kwargs):
        raise AssertionError("An over-budget phase must skip native enumeration")

    monkeypatch.setattr(cuda_rai, "_upstream", unexpected_work)
    # A single pair at this order already has far more than 8M combinations.
    assert getattr(learner, method)([(0, 1)], 50) is None


@pytest.mark.parametrize("method", ["_speculate", "_speculate_fast"])
def test_many_small_pairs_are_bounded_before_native_enumeration(method, monkeypatch):
    learner = object.__new__(cuda_rai.RAIStableCudaLearner)
    learner.graph = _complete_graph(5)
    monkeypatch.setattr(cuda_rai, "_SPEC_MAX_TESTS", 9)

    def unexpected_work(*args, **kwargs):
        raise AssertionError("The aggregate phase budget must be checked first")

    monkeypatch.setattr(cuda_rai, "_upstream", unexpected_work)
    assert getattr(learner, method)(list(combinations(range(5), 2)), 0) is None


@pytest.mark.parametrize("method", ["_speculate", "_speculate_fast"])
def test_admitted_phase_delegates_to_original_cuda_scheduler(method, monkeypatch):
    calls = []
    result = object()

    class OriginalLearner(cuda_rai._CPUReferenceLearner):
        def _speculate(self, pairs, order):
            calls.append((pairs, order))
            return result

        _speculate_fast = _speculate

    monkeypatch.setattr(cuda_rai, "_INTEGRATED_CLASS", None)
    monkeypatch.setattr(
        cuda_rai,
        "_upstream",
        lambda: SimpleNamespace(RAIStableCudaLearner=OriginalLearner),
    )
    learner = object.__new__(cuda_rai._integrated_class())
    learner.graph = _complete_graph(4)
    assert getattr(learner, method)([(0, 1)], 1) is result
    assert calls == [([(0, 1)], 1)]


def test_lazy_subclass_initializes_cpu_contracts_without_original_gpu_constructor(
    monkeypatch,
):
    class OriginalLearner(cuda_rai._CPUReferenceLearner):
        def __init__(self, *args, **kwargs):
            raise AssertionError("The upstream batcher factory must not run")

        def learn_structure(self):
            return "upstream native loop"

    monkeypatch.setattr(cuda_rai, "_INTEGRATED_CLASS", None)
    monkeypatch.setattr(
        cuda_rai,
        "_upstream",
        lambda: SimpleNamespace(RAIStableCudaLearner=OriginalLearner, _FastPDAG=_PDAG),
    )
    gpu = object()
    calls = []

    def batcher(cit, dtype=None):
        calls.append(cit)
        return gpu

    monkeypatch.setattr(cuda_rai, "_build_gpu_batcher", batcher)
    data = np.random.default_rng(43).normal(size=(100, 3))
    learner = cuda_rai.RAIStableCudaLearner(data, 0.05, "fisherz", ["A", "B", "C"])
    assert isinstance(learner, OriginalLearner)
    assert isinstance(learner, cuda_rai.RAIStableCudaLearner)
    assert learner.learn_structure() == "upstream native loop"
    assert type(learner).learn_structure is OriginalLearner.learn_structure
    assert calls == [learner.ci_test]
    assert learner.oracle.tester is learner.ci_test
    assert learner._gpu is gpu
    assert learner.stats["gpu_tests"] == 0
