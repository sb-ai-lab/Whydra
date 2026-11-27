"""Depth caps are search restrictions, never fabricated CI answers."""

import warnings

import numpy as np
import pytest

from whydra.algorithms import wrappers as W
from whydra.algorithms.graph_core import CIT, CausalGraph

ALGORITHMS = [
    W.StandalonePCStable,
    W.ParallelPCStable,
    W.StandaloneFCIStable,
    W.StandaloneFCIStableCached,
    W.StandaloneFCIStableParallelCached,
    W.StandaloneCFCIStable,
    W.StandaloneCFCIStableParallelCached,
    W.StandaloneRFCIStable,
    W.StandaloneGFCIStable,
    W.StandaloneFCIPlusStable,
    W.StandaloneFCIPlusStableParallelCached,
    W.StandaloneRAIStable,
]


def dependent_data():
    rng = np.random.default_rng(243)
    return rng.normal(size=(300, 6)) + rng.normal(size=(300, 1)) * 2


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("limit", [0, 1])
def test_public_cap_instruments_every_ci_call(algorithm, limit, monkeypatch):
    calls = []

    def oracle(self, x, y, condition_set=None):
        size = len(condition_set or ())
        assert size <= limit, (algorithm.__name__, x, y, condition_set)
        calls.append(size)
        # A dense dependent oracle forces all available skeleton orders.
        return 0.0

    monkeypatch.setattr(CIT, "__call__", oracle)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        graph = algorithm(max_cond_set_size=limit).run(dependent_data())
    assert calls
    assert max(calls) == limit
    assert graph.metadata["max_cond_set_size"] == limit
    assert graph.metadata["depth_limited"] is True


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("invalid", [-1, True, 1.5, "0", float("inf")])
def test_invalid_limits_fail_before_ci(algorithm, invalid):
    with pytest.raises(ValueError, match="max_cond_set_size"):
        algorithm(max_cond_set_size=invalid).run(dependent_data())


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_none_and_run_override_preserve_unrestricted_result(algorithm):
    data = dependent_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = algorithm().run(data)
        actual = algorithm(max_cond_set_size=0).run(data, max_cond_set_size=None)
    np.testing.assert_array_equal(actual.graph, expected.graph)
    assert actual.metadata["max_cond_set_size"] is None
    assert actual.metadata["depth_limited"] is False


@pytest.mark.parametrize(
    "algorithm", [W.StandalonePCStable, W.StandaloneFCIStable, W.StandaloneRAIStable]
)
def test_chain_requires_conditional_test_to_remove_false_adjacency(algorithm):
    rng = np.random.default_rng(11)
    x = rng.normal(size=5000)
    middle = x + rng.normal(size=len(x))
    y = middle + rng.normal(size=len(x))
    data = np.column_stack((x, middle, y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        marginal = algorithm(max_cond_set_size=0).run(data)
        conditional = algorithm(max_cond_set_size=1).run(data)
    assert marginal.graph[0, 2] != 0
    assert conditional.graph[0, 2] == 0


def graph_with_budget(limit, p=5):
    cg = CausalGraph(p, [str(i) for i in range(p)])
    cg.max_cond_set_size = limit
    return cg


@pytest.mark.parametrize("module_name", ["cfci_algo", "fci_algo_paralel_cash"])
def test_incomplete_conservative_search_is_ambiguous(module_name):
    from importlib import import_module

    module = import_module("whydra.algorithms.standalone." + module_name)
    calls = []

    def oracle(_stats, x, y, conditions):
        calls.append(tuple(conditions))
        return 1.0

    oracle.max_cond_set_size = 0
    result, separators = module.checkTriple(
        0, 1, 2, [1, 3], [1, 4], set(), set(), None, 0.05, oracle, maj_rule=True
    )
    assert calls
    assert all(not s for s in calls)
    assert result == 3
    assert separators["sepsetA"] == set()


@pytest.mark.parametrize(
    "module_name",
    ["fci_algo_cash", "cfci_algo", "fci_algo_paralel_cash", "fci_algo_v2"],
)
@pytest.mark.parametrize("limit", [0, 1, 2])
def test_pds_scheduler_bounds_nonlocal_candidates(module_name, limit, monkeypatch):
    from importlib import import_module

    module = import_module("whydra.algorithms.standalone." + module_name)
    if module_name == "fci_algo_v2":
        monkeypatch.setattr(module, "_possible_dsep", lambda *_: list(range(5)))
    else:
        monkeypatch.setattr(module, "qreach", lambda *_: list(range(5)))
    matrix = np.zeros((5, 5), dtype=int)
    for x, y in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        matrix[x, y] = matrix[y, x] = 2
    calls = []

    def oracle(_stats, x, y, conditions):
        assert len(conditions) <= limit
        calls.append(len(conditions))
        return 0.0

    oracle.max_cond_set_size = limit
    method = getattr(module, "pdsep", getattr(module, "pdsep_v2", None))
    options = {"use_pds_path": False} if module_name == "fci_algo_v2" else {}
    result = method(
        {"sk": matrix},
        None,
        oracle,
        5,
        {},
        0.05,
        np.full((5, 5), -np.inf),
        unfVect=set(),
        **options,
    )
    minimum = 2 if module_name == "fci_algo_v2" else 1
    assert not calls if limit < minimum else max(calls) == limit
    np.testing.assert_array_equal(result["G"] != 0, matrix != 0)
    assert result["sepset"] == {}  # no invented empty separators


@pytest.mark.parametrize("module_name", ["fci_plus_algo", "fci_plus_algo_paralel_cash"])
def test_fci_plus_augmentation_and_hierarchy_skip_oversized_tests(
    module_name, monkeypatch
):
    from importlib import import_module

    module = import_module("whydra.algorithms.standalone." + module_name)
    cg = graph_with_budget(0)
    cg.G.graph[:] = 0
    cg.G.graph[0, 2] = cg.G.graph[2, 0] = 2
    cg.sepset[0, 1] = cg.sepset[1, 0] = [()]
    calls = []
    cg.set_ind_test(lambda *args: calls.append(args) or 0.0)
    before = cg.G.graph.copy()
    module.AugmentGraph(cg, 5, 0.05)
    assert calls == []
    np.testing.assert_array_equal(cg.G.graph, before)
    cg.max_cond_set_size = 1
    monkeypatch.setattr(module, "_get_hie", lambda *_: set(range(5)))
    found, _ = module._dsep(cg, 0.05, 0, 1, {2}, {3}, 1, 5)
    # cap the closure, not just seed sizes
    assert not found
    assert calls == []


def test_rfci_incomplete_path_check_is_unknown():
    from whydra.algorithms.standalone import rfci_algo

    cg = graph_with_budget(0)
    cg.sepset[0, 2] = [3]
    cg.set_ind_test(lambda *_: 0.0)
    assert rfci_algo._check_edges_on_disc_path(cg, 0.05, [0, 1, 2]) is None


@pytest.mark.parametrize("module_name", ["fci_plus_algo", "fci_plus_algo_paralel_cash"])
def test_separator_minimization_rejects_an_oversized_unverified_witness(module_name):
    from importlib import import_module

    module = import_module("whydra.algorithms.standalone." + module_name)
    cg = graph_with_budget(0)
    cg.set_ind_test(lambda *_: 0.0)
    with pytest.raises(ValueError, match="max_cond_set_size"):
        module._min_dsep(cg, {2, 3}, 0, 1)
    assert cg.sepset[0, 1] is None


def test_cit_guard_precedes_cached_answers_and_does_not_fabricate_a_pvalue():
    cit = CIT(dependent_data())
    cit(0, 1, [2])
    before = (cit.cache_hits, cit.cache_misses)
    cit.max_cond_set_size = 0
    with pytest.raises(ValueError, match="max_cond_set_size"):
        cit(0, 1, [2])
    assert (cit.cache_hits, cit.cache_misses) == before


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_categorical_ci_obeys_zero_cap_with_two_workers(algorithm):
    data = np.random.default_rng(25).integers(0, 2, size=(300, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        graph = algorithm(
            indep_test="chisq", n_jobs=2, max_cond_set_size=np.int64(0)
        ).run(data)
    assert graph.metadata["max_cond_set_size"] == 0
    assert graph.metadata["depth_limited"]


@pytest.mark.parametrize("limit", [0, 1])
def test_gfci_legacy_depth_intersects_common_limit(limit, monkeypatch):
    calls = []
    monkeypatch.setattr(
        CIT, "__call__", lambda self, x, y, s=None: calls.append(len(s or ())) or 0.0
    )
    graph = W.StandaloneGFCIStable(
        depth=0, max_cond_set_size=limit, start_from_complete_graph=True
    ).run(dependent_data())
    assert calls
    assert max(calls) == 0
    assert graph.metadata["effective_max_cond_set_size"] == 0
    assert graph.metadata["depth_limited"] is True


PROCESS_ALGORITHMS = [
    (W.ParallelPCStable, {}),
    (
        W.StandaloneFCIStableParallelCached,
        {"skeleton_backend": "processes", "pdsep_backend": "processes"},
    ),
    (
        W.StandaloneCFCIStableParallelCached,
        {"skeleton_backend": "processes", "pdsep_backend": "processes"},
    ),
    (
        W.StandaloneFCIPlusStableParallelCached,
        {
            "skeleton_backend": "processes",
            "augment_backend": "processes",
            "dsep_backend": "processes",
            "process_min_tasks": 0,
        },
    ),
    (
        W.StandaloneRFCIStable,
        {"skeleton_backend": "processes", "orient_backend": "processes"},
    ),
    (
        W.StandaloneGFCIStable,
        {"ci_backend": "processes", "start_from_complete_graph": True},
    ),
]


@pytest.mark.parametrize(("algorithm", "options"), PROCESS_ALGORITHMS)
@pytest.mark.parametrize("limit", [0, 1])
def test_forced_process_backend_never_exceeds_statistical_degrees_of_freedom(
    algorithm, options, limit
):
    # One extra condition makes n-|S|-3 zero. Direct numeric process workers
    # bypass CIT, so this independently catches their oversized requests.
    data = np.random.default_rng(50).normal(size=(limit + 4, 6))
    graph = algorithm(alpha=0.999999, n_jobs=2, max_cond_set_size=limit, **options).run(
        data
    )
    assert np.count_nonzero(graph.graph) == 30
    assert graph.metadata["max_cond_set_size"] == limit


@pytest.mark.parametrize("limit", [0, 1])
def test_nonlocal_pds_process_workers_execute_only_bounded_orders(limit, monkeypatch):
    from whydra.algorithms.standalone import fci_algo_paralel_cash as module

    monkeypatch.setattr(module, "qreach", lambda *_: list(range(5)))
    matrix = np.zeros((5, 5), dtype=int)
    for x, y in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        matrix[x, y] = matrix[y, x] = 2

    def no_parent_ci(*_):
        pytest.fail("Expected numerical process execution")

    no_parent_ci.max_cond_set_size = limit
    result = module.pdsep(
        {"sk": matrix},
        None,
        no_parent_ci,
        5,
        {},
        0.999999,
        np.full((5, 5), -np.inf),
        unfVect=set(),
        n_jobs=2,
        parallel_backend="processes",
        process_min_tasks=0,
        process_ci_info={
            "method": "fisherz",
            "n_samples": limit + 4,
            "corr_matrix": np.eye(5) * 0.2 + 0.8,
        },
    )
    stats = result["pdsep_stats"]
    assert (stats["process_ci_tests"] > 0) == (limit > 0)
    assert stats["thread_ci_tests"] == 0
    np.testing.assert_array_equal(result["G"] != 0, matrix != 0)


@pytest.mark.parametrize("limit", [1, 3])
def test_fci_plus_process_hierarchy_caps_closure_before_dispatch(limit, monkeypatch):
    from whydra.algorithms.standalone import fci_plus_algo_paralel_cash as module

    cg = graph_with_budget(limit)
    cg._fci_plus_process_min_tasks = 0
    monkeypatch.setattr(module, "_get_hie", lambda *_: set(range(5)))
    recorded = []
    found, _ = module._dsep(
        cg,
        0.999999,
        0,
        1,
        {2},
        {3},
        1,
        5,
        n_jobs=2,
        parallel_backend="processes",
        record_process_tests=recorded.append,
        process_ci_info={
            "method": "fisherz",
            "n_samples": limit + 4,
            "corr_matrix": np.eye(5) * 0.2 + 0.8,
        },
    )
    assert not found
    assert bool(recorded) == (limit == 3)
    assert cg.sepset[0, 1] is None
