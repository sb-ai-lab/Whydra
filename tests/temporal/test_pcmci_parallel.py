"""Parallel PCMCI-family searches retain sequential evidence and search order."""

import os

import numpy as np
import pytest
from joblib import effective_n_jobs

from whydra.algorithms.temporal import PCMCI, PCMCIPlus
from whydra.algorithms.temporal._data import coerce_dataset, make_frame
from whydra.algorithms.temporal._engine import parallel as parallel_module
from whydra.algorithms.temporal._engine._joint_ci import NativeParCorr
from whydra.algorithms.temporal._engine.parallel import (
    ParallelPCMCIPipeline,
    ParallelPCMCIPlusPipeline,
)
from whydra.algorithms.temporal._engine.pcmci import PCMCIPipeline
from whydra.algorithms.temporal._engine.pcmci_plus import PCMCIPlusPipeline
from whydra.background_knowledge import BackgroundKnowledge, BKPhase

PIPELINES = [
    (PCMCIPipeline, ParallelPCMCIPipeline, "run_pcmci"),
    (PCMCIPlusPipeline, ParallelPCMCIPlusPipeline, "run_pcmciplus"),
]


def series(n=160, seed=74):
    data = np.random.default_rng(seed).normal(size=(n, 3))
    for index in range(2, n):
        data[index, 0] += 0.3 * data[index - 1, 0]
        data[index, 1] += 0.8 * data[index - 1, 0]
        data[index, 2] += 0.6 * data[index, 1] + 0.5 * data[index - 2, 1]
    return data


def assert_evidence_equal(actual, expected):
    np.testing.assert_array_equal(actual["graph"], expected["graph"])
    for key in ("p_matrix", "val_matrix"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
    assert actual.get("sepsets") == expected.get("sepsets")
    assert actual.get("ambiguous_triples") == expected.get("ambiguous_triples")


@pytest.mark.parametrize("model_class", [PCMCI, PCMCIPlus])
@pytest.mark.parametrize("fdr_method", ["none", "fdr_bh"])
def test_public_parallel_search_matches_serial_with_missing_data_and_bk(
    model_class, fdr_method
):
    data = series()
    data[12, 0] = data[31, 2] = np.nan
    models = []
    for jobs in (1, 2):
        knowledge = (
            BackgroundKnowledge()
            .require("X3[t-1]", "X1", phases=BKPhase.PRE_SEARCH)
            .forbid("X1[t-2]", "X3", phases=BKPhase.PRE_SEARCH)
        )
        model = model_class(
            n_jobs=jobs,
            tau_min=0,
            tau_max=2,
            alpha=0.03,
            max_conds_dim=2,
            max_combinations=2,
            max_conds_py=2,
            max_conds_px=1,
            fdr_method=fdr_method,
            background_knowledge=knowledge,
        )
        model.run(data)
        models.append(model)
    serial, parallel = models
    assert_evidence_equal(parallel.results_, serial.results_)
    np.testing.assert_array_equal(parallel.graph_.graph, serial.graph_.graph)
    np.testing.assert_array_equal(
        parallel.lagged_graph_.graph, serial.lagged_graph_.graph
    )
    assert parallel.parents_ == serial.parents_
    assert parallel.engine_.diagnostics == serial.engine_.diagnostics
    assert parallel.lag_graph_[2, 0, 1] == "-->"
    assert parallel.lag_graph_[0, 2, 2] == ""
    assert parallel.engine_._parallel_worker_pids
    assert os.getpid() not in parallel.engine_._parallel_worker_pids
    assert parallel.engine_._parallel_executor is None


def event_signature(stage, X, Y, Z, alpha):
    return stage, tuple(X), tuple(Y), tuple(Z or []), alpha


@pytest.mark.parametrize(("serial_class", "parallel_class", "method"), PIPELINES)
def test_parallel_search_replays_ci_in_order_and_merges_cache(
    serial_class, parallel_class, method
):
    frame = make_frame([coerce_dataset(series(n=100))], 2)
    serial = serial_class(dataframe=frame, cond_ind_test=NativeParCorr())
    parallel = parallel_class(dataframe=frame, cond_ind_test=NativeParCorr(), n_jobs=2)
    expected_events, actual_events = [], []
    original_test = serial.cond_ind_test.run_test

    def observed_test(
        X, Y, Z=None, tau_max=0, cut_off="2xtau_max", alpha_or_thres=None
    ):
        expected_events.append(event_signature(serial.mode, X, Y, Z, alpha_or_thres))
        return original_test(
            X, Y, Z=Z, tau_max=tau_max, cut_off=cut_off, alpha_or_thres=alpha_or_thres
        )

    serial.cond_ind_test.run_test = observed_test
    parallel._ci_observer = actual_events.append
    options = {"tau_min": 0, "tau_max": 2, "pc_alpha": 0.1, "max_conds_dim": 2}
    expected = getattr(serial, method)(**options)
    actual = getattr(parallel, method)(**options)
    assert_evidence_equal(actual, expected)
    worker_stages = {"pc1", "mci", "pcmci_plus_skeleton"}
    expected_events = [event for event in expected_events if event[0] in worker_stages]
    replayed_events = [
        event_signature(
            event["stage"], event["X"], event["Y"], event["Z"], event["alpha_or_thres"]
        )
        for event in actual_events
    ]
    assert replayed_events == expected_events
    assert {"pc1", "mci" if method == "run_pcmci" else "pcmci_plus_skeleton"} <= {
        event["stage"] for event in actual_events
    }
    for cache_name in ("cached_ci_results", "ci_results"):
        expected_cache = getattr(serial.cond_ind_test, cache_name)
        actual_cache = getattr(parallel.cond_ind_test, cache_name)
        assert actual_cache.keys() == expected_cache.keys()
        for key, value in expected_cache.items():
            np.testing.assert_allclose(actual_cache[key], value, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("serial_class", "parallel_class", "method"), PIPELINES)
def test_parallel_alpha_selection_restores_selected_parent_and_graph_state(
    serial_class, parallel_class, method, monkeypatch
):
    original_parallel = parallel_module.Parallel
    pools = []

    def observed_parallel(*args, **kwargs):
        pool = original_parallel(*args, **kwargs)
        pools.append(pool)
        return pool

    monkeypatch.setattr(parallel_module, "Parallel", observed_parallel)
    frame = make_frame([coerce_dataset(series(n=90))], 1)
    serial = serial_class(dataframe=frame, cond_ind_test=NativeParCorr())
    parallel = parallel_class(dataframe=frame, cond_ind_test=NativeParCorr(), n_jobs=2)
    options = {
        "tau_min": 1,
        "tau_max": 1,
        "pc_alpha": [0.01, 0.1],
        "max_conds_dim": 1,
        "fdr_method": "fdr_bh",
    }
    if method == "run_pcmci":
        options["save_iterations"] = True
    expected = getattr(serial, method)(**options)
    actual = getattr(parallel, method)(**options)
    assert_evidence_equal(actual, expected)
    assert parallel.all_parents == serial.all_parents
    assert parallel.iterations == serial.iterations
    if method == "run_pcmciplus":
        assert actual["optimal_alpha"] == expected["optimal_alpha"]
        assert actual["alpha_scores"].keys() == expected["alpha_scores"].keys()
        np.testing.assert_allclose(
            list(actual["alpha_scores"].values()),
            list(expected["alpha_scores"].values()),
            rtol=1e-12,
        )
        assert parallel.all_lagged_parents == serial.all_lagged_parents
    assert parallel._parallel_worker_pids
    assert parallel._parallel_executor is None
    # Parent screening, final tests and recursive alpha candidates share a pool.
    assert len(pools) == 1


@pytest.mark.parametrize(("serial_class", "parallel_class", "method"), PIPELINES)
def test_single_job_parallel_engine_falls_back_to_serial(
    serial_class, parallel_class, method
):
    frame = make_frame([coerce_dataset(series(n=60))], 1)
    serial = serial_class(dataframe=frame, cond_ind_test=NativeParCorr())
    parallel = parallel_class(dataframe=frame, cond_ind_test=NativeParCorr(), n_jobs=1)
    options = {"tau_min": 0, "tau_max": 1, "pc_alpha": 0.05, "max_conds_dim": 1}
    assert_evidence_equal(
        getattr(parallel, method)(**options), getattr(serial, method)(**options)
    )
    assert not parallel._parallel_worker_pids
    assert parallel._parallel_executor is None


@pytest.mark.parametrize("model_class", [PCMCI, PCMCIPlus])
def test_all_cpus_and_run_override_do_not_leak_parallel_state(model_class, monkeypatch):
    # Retain the -1 contract without launching a process per host CPU in CI.
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "2")
    data = series(n=70)
    model = model_class(n_jobs=1, max_conds_dim=1)
    expected = model_class(max_conds_dim=1)
    expected.run(data)
    model.run(data, n_jobs=-1)
    assert model.engine_.n_jobs == -1
    assert_evidence_equal(model.results_, expected.results_)
    if effective_n_jobs(-1) > 1:
        assert model.engine_._parallel_worker_pids
        assert os.getpid() not in model.engine_._parallel_worker_pids
    model.run(data)
    assert_evidence_equal(model.results_, expected.results_)
    assert model.n_jobs == 1
    assert not hasattr(model.engine_, "_parallel_executor")


@pytest.mark.parametrize("model_class", [PCMCI, PCMCIPlus])
def test_zero_lag_only_parallel_search_still_tests_contemporaneous_edges(model_class):
    data = np.random.default_rng(95).normal(size=(120, 3))
    data[:, 1] += 1.3 * data[:, 0]
    models = []
    for jobs in (1, 2):
        model = model_class(n_jobs=jobs, tau_min=0, tau_max=0, alpha=0.01)
        model.run(data)
        models.append(model)
    serial, parallel = models
    assert_evidence_equal(parallel.results_, serial.results_)
    assert parallel.lag_graph_.shape == (3, 3, 1)
    assert parallel.lag_graph_[0, 1, 0] != ""
    assert parallel.p_matrix_[0, 1, 0] < 0.01
    assert parallel.parents_ == serial.parents_ == {0: [], 1: [], 2: []}
    assert parallel.engine_._parallel_worker_pids
    assert os.getpid() not in parallel.engine_._parallel_worker_pids


class ConfidenceCI:
    """Picklable analytic CI with directional evidence and known intervals."""

    significance = "analytic"
    confidence = True

    def __init__(self):
        self.interval = np.empty(2)

    def set_dataframe(self, dataframe):
        self.dataframe = dataframe

    def run_test(self, X, Y, Z=None, tau_max=0, alpha_or_thres=None, **kwargs):
        value = 0.2 + 0.1 * X[0][0] - 0.05 * Y[0][0]
        probability = 0.03 + 0.01 * Y[0][0]
        return value, probability, probability <= alpha_or_thres

    def get_confidence(self, X, Y, Z=None, tau_max=0):
        value = self.run_test(X, Y, Z=Z, tau_max=tau_max, alpha_or_thres=0.05)[0]
        self.interval[:] = value - 0.1, value + 0.1
        return self.interval


@pytest.mark.parametrize("val_only", [False, True])
def test_parallel_mci_retains_confidence_intervals_and_val_only(val_only):
    frame = make_frame([coerce_dataset(series(n=30))], 2)
    serial = PCMCIPipeline(dataframe=frame, cond_ind_test=ConfidenceCI())
    parallel = ParallelPCMCIPipeline(
        dataframe=frame, cond_ind_test=ConfidenceCI(), n_jobs=2
    )
    options = {
        "tau_min": 0,
        "tau_max": 2,
        "val_only": val_only,
        "parents": {0: [(0, -1)], 1: [(2, -1)], 2: [(1, -2)]},
        "max_conds_py": 1,
        "max_conds_px": 1,
        "link_assumptions": {
            0: {(1, 0): "o?o"},
            1: {(0, 0): "o?o", (2, -2): "-?>"},
            2: {(1, -1): "-->"},
        },
    }
    expected = serial.run_mci(**options)
    actual = parallel.run_mci(**options)
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    assert actual["conf_matrix"] is not None
    np.testing.assert_allclose(actual["conf_matrix"][0, 1, 0], [0.05, 0.25])
    np.testing.assert_allclose(actual["conf_matrix"][1, 0, 0], [0.05, 0.25])
    np.testing.assert_allclose(actual["conf_matrix"][2, 1, 2], [0.25, 0.45])
    np.testing.assert_allclose(actual["conf_matrix"][1, 2, 1], [0.1, 0.3])
    assert parallel._parallel_worker_pids
    assert os.getpid() not in parallel._parallel_worker_pids
    assert parallel._parallel_executor is None


class RetriableCI(ConfidenceCI):
    confidence = False

    def __init__(self):
        super().__init__()
        self.fail = True

    def run_test(self, X, Y, Z=None, tau_max=0, alpha_or_thres=None, **kwargs):
        if self.fail:
            raise ValueError(f"controlled worker failure pid={os.getpid()}")
        return super().run_test(
            X, Y, Z=Z, tau_max=tau_max, alpha_or_thres=alpha_or_thres, **kwargs
        )


def test_mci_worker_failure_releases_pool_and_same_engine_can_retry():
    frame = make_frame([coerce_dataset(series(n=30))], 1)
    search = ParallelPCMCIPipeline(
        dataframe=frame, cond_ind_test=RetriableCI(), n_jobs=2
    )
    with pytest.raises(ValueError, match="controlled worker failure") as failure:
        search.run_mci(tau_min=1, tau_max=1)
    assert f"pid={os.getpid()}" not in str(failure.value)
    assert search._parallel_executor is None
    search.cond_ind_test.fail = False
    actual = search.run_mci(tau_min=1, tau_max=1)
    assert np.count_nonzero(actual["graph"]) == 9
    assert search._parallel_worker_pids
    assert os.getpid() not in search._parallel_worker_pids
    assert search._parallel_executor is None
