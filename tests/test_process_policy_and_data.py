"""Cost learning does no extra CI work; data revisions cannot cross fit leases."""

import os

import numpy as np
import pytest

from whydra.algorithms.graph_core import worker_data
from whydra.algorithms.graph_core.execution_policy import ProcessCostModel
from whydra.algorithms.graph_core.process_execution import (
    plan_process_work,
    process_map,
)


def test_cold_short_work_stays_local_but_warm_work_uses_processes():
    model = ProcessCostModel(cold_seconds=2)
    cold = model.choose(10000, 4, family="skeleton:1")
    warm = model.choose(
        10000,
        4,
        family="skeleton:1",
        pool_states={2: {"warm": True}, 4: {"warm": True}},
    )
    assert cold.jobs == 1
    assert warm.jobs == 4


def test_observed_slow_four_worker_pool_selects_two_without_extra_work():
    model = ProcessCostModel(cold_seconds=2)
    states = {2: {"warm": True}, 4: {"warm": True}}
    for jobs, seconds in [(1, 0.100), (2, 0.060), (4, 0.140)]:
        plan = model.choose(
            10000, jobs, family="augment", backend="processes", pool_states=states
        )
        model.observe(plan, seconds)
    chosen = model.choose(10000, 4, family="augment", pool_states=states)
    assert chosen.jobs == 2
    assert chosen.record["reason"] == "predicted_process_gain"


def test_cold_samples_do_not_poison_warm_throughput():
    model = ProcessCostModel(cold_seconds=2)
    cold = model.choose(10000, 4, family="augment", backend="processes")
    model.observe(cold, 5.0)
    warm = model.choose(10000, 4, family="augment", pool_states={4: {"warm": True}})
    assert warm.jobs == 4
    assert cold.record["estimated_startup_seconds"] > 4


def test_explicit_processes_remains_explicit_and_counts_are_bounded():
    model = ProcessCostModel()
    assert model.choose(1000, 4, family="x", backend="processes").jobs == 4
    assert model.choose(1000, 4, family="x", backend="processes", tasks=2).jobs == 2
    assert (
        model.choose(1000, 4, family="x", backend="processes", min_work=2000).jobs == 1
    )


def test_learning_and_diagnostics_are_bounded():
    model = ProcessCostModel(max_entries=3)
    stats = {}
    for i in range(70):
        plan = model.choose(i + 1, 1, family=str(i), stats=stats)
        model.observe(plan, 0.001)
    assert len(model._samples) == len(model._serial_rates) == 3
    assert len(stats["process_policy"]) == 64
    assert stats["process_policy"][-1]["observed_seconds"] == 0.001


@pytest.mark.parametrize("work", [-1, float("inf"), float("nan")])
def test_invalid_cost_estimates_rejected(work):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ProcessCostModel().choose(work, 4, family="x")


def _inspect_shared(array, index):
    return os.getpid(), id(array), bool(array.flags.writeable), float(array.flat[index])


def test_workers_reuse_readonly_attachment_and_mutation_gets_a_new_revision():
    if not worker_data._SHARING_SUPPORTED:
        pytest.skip("safe untracked shared memory unavailable on this Python")
    array = np.arange(16.0, dtype=np.float64).reshape(4, 4)
    plan = plan_process_work(100, 2, family="test_shared", backend="processes")
    first = process_map(
        _inspect_shared, [(array, 3)] * 8, n_jobs=2, plan=plan, shared_arg_indices=(0,)
    )
    second = process_map(
        _inspect_shared, [(array, 3)] * 8, n_jobs=2, shared_arg_indices=(0,)
    )
    assert all(
        pid != os.getpid() and not writable and value == 3.0
        for pid, _identity, writable, value in first + second
    )
    for pid in {r[0] for r in first + second}:
        assert len({r[1] for r in first + second if r[0] == pid}) == 1
    assert plan.record["shared_arrays"] == 1
    assert plan.record["shared_bytes"] == array.nbytes
    array.flat[3] = 99.0
    third = process_map(
        _inspect_shared, [(array, 3)] * 4, n_jobs=2, shared_arg_indices=(0,)
    )
    assert all(row[-1] == 99.0 for row in third)


def test_active_dataset_revision_is_not_unlinked_by_another_fit():
    if not worker_data._SHARING_SUPPORTED:
        pytest.skip("safe untracked shared memory unavailable on this Python")
    array = np.eye(3)
    old = worker_data.acquire_array(array)
    try:
        array[0, 0] = 7
        new = worker_data.acquire_array(array)
        try:
            assert old["ref"].name != new["ref"].name
            old_memory = worker_data._open_memory(name=old["ref"].name)
            try:
                values = np.ndarray(
                    old["ref"].shape, dtype=array.dtype, buffer=old_memory.buf
                )
                assert values[0, 0] == 1
                del values
            finally:
                old_memory.close()
        finally:
            worker_data.release_array(new)
    finally:
        worker_data.release_array(old)


def test_local_work_never_allocates_shared_memory(monkeypatch):
    def fail(_array):
        raise AssertionError("serial work must not register a dataset")

    monkeypatch.setattr(
        "whydra.algorithms.graph_core.process_execution.acquire_array", fail
    )
    values = process_map(np.sum, [(np.eye(2),)], n_jobs=1, shared_arg_indices=(0,))
    assert values == [2.0]


def test_structured_arrays_use_portable_serialization_fallback():
    assert worker_data.acquire_array(np.zeros(3, dtype=[("value", "f8")])) is None


def test_active_leases_obey_shared_registry_budget(monkeypatch):
    if not worker_data._SHARING_SUPPORTED:
        pytest.skip("safe untracked shared memory unavailable on this Python")
    worker_data.clear_shared_data()
    monkeypatch.setattr(worker_data, "_MAX_ENTRIES", 1)
    first, second = np.eye(2), np.eye(3)
    lease = worker_data.acquire_array(first)
    try:
        assert lease is not None
        assert worker_data.acquire_array(second) is None
    finally:
        worker_data.release_array(lease)
    following = worker_data.acquire_array(second)
    try:
        assert following is not None
    finally:
        worker_data.release_array(following)


def test_failed_submission_releases_dataset_lease(monkeypatch):
    from contextlib import contextmanager

    from whydra.algorithms.graph_core import process_execution

    class BrokenSubmit:
        def submit(self, *args):
            raise RuntimeError("submission failed")

    @contextmanager
    def pool(_jobs):
        yield BrokenSubmit()

    monkeypatch.setattr(process_execution, "_process_pool", pool)
    array = np.eye(2)
    with pytest.raises(RuntimeError, match="submission failed"):
        process_map(np.sum, [(array,)], n_jobs=2, shared_arg_indices=(0,))
    with worker_data._LOCK:
        assert all(
            entry["users"] == 0
            for entry in worker_data._ENTRIES.values()
            if entry["array"]() is array
        )
