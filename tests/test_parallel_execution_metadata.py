"""Public diagnostics describe dispatched work independently of n_jobs."""

import json

import numpy as np
import pytest

from whydra.algorithms import wrappers as w


@pytest.mark.parametrize(
    "cls",
    [w.StandalonePCStable, w.ParallelPCStable, w.StandaloneFCIStableParallelCached],
)
def test_small_public_fits_report_serial_fallback_and_serializable_metadata(cls):
    data = np.random.default_rng(721).normal(size=(200, 5))
    kwargs = {"skeleton_backend": "threads"}
    if cls is w.StandaloneFCIStableParallelCached:
        kwargs["pdsep_backend"] = "threads"
    one = cls(n_jobs=1, **kwargs).run(data)
    many = cls(n_jobs=4, **kwargs).run(data)
    np.testing.assert_array_equal(one.graph, many.graph)
    assert many.metadata["effective_n_jobs"] == 4
    assert many.metadata["execution"]["configured_n_jobs"] == 4
    assert many.metadata["execution"]["actual_backend"] == "serial"
    stats = many.metadata["ci_batch_execution"]
    assert stats["batches"]["serial"] > 0
    assert stats["tasks"]["threads"] == stats["tasks"]["processes"] == 0
    assert stats["fallback_reasons"]
    json.dumps(many.metadata, allow_nan=False)


def test_fresh_public_fits_do_not_share_or_mutate_execution_counters():
    data = np.random.default_rng(722).normal(size=(200, 5))
    algorithm = w.StandalonePCStable(n_jobs=1)
    first = algorithm.run(data)
    snapshot = json.dumps(first.metadata, sort_keys=True)
    second = algorithm.run(data)
    assert json.dumps(first.metadata, sort_keys=True) == snapshot
    for field in ("batches", "tasks", "queries", "queries_by_backend"):
        assert (
            first.metadata["ci_batch_execution"][field]
            == second.metadata["ci_batch_execution"][field]
        )
    assert first.metadata["pc_execution"] is not second.metadata["pc_execution"]


def test_requested_pdsep_backend_alone_does_not_claim_parallel_execution():
    # GFCI retains its legacy requested-backend field. Observed dispatch must
    # still describe a one-worker fit correctly.
    data = np.random.default_rng(723).normal(size=(200, 5))
    graph = w.StandaloneGFCIStable(n_jobs=1).run(data)
    assert graph.metadata["execution"]["actual_backend"] == "serial"
    assert graph.metadata["execution"]["scope"] == "observed_ci_dispatch"
