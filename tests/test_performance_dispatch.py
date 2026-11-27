"""Execution-policy changes must retain live PDS results and expose dispatch."""

from itertools import combinations

import numpy as np
import pytest

from tests.test_fci_v2_pdsep import (
    ALPHA,
    SKELETON_P,
    _empty_sepset,
    _make_oracle,
    _skeleton_matrix,
)
from whydra.algorithms.standalone import fci_algo_v2 as fci
from whydra.algorithms.wrappers import StandaloneFCIStable


@pytest.mark.parametrize("maximum", [1, 2, 3, float("inf")])
def test_pds_work_bound_matches_enumerated_eligible_subsets(maximum):
    context = {
        "pds_a": [2, 3, 4, 5],
        "pds_b": [2, 4, 5],
        "adj_a": [2, 3],
        "adj_b": [3, 4],
    }
    actual = 0
    for name in ("pds_a", "pds_b"):
        pool = context[name]
        upper = len(pool) if maximum == float("inf") else min(len(pool), maximum)
        for size in range(2, upper + 1):
            actual += sum(
                not (set(s) <= set(context["adj_a"]) or set(s) <= set(context["adj_b"]))
                for s in combinations(pool, size)
            )
    assert fci._context_test_bound(context, maximum, 10**9) == actual
    if actual:
        assert fci._context_test_bound(context, maximum, 1) >= 1


def _run(jobs, threshold):
    independent = {(0, 1): {frozenset({3, 4})}, (2, 4): {frozenset({1, 5})}}
    return fci.pdsep_v2(
        {"sk": _skeleton_matrix()},
        None,
        _make_oracle(independent, []),
        SKELETON_P,
        _empty_sepset(),
        ALPHA,
        np.full((SKELETON_P, SKELETON_P), -np.inf),
        n_jobs=jobs,
        parallel_min_tests=threshold,
    )


def _same_result(first, second):
    np.testing.assert_array_equal(first["G"], second["G"])
    np.testing.assert_array_equal(first["pMax"], second["pMax"])
    assert first["sepset"] == second["sepset"]
    assert first["max_ord"] == second["max_ord"]


def test_small_pds_does_not_create_processes_and_retains_live_deletions(monkeypatch):
    reference = _run(1, 0)

    def forbidden(*args, **kwargs):
        raise AssertionError("Small PDS must not initialize joblib workers")

    monkeypatch.setattr(fci, "Parallel", forbidden)
    actual = _run(4, 10**9)
    _same_result(reference, actual)
    assert actual["execution"]["requested_workers"] == 4
    assert actual["execution"]["effective_workers"] == 1
    assert actual["execution"]["backend"] == "serial"
    assert actual["execution"]["estimated_queries"] > 0


def test_large_pds_keeps_speculation_and_revalidates_live_context(monkeypatch):
    reference = _run(1, 0)
    dispatched = []

    class InlineExecutor:
        def __init__(self, n_jobs):
            dispatched.append(n_jobs)

        def __call__(self, tasks):
            return [function(*args, **kwargs) for function, args, kwargs in tasks]

    monkeypatch.setattr(fci, "Parallel", InlineExecutor)
    actual = _run(4, 1)
    _same_result(reference, actual)
    assert dispatched == [4]
    assert actual["execution"]["backend"] == "processes"
    assert actual["execution"]["speculative_hits"] > 0


def test_public_fci_retains_requested_options_and_per_phase_dispatch():
    data = np.random.default_rng(83).normal(size=(200, 8))
    graph = StandaloneFCIStable(n_jobs=4).run(data)
    assert graph.metadata["effective_n_jobs"] == 4
    passes = graph.metadata["execution"]["passes"]
    assert any(p["phase"] == "skeleton_fci" for p in passes)
    assert passes[-1]["phase"] == "pdsep_fci"
    assert passes[-1]["requested_workers"] == 4
    assert passes[-1]["effective_workers"] == 1
