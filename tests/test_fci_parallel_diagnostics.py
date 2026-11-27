"""FCI diagnostics must not add work to quiet parallel skeleton rounds."""

from itertools import combinations

import numpy as np
import pytest

from whydra.algorithms.graph_core import CausalGraph
from whydra.algorithms.standalone import skeleton


def _discovery(verbose):
    cg = CausalGraph(6, [f"X{i}" for i in range(6)])
    cg.G.graph[:] = 0
    for x, y in ((0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (2, 4), (4, 5)):
        cg.G.graph[x, y] = cg.G.graph[y, x] = -1
    cg.pMax = np.full((6, 6), -np.inf)
    np.fill_diagonal(cg.pMax, 1)
    cg.set_ind_test(
        lambda x, y, conditions: 0.8 if (x + y + sum(conditions)) % 3 == 0 else 0.01
    )
    return skeleton.SkeletonDiscovery(cg, n_jobs=2, verbose=verbose)


def test_quiet_fci_skips_operation_estimate(monkeypatch, capsys):
    discovery = _discovery(verbose=False)
    neighbor_calls = []
    neighbors = discovery.cg.neighbors

    def counted_neighbors(node):
        neighbor_calls.append(node)
        return neighbors(node)

    def unexpected_estimate(*args):
        raise AssertionError("Quiet FCI must not compute a diagnostic CI estimate")

    monkeypatch.setattr(discovery.cg, "neighbors", counted_neighbors)
    monkeypatch.setattr(skeleton.math, "comb", unexpected_estimate)
    discovery._process_depth_parallel_fci(1, {}, None)

    # In quiet mode, each worker reads its neighbors once. A diagnostic
    # estimate would read the neighbors again and rescan the adjacency row.
    assert sorted(neighbor_calls) == list(range(discovery.no_of_var))
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("depth", [0, 1, 2, 4])
def test_verbose_fci_estimate_preserves_results(depth, capsys):
    quiet = _discovery(verbose=False)
    verbose = _discovery(verbose=True)
    graph = verbose.G.copy()
    # Count conditioning sets for each directed pair. A node with too few
    # other neighbors to form a set of this size contributes zero.
    expected = sum(
        sum(1 for _ in combinations([v for v in neighbors if v != y], depth))
        for x in range(len(graph))
        for neighbors in [list(np.flatnonzero(graph[x]))]
        for y in neighbors
    )

    quiet_result = quiet._process_depth_parallel_fci(depth, {}, None)
    verbose_result = verbose._process_depth_parallel_fci(depth, {}, None)

    assert (
        f"Depth {depth}: Estimated max CI tests: {expected}" in capsys.readouterr().out
    )
    assert quiet_result == verbose_result
    assert quiet.ci_tests_performed == verbose.ci_tests_performed
    np.testing.assert_array_equal(quiet.cg.pMax, verbose.cg.pMax)
    np.testing.assert_array_equal(quiet.G, verbose.G)


@pytest.mark.parametrize(
    ("jobs", "precache", "empty", "reason"),
    [
        (1, False, False, "single_worker"),
        (4, False, False, "below_parallel_work_threshold"),
        (4, True, False, "cache_hits_only"),
        (4, False, True, "no_work"),
    ],
)
def test_tiny_fisher_pdsep_reports_serial_without_creating_unused_pool(
    jobs, precache, empty, reason, monkeypatch
):
    from whydra.algorithms.graph_core import CIT
    from whydra.algorithms.standalone import fci_algo_paralel_cash as module
    from whydra.algorithms.standalone.ci_adapters import (
        attach_counted_batch,
        attach_legacy_batch,
    )

    matrix = np.zeros((5, 5), dtype=int)
    if not empty:
        matrix[0, 1] = matrix[1, 0] = 2
    cit = CIT(np.random.default_rng(92).normal(size=(300, 5)))
    if precache:
        for node in (2, 3, 4):
            cit(0, 1, (node,))
    stats = {"total": 0, "pdsep": 0}

    def counted(x, y, conditions):
        stats["total"] += 1
        stats["pdsep"] += 1
        return cit(x, y, conditions)

    attach_counted_batch(counted, cit, stats, {"name": "pdsep"})

    def legacy(_stats, x, y, conditions):
        return counted(x, y, conditions)

    attach_legacy_batch(legacy, counted)
    monkeypatch.setattr(module, "qreach", lambda *_: list(range(5)))

    def unused_pool(*_args, **_kwargs):
        pytest.fail("The built-in batch path must not create an unused outer pool")

    monkeypatch.setattr(module, "Parallel", unused_pool)
    result = module.pdsep(
        {"sk": matrix},
        None,
        legacy,
        5,
        {},
        0.05,
        np.full((5, 5), -np.inf),
        unfVect=set(),
        n_jobs=jobs,
        parallel_backend="processes",
        process_min_tasks=0,
    )
    execution = result["pdsep_stats"]
    assert execution["requested_backend"] == "processes"
    assert execution["actual_backend"] == execution["backend"] == "serial"
    assert execution["process_rounds"] == execution["thread_rounds"] == 0
    assert execution["fallback_reason"] == reason
