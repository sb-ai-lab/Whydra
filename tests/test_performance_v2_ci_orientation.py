"""Cross-edge CI planning and sparse PAG orientation against scalar oracles."""

from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np
import pytest

from whydra.algorithms.graph_core import CIT, CausalGraph
from whydra.algorithms.graph_core import independence_tests as ci
from whydra.algorithms.standalone import fci_algo_v2 as fci
from whydra.algorithms.standalone.skeleton import SkeletonDiscovery
from whydra.background_knowledge import BackgroundKnowledge

from .conftest import run_scalar_queries


def _values():
    data = np.random.default_rng(51).normal(size=(200, 8))
    data[:, 1] += 0.6 * data[:, 0]
    data[:, 3] += 0.7 * data[:, 1] + data[:, 2]
    return data


@pytest.mark.parametrize("capacity", [0, 1, 3, 1000])
def test_serial_chunk_replays_cache_order_duplicates_and_counts(capacity):
    scalar, bulk = (
        CIT(_values(), cache_size=capacity),
        CIT(_values(), cache_size=capacity),
    )
    requests = [
        (x, y, (z,))
        for x, y, z in ((2, 1, 0), (0, 3, 1), (1, 2, 0), (3, 4, 0), (0, 3, 1))
    ]
    expected = [scalar(*request) for request in requests]
    np.testing.assert_array_equal(bulk._serial_batch(requests), expected)
    assert list(bulk._cache.items()) == list(scalar._cache.items())
    assert (bulk.cache_hits, bulk.cache_misses, bulk.cache_evictions) == (
        scalar.cache_hits,
        scalar.cache_misses,
        scalar.cache_evictions,
    )
    assert not bulk._pending


@pytest.mark.parametrize("bad", [(0, 0, ()), (0, 1, (12,)), (0, 1, (0,))])
def test_serial_batch_invalid_request_commits_only_successful_prefix(bad):
    scalar, bulk = CIT(_values()), CIT(_values())
    requests = [(0, 2, ()), (2, 0, ()), bad, (3, 4, ())]
    with pytest.raises(ValueError, match="CI ") as expected:
        run_scalar_queries(scalar, requests)
    with pytest.raises(ValueError, match=str(expected.value)):
        bulk._serial_batch(requests)
    assert list(bulk._cache.items()) == list(scalar._cache.items())
    assert (bulk.cache_hits, bulk.cache_misses) == (
        scalar.cache_hits,
        scalar.cache_misses,
    )
    assert not bulk._pending


def test_serial_batch_numeric_error_commits_only_prefix():
    values = _values()
    values[:, 3] = values[:, 2]
    scalar, bulk = CIT(values), CIT(values)
    requests = [(0, 1, ()), (2, 3, ()), (4, 5, ())]
    with pytest.raises(ValueError, match="singular"):
        run_scalar_queries(scalar, requests)
    with pytest.raises(ValueError, match="singular"):
        bulk._serial_batch(requests)
    assert list(bulk._cache.items()) == list(scalar._cache.items())
    assert bulk.cache_misses == scalar.cache_misses
    assert not bulk._pending


def test_serial_transaction_and_concurrent_public_batch_do_not_deadlock():
    test = CIT(_values())
    requests = [(x, y, ()) for x, y in combinations(range(8), 2)]
    with ThreadPoolExecutor(3) as pool:
        futures = [
            pool.submit(method, requests)
            for method in (test._serial_batch, test.batch, test._serial_batch)
        ]
        results = [future.result(timeout=10) for future in futures]
    for result in results[1:]:
        np.testing.assert_array_equal(result, results[0])
    assert test.cache_misses == len(requests)
    assert not test._pending


@pytest.mark.parametrize("family", ["PC", "FCI"])
def test_bulk_skeleton_matches_legacy_pairs_including_evidence(monkeypatch, family):
    def run():
        values = _values()
        graph = CausalGraph(8, [f"X{i}" for i in range(8)])
        graph.set_ind_test(CIT(values))
        graph.test._decision_alpha = 0.05
        search = SkeletonDiscovery(graph, 0.05)
        result = search.run() if family == "PC" else search.run_fci()
        return graph, result

    actual, result = run()
    monkeypatch.setattr(SkeletonDiscovery, "_bulk_allowed", lambda *args: False)
    expected, reference = run()
    np.testing.assert_array_equal(actual.G.graph, expected.G.graph)
    np.testing.assert_array_equal(actual.sepset, expected.sepset)
    assert actual.separators == expected.separators
    assert actual.separator_members == expected.separator_members
    # Pair-owned kernels may publish a different global LRU traversal.
    assert actual.test._cache.keys() == expected.test._cache.keys()
    for key in actual.test._cache:
        assert actual.test._cache[key] == pytest.approx(
            expected.test._cache[key], rel=1e-12, abs=1e-14
        )
    assert (actual.test.cache_hits, actual.test.cache_misses) == (
        expected.test.cache_hits,
        expected.test.cache_misses,
    )
    if family == "FCI":
        np.testing.assert_allclose(
            result["pMax"], reference["pMax"], rtol=1e-12, atol=1e-14
        )
        assert result["sepset"] == reference["sepset"]


def test_rejected_loworder_bound_batches_exact_reference_inverse(monkeypatch):
    corr = np.array([[1.0, 0.8, 0.8], [0.8, 1.0, 0.7], [0.8, 0.7, 1.0]])
    queries = [(0, 1, (2,)), (1, 2, (0,)), (2, 0, (1,))]
    expected = [ci._fisher_z_reference(corr, 100, *q) for q in queries]
    assert all(ci._low_order_r(corr, *q) is None for q in queries)

    def forbidden(*args):
        raise AssertionError(
            "Finite well-conditioned rejected matrices should use a stacked inverse"
        )

    monkeypatch.setattr(ci, "_fisher_z_reference", forbidden)
    # The guard hook uses the stacked reference inverse. The public batch
    # entry point uses the shared canonical guarded kernel.
    np.testing.assert_array_equal(
        ci._fisher_batch_results(corr, 100, queries), expected
    )


def test_stable_depth_packs_first_order_queries_across_edges(monkeypatch):
    values = np.random.default_rng(57).normal(size=(200, 12))
    graph = CausalGraph(12, [str(i) for i in range(12)])
    graph.set_ind_test(CIT(values))
    batches = []
    original = ci._fisher_batch_results

    def record(corr, samples, requests):
        batches.append(list(requests))
        return original(corr, samples, requests)

    monkeypatch.setattr(ci, "_fisher_batch_results", record)
    SkeletonDiscovery(graph, 0.05)._process_depth(1, None)
    assert any(
        len(batch) > 64 and len({(x, y) for x, y, _ in batch}) > 8 for batch in batches
    )


def test_serial_batch_retains_reference_threshold_after_reversed_hit():
    test = CIT(_values())
    alpha = ci._fisher_z_reference(test.corr_matrix, len(test.data), 3, 0, (1,))
    test._decision_alpha = alpha
    values = test._serial_batch([(3, 0, (1,)), (0, 3, (1,))])
    assert values == [alpha, alpha]


@pytest.mark.parametrize(
    ("sepset", "options", "center_mark"),
    [
        ({(0, 2): set(), (2, 0): set()}, {}, 1),
        ({(0, 2): set()}, {}, 2),
        ({(0, 2): set(), (2, 0): {1}}, {}, 2),
        ({(0, 2): set(), (2, 0): set()}, {"unfVect": {(2, 1, 0)}}, 2),
        ({(0, 2): set(), (2, 0): set()}, {"orientCollider": False}, 2),
    ],
)
def test_v2_requires_bilateral_evidence_for_collider(sepset, options, center_mark):
    skeleton = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    result = fci.udag2pag_v2(skeleton, sepset, 3, **options)
    np.testing.assert_array_equal(
        result, [[0, center_mark, 0], [2, 0, 2], [0, center_mark, 0]]
    )


@pytest.mark.parametrize(("required", "source_mark"), [(True, -1), (False, 2)])
def test_v2_propagates_background_knowledge_to_end_of_path(required, source_mark):
    skeleton = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
    knowledge = BackgroundKnowledge()
    if required:
        knowledge.require("A", "B")
    else:
        knowledge.forbid("B", "A")
    result = fci.udag2pag_v2(
        skeleton, {}, 4, background_knowledge=knowledge, node_names=list("ABCD")
    )
    np.testing.assert_array_equal(
        result, [[0, 1, 0, 0], [source_mark, 0, 1, 0], [0, -1, 0, 1], [0, 0, -1, 0]]
    )


@pytest.mark.parametrize(
    ("rules", "target", "source"),
    [([False] * 10, 2, 2), ([False, True] + [False] * 8, 1, 2), (None, 1, -1)],
)
def test_v2_r2_and_r8_complete_triangle_in_two_steps(rules, target, source):
    triangle = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    knowledge = BackgroundKnowledge().require("A", "B").require("B", "C")
    result = fci.udag2pag_v2(
        triangle,
        {},
        3,
        rules=rules,
        background_knowledge=knowledge,
        node_names=list("ABC"),
    )
    np.testing.assert_array_equal(result, [[0, 1, target], [-1, 0, 1], [source, -1, 0]])
