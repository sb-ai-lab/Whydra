"""Compiled enumeration/reduction preserves evidence and bounded cache events."""

from itertools import combinations

import numpy as np
import pytest

from whydra.algorithms.graph_core import native_fisher
from whydra.algorithms.graph_core.ci_batch import fisher_z_batch_from_corr
from whydra.algorithms.graph_core.independence_tests import (
    _fisher_z_from_corr_reference,
)
from whydra.algorithms.standalone import fisher_skeleton

compiled = pytest.mark.skipif(
    not native_fisher.NATIVE_AVAILABLE, reason="optional C++ extension not built"
)


def _run(depth, retain, outer, seed=None, corr=None):
    if corr is None:
        corr = np.corrcoef(
            np.random.default_rng(2041).normal(size=(200, 7)), rowvar=False
        )
    reference = [
        (1, 2, 3, 4),
        (0, 2, 3, 5),
        (0, 1, 4, 5, 6),
        (0, 1),
        (0, 2),
        (1, 2),
        (2,),
    ]
    return fisher_skeleton.fisher_level_chunk(
        [(0, 1), (0, 2), (1, 2)],
        reference,
        corr,
        200,
        depth,
        0.05,
        seed or {},
        retain > 0,
        outer,
        retain,
    )


def _assert_same(actual, expected):
    for name in (
        "pairs",
        "requests",
        "computed",
        "cit_hits",
        "outer_hits",
        "updates",
        "outer_updates",
        "promotions",
        "discarded",
        "outer_discarded",
    ):
        assert actual[name] == expected[name], name
    assert [(q, type(e), str(e)) for q, e in actual["errors"]] == [
        (q, type(e), str(e)) for q, e in expected["errors"]
    ]


@compiled
@pytest.mark.parametrize("depth", [0, 1, 2, 3])
@pytest.mark.parametrize(
    ("retain", "outer"), [(0, False), (1, True), (8, False), (100, True)]
)
def test_compiled_full_evidence_cache_counts_and_pmax_equal_python(
    monkeypatch, depth, retain, outer
):
    condition = tuple(range(2, 2 + depth))
    seed = {
        (0, 1, condition): (0.73, False),
        (0, 2, tuple(range(3, 3 + depth))): (0.81, True),
    }
    actual = _run(depth, retain, outer, seed)
    assert actual["native_execution"]["backend"] == "cpp"
    numerical = (
        "compiled_low_order_queries"
        if depth <= 1 and native_fisher.NATIVE_MATH_COMPATIBLE
        else "python_numeric_queries"
    )
    assert actual[numerical] > 0
    monkeypatch.setattr(native_fisher, "_fisher_native", None)
    expected = _run(depth, retain, outer, seed)
    _assert_same(actual, expected)


@compiled
@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("fci", [False, True])
def test_compiled_equal_and_neighbor_alpha_use_legacy_boundary(depth, fci):
    corr = np.corrcoef(np.random.default_rng(2042).normal(size=(500, 5)), rowvar=False)
    condition = tuple(range(2, 2 + depth))
    boundary = _fisher_z_from_corr_reference(corr, 500, 0, 1, condition)
    reference = [(1, *condition), (0,), (), (), ()]
    for alpha in (
        np.nextafter(boundary, -np.inf),
        boundary,
        np.nextafter(boundary, np.inf),
    ):
        actual = native_fisher.native_level(
            [(0, 1)],
            reference,
            corr,
            500,
            depth,
            alpha,
            {},
            True,
            False,
            4,
            fci,
            batch=fisher_z_batch_from_corr,
        )
        assert actual["pairs"][0][2][0] == boundary
        assert bool(actual["pairs"][0][3]) == (
            boundary >= alpha if fci else boundary > alpha
        )
        assert actual["python_numeric_queries"] > 0


@compiled
@pytest.mark.parametrize("kind", ["singular", "nan", "dof"])
def test_unsafe_numerical_window_retains_errors_and_nan(monkeypatch, kind):
    corr = np.eye(7)
    if kind == "singular":
        corr[0, 1] = corr[1, 0] = 1.0
    elif kind == "nan":
        corr[0, :] = corr[:, 0] = np.nan
    samples = 3 if kind == "dof" else 200
    args = (
        [(0, 1), (1, 2)],
        [(1, 2), (0, 2), (0, 1), (), (), (), ()],
        corr,
        samples,
        0,
        0.05,
        {},
        True,
        True,
        2,
    )
    actual = fisher_skeleton.fisher_level_chunk(*args)
    monkeypatch.setattr(native_fisher, "_fisher_native", None)
    expected = fisher_skeleton.fisher_level_chunk(*args)
    assert actual["requests"] == expected["requests"]
    assert actual["computed"] == expected["computed"]
    assert [(q, type(e), str(e)) for q, e in actual["errors"]] == [
        (q, type(e), str(e)) for q, e in expected["errors"]
    ]
    np.testing.assert_equal(actual["pairs"], expected["pairs"])


@compiled
def test_higher_order_callback_is_bounded_and_stops_on_same_error_window(monkeypatch):
    reference = [tuple(v for v in range(13) if v != x) for x in range(13)]
    calls = []

    def batch(corr, samples, queries, **kwargs):
        calls.append(len(queries))
        return [ValueError("sentinel") if q == (0, 1, (2, 3)) else 0.1 for q in queries]

    monkeypatch.setattr(fisher_skeleton, "fisher_z_batch_from_corr", batch)
    args = (
        list(combinations(range(13), 2)),
        reference,
        np.eye(13),
        100,
        2,
        0.05,
        {},
        True,
        True,
        3,
    )
    actual = fisher_skeleton.fisher_level_chunk(*args)
    assert calls == [1024]
    calls.clear()
    monkeypatch.setattr(native_fisher, "_fisher_native", None)
    expected = fisher_skeleton.fisher_level_chunk(*args)
    assert calls == [1024]
    _assert_same(actual, expected)


@compiled
@pytest.mark.parametrize("layout", ["fortran", "strided", "readonly"])
def test_native_accepts_supported_buffer_layouts(layout):
    corr = np.corrcoef(np.random.default_rng(2043).normal(size=(200, 7)), rowvar=False)
    expected = _run(1, 100, True, corr=corr)
    if layout == "fortran":
        corr = np.asfortranarray(corr)
    elif layout == "strided":
        backing = np.empty((14, 14))
        backing[::2, ::2] = corr
        corr = backing[::2, ::2]
    else:
        corr.flags.writeable = False
    _assert_same(_run(1, 100, True, corr=corr), expected)


def test_missing_extension_keeps_python_path(monkeypatch):
    monkeypatch.setattr(native_fisher, "_fisher_native", None)
    result = _run(1, 1, True)
    assert "native_execution" not in result
    assert result["computed"] > 0


@compiled
@pytest.mark.parametrize("order", [0, 1])
def test_native_typed_bulk_exact_random_scalar_values(order):
    from whydra.algorithms.graph_core.independence_tests import fisher_z_from_corr

    rng = np.random.default_rng(2044)
    data = rng.normal(size=(400, 12))
    data[:, 1] += 2 * data[:, 0]
    data[:, 3] -= 0.8 * data[:, 2]
    corr = np.corrcoef(data, rowvar=False)
    queries = []
    for _ in range(150):
        x, y, z = map(int, rng.choice(12, size=3, replace=False))
        queries.append((x, y, (z,) if order else ()))
    actual = native_fisher.native_batch(corr, 400, queries, 0.05)
    if not native_fisher.NATIVE_MATH_COMPATIBLE:
        assert actual is None
        return
    assert actual is not None
    assert actual == [
        fisher_z_from_corr(corr, 400, *q, decision_alpha=0.05) for q in queries
    ]


def test_stale_binary_without_batch_falls_back(monkeypatch):
    monkeypatch.setattr(native_fisher, "_fisher_native", object())
    assert native_fisher.native_batch(np.eye(2), 100, [(0, 1, ())], 0.05) is None


@compiled
@pytest.mark.parametrize("depth", [0, 1])
def test_different_libm_keeps_compiled_enumeration_with_canonical_math(
    monkeypatch, depth
):
    expected = _run(depth, 100, True)
    monkeypatch.setattr(native_fisher, "NATIVE_MATH_COMPATIBLE", False)
    actual = _run(depth, 100, True)
    assert actual["native_execution"]["backend"] == "cpp"
    assert actual["compiled_low_order_queries"] == 0
    assert actual["python_numeric_queries"] == actual["computed"] > 0
    _assert_same(actual, expected)


def test_skeleton_preserves_public_batch_override():
    from whydra.algorithms.graph_core import CIT, CausalGraph
    from whydra.algorithms.standalone.skeleton import SkeletonDiscovery

    cit = CIT(np.random.default_rng(2045).normal(size=(100, 4)))
    original = cit.batch
    called = []

    def batch(requests, **kwargs):
        called.append(1)
        return original(requests, **kwargs)

    cit.batch = batch
    cg = CausalGraph(4, [str(i) for i in range(4)])
    cg.set_ind_test(cit)
    discovery = SkeletonDiscovery(cg, n_jobs=1)
    assert discovery._process_depth_pc_processes(0, None) is None
    assert discovery._process_depth_fci_native(0, None, {}) is None
    discovery._process_depth(0, None)
    assert called
