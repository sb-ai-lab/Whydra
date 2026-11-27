"""Small native PDSEP parity cases; no latency/performance assertions."""

import math

import numpy as np
import pytest

from whydra.algorithms.standalone.compact_pdsep import _compact_pair_search


@pytest.fixture
def native():
    module = pytest.importorskip("whydra.algorithms.graph_core._fisher_native")
    if not hasattr(module, "pdsep"):
        pytest.skip("optional PDSEP extension has not been built")
    return module.pdsep


def _reference(descriptors, callback):
    """Drive the original Python generator, independently of native dispatch."""
    remaining, active, results = iter(descriptors), [], []
    while True:
        while len(active) < 4:
            descriptor = next(remaining, None)
            if descriptor is None:
                break
            index, args = descriptor
            generator = _compact_pair_search(*args)
            try:
                requests = next(generator)
            except StopIteration as stopped:
                results.append((index, stopped.value))
            else:
                active.append((index, generator, requests))
        if not active:
            return results
        values = callback([query for _, _, requests in active for query in requests])
        following, offset = [], 0
        for index, generator, requests in active:
            end = offset + len(requests)
            try:
                pending = generator.send(values[offset:end])
            except StopIteration as stopped:
                results.append((index, stopped.value))
            else:
                following.append((index, generator, pending))
            offset = end
        active = following


def _canonical(value):
    if isinstance(value, BaseException):
        return type(value), str(value)
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def _compare(native, descriptors, answer, variables=24):
    histories = [[], []]

    def evaluate(which):
        def callback(requests):
            histories[which].append(list(requests))
            return [answer(*query) for query in requests]

        return callback

    expected = _reference(descriptors, evaluate(0))
    actual = native(np.eye(variables), 200, descriptors, evaluate(1))
    assert actual is not None
    assert histories[1] == histories[0]
    assert _canonical(actual) == _canonical(expected)
    return actual, histories[1]


@pytest.mark.parametrize("retain", [0, 1, 4])
@pytest.mark.parametrize("start_order", [1, 2])
def test_pdsep_native_multi_pair_order_cache_and_completion(
    native, retain, start_order
):
    descriptors = []
    for index in range(7):
        cached = {(2,): (0.2, 0), (3,): (0.1, 1), (2, 4): (0.05, 1)}
        cap = 0 if index == 2 else 3
        descriptors.append(
            (
                30 - index,
                (
                    0,
                    1,
                    (5, 2, 4, 3),
                    (5, 3, 4),
                    (1, 2),
                    cap,
                    0.8,
                    -0.5,
                    cached,
                    retain,
                    start_order,
                ),
            )
        )
    _compare(native, descriptors, lambda x, y, nodes: 0.9 if nodes == (2, 5) else 0.1)


def test_pdsep_native_growing_windows_are_bounded(native):
    nodes = tuple(range(2, 18))
    descriptors = [(0, (0, 1, nodes, nodes, (1,), 5, 2.0, -math.inf, {}, 3, 1))]
    actual, history = _compare(native, descriptors, lambda *query: 0.1)
    sizes = [len(window) for window in history]
    assert 32 in sizes
    assert 128 in sizes
    assert 512 in sizes
    assert 2048 in sizes
    assert max(sizes) == 2048
    assert actual[0][1]["separator"] is None
    assert actual[0][1]["computed"] == sum(sizes)
    assert len(actual[0][1]["updates"]) == 3


@pytest.mark.parametrize("separator_first", [False, True])
def test_pdsep_native_consumes_only_ordered_errors_and_caches_speculative_successes(
    native, separator_first
):
    descriptors = [
        (0, (0, 1, (2, 3, 4, 5), (2, 3, 4, 5), (1,), 2, 0.8, -math.inf, {}, 2, 1))
    ]

    def answer(x, y, nodes):
        if nodes == (3,):
            return ValueError("singular later request")
        if (nodes == (2,) and separator_first) or nodes == (4,):
            return 0.9
        return float("nan")

    actual, _history = _compare(native, descriptors, answer)
    result = actual[0][1]
    assert result["computed"] == 4
    assert result["consumed"] == (1 if separator_first else 2)
    assert (result["error"] is None) == separator_first
    assert ((4,), 0.9) in result["updates"]


def test_pdsep_native_cached_windows_and_duplicate_lru_entries(native):
    # Repeated nodes exercise OrderedDict move-to-end rather than FIFO-only tails.
    cached = {(2,): (0.1, 0), (3,): (0.2, 1), (4,): None}
    descriptors = [
        (0, (0, 1, (2, 3, 3, 4), (2, 3, 3, 4), (1,), 2, 2.0, -math.inf, cached, 2, 1))
    ]
    _compare(native, descriptors, lambda *query: 0.3)
    fully_cached = [
        (
            0,
            (
                0,
                1,
                (2, 3),
                (2, 3),
                (1,),
                1,
                2.0,
                -math.inf,
                {(2,): (0.1, 0), (3,): (0.2, 1)},
                2,
                1,
            ),
        )
    ]
    _actual, history = _compare(
        native, fully_cached, lambda *query: pytest.fail("unexpected miss")
    )
    assert history == [[]]


def test_pdsep_native_callback_failure_propagates_once(native):
    descriptors = [(0, (0, 1, (2, 3), (2, 3), (1,), 1, 0.05, -math.inf, {}, 2, 1))]
    calls = []

    def callback(requests):
        calls.append(requests)
        raise RuntimeError("numeric callback failed")

    with pytest.raises(RuntimeError, match="numeric callback failed"):
        native(np.eye(4), 200, descriptors, callback)
    assert len(calls) == 1


def test_pdsep_native_unsupported_buffer_declines_before_callback(native):
    def callback(requests):
        pytest.fail("unsupported input evaluated a callback")

    assert native(np.eye(3, dtype=np.float32), 200, [], callback) is None
