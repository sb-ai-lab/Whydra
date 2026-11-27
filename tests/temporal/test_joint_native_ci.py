from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.special import betainc

from whydra.algorithms.temporal._engine._joint_ci import (
    NativeParCorr,
    NativeScalarAdapter,
)
from whydra.algorithms.temporal._engine._joint_data import JointDataFrame


@pytest.mark.parametrize("roles", ["x", "y", "z", "xyz"])
def test_component_lags_select_sensor_readings_within_each_recording(roles):
    readings = {}
    masks = {}
    kinds = {}
    lengths = {"night": 12, "day": 9}
    for number, (name, length) in enumerate(lengths.items(), start=1):
        table = 1000 * number + 10 * np.arange(length)[:, None] + np.arange(4)
        readings[name] = table.astype(float)
        kinds[name] = table % 3
        masks[name] = np.zeros(table.shape, dtype=bool)
    masks["night"][2, 2] = True
    masks["night"][8, 1] = True
    masks["day"][4, 3] = True
    readings["day"][6, 0] = np.nan

    frame = JointDataFrame(
        readings,
        mask=masks,
        data_type=kinds,
        vector_vars={0: [(2, -2), (0, 0)], 1: [(1, 0)], 2: [(3, 0)]},
    )
    sampled, groups, nodes, sample_kinds = frame.construct_array(
        X=[(0, -1)],
        Y=[(1, 0)],
        Z=[(2, -1), (0, -1), (2, -1)],
        cut_off="max_lag",
        mask_type=roles,
        return_cleaned_xyz=True,
    )

    assert frame.N == 3
    assert nodes == ([(2, -3), (0, -1)], [(1, 0)], [(3, -1)])
    assert groups.tolist() == [0, 0, 1, 2]
    excluded = {"night": set(), "day": {7}}
    for role, name, time in [("x", "night", 5), ("y", "night", 8), ("z", "day", 5)]:
        if role in roles:
            excluded[name].add(time)
    expected = []
    for name, length in lengths.items():
        times = sorted(set(range(3, length)) - excluded[name])
        assert frame.use_indices_dataset_dict[name].tolist() == times
        block = readings[name]
        expected.extend(
            (block[t - 3, 2], block[t - 1, 0], block[t, 1], block[t - 1, 3])
            for t in times
        )
    np.testing.assert_array_equal(sampled.T, expected)
    np.testing.assert_array_equal(sample_kinds.T, np.asarray(expected, dtype=int) % 3)


@pytest.mark.parametrize(
    ("cutoff", "prefix", "suffix"),
    [
        ("2xtau_max", 8, 0),
        ("tau_max", 4, 0),
        ("max_lag", 2, 0),
        ("max_lag_or_tau_max", 4, 0),
        ("2xtau_max_future", 2, 6),
    ],
)
def test_cutoff_discards_the_requested_ends_of_each_recording(cutoff, prefix, suffix):
    runs = {n: np.arange(n, dtype=float)[:, None] for n in (10, 13)}
    frame = JointDataFrame(runs)
    samples, _, _ = frame.construct_array(
        [(0, -2)], [(0, 0)], tau_max=4, cut_off=cutoff
    )
    times = [t for n in runs for t in range(prefix, n - suffix)]
    assert samples[1].tolist() == times
    np.testing.assert_array_equal(samples[0] + 2, times)


@pytest.mark.parametrize("vector", [False, True], ids=["scalar", "paired_sensors"])
def test_lagged_partial_correlation_on_a_balanced_factorial_design(vector):
    a, b, c, noise, context = np.array(list(product((-1, 1), repeat=5))).T
    samples = np.column_stack(
        (a + 4 * context, -3 * a + 4 * noise - 2 * context, b, c, context, -2 * context)
    )
    recordings = {}
    masks = {}
    for name, segment in zip(("west", "east"), np.split(samples, [13]), strict=True):
        block = np.full((len(segment) + 2, 6), 100.0)
        block[:-2, (0, 2)] = segment[:, (0, 2)]
        block[1:-1, (1, 3, 4, 5)] = segment[:, (1, 3, 4, 5)]
        recordings[name] = block
        masks[name] = np.zeros_like(block, dtype=bool)
        masks[name][-1, 1] = True

    groups = {0: [(0, 0)], 1: [(1, 0)], 2: [(4, 0), (5, 0)]}
    if vector:
        groups[0].append((2, 0))
        groups[1].append((3, 0))
    ci = NativeParCorr(multivariate=vector)
    ci.set_dataframe(JointDataFrame(recordings, mask=masks, vector_vars=groups))
    result = ci.run_test(
        [(0, -1)], [(1, 0)], [(2, 0)], tau_max=1, cut_off="max_lag", alpha_or_thres=0.05
    )

    endpoint_count = 4 if vector else 2
    dof = len(samples) - endpoint_count - 2
    comparisons = 4 if vector else 1
    p = comparisons * betainc(dof / 2, 0.5, 1 - (3 / 5) ** 2)
    assert result[0] == pytest.approx(-3 / 5, abs=1e-13)
    assert result[1] == pytest.approx(p, rel=1e-11, abs=1e-14)
    assert result[2] == (p <= 0.05)


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("corrected", [False, True])
def test_score_gain_matches_the_known_unexplained_variance(vector, corrected):
    signal, noise, untouched = np.array(list(product((-1, 1), repeat=3))).T
    n = len(signal)
    data = np.column_stack((3 * signal + 4 * noise, untouched, signal))
    target = [(0, 0), (1, 0)] if vector else [(0, 0)]
    ci = NativeParCorr(multivariate=vector)
    ci.set_dataframe(JointDataFrame(data, vector_vars={0: target, 1: [(2, 0)]}))

    null = ci.get_model_selection_criterion(0, [], corrected_aic=corrected)
    fitted = ci.get_model_selection_criterion(0, [(1, 0)], corrected_aic=corrected)
    k = 2 * len(target) - 1
    gain = n * np.log(16 / 25) / len(target) + 2
    if corrected:
        gain += 2 * (k + 1) * (k + 2) / (n - k - 2) - 2 * k * (k + 1) / (n - k - 1)
    assert fitted - null == pytest.approx(gain, abs=1e-12)


def test_scalar_callback_reuses_samples_only_for_the_same_frame_and_window():
    callback = Mock()
    callback.test.return_value = SimpleNamespace(statistic=0.5, p_value=0.25)
    adapter = NativeScalarAdapter(callback)
    signal = np.square(np.arange(9, dtype=float))
    obs = np.column_stack((signal, -signal))
    mask = np.zeros_like(obs, dtype=bool)
    mask[4, 1] = True
    first = JointDataFrame(obs, mask=mask)
    adapter.set_dataframe(first)
    query = {"X": [(1, -1)], "Y": [(0, 0)], "tau_max": 2}
    assert adapter.run_test(**query, alpha_or_thres=0.2)[2] is False
    adapter.set_dataframe(first)
    assert adapter.run_test(**query, alpha_or_thres=0.25)[2] is True
    assert callback.test.call_count == 1
    x, y, z, passed = callback.test.call_args.args
    assert (x, y, z) == (0, 1, [])
    target_times = np.array([4, 6, 7, 8])
    np.testing.assert_array_equal(
        passed, np.column_stack((-signal[target_times - 1], signal[target_times]))
    )

    adapter.run_test(**query, cut_off="max_lag")
    assert callback.test.call_args.args[-1].shape == (7, 2)
    callback.test.return_value = SimpleNamespace(statistic=-0.75, p_value=0.125)
    adapter.set_dataframe(JointDataFrame(obs[::-1]))
    assert adapter.run_test(**query) == (-0.75, 0.125)
    assert callback.test.call_count == 3


def test_three_rows_cannot_estimate_a_conditional_correlation():
    ci = NativeParCorr()
    ci.set_dataframe(JointDataFrame(np.eye(3)))
    with pytest.raises(ValueError, match="insufficient samples"):
        ci.run_test([(1, 0)], [(2, 0)], [(0, 0)])


def test_constant_endpoint_has_no_evidence_of_a_link():
    ci = NativeParCorr()
    ci.set_dataframe(JointDataFrame(np.column_stack((np.ones(6), np.arange(6)))))
    assert ci.run_test([(0, 0)], [(1, 0)], alpha_or_thres=0.05) == (0.0, 1.0, False)
