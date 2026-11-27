import numpy as np
import pytest
from scipy.stats import chi2

from whydra.algorithms.graph_core import CIT
from whydra.algorithms.graph_core import independence_tests as ci


def _chi_square_loop_reference(
    cit,
    X,
    Y,
    condition_set,
):
    """Compute the original loop-based chi-square p-value for comparison.

    Args:
        cit: Initialized categorical independence tester.
        X: Index of the first tested variable.
        Y: Index of the second tested variable.
        condition_set: Indices of conditioning variables.

    Returns:
        The p-value returned by the original implementation.
    """
    X_data = cit.data_int[:, X]
    Y_data = cit.data_int[:, Y]
    if len(condition_set) == 0:
        obs = np.bincount(
            X_data * cit.cardinalities[Y] + Y_data,
            minlength=cit.cardinalities[X] * cit.cardinalities[Y],
        ).reshape(cit.cardinalities[X], cit.cardinalities[Y])
        row_sum = obs.sum(axis=1)
        col_sum = obs.sum(axis=0)
        expected = np.outer(row_sum, col_sum) / cit.n_samples
        valid = expected > 0
        stat = np.sum((obs[valid] - expected[valid]) ** 2 / expected[valid])
        df = (cit.cardinalities[X] - 1) * (cit.cardinalities[Y] - 1)
        return chi2.sf(stat, df)
    if len(condition_set) == 1:
        Z_data = cit.data_int[:, condition_set[0]]
    else:
        Z_cols = cit.data_int[:, condition_set]
        dims = cit.cardinalities[list(condition_set)]
        Z_data = np.ravel_multi_index(Z_cols.T, dims)
    unique_z = np.unique(Z_data)
    total_stat = 0
    total_df = 0
    for z_val in unique_z:
        mask = Z_data == z_val
        x_sub = X_data[mask]
        y_sub = Y_data[mask]
        n_sub = len(x_sub)
        if n_sub < 2:
            continue
        obs = np.bincount(
            x_sub * cit.cardinalities[Y] + y_sub,
            minlength=cit.cardinalities[X] * cit.cardinalities[Y],
        ).reshape(cit.cardinalities[X], cit.cardinalities[Y])
        obs = obs[obs.sum(axis=1) > 0, :]
        obs = obs[:, obs.sum(axis=0) > 0]
        if obs.shape[0] < 2 or obs.shape[1] < 2:
            continue
        row_sum = obs.sum(axis=1)
        col_sum = obs.sum(axis=0)
        expected = np.outer(row_sum, col_sum) / n_sub
        valid = expected > 0
        if valid.sum() > 0:
            stat = np.sum((obs[valid] - expected[valid]) ** 2 / expected[valid])
            df = (obs.shape[0] - 1) * (obs.shape[1] - 1)
            total_stat += stat
            total_df += df
    if total_df == 0:
        return 1.0
    return chi2.sf(total_stat, total_df)


@pytest.mark.parametrize("seed", [17, 91])
@pytest.mark.parametrize("dependent", [False, True], ids=["independent", "dependent"])
@pytest.mark.parametrize(
    ("dims", "compressed"),
    [
        ((5,), False),
        ((3, 5), False),
        ((3, 4, 5), False),
        ((2, 3, 4, 5), False),
        ((3, 4, 5), True),
        ((2, 3, 4, 5), True),
    ],
)
def test_chi_square_conditional_matches_loop_reference(
    dims,
    compressed,
    dependent,
    seed,
):
    """Compare seeded categorical tables against the original conditional loop.

    Args:
        dims: Conditioning cardinalities, spanning one to four columns.
        compressed: Whether the data must select compressed stratum indexing.
        dependent: Whether diagonal counts introduce conditional dependence.
        seed: Seed for cell perturbations and observation ordering.

    Returns:
        None; asserts coverage, numerical agreement, and matching decisions.
    """
    rng = np.random.default_rng(seed)
    cx, cy = 4, 5
    k_full = int(np.prod(dims))
    if compressed:
        strata = np.array([[level % dim for dim in dims] for level in range(5)])
    else:
        strata = np.array(list(np.ndindex(*dims)))

    rows = []
    for stratum in strata:
        counts = np.ones((cx, cy), dtype=int)
        counts.flat[rng.choice(cx * cy, size=4, replace=False)] += 1
        if dependent:
            counts[np.arange(cx), np.arange(cx)] += 6
        for x, y in np.ndindex(cx, cy):
            rows.extend([[x, y, *stratum]] * int(counts[x, y]))
    data = np.array(rows)
    rng.shuffle(data)
    cit = CIT(data, method="chisq")
    condition_set = list(range(2, 2 + len(dims)))

    assert tuple(cit.cardinalities) == (cx, cy, *dims)
    assert (k_full * cx * cy > 4 * cit.n_samples) == compressed
    observed_strata, stratum_counts = np.unique(
        data[:, condition_set],
        axis=0,
        return_counts=True,
    )
    assert len(observed_strata) == len(strata)
    assert (len(observed_strata) < k_full) == compressed
    assert np.all(stratum_counts >= 2)
    for stratum in observed_strata:
        subset = data[np.all(data[:, condition_set] == stratum, axis=1)]
        assert len(np.unique(subset[:, 0])) == cx
        assert len(np.unique(subset[:, 1])) == cy

    expected = _chi_square_loop_reference(cit, 0, 1, condition_set)
    with np.errstate(divide="raise", invalid="raise"):
        actual = cit.chi_square(0, 1, condition_set)
    assert actual == pytest.approx(expected, abs=1e-12, rel=0)
    assert (actual > 0.05) == (expected > 0.05)
    assert (expected > 0.05) == (not dependent)


@pytest.mark.parametrize("usable", [False, True], ids=["all-skipped", "mixed"])
def test_chi_square_conditional_skips_degenerate_strata(usable):
    """Verify singleton and constant-variable strata against the original loop.

    Args:
        usable: Whether to include a stratum with positive degrees of freedom.

    Returns:
        None; asserts each degenerate case and equality with the reference.
    """
    rows = [[0, 0, 0]]
    rows.extend([[0, y, 1] for y in range(4)])
    rows.extend([[x, 0, 2] for x in range(3)])
    rows.extend([[1, 1, 3]] * 2)
    if usable:
        rows.extend([[x, y, 4] for x in (0, 2) for y in (1, 3)])
        rows.extend([[2, 3, 4]] * 5)
    else:
        rows.extend([[2, 3, 4]] * 2)
    data = np.array(rows)
    np.random.default_rng(42).shuffle(data)
    cit = CIT(data, method="chisq")
    assert tuple(cit.cardinalities) == (3, 4, 5)
    assert (4 * cit.n_samples >= 5 * 3 * 4) == usable
    assert np.count_nonzero(data[:, 2] == 0) == 1
    constant_x = data[data[:, 2] == 1]
    constant_y = data[data[:, 2] == 2]
    assert len(constant_x) > 1
    assert len(np.unique(constant_x[:, 0])) == 1
    assert len(np.unique(constant_x[:, 1])) == 4
    assert len(constant_y) > 1
    assert len(np.unique(constant_y[:, 1])) == 1
    assert len(np.unique(constant_y[:, 0])) == 3
    last = data[data[:, 2] == 4]
    assert len(np.unique(last[:, 0])) == (2 if usable else 1)
    assert len(np.unique(last[:, 1])) == (2 if usable else 1)
    if usable:
        assert len(np.unique(last[:, 0])) < cit.cardinalities[0]
        assert len(np.unique(last[:, 1])) < cit.cardinalities[1]

    expected = _chi_square_loop_reference(cit, 0, 1, [2])
    with np.errstate(divide="raise", invalid="raise"):
        actual = cit.chi_square(0, 1, [2])
    assert actual == pytest.approx(expected, abs=1e-12, rel=0)
    assert (actual > 0.05) == (expected > 0.05)
    if not usable:
        assert actual == expected == 1.0


@pytest.mark.parametrize(
    ("dtype", "cards", "n"),
    [
        (np.int8, (3, 3, 4, 5, 4), 1000),
        (np.uint8, (3, 3, 4, 5, 4), 1000),
        (np.int16, (5, 5, 16, 16, 8), 14000),
        (np.uint64, (3, 3, 4, 5, 4), 1000),
    ],
)
def test_bounded_table_matches_sparse_decisions_for_integer_codes(
    monkeypatch, dtype, cards, n
):
    rng = np.random.default_rng(504)
    data = np.column_stack([rng.integers(card, size=n) for card in cards]).astype(dtype)
    args = (data, cards, n, 0, 1, (2, 3, 4))
    assert np.prod(cards) <= min(4 * n, ci._DENSE_CHISQ_CELL_LIMIT)
    dense = ci.chi_square_from_encoded(*args)
    monkeypatch.setattr(ci, "_DENSE_CHISQ_CELL_LIMIT", 0)
    sparse = ci.chi_square_from_encoded(data.astype(np.intp), *args[1:])
    # Exact equality keeps decisions identical even when alpha equals the p-value.
    assert dense == sparse
    assert ci.chi_square_from_encoded(*args) == sparse


def test_many_single_level_conditions_do_not_limit_dense_index_dimensions():
    rng = np.random.default_rng(503)
    data = np.zeros((127, 72), dtype=np.int8)
    data[:, :2] = rng.integers(0, 2, size=(127, 2))
    cards = (2, 2) + (1,) * 70
    with_many = ci.chi_square_from_encoded(
        data, cards, len(data), 0, 1, tuple(range(2, 72))
    )
    with_one = ci.chi_square_from_encoded(data, cards, len(data), 0, 1, (2,))
    assert with_many == with_one
