import os
import warnings
from math import log, sqrt

import numpy as np
from scipy.stats import norm, chi2


def _resolve_profile_decorator():
    """Return `line_profiler.profile`, or a no-op when profiling is off.

    `chi_square` sits on the hot path of skeleton search, so the decorator stays
    inactive in a normal install and `line_profiler` stays a dev-only
    dependency. Set ``WHYDRA_PROFILE=1`` (and run under ``kernprof``) to enable.
    """
    if os.environ.get("WHYDRA_PROFILE", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return lambda func: func
    try:
        from line_profiler import profile as line_profile
    except ImportError:
        warnings.warn(
            "WHYDRA_PROFILE is set but line_profiler is not installed; "
            "install it with `pip install line_profiler` to collect line timings.",
            RuntimeWarning,
            stacklevel=2,
        )
        return lambda func: func
    return line_profile


profile = _resolve_profile_decorator()


# Independence tests grouped by the data type they assume. Score-based phases
# (GES / xGES) have to agree with the CI test about that data type, otherwise
# the initial CPDAG is built under the wrong likelihood.
#
# NOTE: membership here means "assumes this data type", NOT "is implemented".
# `CIT` currently implements only `fisherz` and `chisq`; `parcorr` and `kci` are
# listed so score resolution already classifies them correctly, but calling them
# raises NotImplementedError. See IMPLEMENTED_TESTS below.
DISCRETE_TESTS = frozenset({"chisq"})
CONTINUOUS_TESTS = frozenset({"fisherz", "parcorr", "kci"})

#: The tests `CIT.__call__` can actually run.
IMPLEMENTED_TESTS = frozenset({"fisherz", "chisq"})


def is_discrete_test(method) -> bool:
    """True if `method` assumes categorical data."""
    return str(method) in DISCRETE_TESTS


def fisher_z_from_corr(corr_matrix, n_samples, x, y, condition_set) -> float:
    """Fisher-Z p-value for ``x ⟂ y | condition_set`` from a correlation matrix.

    Kept as a free function so that every backend computes Fisher-Z the same
    way: `CIT.fisher_z` uses it on its own cached matrix, and the process-based
    skeleton workers use it on a matrix shipped to the worker, where a `CIT`
    instance is not available.
    """
    var = [x, y] + list(condition_set)
    sub_corr = corr_matrix[np.ix_(var, var)]

    # Fisher-Z needs n - |S| - 3 degrees of freedom. With few observations and a
    # deep conditioning set that goes non-positive; the statistic is undefined
    # rather than extreme, so report "independence not demonstrable" instead of
    # letting sqrt() raise from the middle of the search.
    dof = n_samples - len(condition_set) - 3
    if dof <= 0:
        warnings.warn(
            f"Fisher-Z has no degrees of freedom left "
            f"(n_samples={n_samples}, |condition_set|={len(condition_set)}); "
            "returning p=1.0. Limit the search depth or use more observations.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 1.0

    try:
        inv = np.linalg.inv(sub_corr)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(sub_corr)

    # A zero-variance column makes the whole correlation row NaN. Clamping that
    # NaN into the [-cut_at, cut_at] range would turn it into r = -0.9999999 —
    # comparisons against NaN are all false, so max() returns its first argument
    # — and the test would report p = 0.0, i.e. a perfect dependence on a
    # constant. Propagate the NaN instead: undefined must stay undefined.
    denominator = inv[0, 0] * inv[1, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = -inv[0, 1] / sqrt(abs(denominator)) if denominator != 0 else float("nan")
    if not np.isfinite(r):
        return float("nan")

    cut_at = 0.9999999
    r = min(cut_at, max(-cut_at, r))
    z = 0.5 * log((1 + r) / (1 - r))
    stat = sqrt(dof) * abs(z)
    return 2 * (1 - norm.cdf(stat))


def _check_shape(data) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"CIT expects a 2-D (n_samples, n_features) array, got shape {data.shape}")
    return data


class DegenerateColumnWarning(UserWarning):
    """Данные содержат переменную, по которой тест независимости не определён."""


def _report_degenerate(kind: str, columns, strict: bool, hint: str) -> None:
    """Сообщить о вырожденных столбцах: предупреждением или ошибкой.

    По умолчанию — предупреждение, а не отказ. Вырожденные столбцы встречаются
    в реальных наборах (например, шесть таких в bnlearn/water), и эталонные
    реализации их переваривают; жёсткий отказ сделал бы сверку невозможной.
    Строгий режим включается параметром ``strict_degenerate=True``.
    """
    if not len(columns):
        return
    message = (
        f"{kind} is undefined for columns {list(columns)}: {hint}. "
        "Tests touching them return NaN, and NaN never exceeds alpha, so their "
        "edges are never removed. Drop these variables or pass "
        "strict_degenerate=True to fail instead."
    )
    if strict:
        raise ValueError(message)
    warnings.warn(message, DegenerateColumnWarning, stacklevel=3)


def _missing_value_columns(data):
    """Столбцы с NaN: corrcoef заполняет их строку NaN."""
    data = _check_shape(data).astype(float)
    return np.flatnonzero(np.isnan(data).any(axis=0)).tolist()


def _degenerate_variance_columns(data):
    """Столбцы без пропусков с нулевой дисперсией."""
    data = _check_shape(data).astype(float)
    complete = ~np.isnan(data).any(axis=0)
    degenerate = np.zeros(data.shape[1], dtype=bool)
    if np.any(complete):
        variances = np.var(data[:, complete], axis=0)
        degenerate[complete] = ~(variances > 0)
    return np.flatnonzero(degenerate).tolist()


def _single_level_columns(cardinalities):
    """Столбцы с одним наблюдаемым уровнем: df = 0 и chi2.sf(stat, 0) = NaN."""
    return np.flatnonzero(np.asarray(cardinalities) < 2).tolist()


def resolve_score_func(score_func, indep_test) -> str:
    """
    Resolve a GES score name against the CI test.

    'auto' picks the score matching the data type the CI test assumes:
    BDeu for discrete tests, Gaussian BIC otherwise. Any explicit name is
    passed through untouched, so a deliberate choice always wins.
    """
    if score_func not in (None, "auto"):
        return score_func
    return "local_score_BDeu" if is_discrete_test(indep_test) else "local_score_BIC"


class CIT:
    def __init__(self, data, method='fisherz', strict_degenerate=False):
        self.data = data
        self.method = method
        self.strict_degenerate = bool(strict_degenerate)
        self.n_samples = data.shape[0]
        if method == 'fisherz':
            _report_degenerate(
                "Fisher-Z", _missing_value_columns(data),
                self.strict_degenerate, "missing values (NaN)",
            )
            _report_degenerate(
                "Fisher-Z", _degenerate_variance_columns(data),
                self.strict_degenerate, "zero variance",
            )
            self.corr_matrix = np.corrcoef(data.T)
        elif method == 'chisq':
            self.data_int = np.zeros_like(data, dtype=int)
            self.cardinalities = []
            for c in range(data.shape[1]):
                u, inv = np.unique(data[:, c], return_inverse=True)
                self.data_int[:, c] = inv
                self.cardinalities.append(len(u))
            self.cardinalities = np.array(self.cardinalities)
            _report_degenerate(
                "chi-square", _single_level_columns(self.cardinalities),
                self.strict_degenerate, "a single observed level",
            )

    def __call__(self, X, Y, condition_set=None):
        if condition_set is None:
            condition_set = []
        if self.method == 'fisherz':
            return self.fisher_z(X, Y, condition_set)
        if self.method == 'chisq':
            return self.chi_square(X, Y, condition_set)
        raise NotImplementedError(
            f"Test {self.method!r} is not implemented locally; "
            f"available: {sorted(IMPLEMENTED_TESTS)}"
        )

    def fisher_z(self, X, Y, condition_set):
        return fisher_z_from_corr(self.corr_matrix, self.n_samples, X, Y, condition_set)

    @profile
    def chi_square(self, X, Y, condition_set):
        X_data = self.data_int[:, X]
        Y_data = self.data_int[:, Y]
        if len(condition_set) == 0:
            obs = np.bincount(
                X_data * self.cardinalities[Y] + Y_data,
                minlength=self.cardinalities[X] * self.cardinalities[Y]
            ).reshape(self.cardinalities[X], self.cardinalities[Y])
            row_sum = obs.sum(axis=1)
            col_sum = obs.sum(axis=0)
            expected = np.outer(row_sum, col_sum) / self.n_samples
            valid = expected > 0
            stat = np.sum((obs[valid] - expected[valid]) ** 2 / expected[valid])
            df = (self.cardinalities[X] - 1) * (self.cardinalities[Y] - 1)
            return chi2.sf(stat, df)
        if len(condition_set) == 1:
            Z_data = self.data_int[:, condition_set[0]]
        else:
            Z_cols = self.data_int[:, condition_set]
            dims = self.cardinalities[list(condition_set)]
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
                x_sub * self.cardinalities[Y] + y_sub,
                minlength=self.cardinalities[X] * self.cardinalities[Y]
            ).reshape(self.cardinalities[X], self.cardinalities[Y])
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


__all__ = [
    "CIT",
    "DegenerateColumnWarning",
    "DISCRETE_TESTS",
    "CONTINUOUS_TESTS",
    "IMPLEMENTED_TESTS",
    "fisher_z_from_corr",
    "is_discrete_test",
    "resolve_score_func",
]
