"""Conditional-independence tests: Fisher-Z and Pearson chi-square."""

import os
import threading
import warnings
from collections import OrderedDict
from concurrent.futures import Future
from importlib import import_module
from itertools import islice
from math import erfc, isclose, isfinite, log, sqrt
from numbers import Integral
from typing import Any

import numpy as np

from .conditioning import conditioning_allowed, validate_max_cond_set_size
from .linalg import fisher_inverse
from .validation import validate_data

_CACHE_MISSING = object()


def _resolve_profile_decorator():
    """Return `line_profiler.profile`, or a no-op when profiling is off.

    `chi_square` sits on the hot path of skeleton search, so the decorator stays
    inactive in a normal install and `line_profiler` stays a dev-only
    dependency. Set ``WHYDRA_PROFILE=1`` (and run under ``kernprof``) to enable.
    """
    if os.environ.get("WHYDRA_PROFILE", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return lambda func: func
    try:
        line_profile = import_module("line_profiler").profile
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
    """Check whether `method` assumes categorical data."""
    return str(method) in DISCRETE_TESTS


_FISHER_EPS = float(np.finfo(np.float64).eps)


def _fisher_probability(numerator, denominator, dof):
    """One libm operation order for scalar and batched low-order tests."""
    r = numerator / sqrt(denominator)
    r = min(0.9999999, max(-0.9999999, r))
    z = 0.5 * log((1 + r) / (1 - r))
    return erfc(sqrt(dof) * abs(z) / sqrt(2.0)), r


def _fisher_near_decision(value, r, dof, condition_bound, decision_alpha):
    if decision_alpha is None:
        return False
    # det / trace**k bounds lambda_min / lambda_max for an SPD k x k
    # matrix. Allow ample rounding error for the small closed form and the
    # reference LAPACK solve, then propagate through Fisher's derivative.
    # Borderline decisions use the original kernel, including its exact > or
    # >= comparison at the caller. This is deliberately more conservative
    # than the observed ULP error on well-conditioned inputs.
    uncertainty = (
        256
        * _FISHER_EPS
        * condition_bound
        * max(1.0, sqrt(dof))
        / max(1e-15, 1 - r * r)
    )
    return abs(value - decision_alpha) <= uncertainty


def _fisher_low_order_components(corr, x, y, conditions):
    """Return closed-form operands only for safely conditioned float64 input.

    The determinant/trace bound is much stricter than the reference 1e-12
    spectral threshold. Uncertain, asymmetric, nonfinite or other-dtype
    matrices retain the reference's eigenvalue and inverse checks.
    """
    if (
        not isinstance(corr, np.ndarray)
        or corr.dtype != np.float64
        or any(
            type(v) is not int and not isinstance(v, np.integer)
            for v in (x, y, *conditions)
        )
    ):
        return None
    a, b, d, reverse_b = (
        float(corr[x, x]),
        float(corr[x, y]),
        float(corr[y, y]),
        float(corr[y, x]),
    )
    if len(conditions) == 0:
        entries = (a, b, d, reverse_b)
        if not all(map(isfinite, entries)) or min(a, d) <= 0:
            return None
        if abs(b - reverse_b) > 8 * _FISHER_EPS * max(1.0, abs(b), abs(reverse_b)):
            return None
        trace = a + d
        trace_power = trace * trace
        determinant = a * d - b * b
        numerator, denominator = b, a * d
    else:
        z = conditions[0]
        c, e, f, reverse_c, reverse_e = (
            float(corr[x, z]),
            float(corr[y, z]),
            float(corr[z, z]),
            float(corr[z, x]),
            float(corr[z, y]),
        )
        entries = (a, b, c, d, e, f, reverse_b, reverse_c, reverse_e)
        if not all(map(isfinite, entries)) or min(a, d, f) <= 0:
            return None
        if any(
            abs(u - v) > 8 * _FISHER_EPS * max(1.0, abs(u), abs(v))
            for u, v in ((b, reverse_b), (c, reverse_c), (e, reverse_e))
        ):
            return None
        first = a * f - c * c
        second = d * f - e * e
        if min(first, second, a * d - b * b) <= 0:
            return None
        numerator = b * f - c * e
        denominator = first * second
        determinant = a * second - b * numerator + c * (b * e - c * d)
        trace = a + d + f
        trace_power = trace * trace * trace
    if (
        not isfinite(trace_power)
        or not isfinite(denominator)
        or denominator <= 0
        or determinant <= trace_power * 1e-6
    ):
        return None
    return numerator, denominator, trace_power / determinant


def fisher_z_from_corr(
    corr_matrix, n_samples, x, y, condition_set, *, decision_alpha=None
) -> float:
    """Canonical Fisher-Z with guarded closed forms for zero/one condition."""
    order = len(condition_set)
    if order <= 1:
        dof = n_samples - order - 3
        if dof > 0:
            components = _fisher_low_order_components(corr_matrix, x, y, condition_set)
            if components is not None:
                numerator, denominator, condition_bound = components
                value, r = _fisher_probability(numerator, denominator, dof)
                if not _fisher_near_decision(
                    value, r, dof, condition_bound, decision_alpha
                ):
                    return value
    return _fisher_z_from_corr_reference(corr_matrix, n_samples, x, y, condition_set)


def _fisher_z_from_corr_reference(corr_matrix, n_samples, x, y, condition_set) -> float:
    """Fisher-Z p-value for ``x ⟂ y | condition_set`` from a correlation matrix.

    The tester and process-based skeleton workers can both use this routine
    with their own correlation matrix. It raises ValueError for nonpositive
    degrees of freedom or a singular or ill-conditioned submatrix. Nonfinite
    entries in the submatrix yield NaN.
    """
    var = [x, y, *condition_set]
    sub_corr = corr_matrix[np.ix_(var, var)]

    # Fisher-Z needs n - |S| - 3 degrees of freedom. With few observations and a
    # deep conditioning set that goes non-positive; the statistic is undefined
    # rather than extreme, so report "independence not demonstrable" instead of
    # letting sqrt() raise from the middle of the search.
    dof = n_samples - len(condition_set) - 3
    if dof <= 0:
        msg = f"Fisher-Z requires positive degrees of freedom: n={n_samples}, |S|={len(condition_set)}"
        raise ValueError(msg)

    if not np.isfinite(sub_corr).all():
        return float("nan")  # diagnosed degenerate columns in legacy warning mode
    eigenvalues = np.linalg.eigvalsh(sub_corr)
    if eigenvalues[0] <= eigenvalues[-1] * 1e-12:
        msg = "Fisher-Z correlation submatrix is singular or ill-conditioned; no pseudoinverse is used"
        raise ValueError(msg)
    inv = fisher_inverse(sub_corr)

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
    return erfc(stat / sqrt(2.0))


def _check_shape(data) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim != 2:
        msg = f"CIT expects a 2-D (n_samples, n_features) array, got shape {data.shape}"
        raise ValueError(msg)
    return data


class DegenerateColumnWarning(UserWarning):
    """The data contain a variable for which the independence test is undefined."""


def _report_degenerate(kind: str, columns, strict: bool, hint: str) -> None:
    """Report degenerate columns with a warning or an error.

    The default is to warn and continue. Degenerate columns occur in real
    datasets (bnlearn/water has six, for example). The reference implementations
    accept these columns, so rejecting them would prevent comparisons.
    Set ``strict_degenerate=True`` to enable strict mode.
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
    """Return indices of columns with NaNs, which yield NaN rows in the correlation matrix."""
    data = _check_shape(data).astype(float)
    return np.flatnonzero(np.isnan(data).any(axis=0)).tolist()


def _degenerate_variance_columns(data):
    """Columns with no missing values and zero variance."""
    data = _check_shape(data).astype(float)
    complete = ~np.isnan(data).any(axis=0)
    degenerate = np.zeros(data.shape[1], dtype=bool)
    if np.any(complete):
        variances = np.var(data[:, complete], axis=0)
        degenerate[complete] = ~(variances > 0)
    return np.flatnonzero(degenerate).tolist()


def _single_level_columns(cardinalities):
    """Find columns with one observed level, for which df = 0 and chi2.sf(stat, 0) = NaN."""
    return np.flatnonzero(np.asarray(cardinalities) < 2).tolist()


def resolve_score_func(score_func, indep_test) -> str:
    """
    Resolve a GES score name against the CI test.

    'auto' picks the score matching the data type the CI test assumes:
    BDeu for discrete tests, Gaussian BIC otherwise. Any explicit name is
    validated against the score registry; a registered explicit choice wins.
    """
    if score_func not in (None, "auto"):
        from ..score_based.scores import validate_score_func

        return validate_score_func(score_func)
    return "local_score_BDeu" if is_discrete_test(indep_test) else "local_score_BIC"


_DENSE_CHISQ_CELL_LIMIT = 1_000_000


def _dense_conditional_chisq(data, cards, x, y, conditions, strata):
    from scipy.stats import chi2

    cx, cy = int(cards[x]), int(cards[y])
    # The caller bounded the product of cardinalities before any multiplication.
    z = data[:, conditions[0]].astype(np.intp, copy=False)
    for column in conditions[1:]:
        z = z * int(cards[column]) + data[:, column].astype(np.intp, copy=False)
    x_codes = data[:, x].astype(np.intp, copy=False)
    y_codes = data[:, y].astype(np.intp, copy=False)
    obs = np.bincount(
        (z * cx + x_codes) * cy + y_codes, minlength=strata * cx * cy
    ).reshape(strata, cx, cy)
    totals = obs.sum(axis=(1, 2))
    row_sums = obs.sum(axis=2)
    col_sums = obs.sum(axis=1)

    # Match the sparse kernel's occupied-cell order and rounding at alpha.
    zi, xi, yi = np.nonzero(obs)
    counts = obs[zi, xi, yi].astype(float)
    terms = totals[zi] * counts**2 / (row_sums[zi, xi].astype(float) * col_sums[zi, yi])
    stats = np.maximum(0.0, np.bincount(zi, weights=terms, minlength=strata) - totals)
    dfs = ((row_sums != 0).sum(axis=1) - 1) * ((col_sums != 0).sum(axis=1) - 1)
    valid = (totals >= 2) & (dfs > 0)
    total_df = int(dfs[valid].sum())
    return float(chi2.sf(stats[valid].sum(), total_df)) if total_df else 1.0


def chi_square_from_encoded(data_int, cardinalities, n_samples, X, Y, condition_set):
    # Fisher-Z uses libm, so it must not pay for scipy.stats at import/first fit.
    """Pearson chi-square p-value from pre-encoded categorical columns."""
    from scipy.stats import chi2

    X_data = data_int[:, X]
    Y_data = data_int[:, Y]
    if len(condition_set) == 0:
        obs = np.bincount(
            X_data * cardinalities[Y] + Y_data,
            minlength=cardinalities[X] * cardinalities[Y],
        ).reshape(cardinalities[X], cardinalities[Y])
        row_sum = obs.sum(axis=1)
        col_sum = obs.sum(axis=0)
        expected = np.outer(row_sum, col_sum) / n_samples
        valid = expected > 0
        stat = np.sum((obs[valid] - expected[valid]) ** 2 / expected[valid])
        df = (cardinalities[X] - 1) * (cardinalities[Y] - 1)
        return chi2.sf(stat, df)
    # Size the Cartesian table in Python integers before encoding or allocating it.
    cell_budget = min(4 * n_samples, _DENSE_CHISQ_CELL_LIMIT)
    xy_cells = int(cardinalities[X]) * int(cardinalities[Y])
    strata = 1
    for column in condition_set:
        strata *= int(cardinalities[column])
        if strata * xy_cells > cell_budget:
            break
    else:
        return _dense_conditional_chisq(
            data_int, cardinalities, X, Y, condition_set, strata
        )

    # Index observed strata only: no Cartesian index can overflow.
    _, z = np.unique(data_int[:, condition_set], axis=0, return_inverse=True)
    joint, counts = np.unique(
        np.column_stack(
            (z, X_data.astype(np.intp, copy=False), Y_data.astype(np.intp, copy=False))
        ),
        axis=0,
        return_counts=True,
    )
    zx, ix = np.unique(joint[:, :2], axis=0, return_inverse=True)
    zy, iy = np.unique(joint[:, [0, 2]], axis=0, return_inverse=True)
    row = np.bincount(ix, weights=counts)
    col = np.bincount(iy, weights=counts)
    totals = np.bincount(z)
    # Sum only occupied cells; the zero-cell contributions are incorporated
    # algebraically by subtracting n_z, preserving Pearson's statistic.
    terms = totals[joint[:, 0]] * counts.astype(float) ** 2 / (row[ix] * col[iy])
    stats = np.maximum(0.0, np.bincount(joint[:, 0], weights=terms) - totals)
    dfs = (np.bincount(zx[:, 0]) - 1) * (np.bincount(zy[:, 0]) - 1)
    valid = (totals >= 2) & (dfs > 0)
    total_df = int(dfs[valid].sum())
    return float(chi2.sf(stats[valid].sum(), total_df)) if total_df else 1.0


class CIT:
    """Conditional-independence tester for continuous and discrete data."""

    _batch_execution_stats: dict[str, Any]

    def __init__(
        self,
        data,
        method="fisherz",
        strict_degenerate=False,
        *,
        cache_size=100000,
        data_type=None,
        max_cond_set_size=None,
    ):
        self.max_cond_set_size = validate_max_cond_set_size(max_cond_set_size)
        if method not in IMPLEMENTED_TESTS:
            msg = f"Test {method!r} is not implemented locally; available: {sorted(IMPLEMENTED_TESTS)}"
            raise NotImplementedError(msg)
        data = validate_data(data, numeric=method == "fisherz")
        if data_type not in (None, "continuous", "categorical"):
            msg = "data_type must be continuous or categorical"
            raise ValueError(msg)
        if (data_type == "continuous" and method == "chisq") or (
            data_type == "categorical" and method == "fisherz"
        ):
            msg = "data_type is incompatible with indep_test"
            raise ValueError(msg)
        if not isinstance(cache_size, int) or cache_size < 0:
            msg = "cache_size must be a nonnegative integer"
            raise ValueError(msg)
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._inflight = {}
        self._cache_origins = {}
        self._decision_reference = {}
        self._cache_generations = {}
        self._cache_generation = 0
        self.cache_size = cache_size
        self.cache_hits = self.cache_misses = self.cache_evictions = 0
        self.decision_alpha = None
        self.data = data
        self.method = method
        self.strict_degenerate = bool(strict_degenerate)
        self.n_samples = data.shape[0]
        if method == "fisherz":
            if self.n_samples <= 3:
                msg = "Fisher-Z needs at least four observations"
                raise ValueError(msg)
            _report_degenerate(
                "Fisher-Z",
                _missing_value_columns(data),
                self.strict_degenerate,
                "missing values (NaN)",
            )
            _report_degenerate(
                "Fisher-Z",
                _degenerate_variance_columns(data),
                self.strict_degenerate,
                "zero variance",
            )
            self._corr_matrix = None
        elif method == "chisq":
            self.data_int = np.zeros_like(data, dtype=int, order="F")
            self.cardinalities = []
            for c in range(data.shape[1]):
                u, inv = np.unique(data[:, c], return_inverse=True)
                self.data_int[:, c] = inv
                self.cardinalities.append(len(u))
            self.cardinalities = np.array(self.cardinalities)
            if data_type is None and np.any(
                self.cardinalities >= max(10, 0.9 * self.n_samples)
            ):
                warnings.warn(
                    "chisq input has almost-unique values; confirm categorical interpretation with data_type='categorical' or choose a continuous CI test",
                    UserWarning,
                    stacklevel=2,
                )
            _report_degenerate(
                "chi-square",
                _single_level_columns(self.cardinalities),
                self.strict_degenerate,
                "a single observed level",
            )

    @property
    def corr_matrix(self):
        """Return the correlation matrix used by Fisher-Z tests."""
        if self.method == "fisherz" and self._corr_matrix is None:
            self._corr_matrix = np.corrcoef(self.data.T)
        return self._corr_matrix

    @corr_matrix.setter
    def corr_matrix(self, value):
        """Replace the cached correlation matrix."""
        self._corr_matrix = None if value is None else np.asarray(value)

    def __call__(self, X, Y, condition_set=None):
        """Evaluate the conditional independence test for X and Y given the conditioning set."""
        conditions = tuple(
            sorted(set(condition_set if condition_set is not None else ()))
        )
        if not conditioning_allowed(self, conditions):
            msg = "CI request exceeds max_cond_set_size; no p-value was computed"
            raise ValueError(msg)
        X, Y = int(X), int(Y)
        if X == Y or X in conditions or Y in conditions:
            msg = "CI variables and conditioning set must be disjoint"
            raise ValueError(msg)
        if any(v < 0 or v >= self.data.shape[1] for v in (X, Y, *conditions)):
            msg = "CI variable index out of bounds"
            raise ValueError(msg)
        key = (min(X, Y), max(X, Y), conditions)
        with self._lock:
            lookup = getattr(self._cache, "get_and_touch", None)
            cached = (
                lookup(key, _CACHE_MISSING)
                if lookup is not None
                else self._cache.get(key, _CACHE_MISSING)
            )
            if cached is not _CACHE_MISSING:
                self.cache_hits += 1
                if lookup is None:
                    self._cache.move_to_end(key)
                return cached
            cache_enabled = bool(self.cache_size)
            pending = None
            if not cache_enabled or key not in self._inflight:
                self.cache_misses += 1
                if cache_enabled:
                    # Most keys have no concurrent duplicate. Allocate the
                    # synchronization object only when a waiter actually arrives.
                    self._inflight[key] = None
            else:
                pending = self._inflight[key]
                if pending is None:
                    pending = Future()
                    self._inflight[key] = pending

        if pending is not None:
            from .ordered_ci import RetryCI

            try:
                result = pending.result()
            except RetryCI:
                return self(X, Y, conditions)
            with self._lock:
                self.cache_hits += 1
                if key in self._cache:
                    self._cache.move_to_end(key)
            return result

        # Independent requests must not hold the shared cache lock while NumPy
        # or SciPy computes the statistic. Only callers of the same key wait.
        try:
            result = (self.fisher_z if self.method == "fisherz" else self.chi_square)(
                X, Y, list(conditions)
            )
        except BaseException as exc:
            if cache_enabled:
                with self._lock:
                    pending = self._inflight.pop(key, None)
                    if pending is not None:
                        pending.set_exception(exc)
            raise

        with self._lock:
            if self.cache_size:
                self._store(key, result, (X, Y))
                result = self._decision_value(X, Y, conditions, result)
                if key in self._cache:
                    replace_value = getattr(self._cache, "replace_value", None)
                    if replace_value is not None:
                        replace_value(key, result)
                    else:
                        self._cache[key] = result
            if cache_enabled:
                pending = self._inflight.pop(key, None)
                if pending is not None:
                    pending.set_result(result)
            return result

    @property
    def _pending(self):
        return self._inflight

    @property
    def _decision_alpha(self):
        return self.decision_alpha

    @_decision_alpha.setter
    def _decision_alpha(self, value):
        self.decision_alpha = value

    def _serial_batch(self, requests):
        from .ordered_ci import ordered_batch

        return ordered_batch(self, requests)

    def _request(self, X, Y, condition_set):
        conditions = tuple(
            sorted(set(condition_set if condition_set is not None else ()))
        )
        if not conditioning_allowed(self, conditions):
            msg = "CI request exceeds max_cond_set_size; no p-value was computed"
            raise ValueError(msg)
        X, Y = int(X), int(Y)
        if X == Y or X in conditions or Y in conditions:
            msg = "CI variables and conditioning set must be disjoint"
            raise ValueError(msg)
        if any(v < 0 or v >= self.data.shape[1] for v in (X, Y, *conditions)):
            msg = "CI variable index out of bounds"
            raise ValueError(msg)
        key = (min(X, Y), max(X, Y), conditions)
        return X, Y, conditions, key

    def _store(self, key, result, origin):
        # Caller holds _lock. Numerical work never runs under the cache lock.
        if self.cache_size:
            self._cache[key] = result
            self._cache_generation += 1
            self._cache_generations[key] = self._cache_generation
            # The canonical key is symmetric, but inverse rounding need not
            # be. Remember reversed first computations for boundary repair.
            if origin[0] > origin[1]:
                self._cache_origins[key] = origin
            else:
                self._cache_origins.pop(key, None)
            self._cache.move_to_end(key)
            if len(self._cache) > self.cache_size:
                evicted, _value = self._cache.popitem(last=False)
                self._cache_origins.pop(evicted, None)
                self._decision_reference.pop(evicted, None)
                self._cache_generations.pop(evicted, None)
                self.cache_evictions += 1

    def _reference_value(self, x, y, conditions, pending_owner=None):
        conditions = tuple(sorted(set(conditions)))
        key = (min(x, y), max(x, y), conditions)
        with self._lock:
            if key in self._decision_reference:
                return self._decision_reference[key]
            generation = self._cache_generations.get(key)
            if key in self._cache:
                x, y = self._cache_origins.get(key, key[:2])
            pending = self._pending.get(key)
        value = _fisher_z_reference(self.corr_matrix, self.n_samples, x, y, conditions)
        with self._lock:
            # The old key may have been evicted and reinserted with a different
            # first orientation during the inverse. Never publish across those
            # generations. Only the owner of a fresh computation can publish
            # into a pending reservation; an external threshold check cannot.
            same_cached = (
                generation is not None
                and self._cache_generations.get(key) == generation
            )
            same_pending = (
                generation is None
                and pending_owner is not None
                and pending is pending_owner
                and self._pending.get(key) is pending_owner
                and key not in self._cache
            )
            if self.cache_size and (same_cached or same_pending):
                self._decision_reference[key] = value
                if key in self._cache:
                    self._cache[key] = value
        return value

    def _decision_value(self, x, y, conditions, value, pending_owner=None):
        alpha = getattr(self, "_decision_alpha", None)
        # Most p-values are far from alpha: avoid the custom-method identity
        # checks and the reference inverse entirely on that hot path.
        if (
            alpha is not None
            and isclose(value, alpha, rel_tol=1e-10, abs_tol=1e-14)
            and _builtin_fisherz(self)
        ):
            return self._reference_value(x, y, conditions, pending_owner)
        return value

    def __getstate__(self):
        from .cache import SharedCacheView

        with self._lock:
            state = {
                k: v
                for k, v in self.__dict__.items()
                if k not in {"_lock", "_inflight", "_condition"}
            }
            state["_unified_cache_snapshot"] = isinstance(self._cache, SharedCacheView)
            state["_cache"] = self._cache.copy()
            return state

    def __setstate__(self, state):
        unified = state.pop("_unified_cache_snapshot", False)
        self.__dict__.update(state)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._inflight = {}
        if unified:
            from .cache import share_ci_cache

            share_ci_cache(self)

    def fisher_z(self, X, Y, condition_set):
        """Evaluate a Fisher-Z conditional-independence test."""
        if fisher_z_from_corr is not _BUILTIN_FISHER_KERNEL:
            return fisher_z_from_corr(
                self.corr_matrix, self.n_samples, X, Y, condition_set
            )
        return fisher_z_from_corr(
            self.corr_matrix,
            self.n_samples,
            X,
            Y,
            condition_set,
            decision_alpha=getattr(self, "decision_alpha", None),
        )

    def batch(
        self,
        requests,
        *,
        n_jobs=None,
        backend="auto",
        batch_size=256,
        return_exceptions=False,
    ):
        """Evaluate bounded Fisher batches, preserving input order and caching."""
        if (
            n_jobs is None
            and backend == "auto"
            and batch_size == 256
            and not return_exceptions
        ):
            return self._serial_batch(requests)
        from .ci_batch import cit_batch

        return cit_batch(
            self,
            requests,
            n_jobs=1 if n_jobs is None else n_jobs,
            backend=backend,
            batch_size=batch_size,
            return_exceptions=return_exceptions,
        )

    @profile
    def chi_square(self, X, Y, condition_set):
        """Evaluate a Pearson chi-square conditional-independence test."""
        return chi_square_from_encoded(
            self.data_int, self.cardinalities, self.n_samples, X, Y, condition_set
        )


_BUILTIN_CIT_CALL = CIT.__call__
_BUILTIN_CIT_BATCH = CIT.batch
_BUILTIN_FISHER_Z = CIT.fisher_z
_BUILTIN_FISHER_KERNEL = fisher_z_from_corr
_BUILTIN_CHI_SQUARE = CIT.chi_square
_BUILTIN_CHISQ_KERNEL = chi_square_from_encoded

__all__ = [
    "CIT",
    "CONTINUOUS_TESTS",
    "DISCRETE_TESTS",
    "IMPLEMENTED_TESTS",
    "DegenerateColumnWarning",
    "fisher_z_from_corr",
    "is_discrete_test",
    "resolve_score_func",
]


# Compatibility with existing branch callers of the shared Fisher boundary.
_CI_BATCH_QUERIES = 256


def fisher_z_from_corr_batch(corr_matrix, n_samples, requests):
    """Return batched Fisher-Z statistics from a correlation matrix."""
    output = []
    requests = iter(requests)
    while chunk := list(islice(requests, _CI_BATCH_QUERIES)):
        from .ci_batch import fisher_z_batch_from_corr

        values = fisher_z_batch_from_corr(
            corr_matrix,
            n_samples,
            chunk,
            n_jobs=1,
            batch_size=_CI_BATCH_QUERIES,
            return_exceptions=True,
        )
        for value in values:
            if isinstance(value, BaseException):
                raise value
            output.append(value)
    return output


def _builtin_fisherz(test):
    from .ci_batch import supports_fisher_batch

    return (
        type(test) is CIT
        and supports_fisher_batch(test)
        and getattr(test.batch, "__func__", None) is _BUILTIN_CIT_BATCH
        and not any(name in test.__dict__ for name in ("__call__", "fisher_z", "batch"))
    )


def _threshold_pvalue(test, x, y, conditions, value, alpha):
    from math import isclose

    if isclose(value, alpha, rel_tol=1e-10, abs_tol=1e-14) and _builtin_fisherz(test):
        return test._reference_value(x, y, conditions)
    return value


__all__ += ["chi_square_from_encoded", "fisher_z_from_corr_batch"]

_fisher_z_reference = _fisher_z_from_corr_reference


_CI_BATCH_QUERIES = 1024

_CI_BATCH_BYTES = 1 << 20


def _low_order_r(corr, x, y, conditions):
    size = len(conditions)
    if size > 1:
        return None
    # The reference builds an integer index array from bool/int mixtures.
    # Scalar NumPy indexing treats bool as a mask instead; normalize only
    # booleans so malformed nonintegral indices still follow their old errors.
    if isinstance(x, (bool, np.bool_)):
        x = int(x)
    if isinstance(y, (bool, np.bool_)):
        y = int(y)
    a, b, d, e = (
        float(corr[x, x]),
        float(corr[x, y]),
        float(corr[y, x]),
        float(corr[y, y]),
    )
    if not all(map(isfinite, (a, b, d, e))):
        return None
    if not size:
        lower, upper = min(a - abs(d), e - abs(d)), max(a + abs(d), e + abs(d))
        determinant, product = a * e - b * d, a * e
        if (
            lower > _LOW_ORDER_RATIO * upper
            and upper > 0
            and determinant > 0
            and product > 0
        ):
            return b / sqrt(product)
        return None
    z = conditions[0]
    if isinstance(z, (bool, np.bool_)):
        z = int(z)
    c, f, g, h, i = (
        float(corr[x, z]),
        float(corr[y, z]),
        float(corr[z, x]),
        float(corr[z, y]),
        float(corr[z, z]),
    )
    if not all(map(isfinite, (c, f, g, h, i))):
        return None
    r0, r1, r2 = abs(d) + abs(g), abs(d) + abs(h), abs(g) + abs(h)
    lower, upper = min(a - r0, e - r1, i - r2), max(a + r0, e + r1, i + r2)
    cofactor00, cofactor11 = e * i - f * h, a * i - c * g
    determinant = a * cofactor00 - b * (d * i - f * g) + c * (d * h - e * g)
    product = cofactor00 * cofactor11
    if (
        lower > _LOW_ORDER_RATIO * upper
        and upper > 0
        and determinant > 0
        and product > 0
    ):
        return (b * i - c * h) / sqrt(product)
    return None


def _fisher_pvalue(r, dof):
    if not isfinite(r):
        return float("nan")
    r = min(0.9999999, max(-0.9999999, float(r)))
    z = 0.5 * log((1 + r) / (1 - r))
    return erfc(sqrt(dof) * abs(z) / sqrt(2.0))


def _fisher_batch_results(corr, n_samples, requests):
    """Bounded numeric batch, retaining per-request errors in original order."""
    results: list[float | Exception | None] = [None] * len(requests)
    groups = {}
    for index, (_x, _y, conditions) in enumerate(requests):
        groups.setdefault(len(conditions), []).append(index)
    for size, indices in groups.items():
        # Four similarly sized work arrays fit the budget. A single very deep
        # query stays scalar, rather than allocating an arbitrarily large stack.
        count = max(
            1, min(_CI_BATCH_QUERIES, _CI_BATCH_BYTES // (32 * (size + 2) ** 2))
        )
        for start in range(0, len(indices), count):
            rows = indices[start : start + count]
            try:
                if n_samples - size - 3 <= 0 or count == 1 or len(rows) == 1:
                    msg = "scalar fallback"
                    raise ValueError(msg)
                if any(
                    not isinstance(v, Integral)
                    for j in rows
                    for v in (requests[j][0], requests[j][1], *requests[j][2])
                ):
                    # np.asarray(..., dtype=intp) must not silently truncate a
                    # malformed float index that the scalar kernel rejects.
                    msg = "scalar index validation"
                    raise ValueError(msg)
                variables = np.asarray(
                    [[requests[j][0], requests[j][1], *requests[j][2]] for j in rows],
                    dtype=np.intp,
                )
                matrices = corr[variables[:, :, None], variables[:, None, :]]
                good = np.isfinite(matrices).all(axis=(1, 2))
                r = np.full(len(rows), np.nan)
                if size <= 1:
                    a, b, d, e = (
                        matrices[:, 0, 0],
                        matrices[:, 0, 1],
                        matrices[:, 1, 0],
                        matrices[:, 1, 1],
                    )
                    if size == 0:
                        lower = np.minimum(a - np.abs(d), e - np.abs(d))
                        upper = np.maximum(a + np.abs(d), e + np.abs(d))
                        determinant, product, numerator = a * e - b * d, a * e, b
                    else:
                        c, f, g, h, i = (
                            matrices[:, 0, 2],
                            matrices[:, 1, 2],
                            matrices[:, 2, 0],
                            matrices[:, 2, 1],
                            matrices[:, 2, 2],
                        )
                        r0, r1, r2 = (
                            np.abs(d) + np.abs(g),
                            np.abs(d) + np.abs(h),
                            np.abs(g) + np.abs(h),
                        )
                        lower = np.minimum(np.minimum(a - r0, e - r1), i - r2)
                        upper = np.maximum(np.maximum(a + r0, e + r1), i + r2)
                        cofactor00, cofactor11 = e * i - f * h, a * i - c * g
                        determinant = (
                            a * cofactor00 - b * (d * i - f * g) + c * (d * h - e * g)
                        )
                        product, numerator = cofactor00 * cofactor11, b * i - c * h
                    good &= (
                        (lower > _LOW_ORDER_RATIO * upper)
                        & (upper > 0)
                        & (determinant > 0)
                        & (product > 0)
                    )
                    r[good] = numerator[good] / np.sqrt(product[good])
                    # A failed cheap bound is not necessarily ill-conditioning.
                    # Run the EXACT spectral/inverse reference path as a stack
                    # for finite rejected matrices; retain scalar diagnostics for
                    # singular inputs. No admission tolerance is relaxed.
                    rejected = np.flatnonzero(
                        np.isfinite(matrices).all(axis=(1, 2)) & ~good
                    )
                    if len(rejected):
                        spectrum = np.linalg.eigvalsh(matrices[rejected])
                        admitted = rejected[spectrum[:, 0] > spectrum[:, -1] * 1e-12]
                        inverse = fisher_inverse(matrices[admitted])
                        denominator = inverse[:, 0, 0] * inverse[:, 1, 1]
                        with np.errstate(divide="ignore", invalid="ignore"):
                            r[admitted] = np.where(
                                denominator != 0,
                                -inverse[:, 0, 1] / np.sqrt(np.abs(denominator)),
                                np.nan,
                            )
                        good[admitted] = True
                else:
                    finite = np.flatnonzero(good)
                    eigenvalues = np.linalg.eigvalsh(matrices[finite])
                    good[finite] &= eigenvalues[:, 0] > eigenvalues[:, -1] * 1e-12
                    admitted = np.flatnonzero(good)
                    inverse = fisher_inverse(matrices[admitted])
                    denominator = inverse[:, 0, 0] * inverse[:, 1, 1]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        r[admitted] = -inverse[:, 0, 1] / np.sqrt(np.abs(denominator))
                for position in np.flatnonzero(good):
                    results[rows[position]] = _fisher_pvalue(
                        float(r[position]), n_samples - size - 3
                    )
            except (
                ValueError,
                np.linalg.LinAlgError,
                FloatingPointError,
                IndexError,
                TypeError,
            ):
                # Isolate bad matrices and preserve which request fails first.
                pass
            for index in rows:
                if results[index] is None:
                    x, y, conditions = requests[index]
                    from .ci_batch import _call_ci

                    results[index] = _call_ci(
                        fisher_z_from_corr, corr, n_samples, x, y, conditions
                    )
    return results


_LEGACY_BATCH_CORE = _fisher_batch_results

_LOW_ORDER_RATIO = 1e-10

_CI_BATCH_QUERIES = 1024

_CI_BATCH_BYTES = 1 << 20
