"""Independent analytic CI tests for the native time-series discovery engines.

Linear residuals are obtained by ordinary least squares. Scalar endpoints use
their Pearson correlation; vector endpoints use the largest absolute component
correlation and a Bonferroni correction. Degrees of freedom count physical
features, including redundant one-hot columns, to preserve the public numeric
convention of the existing J-PCMCI+ API.
"""

from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.stats import pearsonr, t

if TYPE_CHECKING:
    from ._joint_data import JointDataFrame


class _NativeCI:
    significance = "analytic"
    two_sided = True

    if TYPE_CHECKING:

        def _evaluate(self, array, xyz) -> tuple[float, float]: ...

    def __init__(self, *, significance="analytic", mask_type="xyz", verbosity=0):
        if significance != "analytic":
            msg = "Native joint CI supports analytic significance"
            raise ValueError(msg)
        self.mask_type = mask_type
        self.verbosity = verbosity
        self.cached_ci_results = {}
        self.ci_results = {}
        self.residuals = {}
        self.dataframe = None

    def set_dataframe(self, dataframe):
        if dataframe is not self.dataframe:
            # Node/cutoff keys are meaningful only for the attached samples.
            # Reattaching the same frame retains worker-local memoization.
            self.cached_ci_results.clear()
            self.ci_results.clear()
            self.residuals.clear()
        self.dataframe = dataframe

    def _array(self, X, Y, Z, tau_max, cut_off):
        if self.dataframe is None:
            msg = "A dataframe must be set before testing"
            raise ValueError(msg)
        return self.dataframe.construct_array(
            X=X,
            Y=Y,
            Z=Z,
            tau_max=tau_max,
            cut_off=cut_off,
            mask_type=self.mask_type,
            return_cleaned_xyz=True,
        )

    def run_test(
        self,
        X,
        Y,
        Z=None,
        tau_max=0,
        cut_off="2xtau_max",
        alpha_or_thres=None,
        verbosity=0,
    ):
        array, xyz, cleaned, _ = self._array(X, Y, Z or [], tau_max, cut_off)
        # The key describes physical nodes and sample alignment, so changing
        # lag cutoffs cannot accidentally reuse a statistic from other rows.
        groups = tuple(tuple(group) for group in cleaned)
        cache_key = (groups, int(tau_max), cut_off, self.mask_type)
        if cache_key in self.cached_ci_results:
            value, p_value = self.cached_ci_results[cache_key]
        else:
            # Constant components carry no information, including unused
            # levels of a one-hot dummy. Conditions retain nominal dimensions
            # whenever they vary even if their columns are linearly dependent.
            varying = np.std(array, axis=1) != 0.0
            reduced, reduced_xyz = array[varying], xyz[varying]
            if not (np.any(reduced_xyz == 0) and np.any(reduced_xyz == 1)):
                value, p_value = 0.0, 1.0
            else:
                value, p_value = self._evaluate(reduced, reduced_xyz)
            if not np.isfinite(value) or not np.isfinite(p_value):
                msg = "Undefined partial correlation: insufficient samples or degenerate conditioning"
                raise ValueError(msg)
            value, p_value = float(value), float(p_value)
            self.cached_ci_results[cache_key] = (value, p_value)
        dependent = None if alpha_or_thres is None else bool(p_value <= alpha_or_thres)
        self.ci_results[groups] = (value, p_value, dependent)
        if alpha_or_thres is None:
            return value, p_value
        return value, p_value, dependent

    @staticmethod
    def _standardize(array):
        standardized = np.array(array, dtype=float, copy=True)
        standardized -= standardized.mean(axis=1, keepdims=True)
        scales = standardized.std(axis=1, keepdims=True)
        np.divide(standardized, scales, out=standardized, where=scales != 0)
        return standardized

    def get_model_selection_criterion(self, j, parents, tau_max=0, corrected_aic=False):
        """Mean Gaussian AIC over components of a logical target node."""
        array, xyz, _ = cast("JointDataFrame", self.dataframe).construct_array(
            X=[(j, 0)],
            Y=[(j, 0)],
            Z=parents,
            tau_max=tau_max,
            mask_type=self.mask_type,
            cut_off="2xtau_max",
        )
        values = self._standardize(array)
        conditions = values[xyz == 2].T
        targets = values[xyz == 0].T
        if conditions.shape[1]:
            targets -= conditions @ np.linalg.lstsq(conditions, targets, rcond=None)[0]
        # Preserve the public nominal-feature convention: the assembled
        # array includes both endpoint groups, even for a vector target.
        n, dimension = len(targets), array.shape[0] - 1
        rss = np.sum(targets * targets, axis=0)
        if np.any(rss <= 0):
            msg = "Cannot score a deterministic target"
            raise ValueError(msg)
        penalty = 2.0 * dimension
        if corrected_aic:
            if n <= dimension + 1:
                msg = "Insufficient samples for corrected AIC"
                raise ValueError(msg)
            penalty += 2.0 * dimension * (dimension + 1) / (n - dimension - 1)
        return float(np.mean(n * np.log(rss) + penalty))


class NativeParCorr(_NativeCI):
    """Scalar partial correlation and the strongest component pair for vectors."""

    measure = "par_corr"
    residual_based = True

    def __init__(
        self,
        *,
        multivariate=False,
        significance="analytic",
        mask_type="xyz",
        verbosity=0,
        correlation_type="max_corr",
    ):
        if correlation_type != "max_corr":
            msg = "NativeParCorr supports correlation_type='max_corr'"
            raise ValueError(msg)
        super().__init__(
            significance=significance, mask_type=mask_type, verbosity=verbosity
        )
        self.multivariate = bool(multivariate)
        self.measure = "par_corr_mult" if self.multivariate else "par_corr"

    def _evaluate(self, array, xyz):
        dof = array.shape[1] - array.shape[0]
        if dof < 1:
            msg = "Undefined partial correlation: insufficient samples or degenerate conditioning"
            raise ValueError(msg)
        values = self._standardize(array)
        x, y, z = (values[xyz == group] for group in range(3))
        if not self.multivariate and (len(x) != 1 or len(y) != 1):
            msg = "Scalar partial correlation cannot test vector-valued endpoints"
            raise ValueError(msg)
        if len(z):
            design = z.T
            x = (x.T - design @ np.linalg.lstsq(design, x.T, rcond=None)[0]).T
            y = (y.T - design @ np.linalg.lstsq(design, y.T, rcond=None)[0]).T
        if np.any(np.ptp(x, axis=1) == 0) or np.any(np.ptp(y, axis=1) == 0):
            msg = "Undefined partial correlation: a tested endpoint is deterministic given its conditions"
            raise ValueError(msg)
        if self.multivariate:
            with np.errstate(divide="ignore", invalid="ignore"):
                correlation = np.corrcoef(np.concatenate([x, y]))[: len(x), len(x) :]
            if not np.isfinite(correlation).all():
                msg = "Undefined partial correlation: degenerate residual correlation"
                raise ValueError(msg)
            value = correlation.flat[np.argmax(np.abs(correlation))]
        else:
            value = cast("float", pearsonr(x[0], y[0])[0])
        if abs(value) >= 1:
            p_value = 0.0
        else:
            score = abs(value) * np.sqrt(dof / (1.0 - value * value))
            p_value = 2.0 * t.sf(score, dof)
        if self.multivariate:
            p_value *= len(x) * len(y)
        return float(value), float(p_value)


class NativeScalarAdapter(_NativeCI):
    """Use the public scalar CI protocol on the native aligned data array."""

    residual_based = False

    def __init__(self, test, *, significance="analytic", mask_type="xyz", verbosity=0):
        super().__init__(
            significance=significance, mask_type=mask_type, verbosity=verbosity
        )
        self.test = test
        self.measure = type(test).__name__

    def _evaluate(self, array, xyz):
        x, y, z = (np.flatnonzero(xyz == group).tolist() for group in range(3))
        if len(x) != 1 or len(y) != 1:
            msg = "A custom scalar CI test cannot test a vector-valued endpoint"
            raise ValueError(msg)
        result = self.test.test(x[0], y[0], z, array.T)
        if not np.isfinite(result.p_value) or not 0 <= result.p_value <= 1:
            msg = "CI test returned an invalid p-value"
            raise ValueError(msg)
        return result.statistic, result.p_value
