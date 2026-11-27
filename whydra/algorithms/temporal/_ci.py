"""Bridge lag-aligned samples to Whydra's existing CI implementations."""

import numpy as np

from ..graph_core import CIT
from ..graph_core.independence_tests import IMPLEMENTED_TESTS
from ._engine._joint_ci import NativeParCorr, _NativeCI


class WhydraCI(_NativeCI):
    """Evaluate native scalar CIT on each correctly aligned CI sample table."""

    residual_based = False

    def __init__(self, method):
        super().__init__()
        self.method = method
        self.measure = method

    def _evaluate(self, array, xyz):
        x, y, z = (np.flatnonzero(xyz == group).tolist() for group in range(3))
        if len(x) != 1 or len(y) != 1:
            msg = "Whydra scalar CI requires scalar endpoints; use parcorr with joint dummies"
            raise ValueError(msg)
        if self.method == "fisherz" and array.shape[1] <= len(z) + 3:
            msg = "Insufficient samples for Fisher-Z after lag alignment"
            raise ValueError(msg)
        p_value = CIT(array.T, method=self.method, strict_degenerate=True)(
            x[0], y[0], z
        )
        if self.method == "fisherz":
            # PC1 ranks parents by effect size, not by significance.
            values = self._standardize(array)
            residuals = values[[x[0], y[0]]].T
            if z:
                design = values[z].T
                residuals = (
                    residuals
                    - design @ np.linalg.lstsq(design, residuals, rcond=None)[0]
                )
            statistic = np.corrcoef(residuals.T)[0, 1]
        else:
            # Existing categorical CIT exposes only p-values; 1-p gives the
            # same ordering by evidence used for categorical parent selection.
            statistic = 1.0 - p_value
        return statistic, p_value


def make_ci(indep_test, *, multivariate=False):
    if isinstance(indep_test, str):
        method = indep_test.lower()
        if method in {"parcorr", "par_corr"}:
            return NativeParCorr(multivariate=multivariate)
        if method not in IMPLEMENTED_TESTS:
            msg = f"Unsupported temporal indep_test: {indep_test!r}"
            raise ValueError(msg)
        if multivariate:
            msg = (
                'Joint dummy variables require indep_test="parcorr"; '
                "disable dataset_dummy and time_dummy to use scalar CIT"
            )
            raise ValueError(msg)
        return WhydraCI(method)
    if hasattr(indep_test, "as_joint_engine"):
        return indep_test.as_joint_engine(multivariate=multivariate)
    if hasattr(indep_test, "run_test") and hasattr(indep_test, "set_dataframe"):
        return indep_test
    msg = "indep_test must be a supported name or a temporal CI object"
    raise TypeError(msg)
