"""Shared input contracts; no implicit imputation, dropping, or regularization."""

import os

import numpy as np


def _missing_mask(data):
    """Keep NumPy inputs independent of pandas while preserving object NA rules."""
    if data.dtype.kind in "fc":
        return np.isnan(data)
    if data.dtype.kind in "mM":
        return np.isnat(data)
    if data.dtype.kind in "biuSU":
        return np.zeros(data.shape, dtype=bool)
    # Object arrays can contain pandas.NA, NaT, None and mixed Python values.
    # Retain pandas' established handling for these inputs and unusual dtypes.
    from pandas import isna

    return isna(data)


def validate_alpha(alpha: float) -> float:
    """Return alpha as a float after checking that it lies strictly between zero and one."""
    if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha < 1:
        msg = "alpha must be finite and strictly between 0 and 1"
        raise ValueError(msg)
    return float(alpha)


def validate_data(data, *, numeric=False):
    """Validate a nonempty sample-by-feature matrix and optionally convert it to floats."""
    data = np.asarray(data)
    if data.ndim != 2 or min(data.shape, default=0) == 0:
        msg = f"Expected nonempty 2-D (samples, features) data; got {data.shape}"
        raise ValueError(msg)
    missing = _missing_mask(data)
    if missing.any():
        columns = np.flatnonzero(missing.any(axis=0)).tolist()
        msg = f"Data contains missing values (NaN) or None in columns {columns}"
        raise ValueError(msg)
    if numeric:
        try:
            data = data.astype(float)
        except (TypeError, ValueError) as exc:
            msg = "This statistical model requires numeric data"
            raise ValueError(msg) from exc
    if data.dtype.kind in "fci" and not np.isfinite(data).all():
        columns = np.flatnonzero(~np.isfinite(data).all(axis=0)).tolist()
        msg = f"Data contains nonfinite values (NaN/infinity) in columns {columns}"
        raise ValueError(msg)
    if data.dtype.kind == "O":
        columns = sorted(
            {
                j
                for (_, j), value in np.ndenumerate(data)
                if isinstance(value, (float, complex)) and not np.isfinite(value)
            }
        )
        if columns:
            msg = f"Data contains nonfinite values (NaN/infinity) in columns {columns}"
            raise ValueError(msg)
    return data


def validate_node_names(names, n):
    """Return the names as strings after checking for one unique name per column."""
    names = [str(name) for name in names]
    if len(names) != n or len(set(names)) != n:
        msg = "node_names must have one unique name for every column"
        raise ValueError(msg)
    return names


def effective_n_jobs(n_jobs):
    """Resolve a worker count, interpreting negative values relative to the CPU count."""
    if n_jobs is None:
        return 1
    if (
        isinstance(n_jobs, bool)
        or not isinstance(n_jobs, (int, np.integer))
        or n_jobs == 0
    ):
        msg = "n_jobs must be a nonzero integer"
        raise ValueError(msg)
    return (
        int(n_jobs) if n_jobs > 0 else max(1, (os.cpu_count() or 1) + 1 + int(n_jobs))
    )


def validate_endpoint_matrix(matrix, n=None):
    """Return an integer copy after validating the graph's endpoint matrix."""
    matrix = np.asarray(matrix)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or (n is not None and matrix.shape != (n, n))
    ):
        msg = "endpoint matrix must be square and match node_names"
        raise ValueError(msg)
    if not np.isin(matrix, (0, -1, 1, 2)).all() or np.any(np.diag(matrix)):
        msg = "Invalid endpoint codes or nonzero diagonal"
        raise ValueError(msg)
    if np.any((matrix != 0) != (matrix.T != 0)):
        msg = "An edge must have two non-NULL endpoints"
        raise ValueError(msg)
    return matrix.astype(int, copy=True)
