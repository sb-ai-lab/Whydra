"""Narrow workaround for a reproduced Windows ARM64 native inverse crash."""

import platform
import sys

import numpy as np


def _affected_numpy_inverse_runtime():
    if (
        sys.platform != "win32"
        or platform.machine().lower() not in ("arm64", "aarch64")
        or np.__version__.split("+", 1)[0] != "2.4.2"
    ):
        return False
    blas = (
        getattr(np.__config__, "CONFIG", {})
        .get("Build Dependencies", {})
        .get("blas", {})
    )
    return (
        str(blas.get("version", "")).startswith("0.3.31.dev")
        and "openblas" in str(blas.get("name", "")).lower()
    )


# Windows' platform.machine() can perform an expensive system query. Small
# Fisher matrices use NumPy on every supported runtime, so resolve this guard
# only immediately before a large inverse could reach the affected backend.
_USE_SCIPY_INVERSE = None
_SAFE_NUMPY_ORDER_LIMIT = 23


def fisher_inverse(matrices):
    """Invert using the same implementation in scalar and batch Fisher tests.

    On Windows ARM64 with NumPy 2.4.2 and OpenBLAS 0.3.31.dev, inversion
    triggered a native access violation on well-conditioned matrices with
    23 to 31 rows and columns, even when passed individually. Python cannot
    recover from this crash. SciPy's GESV solver also crashed in testing.
    SciPy's GETRF/GETRI routines succeeded on the captured 28×28 inputs and
    on independently generated diagnostic matrices.

    On that runtime, use GETRF/GETRI for supported numeric types from size
    23 upwards. Larger sizes may also be affected. Smaller matrices and
    other runtimes use NumPy. GETRF/GETRI can round differently from GESV,
    so scalar and batch Fisher tests use this same helper. The caller checks
    for singular or ill-conditioned inputs.
    """
    global _USE_SCIPY_INVERSE
    if _USE_SCIPY_INVERSE is False or matrices.shape[-1] < _SAFE_NUMPY_ORDER_LIMIT:
        return np.linalg.inv(matrices)
    if _USE_SCIPY_INVERSE is None:
        _USE_SCIPY_INVERSE = _affected_numpy_inverse_runtime()
    if not _USE_SCIPY_INVERSE:
        return np.linalg.inv(matrices)
    if matrices.dtype.kind in "biu":
        matrices = matrices.astype(np.float64)
    elif matrices.dtype not in (
        np.dtype("float32"),
        np.dtype("float64"),
        np.dtype("complex64"),
        np.dtype("complex128"),
    ):
        # NumPy rejects float16, object and extended precision instead of
        # silently promoting them as SciPy may do. Retain those errors.
        return np.linalg.inv(matrices)
    from scipy.linalg import inv

    if matrices.ndim == 2:
        return inv(matrices, check_finite=False)
    if matrices.size == 0:
        return np.empty_like(matrices)
    order = matrices.shape[-1]
    result = np.empty(matrices.shape, dtype=matrices.dtype)
    destination = result.reshape((-1, order, order))
    for index, matrix in enumerate(matrices.reshape((-1, order, order))):
        destination[index] = inv(matrix, check_finite=False)
    return result
