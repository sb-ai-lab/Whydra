"""Optional fused process-worker kernel with the same bounded Python fallback.

The extension owns enumeration, symmetric deduplication, evidence reduction,
and bounded cache tails at every order. Only safe order-zero/one numerical
windows run completely in C++; higher orders and ambiguous alpha boundaries
call the existing Fisher batch once per window. No alternate inverse or
singularity policy is introduced.
"""

import os
from time import perf_counter

try:
    if os.environ.get("WHYDRA_DISABLE_NATIVE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        msg = "disabled by WHYDRA_DISABLE_NATIVE"
        raise ImportError(msg)
    from . import _fisher_native

    if (
        getattr(_fisher_native, "API_VERSION", None) != 1
        or not callable(getattr(_fisher_native, "level", None))
        or not callable(getattr(_fisher_native, "batch", None))
    ):
        msg = "native Fisher extension ABI is stale; rebuild the optional extension"
        raise ImportError(msg)
except ImportError as exc:
    _fisher_native = None
    NATIVE_UNAVAILABLE_REASON = str(exc)
else:
    NATIVE_UNAVAILABLE_REASON = None

NATIVE_AVAILABLE = _fisher_native is not None


def _compatible_libm():
    """Decline compiled numerics when this compiler's libm differs from Python.

    Enumeration/reduction can still use the extension on such platforms. This
    tiny deterministic check is not a benchmark or an alternative CI test.
    """
    if _fisher_native is None:
        return False
    from array import array
    from math import erfc, log, sqrt

    for r in (-0.93, -0.5, -0.25, -0.001, 0.0, 0.003, 0.12, 0.5, 0.92):
        corr = memoryview(array("d", (1.0, r, r, 1.0))).cast("B").cast("d", (2, 2))
        for samples in (8, 100, 1000):
            z = 0.5 * log((1 + r) / (1 - r))
            expected = erfc(sqrt(samples - 3) * abs(z) / sqrt(2.0))
            actual = _fisher_native.batch(corr, samples, [(0, 1, ())], None)
            if actual != [expected]:
                return False
    return True


NATIVE_MATH_COMPATIBLE = _compatible_libm()


def native_pdsep_available():
    """Check whether the loaded native extension provides Possible-D-SEP search."""
    return _fisher_native is not None and callable(
        getattr(_fisher_native, "pdsep", None)
    )


def native_pdsep(corr, n_samples, descriptors, *, batch):
    """Run bounded first-separator searches with canonical numerical windows."""
    if _fisher_native is None or not native_pdsep_available():
        return None
    if not descriptors:
        return []
    alpha = descriptors[0][1][6]

    def evaluate(requests):
        return batch(
            corr,
            n_samples,
            requests,
            n_jobs=1,
            return_exceptions=True,
            decision_alpha=alpha,
        )

    result = _fisher_native.pdsep(corr, n_samples, descriptors, evaluate)
    if result is None:
        return None
    for _index, pair in result:
        pair.setdefault("native_execution", {"backend": "cpp", "phase": "pdsep"})
    return result


def native_batch(corr, n_samples, queries, alpha):
    """Evaluate a safe low-order window without NumPy temporary matrices."""
    if (
        _fisher_native is None
        or not NATIVE_MATH_COMPATIBLE
        or not hasattr(_fisher_native, "batch")
    ):
        return None
    try:
        return _fisher_native.batch(corr, n_samples, queries, alpha)
    except (TypeError, ValueError, OverflowError):
        # Leave unusual public inputs and per-query errors to the Python API.
        return None


def native_level(
    pairs,
    reference,
    corr,
    n_samples,
    depth,
    alpha,
    seed,
    cache_enabled,
    outer_enabled,
    retain,
    fci,
    *,
    batch,
    window=1024,
    native_math=True,
):
    """Return the normal level result, or None when the extension cannot run."""
    if _fisher_native is None:
        return None
    from . import independence_tests as ci
    from .ci_batch import _empty_execution_stats

    native_math = (
        native_math
        and NATIVE_MATH_COMPATIBLE
        and getattr(ci, "_fisher_batch_results", None)
        is getattr(ci, "_LEGACY_BATCH_CORE", None)
        and ci.fisher_z_from_corr is ci._BUILTIN_FISHER_KERNEL
    )
    numeric = _empty_execution_stats()

    def evaluate(requests):
        return batch(
            corr,
            n_samples,
            requests,
            n_jobs=1,
            batch_size=window,
            return_exceptions=True,
            execution_stats=numeric,
            decision_alpha=alpha,
        )

    started = perf_counter()
    result = _fisher_native.level(
        pairs,
        reference,
        corr,
        n_samples,
        depth,
        alpha,
        seed,
        cache_enabled,
        outer_enabled,
        retain,
        fci,
        evaluate,
        window,
        native_math,
    )
    if result is None:
        return None
    compiled = result["compiled_low_order_queries"]
    if compiled:
        numeric["batches"]["serial"] += 1
        numeric["tasks"]["serial"] += (compiled + window - 1) // window
        numeric["queries_by_backend"]["serial"] += compiled
        numeric["queries"] += compiled
        numeric["requested_backends"]["auto"] = (
            numeric["requested_backends"].get("auto", 0) + 1
        )
        numeric["actual_backend"] = "serial"
    # Fused execution includes enumeration/reduction and must not be reported
    # as pure numerical compute_seconds. Python callback timings remain there.
    result["native_execution"] = {
        "backend": "cpp",
        "depth": depth,
        "seconds": perf_counter() - started,
        "compiled_low_order_queries": compiled,
        "python_numeric_queries": result["python_numeric_queries"],
        "windows": result["native_windows"],
    }
    result["numeric"] = numeric
    return result
