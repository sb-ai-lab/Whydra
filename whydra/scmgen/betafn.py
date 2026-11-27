"""Evaluate beta distribution functions for outcome generation and ground truth."""

import numpy as np
from scipy import special

_LANCZOS_G = 7.0
_LANCZOS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)
_HALF_LOG_2PI = 0.5 * np.log(2.0 * np.pi)


def gammaln(x) -> np.ndarray:
    """Compute ln Γ(x) for x > 0 using nine Lanczos terms (g = 7).

    For x < 0.5, use the recurrence Γ(x + 1) = x Γ(x).
    """
    x = np.asarray(x, dtype=float)
    small = x < 0.5
    z = np.where(small, x + 1.0, x) - 1.0
    acc = np.full(x.shape, _LANCZOS[0])
    for i in range(1, 9):
        acc = acc + _LANCZOS[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    out = _HALF_LOG_2PI + (z + 0.5) * np.log(t) - t + np.log(acc)
    return np.where(small, out - np.log(np.where(small, x, 1.0)), out)


def betaln(a, b) -> np.ndarray:
    """Return the natural logarithm of the beta function B(a, b)."""
    return gammaln(a) + gammaln(b) - gammaln(np.add(a, b))


def _beta_values(a, b, value, *, inverse, lnb=None) -> np.ndarray:
    a, b, value = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
        np.asarray(value, dtype=np.float64),
    )
    if lnb is not None:
        np.broadcast_to(np.asarray(lnb, dtype=np.float64), value.shape)

    out = np.where(value >= 1.0, 1.0, 0.0)
    positive = (a > 0.0) & (b > 0.0)
    interior = (value > 0.0) & (value < 1.0)
    evaluate = special.betaincinv if inverse else special.betainc
    evaluate(a, b, value, out=out, where=positive & interior)

    if not np.all(positive):
        support = (value >= 0.0) & (value <= 1.0)
        both_zero = (a == 0.0) & (b == 0.0)
        out[(~positive & interior) | (both_zero & support)] = np.nan
        at_zero = support & (a == 0.0) & (b > 0.0)
        at_one = support & (a > 0.0) & (b == 0.0)
        out[at_zero] = 0.0 if inverse else 1.0
        out[at_one] = 1.0 if inverse else value[at_one] == 1.0
    return out


def betainc(a, b, x, lnb=None) -> np.ndarray:
    """Compute P(Beta(a, b) ≤ x).

    When one shape parameter is zero and the other is positive, all mass is
    at 0 for a = 0 or at 1 for b = 0. When both are zero, the result is NaN
    for x in [0, 1]. Return 0 if x is negative or NaN, and 1 if x exceeds 1.
    The optional precomputed log beta value, ``lnb``, must broadcast
    to the result shape. SciPy computes its own normalization.
    """
    return _beta_values(a, b, x, inverse=False, lnb=lnb)


def betaincinv(a, b, p) -> np.ndarray:
    """Compute the Beta(a, b) quantile.

    For p in [0, 1], one zero shape and one positive shape give a point mass:
    return 0 if a = 0 or 1 if b = 0. If both shapes are zero, return NaN.
    For p below 0 or NaN, return 0. For p above 1, return 1.
    """
    return _beta_values(a, b, p, inverse=True)
