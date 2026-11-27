"""CPU Fisher-Z fallback and experimental Schur-complement variants."""

from collections.abc import Sequence
from math import erfc, log, sqrt
from typing import overload

import numpy as np
from scipy.special import erfc as erfc_vec
from scipy.special import ndtri_exp

CUT_AT = 0.9999999


def _fisher_dof(n_samples: int, k: int) -> int:
    dof = n_samples - k - 3
    if dof <= 0:
        msg = f"Fisher-Z requires positive degrees of freedom: n={n_samples}, |S|={k}"
        raise ValueError(msg)
    return dof


def fisher_z_threshold(alpha: float) -> float:
    """Compute the threshold in log space so subnormal alpha values survive.

    Recheck boundary decisions with the CPU test because the inverse is rounded.
    """
    if not 0.0 < alpha < 1.0:
        msg = "alpha must be between 0 and 1"
        raise ValueError(msg)
    return float(-ndtri_exp(log(alpha) - log(2.0)))


def _r_from_dets(qxy, qxx, qyy):
    """Compute the partial correlation from the conditional covariance and two conditional variances."""
    return qxy / np.sqrt(np.abs(qxx * qyy))


def _k1_r(cxx, cyy, css, cxy, cxs, cys, eps: float = 0.0):
    inv = 1.0 / (css + eps)
    return _r_from_dets(
        cxy - cxs * cys * inv, cxx - cxs * cxs * inv, cyy - cys * cys * inv
    )


def _r_from_ldl(axx, ayy, axy, ux, uy, D, eps: float):
    """Compute the partial correlation using Schur complements of the form α − uᵀ(D+eps)⁻¹v."""
    inv = 1.0 / (D + eps)
    wx = ux * inv
    qxy = axy - np.sum(uy * wx, axis=-1)
    qxx = axx - np.sum(ux * wx, axis=-1)
    qyy = ayy - np.sum(uy * uy * inv, axis=-1)
    return _r_from_dets(qxy, qxx, qyy)


def _ldlt(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpivoted G = L D Lᵀ. ``G`` is ``(..., k, k)``; ``L`` unit lower, ``D`` ``(..., k)``."""
    G = np.asarray(G, dtype=np.float64)
    k = int(G.shape[-1])
    L = np.zeros(G.shape, dtype=np.float64)
    D = np.zeros(G.shape[:-1], dtype=np.float64)
    for j in range(k):
        acc = np.array(G[..., j, j], dtype=np.float64, copy=True)
        if j > 0:
            acc -= np.sum(L[..., j, :j] * L[..., j, :j] * D[..., :j], axis=-1)
        D[..., j] = acc
        L[..., j, j] = 1.0
        if j + 1 >= k:
            continue
        if j > 0:
            cross = np.sum(
                L[..., j + 1 :, :j]
                * L[..., j, :j][..., None, :]
                * D[..., :j][..., None, :],
                axis=-1,
            )
        else:
            cross = 0.0  # type: ignore[assignment]
        adj = G[..., j + 1 :, j] - cross
        den = D[..., j]
        Lj = np.zeros(adj.shape, dtype=np.float64)
        np.divide(adj, den[..., None], out=Lj, where=np.abs(den)[..., None] > 0.0)
        L[..., j + 1 :, j] = Lj
    return L, D


def _solve_unit_lower(L: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Forward substitution ``L U = rhs`` with unit-diagonal L. ``rhs`` is ``(..., k, nrhs)``."""
    U = np.array(rhs, dtype=np.float64, copy=True)
    k = int(L.shape[-1])
    for i in range(k):
        if i:
            U[..., i, :] -= np.einsum("...t,...tc->...c", L[..., i, :i], U[..., :i, :])
    return U


def partial_corr_r(
    corr: np.ndarray,
    x: int,
    y: int,
    Z: Sequence[int] | None = None,
    eps: float = 0.0,
) -> float:
    """Partial correlation r_{xy·Z} via LDLᵀ of G and (D+eps)⁻¹."""
    C = np.asarray(corr, dtype=np.float64)
    z = np.asarray([] if Z is None else list(Z), dtype=np.int64)
    axx = C[x, x]
    ayy = C[y, y]
    axy = C[x, y]
    k = int(z.size)
    if k == 0:
        return float(axy / sqrt(abs(axx * ayy)))
    if k == 1:
        s = int(z[0])
        return float(_k1_r(axx, ayy, C[s, s], axy, C[x, s], C[y, s], eps))
    cols = np.empty(k + 2, dtype=np.int64)
    cols[0] = x
    cols[1] = y
    cols[2:] = z
    block = C[np.ix_(z, cols)]
    L, D = _ldlt(block[:, 2:])
    U = _solve_unit_lower(L, block[:, :2])
    return float(_r_from_ldl(axx, ayy, axy, U[:, 0], U[:, 1], D, eps))


def partial_corr_r_fused(
    corr: np.ndarray,
    x: int,
    y: int,
    Z: Sequence[int] | None = None,
    eps: float = 0.0,
) -> float:
    """Compute the same partial correlation using fused row LDL and forward substitution, then a Schur complement with (D+eps)⁻¹."""
    C = np.asarray(corr, dtype=np.float64)
    z = np.asarray([] if Z is None else list(Z), dtype=np.int64)
    k = int(z.size)
    axx = float(C[x, x])
    ayy = float(C[y, y])
    axy = float(C[x, y])
    if k == 0:
        return axy / sqrt(abs(axx * ayy))
    L = np.zeros((k, k), dtype=np.float64)
    D = np.zeros(k, dtype=np.float64)
    u_x = np.zeros(k, dtype=np.float64)
    u_y = np.zeros(k, dtype=np.float64)
    for i in range(k):
        zi = int(z[i])
        g = np.empty(i + 1, dtype=np.float64)
        for j in range(i + 1):
            g[j] = C[zi, int(z[j])]
        ax = float(C[zi, x])
        ay = float(C[zi, y])
        for j in range(i):
            s = g[j]
            for t in range(j):
                s -= L[i, t] * L[j, t] * D[t]
            L[i, j] = s / D[j] if D[j] != 0.0 else 0.0
        s = g[i]
        for t in range(i):
            s -= L[i, t] * L[i, t] * D[t]
        D[i] = s
        L[i, i] = 1.0
        rx, ry = ax, ay
        for j in range(i):
            rx -= L[i, j] * u_x[j]
            ry -= L[i, j] * u_y[j]
        u_x[i] = rx
        u_y[i] = ry
    return float(_r_from_ldl(axx, ayy, axy, u_x, u_y, D, eps))


def _p_from_r(r: float, n_samples: int, k: int) -> float:
    dof = _fisher_dof(n_samples, k)
    if not np.isfinite(r):
        return float("nan")
    r = float(np.clip(r, -CUT_AT, CUT_AT))
    z = 0.5 * log((1.0 + r) / (1.0 - r))
    stat = sqrt(dof) * abs(z)
    return erfc(stat / sqrt(2.0))


def fisher_z_schur_chol(
    corr: np.ndarray,
    n_samples: int,
    x: int,
    y: int,
    Z: Sequence[int] | None = None,
    eps: float = 0.0,
) -> float:
    """Compute a Fisher-Z p-value from a Schur-complement partial correlation."""
    if Z is None:
        Z = ()
    Z = tuple(Z)
    r = partial_corr_r(corr, x, y, Z, eps=eps)
    return _p_from_r(r, n_samples, len(Z))


@overload
def _p_from_r_vec(r: np.ndarray, n_samples: int, k: int) -> np.ndarray: ...


@overload
def _p_from_r_vec(r: float, n_samples: int, k: int) -> float: ...


def _p_from_r_vec(r, n_samples: int, k: int) -> float | np.ndarray:
    if np.size(r) == 0:
        return np.asarray(r, dtype=np.float64)
    dof = _fisher_dof(n_samples, k)
    if np.ndim(r) == 0:
        return _p_from_r(float(r), n_samples, k)
    r = np.asarray(r, dtype=np.float64)
    finite = np.isfinite(r)
    r = np.clip(r, -CUT_AT, CUT_AT)
    z = 0.5 * np.log((1.0 + r) / (1.0 - r))
    stat = np.sqrt(dof) * np.abs(z)
    return np.where(finite, erfc_vec(stat / np.sqrt(2.0)), np.nan)


def _as_batch(x, y, S) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    x = np.asarray(x, dtype=np.int64).reshape(-1)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    S = np.asarray(S, dtype=np.int64)
    if S.ndim == 1:
        S = S.reshape(-1, 1) if S.size else np.zeros((x.size, 0), dtype=np.int64)
    if S.shape[0] != x.size or y.size != x.size:
        msg = "x, y, S batch dimensions must match"
        raise ValueError(msg)
    return x, y, S, int(S.shape[1])


def partial_corr_r_batch_inv(
    corr: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """Compute partial correlations using a pseudoinverse if inversion fails."""
    C = np.asarray(corr, dtype=np.float64)
    x, y, S, k = _as_batch(x, y, S)
    B = x.size
    if B == 0:
        return np.zeros(0, dtype=np.float64)
    if k == 0:
        return C[x, y] / np.sqrt(np.abs(C[x, x] * C[y, y]))
    idx = np.concatenate([x[:, None], y[:, None], S], axis=1)
    sub = C[idx[:, :, None], idx[:, None, :]]
    finite = np.isfinite(sub).all(axis=(1, 2))
    result = np.full(B, np.nan, dtype=np.float64)
    if not finite.any():
        return result
    try:
        inv = np.linalg.inv(sub[finite])
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(sub[finite])
    result[finite] = -inv[:, 0, 1] / np.sqrt(np.abs(inv[:, 0, 0] * inv[:, 1, 1]))
    return result


def partial_corr_r_batch_schur(
    corr: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    S: np.ndarray,
    eps: float = 0.0,
) -> np.ndarray:
    """Batched G = L D Lᵀ, L [u_x u_y] = [a_x a_y], scale by (D+eps)⁻¹."""
    C = np.asarray(corr, dtype=np.float64)
    x, y, S, k = _as_batch(x, y, S)
    B = x.size
    if B == 0:
        return np.zeros(0, dtype=np.float64)
    axx = C[x, x]
    ayy = C[y, y]
    axy = C[x, y]
    if k == 0:
        return axy / np.sqrt(np.abs(axx * ayy))
    if k == 1:
        s = S[:, 0]
        return _k1_r(axx, ayy, C[s, s], axy, C[x, s], C[y, s], eps)
    G = C[S[:, :, None], S[:, None, :]]
    a = np.stack([C[S, x[:, None]], C[S, y[:, None]]], axis=-1)
    L, D = _ldlt(G)
    U = _solve_unit_lower(L, a)
    return _r_from_ldl(axx, ayy, axy, U[:, :, 0], U[:, :, 1], D, eps)


def fisher_z_batch_inv(corr, n_samples: int, x, y, S) -> np.ndarray:
    """Fisher-Z p-values for a batch of tests using the inverse-correlation form."""
    x, y, S, k = _as_batch(x, y, S)
    r = partial_corr_r_batch_inv(corr, x, y, S)
    return _p_from_r_vec(r, n_samples, k)


def fisher_z_batch_schur(corr, n_samples: int, x, y, S, eps: float = 0.0) -> np.ndarray:
    """Fisher-Z p-values for a batch of tests using the Schur-complement LDLᵀ form."""
    x, y, S, k = _as_batch(x, y, S)
    r = partial_corr_r_batch_schur(corr, x, y, S, eps=eps)
    return _p_from_r_vec(r, n_samples, k)


class NumpyFisherZ:
    """Batch Fisher-Z with canonical or experimental numerical policies."""

    def __init__(
        self,
        corr: np.ndarray,
        n_samples: int,
        method: str = "schur",
        eps: float = 0.0,
        *,
        decision_alpha: float | None = None,
    ):
        if method not in ("schur", "inv", "canonical"):
            msg = f"method must be 'schur', 'inv' or 'canonical', got {method!r}"
            raise ValueError(msg)
        if method == "canonical" and eps != 0.0:
            msg = "canonical Fisher-Z does not support ridge regularization"
            raise ValueError(msg)
        self.corr = np.asarray(corr, dtype=np.float64)
        self.n_samples = int(n_samples)
        self.method = method
        self.eps = float(eps)
        self.decision_alpha = decision_alpha

    @classmethod
    def for_cuda_fallback(cls, corr, n_samples, eps=0.0):
        """Choose the CPU Fisher-Z implementation for the requested CUDA regularization."""
        return cls(
            corr,
            n_samples,
            method="canonical" if eps == 0.0 else "schur",
            eps=eps,
        )

    def batch(self, x, y, S) -> np.ndarray:
        """Return Fisher-Z p-values using this instance's correlation matrix and method."""
        if self.method == "canonical":
            from .independence_tests import fisher_z_from_corr

            x, y, S, _ = _as_batch(x, y, S)
            return np.fromiter(
                (
                    fisher_z_from_corr(
                        self.corr,
                        self.n_samples,
                        int(a),
                        int(b),
                        tuple(map(int, conditions)),
                        decision_alpha=self.decision_alpha,
                    )
                    for a, b, conditions in zip(x, y, S, strict=False)
                ),
                dtype=np.float64,
                count=x.size,
            )
        if self.method == "inv":
            return fisher_z_batch_inv(self.corr, self.n_samples, x, y, S)
        return fisher_z_batch_schur(self.corr, self.n_samples, x, y, S, eps=self.eps)


def fisher_z_schur_chol_fused(
    corr: np.ndarray,
    n_samples: int,
    x: int,
    y: int,
    Z: Sequence[int] | None = None,
    eps: float = 0.0,
) -> float:
    """Compute a Fisher-Z p-value using the fused Schur-complement calculation."""
    if Z is None:
        Z = ()
    Z = tuple(Z)
    r = partial_corr_r_fused(corr, x, y, Z, eps=eps)
    return _p_from_r(r, n_samples, len(Z))
