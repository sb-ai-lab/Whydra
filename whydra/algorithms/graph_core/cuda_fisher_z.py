"""CUDA Fisher-Z. Correlation matrix stays on device.

k=0..3 are specialized LDLᵀ kernels; k≥4 uses a generic ``K_MAX=96`` kernel
with runtime ``k`` (conditioner blocks up to 96×96) and a tunable
threads-per-test. The kernel gathers
``C[i,j]`` from device ``C`` using its strides plus index arrays
``(x, y, S)``. For k≥2, large (all-S) batches unique ``S`` and factor
``C[S,S]`` once per conditioner. Host p-values are float64 (clip / Fisher Z /
``norm.cdf``).
"""

import math
import sys
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch
from scipy.special import ndtr

from .fisher_z_schur_chol import NumpyFisherZ

CUT_AT = 0.9999999
# All-S k≥2 batches are millions of tests; wavefront launches are a few thousand
# pairs. Grouping by S is a win only when the batch is large.
GROUP_S_MIN_B = 4096
GROUP_S_MIN_K = 2

_LDLT_CUDA = None


def _ldlt_cuda():
    """Prebuilt batched LDLᵀ Fisher-Z (``_whydra_fisher_z_ldlt``)."""
    global _LDLT_CUDA
    if _LDLT_CUDA is None:
        from ._cuda import fisher_z

        _LDLT_CUDA = fisher_z
    return _LDLT_CUDA


_DTYPE_ALIASES = {
    "float64": torch.float64,
    "fp64": torch.float64,
    "f64": torch.float64,
}


def resolve_ci_dtype(dtype: str | torch.dtype | None = None) -> torch.dtype:
    """Return torch.float64 for None, torch.float64, or a float64 alias.

    Accepted strings are float64, fp64, and f64 (case-insensitive).
    This Python interface accepts only float64, even if float32 kernels are built.
    """
    if dtype is None:
        return torch.float64
    if isinstance(dtype, torch.dtype):
        if dtype == torch.float64:
            return dtype
        msg = (
            f"unsupported ci dtype {dtype!r}; CUDA CI is float64 only "
            "(rebuild with WHYDRA_ENABLE_FLOAT32=1 to restore float32)"
        )
        raise ValueError(msg)
    key = str(dtype).lower().strip()
    if key in ("float32", "fp32", "f32"):
        msg = (
            "CUDA CI is float64 only (rebuild with WHYDRA_ENABLE_FLOAT32=1 "
            "to restore float32)"
        )
        raise ValueError(msg)
    if key not in _DTYPE_ALIASES:
        msg = f"unsupported ci dtype {dtype!r}; use float64"
        raise ValueError(msg)
    return _DTYPE_ALIASES[key]


def _p_value_from_r(r, n_samples: int, k: int):
    """Host Fisher-Z p-values from partial correlations (float64).

    Device fused kernels skip this and compare ``√df |Z|`` to ``τ`` instead.
    ``scipy.special.ndtr(x)`` is bit-identical to ``scipy.stats.norm.cdf(x)``
    (the latter is a thin, validated wrapper around it) but ~20× cheaper for
    the small batches the Possible-D-SEP driver issues thousands of times, and
    ~3× cheaper for the full-depth skeleton batches.
    """
    r = np.clip(np.asarray(r, dtype=np.float64), -CUT_AT, CUT_AT)
    Z = 0.5 * np.log((1.0 + r) / (1.0 - r))
    stat = np.sqrt(n_samples - k - 3) * np.abs(Z)
    p = 2.0 * (1.0 - ndtr(stat))
    if np.ndim(p) == 0:
        return float(p)
    return p


def corrcoef_from_data(
    data: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device = "cuda",
    dtype: str | torch.dtype | None = None,
) -> torch.Tensor:
    """Pearson ``C`` on GPU. Matches ``np.corrcoef(data.T)`` (``n×p`` samples×vars).

    Centers columns, ``C = (XᵀX) / (σ ⊗ σ)``. Degenerate columns stay NaN.
    CUDA: cuBLAS GEMM plus elementwise mean / sqrt / divide; default ``float64``.
    """
    dt = resolve_ci_dtype(dtype)
    dev = torch.device(device)
    if torch.is_tensor(data):
        X = data.to(device=dev, dtype=dt)  # type: ignore[union-attr]
    else:
        X = torch.as_tensor(np.ascontiguousarray(data), device=dev, dtype=dt)
    if X.dim() != 2:
        msg = "data must be n\N{MULTIPLICATION SIGN}p"
        raise ValueError(msg)
    if int(X.size(0)) < 2:
        msg = "need at least 2 samples for corrcoef"
        raise ValueError(msg)
    X = X - X.mean(dim=0, keepdim=True)
    gram = X.T @ X
    std = gram.diag().clamp(min=0).sqrt()
    C = gram / std.unsqueeze(1) / std.unsqueeze(0)
    finite = std > 0
    diag = torch.where(finite, torch.ones_like(std), C.diag())
    C.diagonal().copy_(diag)
    if not C.is_contiguous():
        C = C.contiguous()
    return C


class _FisherBatch(Protocol):
    def batch(self, x, y, S, /) -> np.ndarray: ...


class CudaFisherZ:
    """Fisher-Z batcher on GPU (LDLᵀ). ``dtype`` is ``float64``.

    ``eps`` is the diagonal ridge ``(D+eps)⁻¹`` (same as CPU ``NumpyFisherZ``).
    ``batch_r`` launches ``fisher_z_ldlt.cu``: CTA size 256, one thread per test
    for ``k≤3`` (``TPT=1``), then 8/16/32 threads per test for larger ``k``.
    Large ``k`` packs fewer tests per CTA so the ``k×k`` factor fits
    in opt-in shared memory. The fused skeleton inlines LDLᵀ for ``k≤32``;
    ``k=33..96`` uses fill + ``batch_r``.
    """

    def __init__(
        self,
        corr: np.ndarray | torch.Tensor,
        n_samples: int,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
        eps: float = 0.0,
    ):
        if not torch.cuda.is_available():
            msg = "CudaFisherZ requires CUDA"
            raise RuntimeError(msg)
        self.device = torch.device(device)
        self.n_samples = int(n_samples)
        self.dtype = resolve_ci_dtype(dtype)
        self.eps = float(eps)
        self._corr_np: np.ndarray | None
        if torch.is_tensor(corr):
            self.corr = corr.to(device=self.device, dtype=self.dtype)  # type: ignore[union-attr]
            self._corr_np = None
        else:
            self._corr_np = np.ascontiguousarray(np.asarray(corr))
            self.corr = torch.as_tensor(
                self._corr_np, device=self.device, dtype=self.dtype
            )
        if self.corr.dim() != 2 or int(self.corr.size(0)) != int(self.corr.size(1)):
            msg = "corr must be a square matrix"
            raise ValueError(msg)
        self._cpu_impl: NumpyFisherZ | None = None
        self._ldlt = _ldlt_cuda()
        self._k_max = int(self._ldlt.k_max())

    @classmethod
    def from_data(
        cls,
        data: np.ndarray | torch.Tensor,
        n_samples: int | None = None,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
        eps: float = 0.0,
    ) -> "CudaFisherZ":
        """Upload ``n×p`` samples and form ``C`` on device (no host ``corrcoef``).

        Corrcoef is a GEMM on the CI dtype; the batcher then keeps ``C`` resident.
        """
        C = corrcoef_from_data(data, device=device, dtype=dtype)
        if n_samples is None:
            n_samples = int(data.shape[0])
        return cls(C, int(n_samples), device=device, dtype=dtype, eps=eps)

    def host_corr(self) -> np.ndarray:
        """Return the cached p×p NumPy correlation matrix, copying from the device if needed."""
        if self._corr_np is None:
            self._corr_np = np.ascontiguousarray(self.corr.detach().cpu().numpy())
        return self._corr_np

    @property
    def _cpu(self) -> _FisherBatch:
        if self._cpu_impl is None:
            self._cpu_impl = NumpyFisherZ(
                self.host_corr(), self.n_samples, method="schur", eps=self.eps
            )
        return self._cpu_impl

    def __call__(
        self, x: int, y: int, condition_set: Sequence[int] | None = None
    ) -> float:
        """Single-test p-value via a one-row ``batch`` (device LDLᵀ, host ``cdf``)."""
        if condition_set is None:
            condition_set = []
        S = np.asarray(list(condition_set), dtype=np.int64).reshape(1, -1)
        p = self.batch(
            np.array([x], dtype=np.int64),
            np.array([y], dtype=np.int64),
            S,
        )
        return float(p[0])

    def _to_int32_cuda(self, t, *, ndim: int) -> torch.Tensor:
        if torch.is_tensor(t):
            out = t.to(device=self.device, dtype=torch.int32)
        else:
            out = torch.as_tensor(np.asarray(t, dtype=np.int32), device=self.device)
        if ndim == 1:
            out = out.reshape(-1)
        if not out.is_contiguous():
            out = out.contiguous()
        return out

    def _prepare_batch(self, x, y, S):
        x_t = self._to_int32_cuda(x, ndim=1)
        y_t = self._to_int32_cuda(y, ndim=1)
        B = int(x_t.numel())
        if torch.is_tensor(S):
            S_t = S.to(device=self.device, dtype=torch.int32)
            if S_t.ndim == 1:
                S_t = S_t.reshape(-1, 1) if S_t.numel() else S_t.reshape(B, 0)
        else:
            S_np = np.asarray(S, dtype=np.int32)
            if S_np.ndim == 1:
                S_np = (
                    S_np.reshape(-1, 1)
                    if S_np.size
                    else np.zeros((B, 0), dtype=np.int32)
                )
            S_t = torch.as_tensor(S_np, device=self.device)
        if B == 0:
            k = int(S_t.size(1)) if S_t.ndim == 2 else 0
            S_t = torch.empty((0, k), dtype=torch.int32, device=self.device)
            return x_t, y_t, S_t, 0, k
        S_t = S_t.reshape(B, -1)
        if not S_t.is_contiguous():
            S_t = S_t.contiguous()
        if int(y_t.numel()) != B or int(S_t.size(0)) != B:
            msg = "x, y, S batch dimensions must match"
            raise ValueError(msg)
        return x_t, y_t, S_t, B, int(S_t.size(1))

    def _unique_conditioners(
        self, S: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the unique rows of ``S`` (U, k) and their inverse row indices (B,)."""
        k = int(S.size(1))
        p = int(self.corr.size(0))
        if p > 1 and k > 0 and k * math.log(p) < 62.0 * math.log(2.0):
            key = S[:, 0].to(torch.int64)
            for j in range(1, k):
                key = key * p + S[:, j].to(torch.int64)
            ukey, inverse = torch.unique(key, return_inverse=True)
            U = int(ukey.numel())
            unique_S = torch.empty((U, k), dtype=S.dtype, device=S.device)
            tmp = ukey
            for j in range(k - 1, -1, -1):
                unique_S[:, j] = (tmp % p).to(S.dtype)
                tmp = torch.div(tmp, p, rounding_mode="floor")
            return unique_S.contiguous(), inverse.to(torch.int32).contiguous()
        unique_S, inverse = torch.unique(S, dim=0, return_inverse=True)
        return unique_S.contiguous(), inverse.to(torch.int32).contiguous()

    def _batch_r_grouped(
        self,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        S_t: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        unique_S, inverse = self._unique_conditioners(S_t)
        U = int(unique_S.size(0))
        k = int(S_t.size(1))
        L = torch.empty((U, k, k), device=self.device, dtype=self.dtype)
        D = torch.empty((U, k), device=self.device, dtype=self.dtype)
        self._ldlt.ldlt_factor(self.corr, unique_S, L, D, self.eps)
        self._ldlt.ldlt_apply(
            self.corr, x_t, y_t, unique_S, inverse, L, D, out, self.eps
        )
        return out

    def _batch_r_from_unique(
        self,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        unique_S: torch.Tensor,
        inverse: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        unique_S = unique_S.to(device=self.device, dtype=torch.int32)
        if not unique_S.is_contiguous():
            unique_S = unique_S.contiguous()
        inverse = inverse.to(device=self.device, dtype=torch.int32)
        if not inverse.is_contiguous():
            inverse = inverse.contiguous()
        U = int(unique_S.size(0))
        k = int(unique_S.size(1))
        L = torch.empty((U, k, k), device=self.device, dtype=self.dtype)
        D = torch.empty((U, k), device=self.device, dtype=self.dtype)
        self._ldlt.ldlt_factor(self.corr, unique_S, L, D, self.eps)
        self._ldlt.ldlt_apply(
            self.corr, x_t, y_t, unique_S, inverse, L, D, out, self.eps
        )
        return out

    def batch_r(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        S: np.ndarray | torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        group_s: bool | None = None,
        unique_S: torch.Tensor | None = None,
        inverse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return partial correlations for B tests as a tensor on the device.

        The int32 vectors x and y identify the two nodes in each test. The
        (B, k) matrix S contains the k conditioning indices per test.
        The default kernel factors each test's conditioning matrix separately
        with 256 threads per block. For k≥2, pass ``unique_S`` and
        ``inverse`` to share an LDLᵀ factorization across tests that use the
        same conditioning set. If the unique rows are unavailable,
        ``group_s=True`` computes them from S. The correlation matrix and
        result use float64.
        """
        x_t, y_t, S_t, B, k = self._prepare_batch(x, y, S)
        if B == 0:
            if out is not None:
                return out
            return torch.empty(0, device=self.device, dtype=self.dtype)
        if k > self._k_max:
            print(
                f"[cuda fisher_z] k={k} > k_max={self._k_max} "
                f"(k>=97 exceeds CUDA LDLT)",
                file=sys.stderr,
                flush=True,
            )
            msg = f"k={k} exceeds CUDA K_MAX={self._k_max}"
            raise ValueError(msg)
        if out is None:
            out = torch.empty(B, device=self.device, dtype=self.dtype)
        elif int(out.numel()) < B:
            msg = "batch_r out is too small"
            raise ValueError(msg)
        else:
            out = out.reshape(-1)[:B]
            if not out.is_contiguous():
                out = out.contiguous()
        if unique_S is not None and k >= GROUP_S_MIN_K:
            if inverse is None:
                msg = "unique_S requires inverse"
                raise ValueError(msg)
            return self._batch_r_from_unique(x_t, y_t, unique_S, inverse, out)
        if group_s is None:
            group_s = False
        elif group_s:
            group_s = k >= GROUP_S_MIN_K
        if group_s:
            return self._batch_r_grouped(x_t, y_t, S_t, out)
        self._ldlt.ldlt_batch(self.corr, x_t, y_t, S_t, out, self.eps)
        return out

    def batch(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        S: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Vectorized Fisher-Z. ``x,y`` shape (B,), ``S`` shape (B, k) with k>=0.

        ``C`` stays on device. The kernel gathers ``C[i,j]`` from indices using
        ``C``'s strides. Host work is index transfer (if needed) and p-values.

        This is the per-test path used by the Possible-D-SEP driver, where a
        call carries a handful of tests but happens thousands of times; it
        inlines the single ``ldlt_batch`` launch instead of routing through
        ``batch_r`` (which would re-run ``_prepare_batch`` on the already
        prepared tensors).
        """
        x_t, y_t, S_t, B, k = self._prepare_batch(x, y, S)
        if B == 0:
            return np.array([], dtype=np.float64)
        if k > self._k_max:
            print(
                f"[cuda fisher_z] k={k} > k_max={self._k_max} "
                f"(k>=97 exceeds CUDA LDLT); CPU fallback",
                file=sys.stderr,
                flush=True,
            )
            return self._cpu.batch(
                x_t.detach().cpu().numpy(),
                y_t.detach().cpu().numpy(),
                S_t.detach().cpu().numpy(),
            )
        out = torch.empty(B, device=self.device, dtype=self.dtype)
        self._ldlt.ldlt_batch(self.corr, x_t, y_t, S_t, out, self.eps)
        return np.asarray(
            _p_value_from_r(out.detach().cpu().numpy(), self.n_samples, k)
        )

    def batch_pvalues(
        self,
        x: int | np.ndarray,
        y: int | np.ndarray,
        S: np.ndarray,
    ) -> np.ndarray:
        """Lean Fisher-Z p-values for a small, repeated index batch.

        Semantically identical to ``batch`` but tuned for the Possible-D-SEP
        driver: thousands of calls carrying a handful of tests each, where the
        per-call host/torch overhead (three ``as_tensor`` conversions, double
        ``_prepare_batch``) dwarfs the kernel. The ``x`` / ``y`` / ``S``
        indices are packed into ONE contiguous ``int32`` buffer, moved to the
        device with a single ``as_tensor``, and sliced into the contiguous
        ``x`` / ``y`` / ``S`` views the kernel requires; the output reuses a
        cached device buffer. ``x``/``y`` may be scalars applied to every row.
        """
        S = np.asarray(S, dtype=np.int32)
        if S.ndim == 1:
            S = S.reshape(S.size, 1) if S.size else S.reshape(0, 0)
        B = int(S.shape[0])
        if B == 0:
            return np.empty(0, dtype=np.float64)
        k = int(S.shape[1])
        if k > self._k_max:
            return self.batch(np.asarray(x), np.asarray(y), S)
        xs = np.asarray(x, dtype=np.int32).reshape(-1)
        ys = np.asarray(y, dtype=np.int32).reshape(-1)
        if xs.size == 1:
            xs = np.full(B, int(xs[0]), dtype=np.int32)
        if ys.size == 1:
            ys = np.full(B, int(ys[0]), dtype=np.int32)
        comb = np.empty(2 * B + B * k, dtype=np.int32)
        comb[:B] = xs
        comb[B : 2 * B] = ys
        if k:
            comb[2 * B :] = S.reshape(-1)
        t = torch.as_tensor(comb, device=self.device)
        x_t = t[:B]
        y_t = t[B : 2 * B]
        S_t = t[2 * B :].reshape(B, k)
        out = self._out_buf(B)
        self._ldlt.ldlt_batch(self.corr, x_t, y_t, S_t, out[:B], self.eps)
        return np.asarray(
            _p_value_from_r(out[:B].detach().cpu().numpy(), self.n_samples, k)
        )

    def _out_buf(self, B: int) -> torch.Tensor:
        buf = getattr(self, "_out_cache", None)
        if buf is None or int(buf.numel()) < B:
            buf = torch.empty(max(B, 64), device=self.device, dtype=self.dtype)
            self._out_cache = buf
        return buf
