"""Fisher-Z batcher with CPU fallback and lazy CUDA imports."""

from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from .cpu_fisher_z import NumpyFisherZ, _as_batch
from .cpu_fisher_z import _p_from_r_vec as _p_value_from_r

if TYPE_CHECKING:
    from .cuda_fisher_z import CudaFisherZ as _CudaFisherBase
else:
    _CudaFisherBase = object

_CUDA_CLASS: "type[CudaFisherZ] | None" = None


def _cuda_class() -> "type[CudaFisherZ]":
    global _CUDA_CLASS
    if _CUDA_CLASS is None:
        from .cuda_fisher_z import CudaFisherZ as OriginalCudaFisherZ

        _CUDA_CLASS = cast(
            "type[CudaFisherZ]",
            type(
                "_BoundCudaFisherZ",
                (CudaFisherZ, OriginalCudaFisherZ),
                {"__module__": __name__},
            ),
        )
    return _CUDA_CLASS


class _CPUReference(Protocol):
    decision_alpha: float | None

    def __call__(self, x: int, y: int, conditions: tuple[int, ...], /) -> float: ...


class _CITFisherBatch:
    """Adapt the configured CPU tester without losing its cache or guards."""

    def __init__(self, test: _CPUReference):
        self.test = test

    def batch(self, x, y, S) -> np.ndarray:
        x, y, S, _ = _as_batch(x, y, S)
        return np.fromiter(
            (
                self.test(int(a), int(b), tuple(map(int, conditions)))
                for a, b, conditions in zip(x, y, S, strict=False)
            ),
            dtype=np.float64,
            count=x.size,
        )


class CudaFisherZ(_CudaFisherBase):
    """Use the configured CI test for unregularized CPU fallback."""

    _bridge_cpu_impl: NumpyFisherZ | _CITFisherBatch | None = None
    cpu_test: _CPUReference | None = None
    decision_alpha: float | None = None

    def __new__(cls, *args, **kwargs):
        """Allocate the CUDA implementation when constructing the public bridge class."""
        return object.__new__(_cuda_class() if cls is CudaFisherZ else cls)

    @classmethod
    def from_data(cls, *args, **kwargs):
        """Build the bridge from observations using the upstream CUDA factory."""
        from .cuda_fisher_z import CudaFisherZ as OriginalCudaFisherZ

        bound_class = _cuda_class() if cls is CudaFisherZ else cls
        # Bind the upstream factory to the adapter class.
        factory = OriginalCudaFisherZ.__dict__["from_data"]
        return factory.__func__(bound_class, *args, **kwargs)

    @property
    def _cpu(self) -> NumpyFisherZ | _CITFisherBatch:
        tester = self.cpu_test
        alpha = self.decision_alpha
        if alpha is None:
            alpha = tester.decision_alpha if tester is not None else None
        implementation = self._bridge_cpu_impl
        if tester is not None and self.eps == 0.0:
            from .ci_batch import configure_fisher_alpha

            configure_fisher_alpha(tester, alpha)
            if not isinstance(implementation, _CITFisherBatch) or (
                implementation.test is not tester
            ):
                implementation = self._bridge_cpu_impl = _CITFisherBatch(tester)
            return implementation
        if not isinstance(implementation, NumpyFisherZ):
            implementation = self._bridge_cpu_impl = NumpyFisherZ.for_cuda_fallback(
                self.host_corr(), self.n_samples, eps=self.eps
            )
        implementation.decision_alpha = alpha
        return implementation

    def _validate_dof(self, k: int) -> None:
        if self.n_samples - k - 3 <= 0:
            msg = (
                "Fisher-Z requires positive degrees of freedom: "
                f"n={self.n_samples}, |S|={k}"
            )
            raise ValueError(msg)

    def batch(self, x, y, S) -> np.ndarray:
        """Return Fisher-Z p-values for a batch of node pairs and conditioning sets."""
        x_t, y_t, S_t, B, k = self._prepare_batch(x, y, S)
        if B == 0:
            return np.empty(0, dtype=np.float64)
        self._validate_dof(k)
        if k > self._k_max:
            return self._cpu.batch(
                x_t.detach().cpu().numpy(),
                y_t.detach().cpu().numpy(),
                S_t.detach().cpu().numpy(),
            )
        partial = self.batch_r(x_t, y_t, S_t)
        return _p_value_from_r(partial.detach().cpu().numpy(), self.n_samples, k)

    def batch_pvalues(self, x, y, S) -> np.ndarray:
        """Accept one endpoint index or one per conditioning-set row."""
        S = np.asarray(S, dtype=np.int32)
        if S.ndim == 1:
            S = S.reshape(S.size, 1) if S.size else S.reshape(0, 0)
        B = int(S.shape[0])
        if B == 0:
            return np.empty(0, dtype=np.float64)
        k = int(S.shape[1])
        self._validate_dof(k)
        xs = np.asarray(x, dtype=np.int32).reshape(-1)
        ys = np.asarray(y, dtype=np.int32).reshape(-1)
        if xs.size == 1:
            xs = np.full(B, int(xs[0]), dtype=np.int32)
        if ys.size == 1:
            ys = np.full(B, int(ys[0]), dtype=np.int32)
        if xs.size != B or ys.size != B:
            msg = "x, y, S batch dimensions must match"
            raise ValueError(msg)
        if k > self._k_max:
            return self._cpu.batch(xs, ys, S)
        return self.batch(xs, ys, S)
