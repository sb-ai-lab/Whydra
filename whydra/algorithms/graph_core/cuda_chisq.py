"""CUDA PC/FCI χ².

Tiled V×C SMEM histograms on device (V ∈ {4,8,16,32}). k=0 and k=1
walk X-CSR strips and accumulate Pearson as n Σ Q_b/c_b − n. k=1
does that per Z-stratum (drop empty rows/cols). k≥2 counting-sorts
samples into a packed-Z CSR when C^k ≤ 1024, otherwise LSD by each
Z column, then reuses the k=1 Pearson on occupied runs (≤ n).

When the k≥2 SMEM index buffers do not fit (n=5000 on a 64 KB part),
``CudaChisqHost`` moves them to a cached global-memory workspace with
persistent CTAs (``WHYDRA_CHISQ_GMEM=1``, the default). The sequential
``CIT.chi_square`` loop is used only if GMEM is disabled or its
workspace cannot be allocated. A k=1 GMEM variant is available behind
``WHYDRA_CHISQ_GMEM_K1=1``; it is faster only for some (n, C) and is
not the default.
"""

import os
import sys
from collections.abc import Sequence

import numpy as np
import torch

from .independence_tests import CIT


def _csr_from_data_int(
    data_int: np.ndarray, cards: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Once-per-dataset CSR: sample indices grouped by sorted unique id.

    ``offsets[j, u] : offsets[j, u+1]`` are rows ``t`` with ``data_int[t, j] == u``.
    Uniques are already ``0 .. c_j-1`` after ``CIT`` recoding.
    """
    data_int = np.asarray(data_int, dtype=np.int32)
    cards = np.asarray(cards, dtype=np.int32)
    n, p = data_int.shape
    C = int(cards.max()) if p else 1
    offsets = np.zeros((p, C + 1), dtype=np.int32)
    indices = np.empty((p, n), dtype=np.int32)
    for j in range(p):
        col = data_int[:, j]
        cj = int(cards[j])
        counts = np.bincount(col, minlength=max(cj, 1))
        offsets[j, 1 : cj + 1] = np.cumsum(counts)
        if cj < C:
            offsets[j, cj + 1 :] = n
        indices[j] = np.argsort(col, kind="stable").astype(np.int32)
    return offsets, indices


_CHISQ_CUDA = None

# Global-memory fallback for k>=2 when the SMEM index buffers do not fit.
# One persistent CTA per workspace slot; the CTA loops over its share of tests.
_GMEM_GRID = 512
_GMEM_BUDGET_BYTES = 64 << 20


def _gmem_enabled() -> bool:
    """Whether k>=2 should try the global-memory kernel before the host path."""
    s = os.environ.get("WHYDRA_CHISQ_GMEM")
    if s is None:
        return True
    return not (s.startswith("0") and len(s) == 1)


def _gmem_k1_enabled() -> bool:
    """Opt-in k=1 global-memory variant (measured, but not a consistent win)."""
    return os.environ.get("WHYDRA_CHISQ_GMEM_K1", "0") not in (
        "0",
        "",
        "false",
        "False",
    )


def _gmem_grid(B: int, n: int) -> int:
    per_cta = 2 * 4 * max(int(n), 1)
    by_mem = max(1, _GMEM_BUDGET_BYTES // per_cta)
    return max(1, min(int(B), _GMEM_GRID, by_mem))


def is_host_ci(batcher) -> bool:
    """Check whether CI uses χ² rather than the fused Fisher-Z LDLᵀ test."""
    return getattr(batcher, "ci_method", None) == "chisq"


def _as_numpy(t) -> np.ndarray:
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _chisq_cuda():
    global _CHISQ_CUDA
    if _CHISQ_CUDA is None:
        from ._cuda import chisq

        _CHISQ_CUDA = chisq
    return _CHISQ_CUDA


class CudaChisqHost:
    """χ² batcher: CUDA k=0, k=1, and k≥2 over observed Z packs.

    ``device`` is the graph device. Recoded ``n×p`` codes are stored
    column-major ``uint16`` on the GPU (cardinality up to 1024). A column
    CSR of sample indices (grouped by unique id) is built once from
    ``data_int``. ``tile_v`` in ``{4,8,16,32}`` forces that strip height;
    ``None`` uses ``tile_v_for_card(C)``. k≥2 counting-sorts into a
    packed-Z CSR when C^k ≤ 1024, otherwise LSD by each conditioner
    column, and sums Pearson χ² over occupied strata (≤ n), not the
    Cartesian product of conditioner cards. If the two n-element index
    buffers do not fit SMEM, the same kernel runs with them in a cached
    global-memory workspace; the host loop is the last resort.
    """

    ci_method = "chisq"

    def __init__(
        self,
        cit: CIT,
        device: str | torch.device = "cuda",
        tile_v: int | None = None,
    ):
        if not torch.cuda.is_available():
            msg = "CudaChisqHost requires CUDA"
            raise RuntimeError(msg)
        if getattr(cit, "method", None) != "chisq":
            msg = f"CudaChisqHost needs CIT(method='chisq'), got {getattr(cit, 'method', None)!r}"
            raise ValueError(msg)
        self.device = torch.device(device)
        self.cit = cit
        self.n_samples = int(cit.n_samples)
        self.dtype = torch.float64
        self.eps = 0.0
        self.corr = None
        from .cuda_graph_batch import _graph_batch_cuda

        self._k_max = int(_graph_batch_cuda().k_max())
        ext = _chisq_cuda()
        self._c_full = int(ext.c_full())
        cards = np.asarray(cit.cardinalities, dtype=np.int32)
        if cards.size and int(cards.max()) > 1023:  # type: ignore[call-overload]
            msg = "CudaChisqHost stores codes as uint16; cardinality > 1023"
            raise ValueError(msg)
        colmaj = np.ascontiguousarray(cit.data_int.T, dtype=np.uint16)
        self._data = torch.as_tensor(colmaj, device=self.device)
        self._cards = torch.as_tensor(cards, device=self.device)
        self._cards_np = cards
        csr_off, csr_idx = _csr_from_data_int(cit.data_int, cards)
        self._csr_off = torch.as_tensor(csr_off, device=self.device)
        self._csr_idx = torch.as_tensor(csr_idx, device=self.device)
        self._ext = ext
        max_card = int(cards.max()) if cards.size else 1  # type: ignore[call-overload]
        self._max_card = max_card
        if tile_v is None:
            self._tile_v = 0
        else:
            if tile_v not in (4, 8, 16, 32):
                msg = "tile_v must be 4, 8, 16, or 32"
                raise ValueError(msg)
            self._tile_v = int(tile_v)
        self._ws: tuple[torch.Tensor, torch.Tensor] | None = None
        self._ws_key: tuple[object, ...] | None = None

    def _workspace(self, grid: int):
        """Two ``int32`` index buffers per workspace slot, cached by (grid, n)."""
        key = (int(grid), int(self.n_samples), self.device)
        if key == self._ws_key:
            return self._ws
        size = int(grid) * int(self.n_samples)
        try:
            self._ws = (
                torch.empty(size, dtype=torch.int32, device=self.device),
                torch.empty(size, dtype=torch.int32, device=self.device),
            )
        except torch.cuda.OutOfMemoryError:
            self._ws = None
        self._ws_key = key
        return self._ws

    @classmethod
    def from_data(
        cls,
        data: np.ndarray,
        device: str | torch.device = "cuda",
        tile_v: int | None = None,
    ) -> "CudaChisqHost":
        """Build a CUDA chi-square tester from integer-coded data."""
        return cls(
            CIT(np.asarray(data), method="chisq"),
            device=device,
            tile_v=tile_v,
        )

    def __call__(
        self, x: int, y: int, condition_set: Sequence[int] | None = None
    ) -> float:
        """Return the chi-square p-value for a single conditional independence request."""
        if condition_set is None:
            condition_set = []
        p = self.batch(
            np.array([x], dtype=np.int64),
            np.array([y], dtype=np.int64),
            np.asarray(condition_set, dtype=np.int64).reshape(1, -1),
        )
        return float(p[0])

    def _to_int32_cuda(self, t) -> torch.Tensor:
        if torch.is_tensor(t):
            out = t.to(device=self.device, dtype=torch.int32).reshape(-1)
        else:
            out = torch.as_tensor(
                np.asarray(t, dtype=np.int32).reshape(-1), device=self.device
            )
        return out.contiguous()

    def _z_from_S(self, S, B: int) -> torch.Tensor:
        if torch.is_tensor(S):
            t = S.to(device=self.device, dtype=torch.int32)
            if t.ndim == 2:
                t = t[:, 0] if t.numel() else t.reshape(-1)
            return t.reshape(-1).contiguous()
        S_np = np.asarray(S, dtype=np.int32)
        if S_np.ndim == 2:
            S_np = S_np[:, 0] if S_np.size else np.zeros(B, dtype=np.int32)
        return torch.as_tensor(
            np.asarray(S_np, dtype=np.int32).reshape(-1), device=self.device
        )

    def _p_from_stat_df_t(self, stat, df, *, k: int, x, y, S=None) -> torch.Tensor:
        """χ² p-values on device. ``chi2.sf(x, ν) = gammaincc(ν/2, x/2)``."""
        pos = df > 0
        zero = df == 0
        sf = torch.special.gammaincc(df.to(torch.float64) * 0.5, stat * 0.5)
        p = torch.where(pos, sf, torch.full_like(sf, float("nan")))
        if k >= 1:
            p = torch.where(zero, torch.ones_like(p), p)
        return p

    def _batch_k0(self, x, y) -> torch.Tensor:
        x_t = self._to_int32_cuda(x)
        y_t = self._to_int32_cuda(y)
        B = int(x_t.numel())
        if B == 0:
            return torch.zeros(0, dtype=torch.float64, device=self.device)
        if int(y_t.numel()) != B:
            msg = "x, y batch dimensions must match"
            raise ValueError(msg)
        stat = torch.empty(B, dtype=torch.float64, device=self.device)
        df = torch.empty(B, dtype=torch.int32, device=self.device)
        self._ext.k0(
            self._data,
            self._cards,
            x_t,
            y_t,
            stat,
            df,
            int(self._max_card),
            self._csr_off,
            self._csr_idx,
            int(self._tile_v),
        )
        return self._p_from_stat_df_t(stat, df, k=0, x=x_t, y=y_t)

    def _batch_k1(self, x, y, S) -> torch.Tensor:
        x_t = self._to_int32_cuda(x)
        y_t = self._to_int32_cuda(y)
        B = int(x_t.numel())
        if B == 0:
            return torch.zeros(0, dtype=torch.float64, device=self.device)
        if int(y_t.numel()) != B:
            msg = "x, y batch dimensions must match"
            raise ValueError(msg)
        z_t = self._z_from_S(S, B)
        if int(z_t.numel()) != B:
            msg = "x, y, S batch dimensions must match"
            raise ValueError(msg)
        stat = torch.empty(B, dtype=torch.float64, device=self.device)
        df = torch.empty(B, dtype=torch.int32, device=self.device)
        if _gmem_k1_enabled() and hasattr(self._ext, "k1_gmem"):
            grid = _gmem_grid(B, self.n_samples)
            ws = self._workspace(grid)
            if ws is not None:
                self._ext.k1_gmem(
                    self._data,
                    self._cards,
                    x_t,
                    y_t,
                    z_t,
                    stat,
                    df,
                    int(self._max_card),
                    self._csr_off,
                    self._csr_idx,
                    int(self._tile_v),
                    ws[0],
                    grid,
                )
                return self._p_from_stat_df_t(stat, df, k=1, x=x_t, y=y_t, S=z_t)
        self._ext.k1(
            self._data,
            self._cards,
            x_t,
            y_t,
            z_t,
            stat,
            df,
            1,
            int(self._max_card),
            self._csr_off,
            self._csr_idx,
            int(self._tile_v),
        )
        return self._p_from_stat_df_t(stat, df, k=1, x=x_t, y=y_t, S=z_t)

    def _S_int32(self, S, B: int) -> torch.Tensor:
        if torch.is_tensor(S):
            t = S.to(device=self.device, dtype=torch.int32)
            if t.ndim == 1:
                t = t.reshape(B, -1)
            return t.contiguous()
        S_np = np.asarray(S, dtype=np.int32)
        if S_np.ndim == 1:
            S_np = (
                S_np.reshape(B, -1) if S_np.size else np.zeros((B, 0), dtype=np.int32)
            )
        return torch.as_tensor(S_np.reshape(B, -1), device=self.device).contiguous()

    def _sequential_batch(self, x, y, S) -> np.ndarray:
        x_np = np.asarray(_as_numpy(x), dtype=np.int64).reshape(-1)
        y_np = np.asarray(_as_numpy(y), dtype=np.int64).reshape(-1)
        B = int(x_np.size)
        S_np = np.asarray(_as_numpy(S), dtype=np.int64)
        if B == 0:
            return np.zeros(0, dtype=np.float64)
        if S_np.ndim == 1:
            S_np = (
                S_np.reshape(B, -1) if S_np.size else np.zeros((B, 0), dtype=np.int64)
            )
        S_np = S_np.reshape(B, -1)
        if int(y_np.size) != B or int(S_np.shape[0]) != B:
            msg = "x, y, S batch dimensions must match"
            raise ValueError(msg)
        out = np.empty(B, dtype=np.float64)
        chi = self.cit.chi_square
        for b in range(B):
            out[b] = chi(int(x_np[b]), int(y_np[b]), [int(v) for v in S_np[b]])
        return out

    def _pack_tiled_fits(self) -> bool:
        n = int(self.n_samples)
        C = max(int(self._max_card), 1)
        cap = int(self._ext.optin_smem()) - int(self._ext.static_smem_reserve())
        V = int(self._tile_v) if self._tile_v else int(self._ext.tile_v_for_card(C))
        hist = 4 * V * C
        pad = 16
        xy = 4 * n
        two_idx = 8 * n
        return (hist + xy + pad + two_idx) <= cap or (hist + pad + two_idx) <= cap

    def _batch_kge2(self, x, y, S, k: int) -> torch.Tensor:
        x_t = self._to_int32_cuda(x)
        y_t = self._to_int32_cuda(y)
        B = int(x_t.numel())
        if B == 0:
            return torch.zeros(0, dtype=torch.float64, device=self.device)
        if int(y_t.numel()) != B:
            msg = "x, y batch dimensions must match"
            raise ValueError(msg)
        S_t = self._S_int32(S, B)
        if int(S_t.size(0)) != B:
            msg = "x, y, S batch dimensions must match"
            raise ValueError(msg)
        if not self._pack_tiled_fits():
            if _gmem_enabled() and hasattr(self._ext, "ks_gmem"):
                grid = _gmem_grid(B, self.n_samples)
                ws = self._workspace(grid)
                if ws is not None:
                    stat = torch.empty(B, dtype=torch.float64, device=self.device)
                    df = torch.empty(B, dtype=torch.int32, device=self.device)
                    self._ext.ks_gmem(
                        self._data,
                        self._cards,
                        x_t,
                        y_t,
                        S_t,
                        stat,
                        df,
                        int(k),
                        int(self._max_card),
                        int(self._tile_v),
                        ws[0],
                        ws[1],
                        grid,
                    )
                    return self._p_from_stat_df_t(stat, df, k=k, x=x_t, y=y_t, S=S_t)
            print(
                f"[cuda chisq] k={k} B={B} n={self.n_samples} "
                f"C={int(self._max_card)} packed-Z SMEM does not fit opt-in; "
                f"sequential CIT.chi_square",
                file=sys.stderr,
                flush=True,
            )
            return torch.as_tensor(
                self._sequential_batch(x_t, y_t, S_t), device=self.device
            )
        stat = torch.empty(B, dtype=torch.float64, device=self.device)
        df = torch.empty(B, dtype=torch.int32, device=self.device)
        self._ext.ks(
            self._data,
            self._cards,
            x_t,
            y_t,
            S_t,
            stat,
            df,
            int(k),
            1,
            int(self._max_card),
            int(self._tile_v),
        )
        return self._p_from_stat_df_t(stat, df, k=k, x=x_t, y=y_t, S=S_t)

    def batch_t(self, x, y, S) -> torch.Tensor:
        """p-values on ``device``. ``x,y`` (B,), ``S`` (B, k)."""
        if torch.is_tensor(S):
            k = int(S.size(1)) if S.ndim == 2 else (1 if S.numel() else 0)
        else:
            S_np = np.asarray(S)
            k = int(S_np.shape[1]) if S_np.ndim == 2 else (1 if S_np.size else 0)
        if k == 0:
            return self._batch_k0(x, y)
        if k == 1:
            return self._batch_k1(x, y, S)
        return self._batch_kge2(x, y, S, k)

    def batch(self, x, y, S) -> np.ndarray:
        """p-values as NumPy. ``x,y`` (B,), ``S`` (B, k)."""
        return self.batch_t(x, y, S).detach().cpu().numpy()
