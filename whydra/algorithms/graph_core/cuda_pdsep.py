r"""CUDA pdsep colliders + qreach + resident CI (FCI-stable v2).

Sepsets are PC-stable ``ranges[p,p,2]`` + 1D ``S``. After qreach, ``amat`` and
the Possible-D-SEP mask stay on device. CI unranks ``combinations(Pdsep(x)\{y}, k)``
(same lex order as sequential), skips ``S ⊆ adj(x)`` on the frozen snapshot,
and first-hit deletes after a last-writer walk.

Adjacency is frozen from the amat snapshot; CI is pair-major over all ``x``;
a one-CTA walk commits sepsets in ``x`` order. Deletions apply in bulk.
Matches ``pdsep_v2`` with ``n_jobs=1``.

Fused CI unranks ``S`` and runs Fisher-Z in one kernel (no ``(x,y,S)`` buffer).
``WHYDRA_PDSEP_FUSED=0`` restores fill + ``batch_r``.
"""

import os
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.stats import norm

if TYPE_CHECKING:
    from .cuda_graph_batch import SepsetCSR

from .cuda_chisq import is_host_ci

USE_PDSEP_FUSED = os.environ.get("WHYDRA_PDSEP_FUSED", "1") != "0"

_PDSEP_CUDA = None


def _pdsep_cuda():
    """Prebuilt ``pdsep.cu`` (colliders + Possible-D-SEP BFS)."""
    global _PDSEP_CUDA
    if _PDSEP_CUDA is None:
        from ._cuda import pdsep

        _PDSEP_CUDA = pdsep
    return _PDSEP_CUDA


def sepset_dict_to_csr(sepset: dict, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Pack ``dict[(i,j)] → iterable`` into 1D ``S`` and ``ranges[p,p,2]``."""
    ranges = np.full((p, p, 2), -1, dtype=np.int64)
    chunks: list[np.ndarray] = []
    pos = 0
    for (x, y), members in sepset.items():
        if x == y:
            continue
        if not members:
            continue
        arr = np.fromiter((int(v) for v in members), dtype=np.int64)
        if arr.size == 0:
            continue
        chunks.append(arr)
        ranges[int(x), int(y), 0] = pos
        ranges[int(x), int(y), 1] = pos + int(arr.size)
        pos += int(arr.size)
    S = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
    return S, ranges


def sepset_csr_to_dict(S: np.ndarray, ranges: np.ndarray, p: int) -> dict:
    """Non-empty CSR slices → ``dict[(i,j)] → set`` (tests / sparse compare)."""
    S = np.asarray(S, dtype=np.int64)
    ranges = np.asarray(ranges, dtype=np.int64)
    sepset = {}
    nz = np.argwhere(ranges[:, :, 0] >= 0)
    for i, j in nz:
        i = int(i)
        j = int(j)
        if i == j:
            continue
        start = int(ranges[i, j, 0])
        end = int(ranges[i, j, 1])
        if end <= start:
            continue
        sepset[(i, j)] = {int(v) for v in S[start:end]}
    return sepset


class SepsetStore:
    """Host ``ranges[p,p,2]`` + 1D ``S``. Late writes append; ``.get`` matches udag2pag."""

    def __init__(self, S: np.ndarray, ranges: np.ndarray):
        self.S = np.ascontiguousarray(S, dtype=np.int64).reshape(-1)
        self.ranges = np.array(ranges, dtype=np.int64, copy=True)
        self._extra: list[int] = []

    def packed(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the flattened sepset members and the ``ranges[p, p, 2]`` table."""
        if not self._extra:
            return self.S, self.ranges
        extra = np.asarray(self._extra, dtype=np.int64)
        if self.S.size == 0:
            return extra, self.ranges
        return np.concatenate([self.S, extra]), self.ranges

    def write(self, x: int, y: int, members, *, mirror: bool = True) -> None:
        """Append a separating set for ``(x, y)`` and update its range."""
        seq = [int(v) for v in members]
        base = int(self.S.size) + len(self._extra)
        if seq:
            self._extra.extend(seq)
            start, end = base, base + len(seq)
        else:
            start, end = 0, 0
        self.ranges[x, y, 0] = start
        self.ranges[x, y, 1] = end
        if mirror:
            self.ranges[y, x, 0] = start
            self.ranges[y, x, 1] = end

    def get(self, key, default=()):
        """Return the separating set stored for ``(x, y)``, or ``default``."""
        x, z = key
        start = int(self.ranges[x, z, 0])
        end = int(self.ranges[x, z, 1])
        if start < 0 or end <= start:
            return default
        n0 = int(self.S.size)
        if end <= n0:
            return self.S[start:end]
        flat, _ = self.packed()
        return flat[start:end]


def _unf_tensor(unfVect, p: int, device) -> torch.Tensor:
    if not unfVect:
        return torch.zeros((0, 3), dtype=torch.int32, device=device)
    rows = []
    for t in unfVect:
        if len(t) != 3:
            continue
        a, b, c = (int(t[0]), int(t[1]), int(t[2]))
        if 0 <= a < p and 0 <= b < p and 0 <= c < p:
            rows.append((a, b, c))
    if not rows:
        return torch.zeros((0, 3), dtype=torch.int32, device=device)
    return torch.as_tensor(rows, dtype=torch.int32, device=device)


def mark_colliders_qreach(
    sk,
    S,
    ranges,
    unfVect,
    *,
    device: str | torch.device = "cuda",
    qreach_cpu,
    to_host: bool = True,
):
    """Collider marks then all-source qreach. ``amat`` stays 0/1/2 int8.

    ``sk``, ``S``, and ``ranges`` may already be CUDA tensors (no H2D).
    ``mark_colliders``: one CTA, centres ``y`` ascending, sources ``x``
    parallel within a centre. ``qreach_all``: waves of **one CTA**, up to
    **128 threads**, one BFS source per thread (integer queues). ``qreach_cpu(x,
    amat)`` is the Python BFS used on queue overflow.
    Returns ``(amat, pdsep_mask, t_colliders, t_qreach)``. Host numpy if
    ``to_host``, else CUDA tensors.
    """
    import time

    ext = _pdsep_cuda()
    dev = torch.device(device)
    if torch.is_tensor(sk):
        amat = sk.to(device=dev, dtype=torch.int8).contiguous()
        p = int(amat.size(0))
    else:
        p = int(sk.shape[0])
        amat = torch.as_tensor(np.ascontiguousarray(sk, dtype=np.int8), device=dev)
    if torch.is_tensor(S):
        S_t = S.to(device=dev, dtype=torch.int64).reshape(-1).contiguous()
    else:
        S_t = torch.as_tensor(np.ascontiguousarray(S, dtype=np.int64), device=dev)
    if torch.is_tensor(ranges):
        ranges_t = ranges.to(device=dev, dtype=torch.int64).contiguous()
    else:
        ranges_t = torch.as_tensor(
            np.ascontiguousarray(ranges, dtype=np.int64), device=dev
        )
    if int(S_t.numel()) == 0:
        S_t = torch.zeros(1, dtype=torch.int64, device=dev)
    t0 = time.perf_counter()
    unf_t = _unf_tensor(unfVect, p, dev)
    ext.mark_colliders(amat, S_t, ranges_t, unf_t)
    torch.cuda.synchronize(dev)
    t_colliders = time.perf_counter() - t0
    t0 = time.perf_counter()
    mask, overflow = ext.qreach_all(amat)
    torch.cuda.synchronize(dev)
    if bool(overflow.any()):
        ov = overflow.detach().cpu().numpy()
        amat_np = amat.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()
        for x in np.flatnonzero(ov):
            psep = qreach_cpu(int(x), amat_np)
            mask_np[int(x), :] = 0
            if psep:
                mask_np[int(x), np.asarray(psep, dtype=np.int64)] = 1
        mask.copy_(torch.as_tensor(mask_np, device=dev))
    t_qreach = time.perf_counter() - t0
    if to_host:
        return (
            amat.detach().cpu().numpy(),
            mask.detach().cpu().numpy(),
            t_colliders,
            t_qreach,
        )
    return amat, mask, t_colliders, t_qreach


def _csr_from_packed(S, ranges, device) -> "SepsetCSR":
    from .cuda_graph_batch import SepsetCSR, _graph_batch_cuda

    if torch.is_tensor(ranges):
        ranges_t = ranges.to(device=device, dtype=torch.int64).contiguous()
    else:
        ranges_t = torch.as_tensor(
            np.ascontiguousarray(ranges, dtype=np.int64), device=device
        )
    if torch.is_tensor(S):
        S_t = S.to(device=device, dtype=torch.int64).reshape(-1).contiguous()
    else:
        S_t = torch.as_tensor(
            np.ascontiguousarray(S, dtype=np.int64).reshape(-1), device=device
        )
    p = int(ranges_t.size(0))
    k_max = int(_graph_batch_cuda().k_max())
    csr = SepsetCSR(p, device, k_max=k_max)
    n = int(S_t.numel())
    if n:
        csr._ensure_capacity(n + p * k_max, "sepset CSR buffer (PDS handoff)")
        csr.S[:n].copy_(S_t)
        csr.nnz.fill_(n)
        csr._nnz_host = n
    csr.ranges.copy_(ranges_t)
    return csr


def _s_subset_adj(adj_x: torch.Tensor, Sb: torch.Tensor, px=None):
    """``S ⊆ adj(x)`` per test. ``adj_x`` is int8 ``[p]`` or ``[p, p]``."""
    if adj_x.dim() == 2:
        if px is None:
            msg = "px required for 2D adj_x"
            raise ValueError(msg)
        return adj_x[px.long().unsqueeze(1), Sb.long()].bool().all(dim=1)
    return adj_x[Sb.long()].bool().all(dim=1)


def _pdsep_rank_stripes(
    batcher,
    amat: torch.Tensor,
    nbrs_pd: torch.Tensor,
    deg_pd: torch.Tensor,
    px0: torch.Tensor,
    py0: torch.Tensor,
    ncomb: torch.Tensor,
    k: int,
    tau: float,
    adj_frozen: torch.Tensor,
    pmax_t: torch.Tensor,
    csr,
    fused: bool,
    ext,
    rb_empty: torch.Tensor,
    write_g: bool = False,
    alive: torch.Tensor | None = None,
    deleted: torch.Tensor | None = None,
    alpha: float = 0.05,
    hit_k: torch.Tensor | None = None,
    hit_rank: torch.Tensor | None = None,
) -> None:
    """One rank stripe of Possible-D-SEP CI over the live directed pairs.

    Fused ``ci_from_comb`` for ``k ≤ 32``; ``k = 33..96`` uses fill +
    ``batch_r`` (96×96 LDLᵀ). ``amat`` stays frozen; hits go into ``alive`` /
    ``deleted``. Skip ``S ⊆ adj_frozen``. ``adj_frozen`` is int8 ``[p, p]``.
    When ``hit_k`` is set, first-hits are stored for a later last-writer walk
    instead of ``csr.claim``.
    """
    from .cuda_graph_batch import (
        ALL_S_CHUNK,
        RANK_MAX,
        _cuda_step,
        _offsets_from_counts,
        delete_if_indep,
        fill_batch,
        first_hit_from_p_t,
        first_hit_ranks,
        prefix_pmax_t,
    )

    E0 = int(px0.numel())
    if E0 == 0:
        return
    max_n = int(ncomb.max().item())
    if max_n <= 0:
        return
    dev = amat.device
    rank_base = 0
    while rank_base < max_n:
        present = alive if alive is not None else amat[px0.long(), py0.long()] != 0
        remain = (ncomb - rank_base).clamp(min=0)
        keep = present & (remain > 0)
        n_live = int(keep.sum().item())
        if n_live == 0:
            break
        idx = keep.nonzero(as_tuple=False).reshape(-1)
        px = px0.index_select(0, idx)
        py = py0.index_select(0, idx)
        n_this = remain.index_select(0, idx)
        W = max(1, int(ALL_S_CHUNK) // n_live)
        n_this = n_this.clamp(max=W)
        off = _offsets_from_counts(n_this)
        B = int(off[-1].item())
        if B == 0:
            rank_base += W
            continue
        host_ci = is_host_ci(batcher)
        r = None
        use_fused = (not host_ci) and bool(fused) and k <= int(ext.fused_k_max())
        if host_ci:
            with _cuda_step("fill"):
                xb, yb, Sb = fill_batch(
                    nbrs_pd, deg_pd, px, py, off, k, rank_base=rank_base
                )
            if k > 0:
                valid = ~_s_subset_adj(adj_frozen, Sb, xb)
            else:
                valid = torch.ones(B, dtype=torch.bool, device=dev)
            pvals = torch.full((B,), float("nan"), dtype=torch.float64, device=dev)
            if bool(valid.any()):
                vidx = valid.nonzero(as_tuple=False).reshape(-1)
                xb_v = xb.index_select(0, vidx)
                yb_v = yb.index_select(0, vidx)
                Sb_v = Sb.index_select(0, vidx)
                with _cuda_step("chisq"):
                    p_v = batcher.batch_t(xb_v, yb_v, Sb_v)
                pvals.index_copy_(0, vidx, p_v)
            with _cuda_step("first_hit"):
                first = first_hit_from_p_t(pvals, off, alpha, True)
            if B and int(px.numel()):
                with _cuda_step("prefix_pmax"):
                    prefix_pmax_t(pvals, px, py, off, first, pmax_t, k=k)
        elif use_fused:
            first = torch.full(
                (int(px.numel()),), RANK_MAX, dtype=torch.int64, device=dev
            )
            r = torch.full((B,), float("nan"), dtype=batcher.corr.dtype, device=dev)
            ext.ci_from_comb(
                batcher.corr,
                amat,
                nbrs_pd,
                deg_pd,
                px,
                py,
                off,
                int(k),
                int(rank_base),
                rb_empty,
                first,
                int(B),
                int(batcher.n_samples),
                float(tau),
                float(batcher.eps),
                True,
                bool(write_g),
                r,
                adj_frozen,
                # No PC-stable union marking on the pdsep path: pass the
                # empty trailing tensors of the extended ci_from_comb.
                torch.empty(0, dtype=torch.uint8, device=dev),
                torch.empty(0, dtype=torch.int32, device=dev),
                torch.empty(0, dtype=torch.uint8, device=dev),
                torch.empty(0, dtype=torch.int64, device=dev),
            )
        else:
            xb, yb, Sb = fill_batch(
                nbrs_pd, deg_pd, px, py, off, k, rank_base=rank_base
            )
            r = torch.full((B,), float("nan"), dtype=batcher.corr.dtype, device=dev)
            if k > 0:
                valid = ~_s_subset_adj(adj_frozen, Sb, xb)
            else:
                valid = torch.ones(B, dtype=torch.bool, device=dev)
            if bool(valid.any()):
                vidx = valid.nonzero(as_tuple=False).reshape(-1)
                r_v = batcher.batch_r(
                    xb.index_select(0, vidx),
                    yb.index_select(0, vidx),
                    Sb.index_select(0, vidx),
                )
                r.index_copy_(0, vidx, r_v.to(dtype=r.dtype))
            first = first_hit_ranks(r, off, batcher.n_samples, k, tau, True)
        if r is not None and B and int(px.numel()):
            ext.prefix_pmax(
                r,
                off,
                first,
                px,
                py,
                pmax_t,
                int(batcher.n_samples),
                int(k),
                int(RANK_MAX),
            )
        hit = first < RANK_MAX
        if bool(hit.any()):
            e_idx = hit.nonzero(as_tuple=False).reshape(-1)
            hx = px.index_select(0, e_idx)
            hy = py.index_select(0, e_idx)
            orig = idx.index_select(0, e_idx)
            if write_g and not use_fused:
                delete_if_indep(
                    amat,
                    hx,
                    hy,
                    torch.ones(int(e_idx.numel()), dtype=torch.uint8, device=dev),
                )
            if alive is not None:
                alive.index_fill_(0, orig, False)
            if deleted is not None:
                deleted[hx.long(), hy.long()] = 1
                deleted[hy.long(), hx.long()] = 1
            if hit_k is not None:
                hit_k.index_fill_(0, orig, int(k))
                assert hit_rank is not None
                abs_rank = first.index_select(0, e_idx) + int(rank_base)
                hit_rank.index_copy_(0, orig, abs_rank)
            else:
                csr.claim(px, py, first, nbrs_pd, deg_pd, k, rank_base, None)
                csr.ranges[hy.long(), hx.long(), :] = csr.ranges[
                    hx.long(), hy.long(), :
                ]
        rank_base += W


def run_pdsep_ci(
    batcher,
    amat: torch.Tensor,
    pdsep_mask: torch.Tensor,
    pmax: np.ndarray | torch.Tensor,
    sep_store: SepsetStore | None,
    alpha: float,
    m_max=float("inf"),
    fused: bool | None = None,
    *,
    to_host: bool = True,
    csr=None,
) -> float:
    """Run rank-striped Possible-D-SEP CI on a resident ``amat``.

    Update ``amat``, ``pmax``, and the separating sets in place and return the CI wall
    time in seconds.

    ``to_host=True`` (default) copies CSR and ``pMax`` to ``sep_store`` / the
    host ``pmax`` array at the end. ``to_host=False`` keeps them on device;
    the packed CSR is stored on ``run_pdsep_ci.last_csr``.

    Freeze adjacency from the snapshot. CI is pair-major over all ``x`` at
    each ``k`` (one fused grid). First-hit ranks stay on device; a single CTA
    then walks ``x = 0..p-1`` with ``__syncthreads`` so sepset last-writer
    matches ``pdsep_v2`` / ``n_jobs=1``. Deletions apply after the walk.

    Fused ``ci_from_comb`` for ``k <= 32``; larger ``k`` (up to 96) uses fill +
    ``batch_r``. Skip ``S`` contained in ``adj(x)``.
    """
    import time

    from . import cuda_graph_batch as _gb
    from .cuda_graph_batch import (
        _graph_batch_cuda,
        compact_adj,
        count_tests,
        emit_pairs,
    )

    if fused is None:
        fused = USE_PDSEP_FUSED
    if is_host_ci(batcher):
        fused = False
    t0 = time.perf_counter()
    dev = amat.device
    p = int(amat.size(0))
    tau = float(norm.ppf(1.0 - alpha / 2.0))
    from .cuda_graph_batch import _pmax_device

    pmax_t = _pmax_device(pmax, dev)
    assert pmax_t is not None
    if csr is None:
        assert sep_store is not None
        S0, R0 = sep_store.packed()
        csr = _csr_from_packed(S0, R0, dev)

    pd_adj = (pdsep_mask != 0).to(dtype=torch.int8)
    nbrs_pd, deg_pd = compact_adj(pd_adj, device=dev)
    k_lim = int(m_max) if np.isfinite(m_max) else 10**9
    k_lim = min(k_lim, int(batcher._k_max))
    ext = _graph_batch_cuda()
    rb_empty = torch.empty(0, dtype=torch.int64, device=dev)

    adj_snap = amat != 0
    deleted = torch.zeros((p, p), dtype=torch.uint8, device=dev)
    adj_i8 = adj_snap.to(dtype=torch.int8).contiguous()
    nbrs_adj, deg_adj = compact_adj(adj_i8, device=dev)
    px0, py0 = emit_pairs(nbrs_adj, deg_adj)
    outside = (pdsep_mask != 0) & (~adj_snap)
    outside.fill_diagonal_(False)
    has_out = outside.any(dim=1)
    if int(px0.numel()):
        keep = has_out[px0.long()]
        if bool(keep.any()):
            idx = keep.nonzero(as_tuple=False).reshape(-1)
            px0 = px0.index_select(0, idx)
            py0 = py0.index_select(0, idx)
        else:
            px0 = px0[:0]
            py0 = py0[:0]
    E0 = int(px0.numel())
    counts = torch.zeros(p, dtype=torch.int64, device=dev)
    if E0:
        counts.scatter_add_(
            0, px0.long(), torch.ones(E0, dtype=torch.int64, device=dev)
        )
    row_off = torch.zeros(p + 1, dtype=torch.int64, device=dev)
    if p:
        torch.cumsum(counts, 0, out=row_off[1:])
    alive = torch.ones(E0, dtype=torch.bool, device=dev)
    hit_k = torch.zeros(E0, dtype=torch.int32, device=dev)
    hit_rank = torch.zeros(E0, dtype=torch.int64, device=dev)
    max_pd = int(deg_pd.max().item()) if p else 0
    max_k = min(k_lim, max_pd)
    for k in range(1, max_k + 1):
        if _gb._PROF is not None:
            _gb._PROF.k = k
        if E0 == 0:
            break
        ncomb = count_tests(nbrs_pd, deg_pd, px0, py0, k)
        if int(ncomb.max().item()) <= 0:
            break
        _pdsep_rank_stripes(
            px0=px0,
            py0=py0,
            ncomb=ncomb,
            k=k,
            adj_frozen=adj_i8,
            alive=alive,
            deleted=deleted,
            write_g=False,
            hit_k=hit_k,
            hit_rank=hit_rank,
            batcher=batcher,
            amat=amat,
            nbrs_pd=nbrs_pd,
            deg_pd=deg_pd,
            tau=tau,
            pmax_t=pmax_t,
            csr=csr,
            fused=fused,
            ext=ext,
            rb_empty=rb_empty,
            alpha=alpha,
        )
    if E0:
        csr.commit_walk(py0, hit_k, hit_rank, row_off, nbrs_pd, deg_pd)
    amat.masked_fill_(deleted != 0, 0)

    torch.cuda.synchronize(dev)
    S_out, ranges_out = csr.pack()
    vars(run_pdsep_ci).update(last_csr=(S_out, ranges_out), last_pmax=pmax_t)
    if to_host:
        if sep_store is not None:
            sep_store.S = S_out.detach().cpu().numpy()
            sep_store.ranges = ranges_out.detach().cpu().numpy()
            sep_store._extra = []
        assert pmax is not None
        if isinstance(pmax, torch.Tensor):
            pmax.copy_(pmax_t)
        else:
            np.copyto(np.asarray(pmax), pmax_t.detach().cpu().numpy())
    return time.perf_counter() - t0


def mark_colliders_only(
    pag: np.ndarray,
    S: np.ndarray,
    ranges: np.ndarray,
    unfVect,
    *,
    device: str | torch.device = "cuda",
) -> None:
    """In-place collider arrows (value 2) on a 0/1 PAG. Same kernel as pdsep.

    CUDA: one CTA, centres ``y`` ascending, sources ``x`` parallel within a
    centre; ``int8`` stores, CSR ``int64`` scans.
    """
    ext = _pdsep_cuda()
    dev = torch.device(device)
    p = int(pag.shape[0])
    amat = torch.as_tensor(np.ascontiguousarray(pag, dtype=np.int8), device=dev)
    S_t = torch.as_tensor(np.ascontiguousarray(S, dtype=np.int64), device=dev)
    ranges_t = torch.as_tensor(np.ascontiguousarray(ranges, dtype=np.int64), device=dev)
    unf_t = _unf_tensor(unfVect, p, dev)
    ext.mark_colliders(amat, S_t, ranges_t, unf_t)
    torch.cuda.synchronize(dev)
    np.copyto(pag, amat.detach().cpu().numpy())


__all__ = [
    "SepsetStore",
    "mark_colliders_only",
    "mark_colliders_qreach",
    "run_pdsep_ci",
    "sepset_csr_to_dict",
    "sepset_dict_to_csr",
]
