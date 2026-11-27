r"""CUDA batch builder for PC/FCI skeleton discovery and possible-D-SEP.

At each depth k, freeze the neighbors and test conditioning sets of size k.
The default fused kernel groups tests by live node pair. It assigns a CUDA
block to each pair, or one thread per pair at k=0. For each ordered pair
(x, y), enumerate k-element subsets of x's neighbors other than y in
lexicographic order, as in the CPU PC-stable search.

Both PC-stable and the FCI-stable skeleton search accumulate the members of
every separating set found at that depth. They merge the two directions of a
pair, store the union for both (x, y) and (y, x), and remove edges after the
depth completes. The union can contain more variables than any single set
tested.

Entry points
    ``run_skeleton``  Run the PC or FCI skeleton search on a device-resident
                      graph. Both can use the fused C++ driver. The Python
                      loop handles batchers that use the host CI interface
                      and builds where the fused driver is unavailable.
    ``run_depth``     Run one depth in the Python loop.
    ``SepsetCSR``     device sepset CSR (``ranges`` + 1-D ``S``).
                      ``claim``/``commit_union_fused``/``commit_walk`` wrap the
                      corresponding bindings; ``commit_union`` is pure
                      Python/torch.
    ``GraphBatchState``  legacy wavefront state retained for API compatibility;
                      no in-repo caller.

Memory / layout model (see ``graph_batch.cu`` for the kernel side)
    ``G`` is CUDA int8 ``[p, p]``; ``nbrs`` int32 ``[p, max_d]`` packed in
    increasing index order with ``deg`` int32 ``[p]``. Sepsets are
    ``ranges[p, p, 2] = (start, end)`` into 1-D int64 ``S``: ``(-1, -1)`` is a
    still-present edge and ``(0, 0)`` an empty sepset (k = 0).

Invariants (identical across the C++/Python paths; pinned by
``tests/test_cuda_pc_sepset_union.py``)
    * PC uses the strict ``p > alpha`` test; FCI / possible-D-SEP uses
      ``p >= alpha``.
    * The skeleton searches store the union of separating-set members.
      The later FCI Possible-D-SEP phase uses first hits.
    * k = 0 is symmetric only on the pair-major + workspace fast path
      (upper triangle plus a mirrored empty sepset); the flat/Python fallback
      and k >= 1 stay directed.
    * The C++ driver and the Python fallback return identical ``G`` and
      sepsets; their test counts differ only by the documented k = 0 factor
      of two on the workspace path.

CUDA counterparts: ``graph_batch.cu`` (bindings ``pc_depth_union`` /
``pc_skeleton_union``) and ``fisher_z_ldlt.cu`` (the LDLT partial-correlation
kernels). See ``docs/pc_stable_cuda.md`` and ``docs/fci_stable_cuda.md`` for
CTA layout.
"""

import math
import sys
import time
from contextlib import contextmanager
from typing import Any, Literal, overload

import numpy as np
import torch
from scipy.stats import norm

from .cuda_chisq import is_host_ci
from .cuda_fisher_z import CudaFisherZ, _p_value_from_r

B_MAX = 1 << 26
# Max tests per rank-striped pair-major chunk. W = max(1, ALL_S_CHUNK / E_live).
# 2^26 is one giant chunk (old all-S). Smaller values skip leftover S after hits.
ALL_S_CHUNK = 1 << 20
RANK_MAX = int(np.iinfo(np.int64).max)
S_BITS_MAX = 64_000_000
# Pair-major is faster than S-major. Rank stripes + fused CI are the PC path.
USE_S_MAJOR = False
S_MAJOR_MIN_K = 2
USE_FUSED = True

_GRAPH_BATCH_CUDA = None
# Set by the profiler: object with ``add(name, dt)``. Each ``_cuda_step``
# synchronizes before and after so GPU time is not charged to a later ``.item()``.
_PROF = None
depth_stats = {"n_tests": 0, "n_chunks": 0}


class CudaStepLog:
    """Per-depth, per-``_cuda_step`` wall times. Enable with ``cuda_profile()``."""

    def __init__(self):
        self.k = -1
        self.phase = "skeleton"
        self.n_waves = 0
        self.records: list[dict] = []

    def add(self, name: str, dt: float) -> None:
        """Record one measured CUDA step."""
        self.records.append(
            {
                "phase": self.phase,
                "k": int(self.k),
                "step": str(name),
                "dt": float(dt),
            }
        )

    def rows(self) -> list[dict]:
        """Return the recorded step samples."""
        return list(self.records)

    def format_table(self) -> str:
        """Format the recorded step samples as a fixed-width table."""
        if not self.records:
            return "(no cuda step samples)"
        lines = [f"{'phase':<12} {'k':>4}  {'step':<16} {'seconds':>12}"]
        tot = 0.0
        for r in self.records:
            lines.append(
                f"{r['phase']:<12} {r['k']:>4}  {r['step']:<16} {r['dt']:12.6f}"
            )
            tot += r["dt"]
        lines.append(f"{'total':<12} {'':>4}  {'':<16} {tot:12.6f}")
        by_k: dict[int, float] = {}
        for r in self.records:
            if r["phase"] == "skeleton":
                by_k[r["k"]] = by_k.get(r["k"], 0.0) + r["dt"]
        if by_k:
            lines.append("")
            lines.append(f"{'k':>4}  {'skeleton_s':>12}")
            for k in sorted(by_k):
                lines.append(f"{k:>4}  {by_k[k]:12.6f}")
        return "\n".join(lines)


@contextmanager
def cuda_profile():
    """Collect ``_cuda_step`` times. Restores the previous profiler on exit."""
    global _PROF
    log = CudaStepLog()
    prev = _PROF
    _PROF = log
    try:
        yield log
    finally:
        _PROF = prev


@contextmanager
def _cuda_step(name: str):
    p = _PROF
    if p is None:
        yield
        return
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize()
    p.add(name, time.perf_counter() - t0)


def first_hit_from_p_t(
    pvals: torch.Tensor,
    offsets: torch.Tensor,
    alpha: float,
    fci: bool,
) -> torch.Tensor:
    """First independent local rank per pair from device p-values.

    ``fci=True`` selects the non-strict ``p >= alpha`` rule; ``False`` selects
    PC's strict ``p > alpha``. Pairs with no independent test stay at
    ``RANK_MAX``.
    """
    device = pvals.device
    E = int(offsets.numel()) - 1
    first = torch.full((max(E, 0),), RANK_MAX, dtype=torch.int64, device=device)
    B = int(pvals.numel())
    if B == 0 or E <= 0:
        return first
    b = torch.arange(B, dtype=torch.int64, device=device)
    pair_id = torch.searchsorted(offsets[1:], b, right=True)
    local = b - offsets.index_select(0, pair_id)
    indep = torch.isfinite(pvals)
    passed_threshold = pvals >= float(alpha) if fci else pvals > float(alpha)
    indep = indep & passed_threshold
    vals = torch.where(
        indep, local, torch.full((B,), RANK_MAX, dtype=torch.int64, device=device)
    )
    first.scatter_reduce_(0, pair_id, vals, reduce="amin")
    return first


def prefix_pmax_t(
    pvals: torch.Tensor,
    px: torch.Tensor,
    py: torch.Tensor,
    offsets: torch.Tensor,
    first: torch.Tensor,
    pmax: torch.Tensor,
    *,
    k: int = 1,
) -> None:
    """Device ``maximum.at`` into ``pmax[x, y]`` (and ``pmax[y, x]`` if ``k==0``)."""
    B = int(pvals.numel())
    if B == 0 or pmax is None:
        return
    device = pvals.device
    b = torch.arange(B, dtype=torch.int64, device=device)
    pair_id = torch.searchsorted(offsets[1:], b, right=True)
    local = b - offsets.index_select(0, pair_id)
    fr = first.index_select(0, pair_id)
    keep = ((fr == RANK_MAX) | (local <= fr)) & torch.isfinite(pvals)
    xs = px.index_select(0, pair_id).to(torch.int64)
    ys = py.index_select(0, pair_id).to(torch.int64)
    pdim = int(pmax.size(0))
    lin = xs * pdim + ys
    neg = torch.full((B,), float("-inf"), dtype=torch.float64, device=device)
    vals = torch.where(keep, pvals.to(torch.float64), neg)
    flat = pmax.view(-1)
    flat.scatter_reduce_(0, lin, vals, reduce="amax")
    if int(k) == 0:
        lin_yx = ys * pdim + xs
        flat.scatter_reduce_(0, lin_yx, vals, reduce="amax")


def _graph_batch_cuda():
    """Prebuilt ``graph_batch.cu`` (compact, combinadic, fused ``ci_from_comb``)."""
    global _GRAPH_BATCH_CUDA
    if _GRAPH_BATCH_CUDA is None:
        from ._cuda import graph_batch

        _GRAPH_BATCH_CUDA = graph_batch
    return _GRAPH_BATCH_CUDA


def _as_adj(G, device) -> torch.Tensor:
    """Normalize any adjacency input to a contiguous CUDA int8 ``(G != 0)``.

    Keeps an already-correct CUDA int8 tensor (no copy unless needed); converts
    numpy / CPU / bool inputs, treating every nonzero entry as an edge.
    """
    if torch.is_tensor(G):
        want_cuda = device is None or torch.device(device).type == "cuda"
        if G.dtype == torch.int8 and G.is_cuda and want_cuda:
            return G if G.is_contiguous() else G.contiguous()
        dev = G.device if device is None else torch.device(device)
        t = (G != 0).to(device=dev, dtype=torch.int8)
    else:
        if device is None:
            device = torch.device("cuda")
        t = torch.as_tensor((np.asarray(G) != 0).astype(np.int8), device=device)
    if not t.is_contiguous():
        t = t.contiguous()
    return t


def upload_G(G, device) -> torch.Tensor:
    """One H2D of adjacency ``(G != 0)`` as int8. Resident for the skeleton."""
    return _as_adj(G, device)


def complete_undirected(p: int, device) -> torch.Tensor:
    """Complete undirected adj on CUDA: int8 1 off-diagonal, 0 on the diagonal.

    Same resident tensor as ``upload_G`` of a TAIL—TAIL complete graph, without
    building Python ``Edge`` objects or copying a host matrix.
    """
    G = torch.ones((int(p), int(p)), dtype=torch.int8, device=device)
    G.fill_diagonal_(0)
    return G.contiguous()


def apply_hits(G: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> None:
    """Zero ``G[x,y]`` and ``G[y,x]`` on device. ``x,y`` stay on GPU.

    CUDA: ``ceil(n / 256)`` CTAs × 256 threads, one thread per listed pair,
    plain ``int8`` stores (no atomics).
    """
    if int(x.numel()) == 0:
        return
    _graph_batch_cuda().apply_hits(G, x.contiguous(), y.contiguous())


def edge_present(G: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``present[i] = 1`` iff ``G[x[i], y[i]] != 0``.

    CUDA: one thread per pair, ``uint8`` result, ``int8`` adjacency loads.
    """
    n = int(x.numel())
    present = torch.empty(n, dtype=torch.uint8, device=G.device)
    if n:
        _graph_batch_cuda().edge_present(G, x.contiguous(), y.contiguous(), present)
    return present


def delete_if_indep(
    G: torch.Tensor, x: torch.Tensor, y: torch.Tensor, indep: torch.Tensor
) -> None:
    """If the edge is still present and ``indep[i]``, store ``G[x,y]=G[y,x]=0``.

    CUDA: one thread per pair. Plain stores, no atomic. Writing 0 twice is
    idempotent.
    """
    if int(x.numel()) == 0:
        return
    _graph_batch_cuda().delete_if_indep(
        G, x.contiguous(), y.contiguous(), indep.contiguous()
    )


def hits_from_r(r, n_samples, k, tau, fci: bool) -> torch.Tensor:
    """Independence bits from partial correlations (unfused path).

    CUDA: one thread per test. FP64 Fisher-Z vs ``τ``; ``fci`` selects
    ``p ≥ α`` (``stat ≤ τ``) else ``p > α``.
    """
    n = int(r.numel())
    hit = torch.empty(n, dtype=torch.uint8, device=r.device)
    if n:
        _graph_batch_cuda().hits_from_r(
            r, hit, int(n_samples), int(k), float(tau), bool(fci)
        )
    return hit


def compact_adj(G, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed neighbors in increasing index order (same as ``np.where``).

    CUDA: ``degrees`` then ``compact``, each ``ceil(p / 256)`` CTAs × 256
    threads, one thread per vertex. ``int8`` adjacency → ``int32`` ``deg`` /
    ``nbrs[p, max_d]``.
    """
    ext = _graph_batch_cuda()
    if device is None:
        device = G.device if torch.is_tensor(G) else torch.device("cuda")
    G_t = _as_adj(G, device)
    p = int(G_t.size(0))
    deg = torch.zeros(p, dtype=torch.int32, device=device)
    ext.degrees(G_t, deg)
    max_d = int(deg.max().item()) if p else 0
    max_d = max(max_d, 1)
    nbrs = torch.zeros((p, max_d), dtype=torch.int32, device=device)
    ext.compact(G_t, nbrs, deg)
    return nbrs, deg


def emit_pairs(
    nbrs: torch.Tensor, deg: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Directed pairs ``(x, y)`` for every packed neighbor. Integer only.

    CUDA: one thread per vertex ``x``; writes ``px`` / ``py`` (``int32``).
    """
    ext = _graph_batch_cuda()
    device = deg.device
    p = int(deg.numel())
    row_off = torch.zeros(p + 1, dtype=torch.int64, device=device)
    if p:
        torch.cumsum(deg.to(torch.int64), 0, out=row_off[1:])
    E = int(row_off[-1].item())
    px = torch.empty(E, dtype=torch.int32, device=device)
    py = torch.empty(E, dtype=torch.int32, device=device)
    if E:
        ext.emit_pairs(nbrs, deg, row_off, px, py)
    return px, py


def count_tests(
    nbrs_cand: torch.Tensor,
    deg_cand: torch.Tensor,
    px: torch.Tensor,
    py: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """``C(deg(x)-1, k)`` per directed pair, saturating ``int64`` (``-1`` overflow).

    CUDA: one thread per pair. Integer binomials only.
    """
    ext = _graph_batch_cuda()
    counts = torch.empty(px.numel(), dtype=torch.int64, device=px.device)
    if px.numel():
        ext.count_tests(nbrs_cand, deg_cand, px, py, int(k), counts)
    return counts


def fill_batch(
    nbrs_cand: torch.Tensor,
    deg_cand: torch.Tensor,
    px: torch.Tensor,
    py: torch.Tensor,
    offsets: torch.Tensor,
    k: int,
    rank_base: int = 0,
    rank_bases: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize ``(x, y, S)`` for one rank stripe (unfused path).

    CUDA: one thread per test, combinadic ``unrank_lex`` into ``int32 S[B, k]``.
    The fused skeleton skips this buffer.
    """
    ext = _graph_batch_cuda()
    B = int(offsets[-1].item()) if offsets.numel() else 0
    device = px.device
    x = torch.empty(B, dtype=torch.int32, device=device)
    y = torch.empty(B, dtype=torch.int32, device=device)
    S = torch.empty((B, int(k)), dtype=torch.int32, device=device)
    if B:
        rb = (
            rank_bases
            if rank_bases is not None
            else torch.empty(0, dtype=torch.int64, device=device)
        )
        ext.fill_batch(
            nbrs_cand,
            deg_cand,
            px,
            py,
            offsets,
            int(k),
            int(rank_base),
            rb,
            x,
            y,
            S,
        )
    return x, y, S


def fill_at_ranks(
    nbrs_cand: torch.Tensor,
    deg_cand: torch.Tensor,
    px: torch.Tensor,
    py: torch.Tensor,
    ranks: torch.Tensor,
    k: int,
    x: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
    S: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One conditioner per pair: ``S[e] = combinations(cand(x,y), k)[ranks[e]]``.

    CUDA: one thread per pair, integer combinadic unrank.
    """
    ext = _graph_batch_cuda()
    E = int(px.numel())
    device = px.device
    if x is None:
        x = torch.empty(E, dtype=torch.int32, device=device)
        y = torch.empty(E, dtype=torch.int32, device=device)
        S = torch.empty((E, int(k)), dtype=torch.int32, device=device)
    elif E and int(x.numel()) != E:
        if y is None or S is None:
            msg = "x, y and S must be provided together"
            raise ValueError(msg)
        x, y, S = x[:E], y[:E], S[:E]
    if x is None or y is None or S is None:
        msg = "x, y and S must be provided together"
        raise ValueError(msg)
    if E:
        ext.fill_at_ranks(nbrs_cand, deg_cand, px, py, ranks, int(k), x, y, S)
    return x, y, S


def _offsets_from_counts(counts: torch.Tensor) -> torch.Tensor:
    E = int(counts.numel())
    off = torch.empty(E + 1, dtype=torch.int64, device=counts.device)
    off[0] = 0
    if E:
        torch.cumsum(counts, 0, out=off[1:])
    return off


def unique_S_lex(nbrs: torch.Tensor, deg: torch.Tensor, k: int) -> torch.Tensor:
    """Appearing k-subsets of packed neighbors, in lex order. No test-batch sort.

    k=1,2: mark + ``nonzero`` (row-major = lex). k≥3: combinadic bitset when
    ``C(p,k)`` fits, else unique of per-vertex combinations (size ``sum C(d,k)``).
    """
    ext = _graph_batch_cuda()
    k = int(k)
    device = deg.device
    p = int(deg.numel())
    if k <= 0 or p == 0:
        return torch.empty((0, max(k, 0)), dtype=torch.int32, device=device)
    if k == 1:
        mark = torch.zeros(p, dtype=torch.uint8, device=device)
        ext.mark_s_k1(nbrs, deg, mark)
        idx = mark.nonzero(as_tuple=False).reshape(-1).to(torch.int32)
        return idx.view(-1, 1).contiguous()
    if k == 2:
        mark = torch.zeros((p, p), dtype=torch.uint8, device=device)
        ext.mark_s_k2(nbrs, deg, mark)
        return mark.nonzero(as_tuple=False).to(torch.int32).contiguous()
    ncomb = math.comb(p, k)
    if 0 < ncomb <= S_BITS_MAX:
        bits = torch.zeros(ncomb, dtype=torch.uint8, device=device)
        ext.mark_s_ranks(nbrs, deg, int(k), bits)
        idx = bits.nonzero(as_tuple=False).reshape(-1).to(torch.int64)
        U = int(idx.numel())
        S = torch.empty((U, k), dtype=torch.int32, device=device)
        if U:
            ext.unrank_rows(idx, int(p), int(k), S)
        return S.contiguous()
    counts = torch.empty(p, dtype=torch.int64, device=device)
    ext.count_nbr_combos(deg, int(k), counts)
    off = _offsets_from_counts(counts)
    M = int(off[-1].item())
    raw = torch.empty((M, k), dtype=torch.int32, device=device)
    if M:
        ext.fill_nbr_combos(nbrs, deg, off, int(k), raw)
        return torch.unique(raw, dim=0).contiguous()
    return raw


def fill_s_major(
    G: torch.Tensor,
    nbrs: torch.Tensor,
    deg: torch.Tensor,
    unique_S: torch.Tensor,
    *,
    undirected: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack tests grouped in lexicographic order by ``unique_S``.

    Return ``x, y, S, counts, offsets``.
    """
    ext = _graph_batch_cuda()
    U = int(unique_S.size(0))
    k = int(unique_S.size(1)) if unique_S.ndim == 2 else 0
    device = unique_S.device
    counts = torch.empty(U, dtype=torch.int64, device=device)
    if U:
        ext.count_s_tests(G, nbrs, deg, unique_S.contiguous(), int(undirected), counts)
    off = _offsets_from_counts(counts)
    B = int(off[-1].item()) if off.numel() else 0
    x = torch.empty(B, dtype=torch.int32, device=device)
    y = torch.empty(B, dtype=torch.int32, device=device)
    S = torch.empty((B, k), dtype=torch.int32, device=device)
    if B:
        ext.fill_s_tests(
            G, nbrs, deg, unique_S.contiguous(), off, int(undirected), x, y, S
        )
    return x, y, S, counts, off


def _iter_s_chunks(counts_np: np.ndarray, b_max: int):
    U = int(counts_np.size)
    u0 = 0
    while u0 < U:
        n = int(counts_np[u0])
        if n > b_max:
            yield u0, u0 + 1
            u0 += 1
            continue
        acc = 0
        u1 = u0
        while (
            u1 < U and int(counts_np[u1]) <= b_max and acc + int(counts_np[u1]) <= b_max
        ):
            acc += int(counts_np[u1])
            u1 += 1
        if u1 == u0:
            u1 = u0 + 1
        yield u0, u1
        u0 = u1


def _iter_chunks(counts_np: np.ndarray, b_max: int):
    E = int(counts_np.size)
    e0 = 0
    while e0 < E:
        n = int(counts_np[e0])
        if n > b_max:
            r0 = 0
            while r0 < n:
                r1 = min(r0 + b_max, n)
                yield ("rank", e0, r0, r1)
                r0 = r1
            e0 += 1
            continue
        acc = 0
        e1 = e0
        while (
            e1 < E and int(counts_np[e1]) <= b_max and acc + int(counts_np[e1]) <= b_max
        ):
            acc += int(counts_np[e1])
            e1 += 1
        if e1 == e0:
            e1 = e0 + 1
        yield ("range", e0, e1)
        e0 = e1


def live_pairs(
    G_pairs,
    G_cand,
    k: int,
    device=None,
    nbrs=None,
    deg=None,
    *,
    undirected: bool = True,
    k0_unique: bool = True,
):
    r"""Live pairs at depth ``k``.

    Directed (``undirected=False``): each ``(x, y)`` with ``y ∈ nbrs(x)`` tests
    ``combinations(nbrs(x)\{y}, k)``. PC-stable uses this with ``k0_unique=False``
    so k=0 keeps both directions. FCI can keep only pairs with ``x < y`` at k=0.

    Undirected (``undirected=True``): unique ``x < y``, concatenating both
    endpoint candidate lists (legacy PC / all-S dumps).
    """
    k = int(k)
    if device is None:
        device = G_pairs.device if torch.is_tensor(G_pairs) else torch.device("cuda")
    G_t = _as_adj(G_pairs, device)
    if nbrs is None:
        nbrs_p, deg_p = compact_adj(G_t, device)
    else:
        assert deg is not None
        nbrs_p, deg_p = nbrs, deg
    if G_cand is None or G_cand is G_pairs:
        nbrs_c, deg_c = nbrs_p, deg_p
    else:
        G_c = _as_adj(G_cand, device)
        if G_c is G_t:
            nbrs_c, deg_c = nbrs_p, deg_p
        else:
            nbrs_c, deg_c = compact_adj(G_c, device)
    px0, py0 = emit_pairs(nbrs_p, deg_p)
    n0 = int(px0.numel())
    ext = _graph_batch_cuda()
    empty_i = torch.empty(0, dtype=torch.int32, device=device)
    empty_c = torch.empty(0, dtype=torch.int64, device=device)
    if n0 == 0:
        return nbrs_c, deg_c, empty_i, empty_i, empty_c
    E_t = torch.zeros(1, dtype=torch.int32, device=device)
    px = torch.empty(n0, dtype=torch.int32, device=device)
    py = torch.empty(n0, dtype=torch.int32, device=device)
    ranks = torch.empty(n0, dtype=torch.int64, device=device)
    ncomb = torch.empty(n0, dtype=torch.int64, device=device)
    if not undirected:
        if k == 0:
            if k0_unique:
                ext.pack_unique(px0, py0, px, py, ranks, ncomb, E_t)
                E = int(E_t.item())
                counts = torch.ones(E, dtype=torch.int64, device=device)
                return nbrs_c, deg_c, px[:E], py[:E], counts
            counts = torch.ones(n0, dtype=torch.int64, device=device)
            return nbrs_c, deg_c, px0, py0, counts
        counts0 = count_tests(nbrs_c, deg_c, px0, py0, k)
        if bool((counts0 < 0).any()):
            msg = "C(n, k) overflow while counting CI tests"
            raise OverflowError(msg)
        ext.pack_ncomb(px0, py0, counts0, px, py, ranks, ncomb, E_t)
        E = int(E_t.item())
        return nbrs_c, deg_c, px[:E], py[:E], ncomb[:E].contiguous()
    ext.pack_unique(px0, py0, px, py, ranks, ncomb, E_t)
    E = int(E_t.item())
    if E == 0:
        return nbrs_c, deg_c, empty_i, empty_i, empty_c
    px_u = px[:E].contiguous()
    py_u = py[:E].contiguous()
    if k == 0:
        counts = torch.ones(E, dtype=torch.int64, device=device)
        return nbrs_c, deg_c, px_u, py_u, counts
    counts_x = count_tests(nbrs_c, deg_c, px_u, py_u, k)
    counts_y = count_tests(nbrs_c, deg_c, py_u, px_u, k)
    if bool((counts_x < 0).any()) or bool((counts_y < 0).any()):
        msg = "C(n, k) overflow while counting CI tests"
        raise OverflowError(msg)
    counts0 = counts_x + counts_y
    keep = counts0 > 0
    if bool(keep.all()):
        return nbrs_c, deg_c, px_u, py_u, counts0
    idx = keep.nonzero(as_tuple=False).reshape(-1)
    return (
        nbrs_c,
        deg_c,
        px_u.index_select(0, idx),
        py_u.index_select(0, idx),
        counts0.index_select(0, idx).contiguous(),
    )


def _iter_all_s_chunks(
    G_pairs,
    G_cand,
    k: int,
    *,
    device=None,
    b_max: int = B_MAX,
    nbrs=None,
    deg=None,
    undirected: bool = True,
):
    """Yield pair-range / rank-window specs at frozen depth ``k`` (no fill)."""
    k = int(k)
    nbrs_c, deg_c, px, py, counts = live_pairs(
        G_pairs,
        G_cand,
        k,
        device=device,
        nbrs=nbrs,
        deg=deg,
        undirected=undirected,
    )
    E = int(px.numel())
    if E == 0:
        return
    counts_np = counts.detach().cpu().numpy()
    for chunk in _iter_chunks(counts_np, int(b_max)):
        if chunk[0] == "range":
            _, e0, e1 = chunk
            yield {
                "kind": "range",
                "nbrs_c": nbrs_c,
                "deg_c": deg_c,
                "px": px[e0:e1],
                "py": py[e0:e1],
                "counts": counts[e0:e1],
                "n": 0,
                "e0": int(e0),
                "e1": int(e1),
                "rank_base": 0,
                "E": E,
                "px_all": px,
                "py_all": py,
                "undirected": undirected,
            }
        else:
            _, e0, r0, r1 = chunk
            yield {
                "kind": "rank",
                "nbrs_c": nbrs_c,
                "deg_c": deg_c,
                "px": px[e0 : e0 + 1],
                "py": py[e0 : e0 + 1],
                "counts": None,
                "n": int(r1 - r0),
                "e0": int(e0),
                "e1": int(e0 + 1),
                "rank_base": int(r0),
                "E": E,
                "px_all": px,
                "py_all": py,
                "undirected": undirected,
            }


def _canon_xy(x: torch.Tensor, y: torch.Tensor):
    lo = torch.minimum(x, y)
    hi = torch.maximum(x, y)
    return lo, hi


def _unique_xy_s(x: torch.Tensor, y: torch.Tensor, S: torch.Tensor):
    """Drop duplicate ``(min(x,y), max(x,y), S)`` rows."""
    if int(x.numel()) == 0:
        return x, y, S
    lo, hi = _canon_xy(x, y)
    rows = torch.cat([lo.view(-1, 1), hi.view(-1, 1), S], dim=1)
    u = torch.unique(rows, dim=0)
    return (
        u[:, 0].contiguous(),
        u[:, 1].contiguous(),
        u[:, 2:].contiguous(),
    )


def _group_offsets(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Exclusive offsets for consecutive equal ``(x, y)`` (rows must be sorted)."""
    B = int(x.numel())
    device = x.device
    if B == 0:
        return torch.zeros(1, dtype=torch.int64, device=device)
    change = torch.ones(B, dtype=torch.bool, device=device)
    if B > 1:
        change[1:] = (x[1:] != x[:-1]) | (y[1:] != y[:-1])
    starts = change.nonzero(as_tuple=False).reshape(-1).to(torch.int64)
    off = torch.empty(int(starts.numel()) + 1, dtype=torch.int64, device=device)
    off[:-1] = starts
    off[-1] = B
    return off


def _fill_stream(spec: dict, px, py, k: int, rank_base: int, n: int):
    off = torch.tensor([0, int(n)], dtype=torch.int64, device=px.device)
    return fill_batch(
        spec["nbrs_c"], spec["deg_c"], px, py, off, k, rank_base=int(rank_base)
    )


def _fill_chunk(spec: dict, k: int):
    """Fill tests for one chunk. Undirected k>=1 uses both endpoint candidate lists."""
    px, py = spec["px"], spec["py"]
    k = int(k)
    if not spec.get("undirected", True):
        if spec["kind"] == "rank":
            n = int(spec["n"])
            off = torch.tensor([0, n], dtype=torch.int64, device=px.device)
            rank_base = int(spec["rank_base"])
        else:
            off = _offsets_from_counts(spec["counts"])
            rank_base = 0
        x, y, S = fill_batch(
            spec["nbrs_c"], spec["deg_c"], px, py, off, k, rank_base=rank_base
        )
        return x, y, S, off, rank_base
    if k == 0:
        if spec["kind"] == "rank":
            n = int(spec["n"])
            off = torch.tensor([0, n], dtype=torch.int64, device=px.device)
            rank_base = int(spec["rank_base"])
        else:
            off = _offsets_from_counts(spec["counts"])
            rank_base = 0
        x, y, S = fill_batch(
            spec["nbrs_c"], spec["deg_c"], px, py, off, k, rank_base=rank_base
        )
        return x, y, S, off, rank_base

    c_x = count_tests(spec["nbrs_c"], spec["deg_c"], px, py, k)
    c_y = count_tests(spec["nbrs_c"], spec["deg_c"], py, px, k)
    if spec["kind"] == "rank":
        n = int(spec["n"])
        rank_base = int(spec["rank_base"])
        cx0 = int(c_x[0].item()) if int(c_x.numel()) else 0
        parts = []
        a0, a1 = max(rank_base, 0), min(rank_base + n, cx0)
        if a1 > a0:
            parts.append(_fill_stream(spec, px, py, k, a0, a1 - a0))
        b0, b1 = (
            max(rank_base, cx0),
            min(rank_base + n, cx0 + int(c_y[0].item() if int(c_y.numel()) else 0)),
        )
        if b1 > b0:
            parts.append(_fill_stream(spec, py, px, k, b0 - cx0, b1 - b0))
        if not parts:
            empty = torch.empty(0, dtype=torch.int32, device=px.device)
            S0 = torch.empty((0, k), dtype=torch.int32, device=px.device)
            off = torch.zeros(1, dtype=torch.int64, device=px.device)
            return empty, empty, S0, off, 0
        x = torch.cat([p[0] for p in parts])
        y = torch.cat([p[1] for p in parts])
        S = torch.cat([p[2] for p in parts], dim=0)
    else:
        x1, y1, S1 = fill_batch(
            spec["nbrs_c"], spec["deg_c"], px, py, _offsets_from_counts(c_x), k
        )
        x2, y2, S2 = fill_batch(
            spec["nbrs_c"], spec["deg_c"], py, px, _offsets_from_counts(c_y), k
        )
        x = torch.cat([x1, x2])
        y = torch.cat([y1, y2])
        S = torch.cat([S1, S2], dim=0)
    x, y, S = _unique_xy_s(x, y, S)
    off = _group_offsets(x, y)
    return x, y, S, off, 0


def build_all_s(
    G_pairs,
    G_cand,
    k: int,
    *,
    device=None,
    b_max: int = B_MAX,
    nbrs=None,
    deg=None,
):
    """Yield every combinadic ``(x, y, S)`` batch at frozen depth ``k``.

    One batch if ``sum C(n,k) <= b_max``; otherwise pair-range / rank chunks.
    Union of batches is all ``itertools.combinations(cand(x,y), k)`` for live
    unique pairs ``x < y``. Does not skip edges deleted later in the depth —
    that is ``run_depth``.
    """
    k = int(k)
    if k >= 1:
        if device is None:
            device = (
                G_pairs.device if torch.is_tensor(G_pairs) else torch.device("cuda")
            )
        G_t = _as_adj(G_pairs, device)
        G_c = G_t if G_cand is None or G_cand is G_pairs else _as_adj(G_cand, device)
        if nbrs is None:
            nbrs_c, deg_c = compact_adj(G_c, device)
        else:
            assert deg is not None
            nbrs_c, deg_c = nbrs, deg
        unique_S = unique_S_lex(nbrs_c, deg_c, k)
        U = int(unique_S.size(0))
        if U == 0:
            return
        counts0 = torch.empty(U, dtype=torch.int64, device=unique_S.device)
        _graph_batch_cuda().count_s_tests(
            G_t, nbrs_c, deg_c, unique_S.contiguous(), 1, counts0
        )
        empty_i = torch.empty(0, dtype=torch.int32, device=unique_S.device)
        for u0, u1 in _iter_s_chunks(counts0.detach().cpu().numpy(), int(b_max)):
            uS = unique_S[u0:u1].contiguous()
            with _cuda_step("fill"):
                x, y, S, _counts, off = fill_s_major(
                    G_t, nbrs_c, deg_c, uS, undirected=True
                )
            yield {
                "x": x,
                "y": y,
                "S": S,
                "px": empty_i,
                "py": empty_i,
                "offsets": off,
                "e0": int(u0),
                "e1": int(u1),
                "rank_base": 0,
                "E": U,
                "px_all": empty_i,
                "py_all": empty_i,
            }
        return
    for spec in _iter_all_s_chunks(
        G_pairs, G_cand, k, device=device, b_max=b_max, nbrs=nbrs, deg=deg
    ):
        with _cuda_step("fill"):
            x, y, S, off, rank_base = _fill_chunk(spec, k)
        yield {
            "x": x,
            "y": y,
            "S": S,
            "px": spec["px"],
            "py": spec["py"],
            "offsets": off,
            "e0": spec["e0"],
            "e1": spec["e1"],
            "rank_base": int(rank_base),
            "E": spec["E"],
            "px_all": spec["px_all"],
            "py_all": spec["py_all"],
        }


def _dedup_hit_parts(parts, k: int, device, *, undirected: bool = True):
    """Keep one sepset per pair (undirected ``(x, y)`` for PC)."""
    if not parts:
        return _empty_hits(k, device, to_host=False)
    x = torch.cat([p[0] for p in parts])
    y = torch.cat([p[1] for p in parts])
    S = torch.cat([p[2] for p in parts], dim=0)
    n = int(x.numel())
    if n == 0:
        return _empty_hits(k, device, to_host=False)
    if undirected:
        lo, hi = _canon_xy(x, y)
        key = lo.to(torch.int64) * 1_000_003 + hi.to(torch.int64)
    else:
        key = x.to(torch.int64) * 1_000_003 + y.to(torch.int64)
    order = torch.argsort(key)
    key_s = key[order]
    keep = torch.ones(n, dtype=torch.bool, device=device)
    if n > 1:
        keep[1:] = key_s[1:] != key_s[:-1]
    pick = order[keep]
    return x.index_select(0, pick), y.index_select(0, pick), S.index_select(0, pick)


def pack_sepset_csr(parts, *, undirected: bool = True, device=None, p: int = 0):
    """Pack device hit batches into ``S`` and ``ranges[p, p, 2]``.

    ``ranges[x, y] = (start, end)`` into 1D ``S``. Still an edge: ``(-1, -1)``.
    Empty sepset: ``(0, 0)``. Undirected pairs write both triangles.
    """
    if device is None:
        device = parts[0][0].device if parts else torch.device("cpu")
    p = int(p)
    empty_S = torch.empty(0, dtype=torch.int64, device=device)
    ranges = torch.full((p, p, 2), -1, dtype=torch.int64, device=device)
    if not parts:
        return empty_S, ranges
    xs, ys, kn, flats = [], [], [], []
    for x, y, S in parts:
        H = int(x.numel())
        if H == 0:
            continue
        k = int(S.shape[1]) if S.ndim == 2 else 0
        if undirected:
            lo, hi = torch.minimum(x, y), torch.maximum(x, y)
            xs.append(lo.to(torch.int64))
            ys.append(hi.to(torch.int64))
        else:
            xs.append(x.to(torch.int64))
            ys.append(y.to(torch.int64))
        kn.append(torch.full((H,), k, dtype=torch.int64, device=device))
        if k > 0:
            flats.append(S.reshape(-1).to(torch.int64))
    if not xs:
        return empty_S, ranges
    x = torch.cat(xs)
    y = torch.cat(ys)
    k_per = torch.cat(kn)
    S_all = torch.cat(flats) if flats else empty_S
    ns = int(x.numel())
    starts = torch.empty(ns, dtype=torch.int64, device=device)
    if ns:
        torch.cumsum(k_per, 0, out=starts)
        starts = starts - k_per
    ends = starts + k_per
    empty = k_per == 0
    starts = torch.where(empty, torch.zeros_like(starts), starts)
    ends = torch.where(empty, torch.zeros_like(ends), ends)
    ranges[x, y, 0] = starts
    ranges[x, y, 1] = ends
    ranges[y, x, 0] = starts
    ranges[y, x, 1] = ends
    return S_all, ranges


def fci_last_writer_mirror(ranges: torch.Tensor) -> torch.Tensor:
    """Match sequential FCI: later ``x`` overwrites both ``(x,y)`` and ``(y,x)``.

    Sequential loops ``x = 0..p-1``, so if both endpoints find a separator the
    stored set is from ``max(x, y)``. Copy that slice into both triangles.
    """
    p = int(ranges.size(0))
    if p == 0:
        return ranges
    i = torch.arange(p, device=ranges.device)
    ii, jj = torch.meshgrid(i, i, indexing="ij")
    mx = torch.maximum(ii, jj)
    mn = torch.minimum(ii, jj)
    hi = ranges[mx, mn]
    lo = ranges[mn, mx]
    pick = torch.where((hi[:, :, 0] >= 0).unsqueeze(-1), hi, lo)
    return torch.where((ii == jj).unsqueeze(-1), ranges, pick)


def _workspace_allocation_error(
    entity: str, elements: int, dtype, device, cause: BaseException
) -> RuntimeError:
    """Build the explicit message used when a CUDA workspace cannot be sized."""
    needed_mib = elements * dtype.itemsize / (1024.0 * 1024.0)
    try:
        free, total = torch.cuda.mem_get_info(device)
        available = (
            f"{free / (1024.0 * 1024.0):.1f} MiB "
            f"(total {total / (1024.0 * 1024.0):.1f} MiB)"
        )
    except Exception:  # Preserve the original exception during OOM diagnostics.
        available = "unknown"
    return RuntimeError(
        f"CUDA workspace allocation failed for {entity}: need "
        f"{needed_mib:.1f} MiB, available {available} on {device}"
    )


class SepsetCSR:
    """Device sepsets: 1D ``S`` plus ``ranges[p, p, 2] = (start, end)``.

    Still an edge: ``(-1, -1)``. Empty sepset (k=0): ``(0, 0)``.
    PC-stable claims ``ranges[x, y]`` only (directed). ``ranges[x,y,0]`` is a
    mutex (``atomicCAS(-1, -2)``) so the first hit of that directed pair wins.

    ``S`` starts small and grows on demand. An allocation failure raises a
    ``RuntimeError`` naming the workspace, the required size and the memory
    currently available on the device.
    """

    def __init__(self, p: int, device, k_max: int = 96):
        p = int(p)
        nmax = p * (p - 1)
        self.p = p
        self.nmax = nmax
        self.k_max = int(k_max)
        self.ranges = torch.full((p, p, 2), -1, dtype=torch.int64, device=device)
        initial = min(nmax * self.k_max, max(4096, 16 * p * self.k_max))
        self.S = self._allocate(initial, "sepset CSR buffer")
        self.nnz = torch.zeros(1, dtype=torch.int64, device=device)
        # Host-side mirror of ``nnz`` so the fused commit can size ``S`` without
        # a device->host sync; refreshed after every write that reaches the host.
        self._nnz_host = 0
        self._overflow = torch.zeros(1, dtype=torch.int32, device=device)
        self._tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._pc_bufs: Any = None
        self._pc_union_buf: Any = None

    def _allocate(self, elements: int, entity: str):
        elements = max(int(elements), 1)
        try:
            return torch.empty(elements, dtype=torch.int64, device=self.ranges.device)
        except torch.OutOfMemoryError as exc:
            raise _workspace_allocation_error(
                entity, elements, torch.int64, self.ranges.device, exc
            ) from exc
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            raise _workspace_allocation_error(
                entity, elements, torch.int64, self.ranges.device, exc
            ) from exc

    def _ensure_capacity(self, needed: int, entity: str) -> None:
        """Grow ``S`` to at least ``needed`` int64 elements, preserving data."""
        needed = int(needed)
        if needed <= int(self.S.numel()):
            return
        grown = self._allocate(needed, entity)
        if self.S.numel():
            grown[: self.S.numel()] = self.S
        self.S = grown

    def _check_overflow(self, stage: str) -> None:
        """Fail loudly if a device-side commit could not fit in ``S``."""
        ovf = int(self._overflow.item())
        if ovf:
            msg = (
                f"{stage}: sepset CSR overflow ({ovf} reservations did not fit); "
                "the workspace needs a larger S buffer"
            )
            raise RuntimeError(msg)

    def commit_union_fused(
        self,
        px: torch.Tensor,
        py: torch.Tensor,
        union: torch.Tensor,
        flags: torch.Tensor,
    ) -> None:
        """Union sepsets for one depth in two device launches.

        Replaces the host torch chain (canonicalize -> argsort -> group -> OR
        bitmap -> cumsum -> scatter) with ``commit_union_mark`` +
        ``commit_union_write``. ``union`` is ``[E0, p]`` uint8 member marks and
        ``flags`` ``[E0]`` uint8 hit flags, both aligned with the directed
        pairs ``(px, py)``; ``rows[e] == e`` here (the caller runs a single
        wave). ``S`` is grown up front to a per-depth upper bound
        (``p`` members x ``E0`` pairs) so the write kernel never overflows.
        """
        E0 = int(px.numel())
        if E0 == 0:
            return
        p = int(self.p)
        bound = self._nnz_host + p * E0
        self._ensure_capacity(bound, "sepset CSR buffer (PC/FCI union commit)")
        if self._tables is None or int(self._tables[1].numel()) != p * p:
            dev = self.ranges.device
            tab = torch.full((2 * p * p,), -1, dtype=torch.int32, device=dev)
            self._tables = (tab, tab[: p * p], tab[p * p :])
        tabs, tab_a, tab_b = self._tables
        tabs.fill_(-1)
        ext = _graph_batch_cuda()
        ext.commit_union_mark(
            px.contiguous(),
            py.contiguous(),
            flags.contiguous(),
            tab_a,
            tab_b,
            p,
        )
        self._overflow.zero_()
        ext.commit_union_write(
            tab_a,
            tab_b,
            union.contiguous(),
            self.ranges,
            self.nnz,
            self.S,
            self._overflow,
        )
        self._check_overflow("PC/FCI union commit")
        self._nnz_host = int(self.nnz.item())

    def claim(
        self,
        px: torch.Tensor,
        py: torch.Tensor,
        first: torch.Tensor,
        nbrs,
        deg,
        k: int,
        rank_base: int,
        rank_bases,
    ) -> None:
        """Record first-hit sepsets for this stripe.

        CUDA: one thread per directed pair. ``int64`` ``atomicCAS`` on
        ``ranges[x,y]``; winner unranks ``S`` into the CSR buffer.
        """
        if int(px.numel()) == 0:
            return
        device = px.device
        k = int(k)
        if k > 0:
            used = int(self.nnz.item())
            self._nnz_host = used
            self._ensure_capacity(
                used + int(px.numel()) * k,
                "sepset CSR buffer (first-hit claim)",
            )
        rb = (
            rank_bases
            if rank_bases is not None
            else torch.empty(0, dtype=torch.int64, device=device)
        )
        self._overflow.zero_()
        _graph_batch_cuda().claim_sepsets(
            px.contiguous(),
            py.contiguous(),
            first.contiguous(),
            rb.contiguous(),
            int(rank_base),
            RANK_MAX,
            nbrs,
            deg,
            self.ranges,
            self.nnz,
            self.S,
            self._overflow,
            int(k),
        )
        self._check_overflow("first-hit claim")

    def commit_union(
        self,
        px: torch.Tensor,
        py: torch.Tensor,
        union: torch.Tensor | None,
        flags: torch.Tensor,
    ) -> None:
        """PC-stable: append union-of-members sepsets for this depth's hits.

        As in ``parallel_pc_stable``, each separated pair gets one record holding
        the sorted union of every separating conditioning set's members. Both
        symmetric cells refer to that record. The union merges the two directed
        enumerations ``(x, y)`` and ``(y, x)``.
        ``flags`` marks directed pairs with at least one separator this depth
        (k = 0 unions are empty); ``union[e, v]`` marks member ``v`` of pair
        row ``e``.
        """
        if not bool(flags.any()):
            return
        # k = 0 (``union is None``): every separated pair gets the EMPTY
        # sepset, i.e. ranges = (0, 0) on both triangles, and nnz stays 0.
        # Records for both directions are already present in px/py, so skip
        # the canonicalization / bitmap merge entirely.
        if union is None:
            idx = flags.to(torch.bool).nonzero(as_tuple=False).reshape(-1)
            x = px.index_select(0, idx).to(torch.int64)
            y = py.index_select(0, idx).to(torch.int64)
            z = torch.zeros_like(x)
            self.ranges[x, y, 0] = z
            self.ranges[x, y, 1] = z
            return
        # Whole merge runs on device: the [E0, p] union bitmap can be tens of
        # MB (p=300, k=1: 53k x 300) and the host numpy pass over it dominated
        # the skeleton for large p.
        p = int(self.p)
        dev = self.ranges.device
        px = px.contiguous()
        py = py.contiguous()
        idx = flags.to(torch.bool).nonzero(as_tuple=False).reshape(-1)
        cx = torch.minimum(px.index_select(0, idx), py.index_select(0, idx))
        cy = torch.maximum(px.index_select(0, idx), py.index_select(0, idx))
        key = cx.to(torch.int64) * p + cy.to(torch.int64)
        order = torch.argsort(key, stable=True)
        key_s = key.index_select(0, order)
        idx_s = idx.index_select(0, order)
        H = int(key_s.numel())
        first = torch.ones(H, dtype=torch.bool, device=dev)
        if H > 1:
            first[1:] = key_s[1:] != key_s[:-1]
        starts_idx = first.nonzero(as_tuple=False).reshape(-1)
        uniq = key_s.index_select(0, starts_idx)
        K = int(uniq.numel())
        nrows = torch.empty(K, dtype=torch.int64, device=dev)
        nrows[:-1] = starts_idx[1:] - starts_idx[:-1]
        nrows[-1] = H - starts_idx[-1]
        if union is not None:
            # Each canonical key owns <= 2 directed rows (one per direction);
            # OR the first row of each key with its remaining rows.
            orred = union.index_select(0, idx_s.index_select(0, starts_idx)).clone()
            extra = 1
            while True:
                sel = nrows > extra
                if not bool(sel.any()):
                    break
                rows = idx_s.index_select(0, starts_idx[sel] + extra)
                orred[sel] |= union.index_select(0, rows)
                extra += 1
            member_lens = orred.sum(dim=1).to(torch.int64)
            nz = orred.reshape(-1).nonzero(as_tuple=False).reshape(-1)
            flat = (nz % p).to(self.S.dtype)
        else:
            member_lens = torch.zeros(K, dtype=torch.int64, device=dev)
            flat = None
        total = int(member_lens.sum().item())
        nnz = int(self.nnz.item())
        self._ensure_capacity(nnz + total, "sepset CSR buffer (PC union commit)")
        self._nnz_host = nnz + total
        ends = nnz + torch.cumsum(member_lens, 0)
        starts = ends - member_lens
        if total:
            # Row-major nonzero order is (key, member) so per-key member lists
            # are already concatenated in increasing column order.
            assert flat is not None
            self.S[nnz : nnz + total] = flat
        # One batched scatter into the CSR range map (both triangles).
        ux = uniq // p
        uy = uniq % p
        self.ranges[ux, uy, 0] = starts
        self.ranges[ux, uy, 1] = ends
        self.ranges[uy, ux, 0] = starts
        self.ranges[uy, ux, 1] = ends
        self.nnz.fill_(nnz + total)

    def commit_walk(
        self,
        py: torch.Tensor,
        hit_k: torch.Tensor,
        hit_rank: torch.Tensor,
        row_off: torch.Tensor,
        nbrs,
        deg,
    ) -> None:
        r"""Last-writer sepsets: one CTA walks ``x = 0..p-1`` with ``__syncthreads``.

        ``hit_k[e] >= 1`` means directed pair ``e`` (x-major, ``py[e]``) found a
        first-hit at combinadic ``hit_rank[e]`` of ``combinations(Pdsep(x)\{y}, k)``.
        Later ``x`` overwrite both triangles, matching sequential ``pdsep_v2``.
        """
        if int(py.numel()) == 0:
            return
        used = int(self.nnz.item())
        added = int(hit_k.clamp(min=0).sum().item())
        self._ensure_capacity(used + added, "sepset CSR buffer (Possible-D-SEP commit)")
        self._nnz_host = used + added
        self._overflow.zero_()
        _graph_batch_cuda().pdsep_commit_walk(
            py.contiguous(),
            hit_k.contiguous(),
            hit_rank.contiguous(),
            row_off.contiguous(),
            nbrs,
            deg,
            self.ranges,
            self.nnz,
            self.S,
            self._overflow,
        )
        self._check_overflow("Possible-D-SEP commit")

    def pack(self):
        """Slice used ``S``; ``ranges`` is the full ``[p, p, 2]`` map."""
        nnz = int(self.nnz.item())
        return self.S[:nnz].contiguous(), self.ranges


def first_hit_ranks(r, offsets, n_samples, k, tau, fci: bool) -> torch.Tensor:
    """First independent local rank per pair (unfused). ``RANK_MAX`` if none.

    CUDA: one thread per test, ``atomicMin`` into ``int64 first[E]``.
    """
    E = int(offsets.numel()) - 1
    first = torch.full((max(E, 0),), RANK_MAX, dtype=torch.int64, device=r.device)
    if int(r.numel()) == 0 or E <= 0:
        return first
    _graph_batch_cuda().first_hit(
        r, offsets, first, int(n_samples), int(k), float(tau), bool(fci)
    )
    return first


def _prefix_pmax(pvals, px, py, offsets, first, pmax) -> None:
    B = int(pvals.shape[0])
    if B == 0:
        return
    pair_id = np.searchsorted(offsets[1:], np.arange(B), side="right")
    local = np.arange(B) - offsets[pair_id]
    fr = first[pair_id]
    keep = ((fr == RANK_MAX) | (local <= fr)) & np.isfinite(pvals)
    if not np.any(keep):
        return
    np.maximum.at(pmax, (px[pair_id][keep], py[pair_id][keep]), pvals[keep])


def _pmax_device(pmax, device) -> torch.Tensor | None:
    """Resident float64 ``pMax`` on GPU."""
    if pmax is None:
        return None
    if torch.is_tensor(pmax):
        t = pmax
        if t.dtype != torch.float64:
            t = t.to(dtype=torch.float64)
        if t.device != torch.device(device):
            t = t.to(device)
        return t.contiguous()
    return torch.as_tensor(
        np.asarray(pmax, dtype=np.float64), device=device
    ).contiguous()


def _pmax_result(pmax_in, pmax_work, *, to_host: bool):
    if pmax_work is None:
        return None
    keep_device = torch.is_tensor(pmax_in) and not to_host
    if keep_device:
        return pmax_work
    if torch.is_tensor(pmax_work):
        return pmax_work.detach().cpu().numpy()
    return pmax_work


class GraphBatchState:
    """Resident PC/FCI wavefront state. Live pairs stay on GPU.

    Per wave the host D2H only ``E`` (0 means stop).
    """

    def __init__(self, G: torch.Tensor, batcher: CudaFisherZ, alpha: float):
        self.G = G
        self.batcher = batcher
        self.device = G.device
        self.alpha = float(alpha)
        self.tau = float(norm.ppf(1.0 - alpha / 2.0))
        self.k = 0
        self.nbrs = torch.empty((0, 0), dtype=torch.int32, device=self.device)
        self.deg = torch.empty(0, dtype=torch.int32, device=self.device)
        self.nbrs_c = torch.empty((0, 0), dtype=torch.int32, device=self.device)
        self.deg_c = torch.empty(0, dtype=torch.int32, device=self.device)
        self.E_t = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.H_t = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.src = 0
        self.px = torch.empty(0, dtype=torch.int32, device=self.device)
        self.py = torch.empty(0, dtype=torch.int32, device=self.device)
        self.ranks = torch.empty(0, dtype=torch.int64, device=self.device)
        self.ncomb = torch.empty(0, dtype=torch.int64, device=self.device)
        self.px_b = torch.empty(0, dtype=torch.int32, device=self.device)
        self.py_b = torch.empty(0, dtype=torch.int32, device=self.device)
        self.ranks_b = torch.empty(0, dtype=torch.int64, device=self.device)
        self.ncomb_b = torch.empty(0, dtype=torch.int64, device=self.device)
        self.x = torch.empty(0, dtype=torch.int32, device=self.device)
        self.y = torch.empty(0, dtype=torch.int32, device=self.device)
        self.S = torch.empty((0, 0), dtype=torch.int32, device=self.device)
        self.r = torch.empty(0, dtype=batcher.dtype, device=self.device)
        self.hit = torch.empty(0, dtype=torch.uint8, device=self.device)
        self.hit_x = torch.empty(0, dtype=torch.int32, device=self.device)
        self.hit_y = torch.empty(0, dtype=torch.int32, device=self.device)
        self.hit_S = torch.empty((0, 0), dtype=torch.int32, device=self.device)

    def _alloc(self, cap: int, k: int) -> None:
        d, k = self.device, int(k)
        cap = max(int(cap), 1)

        def i32(n):
            return torch.empty(n, dtype=torch.int32, device=d)

        def i64(n):
            return torch.empty(n, dtype=torch.int64, device=d)

        self.px, self.py, self.ranks, self.ncomb = (
            i32(cap),
            i32(cap),
            i64(cap),
            i64(cap),
        )
        self.px_b, self.py_b, self.ranks_b, self.ncomb_b = (
            i32(cap),
            i32(cap),
            i64(cap),
            i64(cap),
        )
        self.x, self.y = i32(cap), i32(cap)
        self.S = torch.empty((cap, k), dtype=torch.int32, device=d)
        self.r = torch.empty(cap, dtype=self.batcher.dtype, device=d)
        self.hit = torch.empty(cap, dtype=torch.uint8, device=d)
        self.hit_x, self.hit_y = i32(cap), i32(cap)
        self.hit_S = torch.empty((cap, k), dtype=torch.int32, device=d)
        if _PROF is not None:
            torch.cuda.synchronize()

    def init_depth(self, k: int, G_cand=None, nbrs=None, deg=None) -> int:
        """Compact ``G``, emit pairs, pack live list. D2H ``E`` once."""
        ext = _graph_batch_cuda()
        k = int(k)
        self.k = k
        if nbrs is None:
            with _cuda_step("compact"):
                self.nbrs, self.deg = compact_adj(self.G, self.device)
        else:
            assert deg is not None
            self.nbrs, self.deg = nbrs, deg
        if G_cand is None or G_cand is self.G:
            self.nbrs_c, self.deg_c = self.nbrs, self.deg
        else:
            with _cuda_step("compact"):
                self.nbrs_c, self.deg_c = compact_adj(G_cand, self.device)
        with _cuda_step("emit"):
            px0, py0 = emit_pairs(self.nbrs, self.deg)
        n0 = int(px0.numel())
        with _cuda_step("alloc"):
            self._alloc(n0, k)
        with _cuda_step("zero_E"):
            self.E_t.zero_()
            self.H_t.zero_()
        self.src = 0
        if n0 == 0:
            return 0
        if k == 0:
            with _cuda_step("pack"):
                ext.pack_unique(
                    px0, py0, self.px, self.py, self.ranks, self.ncomb, self.E_t
                )
        else:
            with _cuda_step("count"):
                counts = count_tests(self.nbrs_c, self.deg_c, px0, py0, k)
            with _cuda_step("overflow"):
                overflow = bool((counts < 0).any())
            if overflow:
                msg = "C(n, k) overflow while counting CI tests"
                raise OverflowError(msg)
            with _cuda_step("pack"):
                ext.pack_ncomb(
                    px0, py0, counts, self.px, self.py, self.ranks, self.ncomb, self.E_t
                )
        with _cuda_step("E_item"):
            E = int(self.E_t.item())
        return E

    def _live(self, E: int):
        if self.src == 0:
            return self.px[:E], self.py[:E], self.ranks[:E], self.ncomb[:E]
        return self.px_b[:E], self.py_b[:E], self.ranks_b[:E], self.ncomb_b[:E]

    def _dst(self):
        if self.src == 0:
            return self.px_b, self.py_b, self.ranks_b, self.ncomb_b
        return self.px, self.py, self.ranks, self.ncomb

    def wave(self, E: int, *, fci: bool, pmax) -> tuple[int, object]:
        """Evaluate one batch and return (remaining pair count, pmax).

        Fill the conditioning sets, compute LDLᵀ partial correlations, record
        hits, and compact the live pairs. Copy their count to the host. In FCI
        mode, update the supplied pmax with the largest p-values seen.
        """
        ext = _graph_batch_cuda()
        k = self.k
        px, py, ranks, ncomb = self._live(E)
        x, y, S = self.x[:E], self.y[:E], self.S[:E]
        with _cuda_step("fill"):
            fill_at_ranks(self.nbrs_c, self.deg_c, px, py, ranks, k, x=x, y=y, S=S)
        with _cuda_step("ldlt"):
            r = self.batcher.batch_r(x, y, S, out=self.r[:E])
        hit = self.hit[:E]
        with _cuda_step("hits"):
            ext.hits_from_r(r, hit, self.batcher.n_samples, k, self.tau, bool(fci))
        if fci:
            pvals = np.asarray(
                _p_value_from_r(r.detach().cpu().numpy(), self.batcher.n_samples, k)
            )
            px_np = px.detach().cpu().numpy()
            py_np = py.detach().cpu().numpy()
            ok = np.isfinite(pvals)
            np.maximum.at(pmax, (px_np[ok], py_np[ok]), pvals[ok])
            if k == 0:
                np.maximum.at(pmax, (py_np[ok], px_np[ok]), pvals[ok])
        with _cuda_step("zero_E"):
            self.E_t.zero_()
        dx, dy, dr, dn = self._dst()
        with _cuda_step("compact_live"):
            ext.compact_live(
                px,
                py,
                ranks,
                ncomb,
                hit,
                S,
                int(k),
                int(E),
                dx,
                dy,
                dr,
                dn,
                self.hit_x,
                self.hit_y,
                self.hit_S,
                self.E_t,
                self.H_t,
            )
        self.src = 1 - self.src
        if _PROF is not None:
            _PROF.n_waves += 1
        with _cuda_step("E_item"):
            E_next = int(self.E_t.item())
        return E_next, pmax

    def hits(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the materialized hit rows as ``(x, y, S)`` tensors."""
        with _cuda_step("H_item"):
            H = int(self.H_t.item())
        if H == 0:
            return _empty_hits(self.k, self.device, to_host=False)
        return self.hit_x[:H], self.hit_y[:H], self.hit_S[:H]


@overload
def _empty_hits(
    k: int, device, to_host: Literal[False]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@overload
def _empty_hits(
    k: int, device, to_host: Literal[True]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@overload
def _empty_hits(
    k: int, device, to_host: bool
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[np.ndarray, np.ndarray, np.ndarray]
): ...


def _empty_hits(k: int, device, to_host: bool):
    if to_host:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros((0, k), dtype=np.int32),
        )
    return (
        torch.zeros(0, dtype=torch.int32, device=device),
        torch.zeros(0, dtype=torch.int32, device=device),
        torch.zeros((0, k), dtype=torch.int32, device=device),
    )


def _mark_union(
    union: torch.Tensor | None,
    hit_flags: torch.Tensor | None,
    rows: torch.Tensor,
    offsets: torch.Tensor,
    indep: torch.Tensor,
    S: torch.Tensor | None,
) -> None:
    """Mark separating-member union bits for the independent tests of a stripe.

    PC-stable parity with ``parallel_pc_stable``: the sepset of a pair is the
    union of the members of EVERY separating conditioning set of the depth
    (one record per pair, both directed enumerations merged).  ``rows[e]`` is
    the pair's row in the depth-level union bitmap ``[E0, p]``; ``indep`` and
    ``S`` are aligned with the stripe's tests; ``offsets`` maps tests to
    stripe-local pair indices.  Byte stores are idempotent, so duplicate
    marks from parallel tests are fine.
    """
    if union is None and hit_flags is None:
        return
    idx = indep.nonzero(as_tuple=False).reshape(-1)
    if int(idx.numel()) == 0:
        return
    pair_id = torch.searchsorted(offsets[1:], idx, right=True)
    rows_sel = rows.index_select(0, pair_id)
    if hit_flags is not None:
        hit_flags[rows_sel] = 1
    if union is not None and S is not None and S.numel():
        k = int(S.size(1))
        S_sel = S.index_select(0, idx).reshape(-1)
        flat_rows = rows_sel.unsqueeze(1).expand(-1, k).reshape(-1)
        union[flat_rows, S_sel] = 1


def _ci_stream(
    batcher: Any,
    G_work: torch.Tensor,
    nbrs_c: torch.Tensor,
    deg_c: torch.Tensor,
    px: torch.Tensor,
    py: torch.Tensor,
    n_this: torch.Tensor,
    k: int,
    rank_base: int,
    rank_bases: torch.Tensor | None,
    tau: float,
    fci: bool,
    pmax_work,
    hit_parts: list,
    csr: SepsetCSR | None = None,
    write_g: bool = True,
    alpha: float = 0.05,
    rows: torch.Tensor | None = None,
    union: torch.Tensor | None = None,
    hit_flags: torch.Tensor | None = None,
) -> int:
    """Run one directed rank stripe. Returns the number of CI tests.

    Default: fused ``ci_from_comb`` for ``k ≤ 32`` — ``ceil(B / 256)`` CTAs × 256
    threads, one thread per test. ``k = 33..96`` unranks ``S`` then
    ``CudaFisherZ.batch_r``. Each fused thread unranks ``S`` (integer), runs
    LDLᵀ (FP64 by default), and ``atomicMin`` the first independent rank.
    PC-stable sets ``write_g=False`` so ``G`` stays frozen until the depth ends.
    ``CudaChisqHost`` skips fusion: CUDA χ² for all k (k≥2 packs S into one Z).
    PC union mode (``union``/``hit_flags``/``rows``): instead of (or in
    addition to) the first-hit rank, mark the union of separating-set members
    per pair. The CPU reference scans past the first hit and stores that
    union as the pair's single separating-set record.
    """
    keep = n_this > 0
    if int(px.numel()) == 0 or not bool(keep.any()):
        return 0
    if not bool(keep.all()):
        idx = keep.nonzero(as_tuple=False).reshape(-1)
        px = px.index_select(0, idx)
        py = py.index_select(0, idx)
        n_this = n_this.index_select(0, idx)
        if rank_bases is not None:
            rank_bases = rank_bases.index_select(0, idx)
        if rows is not None:
            rows = rows.index_select(0, idx)
    off = _offsets_from_counts(n_this)
    with _cuda_step("offsets_item"):
        B = int(off[-1].item()) if off.numel() else 0
    if B == 0:
        return 0
    E = int(px.numel())
    device = px.device
    k = int(k)
    ext = _graph_batch_cuda()
    rb = (
        rank_bases
        if rank_bases is not None
        else torch.empty(0, dtype=torch.int64, device=device)
    )
    host_ci = is_host_ci(batcher)
    use_fused = (not host_ci) and USE_FUSED and k <= int(ext.fused_k_max())
    first = torch.full((E,), RANK_MAX, dtype=torch.int64, device=device)
    off = off.contiguous()
    px = px.contiguous()
    py = py.contiguous()
    if host_ci:
        with _cuda_step("fill"):
            x, y, S = fill_batch(
                nbrs_c,
                deg_c,
                px,
                py,
                off,
                k,
                rank_base=rank_base,
                rank_bases=rank_bases,
            )
        with _cuda_step("chisq"):
            pvals = batcher.batch_t(x, y, S)
        if fci:
            indep = torch.isfinite(pvals) & (pvals >= float(alpha))
        else:
            indep = torch.isfinite(pvals) & (pvals > float(alpha))
        indep_u8 = indep.to(torch.uint8)
        if write_g:
            with _cuda_step("delete"):
                delete_if_indep(G_work, x, y, indep_u8)
        with _cuda_step("first_hit"):
            first = first_hit_from_p_t(pvals, off, alpha, fci)
        if union is not None or hit_flags is not None:
            assert rows is not None
            _mark_union(union, hit_flags, rows, off, indep, S if k > 0 else None)
        if fci and pmax_work is not None and B and E:
            with _cuda_step("prefix_pmax"):
                prefix_pmax_t(pvals, px, py, off, first, pmax_work, k=k)
    elif use_fused:
        r_out = torch.empty(0, dtype=batcher.corr.dtype, device=device)
        if fci:
            r_out = torch.full(
                (B,), float("nan"), dtype=batcher.corr.dtype, device=device
            )
        with _cuda_step("fused"):
            ext.ci_from_comb(
                batcher.corr,
                G_work,
                nbrs_c,
                deg_c,
                px,
                py,
                off,
                int(k),
                int(rank_base),
                rb,
                first,
                int(B),
                int(batcher.n_samples),
                float(tau),
                float(batcher.eps),
                bool(fci),
                bool(write_g),
                r_out,
                torch.empty(0, dtype=torch.int8, device=device),
                union
                if union is not None
                else torch.empty(0, dtype=torch.uint8, device=device),
                rows.to(torch.int32)
                if rows is not None
                else torch.empty(0, dtype=torch.int32, device=device),
                hit_flags
                if hit_flags is not None
                else torch.empty(0, dtype=torch.uint8, device=device),
                torch.empty(0, dtype=torch.int64, device=device),
            )
        if fci and pmax_work is not None and B and E:
            with _cuda_step("prefix_pmax"):
                ext.prefix_pmax(
                    r_out,
                    off,
                    first,
                    px,
                    py,
                    pmax_work,
                    int(batcher.n_samples),
                    int(k),
                    int(RANK_MAX),
                )
    else:
        with _cuda_step("fill"):
            x, y, S = fill_batch(
                nbrs_c,
                deg_c,
                px,
                py,
                off,
                k,
                rank_base=rank_base,
                rank_bases=rank_bases,
            )
        with _cuda_step("ldlt"):
            r = batcher.batch_r(x, y, S)
        with _cuda_step("hits"):
            indep = hits_from_r(r, batcher.n_samples, k, tau, fci)
        if write_g:
            with _cuda_step("delete"):
                delete_if_indep(G_work, x, y, indep)
        if union is not None or hit_flags is not None:
            assert rows is not None
            _mark_union(union, hit_flags, rows, off, indep.bool(), S if k > 0 else None)
        with _cuda_step("first_hit"):
            first = first_hit_ranks(r, off, batcher.n_samples, k, tau, fci)
        if fci and pmax_work is not None and B and E:
            pvals = np.asarray(
                _p_value_from_r(r.detach().cpu().numpy(), batcher.n_samples, k),
                dtype=np.float64,
            )
            off_np = off.detach().cpu().numpy()
            first_np = first.detach().cpu().numpy()
            px_np = px.detach().cpu().numpy()
            py_np = py.detach().cpu().numpy()
            pmax_np = (
                pmax_work.detach().cpu().numpy()
                if torch.is_tensor(pmax_work)
                else pmax_work
            )
            _prefix_pmax(pvals, px_np, py_np, off_np, first_np, pmax_np)
            if k == 0:
                _prefix_pmax(pvals, py_np, px_np, off_np, first_np, pmax_np)
            if torch.is_tensor(pmax_work):
                pmax_work.copy_(torch.as_tensor(pmax_np, device=pmax_work.device))
    hit = first < RANK_MAX
    with _cuda_step("hit_mask"):
        any_hit = bool(hit.any())
    if csr is not None and union is None:
        if any_hit:
            with _cuda_step("claim"):
                csr.claim(px, py, first, nbrs_c, deg_c, k, rank_base, rank_bases)
        return B
    if any_hit:
        e_idx = hit.nonzero(as_tuple=False).reshape(-1)
        hx = px.index_select(0, e_idx)
        hy = py.index_select(0, e_idx)
        local = first.index_select(0, e_idx)
        if rank_bases is not None:
            ranks = local + rank_bases.index_select(0, e_idx)
        else:
            ranks = local + int(rank_base)
        if k <= 0:
            hS = torch.zeros((int(e_idx.numel()), 0), dtype=torch.int32, device=device)
        else:
            with _cuda_step("hit_S"):
                _, _, hS = fill_at_ranks(nbrs_c, deg_c, hx, hy, ranks, k)
        hit_parts.append((hx, hy, hS))
    return B


def _ci_stream_union(
    batcher,
    G_work: torch.Tensor,
    nbrs_c,
    deg_c,
    px: torch.Tensor,
    py: torch.Tensor,
    n_this: torch.Tensor,
    k: int,
    rank_base: int,
    tau: float,
    alpha: float,
    pmax_host,
    union_global: "UnionAccum",
    depth_hit_pairs: list,
) -> int:
    """One rank stripe of FCI-stable tests; collect ALL hits (union sepsets).

    Sequential FAS semantics: every conditioning set of every still-present
    directed pair is tested (no early exit after a hit), each independent
    test's ``S`` is unioned into that directed pair's sepset (both triangles
    are merged at pack time), and ``pMax`` takes the max over ALL tests.
    For Fisher-Z, convert partial correlations to p-values on the host
    using ``_p_value_from_r``. For chi-square, read the batcher's p-values.
    FCI counts a finite p-value at or above alpha as independent.

    The union itself is recorded with vectorized NumPy appends
    (:class:`UnionAccum`); the previous per-test Python ``dict`` loop was the
    dominant FCI-skeleton cost at small ``p``.
    """
    keep = n_this > 0
    if int(px.numel()) == 0 or not bool(keep.any()):
        return 0
    if not bool(keep.all()):
        idx = keep.nonzero(as_tuple=False).reshape(-1)
        px = px.index_select(0, idx)
        py = py.index_select(0, idx)
        n_this = n_this.index_select(0, idx)
    off = _offsets_from_counts(n_this)
    with _cuda_step("offsets_item"):
        B = int(off[-1].item()) if off.numel() else 0
    if B == 0:
        return 0
    k = int(k)
    with _cuda_step("fill"):
        x, y, S = fill_batch(nbrs_c, deg_c, px, py, off, k, rank_base=rank_base)
    if is_host_ci(batcher):
        with _cuda_step("chisq"):
            pvals = batcher.batch_t(x, y, S).detach().cpu().numpy().astype(np.float64)
    else:
        with _cuda_step("ldlt"):
            r = batcher.batch_r(x, y, S)
        pvals = np.asarray(
            _p_value_from_r(r.detach().cpu().numpy(), batcher.n_samples, k),
            dtype=np.float64,
        )
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    fin = np.isfinite(pvals)
    if fin.any():
        np.maximum.at(pmax_host, (x_np[fin], y_np[fin]), pvals[fin])
    indep = fin & (pvals >= float(alpha))
    if indep.any():
        sel = np.flatnonzero(indep)
        S_np: np.ndarray = (
            S.detach().cpu().numpy() if k > 0 else np.zeros((len(sel), 0), np.int32)
        )
        union_global.add(x_np[sel], y_np[sel], S_np[sel] if k > 0 else S_np)
        depth_hit_pairs.append((x_np[sel], y_np[sel]))
    return B


def _run_rank_stripes(
    batcher: CudaFisherZ,
    G_work: torch.Tensor,
    nbrs_c,
    deg_c,
    px0: torch.Tensor,
    py0: torch.Tensor,
    ncomb: torch.Tensor,
    k: int,
    tau: float,
    fci: bool,
    undirected: bool,
    b_max: int,
    pmax_work,
    hit_parts: list,
    csr: SepsetCSR | None = None,
    write_g: bool = True,
    alpha: float = 0.05,
    union: torch.Tensor | None = None,
    hit_flags: torch.Tensor | None = None,
    union_global: Any = None,
    pmax_host=None,
) -> tuple[int, int]:
    """Rank-striped pair-major. Returns ``(n_tests, n_chunks)``.

    Host loop: keep live pairs, take the next ``W = max(1, ALL_S_CHUNK / E)``
    combinadic ranks, call ``_ci_stream``. Integer pair bookkeeping; CI is fused.
    PC union mode (``union`` / ``hit_flags`` sized to ``px0``): every
    independent test marks the pair's separating-member union row; pairs are
    NOT dropped after their first separator (the CPU reference tests every S
    of the depth), and the depth-end deletion comes from the hit flags.

    ``union_global`` (FCI): union-of-all-hits sepsets. No early exit after a
    hit, no CSR claim; per-depth hit pairs delete ``G`` after the depth.
    """
    n_tests = 0
    n_chunks = 0
    E0 = int(px0.numel())
    if E0 == 0:
        return 0, 0
    k = int(k)
    rank_base = 0
    px_i = px0.to(torch.int64)
    py_i = py0.to(torch.int64)
    depth_hit_pairs: list = []
    with _cuda_step("max_ncomb"):
        max_n = int(ncomb.max().item()) if int(ncomb.numel()) else 0
    while rank_base < max_n:
        with _cuda_step("present"):
            present = edge_present(G_work, px0, py0)
        remain = (ncomb - rank_base).clamp(min=0)
        keep = present.to(torch.bool) & (remain > 0)
        if union is None and union_global is None and csr is not None and not write_g:
            keep = keep & (csr.ranges[px_i, py_i, 0] < 0)
        with _cuda_step("n_live"):
            n_live = int(keep.sum().item())
        if n_live == 0:
            break
        with _cuda_step("select_live"):
            idx = keep.nonzero(as_tuple=False).reshape(-1)
            px = px0.index_select(0, idx)
            py = py0.index_select(0, idx)
        W = max(1, int(b_max) // n_live)
        r1 = rank_base + W
        n_chunks += 1
        if union_global is not None:
            n_this = remain.index_select(0, idx).clamp(max=W)
            n_tests += _ci_stream_union(
                batcher,
                G_work,
                nbrs_c,
                deg_c,
                px,
                py,
                n_this,
                k,
                rank_base,
                tau,
                alpha,
                pmax_host,
                union_global,
                depth_hit_pairs,
            )
        elif undirected and k >= 1:
            with _cuda_step("count_ncomb"):
                c_x = count_tests(nbrs_c, deg_c, px, py, k)
                c_y = count_tests(nbrs_c, deg_c, py, px, k)
                n_x = (c_x.clamp(max=r1) - rank_base).clamp(min=0)
                n_y = ((c_x + c_y).clamp(max=r1) - c_x.clamp(min=rank_base)).clamp(
                    min=0
                )
                rb_y = c_x.clamp(min=rank_base) - c_x
            n_tests += _ci_stream(
                batcher,
                G_work,
                nbrs_c,
                deg_c,
                px,
                py,
                n_x,
                k,
                rank_base,
                None,
                tau,
                fci,
                pmax_work,
                hit_parts,
                csr,
                write_g,
                alpha,
                rows=idx,
                union=union,
                hit_flags=hit_flags,
            )
            n_tests += _ci_stream(
                batcher,
                G_work,
                nbrs_c,
                deg_c,
                py,
                px,
                n_y,
                k,
                0,
                rb_y,
                tau,
                fci,
                pmax_work,
                hit_parts,
                csr,
                write_g,
                alpha,
                rows=idx,
                union=union,
                hit_flags=hit_flags,
            )
        else:
            n_this = remain.index_select(0, idx).clamp(max=W)
            n_tests += _ci_stream(
                batcher,
                G_work,
                nbrs_c,
                deg_c,
                px,
                py,
                n_this,
                k,
                rank_base,
                None,
                tau,
                fci,
                pmax_work,
                hit_parts,
                csr,
                write_g,
                alpha,
                rows=idx,
                union=union,
                hit_flags=hit_flags,
            )
        rank_base += W
        if _PROF is not None:
            _PROF.n_waves += 1
    if not write_g:
        if union is not None and hit_flags is not None:
            with _cuda_step("hit_rows"):
                hit_rows = hit_flags.nonzero(as_tuple=False).reshape(-1)
            if int(hit_rows.numel()):
                with _cuda_step("apply_hits"):
                    apply_hits(
                        G_work,
                        px0.index_select(0, hit_rows),
                        py0.index_select(0, hit_rows),
                    )
        elif union_global is not None:
            if depth_hit_pairs:
                hx = torch.as_tensor(
                    np.concatenate([p[0] for p in depth_hit_pairs]),
                    dtype=torch.int64,
                    device=G_work.device,
                )
                hy = torch.as_tensor(
                    np.concatenate([p[1] for p in depth_hit_pairs]),
                    dtype=torch.int64,
                    device=G_work.device,
                )
                G_work[hx, hy] = 0
                G_work[hy, hx] = 0
        elif csr is not None:
            claimed = csr.ranges[:, :, 0] >= 0
            G_work.masked_fill_(claimed | claimed.transpose(0, 1), 0)
        elif hit_parts:
            for hx, hy, _hS in hit_parts:
                G_work[hx.to(torch.int64), hy.to(torch.int64)] = 0
                G_work[hy.to(torch.int64), hx.to(torch.int64)] = 0
    return n_tests, n_chunks


class UnionAccum:
    """Vectorized union-of-separating-sets accumulator (FCI skeleton FAS).

    Sequential FAS stores, for every directed pair ``(x, y)``, the union of the
    members of EVERY separating conditioning set at EVERY depth.  The obvious
    implementation (``dict[(x, y)] -> set`` updated once per independent test)
    dominated the FCI skeleton wall time at small ``p``.  Instead the ``(x, y)``
    keys and the flattened ``(x, y, member)`` triples of the independent tests
    are appended as NumPy arrays by the per-stripe CI kernels and reduced once,
    vectorized, in :func:`_pack_union_csr`.
    """

    __slots__ = ("member_keys", "p", "pair_keys")

    def __init__(self, p: int):
        self.p = int(p)
        self.pair_keys: list[np.ndarray] = []
        self.member_keys: list[np.ndarray] = []

    def add(self, x_np: np.ndarray, y_np: np.ndarray, S_np: np.ndarray) -> None:
        """Record the independent tests ``(x_np[t], y_np[t], S_np[t])``."""
        if x_np.size == 0:
            return
        p = self.p
        keys = x_np.astype(np.int64) * p + y_np.astype(np.int64)
        self.pair_keys.append(keys)
        if S_np.size:
            mem = S_np.astype(np.int64).reshape(keys.size, -1)
            self.member_keys.append(np.repeat(keys, mem.shape[1]) * p + mem.reshape(-1))


def _pack_union_csr(union_global, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Union sepsets -> CSR. Both triangles hold the union of both directions.

    ``union_global`` is a :class:`UnionAccum`. Sequential FAS stores the union
    of every separating set of the pair in BOTH keys ``(x, y)`` and ``(y, x)``,
    so the two directed unions are merged here and mirrored. Members are
    sorted; empty unions (k=0 hits) are stored as ``(0, 0)`` slices.
    """
    ranges = np.full((p, p, 2), -1, dtype=np.int64)
    if isinstance(union_global, dict):  # legacy dict-of-sets still accepted
        pair_keys: list[np.ndarray] = []
        member_keys: list[np.ndarray] = []
        for (x, y), members in union_global.items():
            keys = np.asarray([int(x) * p + int(y)], dtype=np.int64)
            pair_keys.append(keys)
            if members:
                mem = np.asarray(sorted(int(v) for v in members), dtype=np.int64)
                member_keys.append(keys.repeat(mem.size) * p + mem)
    else:
        pair_keys = union_global.pair_keys
        member_keys = union_global.member_keys

    if pair_keys:
        pk = np.concatenate(pair_keys)
        x = pk // p
        y = pk % p
        und = np.unique(np.minimum(x, y) * p + np.maximum(x, y))
    else:
        und = np.zeros(0, dtype=np.int64)

    if member_keys:
        mk = np.concatenate(member_keys)
        mm = mk % p
        pdir = mk // p
        x = pdir // p
        y = pdir % p
        u2 = np.unique((np.minimum(x, y) * p + np.maximum(x, y)) * p + mm)
        pu = u2 // p
        members = (u2 % p).astype(np.int64)
        upairs, starts, counts = np.unique(pu, return_index=True, return_counts=True)
        S = members
    else:
        upairs = np.zeros(0, dtype=np.int64)
        starts = np.zeros(0, dtype=np.int64)
        counts = np.zeros(0, dtype=np.int64)
        S = np.zeros(0, dtype=np.int64)

    if und.size:
        x = und // p
        y = und % p
        if upairs.size:
            pos = np.searchsorted(upairs, und)
            pos_c = np.clip(pos, 0, upairs.size - 1)
            has = upairs[pos_c] == und
            s = np.where(has, starts[pos_c], 0)
            c = np.where(has, counts[pos_c], 0)
        else:
            s = np.zeros(und.size, dtype=np.int64)
            c = np.zeros(und.size, dtype=np.int64)
        ranges[x, y, 0] = s
        ranges[x, y, 1] = s + c
        ranges[y, x, 0] = ranges[x, y, 0]
        ranges[y, x, 1] = ranges[x, y, 1]
    return S, ranges


def _pc_union_fused_ok(batcher, k: int) -> bool:
    """Check whether a PC-stable union depth can run in a single launch.

    Requires the fused LDLT kernel (no host χ²) and ``k`` inside its range.
    This path tests every ``S`` of every live directed pair in a SINGLE fused
    launch: PC-stable union mode never drops a pair mid-depth (``G`` is frozen
    until the depth ends) and never early-exits, so the rank-stripe
    decomposition only existed to bound the *materialized* ``(x, y, S)``
    buffers of the unfused path. The fused kernel needs no such buffers.
    """
    if not USE_FUSED or k < 0:
        return False
    if is_host_ci(batcher):
        return False
    try:
        if k > int(_graph_batch_cuda().fused_k_max()):
            return False
    except Exception:  # Fall back if the fused kernel is unavailable.
        return False
    return True


def _run_pc_union_depth(
    batcher: CudaFisherZ,
    G_work: torch.Tensor,
    nbrs: torch.Tensor,
    deg: torch.Tensor,
    k: int,
    tau: float,
    alpha: float,
    csr: SepsetCSR,
) -> int:
    """One PC-stable union depth. Returns the test count.

    Prefers the C++ driver ``pc_depth_union`` (live pairs, prefix sum, fused CI,
    union commit, G update — one call, two syncs) and falls back to the
    equivalent Python sequence on extensions built before it existed.
    """
    ext = _graph_batch_cuda()
    if hasattr(ext, "pc_depth_union"):
        return _run_pc_union_depth_cpp(batcher, G_work, nbrs, deg, k, tau, csr)
    return _run_pc_union_depth_py(batcher, G_work, nbrs, deg, k, tau, alpha, csr)


def _run_pc_union_depth_cpp(
    batcher: CudaFisherZ,
    G_work: torch.Tensor,
    nbrs: torch.Tensor,
    deg: torch.Tensor,
    k: int,
    tau: float,
    csr: SepsetCSR,
) -> int:
    """One depth via the C++ ``pc_depth_union`` driver. Returns test count.

    ``csr.S`` is grown to a per-depth upper bound before the call because the
    kernel reserves sepset slots with device-side atomics and cannot reallocate.
    """
    p = int(G_work.size(0))
    max_d = int(nbrs.size(1)) if nbrs.dim() == 2 else 0
    if p == 0 or max_d == 0:
        return 0
    # Per-depth append upper bound: at most one union record per live directed
    # pair, and each record merges the two directed enumerations of size ``k``
    # (at most ``2k`` distinct members, capped by ``p``).
    per_record = min(p, 2 * int(k))
    bound = csr._nnz_host + p * max_d * per_record
    csr._ensure_capacity(bound, "sepset CSR buffer (PC depth)")
    ext = _graph_batch_cuda()
    with _cuda_step("pc_depth"):
        vals = ext.pc_depth_union(
            batcher.corr,
            G_work,
            nbrs,
            deg,
            csr.ranges,
            csr.S,
            csr.nnz,
            float(tau),
            float(batcher.eps),
            int(batcher.n_samples),
            int(k),
        )
    vals = vals.tolist()
    csr._nnz_host = int(vals[1])
    return int(vals[0])


def _run_pc_union_depth_py(
    batcher: CudaFisherZ,
    G_work: torch.Tensor,
    nbrs: torch.Tensor,
    deg: torch.Tensor,
    k: int,
    tau: float,
    alpha: float,
    csr: SepsetCSR,
) -> int:
    r"""One PC-stable depth in a single fused launch. Returns the test count.

    Replaces the rank-striped driver for the fused path: all live directed
    pairs, all ``combinations(nbrs(x)\{y}, k)`` at this depth, one
    ``ci_from_comb`` launch with per-pair offsets. `hit_flags` / the member
    union bitmap are marked inside the kernel; ``commit_union`` turns them into
    CSR records; ``G`` is zeroed for the hit rows afterwards. Pairs live in a
    ``p * max_deg`` upper-bound buffer so no pair count has to come back from
    the device before the launch.
    """
    device = G_work.device
    ext = _graph_batch_cuda()
    p = int(G_work.size(0))
    max_d = int(nbrs.size(1)) if nbrs.dim() == 2 else 0
    if p == 0 or max_d == 0:
        return 0
    cap = p * max_d
    bufs = getattr(csr, "_pc_bufs", None)
    if bufs is None or bufs[0] != cap or bufs[1] != k:
        px0 = torch.empty(cap, dtype=torch.int32, device=device)
        py0 = torch.empty(cap, dtype=torch.int32, device=device)
        ncomb = torch.zeros(cap, dtype=torch.int64, device=device)
        first = torch.full((cap,), RANK_MAX, dtype=torch.int64, device=device)
        E_out = torch.zeros(1, dtype=torch.int32, device=device)
        bufs = (cap, k, px0, py0, ncomb, first, E_out)
        csr._pc_bufs = bufs
    _, _, px0, py0, ncomb, first, E_out = bufs
    ncomb.zero_()
    E_out.zero_()
    with _cuda_step("live_pairs"):
        ext.pc_live_pairs(nbrs, deg, int(k), px0, py0, ncomb, E_out)
    off = _offsets_from_counts(ncomb)
    with _cuda_step("offsets_item"):
        B = int(off[-1].item())
    if B == 0:
        return 0
    empty_64 = torch.empty(0, dtype=torch.int64, device=device)
    empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
    empty_u8 = torch.empty(0, dtype=torch.uint8, device=device)
    empty_f = torch.empty(0, dtype=batcher.corr.dtype, device=device)
    if k == 0:
        first.fill_(RANK_MAX)
        with _cuda_step("fused"):
            ext.ci_from_comb(
                batcher.corr,
                G_work,
                nbrs,
                deg,
                px0,
                py0,
                off,
                0,
                0,
                empty_64,
                first,
                B,
                int(batcher.n_samples),
                float(tau),
                float(batcher.eps),
                False,
                False,
                empty_f,
                empty_u8,
                empty_u8,
                empty_i32,
                empty_u8,
                torch.empty(0, dtype=torch.int64, device=device),
            )
        with _cuda_step("claim"):
            csr.claim(px0, py0, first, nbrs, deg, 0, 0, None)
        with _cuda_step("apply_hits"):
            claimed = csr.ranges[:, :, 0] >= 0
            G_work.masked_fill_(claimed | claimed.transpose(0, 1), 0)
        return B
    # k >= 1: ``first`` is written by the fused kernel but never read here, so
    # it needs no initialization. The member bitmap and hit flags share one
    # zeroed byte buffer (one memset instead of two).
    ubuf = getattr(csr, "_pc_union_buf", None)
    if ubuf is None or ubuf[0] != cap or ubuf[1] != p:
        ubytes = torch.zeros(cap * p + cap, dtype=torch.uint8, device=device)
        union = ubytes[: cap * p].view(cap, p)
        hit_flags = ubytes[cap * p :]
        csr._pc_union_buf = (cap, p, ubytes, union, hit_flags)
    else:
        ubytes, union, hit_flags = ubuf[2], ubuf[3], ubuf[4]
        ubytes.zero_()
    with _cuda_step("fused"):
        ext.ci_from_comb(
            batcher.corr,
            G_work,
            nbrs,
            deg,
            px0,
            py0,
            off,
            int(k),
            0,
            empty_64,
            first,
            B,
            int(batcher.n_samples),
            float(tau),
            float(batcher.eps),
            False,
            False,
            empty_f,
            empty_u8,
            union,
            empty_i32,
            hit_flags,
            torch.empty(0, dtype=torch.int64, device=device),
        )
    with _cuda_step("commit_union"):
        csr.commit_union_fused(px0, py0, union, hit_flags)
    with _cuda_step("apply_hits"):
        ext.zero_flagged(G_work, px0, py0, hit_flags)
    return B


@overload
def run_depth(
    batcher: Any,
    G_pairs,
    G_cand,
    k: int,
    alpha: float,
    *,
    fci: bool = False,
    pmax: torch.Tensor | None = None,
    b_max: int | None = None,
    apply_to: torch.Tensor | None = None,
    to_host: Literal[False],
    state: GraphBatchState | None = None,
    nbrs=None,
    deg=None,
    csr: SepsetCSR | None = None,
    union_global: Any = None,
    pmax_host=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]: ...


@overload
def run_depth(
    batcher: Any,
    G_pairs,
    G_cand,
    k: int,
    alpha: float,
    *,
    fci: bool = False,
    pmax: np.ndarray | torch.Tensor | None = None,
    b_max: int | None = None,
    apply_to: torch.Tensor | None = None,
    to_host: bool = True,
    state: GraphBatchState | None = None,
    nbrs=None,
    deg=None,
    csr: SepsetCSR | None = None,
    union_global: Any = None,
    pmax_host=None,
) -> tuple[
    torch.Tensor | np.ndarray,
    torch.Tensor | np.ndarray,
    torch.Tensor | np.ndarray,
    torch.Tensor | np.ndarray | None,
]: ...


def run_depth(
    batcher: Any,
    G_pairs,
    G_cand,
    k: int,
    alpha: float,
    *,
    fci: bool = False,
    pmax: np.ndarray | torch.Tensor | None = None,
    b_max: int | None = None,
    apply_to: torch.Tensor | None = None,
    to_host: bool = True,
    state: GraphBatchState | None = None,
    nbrs=None,
    deg=None,
    csr: SepsetCSR | None = None,
    union_global: Any = None,
    pmax_host=None,
):
    r"""All combinadic ``S`` at frozen depth ``k``.

    Candidate lists are taken once at the start of the depth. PC-stable tests
    each directed ``(x, y)`` with ``combinations(nbrs(x)\{y}, k)``, records the
    first hit in ``ranges[x, y]``, and zeros ``G`` only after the depth.
    ``union_global`` (FCI): no early exit, the union of every separating set
    is accumulated per directed pair; per-depth hit pairs zero ``G`` after
    the depth. ``pmax_host`` is the host pMax array used in union mode.
    """
    del state
    if b_max is None:
        b_max = ALL_S_CHUNK
    device = batcher.device
    k = int(k)
    tau = float(norm.ppf(1.0 - alpha / 2.0))
    G_t = _as_adj(G_pairs, device)
    G_c = None if G_cand is None else _as_adj(G_cand, device)
    G_work = apply_to if apply_to is not None else G_t
    pmax_work = _pmax_device(pmax, device) if fci else None
    n_tests = 0
    n_chunks = 0
    hit_parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    use_s_major = (
        (not fci)
        and k >= int(S_MAJOR_MIN_K)
        and USE_S_MAJOR
        and not is_host_ci(batcher)
    )
    if use_s_major:
        src = G_c if G_c is not None else G_t
        if nbrs is None:
            nbrs_c, deg_c = compact_adj(src, device)
        else:
            assert deg is not None
            nbrs_c, deg_c = nbrs, deg
        with _cuda_step("unique_S"):
            unique_S = unique_S_lex(nbrs_c, deg_c, k)
        U_all = int(unique_S.size(0))
        if U_all:
            with _cuda_step("count_S"):
                counts0 = torch.empty(U_all, dtype=torch.int64, device=device)
                _graph_batch_cuda().count_s_tests(
                    G_work, nbrs_c, deg_c, unique_S.contiguous(), 1, counts0
                )
            counts_np = counts0.detach().cpu().numpy()
            for u0, u1 in _iter_s_chunks(counts_np, int(b_max)):
                n_chunks += 1
                uS = unique_S[u0:u1].contiguous()
                with _cuda_step("fill"):
                    x, y, S, counts, _off = fill_s_major(
                        G_work, nbrs_c, deg_c, uS, undirected=True
                    )
                B = int(x.numel())
                if B == 0:
                    continue
                n_tests += B
                with _cuda_step("ldlt"):
                    if k >= 2:
                        inv = torch.repeat_interleave(
                            torch.arange(
                                int(uS.size(0)), device=device, dtype=torch.int32
                            ),
                            counts.to(torch.int64),
                        )
                        r = batcher.batch_r(x, y, S, unique_S=uS, inverse=inv)
                    else:
                        r = batcher.batch_r(x, y, S)
                with _cuda_step("hits"):
                    indep = hits_from_r(r, batcher.n_samples, k, tau, fci)
                with _cuda_step("delete"):
                    delete_if_indep(G_work, x, y, indep)
                if bool(indep.any()):
                    e_idx = indep.nonzero(as_tuple=False).reshape(-1)
                    hit_parts.append(
                        (
                            x.index_select(0, e_idx),
                            y.index_select(0, e_idx),
                            S.index_select(0, e_idx),
                        )
                    )
                if _PROF is not None:
                    _PROF.n_waves += 1
    else:
        if (
            (not fci)
            and csr is not None
            and nbrs is not None
            and deg is not None
            and _pc_union_fused_ok(batcher, k)
        ):
            # Single fused launch covers every test of the depth: PC-stable
            # union mode never drops a pair mid-depth and never early-exits, so
            # rank striping bought nothing but host round-trips here.
            n_tests = _run_pc_union_depth(
                batcher,
                G_work,
                nbrs,
                deg,
                k,
                tau,
                alpha,
                csr,
            )
            n_chunks = 1
            depth_stats["n_tests"] = n_tests
            depth_stats["n_chunks"] = n_chunks
            vars(run_depth)["last_n_tests"] = n_tests
            pmax_out = _pmax_result(pmax, pmax_work, to_host=to_host)
            hx, hy, hS = _empty_hits(k, device, to_host)
            return hx, hy, hS, pmax_out
        undirected = False
        write_g = False
        with _cuda_step("live_pairs"):
            nbrs_c, deg_c, px0, py0, ncomb = live_pairs(
                G_t,
                G_c,
                k,
                device=device,
                nbrs=nbrs,
                deg=deg,
                undirected=undirected,
                k0_unique=bool(fci),
            )
        # PC-stable union mode fallback (non-fused / k past the fused kernel):
        # one separating-member union per pair; k = 0 needs no member marks.
        union = None
        hit_flags = None
        if not fci:
            E0 = int(px0.numel())
            if E0:
                hit_flags = torch.zeros(E0, dtype=torch.uint8, device=device)
                if k >= 1:
                    union = torch.zeros(
                        (E0, int(G_work.size(0))), dtype=torch.uint8, device=device
                    )
        n_tests, n_chunks = _run_rank_stripes(
            batcher,
            G_work,
            nbrs_c,
            deg_c,
            px0,
            py0,
            ncomb,
            k,
            tau,
            fci,
            undirected,
            int(b_max),
            pmax_work,
            hit_parts,
            csr,
            write_g,
            alpha,
            union=union,
            hit_flags=hit_flags,
            union_global=union_global,
            pmax_host=pmax_host,
        )
        if not fci and csr is not None and hit_flags is not None:
            with _cuda_step("commit_union"):
                csr.commit_union(px0, py0, union, hit_flags)

    depth_stats["n_tests"] = n_tests
    depth_stats["n_chunks"] = n_chunks
    vars(run_depth)["last_n_tests"] = n_tests
    pmax_out = _pmax_result(pmax, pmax_work, to_host=to_host)
    if csr is not None and not use_s_major:
        hx, hy, hS = _empty_hits(k, device, to_host)
        return hx, hy, hS, pmax_out
    with _cuda_step("dedup_hits"):
        hx, hy, hS = _dedup_hit_parts(hit_parts, k, device, undirected=False)
    if to_host:
        if int(hx.numel()) == 0:
            ex, ey, eS = _empty_hits(k, device, True)
            return ex, ey, eS, pmax_out
        return (
            hx.detach().cpu().numpy(),
            hy.detach().cpu().numpy(),
            hS.detach().cpu().numpy(),
            pmax_out,
        )
    return hx, hy, hS, pmax_out


def _run_cpp_skeleton(
    batcher: CudaFisherZ,
    G_t: torch.Tensor,
    alpha: float,
    *,
    max_depth: int | None,
    k_max: int,
    to_host: bool,
    csr: SepsetCSR,
    fci: bool = False,
    pmax: torch.Tensor | None = None,
):
    """Whole union skeleton in one extension call (PC, or FCI when ``fci``).

    The depth loop, per-depth adjacency compaction, the fused CI and the
    sepset commit all run in C++ (``pc_skeleton_union``); Python only sizes the
    CSR and copies it out. Depths past the fused kernel (``k > 32``) are
    reported as ``fallback_depth`` so the caller resumes with the host
    wavefront.
    """
    ext = _graph_batch_cuda()
    tau = float(norm.ppf(1.0 - alpha / 2.0))
    k_fused = int(min(int(k_max), int(ext.fused_k_max())))
    p = int(G_t.size(0))
    if fci and pmax is None:
        pmax = torch.full((p, p), float("-inf"), dtype=torch.float64, device=G_t.device)
        pmax.diagonal().fill_(1.0)
    # The fused C++ driver cannot reallocate its CSR buffers. If the lazily
    # sized buffer overflows, restore the input graph, resize to the reported
    # demand and run the pass once more.
    G_start = G_t.clone()
    pmax_start = pmax.clone() if fci and pmax is not None else None
    attempts = 0
    while True:
        attempts += 1
        min_abs = (
            torch.full((p * p,), (1 << 63) - 1, dtype=torch.int64, device=G_t.device)
            if fci
            else torch.Tensor()
        )
        with _cuda_step("cpp_skeleton"):
            res = ext.pc_skeleton_union(
                batcher.corr,
                G_t,
                csr.ranges,
                csr.S,
                csr.nnz,
                tau,
                float(batcher.eps),
                int(batcher.n_samples),
                k_fused,
                int(max_depth if max_depth is not None else -1),
                bool(fci),
                pmax if fci else torch.Tensor(),
                min_abs,
            )
        nnz, fb, depth, n_tests, ovf = (int(v) for v in res.tolist())
        if not ovf:
            break
        G_t.copy_(G_start)
        if pmax_start is not None:
            assert pmax is not None
            pmax.copy_(pmax_start)
        csr.ranges.fill_(-1)
        csr.nnz.zero_()
        csr._nnz_host = 0
        csr._ensure_capacity(
            max(nnz, int(csr.S.numel()) * 2),
            "sepset CSR buffer (fused skeleton)",
        )
        if attempts == 4:
            msg = (
                f"pc_skeleton_union: sepset CSR overflow persisted after resizing "
                f"({ovf} reservations did not fit)"
            )
            raise RuntimeError(msg)
    depth_stats["n_tests"] = n_tests
    depth_stats["n_chunks"] = 0
    vars(run_skeleton).update(last_n_tests=n_tests, last_depth=depth)
    S_used = csr.S[:nnz].contiguous()
    hits: tuple[str, Any, Any]
    if to_host:
        hits = ("csr", S_used.detach().cpu().numpy(), csr.ranges.detach().cpu().numpy())
    else:
        hits = ("csr", S_used, csr.ranges)
    pmax_out: Any = None
    if fci:
        assert pmax is not None
        pmax_out = pmax.detach().cpu().numpy() if to_host else pmax
    return hits, pmax_out, (fb if fb >= 0 else None)


def run_skeleton(
    batcher: Any,
    G_t: torch.Tensor,
    alpha: float,
    *,
    max_depth: int | None = None,
    k_max: int = 96,
    fci: bool = False,
    G1_t: torch.Tensor | None = None,
    pmax: np.ndarray | torch.Tensor | None = None,
    to_host: bool = True,
):
    """PC/FCI skeleton on a resident ``G_t``. No H2D of ``G`` between depths.

    Outer ``k = 0, 1, …`` until ``max_degree - 1 ≤ k``. Compacts adjacency,
    runs pair-major CI (fused for ``k ≤ 32``, flat/fallback beyond), claims
    sepsets into CSR. ``to_host=True`` (default) copies CSR
    and ``pMax`` once at the end. ``to_host=False`` keeps them on device for
    a later PC / FCI phase.
    With ``fci=True``, freeze the adjacency at each depth, accept independence
    at ``p ≥ α``, and track ``pMax``. Merge all separating-set members for
    each pair and store the union in both directions. The return value
    includes the first unsupported depth so the caller can continue there.
    The Python loop also accepts the Pearson χ² batcher, ``CudaChisqHost``.
    """
    depth = -1
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    n_tests = 0
    fallback_depth = None
    p = int(G_t.size(0))
    fci_union = bool(fci)
    union_global: dict | UnionAccum = {} if not fci_union else UnionAccum(p)
    pmax_host = None
    if fci_union:
        if pmax is None:
            pmax_host = np.full((p, p), -np.inf)
            np.fill_diagonal(pmax_host, 1.0)
        elif isinstance(pmax, torch.Tensor):
            pmax_host = pmax.detach().cpu().numpy().copy()
        else:
            pmax_host = np.array(pmax, dtype=np.float64, copy=True)
        pmax_work = None
    else:
        pmax_work = _pmax_device(pmax, G_t.device) if fci else None
    csr = SepsetCSR(p, G_t.device, k_max=int(k_max))
    host_ci = is_host_ci(batcher)
    ext = _graph_batch_cuda()
    if (not host_ci) and USE_FUSED and hasattr(ext, "pc_skeleton_union"):
        return _run_cpp_skeleton(
            batcher,
            G_t,
            alpha,
            max_depth=max_depth,
            k_max=k_max,
            to_host=to_host,
            csr=csr,
            fci=fci,
            pmax=pmax if isinstance(pmax, torch.Tensor) else None,
        )
    while True:
        if _PROF is not None:
            _PROF.phase = "skeleton"
            _PROF.k = depth + 1
        with _cuda_step("compact"):
            nbrs, deg = compact_adj(G_t, G_t.device)
        with _cuda_step("max_deg"):
            max_d = int(deg.max().item()) if int(deg.numel()) else 0
        if max_d - 1 <= depth:
            break
        if max_depth is not None and depth + 1 > max_depth:
            break
        depth += 1
        if _PROF is not None:
            _PROF.k = depth
        if depth > int(k_max):
            why = "host χ² combinadic fill" if host_ci else "CUDA LDLT"
            print(
                f"[cuda skeleton] k={depth} > k_max={int(k_max)} "
                f"(k>=97 exceeds {why}); Python combinadic fallback",
                file=sys.stderr,
                flush=True,
            )
            fallback_depth = depth
            break
        G_cand = G1_t if (fci and G1_t is not None) else G_t
        hx, hy, hS, pmax_work = run_depth(
            batcher,
            G_t,
            G_cand,
            depth,
            alpha,
            fci=fci,
            pmax=pmax_work,
            apply_to=G_t,
            to_host=False,
            nbrs=nbrs,
            deg=deg,
            csr=csr,
            union_global=union_global if fci_union else None,
            pmax_host=pmax_host,
        )
        n_k = int(depth_stats["n_tests"])
        n_tests += n_k
        if fci and G1_t is not None:
            with _cuda_step("copy_G1"):
                G1_t.copy_(G_t)
        if int(hx.numel()) > 0:
            with _cuda_step("clone_hits"):
                batches.append((hx.clone(), hy.clone(), hS.clone()))
    vars(run_skeleton).update(last_n_tests=n_tests, last_depth=depth)
    if _PROF is not None:
        _PROF.phase = "epilogue"
        _PROF.k = -1
    if fci_union:
        with _cuda_step("pack_csr"):
            S_u, ranges_u = _pack_union_csr(union_global, p)
            S = torch.as_tensor(S_u, dtype=torch.int64, device=G_t.device)
            ranges = torch.as_tensor(ranges_u, dtype=torch.int64, device=G_t.device)
    elif batches:
        with _cuda_step("pack_csr"):
            S, ranges = pack_sepset_csr(
                batches, undirected=True, device=G_t.device, p=p
            )
    else:
        with _cuda_step("pack_csr"):
            S, ranges = csr.pack()
            if fci:
                ranges = fci_last_writer_mirror(ranges)
    hits: tuple[str, Any, Any]
    if to_host:
        with _cuda_step("d2h_csr"):
            hits = (
                "csr",
                S.detach().cpu().numpy(),
                ranges.detach().cpu().numpy(),
            )
    else:
        hits = ("csr", S, ranges)
    pmax_out: Any = pmax_work
    if fci_union:
        if isinstance(pmax, torch.Tensor):
            pmax.copy_(torch.as_tensor(pmax_host, device=pmax.device))
            pmax_out = pmax_host if to_host else pmax
        else:
            pmax_out = pmax_host
    elif fci and pmax_work is not None and to_host:
        with _cuda_step("d2h_pmax"):
            pmax_out = pmax_work.detach().cpu().numpy()
    return hits, pmax_out, fallback_depth
