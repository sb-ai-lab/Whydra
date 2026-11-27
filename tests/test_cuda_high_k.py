"""PC/FCI CUDA paths at k up to 96 (96×96 LDLᵀ).

Complete ``p = k+2`` makes ``C(k,k)=1`` per directed pair, so combinadic fill
and CI stay cheap. Fused CI is only ``k≤32``; larger ``k`` uses fill +
``CudaFisherZ.batch_r``. Gold is NumPy float64 LDLᵀ on the same ``C``.
"""

import sys
from pathlib import Path

# Script-style CUDA suite: run `python tests/test_cuda_high_k.py`. Its k sweep
# depends on the device smem cap and it uses a Report-style accumulator, so it
# is not a pytest module.
__test__ = False

import numpy as np
import torch
from scipy.stats import norm

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whydra.algorithms.graph_core.cuda_fisher_z import CudaFisherZ, _ldlt_cuda
from whydra.algorithms.graph_core.cuda_graph_batch import (
    SepsetCSR,
    _graph_batch_cuda,
    _offsets_from_counts,
    compact_adj,
    count_tests,
    emit_pairs,
    fill_batch,
    hits_from_r,
    run_depth,
)
from whydra.algorithms.graph_core.cuda_pdsep import _pdsep_rank_stripes
from whydra.algorithms.graph_core.fisher_z_schur_chol import (
    NumpyFisherZ,
    partial_corr_r_batch_schur,
)

K_MAX = 96
FUSED_K_MAX = 32
N_SAMPLES = 800
ALPHA = 0.05
# k=96 needs (k²+3k)·8+4k ≈ 72 KB of dynamic smem: fits datacenter GPUs
# (A40, 99 KB opt-in) but not consumer parts (RTX 2080 Ti, 64 KB opt-in),
# where the runtime k_max cap is 88 and larger k uses the host fallback.
HIGH_K = (32, 33, 48, 64, 96)


def _gaussian(n: int, p: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = np.tril(rng.normal(size=(p, p)), k=-1) * 0.3
    noise = rng.normal(size=(n, p))
    data = noise.copy()
    for j in range(p):
        data[:, j] = data[:, :j] @ w[j, :j] + noise[:, j]
    return data


def _complete(p: int) -> np.ndarray:
    G = np.ones((p, p), dtype=np.int8)
    np.fill_diagonal(G, 0)
    return G


def _complement_S(p: int, x: int, y: int) -> tuple[int, ...]:
    return tuple(i for i in range(p) if i != x and i != y)


def _tau(alpha: float = ALPHA) -> float:
    return float(norm.ppf(1.0 - alpha / 2.0))


def test_caps() -> None:
    fz = _ldlt_cuda()
    gb = _graph_batch_cuda()
    # k_max is the device-fit cap: K_MAX=96 on >=99KB-optin GPUs (A40),
    # 88 on 64KB consumer parts (RTX 2080 Ti, sm_75).
    cap = int(torch.cuda.get_device_properties(0).shared_memory_per_block_optin)
    k = K_MAX
    while k > 4 and (k * k + 3 * k) * 8 + 4 * k > cap:
        k -= 1
    assert int(fz.k_max()) == k, fz.k_max()
    assert int(gb.k_max()) == K_MAX, gb.k_max()
    assert int(gb.fused_k_max()) == FUSED_K_MAX, gb.fused_k_max()
    print(
        f"caps: fisher_z={fz.k_max()} combinadic={gb.k_max()} fused={gb.fused_k_max()}"
    )


def test_count_fill(k: int) -> None:
    p = k + 2
    G = _complete(p)
    nbrs, deg = compact_adj(G)
    px, py = emit_pairs(nbrs, deg)
    counts = count_tests(nbrs, deg, px, py, k).cpu().numpy()
    assert np.all(counts == 1), (k, counts.min(), counts.max())
    off = _offsets_from_counts(torch.as_tensor(counts, device=px.device))
    x, y, S = fill_batch(nbrs, deg, px, py, off, k)
    xs, ys, Ss = x.cpu().numpy(), y.cpu().numpy(), S.cpu().numpy()
    assert xs.size == p * (p - 1)
    for i in range(min(int(xs.size), 8)):
        expect = _complement_S(p, int(xs[i]), int(ys[i]))
        assert tuple(int(v) for v in Ss[i]) == expect, (k, xs[i], ys[i], Ss[i], expect)
    i_last = int(xs.size) - 1
    expect = _complement_S(p, int(xs[i_last]), int(ys[i_last]))
    assert tuple(int(v) for v in Ss[i_last]) == expect
    print(f"count+fill k={k} p={p} B={xs.size}")


def _gpu_cpu(p: int, seed: int):
    data = _gaussian(N_SAMPLES, p, seed)
    gpu = CudaFisherZ.from_data(data, N_SAMPLES)
    cpu = NumpyFisherZ(gpu.host_corr(), N_SAMPLES, method="schur")
    return gpu, cpu


def _atol_r(k: int) -> float:
    return 1e-12 + 1e-13 * max(k, 1) ** 3


def test_run_depth(k: int, *, fci: bool) -> None:
    p = k + 2
    gpu, cpu = _gpu_cpu(p, seed=1000 + k + 17 * int(fci))
    G = _complete(p)
    nbrs, deg = compact_adj(G)
    px, py = emit_pairs(nbrs, deg)
    counts = count_tests(nbrs, deg, px, py, k)
    off = _offsets_from_counts(counts)
    x, y, S = fill_batch(nbrs, deg, px, py, off, k)
    r_t = gpu.batch_r(x, y, S)
    r_gpu = r_t.detach().cpu().numpy()
    r_cpu = partial_corr_r_batch_schur(
        cpu.corr, x.cpu().numpy(), y.cpu().numpy(), S.cpu().numpy()
    )
    dr = float(np.nanmax(np.abs(r_gpu - r_cpu)))
    assert dr <= _atol_r(k), (k, fci, dr)
    indep = hits_from_r(r_t, gpu.n_samples, k, _tau(), fci)
    expect = {
        (int(x[i]), int(y[i])): tuple(int(v) for v in S[i].cpu().numpy())
        for i in torch.nonzero(indep, as_tuple=False).reshape(-1).cpu().tolist()
    }
    rx, ry, S_hit, _ = run_depth(gpu, G, G, k, ALPHA, fci=fci)
    got = {
        (int(rx[i]), int(ry[i])): tuple(int(v) for v in S_hit[i])
        for i in range(len(rx))
    }
    assert set(got) == set(expect), (
        f"fci={fci} k={k} only_gpu={sorted(set(got) - set(expect))[:6]} "
        f"only_fill={sorted(set(expect) - set(got))[:6]}"
    )
    for key, S_g in got.items():
        assert S_g == expect[key], (k, key, S_g, expect[key])
    tag = "FCI" if fci else "PC"
    print(
        f"run_depth {tag} k={k} p={p} hits={len(got)}/{int(x.numel())} max_|r|={dr:.3e}"
    )


def test_fused_vs_unfused_k32() -> None:
    from whydra.algorithms.graph_core import cuda_graph_batch as gb

    k = 32
    p = k + 2
    gpu, _ = _gpu_cpu(p, seed=32)
    G = _complete(p)
    old = gb.USE_FUSED
    try:
        gb.USE_FUSED = True
        Ga = G.copy()
        rx_f, ry_f, _Sf, _ = run_depth(gpu, Ga, Ga, k, ALPHA, fci=False)
        gb.USE_FUSED = False
        Gb = G.copy()
        rx_u, ry_u, _Su, _ = run_depth(gpu, Gb, Gb, k, ALPHA, fci=False)
    finally:
        gb.USE_FUSED = old
    fused = {(int(rx_f[i]), int(ry_f[i])) for i in range(len(rx_f))}
    unf = {(int(rx_u[i]), int(ry_u[i])) for i in range(len(rx_u))}
    assert fused == unf, sorted(fused ^ unf)[:8]
    print(f"fused vs fill+batch_r k=32 hits={len(fused)}")


def test_pdsep_stripe(k: int, *, fused: bool) -> None:
    p = k + 2
    gpu, cpu = _gpu_cpu(p, seed=2000 + k)
    dev = gpu.device
    amat = torch.zeros((p, p), dtype=torch.int8, device=dev)
    amat[0, 1] = amat[1, 0] = 1
    pd_adj = torch.zeros((p, p), dtype=torch.int8, device=dev)
    pd_adj[0, 1:] = 1
    nbrs_pd, deg_pd = compact_adj(pd_adj, device=dev)
    px0 = torch.tensor([0], dtype=torch.int32, device=dev)
    py0 = torch.tensor([1], dtype=torch.int32, device=dev)
    ncomb = count_tests(nbrs_pd, deg_pd, px0, py0, k)
    assert int(ncomb[0].item()) == 1, int(ncomb[0].item())
    pmax_t = torch.full((p, p), -np.inf, dtype=torch.float64, device=dev)
    csr = SepsetCSR(p, dev, k_max=K_MAX)
    adj_frozen = (amat[0] != 0).to(dtype=torch.int8).contiguous()
    ext = _graph_batch_cuda()
    _pdsep_rank_stripes(
        batcher=gpu,
        amat=amat,
        nbrs_pd=nbrs_pd,
        deg_pd=deg_pd,
        px0=px0,
        py0=py0,
        ncomb=ncomb,
        k=k,
        tau=_tau(),
        adj_frozen=adj_frozen,
        pmax_t=pmax_t,
        csr=csr,
        fused=fused,
        ext=ext,
        rb_empty=torch.empty(0, dtype=torch.int64, device=dev),
        write_g=False,
    )
    S_expect = np.arange(2, p, dtype=np.int64)
    r_cpu = float(
        partial_corr_r_batch_schur(
            cpu.corr, np.array([0]), np.array([1]), S_expect.reshape(1, -1)
        )[0]
    )
    r_gpu = float(
        gpu.batch_r(np.array([0]), np.array([1]), S_expect.reshape(1, -1)).cpu()
    )
    assert abs(r_gpu - r_cpu) <= _atol_r(k), (k, r_gpu, r_cpu)
    indep = bool(
        hits_from_r(
            torch.tensor([r_gpu], dtype=gpu.dtype, device=gpu.device),
            gpu.n_samples,
            k,
            _tau(),
            True,
        )[0]
    )
    start = int(csr.ranges[0, 1, 0].cpu())
    end = int(csr.ranges[0, 1, 1].cpu())
    claimed = start >= 0
    assert claimed == indep, (k, fused, claimed, indep, r_gpu)
    if claimed:
        got = csr.S[start:end].detach().cpu().numpy()
        assert np.array_equal(got, S_expect), (got, S_expect)
    print(f"pdsep k={k} fused={fused} indep={indep} r={r_gpu:.6g}")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 2
    km = int(_ldlt_cuda().k_max())
    high_k = tuple(k for k in HIGH_K if k <= km)
    if high_k != HIGH_K:
        print(
            f"device k_max={km}: skipping HIGH_K {tuple(k for k in HIGH_K if k > km)}"
        )
    test_caps()
    test_fused_vs_unfused_k32()
    for k in high_k:
        test_count_fill(k)
        test_run_depth(k, fci=False)
        test_run_depth(k, fci=True)
        test_pdsep_stripe(k, fused=(k <= FUSED_K_MAX))
        if k > FUSED_K_MAX:
            test_pdsep_stripe(k, fused=True)
    print("ALL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
