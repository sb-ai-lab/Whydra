"""CUDA Fisher-Z vs batched NumPy float64 LDLᵀ for every k=0..96.

Baseline is always ``NumpyFisherZ`` on ``np.corrcoef`` float64 (same unpivoted
LDLᵀ). CUDA is checked in float64 against that baseline.
Specialized kernels are k=0..3; generic TPT is 8 / 16 / 32. k=96 is a 96×96
conditioner block.
"""

import sys
from pathlib import Path

# Script-style CUDA suite: run `python tests/test_cuda_fisher_z.py`. It uses a
# Report accumulator and device-specific k sweeps, so it is not a pytest module.
__test__ = False

import numpy as np
import torch

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whydra.algorithms.graph_core.cuda_fisher_z import (
    CudaFisherZ,
    _ldlt_cuda,
    corrcoef_from_data,
)
from whydra.algorithms.graph_core.fisher_z_schur_chol import (
    NumpyFisherZ,
    fisher_z_schur_chol,
    partial_corr_r_batch_schur,
)

K_MAX = 96
N_SAMPLES = 800


def _random_gaussian(n_samples: int, n_vars: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = np.tril(rng.normal(size=(n_vars, n_vars)), k=-1) * 0.3
    noise = rng.normal(size=(n_samples, n_vars))
    data = noise.copy()
    for j in range(n_vars):
        data[:, j] = data[:, :j] @ w[j, :j] + noise[:, j]
    return data


def _atol_r(k: int) -> float:
    """Unpivoted LDLᵀ vs NumPy float64. TPT reductions and C vs np.corrcoef grow with k."""
    return 1e-12 + 1e-13 * max(k, 1) ** 3


def _atol_p(k: int) -> float:
    return 1e-12 + 1e-12 * max(k, 1) ** 3


def _random_batch(p: int, k: int, B: int, rng: np.random.Generator):
    if k > p - 2:
        raise ValueError(f"k={k} needs p>={k + 2}")
    x = rng.integers(0, p, size=B, dtype=np.int32)
    y = rng.integers(0, p, size=B, dtype=np.int32)
    y = np.where(x == y, (y + 1) % p, y).astype(np.int32)
    S = np.empty((B, k), dtype=np.int32)
    for b in range(B):
        mask = np.ones(p, dtype=bool)
        mask[int(x[b])] = False
        mask[int(y[b])] = False
        S[b] = rng.choice(np.flatnonzero(mask), size=k, replace=False)
    return x, y, S


def _fixed_batch(k: int, B: int):
    """x=0, y=1, S=2..k+1. Needs p >= k+2."""
    x = np.zeros(B, dtype=np.int32)
    y = np.ones(B, dtype=np.int32)
    S = np.broadcast_to(np.arange(2, 2 + k, dtype=np.int32), (B, k)).copy()
    return x, y, S


def _max_abs(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return 0.0
    d = np.abs(a - b)
    d = np.where(~np.isfinite(a) & ~np.isfinite(b), 0.0, d)
    if not np.any(np.isfinite(d)):
        return float("inf")
    return float(np.nanmax(d))


class Report:
    def __init__(self) -> None:
        self.n_ok = 0
        self.n_fail = 0
        self.worst_r = 0.0
        self.worst_p = 0.0

    def check(self, ok: bool, msg: str, *, dr: float = 0.0, dp: float = 0.0) -> None:
        self.worst_r = max(self.worst_r, dr)
        self.worst_p = max(self.worst_p, dp)
        tag = "OK  " if ok else "FAIL"
        if not ok:
            self.n_fail += 1
        else:
            self.n_ok += 1
        extra = ""
        if dr or dp:
            extra = f"  max_|r|={dr:.3e} max_|p|={dp:.3e}"
        print(f"{tag} {msg}{extra}", flush=True)


def _compare(
    gpu: CudaFisherZ,
    cpu: NumpyFisherZ,
    x,
    y,
    S,
    *,
    group_s: bool = False,
) -> tuple[float, float, bool]:
    x = np.asarray(x).reshape(-1)
    S = np.asarray(S)
    if S.ndim == 1:
        S = S.reshape(-1, 1) if S.size else np.zeros((x.size, 0), dtype=np.int32)
    k = int(S.shape[1])
    r_gpu = gpu.batch_r(x, y, S, group_s=group_s).detach().cpu().numpy()
    r_cpu = partial_corr_r_batch_schur(cpu.corr, x, y, S, eps=cpu.eps)
    p_gpu = _p_from_r_gpu(gpu, r_gpu, k) if group_s else gpu.batch(x, y, S)
    p_cpu = cpu.batch(x, y, S)
    dr = _max_abs(r_gpu, r_cpu)
    dp = _max_abs(p_gpu, p_cpu)
    finite = np.array_equal(np.isfinite(r_gpu), np.isfinite(r_cpu))
    ar = _atol_r(k)
    ap = _atol_p(k)
    ok = finite and dr <= ar and dp <= ap
    return dr, dp, ok


def _p_from_r_gpu(gpu: CudaFisherZ, r: np.ndarray, k: int) -> np.ndarray:
    from whydra.algorithms.graph_core.cuda_fisher_z import _p_value_from_r

    return _p_value_from_r(r, gpu.n_samples, k)


def _numpy_f64(data: np.ndarray, eps: float = 0.0) -> NumpyFisherZ:
    C = np.ascontiguousarray(np.corrcoef(data.T), dtype=np.float64)
    return NumpyFisherZ(C, int(data.shape[0]), method="schur", eps=eps)


def _gpu_cpu(data: np.ndarray, *, dtype="float64", eps: float = 0.0):
    n = int(data.shape[0])
    gpu = CudaFisherZ.from_data(data, n, dtype=dtype, eps=eps)
    return gpu, _numpy_f64(data, eps=eps)


def test_corrcoef(rep: Report, data: np.ndarray) -> None:
    C_np = np.corrcoef(data.T)
    C_gpu = corrcoef_from_data(data).detach().cpu().numpy()
    d = _max_abs(C_gpu, C_np)
    rep.check(d < 1e-12, f"corrcoef_from_data vs np.corrcoef p={data.shape[1]}", dr=d)


def _expect_tpt(k: int) -> int:
    if k <= 3:
        return 1
    if k <= 8:
        return 8
    if k <= 16:
        return 16
    return 32


def _sweep_B(k: int) -> int:
    if k >= 64:
        return 32
    if k >= 48:
        return 64
    return 128


def _device_k_max() -> int:
    """Largest k whose generic 96-cap kernel fits this GPU's opt-in smem.

    One test needs (k*k + 3k) float64 tile cells plus k int indices:
    (k²+3k)·8 + 4k bytes. Datacenter GPUs (A40, 99 KB opt-in) fit the full
    K_MAX=96; consumer parts (RTX 2080 Ti, sm_75) cap at 64 KB, so k_max
    reports 88 there and larger k goes through the host wavefront fallback.
    """
    cap = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    k = K_MAX
    while k > 4 and (k * k + 3 * k) * 8 + 4 * k > cap:
        k -= 1
    return k


def test_k_max(rep: Report) -> None:
    ext = _ldlt_cuda()
    km = int(ext.k_max())
    tpt = [int(ext.tpt_for_k(k)) for k in range(K_MAX + 1)]
    expect = [_expect_tpt(k) for k in range(K_MAX + 1)]
    ok = km == _device_k_max() and tpt == expect
    rep.check(ok, f"k_max={km} tpt bins k=0..{K_MAX}")


def test_k_too_large(rep: Report, gpu: CudaFisherZ) -> None:
    x = np.zeros(1, dtype=np.int32)
    y = np.ones(1, dtype=np.int32)
    S = np.zeros((1, K_MAX + 1), dtype=np.int32)
    try:
        gpu.batch_r(x, y, S)
        ok = False
    except (ValueError, RuntimeError):
        ok = True
    rep.check(ok, f"k={K_MAX + 1} raises")


def test_empty_batch(rep: Report, gpu: CudaFisherZ, cpu: NumpyFisherZ) -> None:
    x = np.zeros(0, dtype=np.int32)
    y = np.zeros(0, dtype=np.int32)
    S = np.zeros((0, 3), dtype=np.int32)
    r = gpu.batch_r(x, y, S).detach().cpu().numpy()
    p = gpu.batch(x, y, S)
    p_cpu = cpu.batch(x, y, S)
    ok = r.size == 0 and p.size == 0 and p_cpu.size == 0
    rep.check(ok, "empty batch B=0 k=3")


def test_all_k(rep: Report, gpu: CudaFisherZ, cpu: NumpyFisherZ, p: int) -> None:
    rng = np.random.default_rng(11)
    ext = _ldlt_cuda()
    for k in range(0, _device_k_max() + 1):
        B = _sweep_B(k)
        x, y, S = _random_batch(p, k, B, rng)
        dr, dp, ok = _compare(gpu, cpu, x, y, S)
        tpt = int(ext.tpt_for_k(k))
        rep.check(ok, f"all-k k={k:2d} tpt={tpt:2d} p={p} B={B}", dr=dr, dp=dp)


def test_cta_edges(rep: Report, gpu: CudaFisherZ, cpu: NumpyFisherZ, p: int) -> None:
    rng = np.random.default_rng(23)
    km = _device_k_max()
    ks = tuple(k for k in (0, 1, 2, 3, 4, 8, 9, 16, 17, 32, 48, 64, 96) if k <= km)
    bs = (1, 7, 31, 32, 33, 255, 256, 257)
    for k in ks:
        for B in bs:
            if k >= 64 and B > 33:
                continue
            x, y, S = _random_batch(p, k, B, rng)
            dr, dp, ok = _compare(gpu, cpu, x, y, S)
            rep.check(
                ok,
                f"cta k={k:2d} p={p} B={B}",
                dr=dr,
                dp=dp,
            )


def test_p_sizes(rep: Report) -> None:
    rng = np.random.default_rng(29)
    km = _device_k_max()
    for p, ks, B, seed in (
        (8, tuple(range(0, 7)), 64, 3),
        (16, (0, 1, 2, 3, 4, 8, 12, 14), 64, 5),
        (64, (0, 3, 4, 8, 9, 16, 17, 32), 256, 7),
        (96, (0, 32, 48, 64, 94), 64, 13),
        (128, (0, 3, 4, 8, 16, 32, 64, 96), 128, 9),
        (256, (0, 4, 16, 32, 96), 64, 11),
    ):
        data = _random_gaussian(N_SAMPLES, p, seed=seed)
        gpu, cpu = _gpu_cpu(data)
        for k in (k for k in ks if k <= km):
            x, y, S = _random_batch(p, k, B, rng)
            dr, dp, ok = _compare(gpu, cpu, x, y, S)
            rep.check(ok, f"size p={p:3d} k={k:2d} B={B}", dr=dr, dp=dp)


def test_group_s(rep: Report, gpu: CudaFisherZ, cpu: NumpyFisherZ, p: int) -> None:
    rng = np.random.default_rng(31)
    km = _device_k_max()
    for k, n_unique, repeats in (
        (k, n, r)
        for k, n, r in (
            (2, 32, 8),
            (3, 24, 8),
            (4, 16, 8),
            (8, 12, 8),
            (9, 10, 8),
            (16, 8, 8),
            (17, 6, 8),
            (32, 4, 8),
            (48, 4, 8),
            (96, 2, 4),
        )
        if k <= km
    ):
        rows_x, rows_y, rows_s = [], [], []
        for _ in range(n_unique):
            z = rng.choice(p, size=k, replace=False)
            z.sort()
            rest = [i for i in range(p) if i not in set(z.tolist())]
            for _r in range(repeats):
                xy = rng.choice(rest, size=2, replace=False)
                rows_x.append(int(xy[0]))
                rows_y.append(int(xy[1]))
                rows_s.append(z.copy())
        x = np.asarray(rows_x, dtype=np.int32)
        y = np.asarray(rows_y, dtype=np.int32)
        S = np.stack(rows_s, axis=0).astype(np.int32)
        dr_g, dp_g, ok_g = _compare(gpu, cpu, x, y, S, group_s=True)
        r_u = gpu.batch_r(x, y, S, group_s=False).detach().cpu().numpy()
        r_g = gpu.batch_r(x, y, S, group_s=True).detach().cpu().numpy()
        d_ug = _max_abs(r_u, r_g)
        ok = ok_g and d_ug <= _atol_r(k)
        rep.check(
            ok,
            f"group_s k={k:2d} U={n_unique} B={len(x)} ungrouped_|r|={d_ug:.3e}",
            dr=dr_g,
            dp=dp_g,
        )


def test_fixed_s(rep: Report, gpu: CudaFisherZ, cpu: NumpyFisherZ) -> None:
    km = _device_k_max()
    for k in (k for k in (0, 1, 2, 3, 4, 8, 9, 16, 17, 32, 48, 64, 96) if k <= km):
        x, y, S = _fixed_batch(k, B=64)
        dr, dp, ok = _compare(gpu, cpu, x, y, S)
        rep.check(ok, f"fixed-S k={k:2d} B=64", dr=dr, dp=dp)


def test_strided(rep: Report, gpu: CudaFisherZ, cpu: NumpyFisherZ) -> None:
    C_t = gpu.corr.t()
    if C_t.is_contiguous():
        rep.check(False, "strided C.t() unexpectedly contiguous")
        return
    gpu_s = CudaFisherZ(C_t, gpu.n_samples, eps=gpu.eps)
    if gpu_s.corr.is_contiguous():
        rep.check(False, "strided gpu.corr unexpectedly contiguous")
        return
    rng = np.random.default_rng(43)
    p = int(gpu.corr.size(0))
    cpu_s = NumpyFisherZ(gpu_s.host_corr(), gpu.n_samples, method="schur", eps=gpu.eps)
    for k in (k for k in (0, 2, 8, 17, 48, 96) if k <= _device_k_max()):
        x, y, S = _random_batch(p, k, 16 if k >= 48 else 32, rng)
        dr, dp, ok = _compare(gpu_s, cpu_s, x, y, S)
        p_c = gpu.batch(x, y, S)
        p_s = gpu_s.batch(x, y, S)
        dcs = _max_abs(p_c, p_s)
        ok = ok and dcs <= _atol_p(k)
        rep.check(ok, f"strided C.t() k={k}", dr=dr, dp=max(dp, dcs))


def test_singular_eps(rep: Report) -> None:
    C0 = np.array(
        [[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 0.0]],
        dtype=np.float64,
    )
    x = np.array([0], dtype=np.int32)
    y = np.array([1], dtype=np.int32)
    S = np.array([[2]], dtype=np.int32)
    gpu0 = CudaFisherZ(C0, N_SAMPLES)
    cpu0 = NumpyFisherZ(C0, N_SAMPLES, method="schur")
    with np.errstate(divide="ignore", invalid="ignore"):
        p0 = float(gpu0.batch(x, y, S)[0])
        p0n = float(cpu0.batch(x, y, S)[0])
        same_nan = (not np.isfinite(p0)) and (not np.isfinite(p0n))
        gpu_e = CudaFisherZ(C0, N_SAMPLES, eps=1e-12)
        cpu_e = NumpyFisherZ(C0, N_SAMPLES, method="schur", eps=1e-12)
        p_e = float(gpu_e.batch(x, y, S)[0])
        p_en = float(cpu_e.batch(x, y, S)[0])
        ok_eps = np.isfinite(p_e) and abs(p_e - p_en) < 1e-12
        p_ref = fisher_z_schur_chol(C0, N_SAMPLES, 0, 1, (2,), eps=1e-12)
        ok_eps = ok_eps and abs(p_e - p_ref) < 1e-12
    rep.check(same_nan and ok_eps, f"k1 css=0 eps=0 nan; eps=1e-12 p={p_e:.16g}")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 2

    p_full = K_MAX + 2
    data = _random_gaussian(N_SAMPLES, p_full, seed=7)
    gpu, cpu = _gpu_cpu(data)
    rep = Report()

    test_k_max(rep)
    test_k_too_large(rep, gpu)
    test_corrcoef(rep, data)
    test_empty_batch(rep, gpu, cpu)
    test_all_k(rep, gpu, cpu, p_full)
    test_cta_edges(rep, gpu, cpu, p_full)
    test_fixed_s(rep, gpu, cpu)
    test_group_s(rep, gpu, cpu, p_full)
    test_p_sizes(rep)
    test_strided(rep, gpu, cpu)
    test_singular_eps(rep)

    print(
        f"passed={rep.n_ok} failed={rep.n_fail}  "
        f"worst_|r|={rep.worst_r:.3e} worst_|p|={rep.worst_p:.3e}",
        flush=True,
    )
    return 1 if rep.n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
