"""CUDA k=0 / k=1 / packed k≥2 Pearson χ² vs sequential ``CIT.chi_square``."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whydra.algorithms.graph_core.cuda_chisq import CudaChisqHost
from whydra.algorithms.graph_core.independence_tests import CIT


def _mixed(n=800, seed=0):
    rng = np.random.default_rng(seed)
    cols = [
        rng.integers(0, 2, size=n),
        rng.integers(0, 3, size=n),
        rng.integers(0, 5, size=n),
        rng.integers(0, 8, size=n),
        rng.integers(0, 21, size=n),
        np.zeros(n, dtype=np.int64),
        rng.integers(0, 4, size=n),
    ]
    return np.stack(cols, axis=1)


def _compare(gpu, cpu, xs, ys, zs=None):
    both_nan = np.isnan(gpu) & np.isnan(cpu)
    ok = both_nan | np.isclose(gpu, cpu, rtol=1e-10, atol=1e-12, equal_nan=True)
    n_bad = int(np.sum(~ok))
    max_abs = float(np.nanmax(np.abs(gpu - cpu))) if gpu.size else 0.0
    return n_bad, max_abs, np.flatnonzero(~ok)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_chisq_k0_k1_matches_cpu() -> None:
    """Small k=0/k=1 batch must match the sequential Pearson chi-square."""
    data = np.delete(_mixed(n=300, seed=5), 5, axis=1)  # drop the constant column
    cit = CIT(data, method="chisq")
    batcher = CudaChisqHost(cit)
    p = data.shape[1]

    xs, ys = np.triu_indices(p, k=1)
    empty = np.zeros((xs.size, 0), dtype=np.int64)
    gpu0 = batcher.batch(xs, ys, empty)
    cpu0 = np.array(
        [cit.chi_square(int(x), int(y), []) for x, y in zip(xs, ys, strict=False)]
    )
    assert _compare(gpu0, cpu0, xs, ys)[0] == 0

    xs1, ys1, zs1 = [], [], []
    for x in range(p):
        for y in range(x + 1, p):
            for z in range(p):
                if z not in (x, y):
                    xs1.append(x)
                    ys1.append(y)
                    zs1.append(z)
    xa = np.asarray(xs1, dtype=np.int64)
    ya = np.asarray(ys1, dtype=np.int64)
    za = np.asarray(zs1, dtype=np.int64)
    gpu1 = batcher.batch(xa, ya, za.reshape(-1, 1))
    cpu1 = np.array(
        [
            cit.chi_square(int(x), int(y), [int(z)])
            for x, y, z in zip(xa, ya, za, strict=False)
        ]
    )
    assert _compare(gpu1, cpu1, xa, ya, za)[0] == 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_chisq_gmem_k2_k3_matches_cpu() -> None:
    """The global-memory k>=2 fallback must match the CPU without a host call."""
    rng = np.random.default_rng(11)
    n, cardinality, p = 5000, 3, 8
    data = np.stack([rng.integers(0, cardinality, size=n) for _ in range(p)], axis=1)
    cit = CIT(data, method="chisq")
    batcher = CudaChisqHost(cit)
    if batcher._pack_tiled_fits():
        pytest.skip("SMEM path fits on this device")

    xs = np.array([0, 1, 2], dtype=np.int64)
    ys = np.array([3, 4, 5], dtype=np.int64)
    original = CudaChisqHost._sequential_batch

    def _no_host(_self, *_args, **_kwargs):
        raise AssertionError("GMEM fallback called the host reference")

    CudaChisqHost._sequential_batch = _no_host
    try:
        for S in (
            np.array([[6, 7], [5, 7], [4, 6]], dtype=np.int64),
            np.array([[6, 7, 2], [5, 7, 1], [4, 6, 0]], dtype=np.int64),
        ):
            gpu = batcher.batch(xs, ys, S)
            cpu = np.array(
                [
                    cit.chi_square(int(x), int(y), [int(v) for v in row])
                    for x, y, row in zip(xs, ys, S, strict=False)
                ]
            )
            assert _compare(gpu, cpu, xs, ys)[0] == 0
    finally:
        CudaChisqHost._sequential_batch = original


def main() -> int:
    data = _mixed()
    cit = CIT(data, method="chisq")
    batcher = CudaChisqHost(cit)
    p = data.shape[1]
    rc = 0
    off = batcher._csr_off.detach().cpu().numpy()
    idx = batcher._csr_idx.detach().cpu().numpy()
    for j in range(p):
        col = cit.data_int[:, j]
        cj = int(cit.cardinalities[j])
        if int(off[j, cj]) != data.shape[0]:
            print(f"CSR offsets[{j}] last={off[j, cj]} n={data.shape[0]}")
            rc = 1
        for u in range(cj):
            ts = idx[j, off[j, u] : off[j, u + 1]]
            if ts.size != int(np.sum(col == u)) or (ts.size and np.any(col[ts] != u)):
                print(f"CSR col {j} u={u} broken")
                rc = 1
    if rc == 0:
        print(f"CSR columns={p} ok")

    xs, ys = np.triu_indices(p, k=1)
    S0 = np.zeros((xs.size, 0), dtype=np.int64)
    gpu0 = batcher.batch(xs, ys, S0)
    cpu0 = np.array(
        [cit.chi_square(int(x), int(y), []) for x, y in zip(xs, ys, strict=False)]
    )
    n_bad, max_abs, idx = _compare(gpu0, cpu0, xs, ys)
    print(
        f"k=0 pairs={xs.size} c_full={batcher._c_full} "
        f"max_abs={max_abs:.3e} mismatches={n_bad}"
    )
    if n_bad:
        for i in idx[:8]:
            print(f"  ({int(xs[i])},{int(ys[i])}) gpu={gpu0[i]} cpu={cpu0[i]}")
        rc = 1

    xs1, ys1, zs1 = [], [], []
    for x in range(p):
        for y in range(x + 1, p):
            for z in range(p):
                if z in (x, y):
                    continue
                xs1.append(x)
                ys1.append(y)
                zs1.append(z)
    xs1 = np.asarray(xs1, dtype=np.int64)
    ys1 = np.asarray(ys1, dtype=np.int64)
    zs1 = np.asarray(zs1, dtype=np.int64)
    S1 = zs1.reshape(-1, 1)
    gpu1 = batcher.batch(xs1, ys1, S1)
    cpu1 = np.array(
        [
            cit.chi_square(int(x), int(y), [int(z)])
            for x, y, z in zip(xs1, ys1, zs1, strict=False)
        ]
    )
    n_bad, max_abs, idx = _compare(gpu1, cpu1, xs1, ys1, zs1)
    print(f"k=1 tests={xs1.size} max_abs={max_abs:.3e} mismatches={n_bad}")
    if n_bad:
        for i in idx[:8]:
            print(
                f"  ({int(xs1[i])},{int(ys1[i])}|{int(zs1[i])}) "
                f"gpu={gpu1[i]} cpu={cpu1[i]}"
            )
        rc = 1

    n_p1 = int(np.sum(np.isclose(gpu1, 1.0)))
    n_p1_cpu = int(np.sum(np.isclose(cpu1, 1.0)))
    print(f"k=1 p=1.0 (df=0) gpu={n_p1} cpu={n_p1_cpu}")

    rng = np.random.default_rng(1)
    wide = np.stack(
        [rng.integers(0, 32, size=400) for _ in range(4)],
        axis=1,
    )
    cit_w = CIT(wide, method="chisq")
    batcher_w = CudaChisqHost(cit_w)
    xs_w = np.array([0, 0, 1], dtype=np.int64)
    ys_w = np.array([1, 2, 2], dtype=np.int64)
    zs_w = np.array([2, 3, 3], dtype=np.int64)
    gpu_w = batcher_w.batch(xs_w, ys_w, zs_w.reshape(-1, 1))
    cpu_w = np.array(
        [
            cit_w.chi_square(int(x), int(y), [int(z)])
            for x, y, z in zip(xs_w, ys_w, zs_w, strict=False)
        ]
    )
    n_bad, max_abs, idx = _compare(gpu_w, cpu_w, xs_w, ys_w, zs_w)
    print(f"k=1 c=32 max_abs={max_abs:.3e} mismatches={n_bad}")
    if n_bad:
        for i in idx[:8]:
            print(
                f"  ({int(xs_w[i])},{int(ys_w[i])}|{int(zs_w[i])}) "
                f"gpu={gpu_w[i]} cpu={cpu_w[i]}"
            )
        rc = 1

    rng = np.random.default_rng(2)
    wide40 = np.stack(
        [
            rng.integers(0, 40, size=600),
            rng.integers(0, 40, size=600),
            rng.integers(0, 5, size=600),
            rng.integers(0, 50, size=600),
        ],
        axis=1,
    )
    cit40 = CIT(wide40, method="chisq")
    b40 = CudaChisqHost(cit40)
    xs0 = np.array([0, 0, 1, 0], dtype=np.int64)
    ys0 = np.array([1, 2, 2, 3], dtype=np.int64)
    gpu_k0 = b40.batch(xs0, ys0, np.zeros((xs0.size, 0), dtype=np.int64))
    cpu_k0 = np.array(
        [cit40.chi_square(int(x), int(y), []) for x, y in zip(xs0, ys0, strict=False)]
    )
    n_bad, max_abs, idx = _compare(gpu_k0, cpu_k0, xs0, ys0)
    print(
        f"k=0 tiled max_card={b40._max_card} max_abs={max_abs:.3e} mismatches={n_bad}"
    )
    if n_bad:
        for i in idx[:8]:
            print(f"  ({int(xs0[i])},{int(ys0[i])}) gpu={gpu_k0[i]} cpu={cpu_k0[i]}")
        rc = 1
    zs1 = np.array([2, 3, 3, 2], dtype=np.int64)
    gpu_k1 = b40.batch(xs0, ys0, zs1.reshape(-1, 1))
    cpu_k1 = np.array(
        [
            cit40.chi_square(int(x), int(y), [int(z)])
            for x, y, z in zip(xs0, ys0, zs1, strict=False)
        ]
    )
    n_bad, max_abs, idx = _compare(gpu_k1, cpu_k1, xs0, ys0, zs1)
    print(
        f"k=1 tiled max_card={b40._max_card} max_abs={max_abs:.3e} mismatches={n_bad}"
    )
    if n_bad:
        for i in idx[:8]:
            print(
                f"  ({int(xs0[i])},{int(ys0[i])}|{int(zs1[i])}) "
                f"gpu={gpu_k1[i]} cpu={cpu_k1[i]}"
            )
        rc = 1

    xs2, ys2, S2 = [], [], []
    for x in range(p):
        for y in range(x + 1, p):
            for z0 in range(p):
                if z0 in (x, y):
                    continue
                for z1 in range(z0 + 1, p):
                    if z1 in (x, y):
                        continue
                    xs2.append(x)
                    ys2.append(y)
                    S2.append((z0, z1))
    xs2 = np.asarray(xs2, dtype=np.int64)
    ys2 = np.asarray(ys2, dtype=np.int64)
    S2 = np.asarray(S2, dtype=np.int64)
    gpu2 = batcher.batch(xs2, ys2, S2)
    cpu2 = np.array(
        [
            cit.chi_square(int(x), int(y), [int(z0), int(z1)])
            for x, y, (z0, z1) in zip(xs2, ys2, S2, strict=False)
        ]
    )
    n_bad, max_abs, idx = _compare(gpu2, cpu2, xs2, ys2)
    print(f"k=2 tests={xs2.size} max_abs={max_abs:.3e} mismatches={n_bad}")
    if n_bad:
        for i in idx[:8]:
            print(
                f"  ({int(xs2[i])},{int(ys2[i])}|{S2[i].tolist()}) "
                f"gpu={gpu2[i]} cpu={cpu2[i]}"
            )
        rc = 1

    rng = np.random.default_rng(3)
    pack = np.stack(
        [rng.integers(0, 8, size=500) for _ in range(5)],
        axis=1,
    )
    cit_p = CIT(pack, method="chisq")
    bp = CudaChisqHost(cit_p)
    xs_p = np.array([0, 0, 1], dtype=np.int64)
    ys_p = np.array([1, 2, 2], dtype=np.int64)
    S_p = np.array([[2, 3], [3, 4], [3, 4]], dtype=np.int64)
    gpu_p = bp.batch(xs_p, ys_p, S_p)
    cpu_p = np.array(
        [
            cit_p.chi_square(int(x), int(y), [int(z0), int(z1)])
            for x, y, (z0, z1) in zip(xs_p, ys_p, S_p, strict=False)
        ]
    )
    n_bad, max_abs, idx = _compare(gpu_p, cpu_p, xs_p, ys_p)
    print(
        f"k=2 packed cz=64 max_card={bp._max_card} "
        f"max_abs={max_abs:.3e} mismatches={n_bad}"
    )
    if n_bad:
        for i in idx[:8]:
            print(
                f"  ({int(xs_p[i])},{int(ys_p[i])}|{S_p[i].tolist()}) "
                f"gpu={gpu_p[i]} cpu={cpu_p[i]}"
            )
        rc = 1

    gpu40_2 = b40.batch(
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([[2, 3]], dtype=np.int64),
    )
    cpu40_2 = np.array([cit40.chi_square(0, 1, [2, 3])])
    n_bad, max_abs, idx = _compare(gpu40_2, cpu40_2, np.array([0]), np.array([1]))
    print(f"k=2 tiled C=50 cz=250 max_abs={max_abs:.3e} mismatches={n_bad}")
    if n_bad:
        print(f"  gpu={gpu40_2[0]} cpu={cpu40_2[0]}")
        rc = 1

    for C, n_hi in ((512, 800), (1024, 1200)):
        rng_c = np.random.default_rng(C)
        wide_c = np.stack(
            [rng_c.integers(0, C, size=n_hi) for _ in range(4)],
            axis=1,
        )
        cit_c = CIT(wide_c, method="chisq")
        bc = CudaChisqHost(cit_c)
        xs_c = np.array([0, 0, 1], dtype=np.int64)
        ys_c = np.array([1, 2, 2], dtype=np.int64)
        gpu_c0 = bc.batch(xs_c, ys_c, np.zeros((xs_c.size, 0), dtype=np.int64))
        cpu_c0 = np.array(
            [
                cit_c.chi_square(int(x), int(y), [])
                for x, y in zip(xs_c, ys_c, strict=False)
            ]
        )
        n_bad, max_abs, idx = _compare(gpu_c0, cpu_c0, xs_c, ys_c)
        print(
            f"k=0 C={C} V={bc._ext.tile_v_for_card(C)} "
            f"max_abs={max_abs:.3e} mismatches={n_bad}"
        )
        if n_bad:
            for i in idx[:8]:
                print(
                    f"  ({int(xs_c[i])},{int(ys_c[i])}) gpu={gpu_c0[i]} cpu={cpu_c0[i]}"
                )
            rc = 1
        zs_c = np.array([2, 3, 3], dtype=np.int64)
        gpu_c1 = bc.batch(xs_c, ys_c, zs_c.reshape(-1, 1))
        cpu_c1 = np.array(
            [
                cit_c.chi_square(int(x), int(y), [int(z)])
                for x, y, z in zip(xs_c, ys_c, zs_c, strict=False)
            ]
        )
        n_bad, max_abs, idx = _compare(gpu_c1, cpu_c1, xs_c, ys_c, zs_c)
        print(f"k=1 C={C} max_abs={max_abs:.3e} mismatches={n_bad}")
        if n_bad:
            for i in idx[:8]:
                print(
                    f"  ({int(xs_c[i])},{int(ys_c[i])}|{int(zs_c[i])}) "
                    f"gpu={gpu_c1[i]} cpu={cpu_c1[i]}"
                )
            rc = 1
        b4 = CudaChisqHost(cit_c, tile_v=4)
        gpu_v4 = b4.batch(xs_c, ys_c, np.zeros((xs_c.size, 0), dtype=np.int64))
        n_bad, max_abs, idx = _compare(gpu_v4, cpu_c0, xs_c, ys_c)
        print(f"k=0 C={C} force V=4 max_abs={max_abs:.3e} mismatches={n_bad}")
        if n_bad:
            rc = 1

    rng_obs = np.random.default_rng(7)
    n_obs, C_obs = 400, 80
    wide_obs = np.stack(
        [rng_obs.integers(0, C_obs, size=n_obs) for _ in range(5)],
        axis=1,
    )
    cit_obs = CIT(wide_obs, method="chisq")
    b_obs = CudaChisqHost(cit_obs)
    seq_calls = {"n": 0}
    orig_seq = CudaChisqHost._sequential_batch

    def _count_seq(self, *a, **kw):
        seq_calls["n"] += 1
        return orig_seq(self, *a, **kw)

    CudaChisqHost._sequential_batch = _count_seq
    try:
        xs_o = np.array([0, 0, 1], dtype=np.int64)
        ys_o = np.array([1, 2, 2], dtype=np.int64)
        S_o = np.array([[2, 3], [3, 4], [2, 4]], dtype=np.int64)
        gpu_o = b_obs.batch(xs_o, ys_o, S_o)
        S3 = np.array([[2, 3, 4], [1, 3, 4], [0, 3, 4]], dtype=np.int64)
        gpu_o3 = b_obs.batch(xs_o, ys_o, S3)
    finally:
        CudaChisqHost._sequential_batch = orig_seq
    if seq_calls["n"]:
        print(f"observed-Z used sequential n={seq_calls['n']}")
        rc = 1
    cpu_o = np.array(
        [
            cit_obs.chi_square(int(x), int(y), [int(z0), int(z1)])
            for x, y, (z0, z1) in zip(xs_o, ys_o, S_o, strict=False)
        ]
    )
    n_bad, max_abs, idx = _compare(gpu_o, cpu_o, xs_o, ys_o)
    print(
        f"k=2 observed-Z C={C_obs} cz={C_obs * C_obs} n={n_obs} "
        f"max_abs={max_abs:.3e} mismatches={n_bad} seq_calls={seq_calls['n']}"
    )
    if n_bad:
        for i in idx[:8]:
            print(
                f"  ({int(xs_o[i])},{int(ys_o[i])}|{S_o[i].tolist()}) "
                f"gpu={gpu_o[i]} cpu={cpu_o[i]}"
            )
        rc = 1
    cpu_o3 = np.array(
        [
            cit_obs.chi_square(int(x), int(y), [int(z0), int(z1), int(z2)])
            for x, y, (z0, z1, z2) in zip(xs_o, ys_o, S3, strict=False)
        ]
    )
    n_bad, max_abs, idx = _compare(gpu_o3, cpu_o3, xs_o, ys_o)
    print(
        f"k=3 observed-Z C={C_obs} cz={C_obs**3} n={n_obs} "
        f"max_abs={max_abs:.3e} mismatches={n_bad}"
    )
    if n_bad:
        for i in idx[:8]:
            print(
                f"  ({int(xs_o[i])},{int(ys_o[i])}|{S3[i].tolist()}) "
                f"gpu={gpu_o3[i]} cpu={cpu_o3[i]}"
            )
        rc = 1

    # n=5000, C=3 misses the 64 KB-part SMEM budget; the global-memory
    # fallback must handle it without calling the host reference.
    rng_g = np.random.default_rng(11)
    n_g, C_g, p_g = 5000, 3, 8
    data_g = np.stack([rng_g.integers(0, C_g, size=n_g) for _ in range(p_g)], axis=1)
    cit_g = CIT(data_g, method="chisq")
    b_g = CudaChisqHost(cit_g)
    if b_g._pack_tiled_fits():
        print("GMEM test skipped: SMEM path fits on this device")
    else:
        seq_calls = {"n": 0}
        orig_seq = CudaChisqHost._sequential_batch

        def _count_seq_g(self, *a, **kw):
            seq_calls["n"] += 1
            return orig_seq(self, *a, **kw)

        CudaChisqHost._sequential_batch = _count_seq_g
        try:
            xs_g = np.array([0, 1, 2], dtype=np.int64)
            ys_g = np.array([3, 4, 5], dtype=np.int64)
            S_g = np.array([[6, 7], [5, 7], [4, 6]], dtype=np.int64)
            gpu_g = b_g.batch(xs_g, ys_g, S_g)
            S_g3 = np.array([[6, 7, 2], [5, 7, 1], [4, 6, 0]], dtype=np.int64)
            gpu_g3 = b_g.batch(xs_g, ys_g, S_g3)
        finally:
            CudaChisqHost._sequential_batch = orig_seq
        cpu_g = np.array(
            [
                cit_g.chi_square(int(x), int(y), [int(v) for v in S])
                for x, y, S in zip(xs_g, ys_g, S_g, strict=False)
            ]
        )
        cpu_g3 = np.array(
            [
                cit_g.chi_square(int(x), int(y), [int(v) for v in S])
                for x, y, S in zip(xs_g, ys_g, S_g3, strict=False)
            ]
        )
        for label, gp, cp in (("k=2", gpu_g, cpu_g), ("k=3", gpu_g3, cpu_g3)):
            n_bad, max_abs, idx = _compare(gp, cp, xs_g, ys_g)
            print(
                f"{label} GMEM n={n_g} C={C_g} max_abs={max_abs:.3e} "
                f"mismatches={n_bad} seq_calls={seq_calls['n']}"
            )
            if n_bad or seq_calls["n"]:
                rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
