"""PC-stable-cuda skeleton vs sequential CIT.fisher_z (n_jobs=1)."""

import sys
from pathlib import Path

import numpy as np
import torch

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whydra.algorithms.graph_core import CIT, CausalGraph
from whydra.algorithms.standalone.skeleton import SkeletonDiscovery


def _random_gaussian(n_samples: int, n_vars: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n_vars, n_vars))
    W = np.tril(W, k=-1) * 0.4
    noise = rng.normal(size=(n_samples, n_vars))
    data = noise.copy()
    for j in range(n_vars):
        data[:, j] = data[:, :j] @ W[j, :j] + noise[:, j]
    return data


def _sparse_gaussian(
    n_samples: int, n_vars: int, seed: int = 0, max_parents: int = 3
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n_samples, n_vars))
    data = noise.copy()
    for j in range(n_vars):
        m = min(max_parents, j)
        if m == 0:
            continue
        parents = rng.choice(j, size=m, replace=False)
        data[:, j] += data[:, parents] @ (rng.normal(size=m) * 0.5)
    return data


def _sepsets_as_sets(cg) -> dict:
    if getattr(cg, "sepset_ranges", None) is not None:
        S = np.asarray(cg.sepset_S)
        ranges = np.asarray(cg.sepset_ranges)
        out = {}
        p = int(ranges.shape[0])
        for x in range(p):
            for y in range(x + 1, p):
                u = set()
                for a, b in ((x, y), (y, x)):
                    sa, ea = int(ranges[a, b, 0]), int(ranges[a, b, 1])
                    if sa < 0:
                        continue
                    u.add(tuple(sorted(int(v) for v in S[sa:ea])))
                if u:
                    out[(x, y)] = u
                    out[(y, x)] = u
        return out
    if getattr(cg, "sepset_S", None) is not None:
        S = np.asarray(cg.sepset_S)
        IND = np.asarray(cg.sepset_IND)
        out = {}
        ns = int(IND.shape[0])
        for i in range(ns):
            x, y = int(IND[i, 0]), int(IND[i, 1])
            a, b = int(IND[i, 2]), int(IND[i, 3])
            key = tuple(sorted(int(v) for v in S[a:b]))
            out.setdefault((x, y), set()).add(key)
            out.setdefault((y, x), set()).add(key)
        return out
    sepset = cg.sepset
    out = {}
    n = sepset.shape[0]
    for i in range(n):
        for j in range(n):
            s = sepset[i, j]
            if s is None:
                continue
            out[(i, j)] = {tuple(sorted(int(v) for v in S)) for S in s}
    return out


def _run_skeleton(
    data: np.ndarray,
    *,
    device: str | None,
    alpha: float = 0.05,
    dtype: str | None = None,
):
    names = [f"X{i}" for i in range(data.shape[1])]
    cg = CausalGraph(data.shape[1], names, device=device)
    cg.set_ind_test(CIT(data, method="fisherz"))
    SkeletonDiscovery(
        cg,
        alpha=alpha,
        stable=True,
        n_jobs=1,
        show_progress=False,
        device=device,
        dtype=dtype,
    ).run()
    return cg


def _compare(data: np.ndarray, tag: str) -> bool:
    cpu = _run_skeleton(data, device=None)
    cuda = _run_skeleton(data, device="cuda")
    g_cpu = np.array(cpu.G.graph)
    g_cuda = np.array(cuda.G.graph)
    same_adj = np.array_equal(g_cpu, g_cuda)
    cpu_s = _sepsets_as_sets(cpu)
    cuda_s = _sepsets_as_sets(cuda)
    same_sep = cpu_s == cuda_s
    n_edges = int(np.sum(g_cpu != 0) // 2)
    n_sep = len(cpu_s) // 2
    print(
        f"{tag}: adj_equal={same_adj} sepset_equal={same_sep} "
        f"undirected_edges={n_edges} sepset_pairs={n_sep}"
    )
    if not same_adj:
        diff = np.argwhere(g_cpu != g_cuda)
        print(f"  adj diffs (first 8): {diff[:8].tolist()}")
    if not same_sep:
        keys = set(cpu_s) | set(cuda_s)
        shown = 0
        for k in sorted(keys):
            if cpu_s.get(k) != cuda_s.get(k):
                print(f"  sepset {k}: cpu={cpu_s.get(k)} cuda={cuda_s.get(k)}")
                shown += 1
                if shown >= 12:
                    break
    return same_adj and same_sep


def main() -> int:
    from whydra.algorithms.graph_core.cuda_fisher_z import _ldlt_cuda
    from whydra.algorithms.graph_core.cuda_graph_batch import _graph_batch_cuda

    fz, gb = _ldlt_cuda(), _graph_batch_cuda()
    print(
        f"k_max fisher_z={int(fz.k_max())} combinadic={int(gb.k_max())} "
        f"fused={int(gb.fused_k_max())}"
    )
    # fisher_z k_max is the device-fit cap: 96 on GPUs with >=99KB opt-in
    # dynamic smem (A40); 88 on 64KB consumer parts (RTX 2080 Ti, sm_75),
    # where larger k falls back to the host wavefront.
    cap = int(torch.cuda.get_device_properties(0).shared_memory_per_block_optin)
    kfit = 96
    while kfit > 4 and (kfit * kfit + 3 * kfit) * 8 + 4 * kfit > cap:
        kfit -= 1
    if int(fz.k_max()) != kfit or int(gb.k_max()) != 96 or int(gb.fused_k_max()) != 32:
        print("K_MAX mismatch")
        return 1
    ok = True
    ok = _compare(_random_gaussian(600, 12, seed=3), "p=12 dense") and ok
    ok = _compare(_random_gaussian(600, 50, seed=3), "p=50 dense") and ok
    ok = _compare(_sparse_gaussian(600, 100, seed=3), "p=100 sparse") and ok
    ok = _compare(_sparse_gaussian(600, 200, seed=3), "p=200 sparse") and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
