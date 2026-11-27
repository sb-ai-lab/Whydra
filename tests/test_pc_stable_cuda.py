"""PC-stable-cuda vs sequential PC-stable.

Sequential gold: ``n_jobs=1``, ``CIT.fisher_z``, Python Meek.
PC-stable-cuda: fused skeleton + CUDA Meek. Skeleton, sepsets, and CPDAG must match.
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import torch

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whydra.algorithms.graph_core import CIT, CausalGraph
from whydra.algorithms.graph_core.cuda_fisher_z import _ldlt_cuda
from whydra.algorithms.graph_core.cuda_graph_batch import _graph_batch_cuda
from whydra.algorithms.standalone.pc_algo import pc_stable


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


def _dense(n, p, seed=0):
    rng = np.random.default_rng(seed)
    w = np.tril(rng.normal(size=(p, p)), k=-1) * 0.3
    noise = rng.normal(size=(n, p))
    data = noise.copy()
    for j in range(p):
        data[:, j] = data[:, :j] @ w[j, :j] + noise[:, j]
    return data


def _sparse(n, p, seed=0, max_parents=3):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n, p))
    data = noise.copy()
    for j in range(p):
        m = min(max_parents, j)
        if m == 0:
            continue
        parents = rng.choice(j, size=m, replace=False)
        data[:, j] += data[:, parents] @ (rng.normal(size=m) * 0.5)
    return data


def _run(data: np.ndarray, *, device: str | None):
    names = [f"X{i}" for i in range(data.shape[1])]
    cg = CausalGraph(data.shape[1], names, device=device)
    cg.set_ind_test(CIT(data, method="fisherz"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pc_stable(cg, alpha=0.05, n_jobs=1, show_progress=False, device=device)
    return {
        "graph": np.array(cg.G.graph),
        "sep": _sepsets_as_sets(cg),
        "cg": cg,
    }


def _fmt_diff(a: np.ndarray, b: np.ndarray) -> str:
    n = int(np.sum(a != b))
    if n == 0:
        return "equal"
    idx = np.argwhere(a != b)[:6]
    return f"{n} cells, first={idx.tolist()}"


def _compare(data: np.ndarray, tag: str) -> bool:
    print(f"\n{tag}  n={data.shape[0]} p={data.shape[1]}")
    seq = _run(data, device=None)
    cuda = _run(data, device="cuda")
    graph = np.array_equal(seq["graph"], cuda["graph"])
    sep = seq["sep"] == cuda["sep"]
    skel_seq = seq["graph"] != 0
    skel_cuda = cuda["graph"] != 0
    skel = np.array_equal(skel_seq, skel_cuda)
    print(
        f"  PC-stable-cuda vs seq: skel={skel} cpdag={graph} sepset={sep} "
        f"cpdag_diff={_fmt_diff(seq['graph'], cuda['graph'])}"
    )
    if not sep:
        keys = set(seq["sep"]) | set(cuda["sep"])
        shown = 0
        for k in sorted(keys):
            if seq["sep"].get(k) != cuda["sep"].get(k):
                print(
                    f"    sepset {k}: seq={seq['sep'].get(k)} cuda={cuda['sep'].get(k)}"
                )
                shown += 1
                if shown >= 8:
                    break
    return skel and graph and sep


# Sequential PC-stable on this SEM stays under 1 min (n=1000, α=0.05, seed=0).
# p=120 max_parents=8 is ~55 s sequential; skip it here so seq+CUDA still fits.
_SPARSE_CASES = (
    (50, 3),
    (50, 8),
    (50, 16),
    (80, 12),
    (100, 8),
)


def main() -> int:
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
    ok = _compare(_dense(200, 12, seed=0), "p=12 dense SEM") and ok
    for p, max_parents in _SPARSE_CASES:
        ok = (
            _compare(
                _sparse(1000, p, seed=0, max_parents=max_parents),
                f"p={p} max_parents={max_parents}",
            )
            and ok
        )
    print("\nALL_OK" if ok else "\nMISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
