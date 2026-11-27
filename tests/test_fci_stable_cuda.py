"""FCI-stable-cuda vs sequential FCI-stable v2 (PAG + sepsets)."""

import sys
from pathlib import Path

import numpy as np
import torch

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whydra.algorithms.graph_core import CIT, CausalGraph
from whydra.algorithms.graph_core.cuda_fisher_z import _ldlt_cuda
from whydra.algorithms.graph_core.cuda_graph_batch import _graph_batch_cuda
from whydra.algorithms.standalone.fci_algo_v2 import _sepset_as_pairs, fci_stable_v2


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
    pag = fci_stable_v2(
        cg, data, alpha=0.05, n_jobs=1, device=device, show_progress=False
    )
    return pag, _sepset_as_pairs(getattr(fci_stable_v2, "last_sepset", {}))


def _compare(data: np.ndarray, tag: str) -> bool:
    print(f"\n{tag}  n={data.shape[0]} p={data.shape[1]}")
    cpu, cpu_sep = _run(data, device=None)
    cuda, cuda_sep = _run(data, device="cuda")
    sk_equal = np.array_equal(cpu != 0, cuda != 0)
    pag_equal = np.array_equal(cpu, cuda)
    sep_equal = cpu_sep == cuda_sep
    print(
        f"  FCI-stable-cuda vs seq: skel={sk_equal} pag={pag_equal} sepset={sep_equal}"
    )
    return sk_equal and pag_equal and sep_equal


# Sequential FCI-stable v2 on this SEM stays under 1 min (n=1000, α=0.05, seed=0).
_SPARSE_CASES = (
    (32, 12),
    (50, 8),
    (50, 16),
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
    ok = _compare(_sparse(600, 12, seed=3), "p=12 max_parents=3 n=600 seed=3") and ok
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
