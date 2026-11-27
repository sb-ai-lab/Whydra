"""FCI-stable-cuda skeleton vs sequential CIT (n_jobs=1)."""

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


def _sepsets_as_sets(sepset: np.ndarray) -> dict:
    out = {}
    n = sepset.shape[0]
    for i in range(n):
        for j in range(n):
            s = sepset[i, j]
            if s is None:
                continue
            out[(i, j)] = {tuple(sorted(int(v) for v in S)) for S in s}
    return out


def _run_fci_skeleton(data: np.ndarray, *, device: str | None, alpha: float = 0.05):
    names = [f"X{i}" for i in range(data.shape[1])]
    cg = CausalGraph(data.shape[1], names, device=device)
    cg.set_ind_test(CIT(data, method="fisherz"))
    out = SkeletonDiscovery(
        cg,
        alpha=alpha,
        stable=True,
        n_jobs=1,
        show_progress=False,
        device=device,
    ).run_fci()
    return cg, out


def _sepset_dict_as_sets(sepset: dict) -> dict:
    out = {}
    for k, S in sepset.items():
        if not S:
            continue
        out[k] = tuple(sorted(int(v) for v in S))
    return out


def _sepset_from_out(out: dict) -> dict:
    if out.get("sepset_ranges") is not None:
        from whydra.algorithms.graph_core.cuda_pdsep import sepset_csr_to_dict

        p = int(np.asarray(out["sk"]).shape[0])
        d = sepset_csr_to_dict(out["sepset_S"], out["sepset_ranges"], p)
        return {k: tuple(sorted(v)) for k, v in d.items() if v}
    return _sepset_dict_as_sets(out["sepset"])


def main() -> int:
    from whydra.algorithms.graph_core.cuda_fisher_z import _ldlt_cuda
    from whydra.algorithms.graph_core.cuda_graph_batch import _graph_batch_cuda

    fz, gb = _ldlt_cuda(), _graph_batch_cuda()
    print(
        f"k_max fisher_z={int(fz.k_max())} combinadic={int(gb.k_max())} "
        f"fused={int(gb.fused_k_max())}"
    )
    # fisher_z k_max is the device-fit cap: 96 on >=99KB-optin GPUs (A40),
    # 88 on 64KB consumer parts (RTX 2080 Ti, sm_75).
    cap = int(torch.cuda.get_device_properties(0).shared_memory_per_block_optin)
    kfit = 96
    while kfit > 4 and (kfit * kfit + 3 * kfit) * 8 + 4 * kfit > cap:
        kfit -= 1
    if int(fz.k_max()) != kfit or int(gb.k_max()) != 96 or int(gb.fused_k_max()) != 32:
        return 1
    data = _random_gaussian(n_samples=600, n_vars=12, seed=3)
    cpu_cg, cpu = _run_fci_skeleton(data, device=None)
    cuda_cg, cuda = _run_fci_skeleton(data, device="cuda")

    same_sk = np.array_equal(cpu["sk"], cuda["sk"])
    same_pmax = np.allclose(cpu["pMax"], cuda["pMax"], atol=1e-9, equal_nan=True)
    cpu_sep = _sepset_from_out(cpu)
    cuda_sep = _sepset_from_out(cuda)
    same_sep = cpu_sep == cuda_sep
    same_sep_keys = set(cpu_sep) == set(cuda_sep)
    same_cg_sep = _sepsets_as_sets(cpu_cg.sepset) == _sepsets_as_sets(cuda_cg.sepset)
    print(
        f"sk_equal={same_sk} pMax_equal={same_pmax} "
        f"sepset_equal={same_sep} sepset_keys_equal={same_sep_keys} "
        f"cg_sepset_equal={same_cg_sep}"
    )
    if not same_sk:
        print("diff sk:\n", np.array(cpu["sk"]) - np.array(cuda["sk"]))
    if not same_pmax:
        d = np.abs(cpu["pMax"] - cuda["pMax"])
        print("pMax max abs diff", float(np.nanmax(d)))
    # Skeleton, packed-CSR sepset contents (as sets) and pMax within the
    # documented tolerance are the parity contract. ``cg.sepset`` is a
    # CPU-only summary view; the CUDA path keeps the same information in the
    # packed CSR, so it is not compared directly.
    return 0 if same_sk and same_sep and same_pmax else 1


if __name__ == "__main__":
    raise SystemExit(main())
