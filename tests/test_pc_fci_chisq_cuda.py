"""CUDA PC/FCI with host χ² vs sequential ``CIT.chi_square``."""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whydra.algorithms.graph_core import CIT, CausalGraph
from whydra.algorithms.standalone.fci_algo_v2 import _sepset_as_pairs, fci_stable_v2
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


def _discrete(n, p, seed=0, n_states=3, max_parents=2):
    rng = np.random.default_rng(seed)
    data = np.zeros((n, p), dtype=np.int64)
    for j in range(p):
        m = min(max_parents, j)
        if m == 0:
            data[:, j] = rng.integers(0, n_states, size=n)
            continue
        parents = rng.choice(j, size=m, replace=False)
        mix = data[:, parents].sum(axis=1)
        data[:, j] = (mix + rng.integers(0, n_states, size=n)) % n_states
    return data


def _run_pc(data, *, device):
    names = [f"X{i}" for i in range(data.shape[1])]
    cg = CausalGraph(data.shape[1], names, device=device)
    cg.set_ind_test(CIT(data, method="chisq"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pc_stable(cg, alpha=0.05, n_jobs=1, show_progress=False, device=device)
    return np.array(cg.G.graph), _sepsets_as_sets(cg)


def _run_fci(data, *, device):
    names = [f"X{i}" for i in range(data.shape[1])]
    cg = CausalGraph(data.shape[1], names, device=device)
    cg.set_ind_test(CIT(data, method="chisq"))
    pag = fci_stable_v2(
        cg,
        data,
        alpha=0.05,
        indep_test="chisq",
        n_jobs=1,
        device=device,
        show_progress=False,
    )
    return pag, _sepset_as_pairs(getattr(fci_stable_v2, "last_sepset", {}))


def main() -> int:
    data = _discrete(400, 8, seed=0, n_states=3, max_parents=2)
    print(f"discrete n={data.shape[0]} p={data.shape[1]} states=3")

    pc_cpu_g, pc_cpu_s = _run_pc(data, device=None)
    pc_cu_g, pc_cu_s = _run_pc(data, device="cuda")
    pc_ok = np.array_equal(pc_cpu_g, pc_cu_g) and pc_cpu_s == pc_cu_s
    print(
        f"  PC chisq cuda-host vs seq: cpdag={np.array_equal(pc_cpu_g, pc_cu_g)} "
        f"sepset={pc_cpu_s == pc_cu_s}"
    )

    fci_cpu_g, fci_cpu_s = _run_fci(data, device=None)
    fci_cu_g, fci_cu_s = _run_fci(data, device="cuda")
    fci_ok = np.array_equal(fci_cpu_g, fci_cu_g) and fci_cpu_s == fci_cu_s
    print(
        f"  FCI chisq cuda-host vs seq: pag={np.array_equal(fci_cpu_g, fci_cu_g)} "
        f"sepset={fci_cpu_s == fci_cu_s}"
    )
    return 0 if pc_ok and fci_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
