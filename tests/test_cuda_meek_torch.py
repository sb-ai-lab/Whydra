"""CUDA Meek vs original sequential Python Meek on the same post-collider graph."""

import contextlib
import io
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(_TESTS := ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(_TESTS))

from test_pc_stable_cuda import _dense, _fmt_diff, _sparse

from whydra.algorithms.graph_core import CIT, CausalGraph
from whydra.algorithms.graph_core.meek_torch import (
    meek_orient as meek_torch,
)
from whydra.algorithms.graph_core.meek_torch import (
    meek_orient_tensor,
)
from whydra.algorithms.standalone.pc_algo import PCAlgorithm
from whydra.algorithms.standalone.skeleton import SkeletonDiscovery


def _post_collider(data):
    names = [f"X{i}" for i in range(data.shape[1])]
    cg = CausalGraph(data.shape[1], names)
    cg.set_ind_test(CIT(data, method="fisherz"))
    sk = SkeletonDiscovery(cg, alpha=0.05, stable=True, n_jobs=1, show_progress=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sk.run()
        PCAlgorithm(
            cg, alpha=0.05, stable=True, n_jobs=1, show_progress=False
        ).orient_colliders()
    return np.array(cg.G.graph), cg


def _meek_case(data, tag: str) -> bool:
    g0, cg = _post_collider(data)
    names = [n.name for n in cg.nodes]
    cg_py = CausalGraph(len(names), names)
    cg_py.G.graph[:, :] = g0
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        PCAlgorithm(
            cg_py, alpha=0.05, stable=True, n_jobs=1, show_progress=False
        ).apply_meek_rules()
    g_py = np.array(cg_py.G.graph)
    g_th = g0.copy()
    meek_torch(g_th, device="cuda")
    th_ok = np.array_equal(g_py, g_th)
    print(f"{tag}: cuda={('equal' if th_ok else _fmt_diff(g_py, g_th))}")
    return th_ok


def _r2_cpu_only(cg) -> None:
    algo = PCAlgorithm(cg, alpha=0.05, stable=True, n_jobs=1, show_progress=False)
    loop = True
    while loop:
        loop = False
        triangles = cg.find_triangles()
        for i, j, k in triangles:
            for a, b, c in permutations([i, j, k], 3):
                if (
                    cg.is_fully_directed(a, b)
                    and cg.is_fully_directed(b, c)
                    and cg.is_undirected(a, c)
                ):
                    algo._orient_edge_conflict_aware(a, c)
                    loop = True


def _r2_only_case(data, tag: str) -> bool:
    g0, cg = _post_collider(data)
    names = [n.name for n in cg.nodes]
    cg_py = CausalGraph(len(names), names)
    cg_py.G.graph[:, :] = g0
    _r2_cpu_only(cg_py)
    g_py = np.array(cg_py.G.graph)
    t = torch.from_numpy(np.ascontiguousarray(g0, dtype=np.int8)).cuda()
    meek_orient_tensor(t, rules=2)
    g_th = t.cpu().numpy()
    th_ok = np.array_equal(g_py, g_th)
    print(f"{tag} R2-only: cuda={('equal' if th_ok else _fmt_diff(g_py, g_th))}")
    return th_ok


def _r3_cpu_only(cg) -> None:
    algo = PCAlgorithm(cg, alpha=0.05, stable=True, n_jobs=1, show_progress=False)
    loop = True
    while loop:
        loop = False
        kites = cg.find_kites()
        for i, j, k, leaf in kites:
            if (
                cg.is_fully_directed(j, leaf)
                and cg.is_fully_directed(k, leaf)
                and cg.is_undirected(i, leaf)
                and cg.is_undirected(i, j)
                and cg.is_undirected(i, k)
            ):
                algo._orient_edge_conflict_aware(i, leaf)
                loop = True


def _r3_only_case(data, tag: str) -> bool:
    g0, cg = _post_collider(data)
    names = [n.name for n in cg.nodes]
    cg_py = CausalGraph(len(names), names)
    cg_py.G.graph[:, :] = g0
    _r3_cpu_only(cg_py)
    g_py = np.array(cg_py.G.graph)
    t = torch.from_numpy(np.ascontiguousarray(g0, dtype=np.int8)).cuda()
    meek_orient_tensor(t, rules=4)
    g_th = t.cpu().numpy()
    th_ok = np.array_equal(g_py, g_th)
    print(f"{tag} R3-only: cuda={('equal' if th_ok else _fmt_diff(g_py, g_th))}")
    return th_ok


def _pack_kites_like_cuda(g: np.ndarray):
    """CUDA pack: outer l, combinadic (j,k) of N(l), then i in N(l) with adj j,k."""
    p = g.shape[0]
    kites = []
    for leaf in range(p):
        nbrs = np.where(g[leaf] != 0)[0]
        d = int(nbrs.size)
        for a in range(d):
            j = int(nbrs[a])
            for b in range(a + 1, d):
                k = int(nbrs[b])
                if g[j, k] != 0:
                    continue
                for c in range(d):
                    i = int(nbrs[c])
                    if i in (j, k):
                        continue
                    if g[i, j] != 0 and g[i, k] != 0:
                        kites.append((i, j, k, leaf))
    return kites


def _kite_order_case(data, tag: str) -> bool:
    g0, cg = _post_collider(data)
    py = [(int(i), int(j), int(k), int(leaf)) for i, j, k, leaf in cg.find_kites()]
    packed = _pack_kites_like_cuda(g0)
    ok = py == packed
    print(
        f"{tag} kite order: "
        f"{'equal' if ok else f'MISMATCH py={len(py)} pack={len(packed)}'}"
    )
    return ok


def main() -> int:
    ok = True
    ok = _meek_case(_dense(200, 12, seed=0), "p=12 dense") and ok
    ok = _meek_case(_sparse(1000, 50, seed=0), "p=50 sparse") and ok
    ok = _meek_case(_sparse(1000, 100, seed=0), "p=100 sparse") and ok
    ok = _r2_only_case(_dense(200, 12, seed=0), "p=12 dense") and ok
    ok = _r2_only_case(_sparse(1000, 50, seed=0), "p=50 sparse") and ok
    ok = _kite_order_case(_dense(200, 12, seed=0), "p=12 dense") and ok
    ok = _kite_order_case(_sparse(1000, 50, seed=0), "p=50 sparse") and ok
    ok = _r3_only_case(_dense(200, 12, seed=0), "p=12 dense") and ok
    ok = _r3_only_case(_sparse(1000, 50, seed=0), "p=50 sparse") and ok
    print("\nALL_OK" if ok else "\nMISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
