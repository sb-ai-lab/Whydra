"""RAI-stable-cuda graph primitives vs sequential helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whydra.algorithms.graph_core import CausalGraph, Endpoint
from whydra.algorithms.standalone.rai_algo import _PDAG, DIRECTED, UNDIRECTED


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not _cuda_available(), reason="CUDA required"),
]


def _need_cuda():
    import torch

    if not torch.cuda.is_available():
        print("skip: no CUDA")
        return None
    return torch


def _mixed_cpdag(p: int, seed: int = 0) -> np.ndarray:
    """Complete TAIL–TAIL graph with a few directed edges and deletions."""
    rng = np.random.default_rng(seed)
    G = np.full((p, p), Endpoint.TAIL.value, dtype=int)
    np.fill_diagonal(G, 0)
    for _ in range(max(1, p // 3)):
        i, j = rng.choice(p, size=2, replace=False)
        G[i, j] = Endpoint.ARROW.value
        G[j, i] = Endpoint.TAIL.value
    for _ in range(max(1, p // 4)):
        i, j = rng.choice(p, size=2, replace=False)
        G[i, j] = 0
        G[j, i] = 0
    return G


def _pdag_from_endpoints(G: np.ndarray) -> _PDAG:
    """Reference PDAG with the same parents/undirected neighbors as ``G``.

    ``j`` is a parent of ``i`` iff ``G[j, i] == ARROW`` and ``G[i, j] == TAIL``;
    undirected iff both ends are TAIL (the packing rule of
    ``cuda_rai.compact_potential_parents``).
    """
    p = G.shape[0]
    pdag = _PDAG(set(range(p)))
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if G[j, i] == Endpoint.ARROW.value and G[i, j] == Endpoint.TAIL.value:
                pdag._graph[i][DIRECTED].add(j)
            elif G[i, j] == Endpoint.TAIL.value and G[j, i] == Endpoint.TAIL.value:
                pdag._graph[i][UNDIRECTED].add(j)
    return pdag


def _upload_ep(G: np.ndarray, torch):
    from whydra.algorithms.graph_core.cuda_rai import compact_potential_parents

    t = torch.as_tensor(np.ascontiguousarray(G, dtype=np.int8), device="cuda")
    return compact_potential_parents(t)


def test_compact_potential_parents_order():
    torch = _need_cuda()
    p = 12
    G = _mixed_cpdag(p, seed=1)
    pdag = _pdag_from_endpoints(G)
    nbrs, deg, G_pp = _upload_ep(G, torch)
    nbrs_np = nbrs.cpu().numpy()
    deg_np = deg.cpu().numpy()
    gpp = G_pp.cpu().numpy()
    for i in range(p):
        parents = sorted(pdag.parents(i))
        und = sorted(pdag.undirected_neighbors(i))
        want = parents + [u for u in und if u not in parents]
        got = [int(v) for v in nbrs_np[i, : int(deg_np[i])]]
        assert got == want, f"node {i}: {got} != {want}"
        for j in range(p):
            expect = 1 if j in set(want) else 0
            assert int(gpp[i, j]) == expect


def test_pair_masks_and_filter():
    torch = _need_cuda()
    from whydra.algorithms.graph_core.cuda_rai import (
        filter_pairs,
        pair_mask_endogenous,
        pair_mask_exogenous,
    )

    p = 6
    tasks = [({0, 1, 2}, {3, 4}), ({5}, set())]
    endo = pair_mask_endogenous(tasks, p, "cuda")
    exo = pair_mask_exogenous(tasks, p, "cuda")
    endo_np = endo.cpu().numpy()
    exo_np = exo.cpu().numpy()
    assert endo_np[0, 1]
    assert endo_np[1, 2]
    assert not endo_np[0, 5]
    assert not endo_np[5, 5]
    assert exo_np[0, 3]
    assert exo_np[2, 4]
    assert not exo_np[3, 0]
    px = torch.tensor([0, 0, 1], dtype=torch.int32, device="cuda")
    py = torch.tensor([1, 3, 5], dtype=torch.int32, device="cuda")
    fx, fy = filter_pairs(px, py, endo)
    assert fx.tolist() == [0]
    assert fy.tolist() == [1]
    fx, fy = filter_pairs(px, py, exo)
    assert fx.tolist() == [0]
    assert fy.tolist() == [3]


def test_lowest_topo_and_exit():
    # Follow the IntelLabs RAI reference (_get_lowest_topological_set). With
    # multiple layers the descendants include every endogenous node outside
    # the topmost layer, including intermediate nodes as well as leaves.
    torch = _need_cuda()
    from whydra.algorithms.graph_core.cuda_rai import (
        exit_cond,
        lowest_topo,
        partial_topological_layers,
    )

    p = 4
    G = np.full((p, p), Endpoint.TAIL.value, dtype=np.int8)
    np.fill_diagonal(G, 0)
    # 0 → 1 → 2, 3 undirected with 2
    G[0, 1] = Endpoint.ARROW.value
    G[1, 0] = Endpoint.TAIL.value
    G[1, 2] = Endpoint.ARROW.value
    G[2, 1] = Endpoint.TAIL.value
    t = torch.as_tensor(G, device="cuda")
    en = torch.tensor([True, True, True, False], device="cuda")
    # layers within {0,1,2}: leaves {2}, then {1}, topmost {0}
    layers = [
        {int(v) for v in layer.cpu().numpy().nonzero()[0]}
        for layer in partial_topological_layers(t, en)
    ]
    assert layers == [{2}, {1}, {0}], layers
    # d = en - topmost = {1, 2}
    desc = lowest_topo(t, en).cpu().numpy()
    assert list(np.where(desc)[0]) == [1, 2]

    # single layer (directed cycle inside en) -> d = en
    G2 = np.zeros((3, 3), dtype=np.int8)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        G2[i, j] = Endpoint.ARROW.value
        G2[j, i] = Endpoint.TAIL.value
    t2 = torch.as_tensor(G2, device="cuda")
    en2 = torch.tensor([True, True, True], device="cuda")
    desc2 = lowest_topo(t2, en2).cpu().numpy()
    assert list(np.where(desc2)[0]) == [0, 1, 2]

    deg = torch.tensor([1, 2, 1, 0], dtype=torch.int32, device="cuda")
    en_idx = torch.tensor([0, 1, 2], dtype=torch.int64, device="cuda")
    assert exit_cond(deg, en_idx, 2)
    assert not exit_cond(deg, en_idx, 1)


def test_meek_orient_tensor_matches_host():
    torch = _need_cuda()
    from whydra.algorithms.graph_core.meek_torch import meek_orient_tensor
    from whydra.algorithms.standalone.standalone_pc_stable import apply_meek_rules

    G = _mixed_cpdag(10, seed=2).astype(np.int8)
    p = G.shape[0]
    names = [f"X{i}" for i in range(p)]
    cg = CausalGraph(p, names)
    cg.G.graph[:, :] = G
    apply_meek_rules(cg)
    t = torch.as_tensor(G.copy(), device="cuda")
    meek_orient_tensor(t)
    assert np.array_equal(t.cpu().numpy(), np.asarray(cg.G.graph))


def main() -> int:
    try:
        import torch
    except ImportError:
        print("skip: torch not installed")
        return 2
    if not torch.cuda.is_available():
        print("skip: no CUDA")
        return 2
    tests = [
        test_compact_potential_parents_order,
        test_pair_masks_and_filter,
        test_lowest_topo_and_exit,
        test_meek_orient_tensor_matches_host,
    ]
    for fn in tests:
        fn()
        print(f"ok {fn.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
