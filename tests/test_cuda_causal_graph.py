"""CausalGraph setup: sequential Edge loop vs CUDA torch complete adj."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whydra.algorithms.graph_core import CausalGraph, Endpoint
from whydra.algorithms.graph_core.edges import Edge
from whydra.algorithms.graph_core.general_graph import GeneralGraph
from whydra.algorithms.graph_core.nodes import Node


def _edge_loop_graph(p: int):
    names = [f"X{i}" for i in range(p)]
    nodes = [Node(n) for n in names]
    g = GeneralGraph(nodes)
    for i in range(p):
        for j in range(i + 1, p):
            g.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))
    return names, g


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_complete_is_torch_not_numpy():
    from whydra.algorithms.graph_core.cuda_graph_batch import complete_undirected

    p = 40
    names, old = _edge_loop_graph(p)
    want = (old.graph != 0).astype(np.int8)
    G_t = complete_undirected(p, "cuda")
    torch.cuda.synchronize()
    assert G_t.is_cuda
    assert G_t.dtype == torch.int8
    assert np.array_equal(G_t.cpu().numpy(), want)

    cg = CausalGraph(p, names, device="cuda")
    assert cg._G_cuda is not None
    assert np.all(cg.G.graph == 0)
    assert np.array_equal(cg._G_cuda.cpu().numpy(), want)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    test_cuda_complete_is_torch_not_numpy()
    print("ok")
