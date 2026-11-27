"""Export Whydra endpoint matrices to causal-learn.

Whydra stores the endpoint at node ``j`` in ``G[i, j]``. Causal-learn
stores it in ``graph[j, i]``. In the exported ``dpath``, ``dpath[j, i] == 1``
means ``i`` is an ancestor of ``j``, including ``i == j``.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    from causallearn.graph.GeneralGraph import GeneralGraph as CLGeneralGraph


def _cl_dpath_from_cl_graph(g: "torch.Tensor") -> np.ndarray:
    import torch

    reach = ((g == 1) & (g.mT == -1)).to(dtype=torch.float32)
    reach.fill_diagonal_(1.0)
    # Squaring doubles the path length. Reachability may require p - 1 edges.
    steps = max(0, int(reach.size(0)) - 2).bit_length()
    for _ in range(steps):
        reach = torch.sign(reach @ reach)
    torch.cuda.synchronize(g.device)
    return (reach > 0).to(dtype=torch.int64).cpu().numpy()


def _cl_dpath_from_endpoint_G(G: "torch.Tensor") -> np.ndarray:
    g = G.mT.contiguous()
    return _cl_dpath_from_cl_graph(g)


def _cl_dpath(graph: np.ndarray) -> np.ndarray:
    reach = ((graph == 1) & (graph.T == -1)).astype(np.float32)
    np.fill_diagonal(reach, 1.0)
    steps = max(0, int(graph.shape[0]) - 2).bit_length()
    for _ in range(steps):
        reach = (reach @ reach > 0).astype(np.float32)
    return (reach > 0).astype(np.int64)


def to_causallearn_graph(
    G: np.ndarray, names, dpath: np.ndarray | None = None
) -> "CLGeneralGraph":
    """Export ``G`` and compute ancestor reachability if ``dpath`` is omitted."""
    from causallearn.graph.GeneralGraph import GeneralGraph as CLGeneralGraph
    from causallearn.graph.GraphNode import GraphNode as CLGraphNode

    G = np.asarray(G)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        msg = "G must be square"
        raise ValueError(msg)
    p = int(G.shape[0])
    cl = CLGeneralGraph.__new__(CLGeneralGraph)
    cl.nodes = [CLGraphNode(str(n)) for n in names]
    cl.num_vars = p
    cl.node_map = {node: i for i, node in enumerate(cl.nodes)}
    cl.graph = np.ascontiguousarray(G.T, dtype=int)
    cl.dpath = dpath if dpath is not None else _cl_dpath(cl.graph)
    cl.ambiguous_triples = []
    cl.underline_triples = []
    cl.dotted_underline_triples = []
    cl.attributes = {}
    cl.pattern = False
    cl.pag = False
    return cl


def causal_graph_to_causallearn(cg) -> "CLGeneralGraph":
    """Compute ancestor reachability on CUDA when the oriented graph is resident."""
    names = (node.name for node in cg.nodes)
    G_t = getattr(cg, "_G_cuda", None)
    if getattr(cg, "_cuda_endpoints", False) and G_t is not None:
        import torch

        if torch.is_tensor(G_t) and G_t.is_cuda:
            dpath = _cl_dpath_from_endpoint_G(G_t)
            return to_causallearn_graph(cg.G.graph, names, dpath=dpath)
    return to_causallearn_graph(cg.G.graph, names)


__all__ = [
    "_cl_dpath_from_endpoint_G",
    "causal_graph_to_causallearn",
    "to_causallearn_graph",
]
