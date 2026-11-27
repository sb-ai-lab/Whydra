"""Copy an endpoint matrix into a causallearn ``GeneralGraph`` without ``add_edge``.

Whydra stores the mark at j in ``G[i, j]``. Causal-learn stores the mark
at i in ``cl.graph[i, j]``, so the exporter transposes G. Directed ``i → j`` is
``G[i, j] == 1`` and ``G[j, i] == -1``, which is ``cl.graph[j, i] == 1`` and
``cl.graph[i, j] == -1``. Undirected / bidirected cells are symmetric, so the
transpose is a no-op there.

The exporter builds the ancestor matrix ``dpath`` by repeated squaring on
the CPU or GPU. For n nodes, this detects directed paths with at most
``2 ** max(1, n.bit_length() - 1)`` edges, including zero-length paths.
Longer paths can be missed. Sequential PC-stable uses this matrix only for
export. The exporter in ``cpu_graph_convert`` covers all path lengths.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    from causallearn.graph.GeneralGraph import GeneralGraph as CLGeneralGraph
    from causallearn.graph.Node import Node as CLNode


def _cl_dpath_from_cl_graph(g: "torch.Tensor") -> np.ndarray:
    """Find directed paths up to this module's length limit on the GPU.

    In causal-learn layout, i points to j when ``g[j, i] == 1`` and
    ``g[i, j] == -1``. Square the adjacency matrix on the current stream,
    then copy the result to the host once. Entry [j, i] is 1 for a detected
    path from i to j, including i itself.
    """
    import torch

    reach = ((g == 1) & (g.mT == -1)).to(dtype=torch.float32)
    reach.fill_diagonal_(1.0)
    steps = max(1, int(reach.size(0)).bit_length() - 1)
    for _ in range(steps):
        reach = torch.sign(reach @ reach)
    torch.cuda.synchronize(g.device)
    return (reach > 0).to(dtype=torch.int64).cpu().numpy()


def _cl_dpath_from_endpoint_G(G: "torch.Tensor") -> np.ndarray:
    """Find directed paths up to this module's length limit in a Whydra matrix.

    Directed ``i → j`` is ``G[i, j] == 1`` and ``G[j, i] == -1``. No H2D of ``G``.
    """
    g = G.mT.contiguous()
    return _cl_dpath_from_cl_graph(g)


def _cl_dpath(graph: np.ndarray) -> np.ndarray:
    """Find directed paths up to this module's length limit on the CPU.

    In causal-learn layout, i points to j when ``graph[j, i] == 1`` and
    ``graph[i, j] == -1``. Entry [j, i] is 1 for a detected path from i
    to j, including i itself. The CUDA counterpart uses the same bound.
    """
    reach = ((graph == 1) & (graph.T == -1)).astype(np.float32)
    np.fill_diagonal(reach, 1.0)
    steps = max(1, int(graph.shape[0]).bit_length() - 1)
    for _ in range(steps):
        reach = (reach @ reach > 0).astype(np.float32)
    return (reach > 0).astype(np.int64)


def to_causallearn_graph(
    G: np.ndarray, names, dpath: np.ndarray | None = None
) -> "CLGeneralGraph":
    """Export G by copying its integer transpose into a causal-learn graph.

    Pass a precomputed ancestor matrix as ``dpath`` to use it directly.
    Otherwise the exporter computes paths on the CPU up to this module's
    length limit.
    """
    from causallearn.graph.GeneralGraph import GeneralGraph as CLGeneralGraph
    from causallearn.graph.GraphNode import GraphNode as CLGraphNode

    G = np.asarray(G)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        msg = "G must be square"
        raise ValueError(msg)
    p = int(G.shape[0])
    nodes: list[CLNode] = [CLGraphNode(str(n)) for n in names]
    cl = CLGeneralGraph.__new__(CLGeneralGraph)
    cl.nodes = nodes
    cl.num_vars = p
    cl.node_map = {node: i for i, node in enumerate(nodes)}
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
    """Export the graph's endpoints and length-limited ancestor matrix.

    After CUDA orientation, compute ``dpath`` from the endpoint matrix
    already on the device. Otherwise compute it from ``cg.G.graph`` on
    the CPU. Both paths use the same length limit.
    """
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
