"""
Parse, convert, and visualize graphs.

Convert between three representations: GeneralGraph, endpoint matrices, and
NetworkX DiGraph.
"""

import os
import re
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    import networkx as nx
else:

    class _NetworkXNamespace:
        """Resolve public return annotations without eagerly importing NetworkX."""

        def __getattr__(self, name):
            from importlib import import_module

            module = import_module("networkx")
            globals()["nx"] = module
            return getattr(module, name)

    nx = _NetworkXNamespace()
from whydra.algorithms.graph_core import Edge, Endpoint, GeneralGraph
from whydra.algorithms.graph_core import Node as GraphNode
from whydra.background_knowledge import MatrixEncoding, endpoint_codes


def default_node_names(n: int) -> list:
    """Canonical variable names for a graph of ``n`` nodes: ``X1 … Xn``.

    Estimates and ground-truth graphs use the same one-based names so
    their nodes can be matched by name.
    """
    return [f"X{i + 1}" for i in range(n)]


def endpoint_lookup(encoding) -> dict:
    """Map raw endpoint codes of ``encoding`` onto local endpoint values."""
    null, tail, arrow, circle = endpoint_codes(encoding)
    return {
        null: Endpoint.NULL,
        tail: Endpoint.TAIL,
        arrow: Endpoint.ARROW,
        circle: Endpoint.CIRCLE,
    }


# ============================================================
#  ENDPOINT MATRICES AND GENERAL GRAPH
# ============================================================


def get_adj_matrix(g: GeneralGraph) -> np.ndarray:
    """Convert a GeneralGraph to an endpoint matrix.

    mat[i, j] is the mark at node j on the edge between i and j.
    The codes are TAIL=-1, NULL=0, ARROW=1, and CIRCLE=2.
    """
    nodes = g.get_nodes()
    n = len(nodes)
    node_map = {node.get_name(): i for i, node in enumerate(nodes)}

    mat = np.zeros((n, n), dtype=int)

    for edge in g.get_graph_edges():
        i = node_map[edge.get_node1().get_name()]
        j = node_map[edge.get_node2().get_name()]

        mat[i, j] = edge.get_endpoint2().value  # endpoint at j
        mat[j, i] = edge.get_endpoint1().value  # endpoint at i

    return mat


def adj_matrix_to_graph(adj_mat: np.ndarray, node_names: list) -> GeneralGraph:
    """Convert an endpoint matrix back to a GeneralGraph.

    Round floating-point marks, such as values averaged during bagging.
    """
    import warnings

    from ..algorithms.graph_core.validation import (
        validate_endpoint_matrix,
        validate_node_names,
    )

    node_names = validate_node_names(node_names, len(node_names))
    raw = np.asarray(adj_mat)
    if not np.array_equal(raw, np.rint(raw)):
        warnings.warn(
            "Rounding numeric endpoint codes is deprecated; use endpoint voting for consensus",
            DeprecationWarning,
            stacklevel=2,
        )
    adj_mat = validate_endpoint_matrix(np.rint(raw), len(node_names))
    nodes = [GraphNode(name) for name in node_names]
    g = GeneralGraph(nodes)
    n = len(nodes)

    for i in range(n):
        for j in range(i + 1, n):
            val_j = adj_mat[i, j]  # endpoint at j
            val_i = adj_mat[j, i]  # endpoint at i

            ep_j_val = round(val_j)
            ep_i_val = round(val_i)

            if ep_j_val == 0 and ep_i_val == 0:
                continue

            try:
                ep1 = Endpoint(ep_i_val)
                ep2 = Endpoint(ep_j_val)
            except ValueError:
                continue

            edge = Edge(nodes[i], nodes[j], ep1, ep2)
            g.add_edge(edge)

    return g


# ============================================================
#  GENERAL GRAPH → NETWORKX
# ============================================================


def convert_general_graph_to_nx(graph: GeneralGraph) -> "nx.DiGraph":
    """Export all nine endpoint pairs, retaining marks on each directed arc."""
    import networkx as nx

    from ..algorithms.graph_core.validation import validate_endpoint_matrix

    validate_endpoint_matrix(graph.graph, len(graph.nodes))
    result = nx.DiGraph(graph_kind=getattr(graph, "graph_kind", None))
    result.add_nodes_from(node.get_name() for node in graph.get_nodes())
    for edge in graph.get_graph_edges():
        u, v = edge.node1.get_name(), edge.node2.get_name()
        a, b = edge.endpoint1.value, edge.endpoint2.value

        def add(source, target, left, right, kind, **attrs):
            result.add_edge(
                source, target, type=kind, endpoint_u=left, endpoint_v=right, **attrs
            )

        if (a, b) in [(-1, 1), (2, 1)]:
            add(u, v, a, b, "directed", **({"style": "circle"} if a == 2 else {}))
        elif (a, b) in [(1, -1), (1, 2)]:
            add(v, u, b, a, "directed", **({"style": "circle"} if b == 2 else {}))
        else:
            kind = (
                "bidirected" if a == b == 1 else ("undirected" if a == b else "partial")
            )
            attrs = {"style": "circle"} if 2 in (a, b) else {}
            add(u, v, a, b, kind, **attrs)
            add(v, u, b, a, kind, **attrs)
    return result


def from_networkx(graph):
    """Read endpoint attributes from Whydra's NetworkX adapter."""
    names = list(graph.nodes)
    index = {name: i for i, name in enumerate(names)}
    matrix = np.zeros((len(names), len(names)), dtype=int)
    for u, v, attrs in graph.edges(data=True):
        if "endpoint_u" not in attrs or "endpoint_v" not in attrs:
            msg = "NetworkX edges need explicit endpoint_u/endpoint_v attributes"
            raise ValueError(msg)
        i, j = index[u], index[v]
        a, b = int(attrs["endpoint_u"]), int(attrs["endpoint_v"])
        if matrix[i, j] and (matrix[i, j] != b or matrix[j, i] != a):
            msg = "Conflicting endpoint attributes on reverse arcs"
            raise ValueError(msg)
        matrix[i, j], matrix[j, i] = b, a
    result = adj_matrix_to_graph(matrix, names)
    result.graph_kind = graph.graph.get("graph_kind")
    return result


def from_causallearn(graph, *, graph_kind=None):
    """Convert a causal-learn graph to a Whydra graph.

    Transpose the endpoint matrix because causal-learn stores the mark at the
    row node and Whydra stores the mark at the column node. Preserve node names
    and use the supplied graph kind or, by default, the input graph's kind.
    """
    result = adj_matrix_to_graph(
        np.asarray(graph.graph).T, [node.get_name() for node in graph.get_nodes()]
    )
    result.graph_kind = graph_kind or getattr(graph, "graph_kind", None)
    return result


def to_causallearn(graph):
    """Convert a Whydra graph to a graph from the optional causal-learn package.

    Copy node names and the graph kind. Transpose the endpoint matrix so each
    mark refers to the row node, as causal-learn expects.
    """
    from causallearn.graph.GeneralGraph import GeneralGraph as ExternalGraph
    from causallearn.graph.GraphNode import GraphNode as ExternalNode

    from ..algorithms.graph_core.validation import validate_endpoint_matrix

    matrix = validate_endpoint_matrix(graph.graph, len(graph.nodes))
    result = ExternalGraph(
        [ExternalNode(node.get_name()) for node in graph.get_nodes()]
    )
    result.graph = matrix.T.copy()
    # Preserve Whydra metadata on the third-party graph for a round trip.
    vars(result)["graph_kind"] = getattr(graph, "graph_kind", None)
    return result


def pag_matrix_to_general_graph(
    matrix: np.ndarray,
    node_names: list,
    encoding: MatrixEncoding = MatrixEncoding.STANDARD,
) -> GeneralGraph:
    """
    Convert a PAG matrix to a GeneralGraph.

    The default causal-learn encoding used by Whydra is 0 for no edge,
    -1 for a tail, 1 for an arrowhead, and 2 for a circle. For external
    matrices, such as those from R's pcalg, select the pcalg encoding:
    0 for no edge, 1 for a circle, 2 for an arrowhead, and 3 for a tail.
    """
    endpoint_map = endpoint_lookup(encoding)

    nodes = [GraphNode(name) for name in node_names]
    g = GeneralGraph(nodes)
    n = len(nodes)

    for i in range(n):
        for j in range(i + 1, n):
            val_j = int(matrix[i, j])
            val_i = int(matrix[j, i])
            if val_j != 0 or val_i != 0:
                try:
                    end_i, end_j = endpoint_map[val_i], endpoint_map[val_j]
                except KeyError as exc:
                    msg = (
                        f"Endpoint code {exc.args[0]!r} at ({i}, {j}) is not valid for "
                        f"encoding {MatrixEncoding(encoding).value!r}; "
                        "the matrix is probably in the other encoding"
                    )
                    raise ValueError(msg) from None
                g.add_edge(Edge(nodes[i], nodes[j], end_i, end_j))

    return g


def result_to_nx(
    result,
    node_names: list,
    encoding: MatrixEncoding = MatrixEncoding.STANDARD,
) -> "nx.DiGraph":
    """
    Convert an algorithm result to a NetworkX DiGraph.

    Accept a graph, a result whose ``.G`` holds a graph, or a 2D endpoint
    matrix. For a matrix, ``node_names`` supplies the labels in row order
    and ``encoding`` specifies how the matrix represents endpoint marks.
    """
    if isinstance(result, np.ndarray) and result.ndim == 2:
        g = pag_matrix_to_general_graph(result, node_names, encoding)
        return convert_general_graph_to_nx(g)

    graph_obj = getattr(result, "G", result)

    if hasattr(graph_obj, "get_nodes") and hasattr(graph_obj, "get_graph_edges"):
        graph_obj = cast("GeneralGraph", graph_obj)
        return convert_general_graph_to_nx(graph_obj)

    msg = f"Could not convert a result of type {type(result)} to NetworkX"
    raise TypeError(msg)


# ============================================================
#  PARSING GROUND-TRUTH GRAPHS
# ============================================================


def parse_ground_truth_graph(file_path: str) -> GeneralGraph:
    """Read the ground-truth graph file and return a GeneralGraph."""
    with open(file_path) as f:
        content = f.read()

    nodes_match = re.search(r"Graph Nodes:\n(.*?)\n\n", content, re.DOTALL)
    if not nodes_match:
        nodes_match = re.search(
            r"Graph Nodes:\n(.*?)\nGraph Edges:", content, re.DOTALL
        )

    if not nodes_match:
        msg = f"Could not parse nodes from {file_path}"
        raise ValueError(msg)

    nodes_str = nodes_match.group(1).replace("\n", "").strip()
    splitter = ";" if ";" in nodes_str else ","
    node_names = [name.strip() for name in nodes_str.split(splitter) if name.strip()]

    nodes = [GraphNode(name) for name in node_names]
    g = GeneralGraph(nodes)
    node_map = {node.get_name(): node for node in nodes}

    edges_match = re.search(r"Graph Edges:\n(.*?)$", content, re.DOTALL)
    if edges_match:
        edge_lines = edges_match.group(1).strip().split("\n")
        for line in edge_lines:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue

            node1_name = parts[1]
            arrow_str = parts[2]
            node2_name = parts[3]

            if node1_name not in node_map or node2_name not in node_map:
                continue

            def char_to_endpoint(c):
                if c == "-":
                    return Endpoint.TAIL
                if c == ">" or c == "<":
                    return Endpoint.ARROW
                if c == "o":
                    return Endpoint.CIRCLE
                return Endpoint.TAIL

            end1 = char_to_endpoint(arrow_str[0])
            end2 = char_to_endpoint(arrow_str[-1])

            edge = Edge(node_map[node1_name], node_map[node2_name], end1, end2)
            g.add_edge(edge)

    return g


def load_ground_truth(path: str) -> GeneralGraph:
    """Read a ground-truth graph from the given path."""
    if not os.path.exists(path):
        msg = f"Ground truth file not found: {path}"
        raise FileNotFoundError(msg)

    if path.endswith(".npy"):
        adj_matrix = np.load(path)
        n_nodes = adj_matrix.shape[0]
        nodes = [GraphNode(name) for name in default_node_names(n_nodes)]
        g = GeneralGraph(nodes)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_matrix[i, j] != 0:
                    edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW)
                    g.add_edge(edge)
        return g

    return parse_ground_truth_graph(path)


def draw_graph(graph: GeneralGraph, output_dir: str, filename: str):
    """Save the graph as a PNG image."""
    try:
        import networkx as nx

        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, f"{filename}.png")
        nx_graph = convert_general_graph_to_nx(graph)
        nx.nx_pydot.to_pydot(nx_graph).write(full_path, format="png")
        return full_path
    except Exception as e:
        msg = f"Could not save graph visualization; install whydra[viz] and system Graphviz: {e}"
        raise RuntimeError(msg) from e
