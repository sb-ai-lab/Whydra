from typing import Any

import numpy as np

from .edges import Edge
from .endpoints import Endpoint
from .nodes import Node


class GeneralGraph:
    """A graph of named nodes with endpoint marks stored in a square matrix."""

    def __init__(self, nodes: list[Node], *, graph_kind: str | None = None):
        self.nodes = list(nodes)
        names = [node.get_name() for node in self.nodes]
        if len(names) != len(set(names)):
            msg = "Node names must be unique"
            raise ValueError(msg)
        if graph_kind not in (None, "dag", "cpdag", "pag"):
            msg = "graph_kind must be dag, cpdag, or pag"
            raise ValueError(msg)
        self.graph_kind: str | None = graph_kind
        self.metadata: dict[str, Any] = {}
        self.node_map = {node: i for i, node in enumerate(nodes)}
        self.num_vars = len(nodes)
        self.graph = np.zeros((self.num_vars, self.num_vars), dtype=int)

    def get_num_nodes(self):
        """Return the number of nodes in the graph."""
        return self.num_vars

    def get_endpoint(self, node1: Node, node2: Node) -> Endpoint:
        """Return the endpoint mark at ``node1`` on the edge to ``node2``."""
        i, j = self.node_map[node1], self.node_map[node2]
        return Endpoint(int(self.graph[j, i]))

    def get_endpoint_at(self, node: Node, other: Node) -> Endpoint:
        """Return the mark at ``node`` on its edge to ``other``, as in ``get_endpoint``."""
        return self.get_endpoint(node, other)

    def add_edge(self, edge: Edge):
        """Store both marks of an edge in the endpoint matrix."""
        i = self.node_map[edge.node1]
        j = self.node_map[edge.node2]
        self.graph[i, j] = edge.endpoint2.value
        self.graph[j, i] = edge.endpoint1.value

    def remove_edge(self, edge: Edge):
        """Clear both endpoint marks between the edge's nodes."""
        i = self.node_map[edge.node1]
        j = self.node_map[edge.node2]
        self.graph[i, j] = 0
        self.graph[j, i] = 0

    def get_edge(self, node1: Node, node2: Node) -> Edge | None:
        """Return the edge between two nodes, or None when they are not adjacent."""
        i, j = self.node_map[node1], self.node_map[node2]
        if self.graph[i, j] == 0 and self.graph[j, i] == 0:
            return None
        return Edge(
            node1, node2, Endpoint(self.graph[j, i]), Endpoint(self.graph[i, j])
        )

    def has_edge(self, i: int | np.integer, j: int | np.integer) -> bool:
        """Check whether nodes with indices i and j share an edge.

        A mark at either endpoint means an edge exists. ``get_graph_edges`` and
        ``is_adjacent_to`` use this same test for one-sided marks.
        """
        return self.graph[i, j] != 0 or self.graph[j, i] != 0

    def is_adjacent_to(self, node1: Node, node2: Node) -> bool:
        """Check whether the graph contains an edge between two nodes."""
        return self.has_edge(self.node_map[node1], self.node_map[node2])

    def add_undirected_edge(self, node1: Node, node2: Node):
        """Connect two nodes with a tail at each end."""
        self.add_edge(Edge(node1, node2, Endpoint.TAIL, Endpoint.TAIL))

    def add_directed_edge(self, source: Node, target: Node):
        """Add an edge with a tail at the source and an arrowhead at the target."""
        self.add_edge(Edge(source, target, Endpoint.TAIL, Endpoint.ARROW))

    def add_bidirected_edge(self, node1: Node, node2: Node):
        """Connect two nodes with an arrowhead at each end."""
        self.add_edge(Edge(node1, node2, Endpoint.ARROW, Endpoint.ARROW))

    def get_adjacent_nodes(self, node: Node):
        """Return the neighbors of a node in matrix order."""
        idx = self.node_map[node]
        adj_indices = np.where((self.graph[idx, :] != 0) | (self.graph[:, idx] != 0))[0]
        return [self.nodes[i] for i in adj_indices]

    def get_nodes(self):
        """Return the graph's node list in matrix order."""
        return self.nodes

    def get_graph_edges(self):
        """Return an edge for each adjacent pair of distinct nodes."""
        edges = []
        for i in range(self.num_vars):
            for j in range(i + 1, self.num_vars):
                if self.has_edge(i, j):
                    edges.append(self.get_edge(self.nodes[i], self.nodes[j]))
        return edges


__all__ = ["GeneralGraph"]
