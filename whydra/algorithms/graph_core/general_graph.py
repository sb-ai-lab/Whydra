import numpy as np
from typing import List, Optional

from .nodes import Node
from .edges import Edge
from .endpoints import Endpoint


class GeneralGraph:
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes
        self.node_map = {node: i for i, node in enumerate(nodes)}
        self.num_vars = len(nodes)
        self.graph = np.zeros((self.num_vars, self.num_vars), dtype=int)

    def get_num_nodes(self):
        return self.num_vars

    def get_endpoint(self, node1: Node, node2: Node) -> Endpoint:
        """Return the endpoint mark at ``node1`` on the edge to ``node2``."""
        i, j = self.node_map[node1], self.node_map[node2]
        return Endpoint(int(self.graph[j, i]))
    def add_edge(self, edge: Edge):
        i = self.node_map[edge.node1]
        j = self.node_map[edge.node2]
        self.graph[i, j] = edge.endpoint2.value
        self.graph[j, i] = edge.endpoint1.value

    def remove_edge(self, edge: Edge):
        i = self.node_map[edge.node1]
        j = self.node_map[edge.node2]
        self.graph[i, j] = 0
        self.graph[j, i] = 0

    def get_edge(self, node1: Node, node2: Node) -> Optional[Edge]:
        i, j = self.node_map[node1], self.node_map[node2]
        if self.graph[i, j] == 0 and self.graph[j, i] == 0:
            return None
        return Edge(node1, node2, Endpoint(self.graph[j, i]), Endpoint(self.graph[i, j]))

    def has_edge(self, i: int, j: int) -> bool:
        """Single definition of "nodes i and j are adjacent", by index.

        An endpoint mark on either side means the edge exists. Checking only
        one side used to make `get_graph_edges` disagree with `is_adjacent_to`
        on a one-sided mark: the edge was visible to one API and invisible to
        the other.
        """
        return self.graph[i, j] != 0 or self.graph[j, i] != 0

    def is_adjacent_to(self, node1: Node, node2: Node) -> bool:
        return self.has_edge(self.node_map[node1], self.node_map[node2])

    def add_undirected_edge(self, node1: Node, node2: Node):
        self.add_edge(Edge(node1, node2, Endpoint.TAIL, Endpoint.TAIL))

    def add_directed_edge(self, source: Node, target: Node):
        self.add_edge(Edge(source, target, Endpoint.TAIL, Endpoint.ARROW))

    def add_bidirected_edge(self, node1: Node, node2: Node):
        self.add_edge(Edge(node1, node2, Endpoint.ARROW, Endpoint.ARROW))

    def get_adjacent_nodes(self, node: Node):
        idx = self.node_map[node]
        adj_indices = np.where((self.graph[idx, :] != 0) | (self.graph[:, idx] != 0))[0]
        return [self.nodes[i] for i in adj_indices]

    def get_nodes(self):
        return self.nodes

    def get_graph_edges(self):
        edges = []
        for i in range(self.num_vars):
            for j in range(i + 1, self.num_vars):
                if self.has_edge(i, j):
                    edges.append(self.get_edge(self.nodes[i], self.nodes[j]))
        return edges


__all__ = ["GeneralGraph"]
