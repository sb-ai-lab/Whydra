import numpy as np
from itertools import combinations
from typing import List

from .nodes import Node
from .edges import Edge
from .endpoints import Endpoint
from .general_graph import GeneralGraph


class CausalGraph:
    def __init__(self, no_of_var: int, node_names: List[str]):
        node_names = list(node_names)
        if len(node_names) != no_of_var:
            raise ValueError(
                f"no_of_var={no_of_var} does not match len(node_names)={len(node_names)}"
            )
        # Node equality and hashing go by name, and GeneralGraph keys its index
        # map on the nodes themselves. Two nodes sharing a name would collapse
        # into one entry, silently redirecting every operation on the first to
        # the index of the second and dropping edges without a word.
        duplicates = sorted({name for name in node_names if node_names.count(name) > 1})
        if duplicates:
            raise ValueError(f"node_names must be unique; duplicated: {duplicates}")

        self.nodes = [Node(name) for name in node_names]
        self.G = GeneralGraph(self.nodes)
        for i in range(no_of_var):
            for j in range(i + 1, no_of_var):
                self.G.add_edge(Edge(self.nodes[i], self.nodes[j], Endpoint.TAIL, Endpoint.TAIL))
        self.sepset = np.empty((no_of_var, no_of_var), object)
        self.test = None

    def is_adjacent_to(self, node1: Node, node2: Node) -> bool:
        return self.G.is_adjacent_to(node1, node2)

    def remove_edge(self, node1: Node, node2: Node):
        edge = self.G.get_edge(node1, node2)
        if edge:
            self.G.remove_edge(edge)

    def add_undirected_edge(self, node1: Node, node2: Node):
        self.G.add_undirected_edge(node1, node2)

    def add_directed_edge(self, source: Node, target: Node):
        self.G.add_directed_edge(source, target)

    def add_bidirected_edge(self, node1: Node, node2: Node):
        self.G.add_bidirected_edge(node1, node2)

    def set_ind_test(self, test_obj):
        self.test = test_obj

    def ci_test(self, i: int, j: int, S: tuple) -> float:
        return self.test(i, j, S)

    def neighbors(self, i: int):
        return np.where(self.G.graph[i, :] != 0)[0]

    def max_degree(self) -> int:
        return int(np.max(np.sum(self.G.graph != 0, axis=1)))

    def is_fully_directed(self, i, j):
        return self.G.graph[i, j] == Endpoint.ARROW.value and self.G.graph[j, i] == Endpoint.TAIL.value

    def is_undirected(self, i, j):
        return self.G.graph[i, j] == Endpoint.TAIL.value and self.G.graph[j, i] == Endpoint.TAIL.value

    def is_bidirected(self, i, j):
        return self.G.graph[i, j] == Endpoint.ARROW.value and self.G.graph[j, i] == Endpoint.ARROW.value

    def find_unshielded_triples(self):
        triples = []
        adj = self.G.graph
        num_vars = len(self.nodes)
        for j in range(num_vars):
            neighbors = np.where(adj[j, :] != 0)[0]
            if len(neighbors) < 2:
                continue
            for i, k in combinations(neighbors, 2):
                if not self.G.has_edge(i, k):
                    triples.append((i, j, k))
        return triples

    def find_triangles(self):
        triangles = []
        adj = self.G.graph
        num_vars = len(self.nodes)
        edges = []
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                if adj[i, j] != 0:
                    edges.append((i, j))
        for (i, j) in edges:
            neigh_i = set(np.where(adj[i, :] != 0)[0])
            neigh_j = set(np.where(adj[j, :] != 0)[0])
            common = neigh_i.intersection(neigh_j)
            for k in common:
                if k > j:
                    triangles.append((i, j, k))
        return triangles

    def find_kites(self):
        kites = []
        num_vars = len(self.nodes)
        for l in range(num_vars):
            neighbors_l = np.where(self.G.graph[l, :] != 0)[0]
            if len(neighbors_l) < 2:
                continue
            for j, k in combinations(neighbors_l, 2):
                if not self.G.has_edge(j, k):
                    neigh_j = set(np.where(self.G.graph[j, :] != 0)[0])
                    neigh_k = set(np.where(self.G.graph[k, :] != 0)[0])
                    neigh_l = set(neighbors_l)
                    common = neigh_j.intersection(neigh_k).intersection(neigh_l)
                    for i in common:
                        kites.append((i, j, k, l))
        return kites


__all__ = ["CausalGraph"]
