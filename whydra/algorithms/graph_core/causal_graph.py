"""Graph container shared by the CPU and CUDA search paths."""

from collections.abc import Callable, Iterable
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np

from . import edges
from .conditioning import conditioning_allowed
from .endpoints import Endpoint
from .general_graph import GeneralGraph
from .nodes import Node

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    from ...background_knowledge import BKOrientationGuard
    from ._optional_imports import _ParallelRunner
    from .independence_tests import CIT


class CausalGraph:
    """Store the endpoint matrix and separating sets for PC and FCI.

    On the CPU, the initial matrix has a tail at each end of every edge and
    zeros on the diagonal. On CUDA, the initial adjacency is a binary int8
    matrix on the device. The host matrix stays empty until the skeleton
    search writes the surviving edges.
    """

    # Set by the search phase that owns the corresponding diagnostics or resources.
    ci_stats: dict[str, Any]
    execution_stats: dict[str, Any]
    bk_guard: "BKOrientationGuard"
    pMax: np.ndarray
    _cuda_endpoints: bool
    _fci_plus_process_min_tasks: int
    _fci_plus_parallel_phase: dict[str, str]
    _fci_plus_parallel_stats: dict[str, Any]
    _parallel_ci_stats: dict[str, Any]
    _parallel_cit: "CIT"
    _parallel_executor: "ThreadPoolExecutor | _ParallelRunner | None"

    def __init__(
        self, no_of_var: int, node_names: list[str], device: str | None = None
    ):
        node_names = list(node_names)
        if len(node_names) != no_of_var:
            msg = f"no_of_var={no_of_var} does not match len(node_names)={len(node_names)}"
            raise ValueError(msg)
        # Node equality and hashing go by name, and GeneralGraph keys its index
        # map on the nodes themselves. Two nodes sharing a name would collapse
        # into one entry, silently redirecting every operation on the first to
        # the index of the second and dropping edges without a word.
        duplicates = sorted({name for name in node_names if node_names.count(name) > 1})
        if duplicates:
            msg = f"node_names must be unique; duplicated: {duplicates}"
            raise ValueError(msg)

        self.nodes = [Node(name) for name in node_names]
        self.G = GeneralGraph(self.nodes)
        cuda = device is not None and str(device).startswith("cuda")
        if cuda:
            import torch

            from .cuda_graph_batch import complete_undirected

            self._G_cuda: Any = complete_undirected(no_of_var, device)
            torch.cuda.synchronize()
        else:
            self._G_cuda = None
            self.G.graph = np.full(
                (no_of_var, no_of_var), Endpoint.TAIL.value, dtype=np.int8
            )
            np.fill_diagonal(self.G.graph, 0)
        self.sepset: np.ndarray = np.empty((no_of_var, no_of_var), object)
        # sepset is the legacy collider-membership summary in PC. Actual CI
        # separators live separately and must never be reconstructed by union.
        self.separators: dict[tuple[int, int], set[int]] = {}
        self.separator_members: dict[tuple[int, int], set[int]] = {}
        self.test: CIT | Callable[[int, int, Iterable[int]], float] | None = None

        # GPU PC-stable: S[start:end] for ranges[x,y]=(start,end). None if unused.
        self.sepset_S: np.ndarray | None = None
        self.sepset_ranges: np.ndarray | None = None
        self.sepset_IND: np.ndarray | None = None
        self._sepset_ptr: Any = None

    def is_adjacent_to(self, node1: Node, node2: Node) -> bool:
        """Return whether two nodes are adjacent."""
        return self.G.is_adjacent_to(node1, node2)

    def remove_edge(self, node1: Node, node2: Node):
        """Remove the edge between two nodes, if present."""
        edge = self.G.get_edge(node1, node2)
        if edge:
            self.G.remove_edge(edge)

    def add_undirected_edge(self, node1: Node, node2: Node):
        """Add an undirected edge between two nodes."""
        self.G.add_undirected_edge(node1, node2)

    def add_directed_edge(self, source: Node, target: Node):
        """Add a directed edge from source to target."""
        self.G.add_directed_edge(source, target)

    def add_bidirected_edge(self, node1: Node, node2: Node):
        """Add a bidirected edge between two nodes."""
        self.G.add_bidirected_edge(node1, node2)

    def _ensure_sepset_ptr(self):
        if self._sepset_ptr is not None:
            return
        IND = self.sepset_IND
        p = int(self.G.num_vars)
        ptr = np.full((p, p), -1, dtype=np.int64)
        if IND is not None and IND.shape[0] > 0:
            ns = IND.shape[0]
            x = IND[:ns, 0]
            y = IND[:ns, 1]
            idx = np.arange(ns, dtype=np.int64)
            ptr[x, y] = idx
            ptr[y, x] = idx
        self._sepset_ptr = ptr

    def sepset_slice(self, x: int, y: int):
        """Conditioner indices for undirected pair ``(x, y)``, or None if missing.

        GPU path: ``ranges[x,y]=(start,end)`` into ``sepset_S``.
        ``(-1,-1)`` still an edge; ``(0,0)`` empty sepset.
        """
        if self.sepset_ranges is not None:
            start = int(self.sepset_ranges[x, y, 0])
            end = int(self.sepset_ranges[x, y, 1])
            if start < 0:
                return None
            if start == 0 and end == 0:
                return np.zeros(0, dtype=np.int64)
            assert self.sepset_S is not None
            return self.sepset_S[start:end]
        if self.sepset_S is not None:
            self._ensure_sepset_ptr()
            i = int(self._sepset_ptr[x, y])
            if i < 0:
                return None
            assert self.sepset_IND is not None
            start = int(self.sepset_IND[i, 2])
            end = int(self.sepset_IND[i, 3])
            return self.sepset_S[start:end]
        s = self.sepset[x, y]
        if s is None:
            return None
        return np.asarray(s[0], dtype=np.int64)

    def sepset_contains(self, x: int, z: int, y: int) -> bool | None:
        """Check whether a recorded separating set for ``(x, z)`` contains ``y``.

        With packed storage, check both orders of the pair. Return None if
        neither has a record. With object-array storage, check the separating
        sets at ``sepset[x, z]`` and return None if that entry is missing.
        """
        if self.sepset_ranges is not None or self.sepset_S is not None:
            sl = self.sepset_slice(x, z)
            sl_rev = self.sepset_slice(z, x)
            if sl is None and sl_rev is None:
                return None
            if sl is not None and bool(np.any(sl == y)):
                return True
            return bool(sl_rev is not None and bool(np.any(sl_rev == y)))
        s = self.sepset[x, z]
        if s is None:
            return None
        return any(y in S for S in s)

    def set_ind_test(self, test_obj):
        """Attach the conditional-independence test used by the search."""
        self.test = test_obj

    def ci_test(self, i: int, j: int, S: Iterable[int]) -> float:
        """Return the p-value for testing whether i and j are independent given S."""
        if not conditioning_allowed(self, S):
            msg = "CI request exceeds max_cond_set_size; the search must skip this test"
            raise ValueError(msg)
        if self.test is None:
            msg = "Set an independence test before requesting a CI test"
            raise TypeError(msg)
        return self.test(i, j, S)

    def neighbors(self, i: int):
        """Return the neighbours of a node."""
        return np.where(self.G.graph[i, :] != 0)[0]

    def max_degree(self) -> int:
        """Return the maximum node degree."""
        return int(np.max(np.sum(self.G.graph != 0, axis=1)))

    def is_fully_directed(self, i, j):
        """Return whether the edge points from i to j with a tail at i."""
        return (
            self.G.graph[i, j] == Endpoint.ARROW.value
            and self.G.graph[j, i] == Endpoint.TAIL.value
        )

    def is_undirected(self, i, j):
        """Return whether the edge is undirected."""
        return (
            self.G.graph[i, j] == Endpoint.TAIL.value
            and self.G.graph[j, i] == Endpoint.TAIL.value
        )

    def is_bidirected(self, i, j):
        """Return whether the edge is bidirected."""
        return (
            self.G.graph[i, j] == Endpoint.ARROW.value
            and self.G.graph[j, i] == Endpoint.ARROW.value
        )

    def is_ancestor_of(self, i: int, j: int) -> bool:
        """Check whether i is an ancestor of j over fully directed edges.

        A directed path i -> ... -> j must consist exclusively of fully
        directed edges (graph[u, v] == ARROW and graph[v, u] == TAIL);
        bidirected and undirected edges do not count.  A node is never an
        ancestor of itself.

        Parameters
        ----------
        i : int
            Index of the candidate ancestor.
        j : int
            Index of the candidate descendant.

        Returns
        -------
        bool
            True if a fully directed path connects i to j and i differs from j.
        """
        if i == j:
            return False
        graph = self.G.graph
        visited = np.zeros(graph.shape[0], dtype=bool)
        visited[i] = True
        queue = [i]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            children = np.where(
                (graph[u, :] == Endpoint.ARROW.value)
                & (graph[:, u] == Endpoint.TAIL.value)
            )[0]
            for v in children:
                v = int(v)
                if v == j:
                    return True
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return False

    def find_unshielded_triples(self):
        """Return the unshielded triples of the graph."""
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
        """Return the triangles of the graph."""
        triangles = []
        adj = self.G.graph
        num_vars = len(self.nodes)
        edges = []
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                if adj[i, j] != 0:
                    edges.append((i, j))
        for i, j in edges:
            neigh_i = set(np.where(adj[i, :] != 0)[0])
            neigh_j = set(np.where(adj[j, :] != 0)[0])
            common = neigh_i.intersection(neigh_j)
            for k in common:
                if k > j:
                    triangles.append((i, j, k))
        return triangles

    def find_kites(self):
        """Return the kites of the graph."""
        kites = []
        num_vars = len(self.nodes)
        for target in range(num_vars):
            neighbors_l = np.where(self.G.graph[target, :] != 0)[0]
            if len(neighbors_l) < 2:
                continue
            for j, k in combinations(neighbors_l, 2):
                if not self.G.has_edge(j, k):
                    neigh_j = set(np.where(self.G.graph[j, :] != 0)[0])
                    neigh_k = set(np.where(self.G.graph[k, :] != 0)[0])
                    neigh_l = set(neighbors_l)
                    common = neigh_j.intersection(neigh_k).intersection(neigh_l)
                    for i in common:
                        kites.append((i, j, k, target))
        return kites


# Historical callers import Edge from this module.
Edge = edges.Edge

__all__ = ["CausalGraph"]
