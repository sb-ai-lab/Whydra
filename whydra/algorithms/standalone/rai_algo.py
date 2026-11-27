# Adapted from Intel Labs' LearnStructRAI, MixedGraph and PDAG (Apache-2.0).
# Whydra uses its own graph and CI interfaces and parallelizes CI evaluation.
# Upstream: https://github.com/IntelLabs/causality-lab
"""RAI-stable search and its CUDA learner entry point."""

import logging
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import chain, combinations
from math import comb

import numpy as np

from ..graph_core import CIT, CausalGraph, Endpoint
from ..graph_core._optional_imports import Parallel, delayed
from ..graph_core.conditioning import (
    bounded_size,
    configure_conditioning,
    validate_cuda_background_knowledge,
    validate_max_cond_set_size,
)
from ..graph_core.independence_tests import chi_square_from_encoded, fisher_z_from_corr
from .profiler import profiler

logger = logging.getLogger(__name__)

UNDIRECTED = "---"  # Edge mark for an undirected endpoint.
DIRECTED = "<--"  # Edge mark for a directed endpoint (arrowhead at the node).
EDGE_MARKS = (
    UNDIRECTED,
    DIRECTED,
)  # Fixed traversal order. The reference uses a set; checks showed the order does not affect this traversal.

# Use workers only when the phase has enough work to parallelize.
PARALLEL_MIN_PAIRS = 2  # Connected pairs per phase. Below this threshold there are too few pairs to parallelize.
PARALLEL_MIN_COST_PER_JOB = (
    32  # Estimated phase cost must reach PARALLEL_MIN_COST_PER_JOB * n_jobs.
)
#                                     Cost = number of conditioning sets x (1 + order). This approximates
#                                     how much more expensive a higher-order test is than an order-zero test.
PARALLEL_CHUNKS_PER_JOB = 4  # Chunks per phase = PARALLEL_CHUNKS_PER_JOB * n_jobs.
JOBLIB_MAX_NBYTES = "100K"  # joblib shares arrays larger than this threshold with workers through memmap.


class _PDAG:
    """Represent a partially directed graph, ported from IntelLabs' MixedGraph and PDAG.

    Internally, ``_graph[node][DIRECTED]`` is the set of parents of node and
    ``_graph[node][UNDIRECTED]`` is the set of its undirected neighbors. Operations
    follow the reference using the same ``set`` operations because iteration order affects the result.
    """

    def __init__(self, nodes_set: set[int]):
        """Create an empty PDAG on the given set of nodes.

        :param nodes_set: Set of graph nodes.
        """
        self._graph: dict[int, dict[str, set[int]]] = {}
        self.nodes_set = nodes_set
        self.create_empty_graph(self.nodes_set)
        self.orientation_rules = {
            1: self.orient_by_rule_1,
            2: self.orient_by_rule_2,
            3: self.orient_by_rule_3,
            4: self.orient_by_rule_4,
        }

    def create_empty_graph(self, nodes_set: set[int] | None = None) -> None:
        """Initialize empty edge sets for each node.

        :param nodes_set: Set of nodes (defaults to all nodes in the graph).
        """
        if nodes_set is None:
            nodes_set = self.nodes_set
        for node in nodes_set:
            self._graph[node] = {}
            for head_type in EDGE_MARKS:
                self._graph[node][head_type] = set()

    def create_complete_graph(self, nodes_set: set[int] | None = None) -> None:
        """Build a complete undirected graph on the given set of nodes.

        :param nodes_set: Set of nodes (defaults to all nodes in the graph).
        """
        if nodes_set is None:
            nodes_set = self.nodes_set
        self.create_empty_graph(nodes_set)
        for node in nodes_set:
            self._graph[node][UNDIRECTED] = nodes_set - {node}

    def parents(self, target_node: int) -> set[int]:
        """Return the parents of a node.

        :param target_node: Target node.
        :return: Set of parents.
        """
        return self._graph[target_node][DIRECTED]

    def undirected_neighbors(self, node: int) -> set[int]:
        """Return the undirected neighbors of a node.

        :param node: Node.
        :return: Set of neighbors.
        """
        return self._graph[node][UNDIRECTED]

    def fan_in(self, target_node: int) -> int:
        """Count incoming connections (parents plus undirected neighbors).

        :param target_node: Target node.
        :return: Number of incoming connections.
        """
        return len(self.parents(target_node)) + len(
            self.undirected_neighbors(target_node)
        )

    def is_connected(self, node_i: int, node_j: int) -> bool:
        """Check for an edge of any type between two nodes.

        :param node_i: First node.
        :param node_j: Second node.
        :return: True if the nodes are adjacent.
        """
        assert node_i != node_j
        for node_p, node_c in [
            (node_i, node_j),
            (node_j, node_i),
        ]:  # Swap the parent and child roles.
            for edge_mark in EDGE_MARKS:
                if node_p in self._graph[node_c][edge_mark]:
                    return True
        return False

    def find_adjacent_nodes(
        self,
        node_i: int,
        pool_nodes: set[int] | None = None,
        edge_type: tuple[str, str] | None = None,
    ) -> set[int]:
        """Find adjacent nodes, optionally restricting the candidate nodes and edge type.

        :param node_i: Node whose neighbors to find.
        :param pool_nodes: Candidate nodes, or None to consider all nodes.
        :param edge_type: Pair of marks (at node_i, at its neighbor), or None for any edge type.
        :return: Set of adjacent nodes.
        """
        if edge_type is None:
            connected_nodes = set()
            for edge_mark in EDGE_MARKS:
                connected_nodes.update(self._graph[node_i][edge_mark])
        else:
            mark_origin = edge_type[0]
            mark_neighbor = edge_type[1]
            connected_nodes = set(
                filter(
                    lambda neighbor: node_i in self._graph[neighbor][mark_neighbor],
                    self._graph[node_i][mark_origin],
                )
            )
        if pool_nodes is not None:
            connected_nodes = connected_nodes & pool_nodes
        return connected_nodes

    def find_reachable_set(
        self,
        anchor_node: int,
        nodes_pool: set[int],
        edge_type_list: list[tuple[str, str]] | None,
    ) -> set[int]:
        """Use iterative breadth-first search to find the nodes reachable from the starting node.

        :param anchor_node: Starting node.
        :param nodes_pool: Set of allowed nodes.
        :param edge_type_list: Allowed edge types, or None for any type.
        :return: Set of reachable nodes, excluding the starting node.
        """
        if not nodes_pool:
            return set()
        reachable_set = set()
        allowed_pool = set(nodes_pool)
        frontier = {anchor_node}
        while frontier:
            next_frontier = set()
            for node in frontier:
                if edge_type_list is None:
                    next_frontier.update(
                        self.find_adjacent_nodes(node, allowed_pool, None)
                    )
                else:
                    for edge_type in edge_type_list:
                        next_frontier.update(
                            self.find_adjacent_nodes(node, allowed_pool, edge_type)
                        )
            reachable_set.update(next_frontier)
            allowed_pool.difference_update(next_frontier)
            frontier = next_frontier
        return reachable_set

    def find_unconnected_subgraphs(
        self,
        en_nodes: set[int] | None = None,
        sym_edge_mark: str | None = None,
    ) -> list[set[int]]:
        """Split nodes into connected components.

        :param en_nodes: Set of nodes (defaults to all nodes in the graph).
        :param sym_edge_mark: Symmetric edge mark used for connectivity, or None for any edge type.
        :return: List of node sets, one for each connected component.
        """
        if en_nodes is None:
            en_nodes = self.nodes_set
        connected_sets = []
        nodes = en_nodes.copy()
        edge_type_list = None
        if sym_edge_mark in EDGE_MARKS:
            edge_type_list = [(sym_edge_mark, sym_edge_mark)]
        while len(nodes) > 0:
            node_i = nodes.pop()
            reachable_set = self.find_reachable_set(node_i, nodes, edge_type_list)
            nodes.difference_update(reachable_set)
            reachable_set.add(node_i)
            connected_sets.append(reachable_set)
        return connected_sets

    def find_partial_topological_order(
        self, en_nodes: set[int] | None = None
    ) -> list[set[int]]:
        """Recursively find a partial topological order as a list of node sets.

        :param en_nodes: Set of nodes (defaults to all nodes in the graph).
        :return: List of node sets, from leaves to roots.
        """
        if en_nodes is None:
            en_nodes = self.nodes_set
        if len(en_nodes) == 0:
            return []
        parents_set = set()
        for node in en_nodes:
            parents_set.update(
                self.parents(node) & en_nodes
            )  # Nodes that are parents of other nodes.
        leaves_set = (
            en_nodes - parents_set
        )  # Nodes that have no children among the endogenous nodes.
        if len(leaves_set) == 0:
            return [parents_set]  # No further topological ordering could be determined.
        high_topological_ordering = self.find_partial_topological_order(
            parents_set
        )  # Recurse on the remaining nodes.
        return [leaves_set, *high_topological_ordering]

    def orient_edge(self, source_node: int, target_node: int) -> None:
        """Orient the edge as source_node --> target_node.

        :param source_node: Source node.
        :param target_node: Target node.
        """
        self._graph[target_node][DIRECTED].add(source_node)  # Add a directed endpoint.
        self._graph[target_node][UNDIRECTED].discard(
            source_node
        )  # Remove the undirected endpoint.
        self._graph[source_node][UNDIRECTED].discard(
            target_node
        )  # Remove the undirected endpoint.

    def delete_directed_edge(self, source_node: int, target_node: int) -> None:
        """Remove the directed edge source_node --> target_node.

        :param source_node: Source node.
        :param target_node: Target node.
        """
        self._graph[target_node][DIRECTED].discard(source_node)

    def delete_undirected_edge(self, node_i: int, node_j: int) -> None:
        """Remove the undirected edge between two nodes.

        :param node_i: First node.
        :param node_j: Second node.
        """
        self._graph[node_i][UNDIRECTED].discard(node_j)
        self._graph[node_j][UNDIRECTED].discard(node_i)

    def delete_edge(self, node_i: int, node_j: int) -> None:
        """Remove all edges between two nodes (both directed edges and the undirected edge).

        :param node_i: First node.
        :param node_j: Second node.
        """
        self.delete_directed_edge(
            node_i, node_j
        )  # Remove the directed endpoint at node_j.
        self.delete_directed_edge(
            node_j, node_i
        )  # Remove the directed endpoint at node_i.
        self.delete_undirected_edge(node_i, node_j)  # Remove the undirected endpoints.

    def add_edges(
        self,
        parents_set: set[int],
        target_node: int,
        arrowhead_type: str = UNDIRECTED,
    ) -> None:
        """Add edges from a set of nodes to the target node with the given endpoint type.

        :param parents_set: Set of source nodes.
        :param target_node: Target node.
        :param arrowhead_type: Endpoint type (UNDIRECTED or DIRECTED).
        """
        if arrowhead_type != UNDIRECTED and arrowhead_type != DIRECTED:
            raise ValueError
        self._graph[target_node][arrowhead_type] |= parents_set
        if arrowhead_type == UNDIRECTED:
            for parent_node in parents_set:
                self._graph[parent_node][UNDIRECTED].add(target_node)  # Reverse edge.

    def convert_bidirected_to_undirected(self, nodes: set[int] | None = None) -> None:
        """Replace bidirected (conflicting) edges with undirected edges.

        :param nodes: Set of nodes (defaults to all nodes in the graph).
        """
        if nodes is None:
            nodes = self.nodes_set
        for node_x, node_y in combinations(nodes, 2):
            if (
                node_y in self._graph[node_x][DIRECTED]
                and node_x in self._graph[node_y][DIRECTED]
            ):
                self._graph[node_x][DIRECTED].discard(
                    node_y
                )  # Remove the directed endpoints.
                self._graph[node_y][DIRECTED].discard(node_x)
                self._graph[node_x][UNDIRECTED].add(
                    node_y
                )  # Add the undirected endpoints.
                self._graph[node_y][UNDIRECTED].add(node_x)

    def maximally_orient_pattern(
        self, rules_set, en_nodes: set[int] | None = None
    ) -> None:
        """Apply the orientation rules until the graph stops changing.

        :param rules_set: Rule numbers (1 through 4) in the order to apply them.
        :param en_nodes: Set of nodes (defaults to all nodes in the graph).
        """
        if en_nodes is None:
            en_nodes = self.nodes_set
        graph_modified = True
        while graph_modified:
            graph_modified = False
            for rule_idx in rules_set:
                rule = self.orientation_rules[rule_idx]
                graph_modified |= rule(en_nodes)

    def orient_by_rule_1(self, en_nodes: set[int]) -> bool:
        """Apply R1: Z --> X --- Y with Z and Y nonadjacent implies X --> Y.

        :param en_nodes: Set of nodes.
        :return: True if the graph changed.
        """
        graph_modified = False
        for node_y in en_nodes:
            x_nodes = self.undirected_neighbors(
                node_y
            ).copy()  # Neighbors of the current Y.
            for node_x in x_nodes:  # Check every undirected edge incident to Y.
                for node_z in self.parents(node_x):
                    if not self.is_connected(node_z, node_y):
                        self.orient_edge(
                            source_node=node_x, target_node=node_y
                        )  # Orient X --> Y.
                        graph_modified = True
                        break  # X --> Y is now oriented. Move on to the next X --- Y.
        return graph_modified

    def orient_by_rule_2(self, en_nodes: set[int]) -> bool:
        """Apply R2: X --> Z --> Y and X --- Y imply X --> Y.

        :param en_nodes: Set of nodes.
        :return: True if the graph changed.
        """
        graph_modified = False
        for node_y in en_nodes:
            x_nodes = self.undirected_neighbors(node_y).copy()
            for node_x in x_nodes:
                z_nodes = self.parents(node_y)  # Parents connected by directed edges.
                for node_z in z_nodes:
                    if node_x in self.parents(node_z):
                        self.orient_edge(source_node=node_x, target_node=node_y)
                        graph_modified = True
                        break
        return graph_modified

    def orient_by_rule_3(self, en_nodes: set[int]) -> bool:
        """Apply R3: X --- W --> Y, X --- Z --> Y, W and Z nonadjacent, and X --- Y imply X --> Y.

        :param en_nodes: Set of nodes.
        :return: True if the graph changed.
        """
        graph_modified = False
        for node_y in en_nodes:
            x_nodes = self.undirected_neighbors(node_y).copy()
            wz_nodes = self.parents(node_y)
            for node_x in x_nodes:
                wz_nodes_of_x = self.undirected_neighbors(node_x).intersection(
                    wz_nodes
                )  # W and Z are neighbors of X.
                for node_w, node_z in combinations(wz_nodes_of_x, 2):
                    if self.is_connected(node_w, node_z):
                        continue  # Skip adjacent W and Z.
                    self.orient_edge(source_node=node_x, target_node=node_y)
                    graph_modified = True
                    break
        return graph_modified

    def orient_by_rule_4(self, en_nodes: set[int]) -> bool:
        """Apply R4: W --> Z --> Y, X --- Z, X --- W, W and Y nonadjacent, and X --- Y imply X --> Y.

        :param en_nodes: Set of nodes.
        :return: True if the graph changed.
        """
        graph_updated = False
        for node_y in en_nodes:
            x_nodes = self.undirected_neighbors(node_y).copy()
            z_nodes = self.parents(node_y)
            for node_x in x_nodes:
                for node_z in z_nodes:
                    if not self.is_connected(
                        node_z, node_x
                    ):  # Z and X must be adjacent.
                        continue
                    w_nodes = self.parents(node_z).intersection(
                        self.undirected_neighbors(node_x)
                    )
                    if len(w_nodes) > 0:
                        self.orient_edge(source_node=node_x, target_node=node_y)
                        graph_updated = True
                        break
        return graph_updated


class _SeparationSet:
    """Store separating sets for unordered pairs of nodes (ported from SeparationSet)."""

    def __init__(self, nodes_set: set[int]):
        """Create empty separating sets for all unordered pairs of nodes.

        :param nodes_set: Set of nodes.
        """
        self._sepset = {}
        self.nodes_set = nodes_set
        for i, j in combinations(nodes_set, 2):
            hkey = self.get_hash_key(i, j)
            self._sepset[hkey] = set()

    @staticmethod
    def get_hash_key(node_1: int, node_2: int) -> tuple[int, int]:
        """Return the key for a node pair with the smaller node first.

        :param node_1: First node.
        :param node_2: Second node.
        :return: Key for the pair.
        """
        hkey = (node_1, node_2) if node_1 < node_2 else (node_2, node_1)
        return hkey

    def set_sepset(self, node_1: int, node_2: int, sepset: Sequence[int]) -> None:
        """Replace the separating set for a pair of nodes.

        :param node_1: First node.
        :param node_2: Second node.
        :param sepset: Separating set.
        """
        hkey = self.get_hash_key(node_1, node_2)
        self._sepset[hkey] = set(sepset)

    def get_sepset(self, node_1: int, node_2: int) -> set[int]:
        """Return the separating set for a pair of nodes.

        :param node_1: First node.
        :param node_2: Second node.
        :return: Separating set.
        """
        hkey = self.get_hash_key(node_1, node_2)
        return self._sepset[hkey]


def _unique_element_iterator(
    iterable: Iterable[Sequence[int]],
) -> Iterable[tuple[int, ...]]:
    """Yield each conditioning set once, using a sorted tuple as its key.

    :param iterable: Input sequence of tuples.
    :return: Generator of unique tuples in input order.
    """
    seen: set[tuple[int, ...]] = set()
    for item in iterable:
        key = tuple(sorted(item))
        if key in seen:
            continue
        seen.add(key)
        yield tuple(item)


class _CIOracle:
    """Wrap a CI test to determine whether a pair of nodes is conditionally independent."""

    def __init__(self, tester: CIT, alpha: float):
        """Store the independence test and its significance level.

        :param tester: CI test called as tester(x, y, cond_set) -> p-value.
        :param alpha: Significance level.
        """
        self.tester = tester
        self.alpha = alpha

    def cond_indep(self, x: int, y: int, cond_set: Sequence[int]) -> bool:
        """Test whether x and y are conditionally independent given cond_set.

        Pass the conditioning set S in sorted order so its input order does
        not affect the p-value.

        :param x: First node.
        :param y: Second node.
        :param cond_set: Conditioning set.
        :return: True if the p-value exceeds alpha (the test finds independence).
        """
        p_val = self.tester(x, y, tuple(sorted(cond_set)))
        return p_val > self.alpha


def _pot_parents(graph: _PDAG, node: int, other: int) -> set[int]:
    """Return potential parents (parents and undirected neighbors), excluding the other node.

    :param graph: PDAG.
    :param node: Target node.
    :param other: Node to exclude (the other endpoint of the edge under test).
    :return: Set of potential parents.
    """
    return (graph.parents(node) | graph.undirected_neighbors(node)) - {other}


def _cond_sets(
    pot_parents_a: Iterable[int],
    pot_parents_b: Iterable[int],
    order: int,
) -> Iterable[tuple[int, ...]]:
    """Yield distinct conditioning sets of size order from the two pools.

    :param pot_parents_a: First pool of potential parents.
    :param pot_parents_b: Second pool of potential parents.
    :param order: Size of the conditioning set.
    :return: Generator of unique tuples.
    """
    return _unique_element_iterator(
        chain(combinations(pot_parents_a, order), combinations(pot_parents_b, order))
    )


def _make_tester(
    method: str,
    arrays: tuple[np.ndarray, ...],
    n_samples: int,
) -> Callable[[int, int, Sequence[int]], float]:
    """Build a p-value function from the CI test's encoded arrays.

    :param method: ``chisq`` or ``fisherz``.
    :param arrays: chisq — ``(data_int, cardinalities)``; fisherz — ``(corr_matrix,)``.
    :param n_samples: Number of observations.
    :return: Function with signature ``(x, y, S) -> p-value``.
    """
    if method == "chisq":
        data_int, cardinalities = arrays
        return lambda x, y, S: float(
            chi_square_from_encoded(data_int, cardinalities, n_samples, x, y, S)
        )
    if method == "fisherz":
        (corr_matrix,) = arrays
        return lambda x, y, S: fisher_z_from_corr(corr_matrix, n_samples, x, y, S)
    msg = f"Unknown CI test method: {method}"
    raise ValueError(msg)


def _ci_worker(
    method: str,
    arrays: tuple[np.ndarray, ...],
    n_samples: int,
    alpha: float,
    order: int,
    jobs: list[tuple[int, int, tuple[int, ...], tuple[int, ...]]],
) -> list[tuple[int, int, tuple[int, ...], float]]:
    """Precompute CI tests for a chunk of pairs, stopping each pair at its first p > alpha.

    Take conditioning sets from the graph snapshot at the start of the phase.
    This step only precomputes p-values; the main process makes the decisions.
    Rebuild the tester for each chunk. The closure over the memmap arrays is
    cheap to create. Caching it on a worker keeps the memmap alive and makes
    joblib wait a second to clean up its temporary directory when ``Parallel`` exits.

    :param method: ``chisq`` or ``fisherz``.
    :param arrays: Arrays used by the CI test.
    :param n_samples: Number of observations.
    :param alpha: Significance level (used only for early stopping).
    :param order: Size of the conditioning set.
    :param jobs: List of ``(x, y, pot_parents_x, pot_parents_y)`` tuples.
    :return: Flat list of ``(x, y, sorted_cset, p_value)`` tuples.
    """
    tester = _make_tester(method, arrays, n_samples)
    out: list[tuple[int, int, tuple[int, ...], float]] = []
    for x, y, pot_x, pot_y in jobs:
        for cset in _cond_sets(pot_x, pot_y, order):
            key = tuple(sorted(cset))
            p = tester(x, y, key)
            out.append((x, y, key, p))
            if p > alpha:
                break
    return out


class _SpeculativeMemo:
    """Read joblib result chunks lazily in submission order and return p-values."""

    def __init__(
        self,
        results: Iterator[list[tuple[int, int, tuple[int, ...], float]]],
        chunk_of_pair: list[int],
    ):
        """Create a memo backed by the result generator.

        :param results: joblib result generator yielding one list per chunk.
        :param chunk_of_pair: ``chunk_of_pair[i]`` is the chunk index for pair i (-1 if the pair was not connected).
        """
        self._results = results
        self._chunk_of_pair = chunk_of_pair
        self._loaded = 0
        self._memo: dict[tuple[int, int, tuple[int, ...]], float] = {}
        self.n_worker_tests = 0

    def _load_chunk(self, chunk: list[tuple[int, int, tuple[int, ...], float]]) -> None:
        """Store the results of one chunk in the memo.

        :param chunk: List of ``(x, y, sorted_cset, p_value)`` tuples.
        """
        for x, y, key, p in chunk:
            self._memo[(x, y, key)] = p
            self.n_worker_tests += 1
        self._loaded += 1

    def ensure_pair(self, pair_idx: int) -> None:
        """Read chunks until the results for ``pair_idx`` are in the memo.

        :param pair_idx: Position of the pair in this phase.
        """
        target = self._chunk_of_pair[pair_idx]
        if target == -1:
            return
        while self._loaded <= target:
            self._load_chunk(next(self._results))

    def get(self, x: int, y: int, sorted_cset: tuple[int, ...]) -> float | None:
        """Return a cached p-value or None on a miss.

        :param x: First variable in the test call.
        :param y: Second variable.
        :param sorted_cset: Sorted conditioning set.
        :return: The p-value or None.
        """
        return self._memo.get((x, y, sorted_cset))

    def drain(self) -> None:
        """Read all remaining chunks before the next Parallel call."""
        for chunk in self._results:
            self._load_chunk(chunk)


class RAIStableLearner:
    """Port IntelLabs causality-lab's LearnStructRAI to Whydra's graph primitives.

    Sequential mode matches the reference bit for bit. Parallel mode
    (``n_jobs > 1``) speeds up only the CI tests and leaves the result unchanged.
    """

    cg: CausalGraph
    max_cond_set_size: int | None
    alpha: float

    def __init__(
        self,
        data: np.ndarray,
        alpha: float,
        indep_test: str,
        node_names: Sequence[str],
        verbose: bool = False,
    ):
        """Initialize RAI.

        :param data: Data matrix of shape (n_samples, n_vars).
        :param alpha: Significance level for the CI test.
        :param indep_test: Name of the CI test (``fisherz`` or ``chisq``).
        :param node_names: Variable names.
        :param verbose: Log statistics for each phase.
        """
        self.n_vars = data.shape[1]
        self.node_names = list(node_names)
        self.ci_test = CIT(data, method=indep_test)
        self.oracle = _CIOracle(self.ci_test, alpha)
        self.nodes_set = set(range(self.n_vars))
        self.graph = _PDAG(self.nodes_set)
        self.graph.create_complete_graph(self.nodes_set)
        self.sepset = _SeparationSet(self.nodes_set)
        self.verbose = verbose
        self.n_ci_tests = 0
        self.n_jobs = 1
        self._parallel = None
        #: List of ``(x, y, sorted_cset, independent)`` decisions if the caller supplies a list (used in tests).
        self.trace: list[tuple[int, int, tuple[int, ...], bool]] | None = None
        self.stats: dict[str, int | float] = {
            "phases": 0,
            "parallel_phases": 0,
            "ci_total": 0,
            "ci_worker": 0,
            "ci_sync": 0,
        }

    def learn_structure(self, n_jobs: int = 1):
        """Run RAI structure learning and return the graph.

        With ``n_jobs > 1``, joblib workers compute CI tests for each phase in
        advance. The sequential pass makes the decisions, so the result is
        independent of ``n_jobs``. With ``n_jobs <= 1``, no joblib pool is created.

        :param n_jobs: Number of processes for CI tests.
        :return: GeneralGraph whydra (CPDAG).
        """
        self.n_jobs = n_jobs
        if n_jobs > 1:
            with Parallel(
                n_jobs=n_jobs,
                backend="loky",
                max_nbytes=JOBLIB_MAX_NBYTES,
                mmap_mode="r",
                return_as="generator",
                pre_dispatch="all",
            ) as parallel:
                self._parallel = parallel
                try:
                    self._learn()
                finally:
                    self._parallel = None
        else:
            self._parallel = None
            self._learn()
        return self._to_general_graph()

    def _learn(self) -> None:
        """Run the recursion, apply the final R1–R3 rules, and reorient the skeleton."""
        en_nodes = self.nodes_set
        ex_nodes = set()
        with profiler.time_block("standalone_rai_total", tags={"algo": "rai"}):
            self._learn_recursively(en_nodes=en_nodes, ex_nodes=ex_nodes, order=0)
            self.graph.maximally_orient_pattern((1, 2, 3))
            self._re_orient_skeleton()

    def _worker_arrays(self) -> tuple[str, tuple[np.ndarray, ...], int]:
        """Give workers the same ndarray objects every time for the CI test.

        joblib writes each array to a memmap once per ``Parallel`` context.
        A fresh copy on every call would have to be written again.

        :return: ``(method, arrays, n_samples)``.
        """
        method = self.ci_test.method
        if method == "chisq":
            return (
                method,
                (self.ci_test.data_int, np.asarray(self.ci_test.cardinalities)),
                self.ci_test.n_samples,
            )
        if method == "fisherz":
            corr = self.ci_test.corr_matrix
            assert corr is not None
            return method, (corr,), self.ci_test.n_samples
        msg = f"Unknown CI test method: {method}"
        raise ValueError(msg)

    def _re_orient_skeleton(self) -> None:
        """Reset edge directions and reorient the skeleton using the final parents for v-structures, then R1–R3."""
        cpdag_final = _PDAG(self.graph.nodes_set)
        for node in self.graph.nodes_set:
            connected_nodes_set = self.graph.parents(
                node
            ) | self.graph.undirected_neighbors(node)
            cpdag_final.add_edges(
                parents_set=connected_nodes_set,
                target_node=node,
                arrowhead_type=UNDIRECTED,
            )

        for node in self.graph.nodes_set:
            parents_set = self.graph.parents(node)
            for parent_i, parent_j in combinations(parents_set, 2):
                if not self.graph.is_connected(parent_i, parent_j):
                    cpdag_final.orient_edge(
                        source_node=parent_i, target_node=node
                    )  # Orient the v-structure.
                    cpdag_final.orient_edge(source_node=parent_j, target_node=node)

        cpdag_final.maximally_orient_pattern(
            (1, 2, 3)
        )  # Apply orientation rules R1, R2 and R3.
        self.graph = cpdag_final

    def _exit_cond(self, en_nodes: set[int], order: int) -> bool:
        """Stop recursion when no node has more incoming connections than order.

        :param en_nodes: Set of endogenous nodes.
        :param order: Current order of the CI tests.
        :return: True if recursion can stop.
        """
        if order > bounded_size(self, order):
            return True
        return all(self.graph.fan_in(node) <= order for node in en_nodes)

    def _learn_recursively(
        self, en_nodes: set[int], ex_nodes: set[int], order: int
    ) -> None:
        """Run one RAI step: refine and orient the graph, split ancestors from descendants, then recurse.

        :param en_nodes: Endogenous nodes.
        :param ex_nodes: Exogenous nodes.
        :param order: Current order of the CI tests.
        """
        if self._exit_cond(en_nodes, order):
            return
        self._refine_and_orient(en_nodes=en_nodes, ex_nodes=ex_nodes, order=order)
        d_nodes, list_of_ancestors_sets, a_nodes = self._split_ancestors_descendant(
            en_nodes=en_nodes
        )
        for ancestor_set in list_of_ancestors_sets:
            self._learn_recursively(
                en_nodes=ancestor_set, ex_nodes=ex_nodes, order=order + 1
            )
        self._learn_recursively(
            en_nodes=d_nodes, ex_nodes=a_nodes | ex_nodes, order=order + 1
        )

    def _refine_and_orient(
        self, en_nodes: set[int], ex_nodes: set[int], order: int
    ) -> None:
        """Run the exogenous phase, orient, run the endogenous phase, then orient again.

        :param en_nodes: Endogenous nodes.
        :param ex_nodes: Exogenous nodes.
        :param order: Current order of the CI tests.
        """
        self._refine_exogenous_effect(en_nodes=en_nodes, ex_nodes=ex_nodes, order=order)
        self.maximally_orient_edges(en_nodes=en_nodes)
        self._refine_endogenous(en_nodes=en_nodes, order=order)
        self.maximally_orient_edges(en_nodes=en_nodes)

    def _split_ancestors_descendant(
        self, en_nodes: set[int]
    ) -> tuple[set[int], list[set[int]], set[int]]:
        """Split the nodes into descendants and a list of disconnected ancestor sets.

        :param en_nodes: Endogenous nodes.
        :return: (d_nodes, list_of_ancestors_sets, a_nodes).
        """
        d_nodes = self._get_lowest_topological_set(en_nodes)
        a_nodes = en_nodes - d_nodes  # Node sets to split.
        list_of_ancestors_sets = self.graph.find_unconnected_subgraphs(
            a_nodes
        )  # Disconnected sets of ancestors.
        return d_nodes, list_of_ancestors_sets, a_nodes

    def _refine_exogenous_effect(
        self, en_nodes: set[int], ex_nodes: set[int], order: int
    ) -> None:
        """Use CI tests of the given order to prune edges between exogenous and endogenous nodes.

        Visit pairs in reference order: ``for node in en_nodes: for ex in ex_nodes``.
        Call the test with ``(ex, node)``.

        :param en_nodes: Endogenous nodes.
        :param ex_nodes: Exogenous nodes.
        :param order: Current order of the CI tests.
        """
        pairs = [(ex, node) for node in en_nodes for ex in ex_nodes]
        self._refine_pairs(
            pairs, order, phase="exogenous", n_en=len(en_nodes), n_ex=len(ex_nodes)
        )

    def _refine_endogenous(self, en_nodes: set[int], order: int) -> None:
        """Use CI tests of the given order to prune edges between endogenous nodes.

        :param en_nodes: Endogenous nodes.
        :param order: Current order of the CI tests.
        """
        pairs = list(combinations(en_nodes, 2))
        self._refine_pairs(pairs, order, phase="endogenous", n_en=len(en_nodes), n_ex=0)

    def _refine_pairs(
        self,
        pairs: list[tuple[int, int]],
        order: int,
        phase: str,
        n_en: int,
        n_ex: int,
    ) -> None:
        """Run one refinement phase sequentially on the live graph, as in the reference.

        In parallel mode, workers first compute p-values for a graph snapshot
        (``_speculate``). The loop uses these values from the memo and computes any misses.
        It makes all edge removal decisions here, in reference order.

        :param pairs: Pairs in phase order, with test arguments in the same order.
        :param order: Size of the conditioning set.
        :param phase: Phase name for logging.
        :param n_en: Number of endogenous nodes for logging.
        :param n_ex: Number of exogenous nodes for logging.
        """
        memo = self._speculate(pairs, order) if self._parallel is not None else None
        n_start = self.n_ci_tests
        n_sync_start = self.stats["ci_sync"]
        for pair_idx, (a, b) in enumerate(pairs):
            if not self.graph.is_connected(a, b):
                continue
            pot_a = _pot_parents(self.graph, a, b)
            pot_b = _pot_parents(self.graph, b, a)
            if memo is not None:
                memo.ensure_pair(pair_idx)
            for cset in _cond_sets(pot_a, pot_b, order):
                if self._ci_memo(a, b, cset, memo):
                    self.graph.delete_edge(a, b)
                    self.sepset.set_sepset(a, b, cset)
                    break
        if memo is not None:
            memo.drain()
            self.stats["ci_worker"] += memo.n_worker_tests
            self.stats["parallel_phases"] += 1
        self.stats["phases"] += 1
        if self.verbose:
            logger.info(
                f"RAI phase={phase} order={order} en_nodes={n_en} ex_nodes={n_ex} "
                f"pairs={len(pairs)} ci_tests={self.n_ci_tests - n_start} "
                f"ci_sync={self.stats['ci_sync'] - n_sync_start} parallel={memo is not None}"
            )

    def _ci_memo(
        self,
        a: int,
        b: int,
        cset: Sequence[int],
        memo: _SpeculativeMemo | None,
    ) -> bool:
        """Decide conditional independence using a cached p-value or a synchronous test.

        Both paths use the same decision rule: ``ci_test(a, b, sorted(S)) > alpha``.

        :param a: First variable in the test call.
        :param b: Second variable.
        :param cset: Conditioning set (original tuple).
        :param memo: Memo of precomputed p-values, or None.
        :return: True if the variables are judged conditionally independent.
        """
        key = tuple(sorted(cset))
        self.n_ci_tests += 1
        self.stats["ci_total"] += 1
        p = memo.get(a, b, key) if memo is not None else None
        if p is None:
            p = self.ci_test(a, b, key)
            if memo is not None:
                self.stats["ci_sync"] += 1
        independent = p > self.oracle.alpha
        if self.trace is not None:
            self.trace.append((a, b, key, independent))
        return independent

    def _speculate(
        self, pairs: list[tuple[int, int]], order: int
    ) -> _SpeculativeMemo | None:
        """Precompute a phase's CI tests on workers, or return None if the phase is too small.

        Take a snapshot of ``pot_parents`` for each connected pair in the current
        set iteration order. Preserve pair order when splitting chunks for replay.

        :param pairs: Pairs in phase order.
        :param order: Size of the conditioning set.
        :return: A ``_SpeculativeMemo`` or None.
        """
        graph = self.graph
        n_jobs = self.n_jobs
        jobs = []
        chunk_of_pair = [-1] * len(pairs)
        costs = []
        for idx, (a, b) in enumerate(pairs):
            if not graph.is_connected(a, b):
                continue
            if order == 0:
                pot_a = (
                    pot_b
                ) = ()  # The only conditioning set is empty, so the worker does not need the neighbors.
            else:
                pot_a = tuple(_pot_parents(graph, a, b))
                pot_b = tuple(_pot_parents(graph, b, a))
            jobs.append((idx, a, b, pot_a, pot_b))
            costs.append(
                (comb(len(pot_a), order) + comb(len(pot_b), order)) * (1 + order)
            )
        if (
            len(jobs) < PARALLEL_MIN_PAIRS
            or sum(costs) < PARALLEL_MIN_COST_PER_JOB * n_jobs
        ):
            return None
        n_chunks = min(len(jobs), PARALLEL_CHUNKS_PER_JOB * n_jobs)
        threshold = sum(costs) / n_chunks
        buckets: list[list[tuple[int, int, int, tuple[int, ...], tuple[int, ...]]]] = [
            [] for _ in range(n_chunks)
        ]
        chunk_idx = 0
        acc = 0
        for job, cost in zip(jobs, costs, strict=False):
            buckets[chunk_idx].append(job)
            chunk_of_pair[job[0]] = chunk_idx
            acc += cost
            if chunk_idx < n_chunks - 1 and acc >= threshold:
                chunk_idx += 1
                acc = 0
        chunks = [
            bucket for bucket in buckets if bucket
        ]  # Only trailing chunks can be empty.
        method, arrays, n_samples = self._worker_arrays()
        assert self._parallel is not None
        results = self._parallel(
            delayed(_ci_worker)(
                method,
                arrays,
                n_samples,
                self.oracle.alpha,
                order,
                [(a, b, pot_a, pot_b) for (_, a, b, pot_a, pot_b) in chunk],
            )
            for chunk in chunks
        )
        return _SpeculativeMemo(iter(results), chunk_of_pair)

    def _get_lowest_topological_set(self, en_nodes: set[int]) -> set[int]:
        """Return the nodes in the lowest topological layer (descendants).

        :param en_nodes: Endogenous nodes.
        :return: Set of descendant nodes.
        """
        ordered_sets = self.graph.find_partial_topological_order(en_nodes)
        if len(ordered_sets) > 1:  # Two or more layers.
            return en_nodes - ordered_sets[-1]  # All nodes outside the top layer.
        return ordered_sets[0]  # One layer (no split found).

    def maximally_orient_edges(self, en_nodes: set[int]) -> None:
        """Orient v-structures, apply R1–R4, and make conflicting edges undirected.

        :param en_nodes: Endogenous nodes.
        """
        self.orient_v_structures(en_nodes)  # [R0]: Orient v-structures.
        self.graph.maximally_orient_pattern([1, 2, 3, 4])
        self.graph.convert_bidirected_to_undirected(
            en_nodes
        )  # Treat spurious bidirected edges as undirected.

    def orient_v_structures(self, en_nodes: set[int]) -> None:
        """Orient v-structures using separating sets in two passes, as in the reference.

        :param en_nodes: Endogenous nodes.
        """
        pre_neighbors = {}
        pre_parents = {}
        for node in self.graph.nodes_set:
            pre_neighbors[node] = self.graph.undirected_neighbors(
                node
            ).copy()  # Neighbors before changing the graph.

        for node_z in en_nodes:
            xy_nodes = pre_neighbors[node_z]  # Undirected neighbors.
            for node_x, node_y in combinations(xy_nodes, 2):
                if self.graph.is_connected(node_x, node_y):
                    continue  # Skip the pair because the nodes are adjacent.
                if node_z not in self.sepset.get_sepset(node_x, node_y):
                    self.graph.orient_edge(
                        source_node=node_x, target_node=node_z
                    )  # Orient X --> Z.
                    self.graph.orient_edge(
                        source_node=node_y, target_node=node_z
                    )  # Orient Y --> Z.

        for node in self.graph.nodes_set:
            pre_neighbors[node] = self.graph.undirected_neighbors(node).copy()
            pre_parents[node] = self.graph.parents(node).copy()

        for node_z in en_nodes:
            parents_z = pre_parents[node_z]  # Parents before changing the graph.
            neighbors_z = pre_neighbors[node_z]  # Remaining undirected edges X --- Z.
            for node_x in neighbors_z:
                for node_y in parents_z:
                    if self.graph.is_connected(node_x, node_y):
                        continue
                    if node_z not in self.sepset.get_sepset(node_x, node_y):
                        self.graph.orient_edge(
                            source_node=node_x, target_node=node_z
                        )  # Orient only X --> Z.

    def _to_general_graph(self):
        """Convert the internal PDAG to Whydra's GeneralGraph.

        :return: GeneralGraph with an endpoint matrix (TAIL=-1, ARROW=1).
        """
        cg = CausalGraph(self.n_vars, self.node_names)
        cg.G.graph[:] = 0
        for node in self.graph.nodes_set:
            for parent in self.graph.parents(node):  # parent --> node
                cg.G.graph[parent, node] = Endpoint.ARROW.value
                cg.G.graph[node, parent] = Endpoint.TAIL.value
            for nb in self.graph.undirected_neighbors(node):  # nb --- node
                cg.G.graph[nb, node] = Endpoint.TAIL.value
                cg.G.graph[node, nb] = Endpoint.TAIL.value
        return cg.G


class ResultWrapper:
    """Wrap the resulting GeneralGraph in the G attribute."""

    def __init__(self, graph, working=None):
        """Store the graph.

        :param graph: Final GeneralGraph.
        :param working: Working CausalGraph used by the algorithm (cg attribute).
        """
        self.G = graph
        self.cg = working


def rai_stable(
    data: np.ndarray,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    node_names: list[str] | None = None,
    n_jobs: int = -1,
    verbose: bool = False,
    max_cond_set_size=None,
    ci_test=None,
    certify_skeleton=False,
    device: str | None = None,
    dtype: str | None = None,
    **kwargs,
):
    """Run RAI (recursive autonomy identification), ported from the IntelLabs reference.

    RAI does not support background knowledge (``background_knowledge``).
    Passing a value other than None raises NotImplementedError.
    Other unrecognized keyword arguments are ignored for backward compatibility.

    :param data: Data matrix of shape (n_samples, n_vars).
    :param alpha: Significance level for the CI test.
    :param indep_test: Name of the CI test.
    :param node_names: Variable names (None generates X1 through Xn).
    :param n_jobs: Number of CI test processes (a negative value uses all CPU cores).
    :param verbose: Log statistics for each phase.
    :param device: ``"cuda"`` runs RAI-stable-cuda (``n_jobs`` = 1).
    :param dtype: CUDA Fisher-Z data type for RAI-stable-cuda.
    :param kwargs: Backward-compatible arguments (``n_procs``) and ignored parameters.
    :return: ResultWrapper with the final graph in its G attribute.
    """
    cuda = device is not None and str(device).startswith("cuda")
    background_knowledge = kwargs.get("background_knowledge")
    if cuda:
        validate_cuda_background_knowledge(background_knowledge, "rai_stable_cuda")
    elif background_knowledge is not None:
        msg = "RAI does not support background_knowledge"
        raise NotImplementedError(msg)
    max_cond_set_size = validate_max_cond_set_size(max_cond_set_size)
    if node_names is None:
        node_names = [f"X{i + 1}" for i in range(data.shape[1])]
    if "n_procs" in kwargs:
        n_jobs = kwargs["n_procs"]
    if n_jobs < 0:
        n_jobs = os.cpu_count() or 1

    if cuda:
        from ..graph_core.cpu_cuda_rai_bridge import (
            RAIStableCudaLearner,
            _write_back_causal_graph,
        )

        with profiler.time_block(
            "standalone_rai_total", tags={"algo": "rai_stable_cuda"}
        ):
            learner = RAIStableCudaLearner(
                data=data,
                alpha=alpha,
                indep_test=indep_test,
                node_names=node_names,
                verbose=verbose,
                dtype=dtype,
                ci_test=ci_test,
            )
            learner.max_cond_set_size = max_cond_set_size
            learner.ci_test.max_cond_set_size = max_cond_set_size
            final_graph = learner.learn_structure()
        if certify_skeleton:
            from .rai_validation import certify_rai_skeleton

            learner.cg = CausalGraph(data.shape[1], node_names)
            learner.cg.G.graph[:] = final_graph.graph
            for x in learner.nodes_set:
                for y in learner.nodes_set - {x}:
                    if not learner.graph.is_connected(x, y):
                        learner.cg.sepset[x, y] = [
                            tuple(learner.sepset.get_sepset(x, y))
                        ]
            learner.alpha = alpha
            configure_conditioning(learner.cg, max_cond_set_size)
            final_graph = certify_rai_skeleton(learner, n_jobs=n_jobs)
            cg = learner.cg
        else:
            cg = CausalGraph(data.shape[1], node_names)
            _write_back_causal_graph(cg, learner)
        final_graph.metadata.update(
            max_cond_set_size=max_cond_set_size,
            depth_limited=max_cond_set_size is not None,
        )
        return ResultWrapper(final_graph, cg)

    learner = RAIStableLearner(
        data=data,
        alpha=alpha,
        indep_test=indep_test,
        node_names=node_names,
        verbose=verbose,
    )
    learner.max_cond_set_size = max_cond_set_size
    if ci_test is not None:
        learner.ci_test = ci_test
        learner.oracle = _CIOracle(ci_test, alpha)
    learner.ci_test.max_cond_set_size = max_cond_set_size
    final_graph = learner.learn_structure(n_jobs=n_jobs)
    if certify_skeleton:
        from .rai_validation import certify_rai_skeleton

        learner.cg = CausalGraph(learner.n_vars, learner.node_names)
        learner.cg.G.graph[:] = final_graph.graph
        for x in learner.nodes_set:
            for y in learner.nodes_set - {x}:
                if not learner.graph.is_connected(x, y):
                    learner.cg.sepset[x, y] = [tuple(learner.sepset.get_sepset(x, y))]
        learner.alpha = alpha
        configure_conditioning(learner.cg, max_cond_set_size)
        final_graph = certify_rai_skeleton(learner, n_jobs=n_jobs)
    final_graph.metadata.update(
        max_cond_set_size=max_cond_set_size, depth_limited=max_cond_set_size is not None
    )
    return ResultWrapper(final_graph, getattr(learner, "cg", None))
