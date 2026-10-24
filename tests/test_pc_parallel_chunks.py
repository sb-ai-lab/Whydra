"""Tests for the node chunking used by the loky-parallel PC-stable skeleton.

`parallel_pc_stable.skeleton_discovery` hands each joblib task a chunk of
nodes (balanced by the estimated number of CI tests) instead of a single
node.  The merge back into the graph sorts per-node results by node index,
so the only invariants the chunking itself must satisfy are: every node is
assigned exactly once, the partition is deterministic for a given adjacency
snapshot, and the number of chunks never exceeds the requested one.
"""

import math

import numpy as np

from whydra.algorithms.standalone.parallel_pc_stable import (
    _balanced_node_chunks,
    _node_cost,
)

N_VARS = 12
SEED = 3
EDGE_PROBABILITY = 0.4
N_CHUNKS = 5
DEPTHS = (0, 1, 2, 3)


def _random_undirected_adjacency(n_vars: int, seed: int, edge_probability: float) -> np.ndarray:
    """Build a symmetric adjacency matrix with TAIL (-1) marks on both ends.

    Args:
        n_vars: number of nodes.
        seed: random seed.
        edge_probability: probability of an edge between two nodes.

    Returns:
        np.ndarray: (n_vars, n_vars) int matrix, 0 = no edge, -1 = undirected mark.
    """
    rng = np.random.default_rng(seed)
    upper = rng.random((n_vars, n_vars)) < edge_probability
    upper = np.triu(upper, k=1)
    adjacency = upper | upper.T
    return np.where(adjacency, -1, 0).astype(int)


def test_node_cost_counts_ci_tests():
    """Cost equals deg * C(deg - 1, depth) and is 0 when the depth is unreachable."""
    graph = np.zeros((4, 4), dtype=int)
    for i, j in ((0, 1), (0, 2), (0, 3)):
        graph[i, j] = graph[j, i] = -1
    assert _node_cost(graph, 0, 0) == 3
    assert _node_cost(graph, 0, 1) == 3 * math.comb(2, 1)
    assert _node_cost(graph, 0, 2) == 3 * math.comb(2, 2)
    assert _node_cost(graph, 0, 3) == 3 * math.comb(2, 3)
    assert _node_cost(graph, 1, 1) == 1 * math.comb(0, 1)
    assert _node_cost(graph, 1, 2) == 0
    isolated = np.zeros((2, 2), dtype=int)
    assert _node_cost(isolated, 0, 0) == 0


def test_chunks_partition_all_nodes_exactly_once():
    """Every node appears in exactly one chunk and no chunk is empty."""
    graph = _random_undirected_adjacency(N_VARS, SEED, EDGE_PROBABILITY)
    for depth in DEPTHS:
        chunks = _balanced_node_chunks(graph, depth, N_CHUNKS)
        flat = [x for chunk in chunks for x in chunk]
        assert sorted(flat) == list(range(N_VARS))
        assert all(chunk for chunk in chunks)
        assert len(chunks) <= N_CHUNKS
        assert all(isinstance(x, int) for x in flat)


def test_chunks_are_deterministic():
    """The same adjacency snapshot always yields the same partition."""
    graph = _random_undirected_adjacency(N_VARS, SEED, EDGE_PROBABILITY)
    first = _balanced_node_chunks(graph, 1, N_CHUNKS)
    second = _balanced_node_chunks(graph.copy(), 1, N_CHUNKS)
    assert first == second


def test_more_chunks_than_nodes_is_clamped_by_caller_contract():
    """With n_chunks >= n_vars every non-zero-cost node can sit alone; nothing is lost."""
    graph = _random_undirected_adjacency(N_VARS, SEED, EDGE_PROBABILITY)
    chunks = _balanced_node_chunks(graph, 0, N_VARS)
    flat = sorted(x for chunk in chunks for x in chunk)
    assert flat == list(range(N_VARS))
