"""Tests for the Meek-rule cycle guard (causal-learn parity).

causal-learn skips orienting j -> k during Meek orientation when k is
already an ancestor of j over fully directed edges.  Without this guard
whydra can create directed cycles (or spurious bi-directed conflicts)
that causal-learn never produces under ``pc(stable=True, uc_rule=0,
uc_priority=2)``.
"""

import numpy as np
import pytest

from whydra.algorithms.graph_core import CausalGraph
from whydra.algorithms.standalone import parallel_pc_stable, standalone_pc_stable, pc_algo

NODE_NAMES = ["X1", "X2", "X3", "X4"]
N_VARS = 4


def _has_directed_cycle(graph: np.ndarray) -> bool:
    """Check whether the fully directed part of ``graph`` contains a cycle.

    Uses Kahn's algorithm over directed edges i -> j defined by
    ``graph[i, j] == 1 and graph[j, i] == -1``.

    Args:
        graph: adjacency matrix with endpoint marks.

    Returns:
        bool: True if a directed cycle exists.
    """
    n = graph.shape[0]
    directed = (graph == 1) & (graph.T == -1)
    indegree = directed.sum(axis=0)
    queue = [int(v) for v in np.where(indegree == 0)[0]]
    processed = 0
    while queue:
        u = queue.pop()
        processed += 1
        for v in np.where(directed[u])[0]:
            v = int(v)
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
    return processed < n


def _build_cycle_guard_graph() -> CausalGraph:
    """Build the graph 0 -> 1, 1 - 2, 2 -> 3, 3 -> 1 with no other edges.

    Returns:
        CausalGraph: the constructed graph.
    """
    cg = CausalGraph(N_VARS, NODE_NAMES)
    for i, j in ((0, 2), (0, 3)):
        cg.remove_edge(cg.nodes[i], cg.nodes[j])
    cg.add_directed_edge(cg.nodes[0], cg.nodes[1])
    cg.add_directed_edge(cg.nodes[2], cg.nodes[3])
    cg.add_directed_edge(cg.nodes[3], cg.nodes[1])
    return cg


_MEEK_ENTRY_POINTS = [
    (
        "parallel_pc_stable",
        lambda cg: parallel_pc_stable.apply_meek_rules(cg),
    ),
    (
        "standalone_pc_stable",
        lambda cg: standalone_pc_stable.apply_meek_rules(cg),
    ),
    (
        "pc_algo",
        lambda cg: pc_algo.PCAlgorithm(cg, show_progress=False).apply_meek_rules(),
    ),
]


def test_is_ancestor_of_only_counts_fully_directed_paths():
    """Only fully directed edges form ancestry; bidirected/undirected do not."""
    cg = CausalGraph(N_VARS, NODE_NAMES)
    # Keep only 0 - 1 and 1 - 2; remove the rest of the complete graph.
    for i, j in ((0, 2), (0, 3), (1, 3), (2, 3)):
        cg.remove_edge(cg.nodes[i], cg.nodes[j])

    # Undirected chain 0 - 1 - 2: no ancestry.
    assert not cg.is_ancestor_of(0, 2)

    cg.add_directed_edge(cg.nodes[0], cg.nodes[1])
    cg.add_directed_edge(cg.nodes[1], cg.nodes[2])
    assert cg.is_ancestor_of(0, 2)
    assert not cg.is_ancestor_of(2, 0)
    assert not cg.is_ancestor_of(0, 0)

    # A bidirected edge does not extend the directed ancestry.
    cg.add_bidirected_edge(cg.nodes[2], cg.nodes[3])
    assert not cg.is_ancestor_of(0, 3)
    assert not cg.is_ancestor_of(2, 3)


@pytest.mark.parametrize(
    "apply_meek",
    [entry[1] for entry in _MEEK_ENTRY_POINTS],
    ids=[entry[0] for entry in _MEEK_ENTRY_POINTS],
)
def test_meek_cycle_guard_prevents_directed_cycle(apply_meek):
    """R1 skips 1 -> 2 (2 is an ancestor of 1); R2 then orients 2 -> 1.

    On the graph 0 -> 1, 1 - 2, 2 -> 3, 3 -> 1, Meek R1 on the unshielded
    triple 0 -> 1 - 2 wants 1 -> 2, which would close the cycle
    1 -> 2 -> 3 -> 1.  causal-learn skips it, and R2 on the triangle
    {1, 2, 3} orients 2 -> 1 instead.
    """
    cg = _build_cycle_guard_graph()
    apply_meek(cg)

    assert cg.is_fully_directed(2, 1)
    assert not cg.is_bidirected(1, 2)
    assert cg.is_fully_directed(0, 1)
    assert cg.is_fully_directed(2, 3)
    assert cg.is_fully_directed(3, 1)
    assert not _has_directed_cycle(cg.G.graph)
