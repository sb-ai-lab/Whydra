"""Test PDAG and CPDAG primitives on hand-specified matrices.

Expected results follow from the graph definitions and require no sampled data
or version-sensitive numerical thresholds.

C[x,y]=1 and C[y,x]=0 encode x->y. Two ones encode x--y and two zeros
encode the absence of an edge.
"""

import numpy as np
import pytest

from whydra.algorithms.score_based.pdag import (
    complete_pdag,
    dag_to_cpdag,
    exists_semidirected_path,
    is_adjacent,
    is_clique,
    is_directed,
    is_undirected,
    meek_rules,
    na_yx,
    neighbours,
    parents,
    pdag_to_dag,
)


def mat(p, directed=(), undirected=()):
    C = np.zeros((p, p), dtype=np.int8)
    for a, b in directed:
        C[a, b] = 1
    for a, b in undirected:
        C[a, b] = 1
        C[b, a] = 1
    return C


# ============================================================
# Basic edge relationships
# ============================================================


def test_relations_distinguish_directed_from_undirected():
    C = mat(3, directed=[(0, 1)], undirected=[(1, 2)])
    assert is_directed(C, 0, 1)
    assert not is_directed(C, 1, 0)
    assert is_undirected(C, 1, 2)
    assert is_undirected(C, 2, 1)
    assert not is_undirected(C, 0, 1)
    assert is_adjacent(C, 0, 1)
    assert is_adjacent(C, 1, 0)
    assert not is_adjacent(C, 0, 2)


def test_neighbours_parents_and_na_yx():
    # 0 -> 1, 1 -- 2, 2 -- 3, 0 -- 2
    C = mat(4, directed=[(0, 1)], undirected=[(1, 2), (2, 3), (0, 2)])
    assert neighbours(C, 1) == {2}
    assert neighbours(C, 2) == {1, 3, 0}
    assert parents(C, 1) == {0}
    assert parents(C, 2) == set()
    # NA_yx takes neighbors of y adjacent to x. Node 1 is a neighbor of y=2;
    # with x=0, it is the only such neighbor adjacent to x.
    assert na_yx(C, 2, 0) == {1}


@pytest.mark.parametrize(
    ("nodes", "expected"),
    [((), True), ((0,), True), ((0, 1), True), ((0, 1, 2), True), ((0, 1, 3), False)],
)
def test_is_clique(nodes, expected):
    # Nodes 0, 1, and 2 form a triangle. Node 3 connects only to 0.
    C = mat(4, undirected=[(0, 1), (1, 2), (0, 2), (0, 3)])
    assert is_clique(C, nodes) is expected


# ============================================================
# Semidirected paths
# ============================================================


def test_semidirected_path_follows_arrows_and_undirected_edges():
    # 0 -> 1 -- 2 and 3 -> 2. A path from 2 to 3 would go against the arrow.
    C = mat(4, directed=[(0, 1), (3, 2)], undirected=[(1, 2)])
    assert exists_semidirected_path(C, 0, 2)
    assert not exists_semidirected_path(C, 2, 3)
    assert not exists_semidirected_path(C, 2, 0)


def test_semidirected_path_respects_block_set():
    C = mat(3, directed=[(0, 1), (1, 2)])
    assert exists_semidirected_path(C, 0, 2)
    assert not exists_semidirected_path(C, 0, 2, block={1})


# ============================================================
# Meek's orientation rules R1-R4
# ============================================================


def test_meek_r1_orients_away_from_collider_free_triple():
    # 0 -> 1, 1 -- 2, and 0 and 2 nonadjacent imply 1 -> 2.
    C = meek_rules(mat(3, directed=[(0, 1)], undirected=[(1, 2)]))
    assert is_directed(C, 1, 2)


def test_meek_r2_closes_directed_path():
    # 0 -> 1 -> 2, 0 -- 2  =>  0 -> 2
    C = meek_rules(mat(3, directed=[(0, 1), (1, 2)], undirected=[(0, 2)]))
    assert is_directed(C, 0, 2)


def test_meek_r3_orients_from_two_nonadjacent_parents():
    # a=0 -- b=1, 0 -- 2, 0 -- 3, 2 -> 1, and 3 -> 1. Since 2 and 3 are nonadjacent, 0 -> 1.
    C = meek_rules(
        mat(4, directed=[(2, 1), (3, 1)], undirected=[(0, 1), (0, 2), (0, 3)])
    )
    assert is_directed(C, 0, 1)


def test_meek_r4_orients_through_directed_chain():
    # a=0 -- b=1, 0 -- 2, 2 -> 3, 3 -> 1. Nodes 1 and 2 are nonadjacent, and 0 is adjacent to 3, implying 0 -> 1.
    C = meek_rules(
        mat(4, directed=[(2, 3), (3, 1)], undirected=[(0, 1), (0, 2), (0, 3)])
    )
    assert is_directed(C, 0, 1)


def test_meek_is_idempotent_on_a_completed_cpdag():
    C = dag_to_cpdag(mat(4, directed=[(0, 1), (2, 1), (1, 3)]))
    assert np.array_equal(meek_rules(C), C)


def test_meek_leaves_an_unorientable_chain_alone():
    # On 0 -- 1 -- 2 with no collider, neither direction is compelled.
    C = meek_rules(mat(3, undirected=[(0, 1), (1, 2)]))
    assert is_undirected(C, 0, 1)
    assert is_undirected(C, 1, 2)


# ============================================================
# Extending a PDAG to a DAG
# ============================================================


def test_pdag_to_dag_returns_an_already_directed_dag_unchanged():
    dag = mat(3, directed=[(0, 1), (1, 2)])
    assert np.array_equal(pdag_to_dag(dag), dag)


def test_pdag_to_dag_extends_a_chain_without_inventing_a_collider():
    """Accept any orientation that gives a consistent extension.

    Check the defining properties instead of a particular matrix: the skeleton
    and the set of unshielded colliders must be preserved. The chain has none.
    """
    C = mat(3, undirected=[(0, 1), (1, 2)])
    dag = pdag_to_dag(C)
    assert dag is not None

    skeleton = {(0, 1), (1, 2)}
    got = {(min(i, j), max(i, j)) for i in range(3) for j in range(3) if dag[i, j]}
    assert got == skeleton

    colliders = [
        y
        for y in range(3)
        for x in range(3)
        for z in range(3)
        if x < z and dag[x, y] and dag[z, y] and not (dag[x, z] or dag[z, x])
    ]
    assert colliders == []


def test_pdag_to_dag_rejects_an_unextendable_pdag():
    """With 0 -> 3, 1 -> 2, 2 -- 3, either orientation creates a v-structure.

    With 2 -> 3, node 3 has nonadjacent parents 0 and 2. With 3 -> 2, node 2
    has nonadjacent parents 1 and 3. A consistent extension must preserve the
    v-structures, so none exists. This graph has no directed cycle: the test
    exercises the Dor-Tarsi condition beyond acyclicity.
    """
    C = mat(4, directed=[(0, 3), (1, 2)], undirected=[(2, 3)])
    assert pdag_to_dag(C) is None


def test_pdag_to_dag_rejects_a_directed_cycle():
    C = mat(3, directed=[(0, 1), (1, 2), (2, 0)])
    assert pdag_to_dag(C) is None


def test_pdag_to_dag_matches_exhaustive_search_on_all_4_node_pdags():
    """Check all 4^6 edge configurations on four nodes by brute force.

    For each PDAG, try every orientation of its undirected edges. A consistent
    extension exists if any orientation is acyclic and preserves the set of
    v-structures. pdag_to_dag must return None exactly when no extension exists,
    and otherwise return a consistent DAG.
    """
    import itertools

    p = 4
    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]

    def vstructs_dag(dag):
        out = set()
        for y in range(p):
            pa = [x for x in range(p) if dag[x, y]]
            for i, x in enumerate(pa):
                for z in pa[i + 1 :]:
                    if not (dag[x, z] or dag[z, x]):
                        out.add((min(x, z), y, max(x, z)))
        return out

    def vstructs_pdag(C):
        out = set()
        for y in range(p):
            pa = [x for x in range(p) if is_directed(C, x, y)]
            for i, x in enumerate(pa):
                for z in pa[i + 1 :]:
                    if not is_adjacent(C, x, z):
                        out.add((min(x, z), y, max(x, z)))
        return out

    def acyclic(dag):
        indeg = dag.sum(0).copy()
        remaining = set(range(p))
        while remaining:
            ready = [v for v in remaining if indeg[v] == 0]
            if not ready:
                return False
            v = ready[0]
            remaining.discard(v)
            for w in range(p):
                if dag[v, w]:
                    indeg[w] -= 1
            indeg[v] = -1
        return True

    for cfg in itertools.product([0, 1, 2, 3], repeat=len(pairs)):
        C = np.zeros((p, p), dtype=np.int8)
        for (i, j), code in zip(pairs, cfg, strict=False):
            if code == 1:
                C[i, j] = 1
            elif code == 2:
                C[j, i] = 1
            elif code == 3:
                C[i, j] = C[j, i] = 1

        base = np.zeros((p, p), dtype=np.int8)
        for i in range(p):
            for j in range(p):
                if is_directed(C, i, j):
                    base[i, j] = 1
        undirected_pairs = [
            (i, j) for (i, j), code in zip(pairs, cfg, strict=False) if code == 3
        ]
        target = vstructs_pdag(C)

        extendable = False
        for bits in itertools.product([0, 1], repeat=len(undirected_pairs)):
            cand = base.copy()
            for (i, j), bit in zip(undirected_pairs, bits, strict=False):
                if bit:
                    cand[i, j] = 1
                else:
                    cand[j, i] = 1
            if acyclic(cand) and vstructs_dag(cand) == target:
                extendable = True
                break

        got = pdag_to_dag(C)
        if not extendable:
            assert got is None, f"expected None for {cfg}"
        else:
            assert got is not None, f"an extension exists but none was found: {cfg}"
            assert acyclic(got), f"the extension contains a cycle: {cfg}"
            assert vstructs_dag(got) == target, f"v-structures changed: {cfg}"


def test_pdag_to_dag_preserves_existing_arrowheads():
    C = mat(4, directed=[(0, 1), (2, 1)], undirected=[(1, 3)])
    dag = pdag_to_dag(C)
    assert dag is not None
    assert dag[0, 1]
    assert dag[2, 1]
    assert not dag[1, 0]
    assert not dag[1, 2]


# ============================================================
# DAG to CPDAG conversion and completion
# ============================================================


def test_dag_to_cpdag_keeps_collider_and_frees_reversible_edges():
    # The collider 0 -> 2 <- 1 is invariant. Both arrowheads must remain.
    C = dag_to_cpdag(mat(3, directed=[(0, 2), (1, 2)]))
    assert is_directed(C, 0, 2)
    assert is_directed(C, 1, 2)

    # Both edges in the chain 0 -> 1 -> 2 are reversible.
    C = dag_to_cpdag(mat(3, directed=[(0, 1), (1, 2)]))
    assert is_undirected(C, 0, 1)
    assert is_undirected(C, 1, 2)


def test_chain_and_fork_share_one_cpdag():
    """The chain and the fork belong to the same equivalence class."""
    chain = dag_to_cpdag(mat(3, directed=[(0, 1), (1, 2)]))
    fork = dag_to_cpdag(mat(3, directed=[(1, 0), (1, 2)]))
    assert np.array_equal(chain, fork)


def test_complete_pdag_is_idempotent():
    C = dag_to_cpdag(mat(4, directed=[(0, 1), (0, 2), (1, 3), (2, 3)]))
    assert np.array_equal(complete_pdag(C), C)


def test_complete_pdag_reports_failure_on_unextendable_input():
    C = mat(4, directed=[(0, 3), (1, 2)], undirected=[(2, 3)])
    assert complete_pdag(C) is None
