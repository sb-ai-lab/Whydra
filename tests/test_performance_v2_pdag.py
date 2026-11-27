"""Ordered mask operations against the retained scalar graph oracles."""

from itertools import combinations

import numpy as np
import pytest

from whydra.algorithms.score_based import ges, pdag


def oracle_completion(matrix):
    dag = pdag._pdag_to_dag_reference(matrix)
    if dag is None:
        return None
    result = ((dag != 0) | (dag.T != 0)).astype(np.int8)
    for y in range(len(dag)):
        for x, z in combinations(np.flatnonzero(dag[:, y]), 2):
            if not result[x, z]:
                result[y, x] = result[y, z] = 0
    return pdag._meek_rules_reference(result)


@pytest.mark.parametrize("block", range(16))
def test_all_four_node_mixed_graphs_match_scalar_closure_and_extension(block):
    pairs = list(combinations(range(4), 2))
    for value in range(block * 256, (block + 1) * 256):
        matrix = np.zeros((4, 4), dtype=np.int8)
        for a, b in pairs:
            kind, value = value % 4, value // 4
            matrix[a, b], matrix[b, a] = ((0, 0), (1, 0), (0, 1), (1, 1))[kind]
        np.testing.assert_array_equal(
            pdag.meek_rules(matrix), pdag._meek_rules_reference(matrix)
        )
        expected = pdag._pdag_to_dag_reference(matrix)
        actual = pdag.pdag_to_dag(matrix)
        if expected is None:
            assert actual is None
        else:
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("seed", range(16))
def test_larger_mixed_graphs_and_inplace_updates(seed):
    rng = np.random.default_rng(seed)
    size = 6 + seed
    matrix = (rng.random((size, size)) < 0.2).astype(np.int8)
    np.fill_diagonal(matrix, 0)
    untouched = matrix.copy()
    expected = pdag._meek_rules_reference(matrix)
    result = pdag.meek_rules(matrix, inplace=True)
    assert result is matrix
    np.testing.assert_array_equal(result, expected)
    completion = oracle_completion(untouched)
    actual = pdag.complete_pdag(untouched)
    if completion is None:
        assert actual is None
    else:
        np.testing.assert_array_equal(actual, completion)


def test_affected_component_handles_component_merge_and_split():
    matrix = np.zeros((9, 9), dtype=np.int8)
    for a, b in ((0, 1), (1, 2), (3, 4), (6, 7), (7, 8)):
        matrix[a, b] = matrix[b, a] = 1
    before = pdag.complete_pdag(matrix)
    added = ges.apply_insert(before.copy(), 2, 3, ())
    np.testing.assert_array_equal(
        pdag.complete_pdag(added, affected_nodes=(2, 3)), oracle_completion(added)
    )
    removed = ges.apply_delete(pdag.complete_pdag(added), 2, 3, ())
    np.testing.assert_array_equal(
        pdag.complete_pdag(removed, affected_nodes=(2, 3)), oracle_completion(removed)
    )


@pytest.mark.parametrize("size", [0, 1, 2])
def test_empty_and_tiny_graphs(size):
    matrix = np.zeros((size, size), dtype=np.int8)
    np.testing.assert_array_equal(pdag.complete_pdag(matrix), matrix)
