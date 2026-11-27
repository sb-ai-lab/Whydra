from collections import deque

import pytest

from tests._pag_cases import CPU_MARKS, edge_matrix
from whydra.algorithms.standalone.pag_rules import (
    faith_check,
    minDiscPath,
    minUncovCircPath,
    minUncovPdPath,
)
from whydra.algorithms.standalone.rfci_algo import _min_discriminating_path


def test_discriminating_witness_keeps_first_branch_at_equal_distance():
    graph = edge_matrix(
        7, "0 <-> 3, 0 <-> 4, 3 --> 2, 4 --> 2, 6 --> 3, 5 --> 4", CPU_MARKS
    )
    assert minDiscPath(graph, 0, 1, 2) == [6, 3, 0, 1, 2]
    graph[3, 6] = 0
    assert minDiscPath(graph, 0, 1, 2) == [5, 4, 0, 1, 2]
    assert _min_discriminating_path(graph, 0, 1, 2) == [6, 3, 0, 1, 2]


def test_discriminating_witness_admits_second_collider_arrival():
    graph = edge_matrix(
        7,
        "0 <-> 3, 0 <-> 4, 3 --> 2, 4 --> 2, 5 --> 3, 5 <-> 4, 5 --> 2, 6 --> 5",
        CPU_MARKS,
    )
    assert minDiscPath(graph, 0, 1, 2) == [6, 5, 4, 0, 1, 2]
    graph[5, 6] = graph[6, 5] = 0
    assert minDiscPath(graph, 0, 1, 2) == []
    assert _min_discriminating_path(graph, 0, 1, 2) is None


@pytest.mark.parametrize(
    ("ambiguous", "expected"),
    [
        (set(), [0, 1, 2, 4, 5]),
        ({(4, 2, 1)}, [0, 1, 3, 4, 5]),
        ({(4, 5, 0)}, []),
        ({(1, 0, 5)}, []),
    ],
)
def test_uncovered_search_selects_first_admissible_route(ambiguous, expected):
    graph = edge_matrix(
        6, "0 o-o 1, 1 o-o 2, 1 o-o 3, 2 o-o 4, 3 o-o 4, 4 o-o 5", CPU_MARKS
    )
    assert minUncovCircPath(6, graph, (0, 1, 4, 5), ambiguous) == expected
    assert minUncovPdPath(6, graph, 0, 1, 5, ambiguous) == expected


@pytest.mark.parametrize("path_type", [list, tuple, deque])
def test_faith_check_includes_both_wraparound_triples(path_type):
    path = path_type([2, 3, 0, 1])
    assert faith_check(path, {(3, 1, 2)}, 4)
    assert not faith_check(path, {(3, 2, 1)}, 4)
    assert not faith_check(path, {(2, 1, 0)}, 4)
