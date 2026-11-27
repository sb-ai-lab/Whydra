import numpy as np
import pytest

from tests._pag_cases import (
    CASES,
    CIRCLE_CYCLE,
    CPU_MARKS,
    DIRECTED_ROUTE,
    DISCRIMINATING,
    TWO_ROUTES,
    edge_matrix,
)
from whydra.algorithms.standalone.pag_rules import (
    _orientation_guard,
    _Orientations,
    minDiscPath,
    minUncovCircPath,
    minUncovPdPath,
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_rule_changes_exactly_the_expected_endpoints(case):
    graph, expected = case.matrices(CPU_MARKS)
    engine = _Orientations(graph, _orientation_guard(None, None), case.ambiguous)
    engine.run(case.rule, case.sepset)
    np.testing.assert_array_equal(graph, expected)


def test_r4_discriminating_path_includes_both_collider_parents():
    graph = edge_matrix(5, DISCRIMINATING, CPU_MARKS)
    assert minDiscPath(graph, 2, 3, 4) == [0, 1, 2, 3, 4]


def test_r5_uncovered_circle_route():
    graph = edge_matrix(5, CIRCLE_CYCLE, CPU_MARKS)
    assert minUncovCircPath(5, graph, (0, 1, 3, 4), ()) == [0, 1, 2, 3, 4]


def test_r9_uncovered_route():
    graph = edge_matrix(4, DIRECTED_ROUTE, CPU_MARKS)
    assert minUncovPdPath(4, graph, 0, 1, 3, ()) == [0, 1, 2, 3]


def test_r10_uncovered_routes_have_distinct_first_hops():
    graph = edge_matrix(8, TWO_ROUTES, CPU_MARKS)
    assert minUncovPdPath(8, graph, 0, 1, 3, ()) == [0, 1, 2, 3]
    assert minUncovPdPath(8, graph, 0, 4, 6, ()) == [0, 4, 5, 6]
