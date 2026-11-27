import numpy as np

from whydra.algorithms.standalone.pag_rules import (
    PAG_ARROW,
    PAG_CIRCLE,
    legal_path,
    qreach,
)


def test_reachability_can_reenter_a_vertex_with_a_different_predecessor():
    marks = np.zeros((5, 5), dtype=int)
    for left, right in [(0, 1), (0, 4), (1, 2), (1, 3), (1, 4), (3, 4)]:
        marks[left, right] = marks[right, left] = PAG_CIRCLE
    marks[0, 1] = marks[2, 1] = PAG_ARROW
    assert qreach(2, marks) == [0, 1, 3, 4]


def test_direction_of_closure_and_arrowheads_on_an_asymmetric_matrix():
    marks = np.zeros((4, 4), dtype=int)
    marks[0, 1] = marks[1, 2] = marks[2, 3] = PAG_CIRCLE
    marks[2, 0] = PAG_ARROW
    assert not legal_path(0, 1, 2, marks)
    assert qreach(0, marks) == [1]
    marks[0, 1] = marks[2, 1] = PAG_ARROW
    assert legal_path(0, 1, 2, marks)
    assert qreach(0, marks) == [1, 2]
    marks[0, 2] = PAG_CIRCLE
    marks[2, 1] = 0
    assert legal_path(0, 1, 2, marks)


def test_loops_do_not_add_the_source_to_reachability():
    marks = np.full((3, 3), PAG_CIRCLE)
    assert qreach(1, marks) == [0, 2]
    assert not legal_path(0, 1, 0, marks)
    assert qreach(1, np.diag([PAG_ARROW] * 3)) == []
