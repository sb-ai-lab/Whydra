from dataclasses import dataclass

import numpy as np


@dataclass
class ReachabilityCase:
    name: str
    skeleton: np.ndarray
    separators: dict
    ambiguous: set
    marked: np.ndarray
    reachable: tuple


def reachability_cases():
    chain = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
    both = np.array([[0, 2, 0, 0], [1, 0, 2, 0], [0, 2, 0, 1], [0, 0, 2, 0]])
    second = np.array([[0, 1, 0, 0], [1, 0, 2, 0], [0, 1, 0, 1], [0, 0, 2, 0]])
    full = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
    blocked = ((1,), (0, 2, 3), (1, 3), (1, 2))
    diamond = np.array(
        [
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    return (
        ReachabilityCase("two-colliders", chain, {}, set(), both, full),
        ReachabilityCase("separator", chain, {(2, 0): {1}}, set(), second, blocked),
        ReachabilityCase("ambiguous", chain, {}, {(2, 1, 0)}, second, blocked),
        ReachabilityCase(
            "linked-triangles", diamond, {(0, 3): {1, 2}}, set(), diamond, (*full, ())
        ),
    )
