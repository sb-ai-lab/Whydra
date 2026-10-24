"""Tests that PC-stable skeleton discovery records one tuple with the union of members
of ALL separating sets of the current depth, matching causal-learn's
``pc(stable=True, uc_rule=0, uc_priority=2)`` behaviour.

Why the union matters: collider orientation treats an unshielded triple
(i, k, j) as a collider unless k appears in ANY separating set recorded
for the pair (i, j).  If the skeleton phase stops at the first separating
set, spurious colliders appear and the final CPDAG diverges from
causal-learn.  These tests pin the union semantics across all three
skeleton implementations.
"""

import numpy as np
import pytest

from tests.conftest import generate_linear_sem_data
from whydra.algorithms.graph_core import CausalGraph, CIT
from whydra.algorithms.standalone import parallel_pc_stable, standalone_pc_stable
from whydra.algorithms.standalone.skeleton import SkeletonDiscovery

NODE_NAMES = ["A", "X", "B", "Y"]
N_VARS = 4
ALPHA = 0.05
N_SAMPLES = 3000
SEED = 7
WEIGHT = 0.8
NOISE_STD = 1.0

# True DAG: A -> X, A -> B, B -> Y with node order A=0, X=1, B=2, Y=3.
TRUE_WEIGHTS = np.zeros((N_VARS, N_VARS))
TRUE_WEIGHTS[0, 1] = WEIGHT
TRUE_WEIGHTS[0, 2] = WEIGHT
TRUE_WEIGHTS[2, 3] = WEIGHT

TRUE_SKELETON = {
    frozenset((0, 1)),
    frozenset((0, 2)),
    frozenset((2, 3)),
}

DATA = generate_linear_sem_data(
    TRUE_WEIGHTS,
    n_samples=N_SAMPLES,
    noise_std=NOISE_STD,
    seed=SEED,
)


def _make_causal_graph():
    """Build a fresh complete CausalGraph with a Fisher-z CI test.

    Returns:
        CausalGraph: graph wired to the module-level DATA.
    """
    cg = CausalGraph(N_VARS, NODE_NAMES)
    cg.set_ind_test(CIT(DATA, method="fisherz"))
    return cg


_ENTRY_POINTS = [
    (
        "standalone",
        lambda cg: standalone_pc_stable.skeleton_discovery(cg, ALPHA, True, False, False),
    ),
    (
        "parallel_pc_stable_jobs_1",
        lambda cg: parallel_pc_stable.skeleton_discovery(cg, ALPHA, True, False, False, 1),
    ),
    (
        "parallel_pc_stable_jobs_4",
        lambda cg: parallel_pc_stable.skeleton_discovery(cg, ALPHA, True, False, False, 4),
    ),
    (
        "pc_algo_jobs_1",
        lambda cg: SkeletonDiscovery(
            cg,
            alpha=ALPHA,
            stable=True,
            n_jobs=1,
            show_progress=False,
            verbose=False,
        ).run(),
    ),
    (
        "pc_algo_jobs_4",
        lambda cg: SkeletonDiscovery(
            cg,
            alpha=ALPHA,
            stable=True,
            n_jobs=4,
            show_progress=False,
            verbose=False,
        ).run(),
    ),
]


def _recorded_sepset_tuples(cg, i, j):
    """Return the set of recorded sepset tuples for the pair (i, j).

    Args:
        cg: causal graph after skeleton discovery.
        i: first node index.
        j: second node index.

    Returns:
        set[tuple]: set of recorded conditioning sets (empty if None).
    """
    if cg.sepset[i, j] is None:
        return set()
    return {tuple(int(v) for v in S) for S in cg.sepset[i, j]}


@pytest.mark.parametrize(
    "run_skeleton",
    [entry[1] for entry in _ENTRY_POINTS],
    ids=[entry[0] for entry in _ENTRY_POINTS],
)
def test_sepset_union_records_all_separators(run_skeleton):
    """sepset[X, Y] must contain the single union tuple (A, B), symmetrically.

    At depth 1 the stable snapshot of X's neighbours is {A, B, Y}, so both
    S=(A,) and S=(B,) separate X from Y; the stable (union) semantics
    require their members to be merged into one tuple from both sides.
    """
    cg = _make_causal_graph()
    run_skeleton(cg)

    assert cg.sepset[1, 3] == [(0, 2)]
    assert cg.sepset[3, 1] == [(0, 2)]


def test_sepsets_identical_across_entry_points():
    """All three skeleton implementations must record identical sepsets."""
    reference = None
    for entry_id, run_skeleton in _ENTRY_POINTS:
        cg = _make_causal_graph()
        run_skeleton(cg)
        snapshot = {
            (i, j): frozenset(_recorded_sepset_tuples(cg, i, j))
            for i in range(N_VARS)
            for j in range(N_VARS)
        }
        if reference is None:
            reference = snapshot
        else:
            assert snapshot == reference, f"sepset mismatch for {entry_id}"


@pytest.mark.parametrize(
    "run_skeleton",
    [entry[1] for entry in _ENTRY_POINTS],
    ids=[entry[0] for entry in _ENTRY_POINTS],
)
def test_skeleton_adjacency_matches_true_skeleton(run_skeleton):
    """The recovered skeleton adjacency must equal A-X, A-B, B-Y."""
    cg = _make_causal_graph()
    run_skeleton(cg)

    adjacency = cg.G.graph != 0
    edges = {
        frozenset((i, j))
        for i in range(N_VARS)
        for j in range(N_VARS)
        if i != j and adjacency[i, j]
    }
    assert edges == TRUE_SKELETON
