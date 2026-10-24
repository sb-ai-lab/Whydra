"""Protected pairs must stop at the first separating set during skeleton discovery.

Under stable=True the skeleton enumerates every conditioning set of a given depth
to collect the union of sepsets.  A pair protected by background knowledge keeps
its edge and its sepsets are discarded, so it must stop at the first separating
set, as ``standalone_pc_stable.skeleton_discovery`` does.  Before the fix
``SkeletonDiscovery`` and ``parallel_pc_stable.skeleton_discovery`` ran 8 CI
tests on the protected pair in the scenario below instead of 4.  The result
graph does not depend on this; only the number of CI tests does.
"""

import numpy as np
import pytest

from tests.conftest import generate_linear_sem_data
from whydra.algorithms.graph_core import CausalGraph, CIT
from whydra.algorithms.standalone import parallel_pc_stable, standalone_pc_stable
from whydra.algorithms.standalone.skeleton import SkeletonDiscovery

NODE_NAMES = ["A", "X", "B", "Y", "Z"]
N_VARS = 5
ALPHA = 0.05
N_SAMPLES = 3000
SEED = 7
WEIGHT = 0.8
NOISE_STD = 1.0
# A->X, A->B, B->Y, A->Z, B->Z
TRUE_EDGES = [(0, 1), (0, 2), (2, 3), (0, 4), (2, 4)]
# X-Y, not adjacent in the true DAG
PROTECTED_PAIR = (1, 3)
EXPECTED_PROTECTED_CI_CALLS = 4

TRUE_WEIGHTS = np.zeros((N_VARS, N_VARS))
for _i, _j in TRUE_EDGES:
    TRUE_WEIGHTS[_i, _j] = WEIGHT

DATA = generate_linear_sem_data(
    TRUE_WEIGHTS, n_samples=N_SAMPLES, noise_std=NOISE_STD, seed=SEED
)


class CountingCIT:
    """Wrap a CI callable and count calls involving a specific pair.

    Args:
        cit: The conditional independence test callable to wrap.
        pair: The variable pair (indices) whose CI calls are counted.

    Returns:
        None.
    """

    def __init__(self, cit, pair):
        """Store the wrapped CI test and the pair to monitor.

        Args:
            cit: The conditional independence test callable.
            pair: The variable pair (indices) to count calls for.

        Returns:
            None.
        """
        self.cit = cit
        self.pair = frozenset(pair)
        self.calls = 0

    def __call__(self, x, y, S):
        """Delegate to the wrapped CI test, counting calls on the monitored pair.

        Args:
            x: First variable index.
            y: Second variable index.
            S: Conditioning set of variable indices.

        Returns:
            The result of the wrapped CI test.
        """
        if frozenset((int(x), int(y))) == self.pair:
            self.calls += 1
        return self.cit(x, y, S)


def _make_counted_graph():
    """Create a CausalGraph whose CI test counts calls on the protected pair.

    Args:
        None.

    Returns:
        tuple[CausalGraph, CountingCIT]: The graph and the counting wrapper.
    """
    cg = CausalGraph(N_VARS, NODE_NAMES)
    cg.set_ind_test(CIT(DATA, method="fisherz"))
    counter = CountingCIT(cg.test, PROTECTED_PAIR)
    cg.test = counter
    return cg, counter


def _run_standalone(cg):
    """Run the standalone skeleton discovery entry point.

    Args:
        cg: The CausalGraph to run skeleton discovery on.

    Returns:
        None.
    """
    standalone_pc_stable.skeleton_discovery(
        cg, ALPHA, True, False, False, protected_edges=[PROTECTED_PAIR]
    )


def _run_skeleton_threads_j1(cg):
    """Run the SkeletonDiscovery entry point with one job.

    Args:
        cg: The CausalGraph to run skeleton discovery on.

    Returns:
        None.
    """
    SkeletonDiscovery(
        cg,
        alpha=ALPHA,
        stable=True,
        n_jobs=1,
        show_progress=False,
        protected_edges=[PROTECTED_PAIR],
    ).run()


def _run_skeleton_threads_j2(cg):
    """Run the SkeletonDiscovery entry point with two jobs.

    Args:
        cg: The CausalGraph to run skeleton discovery on.

    Returns:
        None.
    """
    SkeletonDiscovery(
        cg,
        alpha=ALPHA,
        stable=True,
        n_jobs=2,
        show_progress=False,
        protected_edges=[PROTECTED_PAIR],
    ).run()


def _run_parallel_j1(cg):
    """Run the parallel_pc_stable skeleton discovery entry point with one job.

    Args:
        cg: The CausalGraph to run skeleton discovery on.

    Returns:
        None.
    """
    parallel_pc_stable.skeleton_discovery(
        cg, ALPHA, True, False, False, n_jobs=1, protected_edges=[PROTECTED_PAIR]
    )


# Note: loky n_jobs>1 for parallel_pc_stable is NOT included (the counter lives
# in worker processes).
ENTRY_POINTS = [
    ("standalone", _run_standalone),
    ("skeleton_threads_j1", _run_skeleton_threads_j1),
    ("skeleton_threads_j2", _run_skeleton_threads_j2),
    ("parallel_j1", _run_parallel_j1),
]


@pytest.mark.parametrize("name,run", ENTRY_POINTS, ids=[e[0] for e in ENTRY_POINTS])
def test_protected_pair_stops_at_first_separating_set(name, run):
    """Verify each entry point stops at the first separating set for the protected pair.

    Args:
        name: Identifier of the entry point under test.
        run: Callable executing skeleton discovery on a CausalGraph.

    Returns:
        None.
    """
    cg, counter = _make_counted_graph()
    run(cg)
    assert counter.calls == EXPECTED_PROTECTED_CI_CALLS, (
        f"{name}: expected {EXPECTED_PROTECTED_CI_CALLS} CI calls on the protected "
        f"pair, got {counter.calls}"
    )
    assert cg.G.graph[1, 3] != 0
    assert cg.G.graph[3, 1] != 0
    assert cg.sepset[1, 3] is None
    assert cg.sepset[3, 1] is None


@pytest.mark.parametrize(
    "name,run",
    [e for e in ENTRY_POINTS if e[0] != "standalone"],
    ids=[e[0] for e in ENTRY_POINTS if e[0] != "standalone"],
)
def test_protection_does_not_change_graph(name, run):
    """Verify that the early stop does not alter the resulting graph.

    Args:
        name: Identifier of the entry point under test.
        run: Callable executing skeleton discovery on a CausalGraph.

    Returns:
        None.
    """
    ref_cg, _ = _make_counted_graph()
    _run_standalone(ref_cg)
    cg, _ = _make_counted_graph()
    run(cg)
    assert np.array_equal(ref_cg.G.graph, cg.G.graph)


def test_unprotected_pair_still_enumerates_all_sets():
    """Verify that ordinary pairs still enumerate every conditioning set in stable mode.

    Args:
        None.

    Returns:
        None.
    """
    cg, counter = _make_counted_graph()
    SkeletonDiscovery(
        cg, alpha=ALPHA, stable=True, n_jobs=1, show_progress=False
    ).run()
    assert counter.calls > EXPECTED_PROTECTED_CI_CALLS
