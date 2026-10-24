"""Regression tests for compact PC separating-member unions, like causal-learn."""

import pytest

from tests.test_pc_sepset_union import _ENTRY_POINTS, _make_causal_graph
from whydra.algorithms.graph_core import CausalGraph
from whydra.algorithms.standalone import parallel_pc_stable, standalone_pc_stable
from whydra.algorithms.standalone.skeleton import SkeletonDiscovery

N_VARS = 18
DEPTH = 8
ALPHA = 0.05
SMALL_N_VARS = 4
NODE_NAMES = [f"X{i}" for i in range(N_VARS)]
SMALL_NODE_NAMES = NODE_NAMES[:SMALL_N_VARS]


class AlwaysIndependent:
    """Picklable CI stub that separates every pair with every conditioning set."""

    def __call__(
        self,
        x,
        y,
        S,
    ) -> float:
        """Return independence for any pair and conditioning set.

        Args:
            x: first node index.
            y: second node index.
            S: conditioning set.

        Returns:
            float: p-value 1.0.
        """
        return 1.0


def test_parallel_worker_records_one_union_per_pair():
    """A dense depth-eight search returns 17 records instead of 218,790.

    Returns:
        None: asserts each record contains the full candidate complement.
    """
    cg = CausalGraph(N_VARS, NODE_NAMES)
    results = parallel_pc_stable._skeleton_discovery_worker(
        [0], cg.G.graph, AlwaysIndependent(), DEPTH, ALPHA, True, frozenset()
    )

    assert len(results) == 1
    x, removals, records = results[0]
    assert x == 0
    assert removals == [(0, y) for y in range(1, N_VARS)]
    assert records == [
        (0, y, tuple(sorted(set(range(N_VARS)) - {0, y})))
        for y in range(1, N_VARS)
    ]
    assert all(type(v) is int for _, _, members in records for v in members)


def test_pc_node_records_one_union_per_pair():
    """The sequential PC path stores one symmetric union per dense pair.

    Returns:
        None: asserts every separated cell has a single tuple of Python ints.
    """
    cg = CausalGraph(N_VARS, NODE_NAMES)
    cg.set_ind_test(AlwaysIndependent())
    discovery = SkeletonDiscovery(cg, alpha=ALPHA, stable=True, show_progress=False)
    discovery._process_node_at_depth(0, cg.neighbors(0), DEPTH)

    for y in range(1, N_VARS):
        expected = [tuple(sorted(set(range(N_VARS)) - {0, y}))]
        assert cg.sepset[0, y] == expected
        assert cg.sepset[y, 0] == expected
        assert all(type(v) is int for v in cg.sepset[0, y][0])


@pytest.mark.parametrize(
    "run_skeleton",
    [entry[1] for entry in _ENTRY_POINTS],
    ids=[entry[0] for entry in _ENTRY_POINTS],
)
def test_full_runs_keep_sepsets_compact(run_skeleton):
    """All PC entry points compact multiple separators into a single tuple.

    Args:
        run_skeleton: skeleton entry point to run on the four-node DAG data.

    Returns:
        None: asserts the union and the one-tuple invariant for all cells.
    """
    cg = _make_causal_graph()
    run_skeleton(cg)

    assert cg.sepset[1, 3] == [(0, 2)]
    assert cg.sepset[3, 1] == [(0, 2)]
    for cell in cg.sepset.flat:
        if cell is not None:
            assert len(cell) == 1
            assert cell[0] == tuple(sorted(set(cell[0])))
            assert all(type(v) is int for v in cell[0])


@pytest.mark.parametrize("stable", [True, False])
@pytest.mark.parametrize(
    "implementation",
    ["standalone", "parallel", "pc"],
)
def test_depth_zero_preserves_empty_separation(implementation, stable):
    """Empty separation is stored as [()] in both stable and non-stable PC.

    Args:
        implementation: skeleton implementation to exercise.
        stable: whether to enumerate all conditioning sets.

    Returns:
        None: asserts off-diagonal empty unions and untouched diagonal cells.
    """
    cg = CausalGraph(SMALL_N_VARS, SMALL_NODE_NAMES)
    cg.set_ind_test(AlwaysIndependent())
    if implementation == "pc":
        SkeletonDiscovery(cg, stable=stable, show_progress=False).run()
    else:
        module = (
            standalone_pc_stable if implementation == "standalone"
            else parallel_pc_stable
        )
        kwargs = {"n_jobs": 1} if implementation == "parallel" else {}
        module.skeleton_discovery(cg, ALPHA, stable, False, False, **kwargs)

    for x in range(SMALL_N_VARS):
        for y in range(SMALL_N_VARS):
            assert cg.sepset[x, y] == (None if x == y else [()])


@pytest.mark.parametrize("implementation", ["standalone", "parallel", "pc"])
def test_merges_members_across_calls_and_directions(implementation):
    """Repeated updates from opposite sides produce a single sorted union.

    Args:
        implementation: local PC merge helper to exercise.

    Returns:
        None: asserts symmetric compact storage after overlapping updates.
    """
    cg = CausalGraph(SMALL_N_VARS, SMALL_NODE_NAMES)
    if implementation == "pc":
        merge = SkeletonDiscovery(cg, show_progress=False)._merge_sepset_union
    else:
        module = (
            standalone_pc_stable if implementation == "standalone"
            else parallel_pc_stable
        )

        def merge(x, y, members):
            """Apply a module-local merge helper to the test graph.

            Args:
                x: first node index.
                y: second node index.
                members: separating-set members to merge.

            Returns:
                None: updates the test graph in place.
            """
            module._append_sepset(cg, x, y, members)

    merge(1, 3, ())
    merge(1, 3, (2,))
    merge(3, 1, (2, 0))
    assert cg.sepset[1, 3] == [(0, 2)]
    assert cg.sepset[3, 1] == [(0, 2)]
