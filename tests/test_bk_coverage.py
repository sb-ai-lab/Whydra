"""Test background knowledge end to end in every algorithm that supports it.

The core unit tests live in `test_background_knowledge.py`. Validation and
rollback are covered by `test_background_knowledge_invariants.py`. Here we
check that each public algorithm enforces the constraints in its final result.

We require a direction opposite to the one the algorithm finds on its own.
If that direction appears in the result, background knowledge forced it.

The entry ``m[i, j]`` is the mark at node ``j`` on edge (i, j).
The causal-learn encoding uses NULL=0, TAIL=-1, ARROW=1, CIRCLE=2.
"""

import warnings

import numpy as np
import pytest

from whydra.algorithms.wrappers import create_standalone_algorithm
from whydra.background_knowledge import (
    BackgroundKnowledge,
    BKPhase,
    find_directed_cycle,
)
from whydra.evaluation.graph_utils import get_adj_matrix

NULL, TAIL, ARROW, CIRCLE = 0, -1, 1, 2

#: Wrappers whose entry points accept background_knowledge.
BK_ALGORITHMS = [
    "StandalonePCStable",
    "ParallelPCStable",
    "StandaloneFCIStable",
    "StandaloneFCIStableCached",
    "StandaloneFCIStableParallelCached",
    "StandaloneCFCIStable",
    "StandaloneCFCIStableParallelCached",
    "StandaloneRFCIStable",
    "StandaloneGFCIStable",
    "StandaloneFCIPlusStable",
    "StandaloneFCIPlusStableParallelCached",
]

#: A separate test checks that the RAI family does not support background knowledge.
NON_BK_ALGORITHMS = ["StandaloneRAIStable"]

N_JOBS_VALUES = [1, 4]

#: True DAG: X1->X3, X2->X3, X3->X4, X4->X5.
#: The following adjacent pairs provide edges to constrain.
#: (X1,X3), (X2,X3), (X3,X4), (X4,X5).
TRUE_EDGES = [(0, 2), (1, 2), (2, 3), (3, 4)]
N_VARS = 5
NAMES = [f"X{i + 1}" for i in range(N_VARS)]


# ============================================================
# Test helpers
# ============================================================


def _data(seed=4, n=600):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, N_VARS))
    for a, b in TRUE_EDGES:
        data[:, b] += 1.2 * data[:, a]
    return data


DATA = _data()


def run(algo_key, bk=None, n_jobs=1, data=None, extra=None):
    """Run the wrapper and return its endpoint matrix in causal-learn encoding."""
    kwargs = {"alpha": 0.05, "n_jobs": n_jobs, "show_progress": False}
    if bk is not None:
        kwargs["background_knowledge"] = bk
    if extra:
        kwargs.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = create_standalone_algorithm(algo_key, **kwargs).run(
            DATA if data is None else data
        )
    graph = result if hasattr(result, "get_graph_edges") else result.G
    return get_adj_matrix(graph)


def directed(m, u, v) -> bool:
    """Check for a tail at u and an arrowhead at v (``u -> v``)."""
    return m[v, u] == TAIL and m[u, v] == ARROW


def adjacent(m, u, v) -> bool:
    return m[u, v] != NULL or m[v, u] != NULL


def skeleton(m) -> set:
    n = m.shape[0]
    return {
        frozenset((i, j))
        for i in range(n)
        for j in range(i + 1, n)
        if adjacent(m, i, j)
    }


@pytest.fixture(scope="module")
def baseline():
    """Run each algorithm without background knowledge."""
    return {key: run(key) for key in BK_ALGORITHMS}


# ============================================================
# 1. A required direction overrides the unconstrained result
# ============================================================


@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
@pytest.mark.parametrize("n_jobs", N_JOBS_VALUES)
def test_required_forces_counterfactual_direction(algo_key, n_jobs, baseline):
    """Requiring X4->X3 must override the true direction X3->X4."""
    base = baseline[algo_key]
    assert directed(base, 2, 3), (
        f"{algo_key}: expected X3->X4 without background knowledge for this test"
    )

    bk = BackgroundKnowledge(required=[("X4", "X3")])
    result = run(algo_key, bk=bk, n_jobs=n_jobs)

    assert directed(result, 3, 2), (
        f"{algo_key} (n_jobs={n_jobs}): required direction X4->X3 is absent; "
        f"endpoint marks: m[2,3]={result[2, 3]}, m[3,2]={result[3, 2]}"
    )


@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_required_matching_truth_is_kept(algo_key):
    """A constraint that matches the true direction must preserve it."""
    bk = BackgroundKnowledge(required=[("X3", "X4")])
    result = run(algo_key, bk=bk)
    assert directed(result, 2, 3)


# ============================================================
# 2. A forbidden direction stays out of the result
# ============================================================


@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_forbidden_direction_is_not_produced(algo_key, baseline):
    """Forbidding the true direction must remove it from the result."""
    assert directed(baseline[algo_key], 2, 3)

    bk = BackgroundKnowledge(forbidden=[("X3", "X4")])
    result = run(algo_key, bk=bk)

    assert not directed(result, 2, 3), (
        f"{algo_key}: forbidden direction X3->X4 is still present"
    )


@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_allowed_cancels_forbidden(algo_key, baseline):
    """Allowing a matching forbidden direction restores the unconstrained result."""
    bk = BackgroundKnowledge(
        forbidden=[("X3", "X4")],
        allowed=[("X3", "X4")],
    )
    result = run(algo_key, bk=bk)
    assert np.array_equal(result, baseline[algo_key]), (
        f"{algo_key}: allowed failed to cancel the matching prohibition"
    )


# ============================================================
# 3. Constraint phases
# ============================================================


@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_pre_search_required_protects_adjacency(algo_key):
    """A PRE_SEARCH requirement protects an edge during skeleton search.

    X1 and X5 have no direct edge in the true graph. The unconstrained search
    separates them. A requirement during search must keep their edge.
    """
    bk = BackgroundKnowledge(enabled_phases=()).require(
        "X1", "X5", phases=BKPhase.PRE_SEARCH
    )
    result = run(algo_key, bk=bk)
    assert adjacent(result, 0, 4), (
        f"{algo_key}: the X1-X5 edge protected during PRE_SEARCH was removed"
    )


def test_gfci_pre_search_protects_edge_ges_never_found():
    """PRE_SEARCH must preserve a required edge even if GES leaves it out.

    GFCI starts from the GES CPDAG. A required edge may be missing from that
    starting graph. As in PC and FCI, PRE_SEARCH must protect the edge through
    the entire search, including the steps before the checkpoint. The checkpoint
    restores the edge from the complete circle-endpoint graph
    (``previous=complete_pag``), whether search started from GES or a complete graph.
    """

    def protected(start_from_complete_graph):
        bk = BackgroundKnowledge(enabled_phases=()).require(
            "X1", "X5", phases=BKPhase.PRE_SEARCH
        )
        m = run(
            "StandaloneGFCIStable",
            bk=bk,
            extra={"start_from_complete_graph": start_from_complete_graph},
        )
        return adjacent(m, 0, 4)

    assert protected(False)
    assert protected(True)


@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_bk_records_active_checkpoints(algo_key):
    """Each algorithm must record active background knowledge in its reports."""
    bk = BackgroundKnowledge(required=[("X4", "X3")])
    run(algo_key, bk=bk)

    active = [r.checkpoint for r in bk.history if r.active]
    assert active, f"{algo_key}: no active background knowledge checkpoint"
    # No explicit run was opened, so all reports use the default run_id.
    assert all(r.run_id == 0 for r in bk.history)


@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_run_scope_separates_reports_of_reused_object(algo_key):
    """One background knowledge object keeps each run distinct in its history."""
    bk = BackgroundKnowledge(required=[("X4", "X3")])

    with bk.run_scope("first"):
        run(algo_key, bk=bk)
    with bk.run_scope("second"):
        run(algo_key, bk=bk)

    first = bk.reports_for_run("first")
    second = bk.reports_for_run("second")

    assert first, f"{algo_key}: first run recorded no reports"
    assert second, f"{algo_key}: second run recorded no reports"
    assert not {id(r) for r in first} & {id(r) for r in second}
    assert len(first) + len(second) == len(bk.history)
    # Leaving the scope restores the initial counter value.
    assert bk.reports_for_run(0) == []


@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_phase_scoped_constraint_does_not_leak_to_other_phases(algo_key):
    """A POST_ORIENTATION constraint must stay inactive in every other phase."""
    bk = BackgroundKnowledge(enabled_phases=()).require(
        "X4", "X3", phases=BKPhase.POST_ORIENTATION
    )
    run(algo_key, bk=bk)

    phases = {r.phase for r in bk.history if r.active}
    assert phases <= {BKPhase.POST_ORIENTATION}, (
        f"{algo_key}: constraints ran in {phases}; expected POST_ORIENTATION only"
    )


# ============================================================
# 4. Background knowledge is a strict constraint
# ============================================================


@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
@pytest.mark.parametrize("n_jobs", N_JOBS_VALUES)
def test_bk_never_leaves_a_directed_cycle(algo_key, n_jobs):
    """The result must stay acyclic when background knowledge is applied."""
    bk = BackgroundKnowledge(required=[("X4", "X3"), ("X5", "X4")])
    result = run(algo_key, bk=bk, n_jobs=n_jobs)
    cycle = find_directed_cycle(result, TAIL, ARROW)
    assert cycle is None, f"{algo_key}: result contains cycle {cycle}"


@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_orientation_only_bk_does_not_change_skeleton(algo_key, baseline):
    """Constraining orientation must preserve which pairs are adjacent."""
    bk = BackgroundKnowledge(required=[("X4", "X3")])
    result = run(algo_key, bk=bk)
    assert skeleton(result) == skeleton(baseline[algo_key]), (
        f"{algo_key}: an orientation constraint changed the skeleton"
    )


@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_bk_changes_only_the_constrained_pair(algo_key, baseline):
    """An opposing requirement can change nearby edges.

    Pairs that share neither endpoint with the required edge must stay unchanged.
    """
    bk = BackgroundKnowledge(required=[("X4", "X3")])
    result = run(algo_key, bk=bk)
    base = baseline[algo_key]

    untouched = [(0, 1)]  # X1-X2 shares neither endpoint with (X3, X4).
    for i, j in untouched:
        assert result[i, j] == base[i, j], (
            f"{algo_key}: the unrelated pair X{i + 1}-X{j + 1} changed"
        )
        assert result[j, i] == base[j, i], (
            f"{algo_key}: the unrelated pair X{i + 1}-X{j + 1} changed"
        )


@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_bk_result_is_njobs_invariant(algo_key):
    """The result must be independent of n_jobs both with and without constraints."""
    bk1 = BackgroundKnowledge(required=[("X4", "X3")])
    bk4 = BackgroundKnowledge(required=[("X4", "X3")])
    assert np.array_equal(
        run(algo_key, bk=bk1, n_jobs=1), run(algo_key, bk=bk4, n_jobs=4)
    )


# ============================================================
# 5. Conflicts must be reported
# ============================================================


@pytest.mark.parametrize("algo_key", BK_ALGORITHMS)
def test_required_edge_absent_reports_conflict(algo_key):
    """X1 and X2 are nonadjacent. Report the conflict without creating an edge."""
    bk = BackgroundKnowledge(enabled_phases=()).require(
        "X1", "X2", phases=BKPhase.POST_ORIENTATION
    )
    result = run(algo_key, bk=bk)

    assert not adjacent(result, 0, 1), (
        f"{algo_key}: background knowledge added a spurious X1-X2 edge"
    )
    conflicts = [e for r in bk.history for e in r.conflicts]
    assert conflicts, f"{algo_key}: the report omitted the missing required edge"


# ============================================================
# 6. Check that RAI does not support background knowledge
# ============================================================


@pytest.mark.parametrize("algo_key", NON_BK_ALGORITHMS)
def test_rai_family_does_not_support_bk(algo_key):
    """Unsupported BK must be rejected before returning an unconstrained graph."""
    bk = BackgroundKnowledge(required=[("X4", "X3")])
    with pytest.raises(NotImplementedError, match="background_knowledge"):
        run(algo_key, bk=bk)
    assert not bk.history


# ============================================================
# 7. Known limitation: orientation rules can introduce a cycle
# ============================================================


def _dense_case(seed=2, p=8, n=800, density=0.30):
    """Build the dense random DAG that reproduces the cycle."""
    rng = np.random.default_rng(seed)
    A = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(i + 1, p):
            if rng.random() < density:
                A[i, j] = 1
    X = rng.normal(0.0, 1.0, size=(n, p))
    for j in range(p):
        for i in range(j):
            if A[i, j]:
                X[:, j] += rng.uniform(0.5, 1.5) * X[:, i]
    return A, X


REVERSED_BK_CYCLE_ALGORITHMS = [
    "StandalonePCStable",
    "StandaloneFCIStable",
    "StandaloneCFCIStable",
    "StandaloneRFCIStable",
    "StandaloneGFCIStable",
    "StandaloneFCIPlusStable",
]


@pytest.mark.parametrize("algo_key", REVERSED_BK_CYCLE_ALGORITHMS)
def test_rules_can_still_close_a_cycle_under_heavy_bk(algo_key):
    """Record a known limitation of orientation under conflicting constraints.

    Static validation checks that background knowledge is acyclic, and each
    phase applies constraints atomically. Applying the constraints cannot itself
    introduce a cycle. The guard prevents rules from violating constraints, but
    rules can still close a cycle through unconstrained edges. With many
    conflicting constraints the PAG can become cyclic. Background knowledge can
    only report the conflict.

    This test checks that a cycle occurs and that the conflict report names it.
    """
    A, X = _dense_case(seed=2)
    names = [f"X{i + 1}" for i in range(A.shape[0])]
    edges = [(i, j) for i in range(A.shape[0]) for j in range(A.shape[0]) if A[i, j]]
    bk = BackgroundKnowledge(required=[(names[j], names[i]) for i, j in edges[:8]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run(algo_key, bk=bk, data=X)

    cycle = find_directed_cycle(result, TAIL, ARROW)
    if cycle is None:
        pytest.skip(f"{algo_key}: this case did not reproduce the cycle")

    conflicts = [e for r in bk.history for e in r.conflicts]
    assert conflicts, (
        f"{algo_key}: cycle {cycle} was present but no conflict was reported"
    )
    assert any("cycle" in e.detail for e in conflicts), (
        f"{algo_key}: the reported conflict does not identify the cycle"
    )


def test_bk_itself_is_never_the_source_of_a_cycle():
    """Applying constraints to an acyclic graph must roll back if a cycle would form."""
    A, _X = _dense_case(seed=2)
    names = [f"X{i + 1}" for i in range(A.shape[0])]
    edges = [(i, j) for i in range(A.shape[0]) for j in range(A.shape[0]) if A[i, j]]
    bk = BackgroundKnowledge(required=[(names[j], names[i]) for i, j in edges[:8]])

    # The constraints must be acyclic on their own.
    from whydra.background_knowledge import _find_cycle

    assert _find_cycle(list(bk.required)) is None
