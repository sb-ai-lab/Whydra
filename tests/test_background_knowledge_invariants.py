"""Test strict background knowledge constraints, validation, rollback, and guards.

The two kinds of conflict have different outcomes:

* If the constraints themselves are inconsistent, changing them raises ValueError.
* If applying constraints would introduce a cycle, the checkpoint rejects
  those changes and emits BKConflictWarning.
"""

import warnings

import numpy as np
import pytest

from whydra.algorithms.graph_core import CausalGraph
from whydra.algorithms.standalone.fci_algo_v2 import udag2pag_v2
from whydra.algorithms.standalone.pag_rules import udag2pag
from whydra.algorithms.standalone.pc_algo import PCAlgorithm
from whydra.algorithms.standalone.rfci_algo import _apply_r1_r3, _apply_r5_r10
from whydra.background_knowledge import (
    BackgroundKnowledge,
    BKConflictWarning,
    BKOrientationGuard,
    BKPhase,
    MatrixEncoding,
    find_directed_cycle,
)

NAMES3 = ["A", "B", "C"]

#: Each independent implementation of the orientation rules must preserve the constraints.
ORIENTATION_IMPLS = [udag2pag, udag2pag_v2]
ORIENTATION_IDS = ["pag_rules", "fci_v2"]


# ============================================================
# Validate constraints before applying them
# ============================================================


def test_cycle_in_required_is_rejected():
    bk = BackgroundKnowledge().require("A", "B").require("B", "C")
    with pytest.raises(ValueError, match="directed cycle"):
        bk.require("C", "A")


def test_opposite_required_is_rejected():
    bk = BackgroundKnowledge().require("A", "B")
    with pytest.raises(ValueError, match=r"cycle|Opposite"):
        bk.require("B", "A")


def test_required_and_forbidden_same_direction_is_rejected():
    bk = BackgroundKnowledge().require("A", "B")
    with pytest.raises(ValueError, match="both forbidden and required"):
        bk.forbid("A", "B")


def test_self_loop_is_rejected():
    with pytest.raises(ValueError, match="self-loop"):
        BackgroundKnowledge().require("A", "A")


def test_rejected_constraint_leaves_object_untouched():
    bk = BackgroundKnowledge()
    bk.require("A", "B")
    bk.require("B", "C")

    with pytest.raises(ValueError, match="directed cycle"):
        bk.require("C", "A")

    assert bk.required == {("A", "B"), ("B", "C")}
    assert bk._phase_overrides["required"] == {}


def test_cycle_across_phases_is_rejected():
    bk = BackgroundKnowledge(enabled_phases=())
    bk.require("A", "B", phases=BKPhase.PRE_SEARCH)
    bk.require("B", "C", phases=BKPhase.BETWEEN_ORIENTATION)
    with pytest.raises(ValueError, match="across phases"):
        bk.require("C", "A", phases=BKPhase.POST_ORIENTATION)


def test_constructor_rejects_cyclic_knowledge():
    with pytest.raises(ValueError, match="directed cycle"):
        BackgroundKnowledge(required=[("A", "B"), ("B", "C"), ("C", "A")])


def test_alias_self_loop_is_caught_only_after_resolution():
    # The two references look different until compilation resolves them to one node.
    bk = BackgroundKnowledge().require("A", 0)
    bk.validate()  # Valid before resolving the references.
    with pytest.raises(ValueError, match="self-loop"):
        bk.validate(node_names=NAMES3)


def test_validate_accepts_sound_knowledge():
    bk = BackgroundKnowledge().require("A", "B").forbid("C", "A")
    assert bk.validate(node_names=NAMES3) is bk


# ============================================================
# Apply constraints to a graph atomically
# ============================================================

_CODES = {
    MatrixEncoding.STANDARD: (0, -1, 1, 2),
    MatrixEncoding.PCALG: (0, 3, 2, 1),
}


def _chain_bc_ca(encoding):
    """Build B -> C -> A with an unoriented A o-o B edge."""
    _null, tail, arrow, circle = _CODES[encoding]
    g = np.zeros((3, 3), dtype=int)
    for u, v in [(1, 2), (2, 0)]:
        g[v, u] = tail
        g[u, v] = arrow
    g[1, 0] = g[0, 1] = circle
    return g


@pytest.mark.parametrize("encoding", [MatrixEncoding.STANDARD, MatrixEncoding.PCALG])
@pytest.mark.parametrize("graph_kind", ["cpdag", "pag"])
def test_post_bk_conflicting_with_graph_rolls_back(encoding, graph_kind):
    # The constraints alone are acyclic. The graph already contains B -> C -> A,
    # so adding A -> B would close a cycle.
    graph = _chain_bc_ca(encoding)
    before = graph.copy()
    bk = BackgroundKnowledge(required=[("A", "B")])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = bk.apply_matrix(
            graph, NAMES3, checkpoint="post", encoding=encoding, graph_kind=graph_kind
        )

    assert np.array_equal(graph, before), "the matrix must stay unchanged"
    assert not report.changed
    assert len(report.conflicts) == 1
    assert "directed cycle" in report.conflicts[0].detail
    assert any(w.category is BKConflictWarning for w in caught)


def test_valid_post_bk_is_applied():
    graph = np.zeros((3, 3), dtype=int)
    graph[1, 0] = graph[0, 1] = 2  # A o-o B
    bk = BackgroundKnowledge(required=[("A", "B")])

    report = bk.apply_matrix(
        graph,
        NAMES3,
        checkpoint="post",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="pag",
    )

    assert report.changed
    assert not report.conflicts
    assert graph[1, 0] == -1
    assert graph[0, 1] == 1


def test_phase_application_is_atomic():
    """If one constraint closes a cycle, roll back the entire phase."""
    graph = _chain_bc_ca(MatrixEncoding.STANDARD)
    before = graph.copy()

    # B -> C is harmless. A -> B closes a cycle regardless of application order.
    bk = BackgroundKnowledge(required=[("B", "C"), ("A", "B")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = bk.apply_matrix(
            graph,
            NAMES3,
            checkpoint="post",
            encoding=MatrixEncoding.STANDARD,
            graph_kind="pag",
        )

    assert np.array_equal(graph, before), "constraints must apply atomically"
    assert not report.changed
    assert not [e for e in report.events if e.action == "apply"]


# ============================================================
# Background knowledge and orientation rules
# ============================================================


def test_orientation_rules_cannot_overwrite_required_endpoint():
    guard = BackgroundKnowledge(required=[("A", "B")]).orientation_guard(NAMES3)
    m = np.zeros((3, 3), dtype=int)
    m[1, 0], m[0, 1] = -1, 1  # A -> B

    assert guard.allows_mark(m, 0, 1, 1)  # The required arrowhead at B.
    assert not guard.allows_mark(m, 0, 1, 2)  # A circle at B.
    assert not guard.allows_mark(m, 0, 1, 0)  # Delete the edge.
    assert not guard.allows_mark(
        m, 1, 0, 1
    )  # An arrowhead at A would reverse the edge.
    assert guard.allows_mark(m, 1, 2, 1)  # The other pair is unconstrained.


def test_guard_blocks_forbidden_direction():
    guard = BackgroundKnowledge(forbidden=[("A", "B")]).orientation_guard(NAMES3)
    m = np.zeros((3, 3), dtype=int)
    m[1, 0], m[0, 1] = -1, 2  # Tail at A, circle at B.
    assert not guard.allows_mark(m, 0, 1, 1)  # An arrowhead at B would give A -> B.
    assert guard.allows_mark(m, 0, 1, 2)


def test_empty_guard_allows_everything():
    guard = BKOrientationGuard(None, MatrixEncoding.STANDARD)
    m = np.zeros((2, 2), dtype=int)
    assert not guard
    assert all(guard.allows_mark(m, 0, 1, v) for v in (-1, 0, 1, 2))


def test_pc_applies_bk_before_collider_orientation():
    graph = CausalGraph(3, NAMES3)
    bk = BackgroundKnowledge(required=[("A", "B")])
    PCAlgorithm(graph, background_knowledge=bk, show_progress=False).run()

    checkpoints = [r.checkpoint for r in bk.history if r.active]
    assert checkpoints[0] == "before_orientation"
    assert graph.G.graph[1, 0] == -1
    assert graph.G.graph[0, 1] == 1


def test_meek_rules_respect_required_direction():
    """The required direction must survive a full PC run."""
    graph = CausalGraph(4, ["A", "B", "C", "D"])
    graph.set_ind_test(
        lambda x, y, s: 0.0
    )  # All pairs are dependent, so the graph is complete.
    bk = BackgroundKnowledge(required=[("A", "B")])
    PCAlgorithm(graph, background_knowledge=bk, show_progress=False).run()

    assert graph.G.graph[1, 0] == -1, "The tail at A was overwritten"
    assert graph.G.graph[0, 1] == 1, "The arrowhead at B was overwritten"


# ============================================================
# Cycle detection in both matrix encodings
# ============================================================


@pytest.mark.parametrize(
    ("encoding", "tail", "arrow"),
    [(MatrixEncoding.STANDARD, -1, 1), (MatrixEncoding.PCALG, 3, 2)],
)
def test_cycle_detection_works_in_both_encodings(encoding, tail, arrow):
    g = np.zeros((3, 3), dtype=int)
    for u, v in [(0, 1), (1, 2), (2, 0)]:
        g[v, u], g[u, v] = tail, arrow
    cycle = find_directed_cycle(g, tail, arrow)
    assert cycle is not None
    assert cycle[0] == cycle[-1]
    assert len(set(cycle)) == 3


# ============================================================
# Property test: a required direction survives orientation
# ============================================================
#
# A rule can overwrite the tail of a required edge. If restoring its direction
# would close a cycle, the checkpoint rolls back the entire transaction. Check
# the orientation rules directly so the test can exercise these rare cases.

RNG_SEED = 20260903


def _random_skeleton(rng, p, density):
    m = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(i + 1, p):
            if rng.random() < density:
                m[i, j] = m[j, i] = 2  # CIRCLE
    return m


def _random_sepset(rng, m, p):
    sepset = {}
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if m[i, j] == 0:
                others = [k for k in range(p) if k not in (i, j)]
                size = int(rng.integers(0, len(others) + 1))
                sepset[(i, j)] = set(
                    rng.choice(others, size=size, replace=False).tolist()
                )
            else:
                sepset[(i, j)] = set()
    return sepset


def _random_case(rng):
    """Generate a skeleton, separating sets, and an existing edge to require."""
    p = int(rng.integers(6, 9))
    density = float(rng.choice([0.4, 0.6, 0.8]))
    skel = _random_skeleton(rng, p, density)
    edges = [(i, j) for i in range(p) for j in range(i + 1, p) if skel[i, j] != 0]
    if not edges:
        return None
    a, b = edges[int(rng.integers(0, len(edges)))]
    if rng.random() < 0.5:
        a, b = b, a
    return p, skel, _random_sepset(rng, skel, p), a, b


def _holds(pag, a, b):
    """Check that the PAG has a tail at a and an arrowhead at b for a -> b."""
    return pag[a, b] == 1 and pag[b, a] == -1


ALL_PHASES = (BKPhase.PRE_SEARCH, BKPhase.BETWEEN_ORIENTATION, BKPhase.POST_ORIENTATION)


#: These fixtures exercise udag2pag_v2 when replacing a required tail with an
#: arrowhead would leave the checkpoint unable to restore the direction
#: without creating a cycle. About 0.03% of the sampled cases had this
#: property, so a few hundred random draws can miss both fixtures.
REGRESSION_CASES = [
    {
        "a": 3,
        "b": 5,
        "p": 7,
        "skeleton": [
            [0, 2, 0, 2, 0, 2, 0],
            [2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 2, 0],
            [2, 0, 2, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [2, 0, 2, 2, 0, 0, 2],
            [0, 0, 0, 0, 0, 2, 0],
        ],
        "sepset": {
            (0, 2): [1, 3, 5, 6],
            (0, 4): [1, 2, 6],
            (0, 6): [1, 3, 4, 5],
            (1, 2): [0, 3, 4, 5, 6],
            (1, 3): [0, 2, 4, 5, 6],
            (1, 4): [2, 3, 5, 6],
            (1, 5): [2, 3, 6],
            (1, 6): [3],
            (2, 1): [0, 4, 5, 6],
            (2, 4): [3, 5],
            (2, 6): [5],
            (3, 1): [4],
            (3, 4): [5],
            (3, 6): [0],
            (4, 1): [0, 3],
            (4, 2): [0, 1, 3, 5, 6],
            (4, 3): [0, 1, 2, 5],
            (4, 5): [1, 3],
            (5, 1): [2, 4],
            (5, 4): [0, 1, 2, 6],
            (6, 1): [0, 2, 3, 5],
            (6, 3): [1],
            (6, 4): [0, 1, 2, 3, 5],
        },
    },
    {
        "a": 4,
        "b": 2,
        "p": 7,
        "skeleton": [
            [0, 0, 2, 2, 2, 2, 2],
            [0, 0, 2, 2, 2, 2, 0],
            [2, 2, 0, 2, 2, 2, 2],
            [2, 2, 2, 0, 2, 0, 0],
            [2, 2, 2, 2, 0, 2, 0],
            [2, 2, 2, 0, 2, 0, 0],
            [2, 0, 2, 0, 0, 0, 0],
        ],
        "sepset": {
            (0, 1): [4],
            (1, 0): [2, 3, 4, 5, 6],
            (3, 5): [0, 1, 2, 4],
            (3, 6): [0, 1, 2, 5],
            (4, 6): [0, 1, 2, 3, 5],
            (5, 3): [1, 2, 6],
            (6, 1): [0],
            (6, 3): [0],
            (6, 4): [5],
            (6, 5): [1, 2, 3],
        },
    },
]


@pytest.mark.parametrize("orient", ORIENTATION_IMPLS, ids=ORIENTATION_IDS)
@pytest.mark.parametrize("case", REGRESSION_CASES, ids=["case1", "case2"])
def test_required_survives_orientation_regression(orient, case):
    """Preserve the required direction when restoring it later would create a cycle."""
    p, a, b = case["p"], case["a"], case["b"]
    names = [f"X{i}" for i in range(p)]
    skel = np.array(case["skeleton"], dtype=int)
    sepset = {k: set(v) for k, v in case["sepset"].items()}
    bk = BackgroundKnowledge(required=[(names[a], names[b])], enabled_phases=ALL_PHASES)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BKConflictWarning)
        out = orient(skel, sepset, p, background_knowledge=bk, node_names=names)

    assert _holds(out, a, b), (
        f"Orientation did not preserve the required edge X{a} -> X{b}: "
        f"X{a} has mark {out[b, a]} (expected a tail, -1), "
        f"X{b} has mark {out[a, b]} (expected an arrowhead, 1)"
    )


@pytest.mark.parametrize("orient", ORIENTATION_IMPLS, ids=ORIENTATION_IDS)
def test_required_survives_orientation_on_random_skeletons(orient):
    """Rules R0-R10 must preserve both endpoints of a required edge."""
    rng = np.random.default_rng(RNG_SEED)
    violations = []

    for _ in range(600):
        case = _random_case(rng)
        if case is None:
            continue
        p, skel, sepset, a, b = case
        names = [f"X{i}" for i in range(p)]
        bk = BackgroundKnowledge(
            required=[(names[a], names[b])], enabled_phases=ALL_PHASES
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BKConflictWarning)
            out = orient(
                skel.copy(), sepset, p, background_knowledge=bk, node_names=names
            )

        if not _holds(out, a, b):
            violations.append((p, a, b, skel.tolist()))

    assert not violations, (
        f"{len(violations)} configurations lost the required direction during orientation; "
        f"first: {violations[0]}"
    )


def test_rfci_rules_respect_required_endpoint():
    """Check that RFCI preserves required endpoints during orientation.

    RFCI implements R1–R3 and R5–R10 independently. A comparison of 4,000
    configurations found identical outcomes with and without the guard, and
    a search of 72,000 unguarded configurations found no lost endpoint.
    These samples check the endpoint invariant. The FCI fixtures above
    exercise cases where the guard changes the outcome.
    """
    rng = np.random.default_rng(RNG_SEED)
    violations = []

    for _ in range(600):
        case = _random_case(rng)
        if case is None:
            continue
        p, skel, _sepset, a, b = case
        names = [f"X{i}" for i in range(p)]
        bk = BackgroundKnowledge(
            required=[(names[a], names[b])], enabled_phases=ALL_PHASES
        )

        pag = skel.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BKConflictWarning)
            # Follow rfci_stable: apply background knowledge first, then run the rules.
            bk.apply_matrix(
                pag,
                names,
                phase=BKPhase.BETWEEN_ORIENTATION,
                checkpoint="before_orientation",
                encoding=MatrixEncoding.STANDARD,
                graph_kind="pag",
            )
        if not _holds(pag, a, b):
            continue  # The constraint could not be applied, so there is nothing to check.

        guard = bk.orientation_guard(
            names, phase=BKPhase.BETWEEN_ORIENTATION, encoding=MatrixEncoding.STANDARD
        )
        pag = _apply_r1_r3(pag, guard=guard)
        pag = _apply_r5_r10(pag, p, guard=guard)

        if not _holds(pag, a, b):
            violations.append((p, a, b, skel.tolist()))

    assert not violations, (
        f"RFCI rules overwrote the required direction in {len(violations)} configurations; "
        f"first: {violations[0]}"
    )
