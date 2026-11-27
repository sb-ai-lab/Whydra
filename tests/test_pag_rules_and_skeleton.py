"""Test the shared PAG orientation rules and skeleton search.

The pag_rules module implements rules R0–R10 and their traversal helpers.
The skeleton module stores the separating sets used to orient colliders
across the FCI family.
"""

import numpy as np
import pytest

from whydra.algorithms.graph_core import CIT, CausalGraph
from whydra.algorithms.standalone.pag_init import to_circle_adjacency
from whydra.algorithms.standalone.pag_rules import (
    PAG_ARROW,
    PAG_CIRCLE,
    PAG_NULL,
    PAG_TAIL,
    faith_check,
    legal_path,
    minDiscPath,
    powerset,
    qreach,
    udag2pag,
    updateList,
)
from whydra.algorithms.standalone.skeleton import SkeletonDiscovery
from whydra.background_knowledge import MatrixEncoding

NULL, TAIL, ARROW, CIRCLE = 0, -1, 1, 2


# ============================================================
# Constants: one encoding throughout the library
# ============================================================


def test_pag_constants_are_the_library_encoding():
    from whydra.algorithms.graph_core import Endpoint

    assert (
        Endpoint.NULL.value,
        Endpoint.TAIL.value,
        Endpoint.ARROW.value,
        Endpoint.CIRCLE.value,
    ) == (PAG_NULL, PAG_TAIL, PAG_ARROW, PAG_CIRCLE)


def test_rules_live_in_exactly_one_place():
    """Keep one udag2pag definition so the implementations cannot diverge."""
    import ast
    import pathlib

    import whydra.algorithms.standalone as standalone_pkg

    # Start from the package path. Resolving a relative path from the current
    # working directory found no files when pytest ran from tests/.
    standalone_dir = pathlib.Path(standalone_pkg.__file__).parent

    copies = []
    for path in standalone_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "udag2pag":
                copies.append(path.stem)
    assert copies == ["pag_rules"], f"copies of udag2pag: {copies}"


# ============================================================
# Traversal helpers
# ============================================================


def test_powerset_includes_empty_and_full_sets():
    assert list(powerset([1, 2])) == [(), (1,), (2,), (1, 2)]


def test_update_list_appends_each_node_to_the_path():
    assert updateList([1], [2, 3], [[0]]) == [[0], [1, 2], [1, 3]]


def test_update_list_preserves_nonempty_output_and_copies_each_extension():
    prefix = [1]
    output = [[0]]
    assert updateList(prefix, iter([2, 3]), output) is output
    output[1].append(4)
    assert prefix == [1]
    assert output == [[0], [1, 2, 4], [1, 3]]
    empty = []
    assert updateList(prefix, [2], empty) == [[1, 2]]
    assert empty == []


def test_qreach_returns_reachable_nodes():
    amat = np.zeros((3, 3), dtype=int)
    amat[0, 1] = amat[1, 0] = PAG_CIRCLE
    amat[1, 2] = amat[2, 1] = PAG_CIRCLE
    reachable = qreach(0, amat)
    assert 1 in reachable


def test_qreach_on_an_isolated_node_finds_nothing():
    assert qreach(0, np.zeros((3, 3), dtype=int)) == []


def test_legal_path_requires_adjacency_on_both_legs():
    amat = np.zeros((3, 3), dtype=int)
    amat[0, 1] = amat[1, 0] = PAG_CIRCLE
    assert not legal_path(0, 1, 2, amat), (
        "a path without its second edge must be rejected"
    )
    assert not legal_path(0, 1, 0, amat), "a == c does not form a path"


def test_legal_path_accepts_a_collider():
    amat = np.zeros((3, 3), dtype=int)
    amat[0, 1] = amat[2, 1] = PAG_ARROW  # A *-> B <-* C
    amat[1, 0] = amat[1, 2] = PAG_ARROW
    assert legal_path(0, 1, 2, amat)


def test_min_disc_path_returns_empty_when_none_exists():
    """The caller relies on [] as the empty result when it checks `len(md_path) == 0`."""
    assert minDiscPath(np.zeros((4, 4), dtype=int), 0, 1, 2) == []


def test_faith_check_short_circuits_when_disabled():
    assert faith_check([0, 1, 2], set(), 3, boolean=False) == 0


# ============================================================
#  to_circle_adjacency
# ============================================================


@pytest.mark.parametrize(
    ("encoding", "circle"),
    [
        (MatrixEncoding.STANDARD, CIRCLE),
        (MatrixEncoding.PCALG, 1),
    ],
)
def test_to_circle_adjacency_uses_the_named_encoding(encoding, circle):
    m = np.array([[0, TAIL], [TAIL, 0]])
    out = to_circle_adjacency(m, encoding)
    assert out[0, 1] == circle
    assert out[1, 0] == circle
    assert out[0, 0] == 0


def test_to_circle_adjacency_does_not_mutate_the_input():
    m = np.array([[0, TAIL], [TAIL, 0]])
    before = m.copy()
    to_circle_adjacency(m, MatrixEncoding.STANDARD)
    assert np.array_equal(m, before)


# ============================================================
#  udag2pag
# ============================================================


def test_udag2pag_starts_from_an_all_circle_pag():
    """The skeleton gets o-o endpoints before orientation rules run."""
    skel = np.zeros((3, 3), dtype=int)
    skel[0, 1] = skel[1, 0] = 1
    pag = udag2pag(
        skel.copy(), sepset={}, p=3, rules=[False] * 10, orientCollider=False
    )
    assert pag[0, 1] == PAG_CIRCLE
    assert pag[1, 0] == PAG_CIRCLE


def test_udag2pag_orients_an_unshielded_collider():
    """A o-o C o-o B is a collider if A and B are nonadjacent and C is outside sepset(A,B)."""
    skel = np.zeros((3, 3), dtype=int)
    skel[0, 2] = skel[2, 0] = 1
    skel[1, 2] = skel[2, 1] = 1
    pag = udag2pag(
        skel,
        sepset={(0, 1): set(), (1, 0): set()},
        p=3,
        rules=[False] * 10,
        orientCollider=True,
    )
    assert pag[0, 2] == PAG_ARROW
    assert pag[1, 2] == PAG_ARROW


def test_udag2pag_does_not_orient_when_the_middle_node_separates():
    skel = np.zeros((3, 3), dtype=int)
    skel[0, 2] = skel[2, 0] = 1
    skel[1, 2] = skel[2, 1] = 1
    pag = udag2pag(
        skel,
        sepset={(0, 1): {2}, (1, 0): {2}},
        p=3,
        rules=[False] * 10,
        orientCollider=True,
    )
    assert pag[0, 2] == PAG_CIRCLE
    assert pag[1, 2] == PAG_CIRCLE


def test_require_recorded_sepset_changes_the_missing_sepset_case():
    """The parameter captures the one difference between the families in this orientation step."""
    skel = np.zeros((3, 3), dtype=int)
    skel[0, 2] = skel[2, 0] = 1
    skel[1, 2] = skel[2, 1] = 1

    lenient = udag2pag(
        skel.copy(),
        sepset={},
        p=3,
        rules=[False] * 10,
        orientCollider=True,
        require_recorded_sepset=False,
    )
    strict = udag2pag(
        skel.copy(),
        sepset={},
        p=3,
        rules=[False] * 10,
        orientCollider=True,
        require_recorded_sepset=True,
    )

    assert lenient[0, 2] == PAG_ARROW, "FCI orients when the sepset is missing"
    assert strict[0, 2] == PAG_CIRCLE, (
        "CFCI/FCI+ leave the edge unoriented without a sepset"
    )


def test_udag2pag_does_not_mutate_the_caller_matrix():
    skel = np.zeros((3, 3), dtype=int)
    skel[0, 1] = skel[1, 0] = 1
    before = skel.copy()
    udag2pag(skel, sepset={}, p=3, rules=[False] * 10, orientCollider=False)
    assert np.array_equal(skel, before)


def test_rule_stats_are_reported_when_asked():
    skel = np.zeros((3, 3), dtype=int)
    skel[0, 2] = skel[2, 0] = 1
    skel[1, 2] = skel[2, 1] = 1
    stats = {}
    udag2pag(
        skel,
        sepset={(0, 1): set(), (1, 0): set()},
        p=3,
        rules=[True] * 10,
        orientCollider=True,
        rule_stats=stats,
    )
    assert set(stats) == {f"R{i}" for i in range(11)}
    assert stats["R0"] > 0


# ============================================================
#  SkeletonDiscovery
# ============================================================


def _cg(n, ci):
    cg = CausalGraph(n, [f"X{i + 1}" for i in range(n)])
    cg.set_ind_test(ci)
    return cg


def test_skeleton_removes_every_edge_when_all_pairs_are_independent():
    cg = _cg(4, lambda x, y, S: 1.0)
    SkeletonDiscovery(cg, alpha=0.05, show_progress=False).run()
    assert len(cg.G.get_graph_edges()) == 0


def test_skeleton_keeps_every_edge_when_all_pairs_are_dependent():
    cg = _cg(4, lambda x, y, S: 0.0)
    SkeletonDiscovery(cg, alpha=0.05, show_progress=False).run()
    assert len(cg.G.get_graph_edges()) == 6


def test_protected_edges_survive_and_leave_no_sepset():
    """A protected edge stays in the graph and does not receive a separating set."""
    cg = _cg(3, lambda x, y, S: 1.0)
    SkeletonDiscovery(
        cg, alpha=0.05, show_progress=False, protected_edges={(0, 1)}
    ).run()
    assert cg.G.has_edge(0, 1)
    assert cg.sepset[0, 1] is None
    assert cg.sepset[1, 0] is None
    # The other edges are removed as usual.
    assert not cg.G.has_edge(0, 2)


def test_run_fci_returns_the_expected_payload():
    cg = _cg(3, lambda x, y, S: 0.0)
    out = SkeletonDiscovery(cg, alpha=0.05, show_progress=False).run_fci()
    assert set(out) == {"sk", "pMax", "sepset", "unfTriples", "max_ord"}
    assert out["sk"].shape == (3, 3)


@pytest.mark.parametrize("process_ci_info", [None, {"method": "chisq"}])
def test_run_fci_rejects_process_backend_without_fisherz_info(process_ci_info):
    cg = _cg(3, lambda x, y, S: 0.0)
    skeleton = SkeletonDiscovery(
        cg,
        n_jobs=2,
        show_progress=False,
        parallel_backend="processes",
        process_ci_info=process_ci_info,
    )
    with pytest.raises(ValueError, match=r"process_ci_info with method='fisherz'"):
        skeleton.run_fci()


def test_run_fci_rejects_incomplete_fisherz_process_info():
    cg = _cg(3, lambda x, y, S: 0.0)
    skeleton = SkeletonDiscovery(
        cg,
        n_jobs=2,
        show_progress=False,
        parallel_backend="processes",
        process_ci_info={"method": "fisherz"},
    )
    with pytest.raises(
        ValueError, match=r"missing required keys.*corr_matrix.*n_samples"
    ):
        skeleton.run_fci()


def test_flat_sepset_values_are_always_sets():
    """The value type is independent of both stable and n_jobs."""
    for stable in (True, False):
        for n_jobs in (1, 4):
            cg = _cg(4, lambda x, y, S: 1.0)
            out = SkeletonDiscovery(
                cg, alpha=0.05, stable=stable, n_jobs=n_jobs, show_progress=False
            ).run_fci()
            assert all(isinstance(v, set) for v in out["sepset"].values()), (
                f"stable={stable}, n_jobs={n_jobs}"
            )


def test_stable_skeleton_is_njobs_invariant():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((400, 6))
    for a, b in [(0, 2), (1, 2), (2, 3), (3, 4)]:
        X[:, b] += 1.2 * X[:, a]

    results = []
    for n_jobs in (1, 2, 4):
        cg = _cg(6, CIT(X, "fisherz"))
        out = SkeletonDiscovery(
            cg, alpha=0.05, n_jobs=n_jobs, show_progress=False
        ).run_fci()
        results.append(
            (out["sk"].copy(), {k: tuple(sorted(v)) for k, v in out["sepset"].items()})
        )
    assert all(np.array_equal(results[0][0], r[0]) for r in results[1:])
    assert all(results[0][1] == r[1] for r in results[1:])


def test_cg_sepset_keeps_the_full_list_of_separating_sets():
    """The flat sepset holds one value; cg.sepset keeps every separating set found."""
    cg = _cg(4, lambda x, y, S: 1.0)
    SkeletonDiscovery(cg, alpha=0.05, show_progress=False).run_fci()
    recorded = [cg.sepset[i, j] for i in range(4) for j in range(4) if i != j]
    assert any(v is not None and len(v) >= 1 for v in recorded)
