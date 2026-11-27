"""Check causal search results against independent expectations.

The counterexamples use graph invariants, closed-form scores, and independent
oracles to check the results of the Whydra implementations.
"""

import math

import numpy as np
import pytest

from whydra.algorithms import wrappers as W
from whydra.algorithms.graph_core import CIT
from whydra.algorithms.score_based.scores import LocalScoreCache, local_parent_score
from whydra.algorithms.standalone import (
    fci_plus_algo,
    fci_plus_algo_paralel_cash,
    pag_rules,
)
from whydra.algorithms.standalone.cfci_algo import checkTriple
from whydra.background_knowledge import (
    BackgroundKnowledge,
    BKConflictWarning,
    BKPhase,
    MatrixEncoding,
)


def circle_graph(p, edges):
    matrix = np.zeros((p, p), dtype=int)
    for a, b in edges:
        matrix[a, b] = matrix[b, a] = 2
    return matrix


@pytest.mark.parametrize(
    ("edges", "p"),
    [
        ([(i, (i + 1) % 5) for i in range(5)], 5),
        ([(0, 1), (0, 2), (1, 3), (3, 4), (4, 5), (5, 1)], 6),
    ],
)
def test_r5_preserves_skeleton_and_orients_cycle(edges, p):
    matrix = circle_graph(p, edges)
    rules = [False] * 10
    rules[4] = True
    result = pag_rules.udag2pag(matrix, {}, p, rules=rules, orientCollider=False)
    np.testing.assert_array_equal(result != 0, matrix != 0)
    if p == 5:
        np.testing.assert_array_equal(result[matrix != 0], -np.ones(10, dtype=int))


@pytest.mark.parametrize("module", [fci_plus_algo, fci_plus_algo_paralel_cash])
@pytest.mark.parametrize("entry", [[2, 3, 4], [(2, 3, 4)]])
def test_separator_compatibility_preserves_all_members(module, entry):
    seps = np.empty((5, 5), object)
    seps.fill(None)
    seps[0, 1] = entry
    result = module._sepset_to_pair_dict(seps, 5)
    assert result[(0, 1)] == {2, 3, 4}
    assert (0, 2) not in result  # unknown is not marginal independence


def test_cfci_no_independence_is_ambiguous_and_keeps_pds_evidence():
    result, _ = checkTriple(
        0, 1, 2, [], [], None, None, None, 0.05, lambda *_: 0.0, maj_rule=False
    )
    assert result == 3
    result, seps = checkTriple(
        0, 1, 2, [], [], {1, 3}, {1, 3}, None, 0.05, lambda *_: 0.0, maj_rule=False
    )
    assert result == 2
    assert seps["sepsetA"] == {1, 3}


def test_cfci_majority_decision_does_not_edit_established_separator():
    result, seps = checkTriple(
        0, 1, 2, [3, 4], [3, 4], {1}, {1}, None, 0.05, lambda *a: 1.0, maj_rule=True
    )
    assert result == 1
    assert seps["sepsetA"] == {1}


def test_bdeu_category_relabeling_and_full_cartesian_prior():
    # Only two of four parent configurations observed: q must still be 4.
    data = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 1]])
    expected = sum(
        math.lgamma(0.25)
        - math.lgamma(n + 0.25)
        + sum(math.lgamma(c + 0.125) - math.lgamma(0.125) for c in counts if c)
        for n, counts in [(2, [1, 1]), (2, [0, 2])]
    )
    for table in [data, data + 1, data * 2, data - 1, np.where(data, "yes", "no")]:
        assert local_parent_score(
            table, 2, [0, 1], "local_score_BDeu"
        ) == pytest.approx(expected)
        assert LocalScoreCache(table, "local_score_BDeu").score(
            2, [1, 0]
        ) == pytest.approx(expected)


def latent_data():
    rng = np.random.default_rng(7)
    n = 100000
    l1, l2, l4 = rng.normal(size=(3, n))
    x5 = 0.9 * l2 + rng.normal(size=n)
    x3 = 0.8 * l1 + 1.1 * l2 + rng.normal(size=n)
    x4 = 0.7 * x5 + 1.2 * l4 + rng.normal(size=n)
    x1 = l1 + 0.8 * x4 + rng.normal(size=n)
    x2 = 0.9 * x3 + l4 + rng.normal(size=n)
    return np.column_stack([x1, x2, x3, x4, x5])


def test_fci_pds_counterexample():
    graph = W.StandaloneFCIStable(alpha=0.01, n_jobs=1).run(latent_data())
    assert len(graph.get_graph_edges()) == 6
    assert graph.graph[0, 1] == graph.graph[1, 0] == 0


@pytest.mark.parametrize(
    "algorithm", [W.StandaloneFCIStable, W.StandaloneFCIStableCached]
)
def test_fci_cache_changes_computation_count_only(algorithm):
    data = latent_data()
    off = algorithm(alpha=0.01, use_ci_cache=False, ci_cache_size=0).run(data)
    on = algorithm(alpha=0.01).run(data)
    np.testing.assert_array_equal(on.graph, off.graph)
    assert on.metadata["ci_computations"] < off.metadata["ci_computations"]


@pytest.mark.parametrize("use_pds_path", [False, True])
def test_retained_fci_v2_pds_representation(use_pds_path):
    from whydra.algorithms.graph_core import CausalGraph
    from whydra.algorithms.standalone.fci_algo_v2 import fci_stable_v2

    data = latent_data()
    matrix = fci_stable_v2(
        CausalGraph(5, [f"X{i + 1}" for i in range(5)]),
        data,
        alpha=0.01,
        use_pds_path=use_pds_path,
    )
    assert np.count_nonzero(matrix) // 2 == 6
    assert matrix[0, 1] == matrix[1, 0] == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_ci_rejects_nonfinite_input(value):
    data = np.random.default_rng(12).normal(size=(30, 3))
    data[0, 0] = value
    with pytest.raises(ValueError, match=r"(missing|nonfinite).*columns \[0\]"):
        CIT(data)


def test_fisher_z_rejects_singular_and_insufficient_dof():
    data = np.random.default_rng(13).normal(size=(30, 3))
    data[:, 1] = data[:, 0]
    with pytest.raises(ValueError, match=r"singular|condition"):
        CIT(data)(0, 2, [1])
    with pytest.raises(ValueError, match=r"observations|degrees"):
        CIT(np.arange(6).reshape(3, 2))


CAUSAL_CLASSES = [
    W.StandalonePCStable,
    W.ParallelPCStable,
    W.StandaloneFCIStable,
    W.StandaloneFCIStableCached,
    W.StandaloneFCIStableParallelCached,
    W.StandaloneCFCIStable,
    W.StandaloneCFCIStableParallelCached,
    W.StandaloneRFCIStable,
    W.StandaloneGFCIStable,
    W.StandaloneFCIPlusStable,
    W.StandaloneFCIPlusStableParallelCached,
    W.StandaloneRAIStable,
]


@pytest.mark.parametrize("cls", CAUSAL_CLASSES)
def test_common_wrapper_contract(cls):
    import pandas as pd

    data = pd.DataFrame(
        np.random.default_rng(14).normal(size=(50, 3)), columns=["a", "b", "c"]
    )
    algorithm = cls(n_jobs=1)
    with pytest.raises((TypeError, ValueError), match="alpah"):
        algorithm.run(data, alpah=0.1)
    with pytest.raises(ValueError, match="alpha"):
        algorithm.run(data, alpha=1.2)
    graph = algorithm.run(data, alpha=0.01)
    assert [node.get_name() for node in graph.get_nodes()] == list(data.columns)
    assert graph.graph_kind in {"pag", "cpdag"}
    data["b"] = 1
    with pytest.raises(ValueError, match=r"variance|level"):
        algorithm.run(data, strict_degenerate=True)


def test_pag_bk_rolls_back_arrow_into_ancestor():
    matrix = circle_graph(3, [(0, 1), (1, 2), (0, 2)])
    matrix[0, 1], matrix[1, 0] = 1, -1
    matrix[1, 2], matrix[2, 1] = 1, -1
    matrix[0, 2] = matrix[2, 0] = 1
    before = matrix.copy()
    bk = BackgroundKnowledge()
    bk.require("a", "b")
    with pytest.warns(BKConflictWarning, match="ancestor|ancestral"):
        report = bk.apply_matrix(
            matrix,
            ["a", "b", "c"],
            checkpoint="test",
            encoding=MatrixEncoding.STANDARD,
            graph_kind="pag",
        )
    assert report.conflicts
    np.testing.assert_array_equal(matrix, before)


def test_gfci_missing_separator_needs_ci_evidence_for_both_middles():
    from whydra.algorithms.standalone.gfci_algo import _fill_missing_sepsets_gfci

    matrix = circle_graph(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
    seps = {}
    _fill_missing_sepsets_gfci(matrix, seps, [(1, 3, 2)])
    assert (0, 3) not in seps  # no CI test: do not fabricate a separator

    def oracle(a, b, S):
        return float({a, b} == {0, 3} and set(S) == {1, 2})

    _fill_missing_sepsets_gfci(matrix, seps, [], cit_func=oracle, alpha=0.05)
    assert seps[(0, 3)] == {1, 2}


def test_required_adjacency_has_no_orientation_and_tiers_are_atomic():
    bk = BackgroundKnowledge(enabled_phases=())
    bk.require_adjacency("a", "b")
    matrix = circle_graph(2, [(0, 1)])
    bk.apply_matrix(
        matrix,
        ["a", "b"],
        checkpoint="pre",
        phase=BKPhase.PRE_SEARCH,
        encoding=MatrixEncoding.STANDARD,
        graph_kind="pag",
    )
    np.testing.assert_array_equal(matrix, circle_graph(2, [(0, 1)]))
    assert (0, 1) in bk.required_pairs(["a", "b"], phase=BKPhase.PRE_SEARCH)
    bk = BackgroundKnowledge().require("future", "past")
    before = set(bk.forbidden)
    with pytest.raises(ValueError, match="both forbidden and required"):
        bk.add_tiers({"past": 0, "future": 1})
    assert bk.forbidden == before


def test_rfci_rejected_writes_terminate(monkeypatch):
    from whydra.algorithms.graph_core import CausalGraph
    from whydra.algorithms.standalone import rfci_algo as R

    cg = CausalGraph(4, ["a", "b", "c", "d"])
    cg.G.graph = np.zeros((4, 4), dtype=int)
    cg.G.graph[0, 2], cg.G.graph[2, 0] = 1, -1
    cg.G.graph[1, 0], cg.G.graph[0, 1] = 1, 2
    cg.G.graph[2, 1], cg.G.graph[1, 2] = 2, 2
    cg.sepset[3, 2] = [()]
    monkeypatch.setattr(R, "_min_discriminating_path", lambda *a: [3, 0, 1, 2])
    monkeypatch.setattr(R, "_check_edges_on_disc_path", lambda *a, **k: False)

    class Reject:
        calls = 0

        def set_mark(self, *args):
            self.calls += 1
            assert self.calls < 100, "Rule loop continued without a matrix change"
            return False

    guard = Reject()
    before = cg.G.graph.copy()
    R._apply_discriminating_path_rules(cg, cg.G.graph, 4, 0.05, guard=guard)
    np.testing.assert_array_equal(cg.G.graph, before)
    assert guard.calls > 0


@pytest.mark.parametrize("implementation", ["shared", "v2", "rfci"])
def test_r5_every_pipeline(implementation):
    from whydra.algorithms.standalone import fci_algo_v2, rfci_algo

    cycle = circle_graph(5, [(i, (i + 1) % 5) for i in range(5)])
    rules = [False] * 10
    rules[4] = True
    if implementation == "rfci":
        result = rfci_algo._apply_r5_r10(cycle.copy(), 5, set())
    else:
        fn = (
            pag_rules.udag2pag
            if implementation == "shared"
            else fci_algo_v2.udag2pag_v2
        )
        result = fn(cycle.copy(), {}, 5, rules=rules, orientCollider=False)
    np.testing.assert_array_equal(result, np.where(cycle, -1, 0))


def test_ges_complete_pool_and_report_actual_truncation():
    from whydra.algorithms.score_based.ges import _subsets, ges_search

    complete, capped = {}, {}
    assert len(list(_subsets(range(4), None, statistics=complete))) == 16
    with pytest.warns(RuntimeWarning, match="truncated"):
        assert len(list(_subsets(range(4), 3, statistics=capped))) == 15
    assert complete["truncated_candidate_pools"] == 0
    assert capped["truncated_candidate_pools"] == 1
    data = np.random.default_rng(8).normal(size=(200, 3))
    graph, _ = ges_search(
        data, "local_score_BIC", ["a", "b", "c"], max_operator_set=None
    )
    assert graph.metadata["approximate"] is False


def test_categorical_graphs_invariant_to_bijective_relabeling():
    from whydra.algorithms.wrappers import StandaloneGFCIStable

    rng = np.random.default_rng(12)
    a, b = rng.integers(0, 2, (2, 2000))
    data = np.column_stack([a, b, (a + b) > 0]).astype(int)
    labels = [data, data * 7 + 4, data * 3 - 9, np.where(data, "yes", "no")]
    graphs = [
        StandaloneGFCIStable(indep_test="chisq", data_type="categorical").run(x).graph
        for x in labels
    ]
    assert np.count_nonzero(graphs[0]) > 0
    for graph in graphs[1:]:
        np.testing.assert_array_equal(graph, graphs[0])


def test_pc_collider_cycle_conflict_rejects_both_orientations(monkeypatch):
    from whydra.algorithms.graph_core import CausalGraph
    from whydra.algorithms.standalone.standalone_pc_stable import orient_colliders
    from whydra.background_knowledge import find_directed_cycle

    cg = CausalGraph(4, ["x", "y", "z", "u"])
    cg.G.graph = np.zeros((4, 4), dtype=int)
    for a, b in [(0, 1), (2, 1)]:
        cg.G.graph[a, b] = cg.G.graph[b, a] = -1
    for a, b in [(1, 3), (3, 0)]:
        cg.G.graph[a, b], cg.G.graph[b, a] = 1, -1
    cg.sepset[0, 2] = [()]
    monkeypatch.setattr(cg, "find_unshielded_triples", lambda: [(0, 1, 2)])
    before = cg.G.graph.copy()
    with pytest.warns(RuntimeWarning, match="rejected atomically"):
        orient_colliders(cg)
    np.testing.assert_array_equal(cg.G.graph, before)
    assert find_directed_cycle(cg.G.graph, -1, 1) is None
