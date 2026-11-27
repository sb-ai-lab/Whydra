"""Differential GES checks against the original full graph/score scan.

The reference deliberately uses the public scalar graph relations and delta
functions. It does not use the immutable views, grouped scores or retained
target rows of the accelerated search.
"""

import itertools
import warnings

import numpy as np
import pytest

from whydra.algorithms.score_based import ges
from whydra.algorithms.score_based._ges_scan import (
    ForwardCandidates,
    GraphView,
    MaskedScores,
    _mask,
)
from whydra.algorithms.score_based.pdag import (
    adjacent,
    complete_pdag,
    cpdag_matrix_to_general_graph,
    dag_to_cpdag,
    exists_semidirected_path,
    is_clique,
    na_yx,
    neighbours,
    parents,
)
from whydra.algorithms.score_based.scores import LocalScoreCache


def _reference_subsets(items, cap, statistics):
    items = sorted(items)
    if cap is not None and (not isinstance(cap, int) or cap < 0):
        raise ValueError("max_operator_set must be None or a nonnegative integer")
    truncated = cap is not None and len(items) > cap
    if statistics is not None:
        statistics["truncated_candidate_pools"] += int(truncated)
        statistics["largest_candidate_pool"] = max(
            statistics["largest_candidate_pool"], len(items)
        )
    if truncated:
        warnings.warn(
            "GES candidate pool truncated by max_operator_set; use None for complete enumeration",
            RuntimeWarning,
            stacklevel=1,
        )
    limit = len(items) if cap is None else min(len(items), cap)
    for size in range(limit + 1):
        yield from itertools.combinations(items, size)


def _reference_phase(
    C,
    cache,
    p,
    cap,
    min_delta,
    forward,
    max_parents=None,
    verify_deltas=False,
    statistics=None,
    trace=None,
):
    steps = 0
    while True:
        candidates = []
        for y in range(p):
            for x in range(p):
                if forward:
                    if x == y or x in adjacent(C, y):
                        continue
                    pool = neighbours(C, y) - adjacent(C, x) - {x}
                else:
                    if x not in parents(C, y) | neighbours(C, y):
                        continue
                    pool = na_yx(C, y, x)
                for subset in _reference_subsets(pool, cap, statistics):
                    if forward:
                        if (
                            max_parents is not None
                            and len(parents(C, y) | na_yx(C, y, x) | set(subset) | {x})
                            > max_parents
                        ):
                            continue
                        delta = ges.insert_delta(C, cache, x, y, subset)
                    else:
                        delta = ges.delete_delta(C, cache, x, y, subset)
                    if delta > min_delta:
                        candidates.append((delta, x, y, subset))
        validity = ges.insert_is_valid if forward else ges.delete_is_valid
        move = next(
            (
                c
                for c in sorted(candidates, key=lambda c: (-c[0], c[1], c[2], c[3]))
                if validity(C, c[1], c[2], c[3])
            ),
            None,
        )
        if move is None:
            return C, steps
        delta, x, y, subset = move
        if trace is not None:
            trace.append((forward, x, y, subset))
        before = C.copy()
        apply = ges.apply_insert if forward else ges.apply_delete
        C = complete_pdag(apply(C, x, y, subset))
        assert C is not None
        if verify_deltas:
            ges._check_delta(before, C, cache, delta, "reference")
        steps += 1


def _sample(seed, p=6, categorical=False):
    rng = np.random.default_rng(seed)
    weights = np.triu((rng.random((p, p)) < 0.35) * rng.uniform(0.5, 1.2, (p, p)), 1)
    data = rng.normal(size=(250, p))
    for y in range(p):
        data[:, y] += data[:, :y] @ weights[:y, y]
    if categorical:
        data = np.digitize(data, [-0.5, 0.5])
    permutation = rng.permutation(p)
    return data[:, permutation], (weights != 0)[
        np.ix_(permutation, permutation)
    ].astype(np.int8)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("cap", [None, 0, 2])
@pytest.mark.parametrize("score_name", ["local_score_BIC", "local_score_BDeu"])
def test_optimized_search_matches_full_scan_moves_objective_and_diagnostics(
    monkeypatch, seed, cap, score_name
):
    data, _ = _sample(
        400 + seed, p=5 + seed % 2, categorical=score_name.endswith("BDeu")
    )
    p = data.shape[1]
    names = [f"X{i + 1}" for i in range(p)]
    cache = LocalScoreCache(data, score_name)
    reference_stats = {"truncated_candidate_pools": 0, "largest_candidate_pool": 0}
    reference_trace = []
    max_parents = 2 if seed % 2 else None
    with warnings.catch_warnings(record=True) as reference_warnings:
        warnings.simplefilter("always")
        C, _ = _reference_phase(
            np.zeros((p, p), dtype=np.int8),
            cache,
            p,
            cap,
            0.0,
            True,
            max_parents=max_parents,
            verify_deltas=True,
            statistics=reference_stats,
            trace=reference_trace,
        )
        C, _ = _reference_phase(
            C,
            cache,
            p,
            cap,
            0.0,
            False,
            verify_deltas=True,
            statistics=reference_stats,
            trace=reference_trace,
        )
    expected_score = ges.total_score(C, cache)
    expected_graph = cpdag_matrix_to_general_graph(C, names)
    trace = []
    for name, forward in [("apply_insert", True), ("apply_delete", False)]:
        original = getattr(ges, name)

        def record(matrix, x, y, subset, original=original, forward=forward):
            trace.append((forward, x, y, subset))
            return original(matrix, x, y, subset)

        monkeypatch.setattr(ges, name, record)
    with warnings.catch_warnings(record=True) as optimized_warnings:
        warnings.simplefilter("always")
        graph, score = ges.ges_search(
            data,
            score_name,
            names,
            max_operator_set=cap,
            max_parents=max_parents,
            verify_deltas=True,
        )
    assert trace == reference_trace
    assert score == expected_score
    np.testing.assert_array_equal(graph.graph, expected_graph.graph)
    for key, value in reference_stats.items():
        assert graph.metadata[key] == value
    assert [str(w.message) for w in optimized_warnings] == [
        str(w.message) for w in reference_warnings
    ]


@pytest.mark.parametrize("seed", range(10))
def test_bit_relations_and_validity_match_scalar_primitives(seed):
    _, dag = _sample(700 + seed, p=7)
    for matrix in [dag, dag_to_cpdag(dag)]:
        view = GraphView(matrix)
        for y in range(len(matrix)):
            assert view.parents[y] == _mask(parents(matrix, y))
            assert view.neighbours[y] == _mask(neighbours(matrix, y))
            assert view.adjacent[y] == _mask(adjacent(matrix, y))
            for x in range(len(matrix)):
                if x == y:
                    continue
                pool = sorted(neighbours(matrix, y) - adjacent(matrix, x) - {x})
                for size in range(len(pool) + 1):
                    for subset in itertools.combinations(pool, size):
                        assert view.insert_valid(x, y, subset) == ges.insert_is_valid(
                            matrix, x, y, subset
                        )
                pool = sorted(na_yx(matrix, y, x))
                for size in range(len(pool) + 1):
                    for subset in itertools.combinations(pool, size):
                        assert view.delete_valid(x, y, subset) == ges.delete_is_valid(
                            matrix, x, y, subset
                        )
        block = {1, 4}
        assert view.clique(_mask(block)) == is_clique(matrix, block)
        for source in range(len(matrix)):
            for target in range(len(matrix)):
                assert bool(
                    view.reachable(source, _mask(block)) & (1 << target)
                ) == exists_semidirected_path(matrix, source, target, block)


def test_reused_local_candidates_recheck_remote_paths_after_inplace_mutation():
    class LinearScore:
        def score(self, target, parents):
            return float(len(parents))

    C = np.zeros((5, 5), dtype=np.int8)
    C[0, 1] = C[2, 1] = 1  # a compelled collider
    scan = ForwardCandidates(MaskedScores(LinearScore(), 5), 5, None, None, 0.0)
    old_view = GraphView(C)
    rows = scan.rows(old_view, ges._subsets, ges._record_subset_pool, None)
    assert old_view.insert_valid(3, 0, ())
    C[1, 3] = 1  # same array; adds a path without changing target 0's local signature
    new_view = GraphView(C)
    assert old_view.forward_signature(0) == new_view.forward_signature(0)
    new_rows = scan.rows(new_view, ges._subsets, ges._record_subset_pool, None)
    assert new_rows[0] is rows[0]
    assert not new_view.insert_valid(3, 0, ())
    assert old_view.insert_valid(3, 0, ())  # the old view did not retain mutable C
    assert scan.reused_targets > 0
    # A local change must rebuild that target, even on the same array object.
    C[4, 0] = 1
    final_rows = scan.rows(GraphView(C), ges._subsets, ges._record_subset_pool, None)
    assert final_rows[0] is not new_rows[0]


def test_gfci_with_background_knowledge_matches_reference_score_search(monkeypatch):
    from whydra.algorithms.wrappers import StandaloneGFCIStable
    from whydra.background_knowledge import BackgroundKnowledge, BKPhase

    data, _ = _sample(500, p=5)

    def run():
        knowledge = BackgroundKnowledge(enabled_phases=tuple(BKPhase))
        knowledge.forbid("X5", "X1")
        graph = StandaloneGFCIStable(n_jobs=1, ges_max_operator_set=None).run(
            data, background_knowledge=knowledge
        )
        return graph, knowledge.history

    optimized, optimized_history = run()

    def forward(
        C,
        cache,
        p,
        cap,
        max_parents,
        min_delta,
        verbose,
        verify_deltas=False,
        enumeration=None,
    ):
        return _reference_phase(
            C, cache, p, cap, min_delta, True, max_parents, verify_deltas, enumeration
        )

    def backward(
        C, cache, p, cap, min_delta, verbose, verify_deltas=False, enumeration=None
    ):
        return _reference_phase(
            C,
            cache,
            p,
            cap,
            min_delta,
            False,
            verify_deltas=verify_deltas,
            statistics=enumeration,
        )

    monkeypatch.setattr(ges, "_forward_phase", forward)
    monkeypatch.setattr(ges, "_backward_phase", backward)
    reference, reference_history = run()
    np.testing.assert_array_equal(optimized.graph, reference.graph)
    assert optimized_history == reference_history
