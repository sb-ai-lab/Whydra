"""Test regressions in skeleton encoding, classic FCI, and the Possible-D-SEP limit."""

import itertools

import numpy as np
import pytest

from whydra.algorithms.graph_core import CIT, CausalGraph
from whydra.algorithms.standalone import (
    cfci_algo,
    fci_algo_cash,
    fci_algo_paralel_cash,
    fci_algo_v2,
    standalone_fci_stable,
)
from whydra.algorithms.standalone.pag_rules import PAG_CIRCLE, PAG_TAIL
from whydra.algorithms.wrappers import StandaloneFCIStable

ALPHA = 0.05
PDS_EDGES = [(0, 1), (1, 2), (1, 4), (2, 4), (0, 3)]
PDSEP_MODULES = [
    fci_algo_cash,
    fci_algo_paralel_cash,
    cfci_algo,
    standalone_fci_stable,
    fci_algo_v2,
]


def _skeleton(p, edges, encoding):
    """Return a p-node skeleton matrix with the given edges marked by encoding."""
    sk = np.zeros((p, p), dtype=int)
    for x, y in edges:
        sk[x, y] = sk[y, x] = encoding
    return sk


def _sepset(p):
    """Return a dictionary of empty separating sets for ordered pairs of p nodes."""
    return {(x, y): set() for x, y in itertools.permutations(range(p), 2)}


def _oracle(independent, calls):
    """Return a CI test that reads the independent sets and logs each call in calls."""

    def indep(
        suffStat,
        x,
        y,
        S,
    ):
        """Record x, y, and S and return the p-value. The oracle does not use suffStat."""
        calls.append((x, y, tuple(S)))
        pair = (min(x, y), max(x, y))
        return 1.0 if frozenset(S) in independent.get(pair, ()) else 0.0

    return indep


def _cg(data):
    """Return a CausalGraph with a Fisher-Z test fitted to data."""
    p = data.shape[1]
    cg = CausalGraph(p, [f"X{i + 1}" for i in range(p)])
    cg.set_ind_test(CIT(data, method="fisherz"))
    return cg


def _gaussian_data():
    """Return data from a small linear structural equation model with a fixed seed."""
    data = np.random.default_rng(41).normal(size=(500, 5))
    data[:, 2] += data[:, 0] + data[:, 1]
    data[:, 3] += 0.8 * data[:, 2]
    data[:, 4] += 0.7 * data[:, 3]
    return data


def _confounded_data():
    """Return the observed columns from a structural equation model with two latent nodes."""
    rng = np.random.default_rng(10)
    p = 12
    weights = np.zeros((p, p))
    for src in range(p):
        for dst in range(src + 1, p):
            if rng.random() < 0.3:
                weights[src, dst] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.0)
    data = rng.normal(size=(1000, p))
    for dst in range(p):
        data[:, dst] += data[:, :dst] @ weights[:dst, dst]
    return data[:, 2:]


@pytest.mark.parametrize(
    "module", PDSEP_MODULES, ids=lambda module: module.__name__.split(".")[-1]
)
def test_pdsep_collider_reachability_and_encoding(module, monkeypatch):
    """Check PDS reachability through a collider in both encodings, spying on v2 with monkeypatch."""
    captured = {}
    original = fci_algo_v2._possible_dsep

    def capture(
        amat,
        x,
        y,
        adj=None,
    ):
        """Record PDS for amat, x, y, and adj and return the original traversal result."""
        result = original(amat, x, y, adj=adj)
        captured[(x, y)] = result
        return result

    if module is fci_algo_v2:
        monkeypatch.setattr(module, "_possible_dsep", capture)

    def run(encoding, collider):
        """Run pdsep with the given encoding and collider setting; return its result, PDS, and CI calls."""
        p = 5
        sep = _sepset(p)
        # Suppress every collider except 0 *-> 1 <-* 2. The 1-2-4 triangle
        # provides the semidirected traversal from 1 to 2 that PDS v2 needs.
        for x, y, middle in [(0, 4, 1), (1, 3, 0)]:
            sep[(x, y)] = sep[(y, x)] = {middle}
        if not collider:
            sep[(0, 2)] = sep[(2, 0)] = {1}
        calls = []
        captured.clear()
        pdsep = module.pdsep_v2 if module is fci_algo_v2 else module.pdsep
        kwargs = {"n_jobs": 1} if module in (fci_algo_paralel_cash, fci_algo_v2) else {}
        result = pdsep(
            {"sk": _skeleton(p, PDS_EDGES, encoding)},
            None,
            _oracle({}, calls),
            p,
            sep,
            ALPHA,
            np.full((p, p), -np.inf),
            unfVect=set(),
            **kwargs,
        )
        pds = captured[(0, 3)] if module is fci_algo_v2 else result["allPdsep"][0]
        return result, list(pds), calls

    tail, tail_pds, tail_calls = run(PAG_TAIL, True)
    circle, circle_pds, circle_calls = run(PAG_CIRCLE, True)
    assert 2 in tail_pds
    assert tail_pds == circle_pds
    assert tail_calls == circle_calls
    assert tail_calls
    assert np.array_equal(tail["G"], circle["G"])
    assert tail["sepset"] == circle["sepset"]
    assert tail["allPdsep"] == circle["allPdsep"]
    assert np.array_equal(tail["pMax"], circle["pMax"])
    assert 2 not in run(PAG_TAIL, False)[1]
    assert 2 not in run(PAG_CIRCLE, False)[1]


@pytest.mark.parametrize(
    "module", [fci_algo_cash, fci_algo_paralel_cash, standalone_fci_stable]
)
def test_classic_fci_has_no_conservative_step(module, monkeypatch):
    """Check that classic FCI skips the conservative step, trapping a call with monkeypatch."""

    def forbidden(*args, **kwargs):
        """Fail the test immediately for any arguments, without returning a result."""
        pytest.fail("Classic FCI called pc_cons_intern")

    data = _gaussian_data()
    if module is fci_algo_paralel_cash:
        monkeypatch.setattr(module, "pc_cons_intern", forbidden)
    else:
        assert not hasattr(module, "pc_cons_intern")
        assert not hasattr(module, "checkTriple")

    if module is standalone_fci_stable:
        result = module.fci_stable(data, alpha=ALPHA, indep_test="fisherz")
    else:
        result = module.fci_stable(
            _cg(data), data, alpha=ALPHA, indep_test="fisherz", n_jobs=1
        )
    assert result.shape == (5, 5)


@pytest.mark.parametrize("module", [cfci_algo, fci_algo_paralel_cash])
def test_conservative_triples_support_tail_skeleton(module):
    """Check an ambiguous triple in both encodings without returning a result."""

    def run(encoding):
        """Run the conservative step with this encoding and return the graph dictionary and CI calls."""
        graph = {"sk": _skeleton(3, [(0, 1), (1, 2)], encoding), "sepset": _sepset(3)}
        calls = []
        indep = _oracle({(0, 2): {frozenset(), frozenset({1})}}, calls)
        kwargs = {"n_jobs": 1} if module is fci_algo_paralel_cash else {}
        result = module.pc_cons_intern(
            graph, None, ALPHA, indep, maj_rule=False, **kwargs
        )
        return result, calls

    tail, tail_calls = run(PAG_TAIL)
    circle, circle_calls = run(PAG_CIRCLE)
    assert (0, 1, 2) in tail["unfTriples"]
    assert tail["unfTriples"] == circle["unfTriples"]
    assert tail["sepset"] == circle["sepset"]
    assert tail_calls == circle_calls
    assert tail_calls


def test_fci_v2_m_max_and_wrapper(monkeypatch):
    """Use monkeypatch to check CI set sizes and m_max forwarding without returning a result."""
    data = _confounded_data()
    original = fci_algo_v2.pdsep_v2
    sizes = []
    limits = []

    def spy(
        skel,
        suffStat,
        indepTest,
        p,
        sepSet,
        alpha,
        pMax,
        **kwargs,
    ):
        """Record the limit from kwargs and the CI set sizes, then forward the arguments to pdsep."""
        limits.append(kwargs.get("m_max", float("inf")))

        def indep(
            suff,
            x,
            y,
            S,
        ):
            """Record the size of S, pass suff, x, y, and S to the CI test, and return its p-value."""
            sizes.append(len(S))
            return indepTest(suff, x, y, S)

        return original(skel, suffStat, indep, p, sepSet, alpha, pMax, **kwargs)

    monkeypatch.setattr(fci_algo_v2, "pdsep_v2", spy)
    default = fci_algo_v2.fci_stable_v2(_cg(data), data, alpha=ALPHA, n_jobs=1)
    assert sizes
    assert max(sizes) >= 3
    sizes.clear()
    explicit = fci_algo_v2.fci_stable_v2(
        _cg(data), data, alpha=ALPHA, n_jobs=1, m_max=float("inf")
    )
    assert np.array_equal(default, explicit)
    assert max(sizes) >= 3
    sizes.clear()
    fci_algo_v2.fci_stable_v2(_cg(data), data, alpha=ALPHA, n_jobs=1, m_max=2)
    assert sizes
    assert max(sizes) <= 2
    assert limits[-1] == 2
    sizes.clear()
    # This finite sample produces a cyclic PAG. The public wrapper must
    # report the same violation as the FCI kernel and forward m_max.
    from whydra.background_knowledge import find_ancestral_violation

    raw = fci_algo_v2.fci_stable_v2(_cg(data), data, alpha=ALPHA, n_jobs=1, m_max=2)
    violation = find_ancestral_violation(raw)
    assert violation is not None
    sizes.clear()
    with pytest.warns(RuntimeWarning, match="invalid pag"):
        graph = StandaloneFCIStable(
            alpha=ALPHA, indep_test="fisherz", n_jobs=1, m_max=2
        ).run(data)
    assert graph.metadata["ancestral_violation"] == violation
    assert limits[-1] == 2
    assert sizes
    assert max(sizes) <= 2
    sizes.clear()
    fci_algo_v2.fci_stable_v2(_cg(data), data, alpha=ALPHA, n_jobs=1, m_max=1)
    assert limits[-1] == 1
    assert sizes == []


@pytest.mark.parametrize(
    "module", [fci_algo_cash, cfci_algo, fci_algo_paralel_cash, standalone_fci_stable]
)
def test_pdsep_records_last_unprotected_neighbor_even_without_ci_candidates(module):
    skeleton = _skeleton(7, itertools.combinations(range(4), 2), PAG_TAIL)
    skeleton[4, 5] = PAG_TAIL
    calls = []
    plain = module is standalone_fci_stable
    options = {} if plain else {"protected_edges": [(3, 0), (1, 3), (3, 2)]}
    peaks = np.arange(49).reshape(7, 7) / 100
    result = module.pdsep(
        {"sk": skeleton},
        None,
        _oracle({}, calls),
        7,
        {},
        ALPHA,
        peaks.copy(),
        **options,
    )
    expected_rows = (
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
        if plain
        else [[1, 3, 2], [0, 3, 2], [0, 3, 1], []]
    )
    assert result["allPdsep"] == [*expected_rows, [5], [], []]
    assert calls == []
    assert result["max_ord"] == 0
    assert result["sepset"] == {}
    np.testing.assert_array_equal(result["pMax"], peaks)
    np.testing.assert_array_equal(
        result["G"], (skeleton != 0) * (1 if plain else PAG_CIRCLE)
    )
    if not plain:
        stats = result["pdsep_stats"]
        assert stats["pairs_scanned"] == 7
        assert stats["pairs_with_diffset"] == stats["ci_tests"] == 0


@pytest.mark.parametrize(
    "module", [fci_algo_cash, cfci_algo, fci_algo_paralel_cash, standalone_fci_stable]
)
def test_pdsep_row_history_and_directional_peaks_after_edge_removal(module):
    skeleton = _skeleton(3, [(0, 1), (1, 2)], PAG_CIRCLE)
    evidence = {(0, 2): set(), (2, 0): set()}
    calls = []
    independent = {(0, 1): {frozenset({2})}, (1, 2): {frozenset({0})}}
    result = module.pdsep(
        {"sk": skeleton},
        None,
        _oracle(independent, calls),
        3,
        evidence.copy(),
        ALPHA,
        np.full((3, 3), -np.inf),
    )
    by_level = module is fci_algo_paralel_cash
    second_pair = (2, 1) if by_level else (1, 2)
    assert calls == [(0, 1, (2,)), (*second_pair, (0,))]
    assert result["allPdsep"] == [[2, 1], [0, 2], [0, 1] if by_level else []]
    np.testing.assert_array_equal(result["G"], np.zeros((3, 3), dtype=int))
    expected_peaks = np.full((3, 3), -np.inf)
    expected_peaks[0, 1] = expected_peaks[second_pair] = 1.0
    np.testing.assert_array_equal(result["pMax"], expected_peaks)
    assert result["sepset"] == {
        **evidence,
        (0, 1): {2},
        (1, 0): {2},
        (1, 2): {0},
        (2, 1): {0},
    }
    assert result["max_ord"] == 1


@pytest.mark.parametrize("module", [fci_algo_cash, cfci_algo, fci_algo_paralel_cash])
def test_pdsep_protected_neighbors_can_condition_and_local_subsets_are_skipped(module):
    skeleton = _skeleton(5, [(0, 1), (1, 2), (1, 3), (3, 4)], PAG_CIRCLE)
    evidence = {(0, 2): set(), (0, 3): set(), (2, 3): set(), (1, 4): set()}
    calls = []
    result = module.pdsep(
        {"sk": skeleton},
        None,
        _oracle({(1, 2): {frozenset({3, 4})}}, calls),
        5,
        evidence.copy(),
        ALPHA,
        np.full((5, 5), -np.inf),
        protected_edges=[(1, 0), (3, 1), (3, 4)],
    )
    by_level = module is fci_algo_paralel_cash
    reverse = [(2, 1, (0,)), (2, 1, (3,)), (2, 1, (4,))] if by_level else []
    assert calls == [(1, 2, (4,)), *reverse, (1, 2, (0, 4)), (1, 2, (3, 4))]
    assert result["allPdsep"] == [
        [],
        [0, 3, 4, 2],
        [0, 3, 4, 1] if by_level else [],
        [],
        [],
    ]
    expected_graph = skeleton.copy()
    expected_graph[1, 2] = expected_graph[2, 1] = 0
    np.testing.assert_array_equal(result["G"], expected_graph)
    expected_peaks = np.full((5, 5), -np.inf)
    expected_peaks[1, 2] = 1.0
    if by_level:
        expected_peaks[2, 1] = 0.0
    np.testing.assert_array_equal(result["pMax"], expected_peaks)
    assert result["sepset"] == {**evidence, (1, 2): {3, 4}, (2, 1): {3, 4}}
    assert result["max_ord"] == 2
    stats = result["pdsep_stats"]
    assert (
        stats["pairs_scanned"] == stats["pairs_with_diffset"] == (2 if by_level else 1)
    )
    assert stats["ci_tests"] == len(calls)
