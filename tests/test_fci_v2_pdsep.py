import itertools

import numpy as np

from whydra.algorithms.standalone.fci_algo_v2 import (
    _possible_dsep,
    _side_order,
    pdsep_v2,
)
from whydra.algorithms.standalone.pag_rules import (
    PAG_ARROW,
    PAG_CIRCLE,
)

ALPHA = 0.05
SKELETON_EDGES = [
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (2, 3),
    (3, 4),
    (2, 4),
    (4, 5),
    (3, 5),
    (0, 3),
]
SKELETON_P = 6


def _make_oracle(independent, calls):
    """Return a mock CI test: 1.0 if S is an independent set for the pair, otherwise 0.0."""

    def indep(suffStat, x, y, S):
        calls.append((x, y, tuple(S)))
        key = (min(x, y), max(x, y))
        return 1.0 if frozenset(S) in independent.get(key, ()) else 0.0

    return indep


def _skeleton_matrix(value=1):
    """Build a six-node test skeleton using value to mark each edge."""
    sk = np.zeros((SKELETON_P, SKELETON_P), dtype=int)
    for i, j in SKELETON_EDGES:
        sk[i, j] = value
        sk[j, i] = value
    return sk


def _empty_sepset(p=SKELETON_P):
    """Return a sepset dictionary with an empty set for every ordered pair."""
    return {(i, j): set() for i, j in itertools.permutations(range(p), 2)}


def test_possible_dsep_excludes_endpoints_and_paths_through_y():
    """PDS excludes both endpoints and the search cannot pass through y."""
    # Graph A: the chain 0 o-o 1 o-o 2 o-o 3 plus 0 o-o 3.
    amat = np.zeros((4, 4), dtype=int)
    for i, j in [(0, 1), (1, 2), (2, 3), (0, 3)]:
        amat[i][j] = PAG_CIRCLE
        amat[j][i] = PAG_CIRCLE
    pds = _possible_dsep(amat, 0, 3)
    assert 0 not in pds
    assert 3 not in pds
    assert 1 in pds

    # Graph B has only 0 o-o 3 and 3 o-o 4. The only route to 4 passes through y=3.
    amat = np.zeros((5, 5), dtype=int)
    for i, j in [(0, 3), (3, 4)]:
        amat[i][j] = PAG_CIRCLE
        amat[j][i] = PAG_CIRCLE
    assert 4 not in _possible_dsep(amat, 0, 3)


def test_possible_dsep_order_is_reverse_by_default_name():
    """PDS uses reverse lexicographic order on the names X1..Xp, as in causal-learn."""
    amat = np.zeros((12, 12), dtype=int)
    for j in (1, 9, 10, 11):
        amat[0][j] = PAG_CIRCLE
        amat[j][0] = PAG_CIRCLE
    assert _possible_dsep(amat, 0, 1) == [11, 10, 9]

    amat[0][2] = PAG_CIRCLE
    amat[2][0] = PAG_CIRCLE
    assert _possible_dsep(amat, 0, 1) == [2, 11, 10, 9]


def test_side_order_reproduces_edge_flip():
    """_side_order swaps sides only for an arrowhead at i and a circle or tail at j."""
    amat = np.zeros((2, 2), dtype=int)
    amat[1][0] = PAG_ARROW
    amat[0][1] = PAG_CIRCLE
    assert _side_order(amat, 0, 1) == (1, 0)

    amat = np.zeros((2, 2), dtype=int)
    amat[0][1] = PAG_CIRCLE
    amat[1][0] = PAG_CIRCLE
    assert _side_order(amat, 0, 1) == (0, 1)

    amat = np.zeros((2, 2), dtype=int)
    amat[0][1] = PAG_ARROW
    amat[1][0] = PAG_ARROW
    assert _side_order(amat, 0, 1) == (0, 1)


def test_pdsep_result_independent_of_n_jobs():
    """The pdsep_v2 result must be independent of n_jobs and the skeleton encoding."""
    skel = _skeleton_matrix()
    # These sets lie in the PDS for their respective sides and are not subsets
    # of adj(a) or adj(b), so the search tests them. Removing edge 2-4 changes
    # the PDS for the next edge, 3-4. This exercises the branch that recomputes
    # a speculative result on the current graph.
    independent = {
        (0, 1): {frozenset({3, 4})},
        (2, 4): {frozenset({1, 5})},
    }

    def run(n_jobs, sk_matrix):
        calls = []
        return pdsep_v2(
            {"sk": sk_matrix.copy()},
            None,
            _make_oracle(independent, calls),
            SKELETON_P,
            _empty_sepset(),
            ALPHA,
            np.full((SKELETON_P, SKELETON_P), -np.inf),
            n_jobs=n_jobs,
        )

    r1 = run(1, skel)
    r2 = run(2, skel)

    assert np.array_equal(r1["G"], r2["G"])
    assert r1["sepset"] == r2["sepset"]
    assert np.array_equal(r1["pMax"], r2["pMax"])
    assert r1["G"].sum() < skel.sum()

    for i in range(SKELETON_P):
        for j in range(SKELETON_P):
            if i != j and r1["G"][i, j] == 0 and skel[i, j] != 0:
                assert r1["sepset"][(i, j)] == r1["sepset"][(j, i)]
                assert r1["sepset"][(i, j)]

    for value in (-1, 2):
        r = run(1, _skeleton_matrix(value))
        assert np.array_equal(r["G"], r1["G"])


def test_pdsep_conditioning_sets_have_size_at_least_two_and_skip_adjacent_subsets():
    """Conditioning sets have at least two nodes and are subsets of neither adj(x) nor adj(y)."""
    skel = _skeleton_matrix()
    adj = {
        i: {j for j in range(SKELETON_P) if skel[i, j] != 0} for i in range(SKELETON_P)
    }
    calls = []
    res = pdsep_v2(
        {"sk": skel.copy()},
        None,
        _make_oracle({}, calls),
        SKELETON_P,
        _empty_sepset(),
        ALPHA,
        np.full((SKELETON_P, SKELETON_P), -np.inf),
    )
    assert calls
    for x, y, S in calls:
        assert len(S) >= 2
        assert x != y
        assert x not in S
        assert y not in S
        assert not set(S) <= adj[x]
        assert not set(S) <= adj[y]
    assert np.array_equal(res["G"], skel)


def test_pdsep_protected_edges_are_never_tested_or_removed():
    """Edges protected by background knowledge are neither tested nor removed."""
    skel = _skeleton_matrix()
    calls = []
    res = pdsep_v2(
        {"sk": skel.copy()},
        None,
        lambda suff, x, y, S: (calls.append((x, y, tuple(S))), 1.0)[1],
        SKELETON_P,
        _empty_sepset(),
        ALPHA,
        np.full((SKELETON_P, SKELETON_P), -np.inf),
        protected_edges={(0, 1)},
    )
    G = res["G"]
    assert G[0, 1] != 0
    assert G[1, 0] != 0
    for x, y, _S in calls:
        assert {x, y} != {0, 1}
