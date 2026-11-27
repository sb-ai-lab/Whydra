import numpy as np
import pytest

from whydra.algorithms.standalone import gfci_algo as gfci


@pytest.fixture
def path_graph():
    # 2 -- 0 -- 1 -- 3
    return np.array([[0, 2, 2, 0], [2, 0, 0, 2], [2, 0, 0, 0], [0, 2, 0, 0]])


FORWARD = (0, 1, ())
REVERSE = (1, 0, ())
LEFT = (0, 1, (2,))
RIGHT = (1, 0, (3,))
RESCORE_RIGHT = (0, 1, (3,))
BOTH = [FORWARD, LEFT, REVERSE, RIGHT, LEFT, RESCORE_RIGHT]
NODE_NAMES = ("X", "Y", "A", "Z")


@pytest.mark.parametrize(
    ("values", "expected_trace", "expected_set"),
    [
        ([0, 0, 0, 0], BOTH[:4], None),
        ([0, 1, 0, 0], BOTH[:4], {2}),
        ([0, 0, 1], [FORWARD, LEFT, REVERSE], set()),
        ([1, 0, 0], [FORWARD, REVERSE, RIGHT], set()),
        ([0, 1, 0, 1, 0.9, 0.9], BOTH, {2}),
        ([0, 1, 0, 1, 0.9, np.nan], BOTH, {3}),
        ([0, 1, 0, 1, np.nan, 0.9], BOTH, {3}),
        ([0, 0.5, 0, 1, 0.8, 0.7], BOTH, {2}),
        ([0, 1, 0, 1, 0.9, np.nextafter(0.9, 1.0)], BOTH, {3}),
    ],
)
def test_endpoint_selection(path_graph, values, expected_trace, expected_set):
    trace = []
    scores = iter(values)

    def ci(x, y, cond):
        trace.append((x, y, tuple(cond)))
        return next(scores)

    result = gfci._sepset_subset_of_adjx_or_adjy(
        path_graph, 0, 1, ci, 0.5, 1, False, node_names=NODE_NAMES
    )
    assert result == expected_set
    assert trace == expected_trace


@pytest.mark.parametrize(
    ("left", "right", "expected_name"),
    [({"A", "B"}, {"Z"}, "Z"), ({"A"}, {"Z"}, "A")],
)
def test_tied_separator_prefers_size_then_names(left, right, expected_name):
    def ci(x, y, cond):
        return 0.9

    for names in [("X", "Y", "Z", "B", "A"), ("Y", "X", "A", "B", "Z")]:
        x, y = names.index("X"), names.index("Y")
        from_x = {names.index(name) for name in left}
        from_y = {names.index(name) for name in right}
        for a, b, a_set, b_set in ((x, y, from_x, from_y), (y, x, from_y, from_x)):
            result = gfci._choose_adjacency_separator(
                a, b, ci, a_set, b_set, node_names=names
            )
            assert result == {names.index(expected_name)}


@pytest.mark.parametrize("max_p", [False, True])
def test_search_policy_before_rescoring(path_graph, max_p):
    trace = []

    def ci(x, y, cond):
        trace.append((x, y, tuple(cond)))
        return 0.9 if cond else 0.5

    result = gfci._sepset_subset_of_adjx_or_adjy(
        path_graph, 0, 1, ci, 0.5, 1, max_p, node_names=NODE_NAMES
    )
    if max_p:
        assert trace == BOTH
        assert result == {2}
    else:
        assert trace == [FORWARD, REVERSE, FORWARD, FORWARD]
        assert result == set()


@pytest.mark.parametrize("fail_at", range(len(BOTH)))
def test_ci_failure_preserves_trace_prefix(path_graph, fail_at):
    trace = []
    failure = ValueError("CI failed")

    def ci(x, y, cond):
        trace.append((x, y, tuple(cond)))
        if len(trace) == fail_at + 1:
            raise failure
        return float(bool(cond))

    with pytest.raises(ValueError, match=str(failure)) as exc:
        gfci._sepset_subset_of_adjx_or_adjy(
            path_graph, 0, 1, ci, 0.5, 1, False, node_names=NODE_NAMES
        )
    assert exc.value is failure
    assert trace == BOTH[: fail_at + 1]


@pytest.mark.parametrize("use_cache", [False, True])
def test_adjacency_process_task(path_graph, monkeypatch, use_cache):
    trace = []

    def fisher(corr, n, x, y, cond):
        trace.append((x, y, tuple(cond)))
        return float(bool(cond))

    monkeypatch.setattr(gfci, "_fisher_z_from_corr", fisher)
    results, stats = gfci._gfci_adjacency_process_chunk(
        path_graph,
        [(0, 1)],
        np.eye(4),
        50,
        0.5,
        1,
        True,
        use_cache,
        node_names=NODE_NAMES,
    )
    assert results == [(0, 1, {2})]
    assert trace == ([FORWARD, LEFT, RIGHT] if use_cache else BOTH)
    assert stats == {"calls": 6, "hits": 6 - len(trace), "misses": len(trace)}


@pytest.mark.parametrize("needs_fallback", [False, True])
def test_missing_separator_search(path_graph, needs_fallback):
    pag = np.pad(path_graph, (0, 1))
    pag[0, 1] = pag[1, 0] = 0
    known = {(x, y): set() for x in range(5) for y in range(5) if {x, y} != {0, 1}}
    trace = []

    def ci(x, y, cond):
        trace.append((x, y, tuple(cond)))
        return float(cond == [4] if needs_fallback else bool(cond))

    gfci._fill_missing_sepsets_gfci(
        pag, known, cit_func=ci, alpha=0.5, depth=1, node_names=(*NODE_NAMES, "Q")
    )
    if needs_fallback:
        assert trace == [*BOTH[:4], FORWARD, LEFT, RESCORE_RIGHT, (0, 1, (4,))]
        assert known[0, 1] == known[1, 0] == {4}
    else:
        assert trace == BOTH
        assert known[0, 1] == known[1, 0] == {2}


@pytest.mark.parametrize(
    ("backend", "n_jobs"), [("threads", 1), ("threads", 2), ("processes", 2)]
)
def test_named_tie_matches_across_backends(path_graph, monkeypatch, backend, n_jobs):
    # Two equally noisy proxies of z give equal p-values on opposite sides
    # of X--Y. Their distinct noise keeps the correlation matrix nonsingular.
    rows = np.arange(64)
    z, x_noise, y_noise, proxy1, proxy2 = (2 * ((rows >> k) & 1) - 1 for k in range(5))
    data = np.column_stack((z + x_noise, z + y_noise, z + proxy1 / 4, z + proxy2 / 4))
    names = ("X", "Y", "Z", "A")

    def initial_graph(_, node_names):
        order = [names.index(name) for name in node_names]
        return path_graph[np.ix_(order, order)]

    monkeypatch.setattr(
        gfci, "_score_implementation", lambda _: lambda *a, **kw: (None, 0)
    )
    monkeypatch.setattr(gfci, "_get_cpdag_colliders", lambda *args: set())
    monkeypatch.setattr(gfci, "_ges_cpdag_to_matrix", initial_graph)

    for indices in ([0, 1, 2, 3], [1, 0, 3, 2]):
        ordered_names = [names[i] for i in indices]
        cg = gfci.CausalGraph(4, ordered_names)
        pag = gfci.gfci_stable(
            cg,
            data[:, indices],
            depth=1,
            n_jobs=n_jobs,
            skeleton_backend=backend,
            pdsep_backend="threads",
        )
        x, y = ordered_names.index("X"), ordered_names.index("Y")
        assert (
            cg.gfci_sepset[x, y] == cg.gfci_sepset[y, x] == {ordered_names.index("A")}
        )
        expected = np.array([[0, 0, 2, 0], [0, 0, 0, 2], [2, 0, 0, 0], [0, 2, 0, 0]])
        np.testing.assert_array_equal(
            pag[np.ix_(np.argsort(indices), np.argsort(indices))], expected
        )


def test_collider_pag_and_ci_trace(monkeypatch):
    trace = []

    def ci(self, x, y, cond):
        trace.append((x, y, tuple(cond)))
        return float({x, y} == {0, 2})

    monkeypatch.setattr(gfci.CIT, "__call__", ci)
    cg = gfci.CausalGraph(3, ["A", "B", "C"])
    data = np.random.default_rng(123).normal(size=(20, 3))
    pag = gfci.gfci_stable(
        cg, data, start_from_complete_graph=True, depth=0, use_ci_cache=False
    )
    np.testing.assert_array_equal(pag, [[0, 1, 0], [2, 0, 2], [0, 1, 0]])
    assert cg.gfci_sepset == {(0, 2): set(), (2, 0): set()}
    assert trace == [
        (x, y, ())
        for x, y in [
            (0, 1),
            (1, 0),
            (0, 2),
            (2, 0),
            (0, 2),
            (0, 2),
            (1, 2),
            (2, 1),
            (0, 1),
            (0, 1),
            (1, 2),
            (1, 2),
        ]
    ]
