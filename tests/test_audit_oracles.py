"""D-separation search tests with independently derived endpoint matrices."""

import json
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest

from whydra.algorithms import wrappers as W
from whydra.algorithms.graph_core import CIT

REVERSIBLE_EDGES = json.loads(
    Path(__file__)
    .with_name("fixtures")
    .joinpath("rai_reversible_edges.json")
    .read_text(encoding="utf-8")
)


def reference_cpdag(dag, degree, seed):
    matrix = dag - dag.T
    for parent, child in REVERSIBLE_EDGES[f"{len(dag)},{degree},{seed}"]:
        assert dag[parent, child]
        matrix[parent, child] = -1
    return matrix


@pytest.mark.parametrize("cls", [W.StandaloneRAIStable])
@pytest.mark.parametrize(
    ("p", "seed"), [(p, seed) for p in (8, 12) for seed in range(1000, 1010)]
)
@pytest.mark.parametrize("certify", [False, True])
def test_rai_saved_oracle_grid(cls, p, seed, certify, request):
    if not certify and (p, seed) in {(8, 1000), (12, 1002), (12, 1003), (12, 1005)}:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason="Uncertified RAI disagrees with the oracle on this graph and seed",
            )
        )
    rng = np.random.default_rng(seed)
    dag = np.triu((rng.random((p, p)) < 3 / (p - 1)).astype(int), 1)
    truth = nx.from_numpy_array(dag, create_using=nx.DiGraph)

    def oracle(_self, a, b, S=None):
        return float(
            nx.is_d_separator(truth, {int(a)}, {int(b)}, set(map(int, S or [])))
        )

    with (
        patch.object(CIT, "__call__", oracle),
        pytest.warns(RuntimeWarning, match="experimental"),
    ):
        result = cls(n_jobs=1, certify_skeleton=certify).run(rng.normal(size=(100, p)))
    np.testing.assert_array_equal(result.graph, reference_cpdag(dag, 3, seed))


@pytest.mark.parametrize(
    "cls",
    [
        W.StandaloneFCIStable,
        W.StandaloneFCIStableCached,
        W.StandaloneFCIStableParallelCached,
        W.StandaloneCFCIStable,
        W.StandaloneCFCIStableParallelCached,
        W.StandaloneFCIPlusStable,
        W.StandaloneFCIPlusStableParallelCached,
    ],
)
@pytest.mark.parametrize("jobs", [1, 2])
def test_fci_hidden_network_oracle(cls, jobs):
    names = [f"X{i}" for i in range(1, 8)]
    dag = nx.DiGraph(
        [
            ("X1", "X3"),
            ("X2", "X3"),
            ("L1", "X3"),
            ("L1", "X4"),
            ("X3", "X5"),
            ("X4", "X5"),
            ("X5", "X6"),
            ("L2", "X6"),
            ("L2", "X7"),
            ("X4", "X7"),
        ]
    )
    expected = np.array(
        [
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [2, 2, 0, 2, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 1],
            [0, 0, -1, -1, 0, 1, 0],
            [0, 0, 0, 0, -1, 0, 1],
            [0, 0, 0, 2, 0, 1, 0],
        ]
    )

    def oracle(_self, a, b, S=None):
        return float(
            nx.is_d_separator(
                dag,
                {names[int(a)]},
                {names[int(b)]},
                {names[int(z)] for z in (S or [])},
            )
        )

    with patch.object(CIT, "__call__", oracle):
        actual = cls(n_jobs=jobs).run(
            np.random.default_rng(42).normal(size=(100, len(names))), node_names=names
        )
    np.testing.assert_array_equal(actual.graph, expected)


@pytest.mark.parametrize("cls", [W.StandaloneRAIStable])
@pytest.mark.parametrize(
    ("p", "degree", "seed"),
    [(p, d, s) for p in (5, 8) for d in (1, 2, 3) for s in (101, 102, 103)],
)
def test_rai_oracle_density_and_permuted_names(cls, p, degree, seed):
    rng = np.random.default_rng(seed)
    dag = np.triu((rng.random((p, p)) < degree / (p - 1)).astype(int), 1)
    permutation = rng.permutation(p)
    permuted = dag[np.ix_(permutation, permutation)]
    truth = nx.from_numpy_array(permuted, create_using=nx.DiGraph)

    def oracle(_self, a, b, S=None):
        return float(
            nx.is_d_separator(truth, {int(a)}, {int(b)}, set(map(int, S or [])))
        )

    with (
        patch.object(CIT, "__call__", oracle),
        pytest.warns(RuntimeWarning, match="experimental"),
    ):
        actual = cls(n_jobs=1).run(
            rng.normal(size=(100, p)), node_names=[f"node_{i}" for i in permutation]
        )
    np.testing.assert_array_equal(actual.graph, reference_cpdag(permuted, degree, seed))
