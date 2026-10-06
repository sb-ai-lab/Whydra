import numpy as np
import pytest

import whydra
from whydra import (
    StandaloneFCIStable,
    StandalonePCStable,
    StandaloneRAIOptimized,
    StandaloneRAIStable,
)
from whydra.evaluation.graph_utils import get_adj_matrix

# X1 -> X3 <- X2, X3 -> X4
TRUE_SKELETON = {frozenset({0, 2}), frozenset({1, 2}), frozenset({2, 3})}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 3000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = x1 + x2 + rng.normal(scale=0.5, size=n)
    x4 = x3 + rng.normal(scale=0.5, size=n)
    return np.column_stack([x1, x2, x3, x4])


def skeleton(graph):
    adj = get_adj_matrix(graph)
    n = adj.shape[0]
    return {frozenset({i, j}) for i in range(n) for j in range(i + 1, n) if adj[i, j] or adj[j, i]}


def test_version():
    assert isinstance(whydra.__version__, str)


@pytest.mark.parametrize(
    "algo",
    [
        StandalonePCStable(n_jobs=1),
        StandalonePCStable(n_jobs=2),
        StandaloneFCIStable(n_jobs=1),
        StandaloneRAIStable(n_jobs=1),
        StandaloneRAIOptimized(),
    ],
    ids=["pc_seq", "pc_par", "fci", "rai_stable", "rai_optimized"],
)
def test_recovers_skeleton(algo, data):
    graph = algo.run(data)
    assert graph.get_num_nodes() == data.shape[1]
    assert skeleton(graph) == TRUE_SKELETON
