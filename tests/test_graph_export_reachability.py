"""Ancestry in exported graphs."""

import numpy as np
import pytest

from whydra.algorithms.graph_core.cpu_graph_convert import _cl_dpath


def _graph(size, topology):
    graph = np.zeros((size, size), dtype=int)
    for source in range(size - 1):
        if topology == "disconnected" and source == size // 2:
            continue
        graph[source + 1, source], graph[source, source + 1] = 1, -1
    if topology == "cycle" and size > 2:
        graph[0, size - 1], graph[size - 1, 0] = 1, -1
    return graph


def _expected_ancestry(graph):
    size = len(graph)
    result = np.eye(size, dtype=int)
    children = [
        np.flatnonzero((graph[:, source] == 1) & (graph[source, :] == -1))
        for source in range(size)
    ]
    for ancestor in range(size):
        pending = list(children[ancestor])
        while pending:
            descendant = pending.pop()
            if result[descendant, ancestor]:
                continue
            result[descendant, ancestor] = 1
            pending.extend(children[descendant])
    return result


@pytest.mark.parametrize("size", [0, 1, 2, 3, 4, 5, 6, 7, 10, 17, 33])
@pytest.mark.parametrize("topology", ["chain", "disconnected", "cycle"])
def test_host_export_ancestry_matches_directed_traversal(size, topology):
    graph = _graph(size, topology)
    before = graph.copy()
    np.testing.assert_array_equal(_cl_dpath(graph), _expected_ancestry(graph))
    np.testing.assert_array_equal(graph, before)


@pytest.mark.cuda
@pytest.mark.parametrize("size", [1, 6, 7, 10, 17])
@pytest.mark.parametrize("topology", ["chain", "disconnected", "cycle"])
def test_cuda_export_ancestry_matches_directed_traversal(size, topology):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from whydra.algorithms.graph_core.cpu_graph_convert import _cl_dpath_from_cl_graph

    graph = _graph(size, topology)
    actual = _cl_dpath_from_cl_graph(torch.as_tensor(graph, device="cuda"))
    np.testing.assert_array_equal(actual, _expected_ancestry(graph))
