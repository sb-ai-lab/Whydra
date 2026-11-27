"""CPU/CUDA dispatch and CI test inputs."""

import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from whydra.algorithms.graph_core import CIT, CausalGraph
from whydra.algorithms.standalone.fci_algo_v2 import fci_stable_v2
from whydra.algorithms.standalone.pc_algo import pc_stable
from whydra.algorithms.standalone.skeleton import SkeletonDiscovery


def _graph(tester):
    graph = CausalGraph(tester.data.shape[1], ["A", "B", "C"])
    graph.set_ind_test(tester)
    return graph


@pytest.mark.parametrize("matrix_kind", ["lazy", "cached", "supplied"])
def test_cuda_skeleton_preserves_cpu_inputs_and_fallback_settings(
    matrix_kind, monkeypatch
):
    tester = CIT(np.random.default_rng(43).normal(size=(100, 3)))
    if matrix_kind == "cached":
        _ = tester.corr_matrix
    elif matrix_kind == "supplied":
        tester.corr_matrix = np.array(
            [[1.0, 0.7, 0.2], [0.7, 1.0, 0.3], [0.2, 0.3, 1.0]]
        )
    expected = tester._corr_matrix
    calls = []

    class Batcher:
        def __init__(self, matrix, n_samples, **kwargs):
            calls.append(("matrix", matrix, n_samples, kwargs))

        @classmethod
        def from_data(cls, data, n_samples, **kwargs):
            calls.append(("data", data, n_samples, kwargs))
            return SimpleNamespace()

    monkeypatch.setitem(
        sys.modules,
        "whydra.algorithms.graph_core.cpu_cuda_fisher_bridge",
        SimpleNamespace(CudaFisherZ=Batcher),
    )
    discovery = SkeletonDiscovery(
        _graph(tester), alpha=0.123, device="cuda:0", dtype="float64"
    )
    assert len(calls) == 1
    mode, source, samples, options = calls[0]
    assert mode == ("data" if matrix_kind == "lazy" else "matrix")
    assert source is (tester.data if matrix_kind == "lazy" else expected)
    assert samples == tester.n_samples
    assert options == {"device": "cuda:0", "dtype": "float64"}
    assert tester._corr_matrix is expected
    assert discovery._ci_batcher.cpu_test is tester
    assert discovery._ci_batcher.decision_alpha == tester.decision_alpha == 0.123


@pytest.mark.parametrize("method", ["fisherz", "chisq"])
@pytest.mark.parametrize("entry", ["skeleton", "pc", "fci"])
def test_custom_ci_is_rejected_before_optional_imports(method, entry, monkeypatch):
    class CustomCIT(CIT):
        def __call__(self, *args, **kwargs):
            return 0.75

    data = np.random.default_rng(43).normal(size=(100, 3))
    tester = CustomCIT(data, method=method)
    graph = _graph(tester)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("torch") or name.split(".")[-1] in {
            "cpu_cuda_fisher_bridge",
            "cuda_chisq",
            "meek_torch",
        }:
            raise AssertionError("GPU dependency imported before CI validation")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    if entry == "skeleton":
        run, args = SkeletonDiscovery, (graph,)
    elif entry == "pc":
        run, args = pc_stable, (graph,)
    else:
        run, args = fci_stable_v2, (graph, data)
    with pytest.raises(NotImplementedError, match="custom ci_test"):
        run(*args, device="cuda")


@pytest.mark.parametrize("entry", ["pc", "fci"])
def test_cpu_keeps_custom_ci_semantics(entry):
    class CustomCIT(CIT):
        def __call__(self, *args, **kwargs):
            return 0.75

    data = np.random.default_rng(43).normal(size=(100, 3))
    graph = _graph(CustomCIT(data))
    if entry == "pc":
        pc_stable(graph)
        result = graph.G.graph
    else:
        result = fci_stable_v2(graph, data)
    assert not np.any(result)
