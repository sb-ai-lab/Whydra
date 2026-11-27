"""Optional causal-learn export in the CUDA PC facade."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from whydra.algorithms import wrappers
from whydra.algorithms.graph_core import CausalGraph
from whydra.algorithms.standalone import pc_algo


@pytest.mark.parametrize(
    "algorithm", [wrappers.StandalonePCStable, wrappers.ParallelPCStable]
)
def test_cuda_facade_requests_local_graph_without_optional_export(
    algorithm, monkeypatch
):
    def local_container(size, names, device=None):
        assert device == "cuda"
        return CausalGraph(size, names)

    def backend(cg, **kwargs):
        assert kwargs.pop("_return_local_graph", False)
        cg.G.graph[:] = 0
        cg.G.graph[0, 1], cg.G.graph[1, 0] = 1, -1
        return cg.G

    monkeypatch.setattr(wrappers, "CausalGraph", local_container)
    monkeypatch.setattr(pc_algo, "_run_pc_cuda", backend)
    result = algorithm(device="cuda").run(
        np.random.default_rng(45).normal(size=(100, 3))
    )
    assert result.graph[0, 1] == 1
    assert result.graph[1, 0] == -1
    assert result.metadata["effective_n_jobs"] == 1


@pytest.mark.parametrize("fallback", [None, 1], ids=["resident", "wavefront"])
@pytest.mark.parametrize("return_local", [False, True], ids=["legacy_export", "facade"])
def test_cuda_driver_export_is_optional_and_consistent(
    fallback, return_local, monkeypatch
):
    cg = CausalGraph(3, ["X1", "X2", "X3"])
    exported = object()
    export_calls = []

    class Tensor:
        def __ne__(self, other):
            return cg.G.graph != other

        def masked_fill_(self, mask, value):
            cg.G.graph[mask] = value

    def orient(*args):
        cg.G.graph[0, 1], cg.G.graph[1, 0] = 1, -1

    class Discovery:
        def __init__(self, **kwargs):
            pass

        def _run_cuda_skeleton(self, **kwargs):
            return Tensor(), (None, None, None), None, fallback

        def _epilogue_from_device(self, *args, **kwargs):
            pass

        def _init_progress_bar(self, show):
            return None

        def _continue_search(self, depth):
            return False

        def _close_progress_bar(self, bar):
            pass

    def export(graph):
        assert not return_local, "The public facade must not export to causal-learn"
        export_calls.append(graph)
        return exported

    monkeypatch.setattr(pc_algo, "SkeletonDiscovery", Discovery)
    monkeypatch.setattr(pc_algo, "causal_graph_to_causallearn", export)
    monkeypatch.setattr(pc_algo.PCAlgorithm, "run", lambda self: orient() or self.cg.G)
    monkeypatch.setitem(
        sys.modules,
        "whydra.algorithms.graph_core.meek_torch",
        SimpleNamespace(
            orient_colliders_tensor=orient, meek_orient_tensor=lambda *_: None
        ),
    )
    result = pc_algo._run_pc_cuda(
        cg, show_progress=False, _return_local_graph=return_local
    )
    assert result is (cg.G if return_local else exported)
    assert len(export_calls) == (0 if return_local else 1)
    assert cg.G.graph[0, 1] == 1
    assert cg.G.graph[1, 0] == -1
