"""Early rejection of unsupported background knowledge."""

import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from whydra.algorithms import wrappers
from whydra.algorithms.graph_core import CIT, CausalGraph
from whydra.algorithms.standalone import fci_algo_v2, pc_algo, rai_algo, skeleton
from whydra.background_knowledge import BackgroundKnowledge, BKPhase


def _forbid_optional_imports(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("torch", "causallearn")) or name.split(".")[-1].startswith(
            "cuda_"
        ):
            raise AssertionError("Optional backend imported before BK validation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


@pytest.fixture(params=[False, True], ids=["empty", "constrained"])
def knowledge(request):
    return BackgroundKnowledge(required=[("X1", "X2")] if request.param else [])


@pytest.mark.parametrize(
    "algorithm",
    [
        wrappers.StandalonePCStable,
        wrappers.ParallelPCStable,
        wrappers.StandaloneFCIStable,
        wrappers.StandaloneRAIStable,
    ],
)
@pytest.mark.parametrize("location", ["constructor", "run"])
def test_cuda_facade_rejects_bk_before_allocating(
    algorithm, location, knowledge, monkeypatch
):
    def unexpected_work(*args, **kwargs):
        raise AssertionError("BK validation must precede CI and graph construction")

    _forbid_optional_imports(monkeypatch)
    monkeypatch.setattr(wrappers, "CIT", unexpected_work)
    monkeypatch.setattr(wrappers, "CausalGraph", unexpected_work)
    options = {"background_knowledge": knowledge}
    configured = algorithm(
        device="cuda", **(options if location == "constructor" else {})
    )
    with pytest.raises(NotImplementedError, match="background_knowledge"):
        configured.run(np.ones((10, 3)), **(options if location == "run" else {}))


@pytest.mark.parametrize("entry", ["pc", "pc_driver", "fci"])
def test_direct_cuda_rejects_bk_without_mutating_graph(entry, knowledge, monkeypatch):
    cg = CausalGraph(3, ["X1", "X2", "X3"])
    before = cg.G.graph.copy()
    metadata = dict(cg.G.metadata)
    _forbid_optional_imports(monkeypatch)
    options = {"background_knowledge": knowledge, "device": "cuda"}
    if entry == "pc":
        run, args = pc_algo.pc_stable, (cg,)
    elif entry == "pc_driver":
        run, args = pc_algo._run_pc_cuda, (cg,)
    else:
        run, args = fci_algo_v2.fci_stable_v2, (cg, np.ones((10, 3)))
    with pytest.raises(NotImplementedError, match="background_knowledge"):
        run(*args, **options)
    np.testing.assert_array_equal(cg.G.graph, before)
    assert cg.G.metadata == metadata
    assert cg.test is None


@pytest.mark.parametrize("device", [None, "cpu", "cuda"])
def test_direct_rai_rejects_bk_before_learner_init(device, knowledge, monkeypatch):
    def unexpected_learner(*args, **kwargs):
        raise AssertionError("RAI initialized before rejecting unsupported BK")

    _forbid_optional_imports(monkeypatch)
    monkeypatch.setattr(rai_algo, "RAIStableLearner", unexpected_learner)
    with pytest.raises(NotImplementedError, match="background_knowledge"):
        rai_algo.rai_stable(
            np.ones((10, 3)), device=device, background_knowledge=knowledge, n_jobs=1
        )


@pytest.mark.parametrize(
    "algorithm", [wrappers.StandalonePCStable, wrappers.StandaloneFCIStable]
)
def test_cpu_override_preserves_required_adjacency(algorithm):
    rng = np.random.default_rng(28)
    data = rng.normal(size=(1000, 3))
    knowledge = BackgroundKnowledge(enabled_phases=[BKPhase.PRE_SEARCH])
    knowledge.require_adjacency("X1", "X2")
    result = algorithm(device="cuda").run(
        data, device="cpu", background_knowledge=knowledge
    )
    assert result.graph[0, 1] != 0
    assert result.graph[1, 0] != 0


@pytest.mark.parametrize(
    "container", [list, tuple, iter], ids=["list", "tuple", "iterator"]
)
def test_cuda_skeleton_rejects_protected_edges_before_initialization(
    container, monkeypatch
):
    cg = CausalGraph(3, ["X1", "X2", "X3"])
    before = cg.G.graph.copy()
    metadata = dict(cg.G.metadata)

    def unexpected_configuration(*args):
        raise AssertionError("Protected edges must be rejected before CI configuration")

    _forbid_optional_imports(monkeypatch)
    monkeypatch.setattr(skeleton, "configure_fisher_alpha", unexpected_configuration)
    with pytest.raises(NotImplementedError, match="protected_edges"):
        skeleton.SkeletonDiscovery(
            cg, device="cuda", protected_edges=container([(0, 1)])
        )
    np.testing.assert_array_equal(cg.G.graph, before)
    assert cg.G.metadata == metadata
    assert cg.test is None


@pytest.mark.parametrize(
    "container", [list, tuple, iter], ids=["list", "tuple", "iterator"]
)
def test_cuda_skeleton_accepts_empty_protected_edge_iterables(container, monkeypatch):
    cg = CausalGraph(3, ["X1", "X2", "X3"])
    cg.set_ind_test(CIT(np.random.default_rng(17).normal(size=(100, 3))))
    batcher = SimpleNamespace()
    monkeypatch.setitem(
        sys.modules,
        "whydra.algorithms.graph_core.cpu_cuda_fisher_bridge",
        SimpleNamespace(
            CudaFisherZ=SimpleNamespace(from_data=lambda *args, **kwargs: batcher)
        ),
    )
    discovery = skeleton.SkeletonDiscovery(
        cg, device="cuda", protected_edges=container([])
    )
    assert discovery.protected_edges == set()
    assert discovery._ci_batcher is batcher


@pytest.mark.parametrize(
    "container", [list, tuple, iter], ids=["list", "tuple", "iterator"]
)
def test_cpu_skeleton_keeps_protected_adjacency_from_iterable(container):
    cg = CausalGraph(3, ["X1", "X2", "X3"])
    cg.set_ind_test(lambda *args: 1.0)
    skeleton.SkeletonDiscovery(cg, protected_edges=container([(0, 1)])).run()
    assert cg.G.graph[0, 1] != 0
    assert cg.G.graph[1, 0] != 0
    assert cg.G.graph[0, 2] == cg.G.graph[2, 0] == 0
    assert cg.sepset[0, 1] is None
    assert cg.sepset[1, 0] is None
