"""Small, fast CPU-vs-CUDA parity checks.

Level 1 runs the complete PC-stable and FCI-stable searches on the same seeded
data and compares the final endpoint matrices. Level 2 runs skeleton discovery
only and compares adjacency together with the recorded separating sets.

Both checks are intentionally tiny (p = 6, n = 400) so they can gate every CUDA
change without a benchmark run.
"""

import numpy as np
import pytest
import torch

from whydra import StandaloneFCIStable, StandalonePCStable, StandaloneRAIStable
from whydra.algorithms.graph_core import CIT, CausalGraph
from whydra.algorithms.standalone.skeleton import SkeletonDiscovery
from whydra.background_knowledge import BackgroundKnowledge, BKPhase

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

ALPHA = 0.05
N_VARS = 6
N_SAMPLES = 400
NAMES = [f"X{i}" for i in range(N_VARS)]


def _data() -> np.ndarray:
    rng = np.random.default_rng(7)
    x = rng.standard_normal((N_SAMPLES, N_VARS))
    x[:, 2] = x[:, 0] + x[:, 1] + rng.normal(scale=0.4, size=N_SAMPLES)
    x[:, 3] = x[:, 2] + rng.normal(scale=0.5, size=N_SAMPLES)
    x[:, 4] = x[:, 3] + rng.normal(scale=0.5, size=N_SAMPLES)
    return x


def _run(cls, data: np.ndarray, device: str | None) -> np.ndarray:
    graph = cls(alpha=ALPHA, indep_test="fisherz", n_jobs=1, device=device).run(data)
    return np.asarray(graph.graph)


def _skeleton(data: np.ndarray, device: str | None) -> CausalGraph:
    cg = CausalGraph(N_VARS, NAMES, device=device)
    cg.set_ind_test(CIT(data, method="fisherz"))
    disc = SkeletonDiscovery(cg, ALPHA, n_jobs=1, device=device, show_progress=False)
    disc.run()
    return cg


def test_end_to_end_cpu_vs_cuda_matches() -> None:
    """Full PC, FCI and RAI searches must return the same endpoint matrix."""
    data = _data()
    for cls in (StandalonePCStable, StandaloneFCIStable, StandaloneRAIStable):
        cpu = _run(cls, data, None)
        gpu = _run(cls, data, "cuda")
        assert np.array_equal(gpu, cpu), cls.__name__


def test_skeleton_and_sepsets_cpu_vs_cuda_match() -> None:
    """Skeleton adjacency and separator membership must match exactly."""
    data = _data()
    cpu = _skeleton(data, None)
    gpu = _skeleton(data, "cuda")

    assert np.array_equal(cpu.G.graph != 0, gpu.G.graph != 0)

    for x in range(N_VARS):
        for z in range(N_VARS):
            if x == z:
                continue
            for y in range(N_VARS):
                if y in (x, z):
                    continue
                assert cpu.sepset_contains(x, z, y) == gpu.sepset_contains(x, z, y), (
                    f"sepset membership differs for ({x}, {z} | {y})"
                )


def _conditioning_data() -> np.ndarray:
    """X0 and X1 are independent given X2 but correlated marginally."""
    rng = np.random.default_rng(11)
    z = rng.standard_normal(4000)
    return np.column_stack(
        [z + rng.standard_normal(4000), z + rng.standard_normal(4000), z]
    )


def test_cuda_pc_respects_max_cond_set_size() -> None:
    """The CUDA skeleton must stop at the requested conditioning depth."""
    data = _conditioning_data()
    kwargs = {"max_cond_set_size": 0, "node_names": ["X0", "X1", "X2"]}
    cpu = StandalonePCStable(alpha=ALPHA, indep_test="fisherz", n_jobs=1).run(
        data, **kwargs
    )
    cuda = StandalonePCStable(
        alpha=ALPHA, indep_test="fisherz", n_jobs=1, device="cuda"
    ).run(data, **kwargs)
    assert cpu.graph[0, 1] != 0, "CPU must stop before testing the X2 conditioner"
    assert np.array_equal(np.asarray(cuda.graph), np.asarray(cpu.graph))


def test_cuda_fci_respects_max_cond_set_size() -> None:
    """FCI-cuda must apply the same conditioning-depth cap."""
    data = _conditioning_data()
    kwargs = {"max_cond_set_size": 0, "node_names": ["X0", "X1", "X2"]}
    cpu = StandaloneFCIStable(alpha=ALPHA, indep_test="fisherz", n_jobs=1).run(
        data, **kwargs
    )
    cuda = StandaloneFCIStable(
        alpha=ALPHA, indep_test="fisherz", n_jobs=1, device="cuda"
    ).run(data, **kwargs)
    assert cpu.graph[0, 1] != 0
    assert np.array_equal(np.asarray(cuda.graph), np.asarray(cpu.graph))


def test_meek_numpy_matches_cuda_cycle_guard() -> None:
    """The NumPy Meek backend must reject ancestor-cycle orientations too."""
    from whydra.algorithms.graph_core.meek_numpy import meek_orient as meek_numpy
    from whydra.algorithms.graph_core.meek_torch import meek_orient_tensor

    # i -> a, c -> d -> a, and a -- c. Orienting a -> c would close the cycle
    # a -> c -> d -> a; the sequential and CUDA backends reject it.
    G = np.zeros((4, 4), dtype=np.int8)
    G[0, 1], G[1, 0] = 1, -1
    G[2, 3], G[3, 2] = 1, -1
    G[3, 1], G[1, 3] = 1, -1
    G[1, 2] = G[2, 1] = -1

    expected = G.copy()
    expected[2, 1], expected[1, 2] = 1, -1

    host = G.copy()
    meek_numpy(host)
    device = torch.from_numpy(G.copy()).cuda()
    meek_orient_tensor(device)
    assert np.array_equal(host, expected)
    assert np.array_equal(device.cpu().numpy(), expected)


def test_run_rai_cuda_forwards_cit_and_conditioning_limit() -> None:
    """The lower-level RAI CUDA entry must honor the caller's CIT and limit."""
    from whydra.algorithms.graph_core.cuda_rai import run_rai_cuda
    from whydra.algorithms.standalone.rai_algo import rai_stable

    data = _conditioning_data()
    names = ["X1", "X2", "X3"]
    cg = CausalGraph(3, names)
    cg.set_ind_test(CIT(data, method="fisherz"))
    run_rai_cuda(cg, ALPHA, max_cond_set_size=0)
    expected = rai_stable(
        data,
        alpha=ALPHA,
        indep_test="fisherz",
        n_jobs=1,
        node_names=names,
        max_cond_set_size=0,
        device="cuda",
    )
    assert np.array_equal(cg.G.graph, np.asarray(expected.G.graph))


def test_cuda_rejects_non_stable_pc() -> None:
    """CUDA implements PC-stable only; stable=False must fail loudly."""
    with pytest.raises(ValueError, match="stable=True"):
        StandalonePCStable(
            alpha=ALPHA, indep_test="fisherz", n_jobs=1, device="cuda"
        ).run(_data(), stable=False)


def test_cuda_background_knowledge_is_rejected() -> None:
    data = _conditioning_data()
    names = ["X0", "X1", "X2"]
    bk = BackgroundKnowledge(
        required=[("X0", "X1")],
        enabled_phases=[
            BKPhase.PRE_SEARCH,
            BKPhase.BETWEEN_ORIENTATION,
            BKPhase.POST_ORIENTATION,
        ],
    )
    cpu = StandalonePCStable(alpha=ALPHA, indep_test="fisherz", n_jobs=1).run(
        data, node_names=names, background_knowledge=bk
    )
    cuda = StandalonePCStable(
        alpha=ALPHA, indep_test="fisherz", n_jobs=1, device="cuda"
    )
    with pytest.raises(NotImplementedError, match="background_knowledge"):
        cuda.run(data, node_names=names, background_knowledge=bk)
    assert cpu.graph[0, 1] != 0, "CPU keeps the required adjacency"


def test_cuda_fci_background_knowledge_is_rejected() -> None:
    data = _conditioning_data()
    bk = BackgroundKnowledge(required=[("X0", "X1")])
    with pytest.raises(NotImplementedError, match="background_knowledge"):
        StandaloneFCIStable(
            alpha=ALPHA, indep_test="fisherz", n_jobs=1, device="cuda"
        ).run(data, node_names=["X0", "X1", "X2"], background_knowledge=bk)


def test_cuda_rai_background_knowledge_is_rejected() -> None:
    from whydra.algorithms.standalone.rai_algo import rai_stable

    bk = BackgroundKnowledge(required=[("X0", "X1")])
    with pytest.raises(NotImplementedError, match="background_knowledge"):
        rai_stable(
            _conditioning_data(),
            alpha=ALPHA,
            indep_test="fisherz",
            n_jobs=1,
            device="cuda",
            node_names=["X0", "X1", "X2"],
            background_knowledge=bk,
        )


def test_cuda_rai_forwards_custom_ci_test() -> None:
    """RAI-cuda must use a caller-supplied CIT instead of ignoring it."""
    from whydra.algorithms.standalone.rai_algo import rai_stable

    data = _data()
    cpu = rai_stable(
        data,
        alpha=ALPHA,
        indep_test="fisherz",
        n_jobs=1,
        ci_test=CIT(data, method="fisherz"),
    )
    cuda = rai_stable(
        data,
        alpha=ALPHA,
        indep_test="fisherz",
        n_jobs=1,
        device="cuda",
        ci_test=CIT(data, method="fisherz"),
    )
    assert np.array_equal(np.asarray(cuda.G.graph), np.asarray(cpu.G.graph))


def test_cuda_rai_rejects_arbitrary_ci_test() -> None:
    """Arbitrary CI callables remain CPU-only; do not drop them silently."""
    from whydra.algorithms.standalone.rai_algo import rai_stable

    with pytest.raises(NotImplementedError, match="custom ci_test"):
        rai_stable(
            _data(),
            alpha=ALPHA,
            indep_test="fisherz",
            n_jobs=1,
            device="cuda",
            ci_test=lambda x, y, s: 1.0,
        )


def test_cuda_rai_certify_skeleton_matches_cpu() -> None:
    """CUDA RAI honors ``certify_skeleton`` through CPU refinement."""
    data = _data()
    cpu = StandaloneRAIStable(alpha=ALPHA, indep_test="fisherz", n_jobs=1).run(
        data, certify_skeleton=True
    )
    cuda = StandaloneRAIStable(
        alpha=ALPHA, indep_test="fisherz", n_jobs=1, device="cuda"
    ).run(data, certify_skeleton=True)
    assert np.array_equal(np.asarray(cuda.graph), np.asarray(cpu.graph))


def test_cuda_pc_fallback_keeps_orientation(monkeypatch) -> None:
    """Force the k > fused handover and require the full CPU CPDAG parity."""
    from whydra.algorithms.graph_core import cuda_graph_batch as gb

    real_run_skeleton = gb.run_skeleton

    def forced_fallback(*args, **kwargs):
        kwargs["k_max"] = 0
        return real_run_skeleton(*args, **kwargs)

    monkeypatch.setattr(gb, "run_skeleton", forced_fallback)
    data = _data()
    cpu = StandalonePCStable(alpha=ALPHA, indep_test="fisherz", n_jobs=1).run(data)
    cuda = StandalonePCStable(
        alpha=ALPHA, indep_test="fisherz", n_jobs=1, device="cuda"
    ).run(data)
    assert np.array_equal(np.asarray(cuda.graph), np.asarray(cpu.graph))


def test_epilogue_materializes_packed_sepsets() -> None:
    """The host PC fallback needs ``cg.sepset``, not only the packed CSR."""
    cg = CausalGraph(4, ["X0", "X1", "X2", "X3"], device="cuda")
    cg.set_ind_test(CIT(_data()[:, :4], method="fisherz"))
    disc = SkeletonDiscovery(cg, ALPHA, n_jobs=1, device="cuda", show_progress=False)
    G_t = disc._resident_complete_G()
    S = np.array([1, 2], dtype=np.int64)
    ranges = np.full((4, 4, 2), -1, dtype=np.int64)
    ranges[0, 3] = ranges[3, 0] = (0, 2)
    disc._epilogue_from_device(G_t, ("csr", S, ranges), materialize_host_sepsets=True)
    assert cg.sepset[0, 3] == [(1, 2)]
    assert cg.sepset[3, 0] == [(1, 2)]


def test_sepset_workspace_reports_required_and_available_memory(
    monkeypatch,
) -> None:
    """A failed workspace growth must name the size and the free memory."""
    from whydra.algorithms.graph_core import cuda_graph_batch as gb

    csr = gb.SepsetCSR(4, "cuda")

    def _boom(*args, **kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(gb.torch, "empty", _boom)
    monkeypatch.setattr(
        gb.torch.cuda,
        "mem_get_info",
        lambda *args, **kwargs: (64 * 1024 * 1024, 1024 * 1024 * 1024),
    )
    with pytest.raises(
        RuntimeError,
        match=(
            r"workspace allocation failed for sepset CSR buffer: "
            r"need .* MiB, available .* MiB"
        ),
    ):
        csr._ensure_capacity(csr.S.numel() + 10**6, "sepset CSR buffer")
