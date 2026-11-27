"""Boundary validation for the CUDA sepset CSR and Meek collider inputs."""

import pytest
import torch

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(scope="module")
def cuda_kernels():
    """Load extensions only after the CUDA availability skip has been applied."""
    from whydra.algorithms.graph_core import _cuda

    return _cuda


def _csr(p: int):
    S = torch.zeros(2, dtype=torch.int64, device="cuda")
    ranges = torch.full((p, p, 2), -1, dtype=torch.int64, device="cuda")
    return S, ranges


def test_udag2pag_rejects_ranges_shape(cuda_kernels) -> None:
    pag = torch.zeros((4, 4), dtype=torch.int8, device="cuda")
    S, _ = _csr(4)
    ranges = torch.full((4, 4, 3), -1, dtype=torch.int64, device="cuda")
    with pytest.raises(RuntimeError, match=r"pag_r4.*shape"):
        cuda_kernels.udag2pag.r4(pag, S, ranges)


def test_udag2pag_rejects_ranges_past_s(cuda_kernels) -> None:
    pag = torch.zeros((4, 4), dtype=torch.int8, device="cuda")
    S, ranges = _csr(4)
    ranges[0, 1] = torch.tensor([0, 3], device="cuda")
    with pytest.raises(RuntimeError, match=r"pag_r4.*point past S"):
        cuda_kernels.udag2pag.r4(pag, S, ranges)


def test_udag2pag_rejects_bad_sentinel_and_order(cuda_kernels) -> None:
    pag = torch.zeros((4, 4), dtype=torch.int8, device="cuda")
    S, ranges = _csr(4)
    ranges[0, 1] = torch.tensor([-1, 1], device="cuda")
    with pytest.raises(RuntimeError, match=r"pag_r4.*\(-1,-1\)"):
        cuda_kernels.udag2pag.r4(pag, S, ranges)
    ranges[0, 1] = torch.tensor([2, 1], device="cuda")
    with pytest.raises(RuntimeError, match=r"pag_r4.*end must not precede start"):
        cuda_kernels.udag2pag.r4(pag, S, ranges)


def test_meek_rejects_self_loop(cuda_kernels) -> None:
    G = torch.zeros((4, 4), dtype=torch.int8, device="cuda")
    G[0, 0] = 1
    S, ranges = _csr(4)
    with pytest.raises(RuntimeError, match="zero diagonal"):
        cuda_kernels.meek.orient_colliders(G, S, ranges)


def test_claim_and_commit_walk_report_overflow(monkeypatch) -> None:
    """Device commits must flag overflow instead of writing past ``S``."""
    from whydra.algorithms.graph_core.cuda_graph_batch import SepsetCSR, compact_adj

    p = 4
    G = torch.ones((p, p), dtype=torch.int8, device="cuda")
    G.fill_diagonal_(0)
    nbrs, deg = compact_adj(G, G.device)
    csr = SepsetCSR(p, "cuda")
    monkeypatch.setattr(csr, "_ensure_capacity", lambda *args, **kwargs: None)
    csr.S = torch.zeros(1, dtype=torch.int64, device="cuda")
    px = torch.tensor([0], dtype=torch.int32, device="cuda")
    py = torch.tensor([1], dtype=torch.int32, device="cuda")
    first = torch.tensor([0], dtype=torch.int64, device="cuda")
    with pytest.raises(RuntimeError, match=r"first-hit claim.*overflow"):
        csr.claim(px, py, first, nbrs, deg, 2, 0, None)

    csr = SepsetCSR(p, "cuda")
    monkeypatch.setattr(csr, "_ensure_capacity", lambda *args, **kwargs: None)
    csr.S = torch.zeros(1, dtype=torch.int64, device="cuda")
    row_off = torch.tensor([0, 1, 1, 1, 1], dtype=torch.int64, device="cuda")
    hit_k = torch.tensor([2], dtype=torch.int32, device="cuda")
    hit_rank = torch.tensor([0], dtype=torch.int64, device="cuda")
    with pytest.raises(RuntimeError, match=r"Possible-D-SEP commit.*overflow"):
        csr.commit_walk(py, hit_k, hit_rank, row_off, nbrs, deg)


def test_fci_host_rejects_bad_csr_and_indices(cuda_kernels) -> None:
    amat = torch.zeros((4, 4), dtype=torch.int32)
    S = torch.zeros(1, dtype=torch.int64)
    ranges = torch.full((4, 4, 2), -1, dtype=torch.int32)
    ranges[0, 1] = torch.tensor([0, 2])
    with pytest.raises(RuntimeError, match=r"pds_colliders.*point past S"):
        cuda_kernels.fci_host.pds_colliders(amat, S, ranges)
    with pytest.raises(RuntimeError, match=r"pds_batch.*outside \[0,p\)"):
        cuda_kernels.fci_host.pds_batch(amat, [(0, 0), (4, 1)])
    with pytest.raises(RuntimeError, match=r"pds_blocks.*outside \[0,p\)"):
        cuda_kernels.fci_host.pds_blocks(4, [0, 4], [], [], 2, 100)
