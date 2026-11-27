"""CUDA PC-stable fast paths: pair-major kernels and the cached depth workspace.

These cases pin the pair-major kernel (and its workspace allocation) against
the flat offset-search path, and the k = 0 symmetry reduction against the
directed path. The CPU sepset-union semantics live in
``tests/test_pc_sepset_union.py``.
"""

import numpy as np
import pytest
import torch

from tests.test_pc_sepset_union import ALPHA


def _canonical_sepsets(ranges, S, p):
    """Map each claimed pair to the frozenset of its sepset members.

    The union records are appended in atomic-reservation order, so the raw
    ``S`` layout differs between runs; only the per-pair member set is stable.
    """
    r = (
        ranges.detach().cpu().numpy()
        if hasattr(ranges, "detach")
        else np.asarray(ranges)
    )
    s = S.detach().cpu().numpy() if hasattr(S, "detach") else np.asarray(S)
    out = {}
    for a in range(p):
        for b in range(a + 1, p):
            start, end = int(r[a, b, 0]), int(r[a, b, 1])
            if start >= 0:
                out[(a, b)] = frozenset(int(v) for v in s[start:end])
    return out


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_pair_major_depth_matches_flat_depth(k):
    """Pair-major (and its workspace) must reproduce the flat depth exactly.

    Compares the returned ``(n_tests, nnz, overflow)``, the resulting ``G``
    and the per-pair sepset member sets for one full depth. ``WHYDRA_PC_SMALLK``
    selects the CI kernel and ``WHYDRA_PC_WORKSPACE`` the driver allocation
    path; the arithmetic must not depend on either.
    """
    import os

    import torch
    from scipy.stats import norm

    from whydra.algorithms.graph_core.cuda_fisher_z import CudaFisherZ
    from whydra.algorithms.graph_core.cuda_graph_batch import (
        SepsetCSR,
        _graph_batch_cuda,
        compact_adj,
    )

    p, n = 9, 400
    rng = np.random.default_rng(11)
    data = rng.standard_normal((n, p))
    batcher = CudaFisherZ.from_data(data, device="cuda")
    tau = float(norm.ppf(1.0 - ALPHA / 2.0))
    G0 = torch.ones((p, p), dtype=torch.int8, device="cuda")
    G0.fill_diagonal_(0)
    ext = _graph_batch_cuda()

    def run(smallk, workspace):
        old = {
            key: os.environ.get(key)
            for key in ("WHYDRA_PC_SMALLK", "WHYDRA_PC_WORKSPACE")
        }
        try:
            os.environ["WHYDRA_PC_SMALLK"] = "1" if smallk else "0"
            os.environ["WHYDRA_PC_WORKSPACE"] = "1" if workspace else "0"
            G = G0.clone()
            csr = SepsetCSR(p, "cuda")
            nbrs, deg = compact_adj(G, "cuda")
            ret = ext.pc_depth_union(
                batcher.corr,
                G,
                nbrs,
                deg,
                csr.ranges,
                csr.S,
                csr.nnz,
                tau,
                float(batcher.eps),
                int(batcher.n_samples),
                k,
            )
            torch.cuda.synchronize()
            return (
                ret.tolist(),
                G.cpu(),
                _canonical_sepsets(csr.ranges, csr.S, p),
            )
        finally:
            for key, val in old.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

    ref = run(smallk=False, workspace=False)
    for smallk, workspace in ((True, False), (True, True)):
        got = run(smallk=smallk, workspace=workspace)
        tag = f"smallk={smallk} workspace={workspace} k={k}"
        if k == 0 and workspace:
            # The workspace path enumerates the symmetric upper triangle at
            # k=0 (one test per undirected pair); the flat path is directed.
            assert got[0][0] * 2 == ref[0][0], f"n_tests mismatch {tag}"
            assert got[0][1:] == ref[0][1:], f"counts mismatch {tag}"
        else:
            assert got[0] == ref[0], f"counts mismatch {tag}: {got[0]} != {ref[0]}"
        assert torch.equal(got[1], ref[1]), f"G mismatch {tag}"
        assert got[2] == ref[2], f"sepsets mismatch {tag}"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("p", "seed"),
    [
        (8, 1),
        (8, 2),
        (8, 3),
        (8, 4),
        (10, 1),
        (10, 2),
        (10, 3),
        (10, 4),
    ],
)
def test_k0_symmetry_matches_directed_path(p, seed):
    """k=0 upper-triangle CI must leave skeleton/CPDAG/sepsets unchanged.

    The workspace path halves the k=0 test count (one test per undirected
    pair) and mirrors the empty sepset into both ``ranges`` cells. Compare
    a full CUDA PC-stable run against the directed flat path on 8 cases.
    """
    import os

    import torch

    from whydra.algorithms.graph_core import CIT, CausalGraph
    from whydra.algorithms.standalone import pc_algo

    rng = np.random.default_rng(1000 + seed)
    data = rng.standard_normal((300, p))

    def run(smallk):
        old = {
            key: os.environ.get(key)
            for key in ("WHYDRA_PC_SMALLK", "WHYDRA_PC_WORKSPACE")
        }
        try:
            os.environ["WHYDRA_PC_SMALLK"] = "1" if smallk else "0"
            os.environ["WHYDRA_PC_WORKSPACE"] = "1"
            cg = CausalGraph(p, [f"x{i}" for i in range(p)], device="cuda")
            cg.set_ind_test(CIT(data, method="fisherz"))
            pc_algo.pc_stable(
                cg, alpha=ALPHA, n_jobs=1, show_progress=False, device="cuda"
            )
            torch.cuda.synchronize()
            seps = _canonical_sepsets(cg.sepset_ranges, cg.sepset_S, p)
            return np.array(cg.G.graph), seps
        finally:
            for key, val in old.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

    g_dir, s_dir = run(False)
    g_sym, s_sym = run(True)
    assert np.array_equal(g_dir != 0, g_sym != 0), "skeleton mismatch"
    assert np.array_equal(g_dir, g_sym), "CPDAG mismatch"
    assert s_dir == s_sym, "sepsets mismatch"
