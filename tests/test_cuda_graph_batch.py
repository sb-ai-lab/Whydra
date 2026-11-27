"""CUDA graph_batch: compact adj, combinadic fill, first-sepset vs itertools / CIT."""

import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whydra.algorithms.graph_core import CIT
from whydra.algorithms.graph_core.cuda_fisher_z import CudaFisherZ
from whydra.algorithms.graph_core.cuda_graph_batch import (
    SepsetCSR,
    _graph_batch_cuda,
    _offsets_from_counts,
    apply_hits,
    build_all_s,
    compact_adj,
    count_tests,
    delete_if_indep,
    edge_present,
    emit_pairs,
    fill_batch,
    run_depth,
    upload_G,
)


def _undirected(p: int, edges) -> np.ndarray:
    G = np.zeros((p, p), dtype=np.int8)
    for i, j in edges:
        G[i, j] = 1
        G[j, i] = 1
    return G


def _test_compact() -> None:
    G = _undirected(6, [(0, 1), (0, 3), (0, 4), (2, 5), (3, 4)])
    nbrs, deg = compact_adj(G)
    deg_np = deg.cpu().numpy()
    nbrs_np = nbrs.cpu().numpy()
    for x in range(6):
        expect = np.where(G[x] != 0)[0]
        assert deg_np[x] == len(expect), (x, deg_np[x], expect)
        assert np.array_equal(nbrs_np[x, : deg_np[x]], expect), (x, nbrs_np[x], expect)
    print("compact: ok")


def _test_delete_if_indep() -> None:
    G = _undirected(4, [(0, 1), (0, 2), (1, 2), (2, 3)])
    G_t = upload_G(G, "cuda")
    x = torch.tensor([0, 2], dtype=torch.int32, device=G_t.device)
    y = torch.tensor([1, 3], dtype=torch.int32, device=G_t.device)
    indep = torch.tensor([1, 0], dtype=torch.uint8, device=G_t.device)
    delete_if_indep(G_t, x, y, indep)
    got = G_t.detach().cpu().numpy()
    assert got[0, 1] == 0
    assert got[1, 0] == 0
    assert got[2, 3] == 1
    assert got[3, 2] == 1
    present = edge_present(G_t, x, y).cpu().numpy()
    assert list(present) == [0, 1]
    delete_if_indep(G_t, x, y, torch.ones(2, dtype=torch.uint8, device=G_t.device))
    got = G_t.detach().cpu().numpy()
    assert got[2, 3] == 0
    assert got[3, 2] == 0
    present = edge_present(G_t, x, y).cpu().numpy()
    assert list(present) == [0, 0]
    print("delete_if_indep: ok")


def _test_apply_hits() -> None:
    G = _undirected(6, [(0, 1), (0, 3), (2, 5), (3, 4)])
    G_t = upload_G(G, "cuda")
    x = torch.tensor([0, 2], dtype=torch.int32, device=G_t.device)
    y = torch.tensor([1, 5], dtype=torch.int32, device=G_t.device)
    apply_hits(G_t, x, y)
    got = G_t.detach().cpu().numpy()
    assert got[0, 1] == 0
    assert got[1, 0] == 0
    assert got[2, 5] == 0
    assert got[5, 2] == 0
    assert got[0, 3] == 1
    assert got[3, 4] == 1
    print("apply_hits: ok")


def _test_count_and_fill() -> None:
    G = _undirected(8, [(i, j) for i in range(8) for j in range(i + 1, 8)])
    nbrs, deg = compact_adj(G)
    px, py = emit_pairs(nbrs, deg)
    for k in (0, 1, 2, 3):
        counts = count_tests(nbrs, deg, px, py, k)
        counts_np = counts.cpu().numpy()
        px_np = px.cpu().numpy()
        py_np = py.cpu().numpy()
        expect = []
        for e, (x, y) in enumerate(zip(px_np, py_np, strict=False)):
            cand = [int(v) for v in np.where(G[x] != 0)[0] if int(v) != int(y)]
            c = math.comb(len(cand), k) if len(cand) >= k else 0
            assert int(counts_np[e]) == c, (k, x, y, counts_np[e], c)
            expect.append((int(x), int(y), cand))
        off = _offsets_from_counts(counts)
        x_t, y_t, S_t = fill_batch(nbrs, deg, px, py, off, k)
        S = S_t.cpu().numpy()
        xs = x_t.cpu().numpy()
        ys = y_t.cpu().numpy()
        b = 0
        for x, y, cand in expect:
            for comb in combinations(cand, k):
                assert xs[b] == x, (k, b, xs[b], ys[b], x, y)
                assert ys[b] == y, (k, b, xs[b], ys[b], x, y)
                assert tuple(int(v) for v in S[b]) == tuple(comb), (
                    k,
                    b,
                    S[b].tolist(),
                    comb,
                )
                b += 1
        assert b == int(S.shape[0])
        print(f"count+fill k={k}: B={b} ok")


def _random_gaussian(n_samples: int, n_vars: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = np.tril(rng.normal(size=(n_vars, n_vars)), k=-1) * 0.4
    noise = rng.normal(size=(n_samples, n_vars))
    data = noise.copy()
    for j in range(n_vars):
        data[:, j] = data[:, :j] @ W[j, :j] + noise[:, j]
    return data


def _test_first_sepset() -> None:
    n_samples, p, alpha = 400, 8, 0.05
    data = _random_gaussian(n_samples, p, seed=4)
    cit = CIT(data, method="fisherz")
    gpu = CudaFisherZ(cit.corr_matrix, n_samples)
    G = np.ones((p, p), dtype=np.int8)
    np.fill_diagonal(G, 0)
    for k in (0, 2):
        rx, ry, S_hit, _ = run_depth(gpu, G, G, k, alpha, fci=False)
        expect = {}
        for x in range(p):
            nx = np.where(G[x] != 0)[0]
            for y in nx:
                cand = [int(v) for v in nx if int(v) != int(y)]
                found = None
                if k == 0:
                    pval = float(cit.fisher_z(int(x), int(y), ()))
                    if pval > alpha:
                        found = ()
                else:
                    for S in combinations(cand, k):
                        pval = float(cit.fisher_z(int(x), int(y), S))
                        if pval > alpha:
                            found = tuple(int(v) for v in S)
                            break
                if found is not None:
                    expect[(int(x), int(y))] = found
        got = {
            (int(rx[i]), int(ry[i])): tuple(int(v) for v in S_hit[i])
            for i in range(len(rx))
        }
        assert set(got) == set(expect), (k, set(got) ^ set(expect), got, expect)
        if k > 0:
            for (x, y), S in got.items():
                pval = float(cit.fisher_z(x, y, S))
                assert pval > alpha, (k, x, y, S, pval)
        print(f"first-sepset k={k}: {len(got)} independent pairs, sepsets valid")


def _triples(x, y, S) -> list[tuple[int, int, tuple[int, ...]]]:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        S = S.detach().cpu().numpy()
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    S = np.asarray(S)
    if S.ndim == 1:
        S = S.reshape(-1, 1) if S.size else np.zeros((x.size, 0), dtype=np.int64)
    if S.shape[0] != x.size:
        S = S.reshape(x.size, -1)
    out = []
    for i in range(x.size):
        out.append((int(x[i]), int(y[i]), tuple(int(v) for v in S[i])))
    return out


def _host_directed_tests(G: np.ndarray, k: int):
    tests = []
    p = G.shape[0]
    for x in range(p):
        nx = np.where(G[x] != 0)[0]
        for y in nx:
            cand = [int(v) for v in nx if int(v) != int(y)]
            if k == 0:
                tests.append((int(x), int(y), ()))
                continue
            if len(cand) < k:
                continue
            for S in combinations(cand, k):
                tests.append((int(x), int(y), tuple(int(v) for v in S)))
    return tests


def _host_all_s_tests(G: np.ndarray, k: int):
    p = G.shape[0]
    tests = []
    if k == 0:
        xs, ys = np.triu_indices(p, 1)
        for x, y in zip(xs.tolist(), ys.tolist(), strict=False):
            if G[x, y] != 0:
                tests.append((int(x), int(y), ()))
        return tests
    seen = set()
    for x in range(p):
        nx = np.where(G[x] != 0)[0]
        for y in nx:
            if int(y) <= int(x):
                continue
            for src, other in ((int(x), int(y)), (int(y), int(x))):
                cand = [int(v) for v in np.where(G[src] != 0)[0] if int(v) != other]
                for S in combinations(cand, k):
                    t = (int(x), int(y), tuple(int(v) for v in S))
                    if t not in seen:
                        seen.add(t)
                        tests.append(t)
    return tests


def _gpu_all_s_tests(gpu, G: np.ndarray, k: int, alpha: float, b_max: int = 1 << 26):
    from whydra.algorithms.graph_core import cuda_graph_batch as gb

    tests = []
    orig = gpu.batch_r
    old_fused = gb.USE_FUSED
    gb.USE_FUSED = False

    def wrapped(x, y, S, out=None, **kwargs):
        tests.extend(_triples(x, y, S))
        return orig(x, y, S, out=out, **kwargs)

    gpu.batch_r = wrapped
    try:
        run_depth(gpu, G, G, k, alpha, fci=False, b_max=b_max)
    finally:
        gpu.batch_r = orig
        gb.USE_FUSED = old_fused
    return tests


def _test_build_all_s_matches_itertools() -> None:
    G = _undirected(8, [(i, j) for i in range(8) for j in range(i + 1, 8)])
    for k in (0, 1, 2, 3):
        got = []
        for batch in build_all_s(G, G, k, b_max=11):
            got.extend(_triples(batch["x"], batch["y"], batch["S"]))
        expect = _host_all_s_tests(G, k)
        assert set(got) == set(expect), (k, len(got), len(expect))
        if k >= 2 and got:
            # S-major: batches are grouped by increasing S (no post-fill sort)
            prev = None
            for t in got:
                cur = t[2]
                if prev is not None:
                    assert cur >= prev, (k, prev, cur)
                prev = cur
        print(f"build_all_s k={k}: {len(got)} triples (chunked b_max=11)")
    n_samples, p, alpha = 400, 8, 0.05
    data = _random_gaussian(n_samples, p, seed=4)
    cit = CIT(data, method="fisherz")
    gpu = CudaFisherZ(cit.corr_matrix, n_samples)
    complete = np.ones((p, p), dtype=np.int8)
    np.fill_diagonal(complete, 0)
    leftover = complete.copy()
    leftover[0, 1] = leftover[1, 0] = 0
    leftover[2, 5] = leftover[5, 2] = 0
    for G, tag in ((complete, "complete"), (leftover, "leftover")):
        for k in (0, 1, 2, 3):
            host = _host_directed_tests(G, k)
            gpu_t = _gpu_all_s_tests(gpu, G, k, alpha)
            hs, gs = set(host), set(gpu_t)
            assert gs == hs, (
                f"{tag} k={k} only_gpu={sorted(gs - hs)[:8]} "
                f"n_host={len(host)} n_gpu={len(gpu_t)}"
            )
            print(f"all-S run_depth {tag} k={k}: gpu {len(gs)} ⊆ host {len(hs)}")
            if k >= 1:
                gpu_c = _gpu_all_s_tests(gpu, G, k, alpha, b_max=5)
                assert set(gpu_c) == hs, f"{tag} k={k} b_max=5 test mismatch"


def _test_immediate_delete_graph_stable() -> None:
    """Chunked skip+delete must zero the same edges as one giant all-S batch."""
    n_samples, p, alpha = 400, 8, 0.05
    data = _random_gaussian(n_samples, p, seed=4)
    cit = CIT(data, method="fisherz")
    gpu = CudaFisherZ(cit.corr_matrix, n_samples)
    G = np.ones((p, p), dtype=np.int8)
    np.fill_diagonal(G, 0)
    for k, b_small in ((0, 1), (1, 3), (2, 5)):
        Ga = upload_G(G, "cuda")
        run_depth(gpu, Ga, Ga, k, alpha, apply_to=Ga, b_max=b_small)
        Gb = upload_G(G, "cuda")
        run_depth(gpu, Gb, Gb, k, alpha, apply_to=Gb)
        a = Ga.detach().cpu().numpy()
        b = Gb.detach().cpu().numpy()
        assert np.array_equal(a, b), (k, np.argwhere(a != b)[:8])
        tiny = _gpu_all_s_tests(gpu, G, k, alpha, b_max=b_small)
        full = _host_directed_tests(G, k)
        assert set(tiny) == set(full), k
        print(f"immediate-delete k={k}: G match, {len(tiny)} tests (b_max={b_small})")


def _test_fused_stripes_match_giant() -> None:
    """Fused rank stripes and giant unfused all-S zero the same edges."""
    from whydra.algorithms.graph_core import cuda_graph_batch as gb

    n_samples, p, alpha = 400, 8, 0.05
    data = _random_gaussian(n_samples, p, seed=4)
    cit = CIT(data, method="fisherz")
    gpu = CudaFisherZ(cit.corr_matrix, n_samples)
    G = np.ones((p, p), dtype=np.int8)
    np.fill_diagonal(G, 0)
    old = (gb.USE_FUSED, gb.ALL_S_CHUNK)
    try:
        gb.USE_FUSED = False
        Ga = upload_G(G, "cuda")
        run_depth(gpu, Ga, Ga, 1, alpha, apply_to=Ga, b_max=1 << 26)
        gb.USE_FUSED = True
        Gb = upload_G(G, "cuda")
        run_depth(gpu, Gb, Gb, 1, alpha, apply_to=Gb, b_max=4)
        a = Ga.detach().cpu().numpy()
        b = Gb.detach().cpu().numpy()
        assert np.array_equal(a, b), np.argwhere(a != b)[:8]
        tiny = _gpu_all_s_tests(gpu, G, 1, alpha, b_max=4)
        giant = _gpu_all_s_tests(gpu, G, 1, alpha, b_max=1 << 26)
        assert set(tiny) == set(giant)
        print(
            f"fused stripes k=1: G match, {len(tiny)} tests (b_max=4) "
            f"vs {len(giant)} giant"
        )
    finally:
        gb.USE_FUSED, gb.ALL_S_CHUNK = old


def _test_claim_unique_i() -> None:
    """Duplicate directed (0,1) claims once; (1,0) is a separate cell."""
    G = _undirected(4, [(0, 1), (0, 2), (1, 2), (2, 3)])
    G_t = upload_G(G, "cuda")
    nbrs, deg = compact_adj(G_t)
    csr = SepsetCSR(4, G_t.device)
    px = torch.tensor([0, 1, 0], dtype=torch.int32, device=G_t.device)
    py = torch.tensor([1, 0, 1], dtype=torch.int32, device=G_t.device)
    first = torch.zeros(3, dtype=torch.int64, device=G_t.device)
    csr.claim(px, py, first, nbrs, deg, k=1, rank_base=0, rank_bases=None)
    S, ranges = csr.pack()
    start01 = int(ranges[0, 1, 0].cpu())
    end01 = int(ranges[0, 1, 1].cpu())
    start10 = int(ranges[1, 0, 0].cpu())
    end10 = int(ranges[1, 0, 1].cpu())
    assert start01 >= 0, (start01, end01)
    assert end01 - start01 == 1, (start01, end01)
    assert start10 >= 0, (start10, end10)
    assert end10 - start10 == 1, (start10, end10)
    assert int(ranges[0, 2, 0].cpu()) == -1
    n_claimed = int((ranges[:, :, 0] >= 0).sum().cpu())
    assert n_claimed == 2, n_claimed
    assert list(S[start01:end01].cpu()) == [2], S[start01:end01]
    assert list(S[start10:end10].cpu()) == [2], S[start10:end10]
    print("claim unique i: ok")


def main() -> int:
    _graph_batch_cuda()
    _test_claim_unique_i()
    _test_compact()
    _test_apply_hits()
    _test_delete_if_indep()
    _test_count_and_fill()
    _test_build_all_s_matches_itertools()
    _test_immediate_delete_graph_stable()
    _test_fused_stripes_match_giant()
    _test_first_sepset()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
