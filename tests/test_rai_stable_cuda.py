"""Compare upstream CUDA RAI with the sequential CPU learner.

Direct learner comparisons check the skeleton, sepsets, CI trace, CI count,
and CPDAG. The script also compares final graphs through the public
``rai_stable`` entry point, which uses the CUDA adapter.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whydra.algorithms.graph_core.cuda_fisher_z import _ldlt_cuda
from whydra.algorithms.graph_core.cuda_graph_batch import _graph_batch_cuda
from whydra.algorithms.graph_core.cuda_rai import RAIStableCudaLearner
from whydra.algorithms.standalone.rai_algo import RAIStableLearner, rai_stable


def _sepsets_of(learner) -> dict:
    """Non-empty reference sepsets: ``(x, y) -> {frozenset(S)}``, both directions."""
    out = {}
    for (i, j), S in learner.sepset._sepset.items():
        if not S:
            continue
        u = {frozenset(int(v) for v in S)}
        out[(i, j)] = u
        out[(j, i)] = u
    return out


def _dense(n, p, seed=0):
    rng = np.random.default_rng(seed)
    w = np.tril(rng.normal(size=(p, p)), k=-1) * 0.3
    noise = rng.normal(size=(n, p))
    data = noise.copy()
    for j in range(p):
        data[:, j] = data[:, :j] @ w[j, :j] + noise[:, j]
    return data


def _sparse(n, p, seed=0, max_parents=3):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n, p))
    data = noise.copy()
    for j in range(p):
        m = min(max_parents, j)
        if m == 0:
            continue
        parents = rng.choice(j, size=m, replace=False)
        data[:, j] += data[:, parents] @ (rng.normal(size=m) * 0.5)
    return data


def _names(p):
    return [f"X{i}" for i in range(p)]


def _run_seq(data: np.ndarray):
    learner = RAIStableLearner(
        data=data, alpha=0.05, indep_test="fisherz", node_names=_names(data.shape[1])
    )
    learner.trace = []
    learner.learn_structure(n_jobs=1)
    return {
        "graph": np.asarray(learner._to_general_graph().graph),
        "sep": _sepsets_of(learner),
        "trace": learner.trace,
        "n_ci": learner.n_ci_tests,
    }


def _run_cuda(data: np.ndarray):
    learner = RAIStableCudaLearner(
        data=data, alpha=0.05, indep_test="fisherz", node_names=_names(data.shape[1])
    )
    learner.trace = []
    learner.learn_structure()
    return {
        "graph": np.asarray(learner._to_general_graph().graph),
        "sep": _sepsets_of(learner),
        "trace": learner.trace,
        "n_ci": learner.n_ci_tests,
    }


def _fmt_diff(a: np.ndarray, b: np.ndarray) -> str:
    n = int(np.sum(a != b))
    if n == 0:
        return "equal"
    idx = np.argwhere(a != b)[:6]
    return f"{n} cells, first={idx.tolist()}"


def _fmt_trace_diff(t_seq, t_cuda) -> str:
    for i, (a, b) in enumerate(zip(t_seq, t_cuda, strict=False)):
        if a != b:
            return f"first divergence at #{i}: seq={a} cuda={b}"
    if len(t_seq) != len(t_cuda):
        return f"length {len(t_seq)} vs {len(t_cuda)}"
    return "equal"


def _compare(data: np.ndarray, tag: str) -> bool:
    print(f"\n{tag}  n={data.shape[0]} p={data.shape[1]}")
    seq = _run_seq(data)
    cuda = _run_cuda(data)
    graph = np.array_equal(seq["graph"], cuda["graph"])
    skel = np.array_equal(seq["graph"] != 0, cuda["graph"] != 0)
    sep_ident = seq["sep"] == cuda["sep"]
    trace = seq["trace"] == cuda["trace"]
    n_ci = seq["n_ci"] == cuda["n_ci"]
    print(
        f"  RAI-stable-cuda vs seq: skel={skel} cpdag={graph} "
        f"sepset_ident={sep_ident} trace={trace} n_ci_tests={n_ci} "
        f"cpdag_diff={_fmt_diff(seq['graph'], cuda['graph'])} "
        f"trace_diff={_fmt_trace_diff(seq['trace'], cuda['trace'])}"
    )
    if not sep_ident:
        keys = set(seq["sep"]) | set(cuda["sep"])
        shown = 0
        for k in sorted(keys):
            if seq["sep"].get(k) != cuda["sep"].get(k):
                print(
                    f"    sepset {k}: seq={seq['sep'].get(k)} cuda={cuda['sep'].get(k)}"
                )
                shown += 1
                if shown >= 8:
                    break
    return skel and graph and sep_ident and trace and n_ci


_SPARSE_CASES = (
    (20, 3),
    (30, 3),
    (40, 4),
)


def _api_check(data: np.ndarray) -> bool:
    """Compare final graphs from the public CPU and CUDA entry points."""
    seq = rai_stable(data, alpha=0.05, indep_test="fisherz", n_jobs=1)
    cud = rai_stable(data, alpha=0.05, indep_test="fisherz", device="cuda")
    ok = np.array_equal(np.asarray(seq.G.graph), np.asarray(cud.G.graph))
    print(f"  rai_stable API (n_jobs=1 vs device='cuda'): {ok}")
    return ok


def _pool_order_contract() -> bool:
    """Check order-one replay against the Python reference without using CUDA.

    The k == 1 fast path must visit parents in the order returned by
    pot_parents: iterate (parents | undirected) - {other}. CPython builds a
    new table for the set difference. Once the set outgrows the eight-slot
    minimum, skipping other while iterating the union can change the order
    and hence the first conditioning set tested.

    Across the seeded sweep, the cached-union difference must match the
    reference every time. At least one case must distinguish that order
    from filtering the union in place.
    """
    import random

    rng = random.Random(0xA1)
    mismatches = 0
    filter_differs = 0
    for _ in range(20000):
        size = rng.randint(1, 40)
        elem = rng.sample(range(200), size)
        parents = set(elem[0::2])
        undirected = set(elem[1::2])
        other = rng.choice(sorted(parents | undirected))
        # Compare the reference pool with the k == 1 replay operation:
        # take the cached union, then apply the equivalent of set.__sub__.
        reference = list((parents | undirected) - {other})
        cached_union = parents | undirected
        ours = list(cached_union - {other})
        if ours != reference:
            mismatches += 1
        # Forbidden alternative: iterate the live union, skip ``other``.
        if [v for v in cached_union if v != other] != reference:
            filter_differs += 1
    ok = mismatches == 0 and filter_differs > 0
    print(
        f"  order-1 pool-order contract: mismatches={mismatches} "
        f"(filter-vs-difference witnesses={filter_differs}) -> {ok}"
    )
    return ok


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rai_cuda_matches_sequential() -> None:
    """Dense SEM: graph, sepsets, CI trace and CI count must match the CPU."""
    assert _compare(_dense(200, 8, seed=0), "pytest p=8 dense SEM")


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("skip: no CUDA")
        return 2
    fz, gb = _ldlt_cuda(), _graph_batch_cuda()
    print(
        f"k_max fisher_z={int(fz.k_max())} combinadic={int(gb.k_max())} "
        f"fused={int(gb.fused_k_max())}"
    )
    ok = True
    ok = _pool_order_contract() and ok
    ok = _compare(_dense(200, 8, seed=0), "p=8 dense SEM") and ok
    ok = _api_check(_dense(200, 8, seed=0)) and ok
    ok = _compare(_dense(200, 12, seed=0), "p=12 dense SEM") and ok
    # Dense case whose order-1 potential-parent pools exceed the CPython
    # 8-slot minimum, so the k==1 replay path is exercised on sizeable sets.
    ok = _compare(_dense(300, 16, seed=0), "p=16 dense SEM (order-1 pools > 8)") and ok
    for p, max_parents in _SPARSE_CASES:
        ok = (
            _compare(
                _sparse(500, p, seed=0, max_parents=max_parents),
                f"p={p} max_parents={max_parents}",
            )
            and ok
        )
    print("\nALL_OK" if ok else "\nMISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
