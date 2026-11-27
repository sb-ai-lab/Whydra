"""NumPy Schur/Cholesky Fisher-Z vs CIT.fisher_z (no GPU)."""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whydra.algorithms.graph_core.fisher_z_schur_chol import (
    fisher_z_batch_inv,
    fisher_z_batch_schur,
    fisher_z_schur_chol,
    fisher_z_schur_chol_fused,
    partial_corr_r,
    partial_corr_r_batch_schur,
    partial_corr_r_fused,
)
from whydra.algorithms.graph_core.independence_tests import CIT


def _random_gaussian(n_samples: int, n_vars: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = np.tril(rng.normal(size=(n_vars, n_vars)), k=-1) * 0.3
    noise = rng.normal(size=(n_samples, n_vars))
    data = noise.copy()
    for j in range(n_vars):
        data[:, j] = data[:, :j] @ w[j, :j] + noise[:, j]
    return data


def _cases(n_vars: int, max_k: int):
    yield 0, 1, ()
    yield 0, 2, (1,)
    yield 1, 3, (0, 2)
    for k in range(0, max_k + 1):
        for S in combinations(range(2, n_vars), k):
            yield 0, 1, S


def main() -> int:
    n_samples, n_vars, max_k = 400, 10, 6
    data = _random_gaussian(n_samples, n_vars, seed=7)
    cit = CIT(data, method="fisherz")
    C = cit.corr_matrix

    n_ok = 0
    n_fail = 0
    worst_p = 0.0
    worst_r = 0.0
    for x, y, Z in _cases(n_vars, max_k):
        p_cit = cit.fisher_z(x, y, Z)
        p_np = fisher_z_schur_chol(C, n_samples, x, y, Z)
        p_fused = fisher_z_schur_chol_fused(C, n_samples, x, y, Z)
        r_np = partial_corr_r(C, x, y, Z)
        r_fused = partial_corr_r_fused(C, x, y, Z)
        d_p = abs(p_cit - p_np)
        d_pf = abs(p_cit - p_fused)
        d_r = abs(r_np - r_fused)
        worst_p = max(worst_p, d_p, d_pf)
        worst_r = max(worst_r, d_r)
        ok = (
            np.isfinite(p_np)
            and np.isfinite(p_fused)
            and d_p < 1e-12
            and d_pf < 1e-12
            and d_r < 1e-12
        )
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            print(
                f"FAIL x={x} y={y} Z={Z}  "
                f"cit={p_cit:.16g} np={p_np:.16g} fused={p_fused:.16g}  "
                f"|p|={d_p:.3e} |p_fused|={d_pf:.3e} |r|={d_r:.3e}"
            )

    print(
        f"passed={n_ok} failed={n_fail}  "
        f"worst_|cit-np|={worst_p:.3e} worst_|np-fused_r|={worst_r:.3e}  "
        f"n={n_samples} p={n_vars} max_k={max_k}"
    )
    if n_fail:
        return 1

    by_k: dict[int, list] = {}
    for x, y, Z in _cases(n_vars, max_k):
        by_k.setdefault(len(Z), []).append((x, y, Z))
    worst_b = 0.0
    for k, rows in by_k.items():
        xs = np.array([r[0] for r in rows], dtype=np.int64)
        ys = np.array([r[1] for r in rows], dtype=np.int64)
        Ss = np.array([list(r[2]) for r in rows], dtype=np.int64).reshape(len(rows), k)
        p_cit = np.array([cit.fisher_z(x, y, Z) for x, y, Z in rows])
        p_sch = fisher_z_batch_schur(C, n_samples, xs, ys, Ss)
        p_inv = fisher_z_batch_inv(C, n_samples, xs, ys, Ss)
        d = max(
            float(np.max(np.abs(p_cit - p_sch))), float(np.max(np.abs(p_cit - p_inv)))
        )
        worst_b = max(worst_b, d)
        ok = d < 1e-12
        print(
            f"batch k={k} B={len(rows)} max_|cit-batch|={d:.3e} {'OK' if ok else 'FAIL'}"
        )
        if not ok:
            return 1
    print(f"batch vs CIT worst_|Δp|={worst_b:.3e}")

    # css=0, eps=0: (css+eps)^{-1} is 1/0 → non-finite Schur.
    C0 = np.array(
        [[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 0.0]],
        dtype=np.float64,
    )
    r0 = partial_corr_r(C0, 0, 1, (2,))
    r0f = partial_corr_r_fused(C0, 0, 1, (2,))
    r0b = float(partial_corr_r_batch_schur(C0, [0], [1], [[2]])[0])
    ok_nan = (not np.isfinite(r0)) and (not np.isfinite(r0f)) and (not np.isfinite(r0b))
    eps = 1e-12
    r_eps = partial_corr_r(C0, 0, 1, (2,), eps=eps)
    inv = 1.0 / eps
    expect = (0.5 - 0.3 * 0.4 * inv) / np.sqrt(
        abs((1.0 - 0.3 * 0.3 * inv) * (1.0 - 0.4 * 0.4 * inv))
    )
    ok_eps = np.isfinite(r_eps) and abs(r_eps - expect) < 1e-12
    print(f"k1 css=0 eps=0 r={r0} fused={r0f} batch={r0b} {'OK' if ok_nan else 'FAIL'}")
    print(
        f"k1 css=0 eps={eps:g} r={r_eps:.16g} expect={expect:.16g} {'OK' if ok_eps else 'FAIL'}"
    )
    return 0 if (ok_nan and ok_eps) else 1


def test_fisher_z_schur_chol_matches_cit() -> None:
    """NumPy Schur/Cholesky Fisher-Z reference (no GPU required)."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
