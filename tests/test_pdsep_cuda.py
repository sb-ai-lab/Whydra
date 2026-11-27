"""CUDA collider marks, Possible-D-SEP reachability and separator packing."""

import sys
from itertools import permutations
from pathlib import Path

import numpy as np

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests._pdsep_cases import reachability_cases
from whydra.algorithms.graph_core.cuda_graph_batch import UnionAccum, _pack_union_csr
from whydra.algorithms.graph_core.cuda_pdsep import (
    mark_colliders_qreach,
    sepset_csr_to_dict,
    sepset_dict_to_csr,
)
from whydra.algorithms.standalone.fci_algo_v2 import qreach
from whydra.algorithms.standalone.fci_search import orient_colliders
from whydra.algorithms.standalone.pag_rules import PAG_ARROW, PAG_CIRCLE


def _check_union_pack() -> bool:
    """``_pack_union_csr`` over UnionAccum vs a dict-of-sets oracle.

    Regression: the all-empty-separator case (every hit at ``k=0``, so no
    members at all) must not index an empty member table; empty pairs are
    stored as ``(0, 0)`` slices, absent pairs stay ``-1``.
    """
    import numpy as np

    p = 6
    rng = np.random.default_rng(4)
    acc = UnionAccum(p)
    oracle: dict[tuple[int, int], set] = {}
    for _ in range(40):
        x = int(rng.integers(0, p))
        y = int(rng.integers(0, p))
        if x == y:
            continue
        k = int(rng.integers(0, 4))
        S = rng.choice(p, size=k, replace=False)
        acc.add(
            np.asarray([x], dtype=np.int64),
            np.asarray([y], dtype=np.int64),
            S.reshape(1, -1).astype(np.int64),
        )
        oracle.setdefault((x, y), set()).update(int(v) for v in S)

    # force the all-k=0 case explicitly
    acc.add(
        np.asarray([1], np.int64), np.asarray([2], np.int64), np.zeros((1, 0), np.int64)
    )
    oracle.setdefault((1, 2), set())

    S, ranges = _pack_union_csr(acc, p)
    got = sepset_csr_to_dict(S, ranges, p)
    merged: dict[tuple[int, int], set] = {}
    for (x, y), members in oracle.items():
        merged.setdefault((min(x, y), max(x, y)), set()).update(members)
    # both triangles carry the undirected union (empty unions are dropped)
    want: dict[tuple[int, int], set] = {}
    for (lo, hi), members in merged.items():
        if members:
            want[(lo, hi)] = set(members)
            want[(hi, lo)] = set(members)
    if got != want:
        print("union pack mismatch:", {k: sorted(v) for k, v in got.items()})
        return False
    return True


def _check_collider_cases():
    for case in reachability_cases():
        count = len(case.marked)
        members, offsets = sepset_dict_to_csr(case.separators, count)
        for _ in range(2):
            marks, mask, _, _ = mark_colliders_qreach(
                case.skeleton.copy(),
                members,
                offsets,
                case.ambiguous,
                qreach_cpu=qreach,
            )
            np.testing.assert_array_equal(marks, case.marked, err_msg=case.name)
            for source, expected in enumerate(case.reachable):
                np.testing.assert_array_equal(
                    np.flatnonzero(mask[source]), expected, err_msg=case.name
                )


def _check_random_graphs(rng):
    for _ in range(80):
        n = int(rng.integers(3, 25))
        upper = np.triu(rng.choice([0, 1, 2], (n, n), p=[0.55, 0.4, 0.05]), 1)
        skeleton = upper + upper.T
        separators = {}
        ambiguous = set()
        for _ in range(n):
            a, b, c = rng.choice(n, size=3, replace=False)
            if rng.random() < 0.5:
                separators.setdefault((a, c), set()).add(b)
            else:
                ambiguous.add((a, b, c))

        cpu = np.where(skeleton == 2, PAG_ARROW, np.where(skeleton, PAG_CIRCLE, 0))
        orient_colliders(cpu, separators, ambiguous, missing=())
        expected_marks = np.where(cpu == PAG_ARROW, 2, np.where(cpu, 1, 0))
        members, offsets = sepset_dict_to_csr(separators, n)
        marks, mask, _, _ = mark_colliders_qreach(
            skeleton, members, offsets, ambiguous, qreach_cpu=qreach
        )
        np.testing.assert_array_equal(marks, expected_marks)
        for source in range(n):
            np.testing.assert_array_equal(
                np.flatnonzero(mask[source]), qreach(source, expected_marks)
            )


def main() -> int:
    rng = np.random.default_rng(0)
    _check_collider_cases()
    _check_random_graphs(rng)
    p = 24
    sepset = {ij: set() for ij in permutations(range(p), 2)}
    for _ in range(15):
        x, z = rng.integers(0, p, size=2)
        if x == z:
            continue
        y = int(rng.integers(0, p))
        if y in (x, z):
            continue
        sepset[(x, z)].add(y)
        sepset[(z, x)].add(y)

    S, ranges = sepset_dict_to_csr(sepset, p)
    back = sepset_csr_to_dict(S, ranges, p)
    csr_ok = all(back.get(k, set()) == set(sepset[k]) for k in sepset)
    print(f"csr_roundtrip={csr_ok}")
    union_ok = _check_union_pack()
    print(f"union_pack={union_ok}")

    return 0 if csr_ok and union_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
