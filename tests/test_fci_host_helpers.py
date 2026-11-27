"""C++ FCI host helpers vs the Python reference implementations."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whydra.algorithms.standalone.fci_algo_v2 import (
    _adjacency_lists,
    _enumerate_pds_blocks,
    _enumerate_pds_blocks_py,
    _possible_dsep,
)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 2
    from whydra.algorithms.graph_core import _cuda

    _cuda.preload()
    rng = np.random.default_rng(0)
    rc = 0
    for t in range(100):
        p = int(rng.integers(5, 30))
        amat = rng.integers(0, 4, size=(p, p)).astype(np.int32)
        np.fill_diagonal(amat, 0)
        adj = _adjacency_lists(amat)
        pairs = [(int(rng.integers(0, p)), int(rng.integers(0, p))) for _ in range(8)]
        got = _cuda.fci_host.pds_batch(torch.from_numpy(amat), pairs)
        for (x, y), g in zip(pairs, got, strict=False):
            exp = _possible_dsep(amat, x, y, adj)
            if g != exp:
                print(f"pds_batch mismatch t={t} ({x},{y}): {g} != {exp}")
                rc = 1
                break

        pds = sorted(
            rng.choice(p, size=int(rng.integers(0, p)), replace=False).tolist()
        )
        adj_a = sorted(
            rng.choice(p, size=int(rng.integers(0, p)), replace=False).tolist()
        )
        adj_b = sorted(
            rng.choice(p, size=int(rng.integers(0, p)), replace=False).tolist()
        )
        mm = float("inf") if t % 2 else int(rng.integers(2, 8))
        a = _enumerate_pds_blocks_py(pds, adj_a, adj_b, mm, p, 1 << 16)
        b = _enumerate_pds_blocks(pds, adj_a, adj_b, mm, p, 1 << 16)
        if (a is None) != (b is None) or (
            a is not None
            and (
                [(s, x.shape) for s, x in a] != [(s, x.shape) for s, x in b]
                or any(
                    not np.array_equal(x, y)
                    for (_, x), (_, y) in zip(a, b, strict=False)
                )
            )
        ):
            print(f"pds_blocks mismatch t={t} pds={pds} A={adj_a} B={adj_b} m={mm}")
            rc = 1
            break
    print("FCI host helpers:", "OK" if rc == 0 else "FAILED")
    return rc


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pds_host_helpers_match_python() -> None:
    """C++ PDS traversal, block enumeration and collider pass match Python."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
