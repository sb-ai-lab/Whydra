"""The directional FCI host helper uses a candidate's row for support."""

import numpy as np
import pytest

from whydra.algorithms.standalone.fci_algo_v2 import _possible_dsep


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("support_edge", "expected"),
    [((2, 3), [3, 2, 1]), ((3, 2), [3, 1])],
)
def test_candidate_row_determines_possible_dsep_support(support_edge, expected):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA extensions require a CUDA-capable environment")
    from whydra.algorithms.graph_core import _cuda

    # Source 0 discovers candidate 2 through 1. Only neighbor 3 reaches the
    # source on a semidirected path. Reverse one mark to change its eligibility.
    marks = np.array(
        [
            [0, 1, 0, 2, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    marks[support_edge] = 1
    assert _possible_dsep(marks, 0, 4) == expected
    assert _cuda.fci_host.pds_batch(torch.from_numpy(marks), [(0, 4)]) == [expected]
