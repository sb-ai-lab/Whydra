import numpy as np
import pytest

from whydra.algorithms.graph_core import CIT


def test_scalar_batch_keeps_malformed_request_at_its_input_position():
    data = np.random.default_rng(912).integers(0, 2, size=(80, 3))
    tester = CIT(data, method="chisq")
    first, last = (0, 1, ()), (1, 2, ())
    requests = [first, None, last]

    results = tester.batch(requests, return_exceptions=True)

    assert results[0] == tester(*first)
    assert isinstance(results[1], TypeError)
    assert results[2] == tester(*last)
    with pytest.raises(TypeError):
        tester.batch(requests, return_exceptions=False, n_jobs=1)
