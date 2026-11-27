"""Legacy process workers must retain the FCI >= decision at exact alpha."""

import numpy as np
import pytest

from whydra.algorithms.graph_core.independence_tests import (
    _fisher_z_from_corr_reference,
    fisher_z_from_corr,
)
from whydra.algorithms.standalone.skeleton import _process_fci_direction_fisherz


@pytest.mark.parametrize("stable", [False, True])
def test_legacy_process_direction_keeps_reference_alpha_boundary(stable):
    # Exercise a real floating-point boundary rather than substituting a CI
    # callback. Closed forms may be a few ULPs below the LAPACK p-value.
    for correlation in np.linspace(0.01, 0.4, 80):
        corr = np.array([[1.0, correlation], [correlation, 1.0]])
        alpha = _fisher_z_from_corr_reference(corr, 300, 0, 1, ())
        if fisher_z_from_corr(corr, 300, 0, 1, ()) < alpha:
            break
    else:
        pytest.skip("No lower closed-form rounding witness on this numeric backend")
    adjacency = np.array([[0, 1], [1, 0]])
    removed, separators, maxima, requests = _process_fci_direction_fisherz(
        0, 1, adjacency, corr, 300, 0, alpha, -np.inf, {}, stable=stable
    )
    assert removed == [(0, 1)]
    assert separators == [(0, 1, ())]
    assert maxima == [(0, 1, alpha)]
    assert requests == 1
