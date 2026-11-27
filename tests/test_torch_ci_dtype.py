"""Torch-only CUDA driver helpers that do not need a GPU."""

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.torch


def test_resolve_ci_dtype_is_float64_only() -> None:
    """The public CUDA CI dtype resolver accepts float64 only."""
    from whydra.algorithms.graph_core.cuda_fisher_z import resolve_ci_dtype

    assert resolve_ci_dtype(None) is torch.float64
    assert resolve_ci_dtype("float64") is torch.float64
    assert resolve_ci_dtype(torch.float64) is torch.float64
    with pytest.raises(ValueError, match="float64 only"):
        resolve_ci_dtype("float32")
    with pytest.raises(ValueError, match="unsupported ci dtype"):
        resolve_ci_dtype("bogus")
