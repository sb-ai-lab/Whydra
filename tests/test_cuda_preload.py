"""CUDA cubin preload helpers (no algorithm run)."""

import math

import pytest
import torch

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def test_preload_pc_and_fci_kernels() -> None:
    """Both preload helpers report every extension with finite touch times."""
    from whydra.algorithms.graph_core import _cuda
    from whydra.algorithms.graph_core.cuda_jit import (
        preload_fci_cuda_kernels,
        preload_pc_cuda_kernels,
    )

    pc_modules = ("fisher_z", "chisq", "graph_batch", "meek")
    fci_modules = (*pc_modules, "pdsep", "udag2pag")

    pc = preload_pc_cuda_kernels()
    assert set(pc_modules) <= set(pc)
    for name in pc_modules:
        assert math.isfinite(pc[name]), name
        assert pc[name] >= 0.0, name
        assert pc[f"{name}_touch"] == pc[name], name
        extension = getattr(_cuda, name)
        assert callable(extension.preload), name
    assert math.isfinite(pc["import_ext"])
    assert math.isfinite(pc["total"])

    fci = preload_fci_cuda_kernels()
    assert set(fci_modules) <= set(fci)
    for name in ("pdsep", "udag2pag"):
        assert math.isfinite(fci[name]), name
        assert fci[name] >= 0.0, name
        assert fci[f"{name}_touch"] == fci[name], name
        extension = getattr(_cuda, name)
        assert callable(extension.preload), name
    assert math.isfinite(fci["total"])
