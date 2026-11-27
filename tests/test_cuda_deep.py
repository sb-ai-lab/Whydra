"""Run the deep CUDA script suites as slow pytest items.

The files in ``DEEP_SUITES`` contain exhaustive kernel/reference sweeps and a
``main()`` returning a process status. They are kept as scripts so they can be
run directly, and collected here so ``pytest -m cuda`` exercises the full CUDA
surface for line coverage.

Use ``-m "cuda and not slow"`` for the fast parity gate.
"""

import contextlib
import importlib
import io

import pytest
import torch

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

DEEP_SUITES = (
    "test_cuda_fisher_z",
    "test_cuda_meek_torch",
    "test_pc_stable_cuda",
    "test_cuda_high_k",
    "test_cuda_chisq_k0",
    "test_fci_host_helpers",
    "test_pc_stable_cuda_skeleton",
    "test_fci_stable_cuda",
    "test_fci_stable_cuda_skeleton",
    "test_rai_stable_cuda",
    "test_pdsep_cuda",
    "test_pc_fci_chisq_cuda",
    "test_pag_r1_cuda",
    "test_pag_r2_cuda",
    "test_pag_r3_cuda",
    "test_pag_r4_r5_r6_cuda",
    "test_pag_r7_r8_r9_r10_cuda",
    "test_pag_r9_cuda",
    "test_cuda_graph_batch",
)


@pytest.mark.parametrize("name", DEEP_SUITES)
def test_deep_cuda_suite(name: str) -> None:
    """Run one deep CUDA script suite and require a clean status."""
    module = importlib.import_module(f"tests.{name}")
    with contextlib.redirect_stdout(io.StringIO()):
        status = module.main()
    assert status == 0, name
