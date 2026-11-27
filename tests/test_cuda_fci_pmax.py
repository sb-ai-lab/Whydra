"""FCI CPU/CUDA graph parity and the documented pMax tolerance."""

import warnings

import numpy as np
import pytest
import torch

from whydra.algorithms.graph_core import CIT, CausalGraph
from whydra.algorithms.standalone.fci_algo_v2 import fci_stable_v2

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def test_fci_graph_exact_and_pmax_close() -> None:
    """Graphs are exact; pMax differs only by the documented erfc/cdf ulp."""
    rng = np.random.default_rng(21)
    p = 6
    data = rng.normal(size=(400, p))
    data[:, 2] = data[:, 0] + data[:, 1] + rng.normal(scale=0.4, size=400)
    data[:, 3] = data[:, 2] + rng.normal(scale=0.5, size=400)
    names = [f"X{i}" for i in range(p)]

    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for label, device in (("cpu", None), ("cuda", "cuda")):
            cg = CausalGraph(p, names, device=device)
            cg.set_ind_test(CIT(data, method="fisherz"))
            pag = fci_stable_v2(
                cg, data, alpha=0.05, indep_test="fisherz", n_jobs=1, device=device
            )
            pmax = cg.pMax
            if torch.is_tensor(pmax):
                pmax = pmax.detach().cpu().numpy()
            results[label] = (np.asarray(pag), np.asarray(pmax))

    cpu_graph, cpu_pmax = results["cpu"]
    cuda_graph, cuda_pmax = results["cuda"]
    assert np.array_equal(cpu_graph, cuda_graph)

    assert np.array_equal(np.isfinite(cpu_pmax), np.isfinite(cuda_pmax))
    finite = np.isfinite(cpu_pmax) & np.isfinite(cuda_pmax)
    assert finite.any()
    assert np.max(np.abs(cpu_pmax[finite] - cuda_pmax[finite])) <= 1e-10
