"""Configured CI tests in CUDA RAI."""

import builtins
import sys
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from whydra.algorithms.graph_core import independence_tests as ci
from whydra.algorithms.graph_core.cpu_cuda_rai_bridge import (
    RAIStableCudaLearner,
    _build_gpu_batcher,
)


def _tester(method="fisherz"):
    rng = np.random.default_rng(43)
    data = (
        rng.normal(size=(100, 3))
        if method == "fisherz"
        else rng.integers(0, 3, size=(100, 3))
    )
    return ci.CIT(data, method=method)


def _forbid_gpu_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("torch") or name in ("cuda_fisher_z", "cuda_chisq"):
            raise AssertionError("GPU import before rejecting custom CI semantics")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.mark.parametrize("method", ["fisherz", "chisq"])
@pytest.mark.parametrize("override", ["__call__", "batch", "numeric_method"])
def test_instance_ci_overrides_are_rejected_before_gpu_import(
    method, override, monkeypatch
):
    tester = _tester(method)
    if override == "numeric_method":
        override = "fisher_z" if method == "fisherz" else "chi_square"
    setattr(tester, override, MethodType(lambda self, *args: 0.75, tester))
    _forbid_gpu_imports(monkeypatch)
    with pytest.raises(NotImplementedError, match="custom ci_test"):
        _build_gpu_batcher(tester)


@pytest.mark.parametrize("method", ["fisherz", "chisq"])
@pytest.mark.parametrize("override", ["__call__", "numeric_method", "kernel"])
def test_class_and_kernel_overrides_are_rejected(method, override, monkeypatch):
    tester = _tester(method)
    if override == "kernel":
        name = (
            "fisher_z_from_corr" if method == "fisherz" else "chi_square_from_encoded"
        )
        monkeypatch.setattr(ci, name, lambda *args, **kwargs: 0.75)
    else:
        if override == "numeric_method":
            override = "fisher_z" if method == "fisherz" else "chi_square"
        monkeypatch.setattr(ci.CIT, override, lambda *args, **kwargs: 0.75)
    _forbid_gpu_imports(monkeypatch)
    with pytest.raises(NotImplementedError, match="custom ci_test"):
        _build_gpu_batcher(tester)


@pytest.mark.parametrize(
    "kind", ["callable", "labelled_callable", "subclass", "inherited_subclass"]
)
def test_constructor_rejects_custom_ci_before_gpu_import(kind, monkeypatch):
    tester = _tester()
    if kind in ("subclass", "inherited_subclass"):

        class CustomCIT(ci.CIT):
            if kind == "subclass":

                def __call__(self, *args, **kwargs):
                    return 0.75

        custom = CustomCIT(tester.data)
    else:

        def custom(*args):
            return 0.75

        if kind == "labelled_callable":
            custom.method = "fisherz"
    _forbid_gpu_imports(monkeypatch)
    with pytest.raises(NotImplementedError, match="custom ci_test"):
        RAIStableCudaLearner(
            tester.data, 0.05, "fisherz", ["A", "B", "C"], ci_test=custom
        )


@pytest.mark.parametrize("precomputed", [False, True])
def test_fisher_batcher_preserves_configured_matrix(precomputed, monkeypatch):
    tester = _tester()
    expected = None
    if precomputed:
        tester.corr_matrix = np.array(
            [[1.0, 0.7, 0.2], [0.7, 1.0, 0.3], [0.2, 0.3, 1.0]]
        )
        expected = tester.corr_matrix
        assert not np.allclose(expected, np.corrcoef(tester.data.T))

    class Batcher:
        def __init__(self, corr, n_samples, dtype=None):
            assert precomputed
            self.corr = corr
            assert n_samples == tester.n_samples
            assert dtype == "float64"

        @classmethod
        def from_data(cls, data, n_samples, dtype=None):
            assert not precomputed
            assert data is tester.data
            assert n_samples == tester.n_samples
            assert dtype == "float64"
            return SimpleNamespace(corr=None)

    monkeypatch.setitem(
        sys.modules,
        "whydra.algorithms.graph_core.cpu_cuda_fisher_bridge",
        SimpleNamespace(CudaFisherZ=Batcher),
    )
    batcher = _build_gpu_batcher(tester, dtype="float64")
    assert batcher.corr is expected
    assert tester._corr_matrix is expected


def test_chisq_batcher_preserves_configured_tester(monkeypatch):
    tester = _tester("chisq")
    monkeypatch.setitem(
        sys.modules,
        "whydra.algorithms.graph_core.cuda_chisq",
        SimpleNamespace(CudaChisqHost=lambda cit: SimpleNamespace(cit=cit)),
    )
    assert _build_gpu_batcher(tester).cit is tester
