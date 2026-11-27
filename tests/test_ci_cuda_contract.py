"""CI test compatibility across CUDA algorithms."""

import builtins
import sys
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from whydra.algorithms.graph_core import independence_tests as ci
from whydra.algorithms.graph_core.cpu_cuda_rai_bridge import (
    _build_gpu_batcher,
    _validate_cuda_ci_test,
)
from whydra.algorithms.graph_core.cuda_contracts import validate_cuda_ci_test


def _tester(method):
    rng = np.random.default_rng(43)
    data = (
        rng.normal(size=(100, 3))
        if method == "fisherz"
        else rng.integers(0, 3, size=(100, 3))
    )
    return ci.CIT(data, method=method)


@pytest.fixture
def forbid_cuda_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("torch") or name.split(".")[-1] in {
            "cuda_fisher_z",
            "cuda_chisq",
        }:
            raise AssertionError("CI validation must work without CUDA imports")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.mark.parametrize("method", ["fisherz", "chisq"])
def test_builtin_ci_is_admitted_without_optional_dependencies(
    method, forbid_cuda_imports
):
    assert validate_cuda_ci_test(_tester(method)) == method


@pytest.mark.parametrize("method", ["fisherz", "chisq"])
@pytest.mark.parametrize("override", ["__call__", "batch", "numeric_method"])
def test_instance_overrides_are_rejected(method, override, forbid_cuda_imports):
    tester = _tester(method)
    if override == "numeric_method":
        override = "fisher_z" if method == "fisherz" else "chi_square"
    setattr(tester, override, MethodType(lambda self, *args: 0.75, tester))
    with pytest.raises(NotImplementedError, match="custom ci_test"):
        validate_cuda_ci_test(tester)


@pytest.mark.parametrize("method", ["fisherz", "chisq"])
@pytest.mark.parametrize("override", ["__call__", "batch", "numeric_method", "kernel"])
def test_class_and_kernel_overrides_are_rejected(
    method, override, forbid_cuda_imports, monkeypatch
):
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
    with pytest.raises(NotImplementedError, match="custom ci_test"):
        validate_cuda_ci_test(tester)


@pytest.mark.parametrize("method", ["fisherz", "chisq"])
def test_labelled_callables_and_inherited_subclasses_are_rejected(
    method, forbid_cuda_imports
):
    tester = _tester(method)

    class InheritedCIT(ci.CIT):
        pass

    for custom in (
        SimpleNamespace(method=method),
        InheritedCIT(tester.data, method=method),
    ):
        with pytest.raises(NotImplementedError, match="custom ci_test"):
            validate_cuda_ci_test(custom)


@pytest.mark.parametrize("algorithm", ["PC-stable-cuda", "FCI-stable-cuda"])
def test_error_names_the_backend(algorithm, forbid_cuda_imports):
    with pytest.raises(NotImplementedError, match=f"^{algorithm} requires"):
        validate_cuda_ci_test(SimpleNamespace(method="fisherz"), algorithm)


def test_rai_compatibility_entry_point_preserves_error(forbid_cuda_imports):
    with pytest.raises(NotImplementedError, match=r"^RAI-stable-cuda requires"):
        _validate_cuda_ci_test(SimpleNamespace(method="fisherz"))
    assert _validate_cuda_ci_test(_tester("fisherz")) == "fisherz"


@pytest.mark.parametrize("precomputed", [False, True])
@pytest.mark.parametrize("alpha", [None, 0.05])
def test_rai_batcher_receives_configured_cpu_tester_without_forcing_lazy_matrix(
    precomputed, alpha, monkeypatch
):
    tester = _tester("fisherz")
    tester.decision_alpha = alpha
    if precomputed:
        tester.corr_matrix = np.array(
            [[1.0, 0.7, 0.2], [0.7, 1.0, 0.3], [0.2, 0.3, 1.0]]
        )
    original_matrix = tester._corr_matrix

    class Batcher:
        def __init__(self, corr, n_samples, **kwargs):
            assert precomputed
            assert corr is original_matrix
            assert n_samples == tester.n_samples

        @classmethod
        def from_data(cls, data, **kwargs):
            assert not precomputed
            assert data is tester.data
            return SimpleNamespace()

    monkeypatch.setitem(
        sys.modules,
        "whydra.algorithms.graph_core.cpu_cuda_fisher_bridge",
        SimpleNamespace(CudaFisherZ=Batcher),
    )
    batcher = _build_gpu_batcher(tester)
    assert batcher.cpu_test is tester
    assert batcher.decision_alpha == alpha
    assert tester._corr_matrix is original_matrix
