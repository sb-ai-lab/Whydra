"""CUDA Fisher-Z fallback with a host-only Torch test double."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from whydra.algorithms.graph_core.independence_tests import (
    CIT,
    fisher_z_from_corr,
)


class _Tensor:
    """NumPy test double for tensor transfers."""

    def __init__(self, array):
        self.array = np.asarray(array)
        self.ndim = self.array.ndim

    def to(self, **kwargs):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def reshape(self, *shape):
        return _Tensor(self.array.reshape(*shape))

    def is_contiguous(self):
        return True

    def contiguous(self):
        return self

    def numel(self):
        return self.array.size

    def size(self, axis):
        return self.array.shape[axis]

    def dim(self):
        return self.array.ndim

    def __getitem__(self, key):
        return _Tensor(self.array[key])


@pytest.fixture
def cuda_module(monkeypatch):
    fake_torch = ModuleType("torch")
    fake_torch.Tensor = _Tensor
    fake_torch.dtype = np.dtype
    fake_torch.float64 = np.float64
    fake_torch.int32 = np.int32
    fake_torch.is_tensor = lambda value: isinstance(value, _Tensor)
    fake_torch.as_tensor = lambda value, **kwargs: _Tensor(value)
    fake_torch.empty = lambda shape, **kwargs: _Tensor(np.empty(shape))
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True)
    fake_torch.device = str
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    root = Path(__file__).resolve().parents[1] / "whydra/algorithms/graph_core"
    original_name = "whydra.algorithms.graph_core.cuda_fisher_z"
    original_spec = importlib.util.spec_from_file_location(
        original_name, root / "cuda_fisher_z.py"
    )
    original = importlib.util.module_from_spec(original_spec)
    monkeypatch.setitem(sys.modules, original_name, original)
    original_spec.loader.exec_module(original)
    monkeypatch.setattr(
        original, "_ldlt_cuda", lambda: SimpleNamespace(k_max=lambda: 96)
    )
    name = "whydra.algorithms.graph_core._fisher_fallback_contract_test"
    path = root / "cpu_cuda_fisher_bridge.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    module._test_original = original
    return module


def _batcher(module, corr, *, n_samples=100, k_max=96, tester=None, alpha=None):
    batcher = object.__new__(module._cuda_class())
    batcher._corr_np = np.asarray(corr)
    batcher.n_samples = n_samples
    batcher.eps = 0.0
    batcher.cpu_test = tester
    batcher.decision_alpha = alpha
    batcher._cpu_impl = None
    batcher._k_max = k_max
    batcher.device = "cuda"
    batcher.dtype = np.float64
    batcher.corr = _Tensor(corr)
    return batcher


def _tester(corr, *, limit=None):
    tester = CIT(
        np.random.default_rng(73).normal(size=(100, len(corr))),
        max_cond_set_size=limit,
    )
    tester.corr_matrix = corr
    return tester


def test_bridge_import_does_not_require_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['torch'] = None; "
            "from whydra.algorithms.graph_core.cpu_cuda_fisher_bridge import CudaFisherZ; "
            "assert 'whydra.algorithms.graph_core.cuda_fisher_z' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_facade_lazily_inherits_original_constructor_and_kernel_methods(cuda_module):
    batcher = cuda_module.CudaFisherZ(np.eye(2), 100)
    original = cuda_module._test_original.CudaFisherZ
    assert isinstance(batcher, cuda_module.CudaFisherZ)
    assert isinstance(batcher, original)
    assert type(batcher).__init__ is original.__init__
    assert type(batcher).batch_r is original.batch_r
    assert batcher.n_samples == 100


def test_from_data_keeps_the_bound_class(cuda_module, monkeypatch):
    data = np.ones((25, 3))
    calls = []

    def corrcoef(array, **kwargs):
        calls.append(array)
        return _Tensor(np.eye(3))

    monkeypatch.setattr(cuda_module._test_original, "corrcoef_from_data", corrcoef)
    batcher = cuda_module.CudaFisherZ.from_data(data)
    assert isinstance(batcher, cuda_module.CudaFisherZ)
    assert batcher.n_samples == 25
    assert calls == [data]


def test_matrix_fallback_receives_decision_alpha(cuda_module):
    rho = 0.062379379379379374
    corr = np.array([[1.0, rho], [rho, 1.0]])
    alpha = fisher_z_from_corr(corr, 100, 0, 1, ())
    expected = fisher_z_from_corr(corr, 100, 0, 1, (), decision_alpha=alpha)
    batcher = _batcher(cuda_module, corr, alpha=alpha)
    result = batcher._cpu.batch([0], [1], np.empty((1, 0), int))[0]
    assert result == expected
    assert (result > alpha) == (expected > alpha)
    assert (result >= alpha) == (expected >= alpha)


def test_matrix_fallback_propagates_alpha_without_rounding_assumptions(
    cuda_module, monkeypatch
):
    from whydra.algorithms.graph_core import independence_tests

    seen = []

    def alpha_sensitive(corr, n, x, y, conditions, *, decision_alpha=None):
        seen.append(decision_alpha)
        return decision_alpha if decision_alpha is not None else 0.0

    monkeypatch.setattr(independence_tests, "fisher_z_from_corr", alpha_sensitive)
    batcher = _batcher(cuda_module, np.eye(2), alpha=0.125)
    assert batcher._cpu.batch([0], [1], np.empty((1, 0), int))[0] == 0.125
    batcher.decision_alpha = 0.25
    assert batcher._cpu.batch([0], [1], np.empty((1, 0), int))[0] == 0.25
    assert seen == [0.125, 0.25]


def test_attached_cpu_tester_supplies_its_matrix_and_shared_cache(cuda_module):
    corr = np.array([[1.0, 0.2], [0.2, 1.0]])
    tester = _tester(corr)
    # The matrix mismatch detects accidental reuse of the GPU matrix.
    batcher = _batcher(cuda_module, np.eye(2), tester=tester, alpha=0.05)
    result = batcher._cpu.batch([0, 1], [1, 0], np.empty((2, 0), int))
    assert result[0] == result[1] == tester(0, 1, ())
    assert tester.cache_misses == 1
    assert tester.cache_hits == 2
    assert tester.decision_alpha == 0.05


def test_attached_cpu_tester_retains_conditioning_limit(cuda_module):
    tester = _tester(np.eye(3), limit=0)
    batcher = _batcher(cuda_module, np.eye(3), tester=tester)
    with pytest.raises(ValueError, match="exceeds max_cond_set_size"):
        batcher._cpu.batch([0], [1], [[2]])


def test_cpu_fallback_preserves_singular_input_error(cuda_module):
    corr = np.array([[1.0, 1.0], [1.0, 1.0]])
    batcher = _batcher(cuda_module, np.eye(2), tester=_tester(corr))
    with pytest.raises(ValueError, match="singular or ill-conditioned"):
        batcher._cpu.batch([0], [1], np.empty((1, 0), int))


@pytest.mark.parametrize("method", ["batch", "batch_pvalues"])
def test_host_fallback_batch_preserves_strict_boundary(cuda_module, method):
    corr = np.array([[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tester = _tester(corr)
    alpha = fisher_z_from_corr(corr, 100, 0, 1, (2,))
    batcher = _batcher(cuda_module, corr, k_max=0, tester=tester, alpha=alpha)
    result = getattr(batcher, method)([0], [1], [[2]])[0]
    assert result == tester(0, 1, (2,))
    assert not result > alpha
    assert result >= alpha


@pytest.mark.parametrize("method", ["batch", "batch_pvalues"])
def test_nonpositive_dof_fails_before_kernel(cuda_module, method):
    batcher = _batcher(cuda_module, np.eye(3), n_samples=4)
    # No _ldlt object: reaching a kernel would be an AttributeError.
    with pytest.raises(ValueError, match="positive degrees of freedom"):
        getattr(batcher, method)([0], [1], [[2]])


def test_high_k_small_batch_broadcasts_scalar_pair_on_cpu(cuda_module):
    tester = _tester(np.eye(4))
    batcher = _batcher(cuda_module, np.eye(4), k_max=0, tester=tester)
    actual = batcher.batch_pvalues(0, 1, np.array([[2], [3]]))
    np.testing.assert_array_equal(actual, [tester(0, 1, (2,)), tester(0, 1, (3,))])


@pytest.mark.parametrize("method", ["batch", "batch_pvalues"])
def test_inherited_kernel_uses_stable_host_tail(cuda_module, method):
    corr = np.array([[1.0, 0.8], [0.8, 1.0]])
    batcher = _batcher(cuda_module, corr)

    def kernel(correlation, x, y, conditions, output, eps):
        output.array[:] = 0.8

    batcher._ldlt = SimpleNamespace(ldlt_batch=kernel)
    actual = getattr(batcher, method)([0], [1], np.empty((1, 0), int))[0]
    expected = fisher_z_from_corr(corr, 100, 0, 1, ())
    assert actual == pytest.approx(expected, rel=1e-12, abs=0.0)
    assert actual > 1e-30


def test_explicit_ridge_keeps_experimental_model(cuda_module):
    tester = _tester(np.array([[1.0, 1.0], [1.0, 1.0]]))
    batcher = _batcher(cuda_module, np.eye(2), tester=tester, alpha=0.05)
    batcher.eps = 1e-4
    # CIT has no ridge term.
    assert batcher._cpu.method == "schur"
    assert batcher._cpu.batch([0], [1], np.empty((1, 0), int))[0] == 1.0
