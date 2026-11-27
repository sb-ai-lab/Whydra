"""Guard the reproduced ARM64 GESV crash without executing the unsafe path."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from whydra.algorithms.graph_core import ci_batch, linalg
from whydra.algorithms.graph_core import independence_tests as ci


def test_import_and_small_inverse_do_not_query_machine():
    root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
        import numpy as np
        import platform
        def unexpected():
            raise AssertionError('Small inverse must not query the machine')
        platform.machine = unexpected
        from whydra.algorithms.graph_core import linalg
        assert linalg._USE_SCIPY_INVERSE is None
        assert np.array_equal(linalg.fisher_inverse(np.eye(22)), np.eye(22))
        assert linalg._USE_SCIPY_INVERSE is None
    """),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("affected", [False, True])
def test_first_large_inverse_checks_runtime_before_dispatch_and_caches(
    monkeypatch, affected
):
    from scipy import linalg as scipy_linalg

    events = []
    matrix = np.eye(23)
    monkeypatch.setattr(linalg, "_USE_SCIPY_INVERSE", None)

    def detect():
        events.append("detect")
        return affected

    def numpy_inverse(array):
        assert not affected, "Affected large inverse must never enter NumPy"
        events.append("numpy")
        return array

    def scipy_inverse(array, *, check_finite):
        assert affected
        assert check_finite is False
        events.append("scipy")
        return array

    monkeypatch.setattr(linalg, "_affected_numpy_inverse_runtime", detect)
    monkeypatch.setattr(np.linalg, "inv", numpy_inverse)
    monkeypatch.setattr(scipy_linalg, "inv", scipy_inverse)
    assert linalg.fisher_inverse(matrix) is matrix
    assert linalg.fisher_inverse(matrix) is matrix
    assert linalg._USE_SCIPY_INVERSE is affected
    backend = "scipy" if affected else "numpy"
    assert events == ["detect", backend, backend]


@pytest.mark.parametrize("order", [23, 28, 31, 32])
def test_safe_inverse_preserves_input_and_has_small_residual(monkeypatch, order):
    rng = np.random.default_rng(2390 + order)
    raw = rng.normal(size=(4, order, order))
    matrices = raw @ raw.swapaxes(1, 2) / order + np.eye(order)
    original = matrices.copy()
    monkeypatch.setattr(linalg, "_USE_SCIPY_INVERSE", True)

    def unsafe(*_args, **_kwargs):
        pytest.fail("Affected large matrix must not enter NumPy GESV")

    monkeypatch.setattr(np.linalg, "inv", unsafe)
    batch = linalg.fisher_inverse(matrices)
    scalar = np.stack([linalg.fisher_inverse(matrix) for matrix in matrices])
    assert batch.dtype == np.float64
    np.testing.assert_array_equal(batch, scalar)
    np.testing.assert_array_equal(matrices, original)
    np.testing.assert_allclose(
        matrices @ batch,
        np.broadcast_to(np.eye(order), matrices.shape),
        rtol=0,
        atol=3e-14,
    )


def test_small_inverse_keeps_numpy_path_on_affected_runtime(monkeypatch):
    original = np.linalg.inv
    calls = []

    def observed(array):
        calls.append(array.shape)
        return original(array)

    monkeypatch.setattr(linalg, "_USE_SCIPY_INVERSE", True)
    monkeypatch.setattr(np.linalg, "inv", observed)
    np.testing.assert_array_equal(linalg.fisher_inverse(np.eye(22)), np.eye(22))
    assert calls == [(22, 22)]


def test_high_order_scalar_batch_and_alpha_boundary_share_canonical_inverse(
    monkeypatch,
):
    rng = np.random.default_rng(2391)
    data = rng.normal(size=(500, 28))
    data[:, 1:] += 0.25 * data[:, :1]
    corr = np.corrcoef(data.T)
    queries = [
        (x, y, tuple(z for z in range(28) if z not in (x, y)))
        for x, y in ((0, 1), (1, 0), (2, 3), (4, 7))
    ]
    monkeypatch.setattr(linalg, "_USE_SCIPY_INVERSE", True)

    def unsafe(*_args, **_kwargs):
        pytest.fail("High-order Fisher must use the shared safe inverse")

    monkeypatch.setattr(np.linalg, "inv", unsafe)
    expected = [ci.fisher_z_from_corr(corr, len(data), *query) for query in queries]
    for alpha in (
        expected[0],
        np.nextafter(expected[0], 0.0),
        np.nextafter(expected[0], 1.0),
    ):
        actual = ci_batch.fisher_z_batch_from_corr(
            corr, len(data), queries, decision_alpha=alpha
        )
        scalar = [
            ci.fisher_z_from_corr(corr, len(data), *query, decision_alpha=alpha)
            for query in queries
        ]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual, scalar)
        assert [(value > alpha, value >= alpha) for value in actual] == [
            (value > alpha, value >= alpha) for value in expected
        ]


def test_all_singular_batch_never_calls_any_inverse(monkeypatch):
    corr = np.eye(28)
    corr[0, 1] = corr[1, 0] = 1.0
    query = (0, 1, tuple(range(2, 28)))

    def unexpected(*_args, **_kwargs):
        pytest.fail("Singular matrices must be rejected before inversion")

    monkeypatch.setattr(ci_batch, "fisher_inverse", unexpected)
    monkeypatch.setattr(ci, "fisher_inverse", unexpected)
    with pytest.raises(ValueError, match="singular"):
        ci.fisher_z_from_corr(corr, 100, *query)
    results = ci_batch.fisher_z_batch_from_corr(
        corr, 100, [query, query], return_exceptions=True
    )
    assert all(
        isinstance(value, ValueError) and "singular" in str(value) for value in results
    )


def test_safe_inverse_handles_empty_large_batch(monkeypatch):
    monkeypatch.setattr(linalg, "_USE_SCIPY_INVERSE", True)
    result = linalg.fisher_inverse(np.empty((0, 28, 28)))
    assert result.shape == (0, 28, 28)
    assert result.dtype == np.float64


def test_safe_integer_stack_promotes_to_float_instead_of_truncating(monkeypatch):
    monkeypatch.setattr(linalg, "_USE_SCIPY_INVERSE", True)
    matrices = np.stack([np.eye(28, dtype=int) * 2, np.eye(28, dtype=int) * 4])
    result = linalg.fisher_inverse(matrices)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(
        result, np.stack([np.eye(28) * 0.5, np.eye(28) * 0.25])
    )
    np.testing.assert_array_equal(result[0], linalg.fisher_inverse(matrices[0]))


@pytest.mark.parametrize("dtype", [np.float16, object])
def test_safe_inverse_preserves_numpy_unsupported_dtype_errors(monkeypatch, dtype):
    monkeypatch.setattr(linalg, "_USE_SCIPY_INVERSE", True)
    with pytest.raises(TypeError):
        linalg.fisher_inverse(np.eye(28, dtype=dtype))


@pytest.mark.parametrize(
    ("system", "machine", "version", "blas_version", "expected"),
    [
        ("win32", "ARM64", "2.4.2", "0.3.31.dev", True),
        ("linux", "aarch64", "2.4.2", "0.3.31.dev", False),
        ("win32", "AMD64", "2.4.2", "0.3.31.dev", False),
        ("win32", "ARM64", "2.4.3", "0.3.31.dev", False),
        ("win32", "ARM64", "2.4.2", "0.3.32", False),
    ],
)
def test_runtime_workaround_is_narrowly_scoped(
    monkeypatch, system, machine, version, blas_version, expected
):
    monkeypatch.setattr(linalg.sys, "platform", system)
    monkeypatch.setattr(linalg.platform, "machine", lambda: machine)
    monkeypatch.setattr(np, "__version__", version)
    monkeypatch.setattr(
        np.__config__,
        "CONFIG",
        {
            "Build Dependencies": {
                "blas": {"name": "scipy-openblas", "version": blas_version}
            },
        },
    )
    assert linalg._affected_numpy_inverse_runtime() is expected
