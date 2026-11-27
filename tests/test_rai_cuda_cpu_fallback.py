"""CPU replay through CUDA RAI's host callbacks."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from whydra.algorithms.graph_core.cpu_cuda_rai_bridge import RAIStableCudaLearner
from whydra.algorithms.graph_core.independence_tests import (
    CIT,
    _fisher_z_from_corr_reference,
)


def _tester(n_samples=120):
    tester = CIT(np.random.default_rng(271).normal(size=(n_samples, 4)))
    tester.corr_matrix = np.array(
        [
            [1.0, 0.25, -0.3, 0.1],
            [0.25, 1.0, 0.2, -0.15],
            [-0.3, 0.2, 1.0, 0.05],
            [0.1, -0.15, 0.05, 1.0],
        ]
    )
    return tester


def _replay(tester, alpha=0.05):
    # Skip GPU allocation to exercise the replay callback on CPU.
    learner = object.__new__(RAIStableCudaLearner)
    learner.ci_test = tester
    learner.oracle = SimpleNamespace(alpha=alpha)
    learner.stats = {"ci_total": 0, "ci_sync": 0, "gpu_borderline": 0}
    learner.n_ci_tests = 0
    learner.trace = []
    return learner


@pytest.mark.parametrize("conditions", [(), (2,), (2, 3)])
def test_native_cpu_callback_matches_configured_fisher_at_alpha(conditions):
    tester = _tester()
    expected = _fisher_z_from_corr_reference(
        tester.corr_matrix, tester.n_samples, 0, 1, conditions
    )
    tester.decision_alpha = expected
    actual = _replay(tester, alpha=expected)._cpu_pvalue(0, 1, conditions)
    assert actual == expected
    assert actual == tester(0, 1, conditions)
    assert not actual > expected


@pytest.mark.parametrize("rho", [1.0, 1.0 - 1e-13, 1.01])
def test_native_cpu_callback_rejects_singular_conditional_matrices(rho):
    tester = _tester()
    tester.corr_matrix = np.eye(4)
    tester.corr_matrix[2, 3] = tester.corr_matrix[3, 2] = rho
    with pytest.raises(ValueError, match="singular or ill-conditioned"):
        _replay(tester)._cpu_pvalue(0, 1, (2, 3))


def test_native_cpu_callback_rejects_nonpositive_degrees_of_freedom():
    tester = _tester(n_samples=4)
    with pytest.raises(ValueError, match="positive degrees of freedom"):
        _replay(tester)._cpu_pvalue(0, 1, (2,))


def test_native_cpu_callback_preserves_nan_without_claiming_independence():
    tester = _tester()
    tester.corr_matrix[0, 1] = tester.corr_matrix[1, 0] = np.nan
    learner = _replay(tester)
    assert np.isnan(learner._cpu_pvalue(0, 1, ()))
    assert not learner._ci_memo(0, 1, (), None)


def test_native_cpu_callback_respects_conditioning_limit():
    tester = _tester()
    tester.max_cond_set_size = 0
    with pytest.raises(ValueError, match="exceeds max_cond_set_size"):
        _replay(tester)._cpu_pvalue(0, 1, (2,))


def test_native_cpu_callback_honours_custom_cit_override():
    class CustomCIT(CIT):
        def __call__(self, x, y, conditions=None):
            return 0.75

    tester = CustomCIT(np.random.default_rng(10).normal(size=(30, 3)))
    assert _replay(tester)._cpu_pvalue(0, 1, ()) == 0.75


@pytest.mark.parametrize("memo_value", [None, np.nan, "boundary"])
def test_python_replay_rechecks_missing_nan_and_borderline_memo(memo_value):
    tester = _tester()
    expected = _fisher_z_from_corr_reference(tester.corr_matrix, 120, 0, 1, ())
    alpha = expected if memo_value == "boundary" else expected / 2
    tester.decision_alpha = alpha
    value = expected + 1e-8 if memo_value == "boundary" else memo_value
    memo = SimpleNamespace(get=lambda *_args: value)
    learner = _replay(tester, alpha=alpha)
    decision = learner._ci_memo(0, 1, (), memo)
    assert decision == (tester(0, 1, ()) > alpha)
    assert learner.trace == [(0, 1, (), decision)]
    assert learner.stats["ci_sync"] == 1
    assert learner.stats["gpu_borderline"] == int(memo_value == "boundary")
    assert learner.n_ci_tests == 1


def test_cpu_replay_module_import_does_not_require_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['torch'] = None; "
            "from whydra.algorithms.graph_core.cpu_cuda_rai_bridge import RAIStableCudaLearner",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
