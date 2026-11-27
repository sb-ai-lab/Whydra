"""CUDA test collection and required-runtime checks."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_validation_suite(*, torch_available, cuda_available, required=False):
    source = textwrap.dedent(
        f"""
        import importlib.abc
        import importlib.machinery
        import sys
        import types

        if {torch_available!r}:
            torch = types.ModuleType('torch')
            torch.__spec__ = importlib.machinery.ModuleSpec('torch', loader=None)
            torch.cuda = types.SimpleNamespace(is_available=lambda: {cuda_available!r})
            sys.modules['torch'] = torch
        else:
            sys.modules['torch'] = None

        class MissingExtensions(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith('whydra.algorithms.graph_core._whydra_'):
                    raise ModuleNotFoundError('deliberately missing CUDA extension: ' + fullname)

        sys.meta_path.insert(0, MissingExtensions())
        import pytest
        args = ['-q', '-o', 'addopts=', '-p', 'no:cacheprovider',
                'tests/test_cuda_validation.py']
        if {required!r}:
            args.append('--require-cuda')
        raise SystemExit(pytest.main(args))
        """
    )
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=str(ROOT), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),
        capture_output=True,
        text=True,
        timeout=45,
    )


def test_cpu_torch_does_not_import_missing_cuda_extensions():
    result = _run_validation_suite(torch_available=True, cuda_available=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "6 skipped" in result.stdout


def test_available_cuda_does_not_hide_missing_extensions():
    result = _run_validation_suite(torch_available=True, cuda_available=True)
    assert result.returncode != 0
    assert "whydra CUDA kernels are not built" in result.stdout + result.stderr


@pytest.mark.parametrize(
    ("torch_available", "cuda_available", "message"),
    [
        (False, False, "--require-cuda requires PyTorch"),
        (True, False, "--require-cuda requires an available CUDA device"),
        (True, True, "--require-cuda could not load CUDA extensions"),
    ],
)
def test_required_cuda_fails_for_unavailable_runtime(
    torch_available, cuda_available, message
):
    result = _run_validation_suite(
        torch_available=torch_available, cuda_available=cuda_available, required=True
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert message in result.stdout + result.stderr
