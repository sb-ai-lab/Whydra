"""Optional compiler discovery cannot prevent a pure Python installation."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution
from setuptools.errors import CCompilerError, CompileError, LinkError, PlatformError


def _setup(monkeypatch, mode, cuda_mode="0"):
    monkeypatch.setenv("WHYDRA_BUILD_NATIVE", mode)
    monkeypatch.setenv("WHYDRA_BUILD_CUDA", cuda_mode)
    captured = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "setup.py"))
    return captured


@pytest.mark.parametrize(
    ("platform", "compiler"), [("win32", "nvcc.exe"), ("linux", "nvcc")]
)
def test_cuda_build_finds_toolkit_compiler(monkeypatch, tmp_path, platform, compiler):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / compiler).touch()
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch.utils.cpp_extension",
        SimpleNamespace(
            CUDA_HOME=None, BuildExtension=build_ext, CUDAExtension=Extension
        ),
    )
    config = _setup(monkeypatch, "0", cuda_mode="1")
    assert config["ext_modules"]


@pytest.mark.parametrize(
    "error",
    [
        PlatformError("no compiler"),
        CompileError("compile"),
        LinkError("link"),
        OSError("missing tool"),
    ],
)
def test_optional_compiler_failure_retains_python_install(monkeypatch, error):
    config = _setup(monkeypatch, "auto")
    command = config["cmdclass"]["build_ext"](
        Distribution({"ext_modules": config["ext_modules"]})
    )

    def fail(_self):
        raise error

    monkeypatch.setattr(build_ext, "run", fail)
    with pytest.warns(RuntimeWarning, match="Installing the Python implementation"):
        command.run()
    assert command.extensions == command.distribution.ext_modules == []


def test_required_compiler_failure_is_not_silently_ignored(monkeypatch):
    config = _setup(monkeypatch, "required")
    command = config["cmdclass"]["build_ext"](
        Distribution({"ext_modules": config["ext_modules"]})
    )

    def fail(_self):
        raise PlatformError("no compiler")

    monkeypatch.setattr(build_ext, "run", fail)
    with pytest.raises(PlatformError, match="no compiler"):
        command.run()


def test_disabled_native_distribution_is_platform_independent(monkeypatch):
    config = _setup(monkeypatch, "0")
    assert config["ext_modules"] == []
    assert not Distribution({"ext_modules": config["ext_modules"]}).has_ext_modules()


@pytest.mark.parametrize("mode", ["auto", "required"])
def test_real_compiler_discovery_failure_obeys_build_mode(monkeypatch, tmp_path, mode):
    config = _setup(monkeypatch, mode)
    command = config["cmdclass"]["build_ext"](
        Distribution({"ext_modules": config["ext_modules"]})
    )
    command.ensure_finalized()
    command.compiler = "whydra_missing_compiler"
    command.build_lib = str(tmp_path / "lib")
    command.build_temp = str(tmp_path / "temp")
    if mode == "required":
        with pytest.raises(
            (CCompilerError, PlatformError), match="whydra_missing_compiler"
        ):
            command.run()
    else:
        with pytest.warns(RuntimeWarning, match="Installing the Python implementation"):
            command.run()
        assert command.extensions == []
