"""Optional CPython and CUDA accelerators; neither compiler is required to install.

Torch is not in the isolated PEP 517 build; compile the CUDA kernels with the
environment that already has CUDA PyTorch and nvcc::

    pip install -e . --no-build-isolation
    python setup.py build_ext --inplace

``WHYDRA_BUILD_CUDA=0`` skips the CUDA compile. ``WHYDRA_BUILD_CUDA=1`` fails
if torch or nvcc is missing. ``WHYDRA_ENABLE_FLOAT32=1`` compiles Fisher-Z /
fused-CI float32 specializations (off by default). The plain C++ extension is
controlled by ``WHYDRA_BUILD_NATIVE`` and is optional in the default mode.
"""

import os
import sys
import warnings

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.errors import CCompilerError, ExecError, ModuleError, PlatformError

_KERNELS = (
    ("whydra.algorithms.graph_core._whydra_fisher_z_ldlt", "fisher_z_ldlt.cu"),
    ("whydra.algorithms.graph_core._whydra_chisq", "chisq.cu"),
    ("whydra.algorithms.graph_core._whydra_graph_batch", "graph_batch.cu"),
    ("whydra.algorithms.graph_core._whydra_meek", "meek.cu"),
    ("whydra.algorithms.graph_core._whydra_pdsep", "pdsep.cu"),
    ("whydra.algorithms.graph_core._whydra_udag2pag", "udag2pag.cu"),
    ("whydra.algorithms.graph_core._whydra_rai_host", "rai_host.cu"),
    ("whydra.algorithms.graph_core._whydra_fci_host", "fci_host.cpp"),
)


class OptionalBuildExt(build_ext):
    """Compiler discovery can fail before Extension(optional=True) is checked."""

    def run(self):
        """Build extensions and fall back to Python when optional compilation fails."""
        try:
            super().run()
        except (CCompilerError, ExecError, PlatformError, ModuleError, OSError) as exc:
            if os.environ.get("WHYDRA_BUILD_NATIVE", "auto").lower() in {
                "1",
                "true",
                "on",
                "required",
            }:
                raise
            warnings.warn(
                f"Optional native extension could not be built: {exc}. "
                "Installing the Python implementation.",
                RuntimeWarning,
                stacklevel=1,
            )
            self.extensions = []
            self.distribution.ext_modules = []


def _force_cuda() -> bool:
    return os.environ.get("WHYDRA_BUILD_CUDA") == "1"


def _skip_cuda() -> bool:
    return os.environ.get("WHYDRA_BUILD_CUDA", "1") == "0"


def _cuda_extensions():
    if _skip_cuda():
        print("WHYDRA_BUILD_CUDA=0: skipping CUDA extensions", file=sys.stderr)
        return [], {}

    try:
        import torch
        from torch.utils.cpp_extension import (
            CUDA_HOME,
            BuildExtension,
            CUDAExtension,
        )
    except ImportError:
        if _force_cuda():
            msg = (
                "WHYDRA_BUILD_CUDA=1 requires torch in this environment "
                "(use pip install -e . --no-build-isolation)"
            )
            raise RuntimeError(msg) from None
        print("torch not found: skipping CUDA extensions", file=sys.stderr)
        return [], {}

    cuda_home = os.environ.get("CUDA_HOME") or CUDA_HOME
    nvcc_name = "nvcc.exe" if sys.platform == "win32" else "nvcc"
    nvcc = os.path.join(cuda_home, "bin", nvcc_name) if cuda_home else ""
    if not nvcc or not os.path.isfile(nvcc):
        if _force_cuda():
            msg = "WHYDRA_BUILD_CUDA=1 requires nvcc; set CUDA_HOME to the toolkit"
            raise RuntimeError(msg)
        print("nvcc not found: skipping CUDA extensions", file=sys.stderr)
        return [], {}

    if torch.cuda.is_available() and "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

    include = ["whydra/algorithms/graph_core"]
    nvcc_flags = ["-O3"]
    if os.environ.get("WHYDRA_ENABLE_FLOAT32") == "1":
        nvcc_flags.append("-DWHYDRA_ENABLE_FLOAT32=1")
    cxx_flags = ["-O3"]
    ext_modules = [
        CUDAExtension(
            name=name,
            sources=[f"whydra/algorithms/graph_core/{src}"],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
            include_dirs=include,
        )
        for name, src in _KERNELS
    ]
    return ext_modules, {"build_ext": BuildExtension}


def _native_extensions():
    enabled = os.environ.get("WHYDRA_BUILD_NATIVE", "auto").lower()
    required = enabled in {"1", "true", "on", "required"}
    if enabled in {"0", "false", "off"}:
        return []
    return [
        Extension(
            "whydra.algorithms.graph_core._fisher_native",
            ["whydra/algorithms/graph_core/_fisher_native.cpp"],
            depends=["whydra/algorithms/graph_core/_pdsep_native.hpp"],
            language="c++",
            libraries=[] if sys.platform == "win32" else ["m"],
            # Keep scalar operation order: contraction/fast-math could change CI
            # decisions at alpha. The extension uses the Python buffer protocol,
            # so neither NumPy headers nor Cython are build dependencies.
            extra_compile_args=(
                ["/std:c++17", "/O2", "/fp:strict"]
                if sys.platform == "win32"
                else ["-std=c++17", "-O3", "-ffp-contract=off"]
            ),
            optional=not required,
        )
    ]


cuda_ext_modules, cuda_cmdclass = _cuda_extensions()
native_ext_modules = _native_extensions()

if cuda_ext_modules:
    ext_modules = cuda_ext_modules + native_ext_modules
    cmdclass = cuda_cmdclass
else:
    ext_modules = native_ext_modules
    cmdclass = {"build_ext": OptionalBuildExt} if native_ext_modules else {}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
