"""Touch prebuilt CUDA cubins without running PC-stable or FCI-stable.

Kernels are compiled at install time (``setup.py`` ``CUDAExtension``), not by
``torch.utils.cpp_extension.load``. ``import whydra.algorithms.graph_core._cuda``
``dlopen``s every ``.so``; these helpers additionally call
``cudaFuncGetAttributes`` on every instantiation (``CUDA_MODULE_LOADING=LAZY``
would otherwise defer cubins until first launch).

``preload_pc_cuda_kernels`` / ``preload_fci_cuda_kernels`` import that module
(first call pays ``dlopen`` + cubin touch; later calls are cheap). Graph size
``p`` is a runtime grid; there is no per-``p`` cubin. Empty launches in the
wrappers still return before ``<<<>>>`` and do not load anything extra.
"""

import time

import torch


def _need_cuda() -> None:
    if not torch.cuda.is_available():
        msg = "CUDA is required to preload kernels"
        raise RuntimeError(msg)
    torch.cuda.init()


def _touch(ext, name: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    ext.preload()
    torch.cuda.synchronize()
    return name, time.perf_counter() - t0


def preload_pc_cuda_kernels() -> dict[str, float]:
    """Import prebuilt PC modules and CUDA-context-load every PC kernel.

    Modules: ``_whydra_fisher_z_ldlt``, ``_whydra_chisq``,
    ``_whydra_graph_batch``, ``_whydra_meek``. Importing ``_cuda`` also loads
    FCI modules (``pdsep``, ``udag2pag``) so every symbol is ready.

    Does not run PC-stable. Returns per-module ``nvcc`` (always 0; no runtime
    compile) / ``touch`` / ``total`` seconds plus ``import_ext`` and ``total``
    for the whole call.
    """
    _need_cuda()
    t0 = time.perf_counter()
    from . import _cuda

    out: dict[str, float] = {}
    out["import_ext"] = time.perf_counter() - t0
    for name in ("fisher_z", "chisq", "graph_batch", "meek"):
        _, touch = _touch(getattr(_cuda, name), name)
        out[f"{name}_nvcc"] = 0.0
        out[f"{name}_touch"] = touch
        out[name] = touch
    out["total"] = time.perf_counter() - t0
    return out


def preload_fci_cuda_kernels() -> dict[str, float]:
    """PC preload plus ``_whydra_pdsep`` and ``_whydra_udag2pag``.

    Does not run FCI-stable. FCI reuses the PC skeleton / Fisher-Z / Meek
    cubins, then colliders, qreach, and PAG R1–R10. ``_cuda`` import already
    loaded those modules.
    """
    _need_cuda()
    t0 = time.perf_counter()
    out = preload_pc_cuda_kernels()
    from . import _cuda

    for name in ("pdsep", "udag2pag"):
        _, touch = _touch(getattr(_cuda, name), name)
        out[f"{name}_nvcc"] = 0.0
        out[f"{name}_touch"] = touch
        out[name] = touch
    out["total"] = time.perf_counter() - t0
    return out


__all__ = ["preload_fci_cuda_kernels", "preload_pc_cuda_kernels"]
