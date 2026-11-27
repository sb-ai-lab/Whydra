"""Prebuilt whydra CUDA kernels.

Importing this module ``dlopen``s every compiled extension. Call
:func:`preload` to touch every kernel specialization into the current GPU
context; no runtime nvcc and no ``torch.utils.cpp_extension.load`` are used.

Submodules (same pybind surface as the previous per-file modules)::

    fisher_z, chisq, graph_batch, meek, pdsep, udag2pag, fci_host
"""

import importlib
import os

# Only takes effect if CUDA has not been initialized yet.
os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")

_BUILD_HINT = (
    "whydra CUDA kernels are not built. From the repository root, with CUDA "
    "toolkit and a CUDA PyTorch install:\n"
    "  pip install -e . --no-build-isolation\n"
    "or:\n"
    "  python setup.py build_ext --inplace\n"
    "Set WHYDRA_BUILD_CUDA=0 to skip compiling kernels at install time."
)


def _import_ext(modname: str):
    try:
        return importlib.import_module(f".{modname}", __package__)
    except ModuleNotFoundError as exc:
        raise ImportError(_BUILD_HINT) from exc


fisher_z = _import_ext("_whydra_fisher_z_ldlt")
chisq = _import_ext("_whydra_chisq")
graph_batch = _import_ext("_whydra_graph_batch")
meek = _import_ext("_whydra_meek")
pdsep = _import_ext("_whydra_pdsep")
udag2pag = _import_ext("_whydra_udag2pag")
fci_host = _import_ext("_whydra_fci_host")

_MODULES = {
    "fisher_z": fisher_z,
    "chisq": chisq,
    "graph_batch": graph_batch,
    "meek": meek,
    "pdsep": pdsep,
    "udag2pag": udag2pag,
    "fci_host": fci_host,
}


def preload() -> None:
    """Load every cubin into the GPU context without launching kernels."""
    import torch

    if not torch.cuda.is_available():
        msg = "CUDA is required to load whydra kernels"
        raise RuntimeError(msg)
    torch.cuda.init()
    for ext in _MODULES.values():
        ext.preload()
    torch.cuda.synchronize()


__all__ = [
    "chisq",
    "fisher_z",
    "graph_batch",
    "meek",
    "pdsep",
    "preload",
    "udag2pag",
]
