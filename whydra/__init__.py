from .data.case_repository import BenchmarkCaseRepository
from .algorithms import *
from .algorithms import __all__ as _algorithm_exports
from .algorithms.wrappers import run_standalone_algorithm

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("causal-graph-lib")
except PackageNotFoundError:
    __version__ = "unknown"


__all__ = [
    *_algorithm_exports,
    "BenchmarkCaseRepository",
    "run_standalone_algorithm",
]