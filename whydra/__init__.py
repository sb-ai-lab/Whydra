"""Whydra: fast causal discovery library."""

from .data.case_repository import BenchmarkCaseRepository
from .algorithms import *
from .algorithms import __all__ as _algorithm_exports
from .algorithms.wrappers import run_standalone_algorithm

__version__ = "0.1.1"

__all__ = [
    *_algorithm_exports,
    "BenchmarkCaseRepository",
    "run_standalone_algorithm",
    "__version__",
]
