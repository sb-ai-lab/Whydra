"""Public API, loaded on demand so numerical workers only import their kernel."""

from importlib import import_module
from typing import TYPE_CHECKING

from . import algorithms

if TYPE_CHECKING:
    from .algorithms import PCMCI as PCMCI
    from .algorithms import BackgroundKnowledge as BackgroundKnowledge
    from .algorithms import BKApplicationReport as BKApplicationReport
    from .algorithms import BKConflictWarning as BKConflictWarning
    from .algorithms import BKEvent as BKEvent
    from .algorithms import BKPhase as BKPhase
    from .algorithms import CausalAlgorithm as CausalAlgorithm
    from .algorithms import DagGNN as DagGNN
    from .algorithms import JPCMCIPlus as JPCMCIPlus
    from .algorithms import MatrixEncoding as MatrixEncoding
    from .algorithms import ParallelPCStable as ParallelPCStable
    from .algorithms import PCMCIPlus as PCMCIPlus
    from .algorithms import StandaloneCFCIStable as StandaloneCFCIStable
    from .algorithms import (
        StandaloneCFCIStableParallelCached as StandaloneCFCIStableParallelCached,
    )
    from .algorithms import StandaloneFCIPlusStable as StandaloneFCIPlusStable
    from .algorithms import (
        StandaloneFCIPlusStableParallelCached as StandaloneFCIPlusStableParallelCached,
    )
    from .algorithms import StandaloneFCIStable as StandaloneFCIStable
    from .algorithms import StandaloneFCIStableCached as StandaloneFCIStableCached
    from .algorithms import (
        StandaloneFCIStableParallelCached as StandaloneFCIStableParallelCached,
    )
    from .algorithms import StandaloneGFCIStable as StandaloneGFCIStable
    from .algorithms import StandaloneJPCMCIPlus as StandaloneJPCMCIPlus
    from .algorithms import StandalonePCMCI as StandalonePCMCI
    from .algorithms import StandalonePCMCIPlus as StandalonePCMCIPlus
    from .algorithms import StandalonePCStable as StandalonePCStable
    from .algorithms import StandaloneRAIStable as StandaloneRAIStable
    from .algorithms import StandaloneRFCIStable as StandaloneRFCIStable
    from .algorithms import (
        available_standalone_algorithms as available_standalone_algorithms,
    )
    from .algorithms import create_standalone_algorithm as create_standalone_algorithm
    from .algorithms import run_standalone_algorithm as run_standalone_algorithm
    from .data.case_repository import BenchmarkCaseRepository


__all__ = []
__all__.extend(algorithms.__all__)
__all__ += [
    "BenchmarkCaseRepository",
    "run_standalone_algorithm",
]

_EXPORTS = {name: (".algorithms", name) for name in algorithms.__all__}
_EXPORTS["BenchmarkCaseRepository"] = (
    ".data.case_repository",
    "BenchmarkCaseRepository",
)


def __getattr__(name):
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            value = version("whydra")
        except PackageNotFoundError:
            from importlib.resources import files

            source = files(__name__).joinpath("VERSION").read_text(encoding="ascii")
            value = f"{source.strip()}+source"
    else:
        target = _EXPORTS.get(name)
        if target is None:
            msg = f"module {__name__!r} has no attribute {name!r}"
            raise AttributeError(msg)
        value = getattr(import_module(target[0], __name__), target[1])
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | {"__version__"})
