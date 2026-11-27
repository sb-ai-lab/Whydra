"""Algorithm exports without importing every search implementation."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..background_knowledge import (
        BackgroundKnowledge,
        BKApplicationReport,
        BKConflictWarning,
        BKEvent,
        BKPhase,
        MatrixEncoding,
    )
    from .base import (
        CausalAlgorithm,
    )
    from .temporal import (
        PCMCI,
        JPCMCIPlus,
        PCMCIPlus,
        StandaloneJPCMCIPlus,
        StandalonePCMCI,
        StandalonePCMCIPlus,
    )
    from .wrappers import (
        DagGNN,
        ParallelPCStable,
        StandaloneCFCIStable,
        StandaloneCFCIStableParallelCached,
        StandaloneFCIPlusStable,
        StandaloneFCIPlusStableParallelCached,
        StandaloneFCIStable,
        StandaloneFCIStableCached,
        StandaloneFCIStableParallelCached,
        StandaloneGFCIStable,
        StandalonePCStable,
        StandaloneRAIStable,
        StandaloneRFCIStable,
        available_standalone_algorithms,
        create_standalone_algorithm,
        run_standalone_algorithm,
    )

_TEMPORAL_EXPORTS = (
    "PCMCI",
    "PCMCIPlus",
    "JPCMCIPlus",
    "StandalonePCMCI",
    "StandalonePCMCIPlus",
    "StandaloneJPCMCIPlus",
)

__all__ = [
    "PCMCI",
    "BKApplicationReport",
    "BKConflictWarning",
    "BKEvent",
    "BKPhase",
    "BackgroundKnowledge",
    "CausalAlgorithm",
    "DagGNN",
    "JPCMCIPlus",
    "MatrixEncoding",
    "PCMCIPlus",
    "ParallelPCStable",
    "StandaloneCFCIStable",
    "StandaloneCFCIStableParallelCached",
    "StandaloneFCIPlusStable",
    "StandaloneFCIPlusStableParallelCached",
    "StandaloneFCIStable",
    "StandaloneFCIStableCached",
    "StandaloneFCIStableParallelCached",
    "StandaloneGFCIStable",
    "StandaloneJPCMCIPlus",
    "StandalonePCMCI",
    "StandalonePCMCIPlus",
    "StandalonePCStable",
    "StandaloneRAIStable",
    "StandaloneRFCIStable",
    "available_standalone_algorithms",
    "create_standalone_algorithm",
    "run_standalone_algorithm",
]

_BACKGROUND_EXPORTS = frozenset(
    {
        "BKApplicationReport",
        "BKConflictWarning",
        "BKEvent",
        "BKPhase",
        "BackgroundKnowledge",
        "MatrixEncoding",
    }
)


def __getattr__(name):
    if name not in __all__:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module = (
        "..background_knowledge"
        if name in _BACKGROUND_EXPORTS
        else ".base"
        if name == "CausalAlgorithm"
        else ".temporal"
        if name in _TEMPORAL_EXPORTS
        else ".wrappers"
    )
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
