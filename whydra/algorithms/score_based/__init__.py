"""Score-based exports without loading GES for a local-score calculation."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ges import ges_search
    from .scores import LocalScoreCache, local_parent_score

__all__ = ["LocalScoreCache", "ges_search", "local_parent_score"]


def __getattr__(name):
    if name not in __all__:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module_name = ".ges" if name == "ges_search" else ".scores"
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
