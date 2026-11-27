"""Public graph and CI objects, loaded only when requested.

Importing a numerical worker submodule must not initialize every graph class
or conditional independence test in each freshly spawned interpreter.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .causal_graph import CausalGraph
    from .edges import Edge
    from .endpoints import Endpoint
    from .general_graph import GeneralGraph
    from .independence_tests import (
        CIT,
        CONTINUOUS_TESTS,
        DISCRETE_TESTS,
        IMPLEMENTED_TESTS,
        DegenerateColumnWarning,
        fisher_z_from_corr,
        fisher_z_from_corr_batch,
        is_discrete_test,
        resolve_score_func,
    )
    from .nodes import Node

__all__ = [
    "CIT",
    "CONTINUOUS_TESTS",
    "DISCRETE_TESTS",
    "IMPLEMENTED_TESTS",
    "CausalGraph",
    "DegenerateColumnWarning",
    "Edge",
    "Endpoint",
    "GeneralGraph",
    "Node",
    "fisher_z_from_corr",
    "fisher_z_from_corr_batch",
    "is_discrete_test",
    "resolve_score_func",
]

_GRAPH_MODULES = {
    "Endpoint": ".endpoints",
    "Node": ".nodes",
    "Edge": ".edges",
    "GeneralGraph": ".general_graph",
    "CausalGraph": ".causal_graph",
}


def __getattr__(name):
    if name not in __all__:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module_name = _GRAPH_MODULES.get(name, ".independence_tests")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
