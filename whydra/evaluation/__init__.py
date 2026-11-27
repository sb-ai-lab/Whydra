"""Evaluation helpers, imported only when requested."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph_utils import (
        adj_matrix_to_graph,
        convert_general_graph_to_nx,
        draw_graph,
        get_adj_matrix,
        load_ground_truth,
        pag_matrix_to_general_graph,
        parse_ground_truth_graph,
        result_to_nx,
    )
    from .metrics import (
        calculate_metrics,
        compute_metrics_from_dag_and_result,
        compute_metrics_from_endpoint_matrices,
        convert_pag_to_cpdag_proxy,
        dag_adj_to_cpdag_endpoint_matrix,
        dag_adj_to_general_graph,
        f1_score,
        is_pag_result,
        result_to_endpoint_matrix,
    )


_METRICS = (
    "result_to_endpoint_matrix",
    "is_pag_result",
    "dag_adj_to_cpdag_endpoint_matrix",
    "dag_adj_to_general_graph",
    "convert_pag_to_cpdag_proxy",
    "f1_score",
    "compute_metrics_from_endpoint_matrices",
    "compute_metrics_from_dag_and_result",
    "calculate_metrics",
)
__all__ = [
    "adj_matrix_to_graph",
    "calculate_metrics",
    "compute_metrics_from_dag_and_result",
    "compute_metrics_from_endpoint_matrices",
    "convert_general_graph_to_nx",
    "convert_pag_to_cpdag_proxy",
    "dag_adj_to_cpdag_endpoint_matrix",
    "dag_adj_to_general_graph",
    "draw_graph",
    "f1_score",
    "get_adj_matrix",
    "is_pag_result",
    "load_ground_truth",
    "pag_matrix_to_general_graph",
    "parse_ground_truth_graph",
    "result_to_endpoint_matrix",
    "result_to_nx",
]


def __getattr__(name):
    if name not in __all__:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module = ".metrics" if name in _METRICS else ".graph_utils"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
