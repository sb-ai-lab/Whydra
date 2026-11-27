"""Convert temporal marks to summary and lag-expanded Whydra graphs."""

import numpy as np

from ...evaluation.graph_utils import pag_matrix_to_general_graph

# PCMCI's o-o is an unoriented CPDAG edge, not a latent-variable PAG edge.
_ENDPOINTS = {
    "-->": (-1, 1),
    "<--": (1, -1),
    "o-o": (-1, -1),
    "---": (-1, -1),
    "x-x": (2, 2),
    "<->": (1, 1),
    "o->": (2, 1),
    "<-o": (1, 2),
}


def lagged_node_names(node_names, tau_max):
    """List current nodes first, then append one block per positive lag."""
    names = list(node_names)
    expanded = names + [
        f"{name}[t-{lag}]" for lag in range(1, tau_max + 1) for name in names
    ]
    if len(set(expanded)) != len(expanded):
        msg = "Variable names collide with generated lagged node names"
        raise ValueError(msg)
    return expanded


def temporal_to_matrix(marks):
    """Endpoint matrix for edges from t-lag to t; includes autoregression."""
    marks = np.asarray(marks)
    n, _, depth = marks.shape
    matrix = np.zeros((n * depth, n * depth), dtype=int)
    for i, j, lag in np.ndindex(marks.shape):
        mark = str(marks[i, j, lag]).replace("?", "-")
        if not mark or (lag == 0 and i >= j):
            continue
        source, target = lag * n + i, j
        at_source, at_target = _ENDPOINTS[mark]
        matrix[target, source], matrix[source, target] = at_source, at_target
    return matrix


def matrix_to_temporal(matrix, n, tau_max):
    """Convert a Whydra endpoint matrix back to PCMCI CPDAG marks."""
    lookup = {
        (-1, 1): "-->",
        (1, -1): "<--",
        (-1, -1): "o-o",
        (2, 2): "x-x",
        (1, 1): "<->",
        (2, 1): "o->",
        (1, 2): "<-o",
        (0, 0): "",
    }
    marks = np.full((n, n, tau_max + 1), "", dtype="<U3")
    for i, j, lag in np.ndindex(marks.shape):
        if lag == 0 and i == j:
            continue
        source = lag * n + i
        marks[i, j, lag] = lookup[int(matrix[j, source]), int(matrix[source, j])]
    return marks


def temporal_to_general_graph(marks, node_names, *, graph_mode="summary"):
    """Build a summary or lag-expanded GeneralGraph using native endpoint codes.

    Summary endpoints agree only when every lag supports the same mark.
    Disagreement becomes a circle, including opposite temporal directions;
    it must not be misrepresented as evidence of latent confounding.
    Self-lags exist only in the expanded graph and the original mark tensor.
    """
    marks = np.asarray(marks)
    n = len(node_names)
    if marks.ndim != 3 or marks.shape[:2] != (n, n):
        msg = "Expected temporal marks of shape (n_features, n_features, tau_max + 1)"
        raise ValueError(msg)
    if graph_mode == "lagged":
        return pag_matrix_to_general_graph(
            temporal_to_matrix(marks), lagged_node_names(node_names, marks.shape[2] - 1)
        )
    if graph_mode != "summary":
        msg = "graph_mode must be 'summary' or 'lagged'"
        raise ValueError(msg)
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            endpoints = []
            for lag in range(marks.shape[2]):
                if marks[i, j, lag]:
                    endpoints.append(_ENDPOINTS[str(marks[i, j, lag])])
                if lag and marks[j, i, lag]:
                    endpoints.append(_ENDPOINTS[str(marks[j, i, lag])][::-1])
            if endpoints:
                at_i, at_j = zip(*endpoints, strict=False)
                matrix[j, i] = at_i[0] if len(set(at_i)) == 1 else 2
                matrix[i, j] = at_j[0] if len(set(at_j)) == 1 else 2
    return pag_matrix_to_general_graph(matrix, list(node_names))
