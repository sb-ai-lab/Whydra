"""
Metrics and conversions for causal graphs.

This module defines the shared routines for:
  - Converting an algorithm result to an endpoint matrix.
  - Converting a ground-truth DAG to a CPDAG.
  - Converting a PAG to a CPDAG proxy.
  - Computing SHD, adjacency, arrowhead, and shared-edge metrics.
"""

import warnings
from typing import cast

import numpy as np

from whydra.algorithms.graph_core import Edge, Endpoint, GeneralGraph
from whydra.algorithms.graph_core import Node as GraphNode
from whydra.algorithms.graph_core.validation import (
    validate_endpoint_matrix,
    validate_node_names,
)
from whydra.background_knowledge import MatrixEncoding, endpoint_codes
from whydra.evaluation.graph_utils import default_node_names, get_adj_matrix

# ============================================================
#  ALGORITHM RESULTS TO ENDPOINT MATRICES
# ============================================================


def result_to_endpoint_matrix(
    result,
    node_names: list,
    encoding: MatrixEncoding = MatrixEncoding.STANDARD,
) -> np.ndarray:
    """
    Convert a supported algorithm result to an endpoint matrix.

    Return an n-by-n matrix. Entry mat[i, j] is the mark at node j on the
    edge between i and j: TAIL=-1, NULL=0, ARROW=1, or CIRCLE=2.

    Supported inputs:
      - np.ndarray: a PAG matrix. Whydra returns the default causal-learn
        encoding. Set ``encoding`` when an external matrix uses another
        convention, such as ``MatrixEncoding.PCALG`` from R.
      - A graph, such as the CPDAG returned by PC.
      - A wrapper whose ``.G`` holds the graph, such as a low-level RAI result.
    """
    n = len(node_names)

    if isinstance(result, np.ndarray) and result.ndim == 2:
        null, tail, arrow, circle = endpoint_codes(encoding)
        # Map the input encoding to the standard internal codes.
        to_endpoint = {null: 0, tail: -1, arrow: 1, circle: 2}
        present = {int(v) for v in np.asarray(result).ravel().tolist()}
        unexpected = present - set(to_endpoint)
        if unexpected:
            msg = (
                f"Endpoint codes {sorted(unexpected)} are not valid for encoding "
                f"{MatrixEncoding(encoding).value!r}; the matrix is probably in the "
                "other encoding — pass `encoding=` explicitly"
            )
            raise ValueError(msg)
        if result.shape != (n, n):
            msg = "Endpoint matrix dimensions do not match node_names"
            raise ValueError(msg)
        mat = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                mat[i, j] = to_endpoint[int(result[i, j])]
        return validate_endpoint_matrix(mat, n)

    # PC returns a GeneralGraph. RAI wraps it in the .G attribute.
    graph_obj = getattr(result, "G", result)

    if hasattr(graph_obj, "get_nodes") and hasattr(graph_obj, "get_graph_edges"):
        graph_obj = cast("GeneralGraph", graph_obj)
        names = validate_node_names(
            [node.get_name() for node in graph_obj.get_nodes()], n
        )
        requested = validate_node_names(node_names, n)
        if set(names) != set(requested):
            msg = "Graph node names do not match requested names"
            raise ValueError(msg)
        order = [names.index(name) for name in requested]
        return validate_endpoint_matrix(
            get_adj_matrix(graph_obj)[np.ix_(order, order)], n
        )

    msg = (
        f"Could not convert a result of type {type(result)} to an endpoint matrix. "
        "Expected a 2-D endpoint array (PAG) or a graph exposing get_nodes and "
        "get_graph_edges, either directly or through .G."
    )
    raise TypeError(msg)


def is_pag_result(result) -> bool:
    """Check the graph's kind, looking inside ``.G`` for wrapped graphs.

    If the kind is unspecified, treat any 2D NumPy array as a PAG matrix.
    """
    graph = getattr(result, "G", result)
    kind = getattr(graph, "graph_kind", None)
    if kind is not None:
        return kind == "pag"
    return isinstance(result, np.ndarray) and result.ndim == 2


# ============================================================
#  GROUND-TRUTH CONVERSION FROM DAG TO CPDAG
# ============================================================


def dag_adj_to_cpdag_endpoint_matrix(dag_adj: np.ndarray) -> np.ndarray:
    """
    Convert a binary DAG adjacency matrix (adj[i, j] = 1 for i → j) to CPDAG endpoints.

    Build the CPDAG locally. Keep the DAG skeleton, orient its unshielded
    colliders, then apply Meek's rules to the remaining undirected edges.
    """
    dag = np.asarray(dag_adj)
    if dag.ndim != 2 or dag.shape[0] != dag.shape[1]:
        msg = "dag_adj must be a square matrix"
        raise ValueError(msg)
    n = len(dag)
    if np.any(np.diag(dag)):
        msg = "DAG truth has a self cycle"
        raise ValueError(msg)
    unexpected = set(np.unique(dag).tolist()) - {0, 1}
    if unexpected:
        msg = (
            f"dag_adj must be a binary DAG matrix (0/1, adj[i,j]=1 means i->j), "
            f"got unexpected values {sorted(unexpected)} — looks like an endpoint "
            "matrix (TAIL=-1/ARROW=1/CIRCLE=2) was passed instead"
        )
        raise ValueError(msg)
    from whydra.background_knowledge import _find_cycle

    if _find_cycle(zip(*np.where(dag != 0), strict=False)) is not None:
        msg = "DAG truth must be acyclic"
        raise ValueError(msg)
    skeleton = (dag != 0) | (dag.T != 0)
    mat = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if skeleton[i, j]:
                mat[i, j] = mat[j, i] = Endpoint.TAIL.value

    def adjacent(i, j):
        return bool(skeleton[i, j])

    def undirected(i, j):
        return mat[i, j] == Endpoint.TAIL.value and mat[j, i] == Endpoint.TAIL.value

    def directed(i, j):
        return mat[i, j] == Endpoint.ARROW.value and mat[j, i] == Endpoint.TAIL.value

    def orient(i, j):
        if undirected(i, j):
            mat[i, j] = Endpoint.ARROW.value
            mat[j, i] = Endpoint.TAIL.value
            return True
        return False

    # Unshielded colliders are invariant across a Markov-equivalence class.
    for y in range(n):
        parents = [x for x in range(n) if dag[x, y] != 0]
        for x in range(len(parents)):
            for z in parents[x + 1 :]:
                if not adjacent(parents[x], z):
                    orient(parents[x], y)
                    orient(z, y)

    # Meek R1-R3. Repeating the pass reaches the fixed point for the rules
    # needed by DAG -> CPDAG conversion without an external graph package.
    changed = True
    while changed:
        changed = False
        for a in range(n):
            for b in range(n):
                if not directed(a, b):
                    continue
                for c in range(n):
                    if c in (a, b) or not undirected(b, c):
                        continue
                    if not adjacent(a, c):
                        changed |= orient(b, c)  # R1

        for a in range(n):
            for b in range(n):
                if not directed(a, b):
                    continue
                for c in range(n):
                    if c in (a, b) or not directed(b, c):
                        continue
                    if undirected(a, c):
                        changed |= orient(a, c)  # R2

        for a in range(n):
            for b in range(n):
                if not undirected(a, b):
                    continue
                for c in range(n):
                    if c in (a, b) or not undirected(a, c) or adjacent(b, c):
                        continue
                    for d in range(n):
                        if d in (a, b, c):
                            continue
                        if directed(b, d) and directed(c, d):
                            changed |= orient(a, d)  # R3
    return mat


def dag_adj_to_general_graph(
    dag_adj: np.ndarray, node_names: list | None = None
) -> GeneralGraph:
    """Convert a binary DAG adjacency matrix to a GeneralGraph."""
    n = dag_adj.shape[0]
    if node_names is None:
        node_names = default_node_names(n)

    nodes = [GraphNode(name) for name in node_names]
    g = GeneralGraph(nodes)
    for i in range(n):
        for j in range(n):
            if dag_adj[i, j] != 0:
                g.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW))
    return g


# ============================================================
#  PAG TO CPDAG PROXY
# ============================================================


def convert_pag_to_cpdag_proxy(adj_mat: np.ndarray) -> np.ndarray:
    """
    Convert a PAG endpoint matrix with circles to a CPDAG proxy.

    Treat each uncertain circle (o, code 2) as a tail (code -1). An o→
    edge becomes a directed edge, and o-o becomes undirected. Arrowheads
    (code 1) and existing tails (code -1) keep their marks.
    """
    mat = adj_mat.copy()
    mat[mat == Endpoint.CIRCLE.value] = Endpoint.TAIL.value
    return mat


# ============================================================
#  GRAPH QUALITY METRICS
# ============================================================


def f1_score(prec, rec):
    """Return the harmonic mean of precision and recall, or zero if their sum is zero."""
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def compute_metrics_from_endpoint_matrices(
    true_ep: np.ndarray,
    est_ep: np.ndarray,
) -> dict:
    """
    Compute all metrics for two endpoint matrices using TAIL/NULL/ARROW/CIRCLE codes.

    The caller must convert both inputs to comparable graphs before scoring
    them: convert a ground-truth DAG to a CPDAG and a PAG to its CPDAG proxy as needed.
    """
    true_ep = validate_endpoint_matrix(true_ep)
    est_ep = validate_endpoint_matrix(est_ep, len(true_ep))
    n = true_ep.shape[0]

    # --- Adjacency (skeleton) ---
    true_skel = true_ep != 0
    est_skel = est_ep != 0

    # An edge is present if either endpoint is nonzero. Make adjacency symmetric.
    true_skel_sym = true_skel | true_skel.T
    est_skel_sym = est_skel | est_skel.T
    np.fill_diagonal(true_skel_sym, False)
    np.fill_diagonal(est_skel_sym, False)

    # Count the upper triangle so that each edge is counted once.
    triu = np.triu_indices(n, k=1)
    true_edges = true_skel_sym[triu]
    est_edges = est_skel_sym[triu]

    tp_adj = int(np.sum(true_edges & est_edges))
    fp_adj = int(np.sum(~true_edges & est_edges))
    fn_adj = int(np.sum(true_edges & ~est_edges))

    adj_prec = tp_adj / (tp_adj + fp_adj) if (tp_adj + fp_adj) > 0 else 0.0
    adj_rec = tp_adj / (tp_adj + fn_adj) if (tp_adj + fn_adj) > 0 else 0.0
    adj_f1 = f1_score(adj_prec, adj_rec)

    # --- Arrowhead orientation ---
    true_arrows = true_ep == Endpoint.ARROW.value
    est_arrows = est_ep == Endpoint.ARROW.value

    tp_arr = int(np.sum(true_arrows & est_arrows))
    fp_arr = int(np.sum(~true_arrows & est_arrows))
    fn_arr = int(np.sum(true_arrows & ~est_arrows))

    arrow_prec = tp_arr / (tp_arr + fp_arr) if (tp_arr + fp_arr) > 0 else 0.0
    arrow_rec = tp_arr / (tp_arr + fn_arr) if (tp_arr + fn_arr) > 0 else 0.0
    arrow_f1 = f1_score(arrow_prec, arrow_rec)

    # --- Arrowheads on edges shared by both graphs ---
    common_edges_mask = true_skel & est_skel
    if np.sum(common_edges_mask) > 0:
        true_arrows_ce = true_arrows & common_edges_mask
        est_arrows_ce = est_arrows & common_edges_mask
        tp_arr_ce = int(np.sum(true_arrows_ce & est_arrows_ce))
        fp_arr_ce = int(np.sum(~true_arrows_ce & est_arrows_ce))
        fn_arr_ce = int(np.sum(true_arrows_ce & ~est_arrows_ce))
        arrow_prec_ce = (
            tp_arr_ce / (tp_arr_ce + fp_arr_ce) if (tp_arr_ce + fp_arr_ce) > 0 else 0.0
        )
        arrow_rec_ce = (
            tp_arr_ce / (tp_arr_ce + fn_arr_ce) if (tp_arr_ce + fn_arr_ce) > 0 else 0.0
        )
        arrow_f1_ce = f1_score(arrow_prec_ce, arrow_rec_ce)
    else:
        arrow_prec_ce = arrow_rec_ce = arrow_f1_ce = 0.0

    # --- SHD ---
    # Compute SHD directly. Each pair of nodes counts once if either endpoint
    # differs. Causal-learn uses a different definition of SHD, so its values
    # cannot be compared directly with this score.
    shd = 0
    for i in range(n):
        for j in range(i + 1, n):
            if true_ep[i, j] != est_ep[i, j] or true_ep[j, i] != est_ep[j, i]:
                shd += 1

    # --- Additional statistics ---
    n_edges = int(np.sum(est_edges))
    degrees = np.sum(est_skel_sym, axis=1)
    max_degree = int(np.max(degrees)) if len(degrees) > 0 else 0

    return {
        "SHD": shd,
        "adj_f1": adj_f1,
        "arrow_f1": arrow_f1,
        "arrow_f1_common_edges": arrow_f1_ce,
        "adj_support": tp_adj + fn_adj,
        "arrow_support": tp_arr + fn_arr,
        "adj_f1_defined": bool(tp_adj + fp_adj + fn_adj),
        "arrow_f1_defined": bool(tp_arr + fp_arr + fn_arr),
        "adj_precision": adj_prec,
        "adj_recall": adj_rec,
        "arrow_precision": arrow_prec,
        "arrow_recall": arrow_rec,
        "arrow_precision_common_edges": arrow_prec_ce,
        "arrow_recall_common_edges": arrow_rec_ce,
        # A fraction with a zero denominator is undefined. For example, a perfect
        # prediction on an edgeless graph loses 0% of its edges, not 100%.
        "missing_edges_share": (1.0 - adj_rec) if (tp_adj + fn_adj) > 0 else 0.0,
        "false_edges_share": (1.0 - adj_prec) if (tp_adj + fp_adj) > 0 else 0.0,
        "n_edges": n_edges,
        "max_degree": max_degree,
    }


def compute_metrics_from_dag_and_result(
    true_dag_adj: np.ndarray,
    est_ep_mat: np.ndarray,
    is_pag: bool = False,
) -> dict:
    """Compute metrics from a ground-truth DAG and an estimated endpoint matrix.

    Convert the true DAG to a CPDAG and, when requested, the estimated PAG
    to a CPDAG proxy before computing the metrics.

    Parameters
    ----------
    true_dag_adj : binary DAG matrix (adj[i, j] = 1 denotes i → j)
    est_ep_mat : endpoint matrix (TAIL=-1, NULL=0, ARROW=1, CIRCLE=2)
    is_pag : if True, convert est_ep_mat from a PAG to a CPDAG proxy
    """
    true_ep = dag_adj_to_cpdag_endpoint_matrix(true_dag_adj)
    est_ep = convert_pag_to_cpdag_proxy(est_ep_mat) if is_pag else est_ep_mat
    return compute_metrics_from_endpoint_matrices(true_ep, est_ep)


# ============================================================
#  BACKWARD COMPATIBILITY: calculate_metrics(true_G, est_G)
# ============================================================


def calculate_metrics(
    true_G: GeneralGraph,
    est_G: GeneralGraph,
    *,
    node_mapping=None,
    truth_kind=None,
    pag_proxy=None,
) -> dict:
    """Compute graph metrics from two GeneralGraph objects."""
    true_names = validate_node_names(
        [n.get_name() for n in true_G.get_nodes()], len(true_G.get_nodes())
    )
    est_names = validate_node_names(
        [n.get_name() for n in est_G.get_nodes()], len(est_G.get_nodes())
    )
    if node_mapping is not None:
        est_names = validate_node_names(
            [node_mapping.get(n, n) for n in est_names], len(est_names)
        )
    if set(true_names) != set(est_names):
        msg = "Truth and estimate node names must match, or supply an explicit node_mapping"
        raise ValueError(msg)
    order = [est_names.index(n) for n in true_names]
    est_mat = validate_endpoint_matrix(
        get_adj_matrix(est_G)[np.ix_(order, order)], len(true_names)
    )
    true_raw = validate_endpoint_matrix(get_adj_matrix(true_G), len(true_names))
    kind = truth_kind or getattr(true_G, "graph_kind", None) or "dag"
    if kind == "dag":
        if np.any(
            (true_raw != 0)
            & ~(
                ((true_raw == 1) & (true_raw.T == -1))
                | ((true_raw == -1) & (true_raw.T == 1))
            )
        ):
            msg = "DAG truth must have only directed edges; declare truth_kind for a partial graph"
            raise ValueError(msg)
        true_mat = dag_adj_to_cpdag_endpoint_matrix((true_raw == 1).astype(int))
    elif kind in {"cpdag", "pag"}:
        true_mat = true_raw
    else:
        msg = "truth_kind must be dag, cpdag, or pag"
        raise ValueError(msg)
    estimate_pag = is_pag_result(est_G) or np.any(est_mat == 2)
    approximate = estimate_pag and kind != "pag"
    if approximate:
        if pag_proxy is False:
            msg = "PAG evaluation needs PAG truth, or explicitly select pag_proxy=True for approximate metrics"
            raise ValueError(msg)
        if pag_proxy is None:
            warnings.warn(
                "Implicit PAG-to-CPDAG proxy metrics are deprecated and approximate; pass pag_proxy=True or provide PAG truth",
                DeprecationWarning,
                stacklevel=2,
            )
        est_mat = convert_pag_to_cpdag_proxy(est_mat)
    result = compute_metrics_from_endpoint_matrices(true_mat, est_mat)
    result.update(
        metric_mode="pag_to_cpdag_proxy" if approximate else kind,
        approximate=approximate,
    )
    return result
