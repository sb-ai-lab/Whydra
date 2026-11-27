import importlib
import importlib.util

import numpy as np
import pytest

# On installations without PyTorch, these modules fail at import (directly or
# through a CUDA module). The CUDA tests themselves are skipped when torch is
# present but no GPU is available; here torch is not installed at all, so the
# modules must not be collected.
if importlib.util.find_spec("torch") is None:
    collect_ignore_glob = [
        "test_cuda_*.py",
        "test_fci_host_helpers.py",
        "test_fci_stable_cuda*.py",
        "test_pag_*_cuda.py",
        "test_pc_stable_cuda*.py",
        "test_pdsep_cuda.py",
        "test_rai_stable_cuda.py",
    ]


def pytest_addoption(parser):
    parser.addoption(
        "--require-cuda",
        action="store_true",
        help="Require a CUDA device and all CUDA extensions instead of a CPU-only run.",
    )


def pytest_sessionstart(session):
    if not session.config.getoption("--require-cuda", default=False):
        return
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise pytest.UsageError("--require-cuda requires PyTorch") from exc
    if not torch.cuda.is_available():
        raise pytest.UsageError("--require-cuda requires an available CUDA device")
    try:
        importlib.import_module("whydra.algorithms.graph_core._cuda")
        importlib.import_module("whydra.algorithms.graph_core._whydra_rai_host")
    except (ImportError, OSError) as exc:
        raise pytest.UsageError(
            f"--require-cuda could not load CUDA extensions: {exc}"
        ) from exc


def legacy_random_state():
    """Return NumPy's shared RandomState when a test needs its original seeded draws."""
    # Import on demand so collecting CUDA availability tests needs no SciPy stats backend.
    from scipy.stats import norm

    # A fresh distribution retains the default even if a caller changed norm.random_state.
    return norm().random_state


def run_scalar_queries(ci_test, requests):
    """Run conditional-independence queries in order and propagate the first error."""
    for request in requests:
        ci_test(*request)


def generate_linear_sem_data(W, n_samples=1000, noise_std=1.0, seed=42):
    """
    Generate data from a linear Structural Equation Model.

    Parameters
    ----------
    W : np.ndarray
        Adjacency matrix of shape (n_vars, n_vars) where W[i, j] is the weight from i to j
    n_samples : int
        Number of samples to generate
    noise_std : float
        Standard deviation of Gaussian noise
    seed : int
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Generated data of shape (n_samples, n_vars)
    """
    rng = legacy_random_state()
    rng.seed(seed)
    n_vars = W.shape[0]
    noise = rng.normal(0, noise_std, size=(n_samples, n_vars))
    X = noise.copy()
    for i in range(n_vars):
        for j in range(n_vars):
            if W[j, i] != 0:
                X[:, i] += W[j, i] * X[:, j]
    return X


def get_chain_graph_3nodes():
    """Returns ground truth for X1 -> X2 -> X3"""
    W = np.zeros((3, 3))
    W[0, 1] = 0.8
    W[1, 2] = 0.8
    return W


def get_fork_graph_3nodes():
    """Returns ground truth for X2 -> X1, X2 -> X3 (topologically ordered)"""
    W = np.zeros((3, 3))
    W[0, 1] = 0.8
    W[0, 2] = 0.8
    return W


def get_collider_graph_3nodes():
    """Returns ground truth for X1 -> X2 <- X3"""
    W = np.zeros((3, 3))
    W[0, 1] = 0.8
    W[2, 1] = 0.8
    return W


def get_v_structure_graph_4nodes():
    """Returns ground truth for X1 -> X2 <- X3, X2 -> X4"""
    W = np.zeros((4, 4))
    W[0, 1] = 0.8
    W[2, 1] = 0.8
    W[1, 3] = 0.8
    return W


def count_edges(graph):
    """Count the number of edges in a GeneralGraph"""
    return len(graph.get_graph_edges())


def has_edge(graph, node_i, node_j):
    """Check if there is an edge between two nodes (by index)"""
    nodes = graph.get_nodes()
    return graph.is_adjacent_to(nodes[node_i], nodes[node_j])


def is_directed_edge(graph, source_idx, target_idx):
    """Check if there is a directed edge source -> target"""
    from whydra.algorithms.graph_core import Endpoint

    nodes = graph.get_nodes()
    edge = graph.get_edge(nodes[source_idx], nodes[target_idx])
    if edge is None:
        return False
    return (
        edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW
    )


def is_undirected_edge(graph, node_i, node_j):
    """Check if there is an undirected edge between two nodes"""
    from whydra.algorithms.graph_core import Endpoint

    nodes = graph.get_nodes()
    edge = graph.get_edge(nodes[node_i], nodes[node_j])
    if edge is None:
        return False
    return (
        edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.TAIL
    )
