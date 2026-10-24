import numpy as np


def pytest_addoption(parser):
    """Регистрирует опцию для сверки на реальных данных bnlearn."""
    parser.addoption(
        "--run-bnlearn",
        action="store_true",
        default=False,
        help="запустить сверку с causal-learn на кейсах bnlearn "
             "(долго, расхождения ожидаемы и намеренно не замаскированы)",
    )



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
    np.random.seed(seed)
    n_vars = W.shape[0]
    noise = np.random.normal(0, noise_std, size=(n_samples, n_vars))
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
    from causallearn.graph.Endpoint import Endpoint
    nodes = graph.get_nodes()
    edge = graph.get_edge(nodes[source_idx], nodes[target_idx])
    if edge is None:
        return False
    return (edge.get_endpoint1() == Endpoint.TAIL and 
            edge.get_endpoint2() == Endpoint.ARROW)


def is_undirected_edge(graph, node_i, node_j):
    """Check if there is an undirected edge between two nodes"""
    from causallearn.graph.Endpoint import Endpoint
    nodes = graph.get_nodes()
    edge = graph.get_edge(nodes[node_i], nodes[node_j])
    if edge is None:
        return False
    return (edge.get_endpoint1() == Endpoint.TAIL and 
            edge.get_endpoint2() == Endpoint.TAIL)
