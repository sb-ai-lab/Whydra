import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
standalone_dir = os.path.join(project_root, 'whydra', 'algorithms', 'standalone')
if standalone_dir not in sys.path:
    sys.path.insert(0, standalone_dir)

from whydra.algorithms.wrappers import StandaloneFCIStable, create_standalone_algorithm
from whydra.algorithms.standalone import fci_algo_v2
from whydra.algorithms.graph_core import CausalGraph, CIT

from .conftest import (
    generate_linear_sem_data,
    get_chain_graph_3nodes,
    get_fork_graph_3nodes,
    get_collider_graph_3nodes,
    count_edges,
    has_edge,
)


class TestFCIStableBasic:
    def test_fci_stable_returns_general_graph(self):
        np.random.seed(42)
        data = np.random.randn(100, 3)
        algo = StandaloneFCIStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        assert hasattr(graph, 'get_nodes')

    def test_fci_stable_node_count(self):
        np.random.seed(42)
        n_vars = 4
        data = np.random.randn(200, n_vars)
        algo = StandaloneFCIStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        assert len(graph.get_nodes()) == n_vars

    def test_fci_stable_default_parameters(self):
        algo = StandaloneFCIStable()
        assert algo.alpha == 0.05
        assert algo.indep_test == 'fisherz'
        assert algo.n_jobs == 4
        assert algo.verbose == False

    def test_fci_stable_custom_parameters(self):
        algo = StandaloneFCIStable(alpha=0.01, indep_test='fisherz', n_jobs=2, verbose=True)
        assert algo.alpha == 0.01
        assert algo.indep_test == 'fisherz'
        assert algo.n_jobs == 2
        assert algo.verbose == True


class TestFCIStableChainStructure:
    def test_chain_skeleton_recovery(self):
        W = get_chain_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=1000, noise_std=1.0, seed=42)
        algo = StandaloneFCIStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        
        assert has_edge(graph, 0, 1), "Edge X1-X2 should exist"
        assert has_edge(graph, 1, 2), "Edge X2-X3 should exist"


class TestFCIStableForkStructure:
    def test_fork_skeleton_recovery(self):
        W = get_fork_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=1000, noise_std=1.0, seed=42)
        algo = StandaloneFCIStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        
        assert has_edge(graph, 0, 1), "Edge X1-X2 should exist"
        assert has_edge(graph, 0, 2), "Edge X1-X3 should exist"


class TestFCIStableColliderStructure:
    def test_collider_skeleton_recovery(self):
        W = get_collider_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=1000, noise_std=1.0, seed=42)
        algo = StandaloneFCIStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        
        assert has_edge(graph, 0, 1), "Edge X1-X2 should exist"
        assert has_edge(graph, 1, 2), "Edge X2-X3 should exist"


class TestFCIStableIndependentVariables:
    def test_no_edges_for_independent_data(self):
        np.random.seed(42)
        data = np.random.randn(500, 4)
        algo = StandaloneFCIStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        
        edge_count = count_edges(graph)
        assert edge_count == 0, f"Expected 0 edges for independent data, got {edge_count}"


class TestFCIStableAlphaSensitivity:
    def test_stricter_alpha_fewer_edges(self):
        W = get_chain_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=500, noise_std=1.0, seed=42)
        
        algo_loose = StandaloneFCIStable(alpha=0.1, indep_test='fisherz', n_jobs=1)
        graph_loose = algo_loose.run(data)
        
        algo_strict = StandaloneFCIStable(alpha=0.01, indep_test='fisherz', n_jobs=1)
        graph_strict = algo_strict.run(data)
        
        edges_loose = count_edges(graph_loose)
        edges_strict = count_edges(graph_strict)
        assert edges_strict <= edges_loose, "Stricter alpha should result in fewer or equal edges"


class TestFCIStablePAGMatrix:
    def test_pag_matrix_shape(self):
        np.random.seed(42)
        n_vars = 4
        data = np.random.randn(300, n_vars)
        
        cg = CausalGraph(n_vars, [f"X{i+1}" for i in range(n_vars)])
        cg.set_ind_test(CIT(data, method='fisherz'))
        
        pag_matrix = fci_algo_v2.fci_stable_v2(cg, data, alpha=0.05, n_jobs=1, verbose=False)
        
        assert pag_matrix.shape == (n_vars, n_vars), "PAG matrix should be square"
        assert np.allclose(pag_matrix, pag_matrix.T, atol=1), "PAG matrix should be symmetric in structure"

    def test_pag_matrix_diagonal_zeros(self):
        np.random.seed(42)
        n_vars = 3
        data = np.random.randn(300, n_vars)
        
        cg = CausalGraph(n_vars, [f"X{i+1}" for i in range(n_vars)])
        cg.set_ind_test(CIT(data, method='fisherz'))
        
        pag_matrix = fci_algo_v2.fci_stable_v2(cg, data, alpha=0.05, n_jobs=1, verbose=False)
        
        for i in range(n_vars):
            assert pag_matrix[i, i] == 0, "Diagonal elements should be zero"


class TestFCIStableAlgorithmFactory:
    def test_create_by_canonical_name(self):
        algo = create_standalone_algorithm("standalonefcistable", alpha=0.01)
        assert isinstance(algo, StandaloneFCIStable)
        assert algo.alpha == 0.01

    def test_create_by_short_name(self):
        algo = create_standalone_algorithm("fci_stable", alpha=0.01)
        assert isinstance(algo, StandaloneFCIStable)

    def test_create_by_alias(self):
        algo = create_standalone_algorithm("fcistable", alpha=0.01)
        assert isinstance(algo, StandaloneFCIStable)


class TestFCIStableEdgeCases:
    def test_small_sample_size(self):
        np.random.seed(42)
        data = np.random.randn(30, 3)
        algo = StandaloneFCIStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        assert len(graph.get_nodes()) == 3

    def test_two_variables(self):
        np.random.seed(42)
        data = np.random.randn(200, 2)
        algo = StandaloneFCIStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        assert len(graph.get_nodes()) == 2

    def test_many_variables(self):
        np.random.seed(42)
        data = np.random.randn(200, 8)
        algo = StandaloneFCIStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        assert len(graph.get_nodes()) == 8
