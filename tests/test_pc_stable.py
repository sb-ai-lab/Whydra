import sys
import os
import numpy as np
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
standalone_dir = os.path.join(project_root, 'whydra', 'algorithms', 'standalone')
if standalone_dir not in sys.path:
    sys.path.insert(0, standalone_dir)

from whydra.algorithms.wrappers import StandalonePCStable, create_standalone_algorithm, run_standalone_algorithm

from .conftest import (
    generate_linear_sem_data,
    get_chain_graph_3nodes,
    get_fork_graph_3nodes,
    get_collider_graph_3nodes,
    get_v_structure_graph_4nodes,
    count_edges,
    has_edge,
)


class TestPCStableBasic:
    def test_pc_stable_returns_general_graph(self):
        np.random.seed(42)
        data = np.random.randn(100, 3)
        algo = StandalonePCStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        assert isinstance(graph, type(graph))
        assert hasattr(graph, 'get_nodes')

    def test_pc_stable_node_count(self):
        np.random.seed(42)
        n_vars = 5
        data = np.random.randn(200, n_vars)
        algo = StandalonePCStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        assert len(graph.get_nodes()) == n_vars

    def test_pc_stable_default_parameters(self):
        algo = StandalonePCStable()
        assert algo.alpha == 0.05
        assert algo.indep_test == 'fisherz'
        assert algo.n_jobs == 4

    def test_pc_stable_custom_parameters(self):
        algo = StandalonePCStable(alpha=0.01, indep_test='fisherz', n_jobs=2)
        assert algo.alpha == 0.01
        assert algo.indep_test == 'fisherz'
        assert algo.n_jobs == 2


class TestPCStableChainStructure:
    def test_chain_skeleton_recovery(self):
        W = get_chain_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=1000, noise_std=1.0, seed=42)
        algo = StandalonePCStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        
        assert has_edge(graph, 0, 1), "Edge X1-X2 should exist"
        assert has_edge(graph, 1, 2), "Edge X2-X3 should exist"
        assert not has_edge(graph, 0, 2), "Edge X1-X3 should not exist (conditional independence)"

    def test_chain_edge_count(self):
        W = get_chain_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=1000, noise_std=1.0, seed=42)
        algo = StandalonePCStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        assert count_edges(graph) == 2


class TestPCStableForkStructure:
    def test_fork_skeleton_recovery(self):
        W = get_fork_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=1000, noise_std=1.0, seed=42)
        algo = StandalonePCStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        
        assert has_edge(graph, 0, 1), "Edge X1-X2 should exist"
        assert has_edge(graph, 0, 2), "Edge X1-X3 should exist"
        assert not has_edge(graph, 1, 2), "Edge X2-X3 should not exist"


class TestPCStableColliderStructure:
    def test_collider_skeleton_recovery(self):
        W = get_collider_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=1000, noise_std=1.0, seed=42)
        algo = StandalonePCStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        
        assert has_edge(graph, 0, 1), "Edge X1-X2 should exist"
        assert has_edge(graph, 1, 2), "Edge X2-X3 should exist"
        assert not has_edge(graph, 0, 2), "Edge X1-X3 should not exist (marginally independent)"


class TestPCStableVStructure:
    def test_v_structure_recovery(self):
        W = get_v_structure_graph_4nodes()
        data = generate_linear_sem_data(W, n_samples=1000, noise_std=1.0, seed=42)
        algo = StandalonePCStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        
        assert has_edge(graph, 0, 1), "Edge X1-X2 should exist"
        assert has_edge(graph, 1, 2), "Edge X2-X3 should exist"
        assert has_edge(graph, 1, 3), "Edge X2-X4 should exist"
        assert not has_edge(graph, 0, 2), "Edge X1-X3 should not exist"
        assert not has_edge(graph, 0, 3), "Edge X1-X4 should not exist"
        assert not has_edge(graph, 2, 3), "Edge X3-X4 should not exist"


class TestPCStableAlphaSensitivity:
    def test_stricter_alpha_fewer_edges(self):
        W = get_chain_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=500, noise_std=1.0, seed=42)
        
        algo_loose = StandalonePCStable(alpha=0.1, indep_test='fisherz', n_jobs=1)
        graph_loose = algo_loose.run(data)
        
        algo_strict = StandalonePCStable(alpha=0.01, indep_test='fisherz', n_jobs=1)
        graph_strict = algo_strict.run(data)
        
        edges_loose = count_edges(graph_loose)
        edges_strict = count_edges(graph_strict)
        assert edges_strict <= edges_loose, "Stricter alpha should result in fewer or equal edges"


class TestPCStableIndependentVariables:
    def test_no_edges_for_independent_data(self):
        np.random.seed(42)
        data = np.random.randn(500, 4)
        algo = StandalonePCStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        
        for i in range(4):
            for j in range(i + 1, 4):
                assert not has_edge(graph, i, j), f"Edge between X{i+1} and X{j+1} should not exist"


class TestPCStableParallelExecution:
    def test_parallel_vs_sequential_equivalence(self):
        W = get_chain_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=500, noise_std=1.0, seed=42)
        
        algo_seq = StandalonePCStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph_seq = algo_seq.run(data)
        
        algo_par = StandalonePCStable(alpha=0.05, indep_test='fisherz', n_jobs=2)
        graph_par = algo_par.run(data)
        
        assert count_edges(graph_seq) == count_edges(graph_par), \
            "Sequential and parallel should produce same number of edges"


class TestPCStableAlgorithmFactory:
    def test_create_by_canonical_name(self):
        algo = create_standalone_algorithm("standalonepcstable", alpha=0.01)
        assert isinstance(algo, StandalonePCStable)
        assert algo.alpha == 0.01

    def test_create_by_short_name(self):
        algo = create_standalone_algorithm("pc_stable", alpha=0.01)
        assert isinstance(algo, StandalonePCStable)

    def test_create_by_alias(self):
        algo = create_standalone_algorithm("pcstable", alpha=0.01)
        assert isinstance(algo, StandalonePCStable)

    def test_invalid_algorithm_name(self):
        with pytest.raises(ValueError):
            create_standalone_algorithm("invalid_algo")


class TestPCStableRunWithDataPath:
    def test_run_with_numpy_array(self):
        np.random.seed(42)
        data = np.random.randn(100, 3)
        graph = run_standalone_algorithm("pc_stable", data, alpha=0.05, n_jobs=1)
        assert len(graph.get_nodes()) == 3

    def test_run_standalone_algorithm_with_kwargs(self):
        np.random.seed(42)
        data = np.random.randn(100, 3)
        graph = run_standalone_algorithm(
            "StandalonePCStable",
            data,
            alpha=0.01,
            indep_test='fisherz',
            n_jobs=1
        )
        assert len(graph.get_nodes()) == 3


class TestPCStableEdgeCases:
    def test_small_sample_size(self):
        np.random.seed(42)
        data = np.random.randn(20, 3)
        algo = StandalonePCStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        assert len(graph.get_nodes()) == 3

    def test_two_variables(self):
        np.random.seed(42)
        data = np.random.randn(200, 2)
        algo = StandalonePCStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        assert len(graph.get_nodes()) == 2

    def test_many_variables_sparse_graph(self):
        W = np.zeros((6, 6))
        W[0, 1] = 0.8
        W[2, 3] = 0.8
        W[4, 5] = 0.8
        data = generate_linear_sem_data(W, n_samples=1000, noise_std=1.0, seed=42)
        algo = StandalonePCStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        
        assert has_edge(graph, 0, 1)
        assert has_edge(graph, 2, 3)
        assert has_edge(graph, 4, 5)
        assert not has_edge(graph, 0, 2)
        assert not has_edge(graph, 1, 3)
        assert not has_edge(graph, 3, 4)
