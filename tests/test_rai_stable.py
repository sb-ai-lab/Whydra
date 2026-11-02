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

from whydra.algorithms.wrappers import StandaloneRAIStable, create_standalone_algorithm

from .conftest import (
    generate_linear_sem_data,
    get_chain_graph_3nodes,
    get_fork_graph_3nodes,
    get_collider_graph_3nodes,
    get_v_structure_graph_4nodes,
    count_edges,
    has_edge,
)


class TestRAIStableBasic:
    def test_rai_stable_returns_general_graph(self):
        np.random.seed(42)
        data = np.random.randn(100, 3)
        algo = StandaloneRAIStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        assert hasattr(graph, 'get_nodes')

    def test_rai_stable_node_count(self):
        np.random.seed(42)
        n_vars = 5
        data = np.random.randn(200, n_vars)
        algo = StandaloneRAIStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        assert len(graph.get_nodes()) == n_vars

    def test_rai_stable_default_parameters(self):
        algo = StandaloneRAIStable()
        assert algo.alpha == 0.05
        assert algo.indep_test == 'fisherz'
        assert algo.n_jobs == 4

    def test_rai_stable_custom_parameters(self):
        algo = StandaloneRAIStable(alpha=0.01, indep_test='fisherz', n_jobs=2)
        assert algo.alpha == 0.01
        assert algo.indep_test == 'fisherz'
        assert algo.n_jobs == 2


class TestRAIStableChainStructure:
    def test_chain_skeleton_recovery(self):
        W = get_chain_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=1000, noise_std=1.0, seed=42)
        algo = StandaloneRAIStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        
        assert has_edge(graph, 0, 1), "Edge X1-X2 should exist"
        assert has_edge(graph, 1, 2), "Edge X2-X3 should exist"
        assert not has_edge(graph, 0, 2), "Edge X1-X3 should not exist"

    def test_chain_edge_count(self):
        W = get_chain_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=1000, noise_std=1.0, seed=42)
        algo = StandaloneRAIStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        assert count_edges(graph) == 2


class TestRAIStableForkStructure:
    def test_fork_skeleton_recovery(self):
        W = get_fork_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=1000, noise_std=1.0, seed=42)
        algo = StandaloneRAIStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        
        assert has_edge(graph, 0, 1), "Edge X1-X2 should exist"
        assert has_edge(graph, 0, 2), "Edge X1-X3 should exist"
        assert not has_edge(graph, 1, 2), "Edge X2-X3 should not exist"


class TestRAIStableColliderStructure:
    def test_collider_skeleton_recovery(self):
        W = get_collider_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=1000, noise_std=1.0, seed=42)
        algo = StandaloneRAIStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        
        assert has_edge(graph, 0, 1), "Edge X1-X2 should exist"
        assert has_edge(graph, 1, 2), "Edge X2-X3 should exist"
        assert not has_edge(graph, 0, 2), "Edge X1-X3 should not exist"


class TestRAIStableVStructure:
    def test_v_structure_recovery(self):
        W = get_v_structure_graph_4nodes()
        data = generate_linear_sem_data(W, n_samples=1000, noise_std=1.0, seed=42)
        algo = StandaloneRAIStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        
        assert has_edge(graph, 0, 1), "Edge X1-X2 should exist"
        assert has_edge(graph, 1, 2), "Edge X2-X3 should exist"
        assert has_edge(graph, 1, 3), "Edge X2-X4 should exist"
        assert not has_edge(graph, 0, 2), "Edge X1-X3 should not exist"
        assert not has_edge(graph, 0, 3), "Edge X1-X4 should not exist"
        assert not has_edge(graph, 2, 3), "Edge X3-X4 should not exist"


class TestRAIStableIndependentVariables:
    def test_no_edges_for_independent_data(self):
        np.random.seed(42)
        data = np.random.randn(500, 4)
        algo = StandaloneRAIStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        
        for i in range(4):
            for j in range(i + 1, 4):
                assert not has_edge(graph, i, j), f"Edge between X{i+1} and X{j+1} should not exist"


class TestRAIStableAlphaSensitivity:
    def test_stricter_alpha_fewer_edges(self):
        W = get_chain_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=500, noise_std=1.0, seed=42)
        
        algo_loose = StandaloneRAIStable(alpha=0.1, indep_test='fisherz', n_jobs=1)
        graph_loose = algo_loose.run(data)
        
        algo_strict = StandaloneRAIStable(alpha=0.01, indep_test='fisherz', n_jobs=1)
        graph_strict = algo_strict.run(data)
        
        edges_loose = count_edges(graph_loose)
        edges_strict = count_edges(graph_strict)
        assert edges_strict <= edges_loose, "Stricter alpha should result in fewer or equal edges"


class TestRAIStableParallelExecution:
    def test_parallel_vs_sequential_equivalence(self):
        W = get_chain_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=500, noise_std=1.0, seed=42)
        
        algo_seq = StandaloneRAIStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph_seq = algo_seq.run(data)
        
        algo_par = StandaloneRAIStable(alpha=0.05, indep_test='fisherz', n_jobs=2)
        graph_par = algo_par.run(data)
        
        assert np.array_equal(graph_seq.graph, graph_par.graph), \
            "Sequential and parallel must produce the same endpoint matrix"


class TestRAIStableAlgorithmFactory:
    def test_create_rai_stable_by_canonical_name(self):
        algo = create_standalone_algorithm("standaloneraistable", alpha=0.01)
        assert isinstance(algo, StandaloneRAIStable)
        assert algo.alpha == 0.01

    def test_create_rai_stable_by_short_name(self):
        algo = create_standalone_algorithm("rai_stable", alpha=0.01)
        assert isinstance(algo, StandaloneRAIStable)


class TestRAIStableEdgeCases:
    def test_small_sample_size(self):
        np.random.seed(42)
        data = np.random.randn(30, 3)
        algo = StandaloneRAIStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        assert len(graph.get_nodes()) == 3

    def test_two_variables(self):
        np.random.seed(42)
        data = np.random.randn(200, 2)
        algo = StandaloneRAIStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        assert len(graph.get_nodes()) == 2

    def test_many_variables_sparse_graph(self):
        W = np.zeros((6, 6))
        W[0, 1] = 0.8
        W[2, 3] = 0.8
        W[4, 5] = 0.8
        data = generate_linear_sem_data(W, n_samples=1000, noise_std=1.0, seed=42)
        algo = StandaloneRAIStable(alpha=0.05, indep_test='fisherz', n_jobs=1)
        graph = algo.run(data)
        
        assert has_edge(graph, 0, 1)
        assert has_edge(graph, 2, 3)
        assert has_edge(graph, 4, 5)


class TestRAIStableSpeculativeReplay:
    """Параллельный слой RAI: воркеры считают p-value наперёд, решения принимает replay."""

    @staticmethod
    def _dag_data():
        rng = np.random.default_rng(2024)
        n_vars = 14
        W = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if rng.random() < 0.3:
                    W[i, j] = rng.uniform(0.5, 0.9) * rng.choice([-1, 1])
        return generate_linear_sem_data(W, n_samples=2000, seed=7)

    def test_replay_repeats_sequential_decisions(self):
        """Последовательность решений (x, y, S, independent) при n_jobs=3 та же, что при n_jobs=1."""
        from whydra.algorithms.standalone.rai_algo import RAIStableLearner

        data = self._dag_data()
        names = [f"X{i + 1}" for i in range(data.shape[1])]
        seq = RAIStableLearner(data, 0.05, "fisherz", names)
        seq.trace = []
        graph_seq = seq.learn_structure(n_jobs=1)

        par = RAIStableLearner(data, 0.05, "fisherz", names)
        par.trace = []
        graph_par = par.learn_structure(n_jobs=3)

        assert par.trace == seq.trace
        assert np.array_equal(graph_seq.graph, graph_par.graph)

    def test_workers_take_most_of_the_ci_tests(self):
        """При n_jobs=3 главный процесс считает лишь малую долю CI-тестов (иначе схема выродилась в sync)."""
        from whydra.algorithms.standalone.rai_algo import RAIStableLearner

        data = self._dag_data()
        names = [f"X{i + 1}" for i in range(data.shape[1])]
        par = RAIStableLearner(data, 0.05, "fisherz", names)
        par.learn_structure(n_jobs=3)

        assert par.stats["parallel_phases"] > 0
        assert par.stats["ci_worker"] > 0
        assert par.stats["ci_sync"] < 0.25 * par.stats["ci_total"], par.stats
