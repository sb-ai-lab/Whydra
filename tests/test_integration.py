import sys
import os
import numpy as np
import pytest
import tempfile

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
standalone_dir = os.path.join(project_root, 'whydra', 'algorithms', 'standalone')
if standalone_dir not in sys.path:
    sys.path.insert(0, standalone_dir)

from whydra.algorithms.wrappers import (
    run_standalone_algorithm,
    create_standalone_algorithm,
    available_standalone_algorithms,
    _load_data_like_main,
)

from .conftest import (
    generate_linear_sem_data,
    get_chain_graph_3nodes,
    count_edges,
    has_edge,
)


class TestAvailableAlgorithms:
    def test_available_algorithms_returns_list(self):
        algos = available_standalone_algorithms()
        assert isinstance(algos, list)
        assert len(algos) > 0

    def test_available_algorithms_contains_expected(self):
        algos = available_standalone_algorithms()
        assert "StandalonePCStable" in algos
        assert "StandaloneFCIStable" in algos
        assert "StandaloneRAIStable" in algos


class TestLoadDataLikeMain:
    def test_load_numpy_array(self):
        data = np.array([[1, 2], [3, 4]])
        result = _load_data_like_main(data)
        np.testing.assert_array_equal(result, data)

    def test_load_npy_file(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
        np.save(temp_path, data)
        result = _load_data_like_main(temp_path)
        np.testing.assert_array_equal(result, data)
        os.unlink(temp_path)

    def test_load_pickle_file(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        import pickle
        with open(temp_path, 'wb') as pf:
            pickle.dump(data, pf)
        result = _load_data_like_main(temp_path)
        np.testing.assert_array_equal(result, data)
        os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            _load_data_like_main("nonexistent_file.npy")

    def test_load_invalid_type(self):
        with pytest.raises(TypeError):
            _load_data_like_main([1, 2, 3])


class TestUnifiedAPI:
    def test_run_pc_stable_via_unified_api(self):
        W = get_chain_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=500, noise_std=1.0, seed=42)
        graph = run_standalone_algorithm("pc_stable", data, alpha=0.05, n_jobs=1)
        assert has_edge(graph, 0, 1)
        assert has_edge(graph, 1, 2)

    def test_run_fci_stable_via_unified_api(self):
        W = get_chain_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=500, noise_std=1.0, seed=42)
        graph = run_standalone_algorithm("fci_stable", data, alpha=0.05, n_jobs=1)
        assert has_edge(graph, 0, 1)
        assert has_edge(graph, 1, 2)

    def test_run_rai_stable_via_unified_api(self):
        W = get_chain_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=500, noise_std=1.0, seed=42)
        graph = run_standalone_algorithm("rai_stable", data, alpha=0.05, n_jobs=1)
        assert has_edge(graph, 0, 1)
        assert has_edge(graph, 1, 2)


class TestAlgorithmConsistency:
    def test_pc_vs_rai_on_chain(self):
        W = get_chain_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=1000, noise_std=1.0, seed=42)
        
        pc_graph = run_standalone_algorithm("pc_stable", data, alpha=0.05, n_jobs=1)
        rai_graph = run_standalone_algorithm("rai_stable", data, alpha=0.05, n_jobs=1)
        
        pc_edges = count_edges(pc_graph)
        rai_edges = count_edges(rai_graph)
        
        assert pc_edges == rai_edges, \
            f"PC and RAI should find same skeleton on chain: PC={pc_edges}, RAI={rai_edges}"

    def test_pc_vs_rai_on_independent(self):
        np.random.seed(42)
        data = np.random.randn(500, 4)
        
        pc_graph = run_standalone_algorithm("pc_stable", data, alpha=0.05, n_jobs=1)
        rai_graph = run_standalone_algorithm("rai_stable", data, alpha=0.05, n_jobs=1)
        
        assert count_edges(pc_graph) == 0
        assert count_edges(rai_graph) == 0


class TestRunWithFile:
    def test_run_with_npy_file(self):
        W = get_chain_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=500, noise_std=1.0, seed=42)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
        np.save(temp_path, data)
        graph = run_standalone_algorithm("pc_stable", temp_path, alpha=0.05, n_jobs=1)
        assert has_edge(graph, 0, 1)
        assert has_edge(graph, 1, 2)
        os.unlink(temp_path)

    def test_run_with_pickle_file(self):
        W = get_chain_graph_3nodes()
        data = generate_linear_sem_data(W, n_samples=500, noise_std=1.0, seed=42)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        import pickle
        with open(temp_path, 'wb') as pf:
            pickle.dump(data, pf)
        graph = run_standalone_algorithm("pc_stable", temp_path, alpha=0.05, n_jobs=1)
        assert has_edge(graph, 0, 1)
        assert has_edge(graph, 1, 2)
        os.unlink(temp_path)


class TestAlgorithmNameNormalization:
    def test_case_insensitive_lookup(self):
        algo1 = create_standalone_algorithm("PC_STABLE")
        algo2 = create_standalone_algorithm("pc_stable")
        assert type(algo1) == type(algo2)

    def test_hyphen_to_underscore(self):
        algo1 = create_standalone_algorithm("pc-stable")
        algo2 = create_standalone_algorithm("pc_stable")
        assert type(algo1) == type(algo2)

    def test_space_to_underscore(self):
        algo1 = create_standalone_algorithm("pc stable")
        algo2 = create_standalone_algorithm("pc_stable")
        assert type(algo1) == type(algo2)


class TestErrorHandling:
    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unknown standalone algorithm"):
            run_standalone_algorithm("nonexistent", np.random.randn(100, 3))

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            run_standalone_algorithm("pc_stable", "missing_file.npy", n_jobs=1)
