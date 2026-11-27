"""Public imports must defer optional runtimes without changing returned objects."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def fresh_python(source):
    root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=root)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_plain_package_import_has_no_numerical_or_dataset_side_effects():
    fresh_python("""
        import sys
        import whydra
        assert set(whydra.__all__) <= set(dir(whydra))
        assert '__version__' in dir(whydra)
        for name in ('numpy', 'pandas', 'scipy', 'joblib', 'networkx',
                     'whydra.data', 'whydra.algorithms.wrappers'):
            assert name not in sys.modules, name
        assert isinstance(whydra.__version__, str)
        try:
            whydra.no_such_export
        except AttributeError:
            pass
        else:
            raise AssertionError('Unknown exports must raise AttributeError')
    """)


def test_public_exports_retain_identity_pickle_and_star_imports():
    fresh_python("""
        import pickle
        import sys
        import whydra
        from whydra import ParallelPCStable
        from whydra.algorithms import ParallelPCStable as algorithm_class
        from whydra.algorithms.wrappers import ParallelPCStable as implementation
        assert ParallelPCStable is algorithm_class is implementation
        assert whydra.ParallelPCStable is ParallelPCStable
        assert pickle.loads(pickle.dumps(ParallelPCStable)) is ParallelPCStable
        restored = pickle.loads(pickle.dumps(ParallelPCStable(n_jobs=1)))
        assert type(restored) is ParallelPCStable and restored.n_jobs == 1
        assert 'whydra.algorithms.standalone.parallel_pc_stable' not in sys.modules
        assert 'whydra.algorithms.standalone.dag_gnn.runner' not in sys.modules
        assert 'pandas' not in sys.modules
        namespace = {}
        exec('from whydra import *', namespace)
        assert all(namespace[name] is getattr(whydra, name) for name in whydra.__all__)
        from whydra.data import LoaderRegistry
        assert set(LoaderRegistry.available()) >= {'bnlearn', 'lucas', 'tetrad'}
    """)


def test_public_pc_and_native_kernel_return_the_same_endpoint_matrix():
    fresh_python("""
        import numpy as np
        import sys
        from whydra import ParallelPCStable, run_standalone_algorithm
        from whydra.algorithms.graph_core import CausalGraph, CIT
        from whydra.algorithms.standalone.parallel_pc_stable import pc_stable
        data = np.random.default_rng(48).normal(size=(240, 4))
        data[:, 1] += data[:, 0]
        data[:, 2] += data[:, 1]
        names = [f'X{i+1}' for i in range(4)]
        native_input = CausalGraph(4, names)
        native_input.set_ind_test(CIT(data, method='fisherz'))
        native = pc_stable(native_input, alpha=.05, n_jobs=1,
                           verbose=False, show_progress=False).G
        public = ParallelPCStable(n_jobs=1).run(data)
        by_name = run_standalone_algorithm('parallel_pc_stable', data, n_jobs=1)
        np.testing.assert_array_equal(public.graph, native.graph)
        np.testing.assert_array_equal(by_name.graph, native.graph)
        for name in ('pandas', 'scipy.stats', 'joblib', 'networkx',
                     'whydra.algorithms.standalone.fci_algo_v2'):
            assert name not in sys.modules, name
    """)


def test_optional_algorithm_failures_are_deferred_until_selected():
    fresh_python("""
        import importlib.abc
        import sys
        class MissingOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith('whydra.algorithms.standalone.dag_gnn'):
                    raise ModuleNotFoundError('Simulated missing dependency', name=fullname)
        sys.meta_path.insert(0, MissingOptional())
        from whydra import ParallelPCStable, DagGNN
        ParallelPCStable(n_jobs=1)
        try:
            DagGNN()
        except ImportError as exc:
            assert 'PyTorch' in str(exc)
            assert isinstance(exc.__cause__, ModuleNotFoundError)
        else:
            raise AssertionError('Selected optional algorithm should explain its dependency')
        try:
            DagGNN(typo=True)
        except TypeError as exc:
            assert 'Unsupported DagGNN parameters: typo' in str(exc)
        else:
            raise AssertionError('Unknown parameters must not depend on optional imports')
    """)


def test_graph_annotations_and_networkx_conversion_still_resolve():
    fresh_python("""
        import sys
        import typing
        from whydra.evaluation import graph_utils, get_adj_matrix
        from whydra.algorithms.graph_core import GeneralGraph, Node
        assert get_adj_matrix is graph_utils.get_adj_matrix
        assert 'whydra.evaluation.metrics' not in sys.modules
        assert 'networkx' not in sys.modules
        hints = typing.get_type_hints(graph_utils.convert_general_graph_to_nx)
        import networkx as nx
        assert hints['return'] is nx.DiGraph
        assert typing.get_type_hints(graph_utils.result_to_nx)['return'] is nx.DiGraph
        graph = GeneralGraph([Node('a'), Node('b')])
        graph.add_directed_edge(graph.nodes[0], graph.nodes[1])
        exported = graph_utils.convert_general_graph_to_nx(graph)
        assert list(exported.edges) == [('a', 'b')]
        restored = graph_utils.from_networkx(exported)
        import numpy as np
        np.testing.assert_array_equal(restored.graph, graph.graph)
    """)


def test_dataframe_subclasses_and_named_csv_keep_column_names(tmp_path):
    import numpy as np
    import pandas as pd

    from whydra import ParallelPCStable, run_standalone_algorithm

    class NamedFrame(pd.DataFrame):
        pass

    data = NamedFrame(
        np.random.default_rng(9).normal(size=(100, 3)),
        columns=["left", "middle", "right"],
    )
    direct = ParallelPCStable(n_jobs=1).run(data)
    assert [node.get_name() for node in direct.nodes] == list(data.columns)
    path = tmp_path / "named.csv"
    data.to_csv(path, index=False)
    loaded = run_standalone_algorithm("parallel_pc_stable", path, n_jobs=1)
    assert [node.get_name() for node in loaded.nodes] == list(data.columns)
    np.testing.assert_array_equal(loaded.graph, direct.graph)
