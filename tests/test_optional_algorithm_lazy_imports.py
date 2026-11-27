"""Import boundaries for score and optional neural algorithms.

Backend doubles below test dependency boundaries only; they are never timings
or claims about training with PyTorch.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def fresh_python(source):
    root = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=root,
        env=dict(os.environ, PYTHONPATH=root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_score_namespace_and_local_score_do_not_load_ges_or_graphs():
    fresh_python("""
        import sys
        import pickle
        from whydra.algorithms import score_based
        assert set(score_based.__all__) <= set(dir(score_based))
        assert 'numpy' not in sys.modules
        from whydra.algorithms.score_based import LocalScoreCache, local_parent_score
        from whydra.algorithms.score_based.scores import LocalScoreCache as implementation
        assert LocalScoreCache is implementation
        assert pickle.loads(pickle.dumps(LocalScoreCache)) is implementation
        import numpy as np
        x = np.arange(12., dtype=float)
        y = .5*x + np.array([.3, -.2, .1, .4] * 3)
        data = np.column_stack((x, y))
        actual = local_parent_score(data, 1, [0], 'local_score_BIC')
        design = np.column_stack((np.ones(len(x)), x))
        residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        expected = -len(x)*np.log(residual @ residual / len(x)) - 2*np.log(len(x))
        assert actual == expected
        cache = LocalScoreCache(data, 'local_score_BIC')
        assert cache.score(1, [0]) == actual
        for name in ('whydra.algorithms.score_based.ges', 'whydra.algorithms.score_based.pdag',
                     'whydra.algorithms.graph_core.general_graph', 'pandas', 'scipy', 'joblib'):
            assert name not in sys.modules, name
    """)


def test_score_public_ges_identity_and_small_search():
    fresh_python("""
        import numpy as np
        from whydra.algorithms.score_based import ges_search
        from whydra.algorithms.score_based.ges import ges_search as implementation
        assert ges_search is implementation
        data = np.random.default_rng(310).normal(size=(160, 3))
        data[:, 1] += data[:, 0]
        graph, score = ges_search(data, 'local_score_BIC', ['a', 'b', 'c'])
        direct, direct_score = implementation(data, 'local_score_BIC', ['a', 'b', 'c'])
        assert np.array_equal(graph.graph, direct.graph) and score == direct_score
        assert graph.graph_kind == 'cpdag'
    """)


def test_dag_constructor_keeps_dependency_error_and_delays_unrelated_algorithms():
    fresh_python("""
        import importlib.abc
        import sys
        class MissingBackend(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] == 'torch':
                    raise ModuleNotFoundError('Test backend unavailable', name=fullname)
        sys.meta_path.insert(0, MissingBackend())
        from whydra import DagGNN
        assert 'torch' not in sys.modules
        try:
            DagGNN()
        except ImportError as exc:
            assert 'PyTorch' in str(exc)
            assert exc.__cause__ is not None
        else:
            raise AssertionError('Selected algorithm must explain its missing backend')
        assert 'scipy.stats' not in sys.modules
    """)


def test_dag_runner_defers_training_imports_and_keeps_explicit_export_access():
    fresh_python("""
        import importlib
        import sys
        from types import ModuleType, SimpleNamespace
        import numpy as np
        sys.modules['torch'] = ModuleType('torch')
        runner = importlib.import_module('whydra.algorithms.standalone.dag_gnn.runner')
        assert 'torch.optim' not in sys.modules and 'tqdm' not in sys.modules
        assert 'whydra.algorithms.standalone.dag_gnn.modules' not in sys.modules
        assert 'whydra.algorithms.standalone.dag_gnn.utils' not in sys.modules
        assert set(runner._TRAINING_EXPORTS) <= set(dir(runner))
        model = object()
        utility = object()
        calls = []
        def load(name, package=None):
            calls.append(name)
            if name == '.modules':
                return SimpleNamespace(**{key: model for key in
                    ('MLPEncoder', 'MLPDecoder', 'MLPDiscreteEncoder', 'MLPDiscreteDecoder')})
            if name == '.utils':
                return SimpleNamespace(**{key: utility for key in
                    ('nll_gaussian', 'nll_categorical', 'kl_gaussian_sem', 'matrix_poly')})
            return SimpleNamespace(trange=range)
        runner.import_module = load
        assert runner.MLPEncoder is model
        assert runner.MLPEncoder is model and calls == ['.modules']
        try:
            runner.train_dag_gnn(np.ones((3, 2)), data_type='invalid', device='cpu')
        except ValueError as exc:
            assert 'data_type' in str(exc)
        else:
            raise AssertionError('Original argument validation must still run')
        assert all(name in runner.__dict__ for name in runner._TRAINING_EXPORTS)
        assert runner.matrix_poly is utility
    """)
