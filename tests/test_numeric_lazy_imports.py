"""Numerical and worker paths must not initialize unrelated runtimes."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest


def fresh_python(source):
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_worker_import_does_not_initialize_ci_or_graph_objects():
    fresh_python("""
        import sys
        from whydra.algorithms.graph_core.worker_data import initialize_worker
        from whydra.algorithms import graph_core
        assert callable(initialize_worker)
        assert 'CIT' in dir(graph_core)
        assert 'whydra.algorithms.graph_core.independence_tests' not in sys.modules
        assert 'whydra.algorithms.graph_core.causal_graph' not in sys.modules
        assert 'pandas' not in sys.modules
        assert 'scipy.stats' not in sys.modules
    """)


def test_fisher_and_silent_serial_pc_skip_unneeded_imports():
    fresh_python("""
        import sys
        import numpy as np
        from whydra import ParallelPCStable
        from whydra.algorithms.graph_core import CIT
        data = np.random.default_rng(123).normal(size=(120, 5))
        cit = CIT(data)
        assert 0 <= cit(0, 1, ()) <= 1
        graph = ParallelPCStable(n_jobs=1, show_progress=False).run(data)
        assert graph.graph.shape == (5, 5)
        for name in ('pandas', 'scipy.stats', 'joblib', 'tqdm.auto'):
            assert name not in sys.modules, name
        categorical = np.array([[0, 0]] * 4 + [[0, 1]] * 2 + [[1, 0]] + [[1, 1]] * 5)
        discrete = CIT(categorical, method='chisq')
        assert 0 <= discrete(0, 1, ()) <= 1
        assert 'scipy.stats' in sys.modules
    """)


def test_graph_core_exports_keep_class_identity_and_pickle_support():
    fresh_python("""
        import pickle
        from whydra.algorithms import graph_core
        from whydra.algorithms.graph_core.nodes import Node
        assert graph_core.Node is Node
        assert graph_core.Node is graph_core.Node
        assert pickle.loads(pickle.dumps(Node)) is Node
        assert set(graph_core.__all__) <= set(dir(graph_core))
        for name in graph_core.__all__:
            assert hasattr(graph_core, name), name
        try:
            graph_core.unavailable_export
        except AttributeError:
            pass
        else:
            raise AssertionError('Unknown exports must raise AttributeError')
    """)


@pytest.mark.parametrize(
    "data",
    [
        np.array([[1.0, np.nan], [2.0, 3.0]]),
        np.array([[1 + 0j, 2 + np.nan * 1j], [3, 4]]),
        np.array([[1, None], [2, 3]], dtype=object),
        np.array([["2025-01-01", "NaT"]], dtype="datetime64[D]"),
        np.array([[1, "NaT"]], dtype="timedelta64[D]"),
    ],
)
def test_missing_values_keep_column_diagnostics(data):
    from whydra.algorithms.graph_core.validation import validate_data

    with pytest.raises(ValueError, match=r"missing values.*columns \[1\]"):
        validate_data(data)


def test_pandas_nullable_values_keep_validation_contract():
    import pandas as pd

    from whydra.algorithms.graph_core.validation import validate_data

    frame = pd.DataFrame(
        {
            "a": pd.Series([1, 2], dtype="Int64"),
            "b": pd.Series([pd.NA, 3], dtype="Int64"),
        }
    )
    with pytest.raises(ValueError, match=r"missing values.*columns \[1\]"):
        validate_data(frame)
    valid = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    np.testing.assert_array_equal(validate_data(valid, numeric=True), valid.to_numpy())
