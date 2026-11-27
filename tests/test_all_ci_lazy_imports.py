"""Every public CI family should load only the runtime used by its search."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

FAMILIES = (
    ("StandaloneFCIStable", "fci_algo_v2", "fci_stable_v2"),
    ("StandaloneFCIStableCached", "fci_algo_cash", "fci_stable"),
    ("StandaloneFCIStableParallelCached", "fci_algo_paralel_cash", "fci_stable"),
    ("StandaloneCFCIStable", "cfci_algo", "cfci_stable"),
    ("StandaloneCFCIStableParallelCached", "cfci_algo_paralel_cash", "cfci_stable"),
    ("StandaloneRFCIStable", "rfci_algo", "rfci_stable"),
    ("StandaloneGFCIStable", "gfci_algo", "gfci_stable"),
    ("StandaloneFCIPlusStable", "fci_plus_algo", "fci_plus_stable"),
    (
        "StandaloneFCIPlusStableParallelCached",
        "fci_plus_algo_paralel_cash",
        "fci_plus_stable",
    ),
    ("StandaloneRAIStable", "rai_algo", "rai_stable"),
)


def fresh(source):
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


@pytest.mark.parametrize(("class_name", "module_name", "kernel_name"), FAMILIES)
def test_each_fisher_family_is_lazy_and_matches_its_kernel(
    class_name, module_name, kernel_name
):
    fresh(f"""
        from importlib import import_module
        import sys
        import warnings
        import numpy as np
        import whydra
        cls = getattr(whydra, {class_name!r})
        module_path = 'whydra.algorithms.standalone.' + {module_name!r}
        assert module_path not in sys.modules
        data = np.random.default_rng(831).normal(size=(240, 6))
        data[:, 1] += .9 * data[:, 0]
        data[:, 3] += .8 * data[:, 1]
        extra = {{'start_from_complete_graph': True}} if {class_name!r} == 'StandaloneGFCIStable' else {{}}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            public = cls(n_jobs=1, **extra).run(data)
        for forbidden in ('pandas', 'scipy.stats', 'networkx', 'joblib', 'tqdm',
                          'torch', 'whydra.data'):
            assert forbidden not in sys.modules, forbidden
        assert module_path in sys.modules
        if {class_name!r} == 'StandaloneFCIPlusStableParallelCached':
            assert 'whydra.algorithms.standalone.rfci_algo' not in sys.modules
            assert 'whydra.algorithms.standalone.fci_algo_v2' not in sys.modules
        if {class_name!r} == 'StandaloneGFCIStable':
            assert 'whydra.algorithms.score_based' not in sys.modules
        from whydra.algorithms.graph_core import CausalGraph, CIT
        names = [f'X{{i+1}}' for i in range(data.shape[1])]
        kernel = getattr(import_module(module_path), {kernel_name!r})
        if {class_name!r} == 'StandaloneRAIStable':
            native = kernel(data, alpha=.05, n_jobs=1, ci_test=CIT(data), node_names=names)
        else:
            cg = CausalGraph(data.shape[1], names)
            cg.set_ind_test(CIT(data))
            native = kernel(cg, data, alpha=.05, n_jobs=1, **extra)
        if hasattr(native, 'G'):
            native = native.G
        native_matrix = native.graph if hasattr(native, 'graph') else native
        assert np.array_equal(public.graph, native_matrix)
    """)


def test_fci_plus_keeps_lazy_legacy_demo_exports():
    fresh("""
        import sys
        from whydra.algorithms.standalone import fci_plus_algo_paralel_cash as module
        for name in ('rfci_stable', 'fci_stable_v2', 'make_collider_data'):
            assert name in dir(module)
        assert 'whydra.algorithms.standalone.rfci_algo' not in sys.modules
        assert 'whydra.algorithms.standalone.fci_algo_v2' not in sys.modules
        from whydra.algorithms.standalone.rfci_algo import rfci_stable, make_collider_data
        from whydra.algorithms.standalone.fci_algo_v2 import fci_stable_v2
        assert module.rfci_stable is rfci_stable
        assert module.make_collider_data is make_collider_data
        assert module.fci_stable_v2 is fci_stable_v2
        namespace = {}
        exec('from whydra.algorithms.standalone.fci_plus_algo_paralel_cash import *', namespace)
        assert namespace['rfci_stable'] is rfci_stable
        assert namespace['fci_stable_v2'] is fci_stable_v2
    """)


def test_gfci_lazy_score_exports_retain_identity_and_runtime_override():
    fresh("""
        import sys
        from whydra.algorithms.standalone import gfci_algo as module
        assert 'whydra.algorithms.score_based' not in sys.modules
        assert {'ges_search', 'local_parent_score', '_local_parent_score'} <= set(dir(module))
        from whydra.algorithms.score_based import ges_search, local_parent_score
        assert module.ges_search is ges_search
        assert module.local_parent_score is local_parent_score
        assert module._local_parent_score is local_parent_score
        namespace = {}
        exec('from whydra.algorithms.standalone.gfci_algo import *', namespace)
        assert namespace['ges_search'] is ges_search
        assert namespace['local_parent_score'] is local_parent_score
        import numpy as np
        from whydra import StandaloneGFCIStable
        from whydra.algorithms.graph_core import GeneralGraph, Node
        calls = []
        def custom_score_search(data, score_name, node_names, **kwargs):
            calls.append(score_name)
            return GeneralGraph([Node(name) for name in node_names]), 0.
        module.ges_search = custom_score_search
        assert module._score_implementation('ges_search') is custom_score_search
        data = np.random.default_rng(910).normal(size=(150, 4))
        StandaloneGFCIStable(n_jobs=1).run(data)
        assert len(calls) == 1
        StandaloneGFCIStable(n_jobs=1, start_from_complete_graph=True).run(data)
        assert len(calls) == 1
    """)


def test_shared_backend_snapshot_preserves_serial_coarse_and_mixed_labels():
    from whydra.algorithms.standalone.ci_adapters import observed_phase_backend

    before = {"batches": {"serial": 7, "processes": 2}}
    assert observed_phase_backend(before, before) == "serial"
    after = {"batches": {"serial": 8, "processes": 3}}
    assert observed_phase_backend(after, before) == "processes"
    assert observed_phase_backend(after, before, extra=["threads"]) == "hybrid"
    assert (
        observed_phase_backend(before, before, {"coarse_batches": {"processes": 1}})
        == "processes"
    )
