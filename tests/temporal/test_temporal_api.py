"""Whydra contracts, known temporal graphs and adapter regression tests."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import whydra
from whydra.algorithms.graph_core import CIT, GeneralGraph
from whydra.algorithms.temporal import PCMCI, JPCMCIPlus, PCMCIPlus
from whydra.algorithms.temporal._ci import WhydraCI
from whydra.algorithms.temporal._data import coerce_dataset, make_frame
from whydra.algorithms.temporal.graphs import temporal_to_general_graph
from whydra.background_knowledge import BackgroundKnowledge, BKConflictWarning, BKPhase

MODELS = [PCMCI, PCMCIPlus, JPCMCIPlus]


def lagged_data(n=1600, seed=18):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, 3))
    for t in range(2, n):
        data[t, 1] += 0.9 * data[t - 1, 0]
        data[t, 2] += 0.8 * data[t - 2, 1]
    return data


@pytest.mark.parametrize(
    ("alias", "cls"),
    [
        ("pcmci", PCMCI),
        ("PCMCI+", PCMCIPlus),
        ("pcmci-plus", PCMCIPlus),
        ("j_pcmci_plus", JPCMCIPlus),
        ("J-PCMCI+", JPCMCIPlus),
        ("native_jpcmci_plus", JPCMCIPlus),
    ],
)
def test_registry(alias, cls):
    from whydra.algorithms import wrappers

    assert isinstance(whydra.create_standalone_algorithm(alias), cls)
    assert isinstance(whydra.create_standalone_algorithm(cls.__name__), cls)
    assert cls.__name__ in whydra.available_standalone_algorithms()
    assert getattr(whydra, cls.__name__) is getattr(wrappers, cls.__name__) is cls


@pytest.mark.parametrize("alias", ["orig_j_pcmci_plus", "jpcmci_plus"])
def test_legacy_jpcmci_aliases_are_unavailable(alias):
    with pytest.raises(ValueError, match="Unknown standalone algorithm"):
        whydra.create_standalone_algorithm(alias)


@pytest.mark.parametrize("cls", MODELS)
def test_known_lagged_chain_and_whydra_graph_contract(cls):
    model = cls(alpha=0.001, tau_max=2, n_jobs=1)
    graph = model.run(lagged_data())
    assert isinstance(graph, GeneralGraph)
    assert graph.graph.shape == (3, 3)
    assert model.lag_graph_[0, 1, 1] == "-->"
    assert model.lag_graph_[1, 2, 2] == "-->"
    assert model.lag_graph_[0, 2, 2] == ""
    assert graph.graph[0, 1] == 1
    assert graph.graph[1, 0] == -1
    assert model.lagged_graph_.graph.shape == (9, 9)
    assert model.node_names_ == ("X1", "X2", "X3")
    assert np.isfinite(model.p_matrix_).all()


@pytest.mark.parametrize("cls", MODELS)
def test_run_overrides_constructor_parameters_and_refits(cls, capsys):
    data = lagged_data(n=180)
    model = cls(tau_max=1, alpha=0.01)
    actual = model.run(data, tau_max=2, alpha=0.03, node_names=["a", "b", "c"])
    expected = cls(tau_max=2, alpha=0.03, node_names=["a", "b", "c"])
    np.testing.assert_array_equal(actual.graph, expected.run(data).graph)
    np.testing.assert_allclose(model.p_matrix_, expected.p_matrix_)
    model.run(data)
    assert model.lag_graph_.shape == (3, 3, 2)
    assert model.node_names_ == ("X1", "X2", "X3")
    assert capsys.readouterr().out == ""


def test_lagged_graph_keeps_self_lags_and_summary_merges_conflicting_directions():
    marks = np.full((2, 2, 3), "", dtype="<U3")
    marks[0, 0, 1] = marks[0, 1, 1] = marks[1, 0, 2] = "-->"
    summary = temporal_to_general_graph(marks, ["a", "b"])
    assert summary.graph.tolist() == [[0, 2], [2, 0]]
    expanded = temporal_to_general_graph(marks, ["a", "b"], graph_mode="lagged")
    assert [node.get_name() for node in expanded.nodes] == [
        "a",
        "b",
        "a[t-1]",
        "b[t-1]",
        "a[t-2]",
        "b[t-2]",
    ]
    assert expanded.graph[2, 0] == 1
    assert expanded.graph[0, 2] == -1
    assert expanded.graph[5, 0] == 1
    assert expanded.graph[0, 5] == -1


def test_unoriented_cpdag_and_conflict_marks_are_not_dropped():
    marks = np.full((3, 3, 1), "", dtype="<U3")
    marks[0, 1, 0] = marks[1, 0, 0] = "o-o"
    marks[1, 2, 0] = marks[2, 1, 0] = "x-x"
    g = temporal_to_general_graph(marks, ["a", "b", "c"])
    assert g.graph[0, 1] == g.graph[1, 0] == -1
    assert g.graph[1, 2] == g.graph[2, 1] == 2


def test_lagged_mode_and_npy_facade(tmp_path):
    data = lagged_data(n=160)
    path = tmp_path / "series.npy"
    np.save(path, data)
    expected = PCMCI(tau_max=2, graph_mode="lagged").run(data)
    actual = whydra.run_standalone_algorithm(
        "pcmci", path, tau_max=2, graph_mode="lagged"
    )
    np.testing.assert_array_equal(actual.graph, expected.graph)


def test_csv_facade_preserves_column_names_and_missing_observations(tmp_path):
    data = pd.DataFrame(lagged_data(n=160), columns=["a", "b", "c"])
    data.loc[10, "a"] = np.nan
    path = tmp_path / "series.csv"
    data.to_csv(path, index=False, sep=";")
    expected = PCMCI(tau_max=2).run(data)
    actual = whydra.run_standalone_algorithm("pcmci", path, sep=";", tau_max=2)
    assert [node.get_name() for node in actual.nodes] == ["a", "b", "c"]
    np.testing.assert_array_equal(actual.graph, expected.graph)
    with pytest.raises(ValueError, match="missing values"):
        whydra.run_standalone_algorithm("pc_stable", path, sep=";")


@pytest.mark.parametrize("method", ["fisherz", "chisq"])
def test_whydra_ci_is_reused_after_lag_and_missing_alignment(method):
    rng = np.random.default_rng(4)
    data = (
        rng.normal(size=(100, 3))
        if method == "fisherz"
        else rng.integers(0, 3, (100, 3)).astype(float)
    )
    data[8, 0] = np.nan
    frame = make_frame([coerce_dataset(data)], 2)
    ci = WhydraCI(method)
    ci.set_dataframe(frame)
    x, y, z = [(0, -2)], [(1, 0)], [(2, -1)]
    aligned, _, _ = frame.construct_array(x, y, z, tau_max=2, mask_type="xyz")
    expected = CIT(aligned.T, method=method)(0, 1, [2])
    assert ci.run_test(x, y, z, tau_max=2)[1] == pytest.approx(
        expected, rel=1e-12, abs=1e-14
    )


def test_missing_rows_keep_actual_time_distances_and_environments():
    data = np.column_stack(
        [np.arange(12, dtype=float), np.arange(12, dtype=float) + 10]
    )
    data[5, 0] = np.nan
    frame = make_frame([coerce_dataset(data), coerce_dataset(data + 100)], 1)
    aligned, _, _ = frame.construct_array(
        [(0, -1)], [(0, 0)], tau_max=1, mask_type="xyz"
    )
    np.testing.assert_array_equal(aligned[1] - aligned[0], np.ones(16))


@pytest.mark.parametrize("cls", MODELS)
@pytest.mark.parametrize("phase", list(BKPhase))
def test_forbidden_lag_never_becomes_a_backward_edge(cls, phase):
    bk = BackgroundKnowledge().forbid("X1[t-1]", "X2", phases=phase)
    model = cls(tau_max=2, background_knowledge=bk)
    model.run(lagged_data(n=350))
    assert model.lag_graph_[0, 1, 1] == ""
    assert model.lagged_graph_.graph[3, 1] == model.lagged_graph_.graph[1, 3] == 0
    assert any(report.phase == phase for report in bk.history)


@pytest.mark.parametrize("cls", MODELS)
def test_pre_search_required_lag_retains_real_ci_statistics(cls):
    bk = BackgroundKnowledge().require("X1[t-1]", "X2", phases=BKPhase.PRE_SEARCH)
    model = cls(alpha=1e-8, background_knowledge=bk)
    model.run(np.random.default_rng(81).normal(size=(150, 2)))
    assert model.lag_graph_[0, 1, 1] == "-->"
    assert model.p_matrix_[0, 1, 1] > model.alpha
    assert model.engine_.diagnostics


@pytest.mark.parametrize("cls", MODELS)
def test_later_requirement_never_invents_an_absent_lag(cls):
    bk = BackgroundKnowledge().require("X1[t-1]", "X2", phases=BKPhase.POST_ORIENTATION)
    model = cls(alpha=1e-8, background_knowledge=bk)
    with pytest.warns(BKConflictWarning, match="absent"):
        model.run(np.random.default_rng(81).normal(size=(150, 2)))
    assert model.lag_graph_[0, 1, 1] == ""


@pytest.mark.parametrize("cls", MODELS)
def test_contemporaneous_required_edge_and_forbidden_reverse(cls):
    bk = BackgroundKnowledge().require("X1", "X2", phases=BKPhase.PRE_SEARCH)
    model = cls(tau_min=0, tau_max=0, alpha=1e-8, background_knowledge=bk)
    graph = model.run(np.random.default_rng(81).normal(size=(100, 2)))
    assert graph.graph.tolist() == [[0, 1], [-1, 0]]
    assert model.p_matrix_[0, 1, 0] > model.alpha


def test_background_knowledge_type_and_time_order_validation():
    data = lagged_data(n=80)
    with pytest.raises(TypeError, match="BackgroundKnowledge"):
        PCMCI(background_knowledge={}).run(data)
    with pytest.raises(ValueError, match="past"):
        PCMCI(background_knowledge=BackgroundKnowledge().require("X2", "X1[t-1]")).run(
            data
        )
    with pytest.raises(ValueError, match="outside"):
        PCMCI(background_knowledge=BackgroundKnowledge().require("X1", "X2")).run(data)


def joint_data(seed=11):
    rng = np.random.default_rng(seed)
    weather = rng.normal(size=40)
    blocks = []
    for env in range(5):
        data = rng.normal(size=(40, 2))
        for t in range(1, len(data)):
            data[t, 0] += 0.5 * weather[t - 1] + 0.3 * env
            data[t, 1] += 0.7 * data[t - 1, 0]
        blocks.append(
            pd.DataFrame(
                {
                    "a": data[:, 0],
                    "b": data[:, 1],
                    "weather": weather,
                    "altitude": float(env),
                }
            )
        )
    return blocks


@pytest.mark.parametrize("jobs", [1, 2])
def test_joint_contexts_dummies_and_input_isolation(jobs):
    blocks = joint_data()
    snapshots = [block.copy(deep=True) for block in blocks]
    model = JPCMCIPlus(
        temporal_context=["weather"],
        spatial_context=["altitude"],
        time_dummy=True,
        n_jobs=jobs,
        max_conds_dim=2,
    )
    graph = model.run(blocks)
    assert graph.graph.shape == (4, 4)
    assert model.lag_graph_.shape == (6, 6, 2)
    assert model.node_names_[-2:] == ("__time_dummy__", "__dataset_dummy__")
    assert model.node_roles_[4] == "time_dummy"
    assert any(model.observed_context_parents_.values())
    for block, snapshot in zip(blocks, snapshots, strict=False):
        pd.testing.assert_frame_equal(block, snapshot)
    for target in (3, 4, 5):
        assert "-->" not in model.lag_graph_[:, target, :]
    assert (model.p_matrix_ <= 1).all()


def test_parallel_matches_sequential_with_contexts_missing_data_and_bk():
    blocks = joint_data()
    blocks[1].loc[10, "a"] = np.nan
    results = []
    for jobs in (1, 2):
        bk = BackgroundKnowledge().require("a[t-1]", "b", phases=BKPhase.PRE_SEARCH)
        model = JPCMCIPlus(
            temporal_context=["weather"],
            spatial_context=["altitude"],
            n_jobs=jobs,
            background_knowledge=bk,
            max_conds_dim=2,
        )
        model.run(blocks)
        results.append(model)
    np.testing.assert_array_equal(results[0].lag_graph_, results[1].lag_graph_)
    np.testing.assert_allclose(results[0].p_matrix_, results[1].p_matrix_, atol=1e-12)
    np.testing.assert_allclose(
        results[0].val_matrix_, results[1].val_matrix_, atol=1e-12
    )
    assert results[0].parents_ == results[1].parents_
    assert results[0].sepsets_ == results[1].sepsets_


def test_unequal_environments_and_column_reordering():
    blocks = [block[["a", "b"]] for block in joint_data()]
    blocks[1] = blocks[1].iloc[:-5][["b", "a"]]
    model = JPCMCIPlus()
    graph = model.run(dict(enumerate(blocks)))
    assert graph.graph.shape == (2, 2)
    for data in (blocks, tuple(blocks), dict(enumerate(blocks))):
        from_facade = whydra.run_standalone_algorithm("j_pcmci_plus", data)
        np.testing.assert_array_equal(from_facade.graph, graph.graph)
    with pytest.raises(ValueError, match="aligned"):
        JPCMCIPlus(time_dummy=True).run(blocks)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"tau_max": -1}, "tau_min"),
        ({"tau_max": True}, "integers"),
        ({"alpha": float("nan")}, "alpha"),
        ({"alpha": 1}, "alpha"),
        ({"n_jobs": 0}, "n_jobs"),
        ({"n_jobs": True}, "n_jobs"),
        ({"node_names": ["a"]}, "node_names"),
        ({"graph_mode": "bad"}, "graph_mode"),
        ({"indep_test": "unknown"}, "indep_test"),
        ({"typo": True}, "Unsupported"),
        ({"node_names": ["a", "a", "b"]}, "node_names"),
        ({"node_names": ["a", "a[t-1]", "b"]}, "collide"),
    ],
)
def test_invalid_configuration(params, match):
    with pytest.raises((ValueError, TypeError), match=match):
        PCMCI(**params).run(lagged_data(n=40))


def test_invalid_time_coordinates_and_infinite_or_constant_data():
    data = lagged_data(n=30)
    for values, match in [
        (np.ones((30, 3)), "Constant"),
        (np.full((30, 3), np.inf), "Infinite"),
        (np.ones((3, 3)), "four rows"),
    ]:
        with pytest.raises(ValueError, match=match):
            PCMCI().run(values)
    index = pd.date_range("2026-01-01", periods=31).delete(10)
    with pytest.raises(ValueError, match="regularly spaced"):
        PCMCI().run(pd.DataFrame(data, index=index))
    with pytest.raises(ValueError, match="one alpha"):
        PCMCIPlus(pc_alpha=0.2).run(data)


def test_joint_scalar_ci_requires_disabling_vector_dummies():
    data = np.random.default_rng(42).normal(size=(3, 40, 2))
    with pytest.raises(ValueError, match="dummy"):
        JPCMCIPlus(indep_test="fisherz").run(data)
    options = {"indep_test": "fisherz", "dataset_dummy": False}
    expected = JPCMCIPlus(**options).run(data)
    actual = whydra.run_standalone_algorithm("j_pcmci_plus", data, **options)
    assert actual.graph.shape == (2, 2)
    np.testing.assert_array_equal(actual.graph, expected.graph)


def test_import_and_execution_without_external_temporal_packages():
    code = """
import importlib.abc, sys
class BlockExternal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'time_series_causal_discovery', 'causallearn'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BlockExternal())
import numpy as np
import whydra
names = whydra.available_standalone_algorithms()
assert {'StandalonePCMCI', 'StandalonePCMCIPlus', 'StandaloneJPCMCIPlus'} <= set(names)
assert 'whydra.algorithms.temporal' not in sys.modules
assert 'pandas' not in sys.modules
data = np.random.default_rng(3).normal(size=(60, 3))
for name in ('pcmci', 'pcmci_plus', 'j_pcmci_plus'):
    assert whydra.run_standalone_algorithm(name, data).graph.shape == (3, 3)
"""
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
