import numpy as np
import pytest
from scipy.stats import pearsonr

from whydra.algorithms.temporal import StandalonePCMCI, StandalonePCMCIPlus
from whydra.algorithms.temporal._engine._joint_ci import NativeParCorr
from whydra.algorithms.temporal._engine._joint_data import JointDataFrame
from whydra.algorithms.temporal._engine.pcmci import PCMCIPipeline
from whydra.algorithms.temporal._engine.pcmci_plus import PCMCIPlusPipeline
from whydra.background_knowledge import BackgroundKnowledge, BKPhase


@pytest.fixture
def lagged_design():
    rng = np.random.default_rng(1871)
    rows = rng.normal(size=(64, 5))
    basis, _ = np.linalg.qr(rows - rows.mean(axis=0))
    weights = np.array([0.55, -0.265, 0.10, -0.04])
    data = rng.normal(size=(66, 5))
    data[1:-1, :4] = basis[:, :4]
    data[2:, 4] = basis[:, :4] @ weights + basis[:, 4] * np.sqrt(1 - weights @ weights)
    return data


def test_mci_with_two_time_origins_matches_partial_regression():
    rng = np.random.default_rng(382)
    data = rng.normal(size=(180, 4))
    data[1:, 0] += 0.8 * data[:-1, 2]
    data[2:, 1] += data[:-2, 0] - 0.7 * data[1:-1, 3]
    pipeline = PCMCIPipeline(JointDataFrame(data), NativeParCorr())
    allowed = {1: {(0, -2): "-?>", (2, -1): "-?>"}}
    parents = {0: [(2, -1), (3, -2)], 1: [(0, -2), (3, -1), (3, -2)]}
    result = pipeline.run_mci(
        tau_min=1,
        tau_max=2,
        link_assumptions=allowed,
        parents=parents,
        max_conds_px=1,
        max_conds_py=2,
    )

    times = np.arange(4, len(data))
    endpoints = np.column_stack([data[times - 2, 0], data[times, 1]])
    design = np.column_stack(
        [np.ones(len(times)), data[times - 1, 3], data[times - 3, 2]]
    )
    fitted = design @ np.linalg.lstsq(design, endpoints, rcond=None)[0]
    expected = np.corrcoef((endpoints - fitted).T)[0, 1]
    assert result["val_matrix"][0, 1, 2] == pytest.approx(expected)
    assert result["graph"][0, 1, 2] == "-->"
    assert 0 < result["p_matrix"][2, 1, 1] < 1
    assert result["p_matrix"][3, 1, 1] == 1


@pytest.mark.parametrize("algorithm", [StandalonePCMCI, StandalonePCMCIPlus])
def test_lagged_family_threshold_after_adjustment(lagged_design, algorithm):
    names = ["force", "load", "temp", "offset", "response"]
    knowledge = BackgroundKnowledge().require(
        "offset[t-1]", "response", phases=BKPhase.PRE_SEARCH
    )
    model = algorithm(
        tau_min=1,
        pc_alpha=0.05,
        alpha=0.05,
        fdr_method="fdr_bh",
        node_names=names,
        background_knowledge=knowledge,
        max_conds_dim=0,
        max_conds_px=0,
        max_conds_py=0,
        link_assumptions={4: {(i, -1): "-?>" for i in range(4)}},
    )
    model.run(lagged_design)
    observed = [
        pearsonr(lagged_design[1:-1, i], lagged_design[2:, 4]).pvalue for i in range(4)
    ]
    expected = np.minimum(1.0, np.array(observed) * [4, 2, 4 / 3, 1])
    np.testing.assert_allclose(model.p_matrix_[:4, 4, 1], expected, rtol=1e-10)
    assert observed[1] < 0.05 < expected[1]
    assert expected[3] > 0.5
    assert model.lag_graph_[:4, 4, 1].tolist() == ["-->", "", "", "-->"]


def test_correction_adjusts_only_the_listed_hypotheses():
    engine = PCMCIPipeline(JointDataFrame(np.eye(3)), NativeParCorr())
    pvalues = np.full((3, 3, 3), 0.002)
    coordinates = [(0, 0, 2), (1, 0, 1), (2, 1, 2)]
    assumptions = {
        0: {(0, -2): "-?>", (1, -1): "-?>", (2, 0): "-?>"},
        1: {(2, -2): "-?>"},
    }
    for link, probability in zip(coordinates, [0.019, 0.020, 0.8], strict=True):
        pvalues[link] = probability

    adjusted = engine.get_corrected_pvalues(
        pvalues, tau_max=2, link_assumptions=assumptions
    )
    expected = pvalues.copy()
    for link, probability in zip(coordinates, [0.030, 0.030, 0.8], strict=True):
        expected[link] = probability
    np.testing.assert_allclose(adjusted, expected)
    assert pvalues[0, 0, 2] == 0.019


@pytest.mark.parametrize("reversed_order", [False, True])
def test_alpha_grid_recovers_the_same_two_lagged_predictors(
    lagged_design, reversed_order
):
    grid = [0.0001, 0.05, 0.9]
    if reversed_order:
        grid.reverse()
    options = {
        "pc_alpha": grid,
        "link_assumptions": {4: {(i, -1): "-?>" for i in range(4)}},
        "tau_min": 1,
        "tau_max": 1,
        "max_conds_dim": 0,
    }
    frame = JointDataFrame(lagged_design)
    pc = PCMCIPipeline(frame, NativeParCorr())
    plus = PCMCIPlusPipeline(frame, NativeParCorr())
    estimated = pc.run_pc_stable(**options)
    result = plus.run_pcmciplus(max_conds_px=0, max_conds_py=0, **options)

    assert estimated[4] == [(0, -1), (1, -1)]
    assert pc.iterations[4]["optimal_pc_alpha"] == 0.05
    assert plus.all_lagged_parents == estimated
    assert result["optimal_alpha"] == 0.05
    assert result["alpha_scores"][0.05] < min(
        result["alpha_scores"][a] for a in (0.0001, 0.9)
    )
    np.testing.assert_array_equal(
        np.flatnonzero(result["graph"][:, 4, 1] == "-->"), [0, 1]
    )


def test_zero_lag_statistics_and_intervals_always_come_from_the_same_direction():
    model = PCMCIPipeline(JointDataFrame(np.eye(3)), NativeParCorr())
    prob = np.array([[1, 0.2, 0.7], [0.4, 1, 0.6], [0.7, 0.1, 1]])[:, :, None]
    effect = np.arange(9).reshape(3, 3, 1) + 0.25
    bounds = np.stack((effect - 0.2, effect + 0.3), axis=-1)
    links = {1: {(0, 0): "o?o", (2, 0): "-?>"}, 2: {(0, 0): "o?o"}}

    result = model.symmetrize_p_and_val_matrix(prob, effect, links, bounds)

    for src, dst in [(1, 0), (0, 2), (2, 1)]:
        for a, b in [(src, dst), (dst, src)]:
            assert result["p_matrix"][a, b, 0] == prob[src, dst, 0]
            assert result["val_matrix"][a, b, 0] == effect[src, dst, 0]
            np.testing.assert_array_equal(result["conf_matrix"][a, b], bounds[src, dst])


def test_scoring_keeps_the_collider_while_extending_a_mixed_graph():
    graph = np.full((4, 4, 3), "", dtype="<U3")
    graph[0, 2, 0] = graph[1, 2, 0] = "-->"
    graph[2, 0, 0] = graph[2, 1, 0] = "<--"
    graph[3, 2, 0] = graph[2, 3, 0] = "o-o"
    graph[3, 1, 2] = "-->"
    model = PCMCIPlusPipeline(JointDataFrame(np.eye(4)), NativeParCorr())
    parents = model._scoring_parents(graph)
    edges = {(i, lag, j) for j, family in parents.items() for i, lag in family}
    assert edges == {(0, 0, 2), (1, 0, 2), (2, 0, 3), (3, -2, 1)}


@pytest.mark.parametrize("cycle_mark", ["-->", "x-x"])
def test_scoring_rejects_inconsistent_contemporaneous_graphs(cycle_mark):
    graph = np.full((3, 3, 1), "", dtype="<U3")
    for i in range(3):
        j = (i + 1) % 3
        graph[i, j, 0] = cycle_mark
        graph[j, i, 0] = "<--" if cycle_mark == "-->" else "x-x"
    model = PCMCIPlusPipeline(JointDataFrame(np.eye(3)), NativeParCorr())
    with pytest.raises(ValueError, match="DAG extension"):
        model._scoring_parents(graph)


@pytest.mark.parametrize("algorithm", [StandalonePCMCI, StandalonePCMCIPlus])
def test_unknown_adjustment_does_not_start_screening(
    algorithm, lagged_design, monkeypatch
):
    def unexpected_call(*args, **kwargs):
        pytest.fail("CI was invoked before validating the adjustment")

    monkeypatch.setattr(NativeParCorr, "run_test", unexpected_call)
    with pytest.raises(ValueError, match=r"fdr_method|FDR"):
        algorithm(fdr_method="holm").run(lagged_design)
