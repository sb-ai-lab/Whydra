from decimal import Decimal
from itertools import combinations

import numpy as np
import pytest
from scipy.linalg import hadamard

from whydra.algorithms.temporal._bk import TemporalKnowledge
from whydra.algorithms.temporal._engine._joint_ci import NativeParCorr
from whydra.algorithms.temporal._engine._joint_data import JointDataFrame
from whydra.algorithms.temporal._engine.parallel import ParallelJPCMCIplus
from whydra.background_knowledge import BackgroundKnowledge, BKPhase


class PairEvidence:
    significance = "analytic"

    def __init__(self, by_query):
        self.by_query = by_query

    def set_dataframe(self, dataframe):
        self.dataframe = dataframe

    def run_test(self, X, Y, Z, alpha_or_thres, **kwargs):
        statistic, probability = self.by_query.get(
            (X[0], Y[0][0], tuple(Z)), (0.48, 0.01)
        )
        return statistic, probability, probability <= alpha_or_thres


def joint_engine(samples, test, jobs):
    names = [f"station{i}" for i in range(samples.shape[1])]
    return ParallelJPCMCIplus(
        dict.fromkeys(range(len(names)), "system"),
        dataframe=JointDataFrame(samples, var_names=names),
        cond_ind_test=test,
        n_jobs=jobs,
    )


def estimate_skeleton(engine, candidates, depth, last_subset=None):
    return engine._pcalg_skeleton(
        initial_graph=candidates,
        lagged_parents={},
        mode="contemp_conds",
        pc_alpha=0.05,
        tau_min=0,
        tau_max=candidates.shape[2] - 1,
        max_conds_dim=depth,
        max_combinations=last_subset,
        max_conds_py=None,
        max_conds_px=None,
        max_conds_px_lagged=None,
    )


@pytest.mark.parametrize("jobs", [1, 2])
def test_path_recovered_from_orthogonal_innovations(jobs):
    route = [4, 1, 3, 0, 2]
    innovations = hadamard(64)[:, 1:6]
    samples = np.empty(innovations.shape, dtype=float)
    samples[:, route[0]] = innovations[:, 0]
    for k in range(1, len(route)):
        samples[:, route[k]] = 0.8 * samples[:, route[k - 1]] + 0.6 * innovations[:, k]

    engine = joint_engine(samples, NativeParCorr(), jobs)
    complete = np.full((5, 5, 1), "o?o")
    result = estimate_skeleton(engine, complete, depth=1)
    for left, right in combinations(range(len(route)), 2):
        src, dst = route[left], route[right]
        kept = right == left + 1
        for a, b in ((src, dst), (dst, src)):
            assert bool(result["graph"][a, b, 0]) == kept
            if not kept:
                (separator,) = result["sepsets"][((a, 0), b)]
                assert separator in [(node, 0) for node in route[left + 1 : right]]


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("last_subset", [0, 1, None])
def test_neighbors_removed_within_a_depth_remain_eligible_as_separators(
    jobs, last_subset
):
    candidates = np.full((5, 5, 1), "", dtype="<U3")
    candidates[:4, 4, 0] = candidates[4, :4, 0] = "o?o"
    evidence = PairEvidence(
        {
            ((0, 0), 4, ((2, 0),)): (-0.08, 0.62),
            ((1, 0), 4, ((0, 0),)): (0.06, 0.71),
            ((3, 0), 4, ((0, 0),)): (0.38, 0.03),
            ((3, 0), 4, ((1, 0),)): (-0.34, 0.03),
            ((3, 0), 4, ((2, 0),)): (0.35, 0.03),
        }
    )
    search = joint_engine(np.zeros((6, 5)), evidence, jobs)
    result = estimate_skeleton(search, candidates, depth=1, last_subset=last_subset)
    assert result["sepsets"][((1, 0), 4)] == [(0, 0)]
    assert result["graph"][1, 4, 0] == result["graph"][4, 1, 0] == ""
    if last_subset == 0:
        assert result["graph"][0, 4, 0] == "o?o"
    else:
        assert result["sepsets"][((0, 0), 4)] == [(2, 0)]
        assert result["sepsets"][((4, 0), 0)] == [(2, 0)]
    expected = {0: 0.38, 1: -0.34, None: 0.35}
    assert result["val_matrix"][3, 4, 0] == expected[last_subset]
    assert result["p_matrix"][3, 4, 0] == 0.03


@pytest.mark.parametrize("jobs", [1, 2])
def test_custom_ci_compares_with_the_stored_matrix_precision(jobs):
    candidates = np.full((4, 4, 1), "o?o")
    evidence = PairEvidence(
        {
            ((0, 0), 3, ((1, 0),)): (0.4, Decimal("0.0312500000000000002")),
            ((0, 0), 3, ((2, 0),)): (0.3, Decimal("0.0312500000000000001")),
        }
    )
    result = estimate_skeleton(
        joint_engine(np.zeros((6, 4)), evidence, jobs), candidates, 1
    )
    assert result["p_matrix"][0, 3, 0] == 0.03125
    assert result["val_matrix"][0, 3, 0] == 0.3


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("first_independent", [True, False])
def test_reciprocal_removal_and_statistics_depend_on_the_lag(jobs, first_independent):
    candidates = np.full((3, 3, 2), "", dtype="<U3")
    candidates[0, 2] = candidates[2, 0] = ["o?o", "-?>"]
    evidence = PairEvidence(
        {
            ((0, 0), 2, ()): (-0.17, 0.68) if first_independent else (-0.49, 0.02),
            ((2, 0), 0, ()): (0.09, 0.83),
            ((2, -1), 0, ()): (0.11, 0.72),
        }
    )
    result = estimate_skeleton(
        joint_engine(np.zeros((6, 3)), evidence, jobs), candidates, 0
    )
    assert result["graph"][0, 2].tolist() == ["", "-?>"]
    assert result["graph"][2, 0].tolist() == ["", ""]
    p = 0.68 if first_independent else 0.83
    np.testing.assert_array_equal(result["p_matrix"][0, 2], [p, 0.01])
    np.testing.assert_array_equal(result["p_matrix"][2, 0], [p, 0.72])
    assert result["val_matrix"][0, 2, 0] == (-0.17 if first_independent else -0.49)
    assert result["val_matrix"][2, 0, 0] == (0.0 if first_independent else 0.09)


@pytest.mark.parametrize("jobs", [1, 2])
def test_background_requirement_keeps_measured_lag_among_constant_series(jobs):
    model = joint_engine(np.ones((24, 3)), NativeParCorr(), jobs)
    rule = BackgroundKnowledge().require(
        "station2[t-1]", "station0", phases=BKPhase.PRE_SEARCH
    )
    model._background_knowledge = TemporalKnowledge(rule, model.var_names, 0, 1)
    candidates = np.full((3, 3, 2), "", dtype="<U3")
    candidates[2, 0, 1] = candidates[2, 1, 1] = "-->"
    candidates[0, 1, 1] = "-?>"

    result = estimate_skeleton(model, candidates, depth=0)
    assert result["graph"][2, :, 1].tolist() == ["-->", "-->", ""]
    assert result["graph"][0, 1, 1] == ""
    np.testing.assert_array_equal(result["p_matrix"][2, :, 1], [1, 0, 1])
    assert model.diagnostics == [
        {
            "source": 2,
            "target": 0,
            "lag": 1,
            "p_value": 1.0,
            "stage": "system_search",
            "reason": "required_link",
        }
    ]
