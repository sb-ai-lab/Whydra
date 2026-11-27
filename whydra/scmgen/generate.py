"""Store datasets with complete ground truth and generate paired samples.

`Dataset.latent` keeps track of each row's exogenous noise and data-generating state:
structural mu0/tau, assignment noise u_t, outcome noise, mediator coefficients and
noise, collider noise, and contributions from pathology axes. ``truth.refresh_truth``
computes the dependent ground-truth fields. Reusing each row's exogenous draws
pairs its two potential outcomes and its clean and corrupted samples (S-2).
Recomputing the individual treatment effect keeps it consistent with the row's
current mechanisms.

Since 0.6.0 the model and sample are separate. `sample_scm(config, model_seed)` creates
an SCM with a graph, coefficients, and calibration constants. `scm.sample(n,
sample_seed)` creates clean data, and `apply_pathologies(ds, pathology_seed)` produces
the corrupted data. `generate(config, n, seed)` uses one seed for all three streams.
`generate_split` also draws an independent test population from the same SCM: clean with
test_dgp="none", or subject to the same data-generating axes with "replay".

The clean model and `Dataset.recipe()` identify a dataset (0.7.0). The recipe
describes the model, sample, and axes actually applied with their effective parameters.
`scmgen.io.replay(dir)` reproduces it bit for bit in the reference environment. The
Dataset constructor is an internal API.
"""

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DRIFT_KINDS, PHASE_DGP, ScenarioConfig, pathology_phase
from .graph import SampledGraph, column_node
from .scm import SCM

RECIPE_SCHEMA = 1
TEST_DGP_MODES = ("none", "replay")


@dataclass
class Dataset:
    """Store a dataset and its complete ground truth (O-1, O-2)."""

    x: (
        pd.DataFrame
    )  # Observed covariates, including colliders. Node columns use the names x{i}.
    t: np.ndarray  # Recorded binary treatment, which may differ from t_true after noise_t or missing.
    y: np.ndarray  # Observed outcome, equal to y_true before the measurement phase.
    y0: np.ndarray  # Potential outcome under T=0 (shared noise in the shared coupling).
    y1: np.ndarray  # Potential outcome under T=1.
    tau: (
        np.ndarray
    )  # Structural effect in outcome units, conditional on (W, u_tau, eps_med):
    # Gaussian equals ite with shared noise, binary uses p1-p0, and ziln/beta use mu1-mu0. This
    # oracle conditions on latent variables beyond W.
    mu0: np.ndarray  # Noise-free component of the outcome under T=0.
    mu1: np.ndarray  # Noise-free component under T=1.
    e: np.ndarray  # P(T_true = 1 | W[, S = 1]) in the exported population.
    segments: (
        np.ndarray
    )  # True segments (S-1) shared by tau, pathologies, and curation.
    graph: SampledGraph
    config: ScenarioConfig
    model_seed: (
        int | None
    )  # Model seed for graph, coefficients, and calibration (0.6.0).
    sample_seed: int | None = (
        None  # Seed for row-level exogenous noise. None for the calibration population.
    )
    pathology_seed: int | None = None  # Seed for pathology axes. None for clean data.
    t_true: np.ndarray | None = None  # True treatment received.
    pathology_log: list | None = (
        None  # What was applied and where. Empty for clean data.
    )
    # Ground truth for additional axes. None means the axis was absent.
    z: np.ndarray | None = (
        None  # P-17: assigned treatment (t_true is treatment received). Included in observed data.
    )
    tau_cate: np.ndarray | None = (
        None  # E[tau | W], integrated over u_tau and eps_med. Defaults to tau.
    )
    cvte: np.ndarray | None = (
        None  # Var[tau | W] over u_tau and eps_med, the irreducible part of ITE variation. Defaults to 0.
    )
    tau_itt: np.ndarray | None = (
        None  # P-17: E[Y(T(Z=1)) - Y(T(Z=0)) | W, u_tau, eps_med], the oracle conditional on latent variables.
    )
    compliance: np.ndarray | None = (
        None  # P-17: 0 complier | 1 never-taker | 2 always-taker
    )
    mediators: np.ndarray | None = (
        None  # Mediator values (n × m) under the treatment received. None when m = 0.
    )
    latent: dict | None = (
        None  # Latent state of the data-generating process (see truth.py).
    )
    # --- 0.5.0 ---
    e_assign: np.ndarray | None = (
        None  # P(Z = 1 | W), the assignment propensity. Only present with noncompliance.
    )
    e_source: np.ndarray | None = (
        None  # P(T_true = 1 | W) before selection. Only present with selection.
    )
    ate_source: float | None = (
        None  # Mean tau in the source sample. Only present with selection.
    )
    row_id: np.ndarray | None = (
        None  # Row number in the original sample before selection.
    )
    period: np.ndarray | None = (
        None  # Drift period (0/1) in original row order. None before drift.
    )
    n_requested: int | None = None  # Requested n before selection.
    # --- 0.6.0 ---
    scm: SCM | None = (
        None  # The SCM that generated the data. Serialized in scm.json separately from parquet.
    )
    split: str = "train"  # train | test (generate_split)
    segments_unseen: int = 0  # Rows in grid cells that were empty at calibration and received the nearest ID.
    cluster: np.ndarray | None = (
        None  # P-19: each row's cluster number (an observed column). None without this axis.
    )
    dup_index: np.ndarray | None = (
        None  # P-24: 0 for the original and 1..k for copies. None without this axis.
    )
    x_missing: pd.DataFrame | None = (
        None  # P-26: mask of truly missing X values encoded as sentinels.
    )
    # --- 0.7.0 ---
    tau_itt_cate: np.ndarray | None = (
        None  # E[Y(T(1)) - Y(T(0)) | W], the assignment CATE. Only present with noncompliance.
    )
    tau_cate_selected: np.ndarray | None = (
        None  # E[Y(1) - Y(0) | W, S = 1]. Only present with selection.
    )
    tau_itt_selected: np.ndarray | None = (
        None  # E_ν[complier·ite·ω] / E_ν[ω]; selection + noncompliance
    )
    e_assign_selected: np.ndarray | None = (
        None  # P(Z = 1 | W, S = 1); selection + noncompliance
    )
    n_source: int | None = (
        None  # Row count before the first selection stage. Only present with selection.
    )
    applied_pathologies: list = field(
        default_factory=list
    )  # Recipe for the axes applied by apply_pathologies.
    test_dgp: str | None = (
        None  # generate_split: none or replay, recorded on dirty and test.
    )
    test_target_match: bool | None = (
        None  # generate_split: whether the test target matches the dirty dataset's estimation task.
    )
    # --- 0.11.0 ---
    tau_cate_x: np.ndarray | None = (
        None  # E[ite | x], the oracle without outcome lags. Only present for outcome.pre_periods > 0.
    )

    def __post_init__(self):
        n = self.tau.size
        if self.t_true is None:
            self.t_true = self.t
        if self.pathology_log is None:
            self.pathology_log = []
        if self.tau_cate is None:
            self.tau_cate = self.tau
        if self.cvte is None:
            self.cvte = np.zeros(n)
        if self.latent is None:
            self.latent = {}
        if self.row_id is None:
            self.row_id = np.arange(n, dtype=np.int64)
        if self.n_requested is None:
            self.n_requested = int(n)

    @property
    def seed(self) -> int | None:
        """Return the single 0.5.0-style seed: model_seed when all three streams use it, otherwise None."""
        seeds = {self.model_seed, self.sample_seed, self.pathology_seed} - {None}
        return self.model_seed if seeds == {self.model_seed} else None

    @property
    def ate(self) -> float:
        """Return mean(tau) over exported rows.

        After selection this includes only selected rows. See ate_source and
        ate_selected.
        """
        return float(self.tau.mean())

    @property
    def ate_selected(self) -> float | None:
        """Return mean(tau_cate_selected) over selected rows, or None without selection."""
        return (
            None
            if self.tau_cate_selected is None
            else float(self.tau_cate_selected.mean())
        )

    @property
    def vte_w(self) -> np.ndarray:
        """Return the conditional variance of the realized effect given W.

        Var[Y(1) − Y(0) | W] includes outcome noise. Unlike cvte, it depends on how
        potential outcomes are coupled (`outcome.coupling`). The first access computes
        it with truth.vte_w and caches it in `latent["vte_w"]` until ground truth is
        recomputed (0.8.0).
        """
        assert self.latent is not None
        if "vte_w" not in self.latent:
            from .truth import vte_w

            self.latent["vte_w"] = vte_w(self)
        return self.latent["vte_w"]

    @property
    def y_scale(self) -> np.ndarray | None:
        """Return the natural outcome scale for each row.

        s(x) multiplies all outcome-scale ground-truth fields, with s² multiplying cvte
        and vte_w (0.9.0). Return None if outcome.scale_sigma = 0.
        """
        assert self.latent is not None
        return self.latent.get("y_scale")

    @property
    def ite(self) -> np.ndarray:
        """Return the realized individual effect, y1 - y0."""
        return self.y1 - self.y0

    # Binary mediators (0.10.0): n × m fields from the latent state. None for gaussian or
    # absent mediators.

    def mediator_truth(self) -> dict | None:
        """Return true probabilities, effects, and values for binary mediators.

        The dictionary includes p_m0, p_m1, tau_m, theta, m0, and m1, cached in the
        latent state. Return None for gaussian mediators or when there are no mediators.
        """
        from .truth import mediator_truth

        return mediator_truth(self)

    def _mediator_field(self, key: str):
        mt = self.mediator_truth()
        return None if mt is None else mt[key]

    @property
    def p_m0(self) -> np.ndarray | None:
        """P(M_k = 1 | W, T = 0), n × m."""
        return self._mediator_field("p_m0")

    @property
    def p_m1(self) -> np.ndarray | None:
        """P(M_k = 1 | W, T = 1), n × m."""
        return self._mediator_field("p_m1")

    @property
    def tau_m(self) -> np.ndarray | None:
        """Return p_m1 − p_m0, the effect of T on the mediator given W (n × m)."""
        return self._mediator_field("tau_m")

    @property
    def theta(self) -> np.ndarray | None:
        """Return θ_k(x), the M_k -> Y effect in unscaled outcome units, without s (n × m)."""
        return self._mediator_field("theta")

    @property
    def m0(self) -> np.ndarray | None:
        """Return M_k(T=0) from shared uniform noise as an n × m int8 array."""
        return self._mediator_field("m0")

    @property
    def m1(self) -> np.ndarray | None:
        """M_k(T=1), int8 n × m; `mediators` = M(t_true)."""
        return self._mediator_field("m1")

    @property
    def y_true(self) -> np.ndarray:
        """Return the outcome under the treatment received, before measurement error."""
        return np.where(self.t_true == 1, self.y1, self.y0)

    @property
    def itt(self) -> np.ndarray | None:
        """Return the realized assignment effect, 1[complier]·ite, or None without noncompliance."""
        if self.compliance is None:
            return None
        return np.where(self.compliance == 0, self.ite, 0.0)

    @property
    def n(self) -> int:
        """Return the number of rows in the dataset."""
        return int(self.tau.size)

    def observed_nodes(self) -> list[int]:
        """Return the indices of X nodes with columns in x.

        The hidden_confounder axis can remove some.
        """
        return [column_node(c) for c in self.x.columns if column_node(c) >= 0]

    def to_frame(self, onehot: bool = False) -> pd.DataFrame:
        """Return the O-1 table: X columns, optional z, t, and y. onehot=True expands categorical nodes using x_onehot."""
        df = self.x_onehot() if onehot else self.x.copy()
        if self.z is not None:
            df["z"] = self.z
        df["t"] = self.t
        df["y"] = self.y
        return df

    def _onehot_levels(self) -> dict[str, int]:
        """Map each categorical x column with k >= 3 to its one-hot indicator count."""
        disc = self.graph.discrete or {}
        out = {}
        for c in self.x.columns:
            spec = disc.get(column_node(c))
            if (
                spec is not None
                and spec["kind"] == "categorical"
                and spec["levels"] >= 3
            ):
                out[c] = int(spec["levels"])
        return out

    def column_nodes(self, onehot: bool = False) -> list[int]:
        """Map each feature-matrix column to its graph node index.

        Use the X node index for each column of x or x_onehot(). Use -1 for columns
        without an X node: period, cluster, and mediators or descendants of Y exposed by
        bad_controls.
        """
        levels = self._onehot_levels() if onehot else {}
        return [column_node(c) for c in self.x.columns for _ in range(levels.get(c, 1))]

    def x_onehot(self) -> pd.DataFrame:
        """Expand categorical features into indicator columns.

        Each categorical X node with k >= 3 yields k float 0/1 columns, for example x7=0
        through x7=k-1. A missing value gives NaN in every corresponding indicator.
        Binary and continuous columns pass through. Return a copy of x if it contains no
        such categorical nodes.
        """
        levels = self._onehot_levels()
        if not levels:
            return self.x.copy()
        parts, run = [], []
        for c in self.x.columns:
            if c not in levels:
                run.append(c)
                continue
            if run:
                parts.append(self.x[run])
                run = []
            v = self.x[c].to_numpy(dtype=float)
            ind = (v[:, None] == np.arange(levels[c])).astype(float)
            ind[np.isnan(v)] = np.nan
            parts.append(
                pd.DataFrame(
                    ind,
                    index=self.x.index,
                    columns=[f"{c}={j}" for j in range(levels[c])],
                )
            )
        if run:
            parts.append(self.x[run])
        return pd.concat(parts, axis=1)

    def discrete_truth(self) -> dict | None:
        """Return discrete node parameters and realized sample proportions for each level, or None without discrete nodes."""
        disc = self.graph.discrete
        if not disc:
            return None
        out = {}
        for i, spec in disc.items():
            item = dict(spec)
            col = self.graph.node_names[i]
            if col in self.x.columns:
                v = self.x[col].to_numpy(dtype=float)
                v = v[~np.isnan(v)]
                k = int(spec["levels"])
                item["observed_shares"] = (
                    (np.bincount(v.astype(int), minlength=k) / v.size).tolist()
                    if v.size
                    else [float("nan")] * k
                )
            out[i] = item
        return out

    def achieved(self) -> dict:
        """Summarize the realized sample.

        Data-generating summaries include the fraction of rows with tau = 0, the
        control response or zero share, the standard deviation of tau, and rows in grid
        cells unoccupied at calibration.
        """
        out = {
            "sparsity_achieved": float((self.tau == 0).mean()),
            "ate": self.ate,
            "tau_sd": float(self.tau.std()),
            "segments_unseen": int(self.segments_unseen),
        }
        fam = self.config.outcome.family
        if fam == "binary":
            out["control_rate"] = float(self.y0.mean())
        elif fam == "ziln":
            out["zero_share"] = float((self.y0 == 0).mean())
        elif fam == "beta":
            out["control_mean"] = float(self.y0.mean())
            out["zero_share"] = float((self.y0 == 0).mean())
            out["one_share"] = float((self.y0 == 1).mean())
        elif (
            fam == "lognormal"
        ):  # Row-level realized values (0.11.0). Tau uses the log scale.
            out["control_mean"] = float(self.y0.mean())
            out["zero_share"] = float((self.y0 == 0).mean())
            assert self.latent is not None
            tau_log = self.latent["tau_x"]
            out["tau_log_mean"], out["tau_log_sd"] = (
                float(tau_log.mean()),
                float(tau_log.std()),
            )
        s = self.y_scale
        if s is not None:  # Natural scale (0.9.0), measured over sampled rows.
            y = np.where(self.t_true == 1, self.y1, self.y0)
            c = y - y.mean()
            m2 = float((c * c).mean())
            out["y_kurtosis"] = (
                float((c**4).mean() / (m2 * m2) - 3.0) if m2 > 0 else 0.0
            )
            out["scale_sd_log"] = float(np.log(s).std())
        mt = self.mediator_truth()
        if mt is not None:  # Binary mediators (0.10.0), measured over sampled rows.
            for key in ("p_m0", "p_m1", "tau_m", "theta"):
                out[key] = mt[key].mean(axis=0).tolist()
            assert self.mediators is not None
            out["mediator_rate"] = self.mediators.mean(axis=0).tolist()
        return out

    def design(self) -> dict:
        """Describe the study design and treatment assignment.

        Report assignment by mechanism (rct-cluster only if Z is constant within each
        observed cluster), treatment received, selection, drifts, split, and the
        estimation task (cate_dgp, test_dgp, test_target_match).
        """
        from .truth import cate_dgp

        assert self.pathology_log is not None
        log = [e for e in self.pathology_log if e.get("active", True)]
        kinds = [e.get("kind") for e in log]
        assignment = self.config.treatment.mode
        if any(
            e.get("kind") == "cluster"
            and e.get("assign") == "cluster"
            and e.get("z_constant_within_cluster")
            for e in log
        ):
            assignment = "rct-cluster"
        if any(e.get("kind") == "cluster" and e.get("assign") == "block" for e in log):
            assignment = "rct-block"  # Complete randomization within blocks (0.11.0).
        nc = next((e for e in log if e.get("kind") == "noncompliance"), None)
        return {
            "assignment": assignment,
            "received": f"noncompliance:{nc.get('side', 'one')}" if nc else "compliant",
            "selection": "selection" if "selection" in kinds else "none",
            "policy_drift": "policy_drift" in kinds,
            "effect_drift": "effect_drift" in kinds,
            "split": self.split,
            "cate_dgp": cate_dgp(self),
            "test_dgp": self.test_dgp,
            "test_target_match": self.test_target_match,
            # Potential outcome coupling is a generator assumption (0.8.0). Conditional on
            # W, outcomes are uncorrelated only with independent Y(1) noise and no mediators,
            # since mediator noise enters both outcomes.
            "po_coupling": self.config.outcome.coupling,
            "po_uncorrelated_given_w": (
                self.config.outcome.coupling == "independent"
                and self.config.graph.n_mediators == 0
            ),
            # Natural outcome scale (0.9.0): node and σ. None if scale_sigma =
            # 0.
            "outcome_scale": self._outcome_scale_design(),
            # Mediation (0.10.0): direct edge, M family, and fraction of the effect through
            # M. None without M. achieved() gives means for p_m0, p_m1, tau_m, and theta.
            "mediation": self._mediation_design(),
        }

    def _mediation_design(self) -> dict | None:
        g = self.config.graph
        if g.n_mediators == 0:
            return None
        from .mechanisms import mediator_contributions

        s = 1.0 if self.y_scale is None else self.y_scale
        assert self.latent is not None
        terms = mediator_contributions(self.latent)
        assert terms is not None
        via = terms.tau_w * s  # Effect through M in expectation given W.
        assert self.tau_cate is not None
        total = float(self.tau_cate.mean())
        return {
            "direct_effect": bool(g.direct_effect),
            "family": self.config.mediator.family,
            "n_mediators": int(g.n_mediators),
            "ate_mediated": float(np.mean(via)),
            "share_mediated": float(np.mean(via) / total) if total != 0 else None,
        }

    def _outcome_scale_design(self) -> dict | None:
        oc = self.config.outcome
        if oc.scale_sigma <= 0:
            return None
        j = None if self.scm is None else int(self.scm.params["scale_feature"])
        return {
            "sigma": float(oc.scale_sigma),
            "feature": j,
            "column": None if j is None else self.graph.node_names[j],
        }

    def recipe(self) -> dict:
        """Collect the parameters needed to reproduce this dataset.

        The recipe (0.7.0) records the model, sample, split, and axes actually
        applied, including their effective parameters after overrides and original
        indices for the random streams. scmgen.io.replay uses it for bitwise
        reproduction in the reference environment.
        """
        from .scm import config_sha256

        assert self.n_requested is not None
        return {
            "recipe_schema": RECIPE_SCHEMA,
            "model": {
                "config_sha256": None
                if self.scm is None
                else config_sha256(self.scm.config),
                "model_seed": self.model_seed,
                "n_calib": None if self.scm is None else int(self.scm.n_calib),
            },
            "sample": {
                "sample_seed": self.sample_seed,
                "n_requested": int(self.n_requested),
                "split": self.split,
            },
            "pathologies": {
                "seed": self.pathology_seed,
                "applied": copy.deepcopy(self.applied_pathologies),
            },
            "test_dgp": self.test_dgp,
            "test_target_match": self.test_target_match,
        }

    def dag(self) -> tuple[np.ndarray, list[str], list[str]]:
        """Return the graph after the axes actually applied to this dataset.

        The post-intervention DAG (0.7.0) extends the clean graph using the
        actual arguments of axes marked active in pathology_log:

        * noncompliance adds z -> t, moves assignment edges from t to z, and adds hidden
        compliance -> t. Parents of compliance are the projection's X nodes plus nu_y
        when hidden > 0.

        * targeted_selection, overlap, and policy_drift link the X nodes used to the
        assignment node (z if present, otherwise t). policy_drift also links period to
        assignment.

        * hidden_modifier adds u_tau_{index} -> y. cluster links cluster to y for icc >
        0 and to assignment for icc_e > 0 or assign set to cluster or block.
        covariate_shift links period to the shifted X nodes. effect_drift adds period -> y.

        * Measurement axes add nodes with role measurement: x{i}_obs <- x{i}, plus
        period for measurement_drift and a driver for MAR; y_obs <- y, plus t for by_arm
        and the heteroscedastic feature; and t_obs <- t. Each observed column maps to
        its *_obs node when present (dag_post.observed_columns).

        * hidden_confounder marks its hidden nodes with the hidden role. bad_controls
        adds d{k} <- y for y_descendant.

        Zero-strength axes that have no action add no nodes. Return (adjacency, names,
        roles). In a clean dataset adjacency equals graph.adjacency.
        """
        g = self.graph
        adj = g.adjacency.astype(np.int8).copy()
        names = list(g.node_names)
        assert g.roles_effective is not None
        roles = (
            list(g.roles_effective)
            + ["treatment"]
            + ["mediator"] * self.config.graph.n_mediators
            + ["outcome"]
        )
        it, iy = g.idx_t(), g.idx_y()
        state = {"adj": adj}

        def add(name: str, role: str) -> int:
            a = state["adj"]
            k = a.shape[0]
            new = np.zeros((k + 1, k + 1), dtype=np.int8)
            new[:k, :k] = a
            state["adj"] = new
            names.append(name)
            roles.append(role)
            return k

        def node(name: str, role: str) -> int:
            return names.index(name) if name in names else add(name, role)

        def obs_node(col: str) -> int:
            """Measurement node for column col (x{i}, t, or y) with the true node as parent."""
            src = names.index(col)
            k = node(f"{col}_obs", "measurement")
            state["adj"][src, k] = 1
            return k

        assign = it
        period = None
        from .truth import pre_columns

        pre = pre_columns(self)
        # Outcome lags (0.11.0): past Y(0) depends on the X parents of μ₀ and the unit's
        # persistent component u_y. The same component also contributes to the noise in y.
        if pre:
            iu = add("u_y", "hidden")
            state["adj"][iu, iy] = 1
            parents = [
                i
                for i in range(g.d)
                if g.adjacency[i, iy]
                and g.roles_effective[i]
                in ("confounder", "predictor", "med_confounder")
            ]
            for col in pre:
                k = add(col, "pre_outcome")
                state["adj"][iu, k] = 1
                for i in parents:
                    state["adj"][i, k] = 1
        assert self.pathology_log is not None
        for index, e in enumerate(self.pathology_log):
            if not e.get("active", True):
                continue
            kind = e.get("kind")
            affected = [c for c in e.get("affected_columns", []) if isinstance(c, str)]
            if kind in DRIFT_KINDS and period is None:
                period = add("period", "period")
            a = state["adj"]
            if kind == "noncompliance":
                iz = add("z", "assignment")
                a = state["adj"]
                a[iz, it] = 1
                for i in range(
                    iz
                ):  # z uses all the inputs previously used for assignment.
                    if a[i, it]:
                        a[i, iz], a[i, it] = 1, 0
                ic = add("compliance", "hidden")
                a = state["adj"]
                a[ic, it] = 1
                for j in e.get("score_features", []):
                    a[int(j), ic] = 1
                if float(e.get("hidden", 0.0)) > 0:
                    inu = node("nu_y", "hidden")
                    a = state["adj"]
                    a[inu, iy] = 1
                    a[inu, ic] = 1
                assign = iz
            elif kind == "hidden_modifier":
                iu = add(f"u_tau_{index}", "hidden")
                state["adj"][iu, iy] = 1
            elif kind == "bad_controls":
                for col, what in (e.get("added_columns") or {}).items():
                    if what == "y_descendant":
                        k = add(col, "y_descendant")
                        state["adj"][iy, k] = 1
            elif kind == "hidden_confounder":
                for col in e.get("hidden_columns", []):
                    roles[names.index(col)] = "hidden"
            elif kind in ("targeted_selection", "overlap", "policy_drift"):
                for j in e.get("features", []):
                    a[int(j), assign] = 1
                if kind == "policy_drift":
                    a[period, assign] = 1
            elif kind == "covariate_shift":
                for j in e.get("features", []):
                    a[period, int(j)] = 1
            elif kind == "effect_drift":
                a[period, iy] = 1
            elif kind == "cluster":
                ic = add("cluster", "cluster")
                a = state["adj"]
                if float(e.get("icc_y", 0.0)) > 0:
                    a[ic, iy] = 1
                if float(e.get("icc_e", 0.0)) > 0 or e.get("assign") in (
                    "cluster",
                    "block",
                ):
                    a[ic, assign] = 1
            elif kind == "measurement_drift":
                for col in affected:
                    if column_node(col) >= 0:
                        k = obs_node(col)
                        state["adj"][period, k] = 1
            elif kind in ("noise_x", "missing"):
                driver = e.get("driver")
                for col in affected:
                    if column_node(col) >= 0 or col in ("t", "y"):
                        k = obs_node(col)
                        if driver is not None and column_node(col) >= 0:
                            state["adj"][int(driver), k] = 1
            elif kind == "quantize_y":  # 0.11.0
                if "y" in affected:
                    obs_node("y")
            elif kind == "noise_y":
                if "y" in affected:
                    k = obs_node("y")
                    if e.get("by_arm"):
                        state["adj"][it, k] = 1
            elif kind == "heteroscedastic":
                if "y" in affected:
                    k = obs_node("y")
                    state["adj"][int(e["feature"]), k] = 1
            elif kind == "noise_t":
                if "t" in affected:
                    obs_node("t")
        return state["adj"], names, roles

    def dag_post(self) -> dict:
        """Return the graph after dataset pathologies and map observed columns to its nodes."""
        adj, names, roles = self.dag()
        cols = (
            list(map(str, self.x.columns))
            + (["z"] if self.z is not None else [])
            + ["t", "y"]
        )
        observed = {c: (f"{c}_obs" if f"{c}_obs" in names else c) for c in cols}
        return {
            "adjacency": adj.tolist(),
            "names": names,
            "roles": roles,
            "observed_columns": observed,
        }

    def ground_truth(self) -> dict:
        """Return all ground-truth fields together (O-2)."""
        from .truth import late_realized, truth_targets

        return {
            "y0": self.y0,
            "y1": self.y1,
            "ite": self.ite,
            "tau": self.tau,
            "mu0": self.mu0,
            "mu1": self.mu1,
            "e": self.e,
            "t_true": self.t_true,
            "ate": self.ate,
            "ate_source": self.ate_source,
            "ate_selected": self.ate_selected,
            "segments": self.segments,
            "tau_cate": self.tau_cate,
            "cvte": self.cvte,
            "vte_w": self.vte_w,
            "z": self.z,
            "itt": self.itt,
            "tau_itt": self.tau_itt,
            "tau_itt_cate": self.tau_itt_cate,
            "compliance": self.compliance,
            "e_assign": self.e_assign,
            "e_source": self.e_source,
            "tau_cate_selected": self.tau_cate_selected,
            "tau_itt_selected": self.tau_itt_selected,
            "e_assign_selected": self.e_assign_selected,
            "tau_cate_x": self.tau_cate_x,
            "row_id": self.row_id,
            "period": self.period,
            "n_requested": self.n_requested,
            "n_source": self.n_source,
            "late_realized": late_realized(self),
            "adjacency": self.graph.adjacency,
            "roles": self.graph.roles,
            "roles_effective": self.graph.roles_effective,
            "node_names": self.graph.node_names,
            "discrete": self.discrete_truth(),
            "cluster": self.cluster,
            "dup_index": self.dup_index,
            "targets": truth_targets(self),
            "achieved": self.achieved(),
            "design": self.design(),
            "recipe": self.recipe(),
            "pathology_log": self.pathology_log,
            "config": self.config.to_dict(),
            "seed": self.seed,
            "model_seed": self.model_seed,
            "sample_seed": self.sample_seed,
            "pathology_seed": self.pathology_seed,
            "split": self.split,
        }


def generate(config: ScenarioConfig, n: int, seed: int) -> Dataset:
    """Generate a deterministic dataset from a scenario, n, and one seed (T-1).

    This 0.6.0 wrapper sets model_seed = sample_seed = pathology_seed = seed. A scenario
    with pathologies returns corrupted observations with ground truth and a
    pathology_log. Use generate_pair for paired clean and corrupted data, or
    generate_split to include a test population.
    """
    from .pathologies import (
        apply_pathologies,  # Import lazily because pathologies imports generate.
    )
    from .scm import sample_scm

    clean = sample_scm(config, seed).sample(n, seed)
    if not config.pathologies:
        return clean
    return apply_pathologies(clean, seed)


def generate_pair(config: ScenarioConfig, n: int, seed: int) -> tuple[Dataset, Dataset]:
    """Generate clean and corrupted samples (S-2) with identical seeds, differing only through the applied pathologies."""
    from .pathologies import apply_pathologies
    from .scm import sample_scm

    clean = sample_scm(config, seed).sample(n, seed)
    return clean, apply_pathologies(clean, seed)


def replay_plan(config: ScenarioConfig) -> list:
    """Prepare an intervention plan for the test sample.

    Use the scenario's data-generating axes with their original indices (0.7.0).
    For drift axes set share = 1.0, placing all test rows in period two. Validate the
    plan as a configuration.
    """
    plan = []
    for i, p in enumerate(config.pathologies):
        if pathology_phase(p) != PHASE_DGP:
            continue
        q = copy.deepcopy(p)
        if q.kind in DRIFT_KINDS:
            q.params = {**q.params, "share": 1.0}
        plan.append((i, q))
    check = copy.deepcopy(config)
    check.pathologies = [q for _, q in plan]
    check.validate()
    return plan


def generate_split(
    config: ScenarioConfig,
    n_train: int,
    n_test: int,
    seed: int,
    *,
    test_seed: int | None = None,
    test_dgp: str = "none",
) -> tuple[Dataset, Dataset, Dataset]:
    """Generate a training pair and an independent test sample.

    Return (clean, dirty, test): a paired training sample (S-2) and an independent test
    population from the same SCM. Draw test
    using scm.sample(n_test, test_seed). test_seed defaults to seed + 1. Using the same
    seed for training and test raises ValueError. Call scm.sample directly for common
    random numbers.

    test_dgp="none" (default) produces a clean test sample targeting the source SCM's
    CATE. This matches dirty when design.cate_dgp == "source". Otherwise
    design.test_target_match is False on both dirty and test. test_dgp="replay" applies
    the scenario's data-generating axes to test with the same indices and
    pathology_seed, setting share = 1 for drift axes. This targets the modified CATE and
    sets test_target_match to True. The test dataset has split="test".
    """
    from .pathologies import apply_pathologies
    from .scm import sample_scm
    from .truth import cate_dgp

    if test_dgp not in TEST_DGP_MODES:
        msg = f"test_dgp must be one of {'|'.join(TEST_DGP_MODES)}, got {test_dgp!r}"
        raise ValueError(msg)
    test_seed = seed + 1 if test_seed is None else int(test_seed)
    if test_seed == int(seed):
        msg = "test must be independent of train: test_seed == sample_seed"
        raise ValueError(msg)
    scm = sample_scm(config, seed)
    clean = scm.sample(n_train, seed)
    dirty = apply_pathologies(clean, seed)
    test = scm.sample(n_test, test_seed)
    test.split = "test"
    if test_dgp == "replay":
        plan = replay_plan(config)
        test = apply_pathologies(test, seed, plan=plan)
    match = cate_dgp(dirty) == "source" or test_dgp == "replay"
    dirty.test_dgp = test.test_dgp = test_dgp
    dirty.test_target_match = test.test_target_match = match
    return clean, dirty, test
