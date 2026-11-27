"""Configure a scenario for the structural causal model generator.

A scenario is a declarative configuration that can be serialized to YAML (C-1). Settings
control the experimental design: graph, mechanisms, treatment, outcome, CATE, and data
pathologies.

Loading has been strict since 0.5.0. An unknown key raises ValueError with its path. Use
`from_dict(..., strict=False)` to read older YAML permissively. `validate()` checks
ranges, allowed values, which settings apply to each outcome family, pathology
phase order, and the parameter schema for each axis. `generate` calls it
automatically.

In 0.6.0, `drift` split into four axes: `measurement_drift` shifts observed X,
`covariate_shift` changes P(X) under the same equations, `effect_drift` rotates, scales,
or shifts τ, and `policy_drift` changes the assignment policy. In 0.7.0 the alias was
removed: `kind: drift` raises ValueError with migration guidance. Other additions
include `cluster` (P-19), `duplicate_rows` (P-24), and `missing.params.sentinel` (P-26).
See `PATHOLOGY_NUMBERS` for catalog numbers.

Since 0.7.0, `validate()` checks that numeric config fields and `params` have the right
type and finite values (booleans do not count as numbers) and that feature indices lie
in [0, n_covariates). It rejects repeated `noncompliance`, `effect_drift`, or
`duplicate_rows` axes, cluster-level assignment outside a pure RCT, and
`covariate_shift` after `noncompliance` or `effect_drift`. `CATE_MODIFYING` records
which data-generating axes change CATE(W) in each family.
"""

import dataclasses
import math
import numbers
import pathlib
from dataclasses import dataclass, field

import yaml


def _from_dict(cls, d: dict, strict: bool, path: str):
    """Build a dataclass from a dictionary.

    In strict mode, raise ValueError with the path of an unknown key.
    """
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = [k for k in d if k not in known]
    if unknown and strict:
        msg = f"{path}.{unknown[0]}: unknown configuration key (known keys: {sorted(known)})"
        raise ValueError(msg)
    return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class GraphConfig:
    """Define node roles (G-2) and X–X structure (G-1), or supply a manual X–X block (G-4)."""

    n_confounders: int = (
        3  # X -> T and X -> Y (act as confounders in observational designs)
    )
    n_modifiers: int = 2  # X -> tau (effect modifiers)
    n_predictors: int = 3  # X -> Y (outcome predictors with no link to T)
    n_noise: int = 2  # No edges
    n_instruments: int = 0  # X -> T without X -> Y (relevant in observational designs)
    n_mediators: int = 0  # T -> M -> Y (separate nodes outside X). MediatorConfig sets their mechanism.
    n_colliders: int = 0  # T -> C <- Y (a covariate that can bias adjustment, identified in node roles)
    # Roles around mediators and nodes without direct edges to T, M, or Y (0.10.0, G-2).
    n_med_confounders: int = 0  # X -> M for every mediator and X -> Y: an M–Y confounder (gdiag conf_my), never an ancestor of T
    n_med_predictors: int = (
        0  # X -> M for every mediator, reaching Y only through M (gdiag pred_m)
    )
    n_upstream: int = (
        0  # Only X–X edges, outgoing to any role. No direct edge to T, M, or Y.
    )
    n_side: int = 0  # Only X–X edges, outgoing only to other side nodes. Never an ancestor of T, M, or Y (gdiag side).
    direct_effect: bool = True  # False removes T -> Y and sets tau_x ≡ 0. The entire effect passes through mediators (n_mediators ≥ 1).
    # Optional role strings in X position order. Their length must equal the covariate count and
    # their frequencies must match the role counts. None uses role_order. With a manual adjacency
    # matrix this replaces the block layout, preserving topological order.
    roles: list | None = None
    xx_density: float = 0.0  # X–X edge density across roles, in [0, 1]
    # Optional X–X adjacency matrix (G-4). Since 0.5.0, it supplies only the X–X block: roles
    # determine edges to T and Y. None samples the block. As in sampling, collider columns and
    # noise/instrument edges are zeroed.
    adjacency: list | None = None
    role_order: str = "shuffled"  # Position layout: shuffled or blocks (see graph.py)
    # X–X structures (G-1a): er, scale_free, blocks, layered, and copying. A string selects one
    # structure using the original individual implementation. Since 0.3.0, a list of at least two
    # combines blocks, scale_free, and layered factors. Their matrix product is normalized to mean
    # probability xx_density over eligible pairs. p_in, p_out, and layer_skip affect only its
    # shape. The random draws follow blocks -> scale_free -> layered regardless of list order.
    # copying cannot be combined.
    xx_structure: str | list = "er"
    max_in_degree: int | None = (
        None  # Maximum number of parents for an X node (since 0.3.0). None imposes no limit.
    )
    sf_gamma: float = 2.5  # scale_free (Chung–Lu): power-law tail exponent > 2. Smaller values produce stronger hubs.
    blocks_k: int = 3  # blocks (SBM): number of feature blocks
    p_in: float = 0.4  # blocks: within-block edge density
    p_out: float = 0.02  # blocks: between-block edge density
    layers_l: int = 4  # layered: number of layers (depth of the feature pipeline)
    layer_skip: float = 0.0  # layered: fraction of density for edges that skip layers (0 connects adjacent layers only)
    copy_share: float = (
        0.5  # copying: fraction of clone nodes that inherit parents from a template
    )
    copy_fidelity: float = (
        1.0  # copying: probability of inheriting each template parent
    )
    # Discrete X nodes (since 0.4.0) start continuous, with parental contributions plus eps.
    # Standardize the latent value using SCM calibration constants (since 0.6.0, previously sample
    # statistics) and split it at Φ⁻¹ thresholds for the target proportions. Descendants and
    # mechanisms use the discrete value's level effects (SampledGraph.discrete). Zero counts
    # consume no random draws.
    n_binary: int = 0  # Number of binary X nodes
    n_categorical: int = 0  # Number of categorical nodes
    categorical_levels: int | list = (
        4  # An integer gives every node the same number of levels. A list gives one count per node (length n_categorical).
    )
    binary_rate: float | list = field(
        default_factory=lambda: [0.05, 0.5]
    )  # Fraction of ones: draw uniformly from [lo, hi], or supply one number for all nodes.
    categorical_skew: float = 0.0  # Level probabilities are proportional to 1/rank^skew (0 is uniform, 1 is Zipf).
    discrete_roles: list | None = (
        None  # Eligible roles for discrete positions. None allows all roles except colliders.
    )


@dataclass
class MechanismConfig:
    """Specify structural equation function classes (M-1)."""

    kind: str = (
        "linear"  # linear or nonlinear (smooth trigonometric and quadratic terms)
    )
    interaction: float = 0.0  # Fraction of pairwise interactions in mu0, in [0, 1]


@dataclass
class TauConfig:
    """Configure the CATE function tau(x) separately from mu0 (M-2).

    form="segment" gives a piecewise constant tau over true segments (SegmentConfig).

    For a varying effect, mechanisms.apply_tau constructs the shape, gates rows to zero,
    normalizes to standard deviation amplitude, adds offset, then zeroes gated rows
    again. A gated row has tau = 0 even with a nonzero offset. The latent ATE is
    approximately (1 - sparsity) * offset plus the mean normalized component. That mean
    is near zero for linear and nonlinear forms, but random for segments because their
    levels are uncentered. Use Dataset.ate for the true ATE.

    offset uses the family's latent scale: outcome units for gaussian, logit for
    binary/ziln/beta, and log for lognormal. Since 0.5.0, amplitude = 0 makes the
    variable component zero for any shape, returning tau = offset on every row.
    """

    amplitude: float = 1.0  # Effect scale (standard deviation of tau)
    sparsity: float = 0.0  # Population fraction with tau = 0
    form: str = "linear"  # linear | nonlinear | segment
    offset: float = 0.0  # Shift added after normalizing tau (since 0.2.0)
    # Since 0.11.0, lognormal only: after assembly, multiply tau by exp(κ·σ_μ·mu0(x)) /
    # c_κ, where c_κ is the calibration mean of the multiplier. With κ > 0 the relative effect
    # grows with baseline outcome level.
    level_slope: float = 0.0


@dataclass
class MediatorConfig:
    """Specify mediator mechanisms M_k (0.10.0). Ignored when there are no mediators.

    gaussian (default): M_k = f_k(x) + a_k·T + ε_k, contributing w_k·M_k to Y, with a_k,
    w_k ~ N(0, 1) as before 0.10.0. The term f_k comes from M's X parents
    (med_confounder and med_predictor) and has standard deviation base_amplitude.
    Without them, f_k ≡ 0.

    binary (gaussian Y only): logit P(M_k = 1 | x, t) = c_k + f_k(x) + t·(effect +
    g_k(x))·1[gate is off]. M_k(t) = 1[u_k < p_t] uses a common uniform draw u_k. Its
    contribution to Y is θ_k(x)·M_k, where θ_k(x) = theta·exp(h_k(x))/c_θ and E[θ_k] =
    theta on calibration data. g_k uses M's X parents, and h_k uses modifiers and
    med_confounder nodes.
    """

    family: str = "gaussian"  # gaussian | binary
    base_amplitude: float = 1.0  # Standard deviation of f_k(x), in logit units for binary or M units for gaussian. Used only if M has X parents.
    base_rate: float = 0.1  # binary: mean P(M_k = 1 | T = 0) in the calibration population, set by intercept c_k
    effect: float = 0.5  # binary: mean effect of T on the log odds of M
    effect_amplitude: float = 0.0  # binary: standard deviation of g_k(x). Zero gives a homogeneous T -> M effect.
    effect_sparsity: float = 0.0  # binary: fraction of rows with zero T -> M effect, gated using M's X parents
    theta: float = (
        1.0  # binary: mean M -> Y effect in standard deviations of μ₀ (gaussian Y)
    )
    theta_amplitude: float = (
        0.0  # binary: standard deviation of log θ_k(x). Zero gives a homogeneous θ.
    )


@dataclass
class SegmentConfig:
    """Define true segments (S-1) on a quantile grid over one or two covariates.

    The same segmentation defines segment-specific tau, localizes pathologies, and
    validates curation.
    """

    n_segments: int = 8  # Target segment count (4–32 in the specification). A discrete feature uses its k levels.
    # Grid covariate indices. None chooses two by role priority. Since 0.4.0, 'categorical' and
    # 'binary' select the first node of that kind in index order.
    features: list | None = None


@dataclass
class PathologyConfig:
    """Configure one data pathology by axis, strength, and location.

    kind accepts noise_y, noise_t, noise_x, missing, heteroscedastic, overlap,
    selection, measurement_drift, covariate_shift, effect_drift, policy_drift,
    hidden_confounder, bad_controls, targeted_selection, noncompliance, hidden_modifier,
    cluster, duplicate_rows, or quantize_y (see PATHOLOGY_PARAMS). Version 0.7.0 removed
    drift. Using it raises ValueError with guidance.

    target="segments" applies the pathology to the specified segments or a random
    fraction segment_share of them, chosen deterministically from the run seed.
    Pathologies run in scenario order. Validation raises ValueError if phase order goes
    backward (data generation -> selection -> measurement; see pathology_phase).
    """

    kind: str
    strength: float = 0.5  # Strength in [0, 1]
    target: str = "global"  # global | segments
    segments: list | None = None
    segment_share: float = 0.25
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if (
            self.kind == "drift"
        ):  # Give migration instructions for the removed drift alias.
            msg = (
                "pathologies: 'drift' was split in 0.6.0 and removed in 0.7.0: observed X shift -> "
                "kind='measurement_drift'; shift_tau / scale_tau / shift_ate -> kind='effect_drift' (params); "
                "shift_e -> kind='policy_drift' (strength)"
            )
            raise ValueError(msg)


@dataclass
class TreatmentConfig:
    """Configure treatment assignment (M-3). RCT is the default design."""

    mode: str = "rct"  # rct | obs
    propensity: float = 0.5  # rct: constant e(x), controlling the balance between arms
    confounding: float = (
        1.0  # obs: confounding strength (coefficient scale in the logit)
    )
    # Overlap violations are pathology P-6 (PathologyConfig kind="overlap"), separate from
    # assignment settings.


@dataclass
class OutcomeConfig:
    """Select the outcome family (M-4).

    The family selects the noise model, the outcome scale, and the applicable
    parameters. ``ScenarioConfig.validate`` checks that the settings are compatible.
    """

    family: str = "gaussian"  # gaussian | binary | ziln | beta | lognormal
    snr: float = 1.0  # gaussian: sd(tau) / sd(outcome noise). Keep at 1.0 for binary, ziln, and beta.
    base_rate: float = 0.1  # binary: baseline control conversion probability (logit intercept at mu0 = 0)
    zero_mass: float = (
        0.8  # ziln: proportion of zeros in controls (logit intercept at mu0 = 0)
    )
    heavy_tails: bool = (
        False  # gaussian: use Student's t noise (df=3) in place of normal noise
    )
    # Beta outcomes and coupling of the two potential outcomes (since 0.8.0).
    base_mean: float = 0.3  # beta: mean of the continuous control component at mu0 = 0 (logit intercept)
    beta_phi: float = 10.0  # beta: precision φ (variance of the continuous component = m(1 − m)/(1 + φ))
    zero_mass_beta: float = 0.0  # beta: control mass at 0
    one_mass_beta: float = 0.0  # beta: control mass at 1
    beta_tau_on: str = "mean"  # beta: effect channel, mean | mass | both
    beta_mass_scale: float = 1.0  # beta: multiplier for latent tau in the mass logits (mass and both channels)
    coupling: str = "shared"  # Potential outcome coupling: shared uses common noise, independent draws separate noise for Y(1).
    # Natural outcome scale s(x) = exp(scale_sigma · z(x_j)), gaussian only (0.9.0).
    scale_sigma: float = 0.0  # Log-scale σ. Zero disables scaling and reproduces 0.8.0 data and random draws bit for bit.
    scale_feature: int | None = (
        None  # Scale's X node index. None selects the first continuous confounder, predictor, or modifier.
    )
    # --- 0.11.0: lognormal — log Y(t) = log b + σ_μ·mu0 + t·tau + σ_ε·ε, Y > 0 ---
    base_level: float = (
        1.0  # lognormal: b is the median of Y(0) at mu0 = 0 (log-scale intercept).
    )
    level_sigma: float = (
        1.0  # lognormal: σ_μ is the spread in log baseline explained by X.
    )
    noise_sigma: float = (
        1.0  # lognormal: σ_ε is the log-scale noise standard deviation.
    )
    zero_mass_ln: float = 0.0  # lognormal: point mass at zero, independent of x and T. Only the positive component carries the effect.
    # Pretreatment outcome: K lags of Y(0), for gaussian or lognormal (0.11.0).
    pre_periods: int = 0  # K. Zero disables lags without consuming random draws.
    pre_rho: float = 0.7  # ρ ∈ [0, 1): correlation between the unit's persistent component and period noise
    pre_drift: float = 0.0  # δ: baseline shift per period, in log units for lognormal or sd(mu0) units for gaussian


# Map an X role to its GraphConfig count in block layout order (role_order = "blocks"). Roles
# existing before 0.10.0 keep their relative order. Role strings are part of the graph contract
# (G-6). This is the canonical role table.
ROLE_COUNTERS = {
    "upstream": "n_upstream",
    "confounder": "n_confounders",
    "modifier": "n_modifiers",
    "predictor": "n_predictors",
    "med_confounder": "n_med_confounders",
    "med_predictor": "n_med_predictors",
    "noise": "n_noise",
    "instrument": "n_instruments",
    "collider": "n_colliders",
    "side": "n_side",
}


def check_roles(g: GraphConfig) -> None:
    """Validate graph.roles (0.10.0): d role strings whose frequencies match the role counts."""
    if g.roles is None:
        return
    d = sum(int(getattr(g, attr)) for attr in ROLE_COUNTERS.values())
    if not isinstance(g.roles, list) or len(g.roles) != d:
        msg = f"graph.roles: expected a list of {d} roles (one per covariate), got {g.roles!r}"
        raise ValueError(msg)
    bad = [r for r in g.roles if r not in ROLE_COUNTERS]
    if bad:
        msg = f"graph.roles: unknown role {bad[0]!r} (allowed: {list(ROLE_COUNTERS)})"
        raise ValueError(msg)
    diff = {
        role: (g.roles.count(role), getattr(g, attr))
        for role, attr in ROLE_COUNTERS.items()
        if g.roles.count(role) != getattr(g, attr)
    }
    if diff:
        raise ValueError(
            "graph.roles disagrees with the configured role counts: "
            + ", ".join(
                f"{role} occurs {a} times, graph.{ROLE_COUNTERS[role]} = {b}"
                for role, (a, b) in diff.items()
            )
        )


# Pathology phases: 1 data generation -> 2 selection -> 3 measurement.

BETA_PHI_MAX = (
    1e4  # Upper bound on outcome.beta_phi within betafn's precision range (a, b ≤ 1e4).
)
QUANTIZE_Y_STEP_SD = 0.1  # Default quantize_y step as a fraction of sd(y) when neither params.step nor step_sd is given (0.11.0).

PHASE_DGP, PHASE_SELECTION, PHASE_MEASUREMENT = 1, 2, 3
PHASE_NAMES = {
    PHASE_DGP: "DGP",
    PHASE_SELECTION: "selection",
    PHASE_MEASUREMENT: "measurement",
}
_DRIFT_DGP_PARAMS = ("shift_tau", "shift_e", "shift_ate", "scale_tau")
# Second-period axes: the first creates period from params.share, and the others reuse it.
DRIFT_KINDS = ("measurement_drift", "covariate_shift", "effect_drift", "policy_drift")

_PATHOLOGY_PHASE = {
    "hidden_modifier": PHASE_DGP,
    "targeted_selection": PHASE_DGP,
    "overlap": PHASE_DGP,
    "noncompliance": PHASE_DGP,
    "covariate_shift": PHASE_DGP,
    "effect_drift": PHASE_DGP,
    "policy_drift": PHASE_DGP,
    "cluster": PHASE_DGP,
    "selection": PHASE_SELECTION,
    "duplicate_rows": PHASE_SELECTION,
    "noise_y": PHASE_MEASUREMENT,
    "noise_t": PHASE_MEASUREMENT,
    "noise_x": PHASE_MEASUREMENT,
    "missing": PHASE_MEASUREMENT,
    "heteroscedastic": PHASE_MEASUREMENT,
    "outliers": PHASE_MEASUREMENT,
    "hidden_confounder": PHASE_MEASUREMENT,
    "bad_controls": PHASE_MEASUREMENT,
    "measurement_drift": PHASE_MEASUREMENT,
    "quantize_y": PHASE_MEASUREMENT,
}

# Allowed params keys for each axis.
PATHOLOGY_PARAMS = {
    "noise_y": {"by_arm"},
    "noise_t": set(),
    "noise_x": {"features", "roles", "mode", "bins"},
    "outliers": {"share", "magnitude", "spread", "sides", "by_arm"},
    "missing": {"columns", "mode", "features", "sentinel"},
    "heteroscedastic": {"feature"},
    "overlap": set(),
    "targeted_selection": {"direction"},
    "selection": {"t_weight"},
    "measurement_drift": {"features", "share"},
    "covariate_shift": {"features", "share"},
    "effect_drift": {"shift_tau", "scale_tau", "shift_ate", "share"},
    "policy_drift": {"share"},
    "hidden_confounder": set(),
    "bad_controls": {"expose", "n_y_descendants"},
    "noncompliance": {"side", "hidden"},
    "hidden_modifier": set(),
    "cluster": {
        "n_clusters",
        "size_sigma",
        "icc_e",
        "rho",
        "assign",
        "emit",
        "block_propensity",
    },
    "duplicate_rows": {"k", "noise", "position"},
    "quantize_y": {"step", "step_sd"},
}
PATHOLOGY_KINDS = tuple(PATHOLOGY_PARAMS)
GLOBAL_ONLY = ("hidden_confounder", "bad_controls", "cluster")
# Repeat applications have no defined meaning for these axes. The noncompliance
# restriction dates to 0.5.0 and the others to 0.7.0.
NO_REPEAT = ("noncompliance", "effect_drift", "duplicate_rows")
# These data-generating axes must follow covariate_shift because unit type and rotation are drawn
# using the current X. The other axes add to the logit of e or to τ and survive recomputation of X
# under the 0.7.0 data-generation contract.
AFTER_COVARIATE_SHIFT = ("noncompliance", "effect_drift")
# These axes make e vary within a cluster, so they are incompatible with cluster.assign =
# "cluster" or "block" (0.7.0).
NON_CLUSTER_CONSTANT_E = ("policy_drift", "overlap", "targeted_selection")

# For each outcome family, record which applied data-generating axes change CATE(W) compared with
# the clean SCM (0.7.0). An absent axis leaves the function unchanged in every family.
# Tests compare at identical W.
CATE_MODIFYING = {
    "hidden_modifier": (
        "binary",
        "ziln",
        "beta",
        "lognormal",
    ),  # E_U[σ(a+U) − σ(b+U)] ≠ σ(a) − σ(b); E e^(τ+U) ≠ e^τ
    "cluster": (
        "binary",
        "ziln",
        "beta",
        "lognormal",
    ),  # The cluster intercept shifts the probability difference or the
    # baseline level, which also changes the absolute lognormal effect.
    "effect_drift": ("gaussian", "binary", "ziln", "beta", "lognormal"),
}

# Catalog numbers for pathology axes.
PATHOLOGY_NUMBERS = {
    "noise_y": "P-1",
    "noise_t": "P-2",
    "noise_x": "P-3",
    "missing": "P-4",
    "heteroscedastic": "P-5",
    "overlap": "P-6",
    "selection": "P-7",
    "measurement_drift": "P-8",
    "covariate_shift": "P-8",
    "hidden_confounder": "P-9",
    "bad_controls": "P-10",
    "targeted_selection": "P-12",
    "noncompliance": "P-17",
    "cluster": "P-19",
    "hidden_modifier": "P-20",
    "effect_drift": "P-21",
    "policy_drift": "P-21",
    "duplicate_rows": "P-24",
    "outliers": "P-18",
    "quantize_y": "P-16",
}


def pathology_number(p: PathologyConfig) -> tuple[str, list[str]]:
    """Return the main and additional pathology numbers for the axis and its parameters.

    noise_y.by_arm gives P-14. noise_x.roles including
    confounder gives P-13 (residual confounding), other roles give P-3, and
    mode=quantize gives P-16. For missing, t/y or mnar gives P-15, with an additional
    P-26 for sentinel. Since 0.9.0, outliers.by_arm adds P-14.
    """
    q = p.params
    extra: list[str] = []
    if p.kind == "noise_y" and q.get("by_arm"):
        return "P-14", extra
    if p.kind == "outliers" and q.get("by_arm"):
        extra.append("P-14")
    if p.kind == "noise_x":
        if q.get("mode", "gaussian") == "quantize":
            return "P-16", extra
        if "confounder" in (q.get("roles") or ()):
            return "P-13", extra
    if p.kind == "missing":
        if q.get("sentinel") is not None:
            extra.append("P-26")
        cols = set(q.get("columns", ["x"]))
        if cols & {"t", "y"} or q.get("mode", "mcar") == "mnar":
            return "P-15", extra
    return PATHOLOGY_NUMBERS[p.kind], extra


def _drift_param_active(params: dict, key: str) -> bool:
    v = params.get(key)
    if v is None:
        return False
    return float(v) != (1.0 if key == "scale_tau" else 0.0)


def drift_share(p: PathologyConfig) -> float:
    """Return the fraction of rows in a drift axis's second period (params.share, default 0.5)."""
    return float(p.params.get("share", 0.5))


def pathology_phase(p: PathologyConfig) -> int:
    """Return the axis phase: PHASE_DGP, PHASE_SELECTION, or PHASE_MEASUREMENT."""
    phase = _PATHOLOGY_PHASE.get(p.kind)
    if phase is None:
        msg = f"unknown pathology: {p.kind!r} (known: {list(PATHOLOGY_KINDS)})"
        raise ValueError(msg)
    return phase


@dataclass
class ScenarioConfig:
    """Define a complete scenario (C-1). The benchmark uses its name as the identifier."""

    name: str = "unnamed"
    graph: GraphConfig = field(default_factory=GraphConfig)
    mechanism: MechanismConfig = field(default_factory=MechanismConfig)
    tau: TauConfig = field(default_factory=TauConfig)
    treatment: TreatmentConfig = field(default_factory=TreatmentConfig)
    outcome: OutcomeConfig = field(default_factory=OutcomeConfig)
    segments: SegmentConfig = field(default_factory=SegmentConfig)
    pathologies: list = field(default_factory=list)  # list[PathologyConfig]
    mediator: MediatorConfig = field(default_factory=MediatorConfig)  # 0.10.0

    @property
    def n_covariates(self) -> int:
        """Count the covariates across all configured graph roles."""
        return sum(int(getattr(self.graph, attr)) for attr in ROLE_COUNTERS.values())

    def to_dict(self) -> dict:
        """Convert the scenario configuration and its nested dataclasses to a dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict, strict: bool = True) -> "ScenarioConfig":
        """Load from a dictionary (YAML).

        With strict=True (default since 0.5.0), an unknown key raises ValueError with
        its path.
        """
        d = dict(d)
        for key, sub in [
            ("graph", GraphConfig),
            ("mechanism", MechanismConfig),
            ("tau", TauConfig),
            ("treatment", TreatmentConfig),
            ("outcome", OutcomeConfig),
            ("segments", SegmentConfig),
            ("mediator", MediatorConfig),
        ]:
            if key in d and isinstance(d[key], dict):
                d[key] = _from_dict(sub, d[key], strict, key)
        if "pathologies" in d:
            d["pathologies"] = [
                _from_dict(PathologyConfig, p, strict, f"pathologies[{i}]")
                if isinstance(p, dict)
                else p
                for i, p in enumerate(d["pathologies"])
            ]
        return _from_dict(cls, d, strict, "scenario")

    @classmethod
    def from_yaml(
        cls, path: str | pathlib.Path, strict: bool = True
    ) -> "ScenarioConfig":
        """Read a scenario configuration from a YAML file."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f), strict=strict)

    def to_yaml(self, path: str | pathlib.Path) -> None:
        """Write this scenario configuration to a YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)

    # Scenario validation covers parameter values, family settings and pathology phase order.

    def validate(self) -> None:
        """Validate scenario parameters and the compatibility of its axes.

        Check types, finiteness, ranges, allowed values, applicability, and phase order.
        ValueError includes the field path. graph.sample_graph checks structural
        constraints on adjacency, discrete nodes, and X–X structures because they depend
        on the sampled roles.
        """
        g, m, t, tr, o, s = (
            self.graph,
            self.mechanism,
            self.tau,
            self.treatment,
            self.outcome,
            self.segments,
        )
        for name in (
            *ROLE_COUNTERS.values(),
            "n_mediators",
            "n_binary",
            "n_categorical",
        ):
            _check_count(getattr(g, name), f"graph.{name}")
        if self.n_covariates < 1:
            msg = "graph: at least one covariate is required"
            raise ValueError(msg)
        self._validate_roles_and_mediation()
        _check_range(g.xx_density, "graph.xx_density", 0.0, 1.0)
        _check_range(g.p_in, "graph.p_in", 0.0, 1.0)
        _check_range(g.p_out, "graph.p_out", 0.0, 1.0)
        _check_range(g.layer_skip, "graph.layer_skip", 0.0, 1.0)
        _check_range(g.copy_share, "graph.copy_share", 0.0, 1.0)
        _check_range(g.copy_fidelity, "graph.copy_fidelity", 0.0, 1.0)
        _check_number(g.sf_gamma, "graph.sf_gamma")
        _check_number(g.categorical_skew, "graph.categorical_skew")
        for name in ("blocks_k", "layers_l"):
            v = getattr(g, name)
            if not _is_int(v) or v < 1:
                msg = f"graph.{name} must be an integer >= 1, got {v!r}"
                raise ValueError(msg)
        if g.max_in_degree is not None:
            _check_count(g.max_in_degree, "graph.max_in_degree")
        for v in g.binary_rate if isinstance(g.binary_rate, list) else [g.binary_rate]:
            _check_range(v, "graph.binary_rate", 0.0, 1.0, lo_open=True, hi_open=True)
        for v in (
            g.categorical_levels
            if isinstance(g.categorical_levels, list)
            else [g.categorical_levels]
        ):
            if not _is_int(v) or v < 2:
                msg = f"graph.categorical_levels must be an integer >= 2, got {v!r}"
                raise ValueError(msg)
        _check_enum(m.kind, "mechanism.kind", ("linear", "nonlinear"))
        _check_range(m.interaction, "mechanism.interaction", 0.0, 1.0)
        _check_enum(t.form, "tau.form", ("linear", "nonlinear", "segment"))
        _check_range(t.sparsity, "tau.sparsity", 0.0, 1.0, hi_open=True)
        _check_number(t.amplitude, "tau.amplitude")
        if t.amplitude < 0:
            msg = f"tau.amplitude must be >= 0, got {t.amplitude!r}"
            raise ValueError(msg)
        _check_number(t.offset, "tau.offset")
        _check_enum(tr.mode, "treatment.mode", ("rct", "obs"))
        _check_range(
            tr.propensity, "treatment.propensity", 0.0, 1.0, lo_open=True, hi_open=True
        )
        _check_number(tr.confounding, "treatment.confounding")
        _check_enum(
            o.family,
            "outcome.family",
            ("gaussian", "binary", "ziln", "beta", "lognormal"),
        )
        _check_enum(o.coupling, "outcome.coupling", ("shared", "independent"))
        _check_number(o.snr, "outcome.snr")
        if o.snr <= 0:
            msg = f"outcome.snr must be > 0, got {o.snr!r}"
            raise ValueError(msg)
        _check_range(
            o.base_rate, "outcome.base_rate", 0.0, 1.0, lo_open=True, hi_open=True
        )
        _check_range(
            o.zero_mass, "outcome.zero_mass", 0.0, 1.0, lo_open=True, hi_open=True
        )
        _check_type(
            o.heavy_tails,
            bool,
            f"outcome.heavy_tails must be a boolean, got {o.heavy_tails!r}",
        )
        # Family-specific settings: snr and heavy_tails apply only to gaussian outcomes.
        if o.family != "gaussian":
            if o.snr != 1.0:
                msg = (
                    f"outcome.snr applies only to gaussian outcomes. For {o.family}, leave it at 1.0 "
                    f"(got {o.snr!r}). Set the effect scale with tau.amplitude."
                )
                raise ValueError(msg)
            if o.heavy_tails:
                msg = f"outcome.heavy_tails applies only to gaussian outcomes (family {o.family})"
                raise ValueError(msg)
        _check_number(o.scale_sigma, "outcome.scale_sigma")
        if o.scale_sigma < 0:
            msg = f"outcome.scale_sigma must be >= 0, got {o.scale_sigma!r}"
            raise ValueError(msg)
        if o.scale_sigma > 0 and o.family != "gaussian":
            msg = f"outcome.scale_sigma applies only to gaussian outcomes (family {o.family})"
            raise ValueError(msg)
        if o.scale_feature is not None:
            if o.scale_sigma == 0:
                msg = (
                    "outcome.scale_feature has no effect unless outcome.scale_sigma > 0"
                )
                raise ValueError(msg)
            _check_index(o.scale_feature, "outcome.scale_feature", self.n_covariates)
        if o.family == "beta":
            _check_range(
                o.base_mean, "outcome.base_mean", 0.0, 1.0, lo_open=True, hi_open=True
            )
            _check_range(
                o.beta_phi, "outcome.beta_phi", 0.0, BETA_PHI_MAX, lo_open=True
            )
            _check_range(
                o.zero_mass_beta, "outcome.zero_mass_beta", 0.0, 1.0, hi_open=True
            )
            _check_range(
                o.one_mass_beta, "outcome.one_mass_beta", 0.0, 1.0, hi_open=True
            )
            if o.zero_mass_beta + o.one_mass_beta >= 1.0:
                msg = (
                    f"outcome.zero_mass_beta + outcome.one_mass_beta must be < 1, got "
                    f"{o.zero_mass_beta + o.one_mass_beta!r}"
                )
                raise ValueError(msg)
            _check_enum(o.beta_tau_on, "outcome.beta_tau_on", ("mean", "mass", "both"))
            _check_number(o.beta_mass_scale, "outcome.beta_mass_scale")
            if o.beta_mass_scale < 0:
                msg = f"outcome.beta_mass_scale must be >= 0, got {o.beta_mass_scale!r}"
                raise ValueError(msg)
            if (
                o.beta_tau_on != "mean"
                and o.zero_mass_beta == 0
                and o.one_mass_beta == 0
            ):
                msg = (
                    f"outcome.beta_tau_on={o.beta_tau_on!r} requires mass at 0 or 1 "
                    f"(zero_mass_beta or one_mass_beta > 0): without point masses there is no mass effect channel"
                )
                raise ValueError(msg)
        self._validate_lognormal_and_lags()
        if not _is_int(s.n_segments) or s.n_segments < 1:
            msg = f"segments.n_segments must be an integer >= 1, got {s.n_segments!r}"
            raise ValueError(msg)
        if s.features is not None:
            if not 1 <= len(s.features) <= 2:
                msg = f"segments.features requires one or two features, got {len(s.features)}"
                raise ValueError(msg)
            for f in s.features:
                if not isinstance(f, str):
                    _check_index(f, "segments.features", self.n_covariates)
        self._validate_pathologies()

    def _validate_lognormal_and_lags(self) -> None:
        """Version 0.11.0: lognormal outcomes, tau.level_slope, and pretreatment outcome lags."""
        o, t, g = self.outcome, self.tau, self.graph
        _check_number(t.level_slope, "tau.level_slope")
        if t.level_slope != 0 and o.family != "lognormal":
            msg = (
                f"tau.level_slope applies only to outcome.family='lognormal' (family {o.family}): "
                f"the relative effect is defined only for positive outcomes"
            )
            raise ValueError(msg)
        if o.family == "lognormal":
            _check_number(o.base_level, "outcome.base_level")
            if o.base_level <= 0:
                msg = f"outcome.base_level must be > 0, got {o.base_level!r}"
                raise ValueError(msg)
            _check_nonneg(o.level_sigma, "outcome.level_sigma")
            _check_number(o.noise_sigma, "outcome.noise_sigma")
            if o.noise_sigma <= 0:
                msg = f"outcome.noise_sigma must be > 0, got {o.noise_sigma!r}"
                raise ValueError(msg)
            _check_range(o.zero_mass_ln, "outcome.zero_mass_ln", 0.0, 1.0, hi_open=True)
            if g.n_mediators > 0:
                msg = (
                    "outcome.family='lognormal' does not support mediators (graph.n_mediators > 0): "
                    "the contribution of mediators on the log-outcome scale is undefined"
                )
                raise ValueError(msg)
        _check_count(o.pre_periods, "outcome.pre_periods")
        _check_range(o.pre_rho, "outcome.pre_rho", 0.0, 1.0, hi_open=True)
        _check_number(o.pre_drift, "outcome.pre_drift")
        if o.pre_periods > 0:
            if o.family not in ("gaussian", "lognormal"):
                msg = (
                    "outcome.pre_periods > 0 supports outcome lags only for gaussian and lognormal "
                    f"outcomes (got {o.family})"
                )
                raise ValueError(msg)
            if o.heavy_tails:
                msg = "outcome.pre_periods > 0 does not support heavy_tails: combining a persistent component with t(3) noise does not preserve the t(3) distribution"
                raise ValueError(msg)
            if g.n_mediators > 0:
                msg = "outcome.pre_periods > 0 does not support mediators (graph.n_mediators > 0)"
                raise ValueError(msg)
            if self.treatment.mode != "rct":
                msg = (
                    "outcome.pre_periods > 0 requires treatment.mode='rct' (using lags as placebo outcomes in "
                    "observational designs is outside the scope of 0.11.0)"
                )
                raise ValueError(msg)
            if o.family == "lognormal" and o.zero_mass_ln > 0:
                msg = (
                    "outcome.pre_periods > 0 does not support zero_mass_ln > 0: "
                    "the point mass at zero is undefined for lagged outcomes"
                )
                raise ValueError(msg)

    def _validate_roles_and_mediation(self) -> None:
        """Version 0.10.0: manual role assignment, direct_effect, and MediatorConfig."""
        g, t, md = self.graph, self.tau, self.mediator
        _check_type(
            g.direct_effect,
            bool,
            f"graph.direct_effect must be a boolean, got {g.direct_effect!r}",
        )
        check_roles(g)
        if g.n_upstream + g.n_side > 0 and g.xx_density == 0 and g.adjacency is None:
            msg = (
                "graph.n_upstream or n_side is positive with xx_density = 0 and no explicit adjacency: "
                "without edges between covariates, these roles are indistinguishable from noise"
            )
            raise ValueError(msg)
        n_mp = g.n_med_confounders + g.n_med_predictors
        n_theta = g.n_modifiers + g.n_med_confounders
        _check_enum(md.family, "mediator.family", ("gaussian", "binary"))
        _check_nonneg(md.base_amplitude, "mediator.base_amplitude")
        if md.family == "binary":
            if self.outcome.family != "gaussian":
                msg = (
                    f"mediator.family='binary' applies only to outcome.family='gaussian' "
                    f"(family {self.outcome.family})"
                )
                raise ValueError(msg)
            if g.n_mediators < 1:
                msg = "mediator.family='binary' has no effect without mediators (graph.n_mediators = 0)"
                raise ValueError(msg)
            _check_range(
                md.base_rate, "mediator.base_rate", 0.0, 1.0, lo_open=True, hi_open=True
            )
            _check_number(md.effect, "mediator.effect")
            _check_nonneg(md.effect_amplitude, "mediator.effect_amplitude")
            _check_range(
                md.effect_sparsity, "mediator.effect_sparsity", 0.0, 1.0, hi_open=True
            )
            _check_number(md.theta, "mediator.theta")
            if md.theta == 0:
                msg = "mediator.theta must be nonzero (a mediator with no M -> Y effect contributes nothing)"
                raise ValueError(msg)
            _check_nonneg(md.theta_amplitude, "mediator.theta_amplitude")
            if (md.effect_amplitude > 0 or md.effect_sparsity > 0) and n_mp == 0:
                msg = (
                    "mediator.effect_amplitude or effect_sparsity > 0 requires X parents of the mediator "
                    "(graph.n_med_confounders / n_med_predictors)"
                )
                raise ValueError(msg)
            if md.theta_amplitude > 0 and n_theta == 0:
                msg = (
                    "mediator.theta_amplitude > 0 requires modifiers or med_confounder nodes "
                    "(graph.n_modifiers / n_med_confounders)"
                )
                raise ValueError(msg)
        if not g.direct_effect:
            if g.n_mediators < 1:
                msg = (
                    "graph.direct_effect = False requires n_mediators >= 1: without a direct edge "
                    "or mediators, T has no effect on Y"
                )
                raise ValueError(msg)
            if t.amplitude != 0 or t.offset != 0 or t.form == "segment":
                msg = (
                    "graph.direct_effect = False means there is no direct effect. Set tau.amplitude = 0, "
                    f"tau.offset = 0, and tau.form != 'segment' (got amplitude = {t.amplitude!r}, "
                    f"offset = {t.offset!r}, form = {t.form!r})"
                )
                raise ValueError(msg)
            if g.n_modifiers > 0 and not (
                md.family == "binary" and md.theta_amplitude > 0
            ):
                msg = (
                    "graph.direct_effect = False with n_modifiers > 0 leaves the modifier -> Y edge unused. "
                    "With no direct effect, a modifier requires mediator.family = 'binary' "
                    "with theta_amplitude > 0."
                )
                raise ValueError(msg)

    def _validate_pathologies(self) -> None:
        last_phase, last_kind = 0, None
        shares: dict[str, float] = {}
        seen: list[str] = []
        kinds_all = [p.kind for p in self.pathologies if isinstance(p, PathologyConfig)]
        for i, p in enumerate(self.pathologies):
            path = f"pathologies[{i}]"
            _check_type(p, PathologyConfig, f"{path}: expected PathologyConfig")
            phase = pathology_phase(p)  # Raises ValueError for an unknown axis.
            _check_range(p.strength, f"{path}.strength", 0.0, 1.0)
            _check_enum(p.target, f"{path}.target", ("global", "segments"))
            _check_range(p.segment_share, f"{path}.segment_share", 0.0, 1.0)
            if p.segments is not None:
                for sgm in p.segments:
                    if not _is_int(sgm) or sgm < 0:
                        msg = f"{path}.segments: segment IDs must be nonnegative integers, got {sgm!r}"
                        raise ValueError(msg)
            if p.kind in GLOBAL_ONLY and p.target != "global":
                msg = f"{path}: pathology {p.kind} is inherently global: target must be 'global'"
                raise ValueError(msg)
            _check_type(p.params, dict, f"{path}.params must be a dictionary")
            unknown = set(p.params) - PATHOLOGY_PARAMS[p.kind]
            if unknown:
                msg = (
                    f"{path}.params.{min(unknown)}: unknown parameter for {p.kind} "
                    f"(allowed: {sorted(PATHOLOGY_PARAMS[p.kind])})"
                )
                raise ValueError(msg)
            self._validate_params(p, path, kinds_all)
            if phase < last_phase:
                msg = (
                    f"{path}: pathology {p.kind} (phase {PHASE_NAMES[phase]}) cannot follow "
                    f"{last_kind} (phase {PHASE_NAMES[last_phase]}). Required order: DGP -> selection -> measurement"
                )
                raise ValueError(msg)
            if p.kind in NO_REPEAT and p.kind in seen:
                msg = (
                    f"{path}: repeating {p.kind} in one scenario is undefined: "
                    f"use a single axis to express the combined action"
                )
                raise ValueError(msg)
            if p.kind in DRIFT_KINDS:
                share = drift_share(p)
                for other, s0 in shares.items():
                    if s0 != share:
                        msg = (
                            f"{path}.params.share = {share} differs from share = {s0} for {other}: "
                            "all drift mechanisms in a scenario share one period indicator"
                        )
                        raise ValueError(msg)
                shares[p.kind] = share
            if p.kind == "covariate_shift":
                bad = [k for k in seen if k in AFTER_COVARIATE_SHIFT]
                if bad:
                    msg = (
                        f"{path}: covariate_shift recomputes second-period X / mu0_x / tau_x / e from the SCM "
                        f"and must precede the data-generating axis {bad[0]} (unit type or rotation is drawn "
                        f"using X)"
                    )
                    raise ValueError(msg)
            seen.append(p.kind)
            last_phase, last_kind = phase, p.kind

    def _validate_params(self, p: PathologyConfig, path: str, kinds_all: list) -> None:
        fam, mode = self.outcome.family, self.treatment.mode
        q = p.params
        d = self.n_covariates
        if p.kind in ("overlap", "targeted_selection") and mode != "obs":
            msg = f"{path}: {p.kind} requires treatment.mode='obs'"
            raise ValueError(msg)
        if p.kind == "heteroscedastic" and fam == "binary":
            msg = f"{path}: heteroscedastic does not apply to binary outcomes"
            raise ValueError(msg)
        if (
            p.kind in ("hidden_modifier", "effect_drift")
            and not self.graph.direct_effect
        ):
            msg = (
                f"{path}: {p.kind} changes the direct T -> Y effect, which is absent when "
                f"graph.direct_effect = False (drift or a hidden component of the mediated effect is outside scope)"
            )
            raise ValueError(msg)
        if "features" in q and q["features"] is not None:
            if not isinstance(q["features"], list) or not q["features"]:
                msg = f"{path}.params.features must be a nonempty list of node indices"
                raise ValueError(msg)
            for f in q["features"]:
                _check_index(f, f"{path}.params.features", d)
        if "feature" in q:
            _check_index(q["feature"], f"{path}.params.feature", d)
        if p.kind in ("noise_y", "outliers") and q.get("by_arm") is not None:
            if not isinstance(q["by_arm"], dict) or not q["by_arm"]:
                msg = f"{path}.params.by_arm must be a dictionary {{0: k0, 1: k1}}"
                raise ValueError(msg)
            for k, v in q["by_arm"].items():
                if str(k) not in ("0", "1"):
                    msg = f"{path}.params.by_arm: expected a key of 0 or 1, got {k!r}"
                    raise ValueError(msg)
                _check_number(v, f"{path}.params.by_arm[{k}]")
                if v < 0:
                    msg = f"{path}.params.by_arm[{k}] must be >= 0"
                    raise ValueError(msg)
        if p.kind == "outliers":  # 0.9.0
            if self.outcome.family in ("binary", "beta"):
                msg = (
                    f"{path}: outliers is unsupported for {self.outcome.family} outcomes "
                    "because additive recording errors can put values outside the outcome's allowed range"
                )
                raise ValueError(msg)
            _check_enum(
                q.get("sides", "both"), f"{path}.params.sides", ("both", "up", "down")
            )
            if (
                self.outcome.family in ("ziln", "lognormal")
                and q.get("sides", "both") != "up"
            ):
                msg = (
                    f"{path}.params.sides: for {self.outcome.family}, outliers can only increase positive "
                    f"outcomes (sides='up') to keep them positive. Got {q.get('sides', 'both')!r}."
                )
                raise ValueError(msg)
            if q.get("share") is not None:
                _check_range(q["share"], f"{path}.params.share", 0.0, 1.0)
            if "magnitude" in q:
                _check_number(q["magnitude"], f"{path}.params.magnitude")
                if q["magnitude"] <= 0:
                    msg = f"{path}.params.magnitude must be > 0, got {q['magnitude']!r}"
                    raise ValueError(msg)
            if "spread" in q:
                _check_number(q["spread"], f"{path}.params.spread")
                if q["spread"] < 0:
                    msg = f"{path}.params.spread must be >= 0, got {q['spread']!r}"
                    raise ValueError(msg)
        if p.kind == "noise_x":
            _check_enum(
                q.get("mode", "gaussian"),
                f"{path}.params.mode",
                ("gaussian", "quantize"),
            )
            if "bins" in q and (not _is_int(q["bins"]) or q["bins"] < 2):
                msg = f"{path}.params.bins must be an integer >= 2"
                raise ValueError(msg)
            if "roles" in q and q["roles"] is not None:
                bad = set(q["roles"]) - set(ROLE_COUNTERS)
                if bad:
                    msg = f"{path}.params.roles: unknown role {min(bad)!r}"
                    raise ValueError(msg)
        if p.kind == "missing":
            _check_enum(
                q.get("mode", "mcar"), f"{path}.params.mode", ("mcar", "mar", "mnar")
            )
            cols = set(q.get("columns", ["x"]))
            bad = cols - {"x", "t", "y"}
            if bad:
                msg = f"{path}.params.columns: expected x, t, or y, got {sorted(bad)}"
                raise ValueError(msg)
            sent = q.get("sentinel")
            if sent is not None:
                if cols & {"t", "y"}:
                    msg = f"{path}.params.sentinel: sentinels cannot encode t or y. columns must contain only x."
                    raise ValueError(msg)
                if isinstance(sent, dict):
                    for k, v in sent.items():
                        _check_number(v, f"{path}.params.sentinel[{k}]")
                elif sent != "auto":
                    _check_number(sent, f"{path}.params.sentinel")
        if p.kind == "selection":
            _check_number(q.get("t_weight", 1.0), f"{path}.params.t_weight")
        if p.kind == "targeted_selection":
            _check_number(q.get("direction", -1.0), f"{path}.params.direction")
        if p.kind == "noncompliance":
            _check_enum(q.get("side", "one"), f"{path}.params.side", ("one", "two"))
            _check_range(q.get("hidden", 0.5), f"{path}.params.hidden", 0.0, 1.0)
            if self.outcome.pre_periods > 0 and q.get("hidden", 0.5) > 0:
                msg = (
                    f"{path}.params.hidden > 0 is incompatible with outcome.pre_periods > 0: the compliance-type "
                    "score uses outcome noise that the lags partially reveal. Set hidden: 0."
                )
                raise ValueError(msg)
        # Outcome rounding is defined only for continuous measurements.
        if p.kind == "quantize_y":
            if fam in ("binary", "beta"):
                msg = f"{path}: quantize_y does not apply to family {fam}"
                raise ValueError(msg)
            given = [k for k in ("step", "step_sd") if q.get(k) is not None]
            if len(given) > 1:
                msg = (
                    f"{path}.params: specify at most one of step or step_sd (the default is step_sd = "
                    f"{QUANTIZE_Y_STEP_SD}), got {given}"
                )
                raise ValueError(msg)
            for k in given:
                _check_number(q[k], f"{path}.params.{k}")
                if q[k] <= 0:
                    msg = f"{path}.params.{k} must be > 0, got {q[k]!r}"
                    raise ValueError(msg)
        if p.kind == "bad_controls":
            bad = set(q.get("expose", ["mediator", "y_descendant"])) - {
                "mediator",
                "y_descendant",
            }
            if bad:
                msg = f"{path}.params.expose: expected mediator or y_descendant, got {sorted(bad)}"
                raise ValueError(msg)
            if "n_y_descendants" in q and (
                not _is_int(q["n_y_descendants"]) or q["n_y_descendants"] < 0
            ):
                msg = f"{path}.params.n_y_descendants must be a nonnegative integer"
                raise ValueError(msg)
        if p.kind in DRIFT_KINDS:
            _check_range(
                q.get("share", 0.5), f"{path}.params.share", 0.0, 1.0, lo_open=True
            )
        if p.kind == "effect_drift":
            if "shift_tau" in q:
                _check_range(q["shift_tau"], f"{path}.params.shift_tau", 0.0, 1.0)
            if "scale_tau" in q:
                _check_number(q["scale_tau"], f"{path}.params.scale_tau")
                if q["scale_tau"] < 0:
                    msg = f"{path}.params.scale_tau must be >= 0"
                    raise ValueError(msg)
            if "shift_ate" in q:
                _check_number(q["shift_ate"], f"{path}.params.shift_ate")
            if not any(
                _drift_param_active(q, k)
                for k in ("shift_tau", "scale_tau", "shift_ate")
            ):
                msg = (
                    f"{path}: effect_drift requires at least one of shift_tau, scale_tau, or shift_ate "
                    f"(the axis does not use strength)"
                )
                raise ValueError(msg)
        if p.kind == "cluster":
            if "n_clusters" in q and (
                not _is_int(q["n_clusters"]) or q["n_clusters"] < 2
            ):
                msg = f"{path}.params.n_clusters must be an integer >= 2"
                raise ValueError(msg)
            if "size_sigma" in q:
                _check_number(q["size_sigma"], f"{path}.params.size_sigma")
                if q["size_sigma"] < 0:
                    msg = f"{path}.params.size_sigma must be >= 0"
                    raise ValueError(msg)
            icc_e = q.get("icc_e", 0.0)
            _check_range(icc_e, f"{path}.params.icc_e", 0.0, 1.0, hi_open=True)
            _check_range(q.get("rho", 1.0), f"{path}.params.rho", -1.0, 1.0)
            assign = q.get("assign", "individual")
            _check_enum(
                assign, f"{path}.params.assign", ("individual", "cluster", "block")
            )
            bp = q.get("block_propensity")
            if bp is not None:
                if assign != "block":
                    msg = f"{path}.params.block_propensity has no effect unless assign='block'"
                    raise ValueError(msg)
                if isinstance(bp, list):
                    if len(bp) != int(q.get("n_clusters", 20)):
                        msg = (
                            f"{path}.params.block_propensity: expected a list of length n_clusters = "
                            f"{int(q.get('n_clusters', 20))}, got {len(bp)}"
                        )
                        raise ValueError(msg)
                    for v in bp:
                        _check_range(v, f"{path}.params.block_propensity[]", 0.0, 1.0)
                else:
                    _check_number(bp, f"{path}.params.block_propensity")
                    if bp <= 0:
                        msg = (
                            f"{path}.params.block_propensity (spread of the block proportions on the logit scale) must be > 0, "
                            f"got {bp!r}"
                        )
                        raise ValueError(msg)
            if "emit" in q and not isinstance(q["emit"], bool):
                msg = f"{path}.params.emit must be a boolean"
                raise ValueError(msg)
            if p.strength >= 1.0:
                msg = f"{path}.strength (ICC) must be < 1"
                raise ValueError(msg)
            if icc_e > 0 and mode != "obs":
                msg = f"{path}.params.icc_e requires treatment.mode='obs'"
                raise ValueError(msg)
            if (
                p.strength == 0
                and not q.get("emit", False)
                and (icc_e > 0 or assign in ("cluster", "block"))
            ):
                msg = (
                    f"{path}: cluster has no effect at strength = 0. To use icc_e or assign={assign} with "
                    f"no effect on Y, set params.emit = true"
                )
                raise ValueError(msg)
            if assign in ("cluster", "block"):
                # Cluster randomization requires a constant assignment probability within each cluster.
                if icc_e > 0:
                    msg = (
                        f"{path}.params.assign={assign!r} is incompatible with icc_e > 0: "
                        "assignment probability varies within a cluster"
                    )
                    raise ValueError(msg)
                if mode != "rct":
                    msg = (
                        f"{path}.params.assign={assign!r} requires treatment.mode='rct'. In observational designs e depends "
                        f"on each row's X and varies within a cluster"
                    )
                    raise ValueError(msg)
                incompatible = [k for k in kinds_all if k in NON_CLUSTER_CONSTANT_E]
                if incompatible:
                    msg = (
                        f"{path}.params.assign={assign!r} is incompatible with {incompatible[0]}: "
                        "assignment is no longer constant within a cluster"
                    )
                    raise ValueError(msg)
        if p.kind == "duplicate_rows":
            if "k" in q and (not _is_int(q["k"]) or q["k"] < 1):
                msg = f"{path}.params.k must be an integer >= 1"
                raise ValueError(msg)
            if "noise" in q:
                _check_number(q["noise"], f"{path}.params.noise")
                if q["noise"] < 0:
                    msg = f"{path}.params.noise must be >= 0"
                    raise ValueError(msg)
            _check_enum(
                q.get("position", "end"), f"{path}.params.position", ("end", "shuffle")
            )


def _is_number(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


def _check_type(value: object, expected: type, message: str) -> None:
    """Invalid configuration field types raise ValueError, just like invalid values."""
    if isinstance(value, expected):
        return
    raise ValueError(message)


def _is_int(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def _check_number(v, path: str) -> None:
    """Require a finite number (int or float, excluding bool)."""
    if not _is_number(v):
        msg = f"{path} must be a number, got {v!r}"
        raise ValueError(msg)
    if not math.isfinite(v):
        msg = f"{path} must be finite, got {v!r}"
        raise ValueError(msg)


def _check_nonneg(v, path: str) -> None:
    _check_number(v, path)
    if v < 0:
        msg = f"{path} must be >= 0, got {v!r}"
        raise ValueError(msg)


def _check_count(v, path: str) -> None:
    if not _is_int(v) or v < 0:
        msg = f"{path} must be a nonnegative integer, got {v!r}"
        raise ValueError(msg)


def _check_index(v, path: str, n: int) -> None:
    """Validate an X node index: an integer in [0, n)."""
    if not _is_int(v) or not 0 <= v < n:
        msg = f"{path}: node index must be an integer in [0, {n}), got {v!r}"
        raise ValueError(msg)


def _check_range(
    v, path: str, lo: float, hi: float, lo_open: bool = False, hi_open: bool = False
) -> None:
    _check_number(v, path)
    ok_lo = v > lo if lo_open else v >= lo
    ok_hi = v < hi if hi_open else v <= hi
    if not (ok_lo and ok_hi):
        lb, rb = ("(" if lo_open else "["), (")" if hi_open else "]")
        msg = f"{path} must lie in {lb}{lo}, {hi}{rb}, got {v!r}"
        raise ValueError(msg)


def _check_enum(v, path: str, allowed: tuple) -> None:
    if v not in allowed:
        msg = f"{path} must be one of {'|'.join(allowed)}, got {v!r}"
        raise ValueError(msg)
