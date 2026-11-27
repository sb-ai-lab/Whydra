"""Define an SCM independently of the sample drawn from it.

Model ground truth is independent of n. The structural equations use the graph,
coefficients (`params`, drawn with model_seed), and normalization constants (`calib`,
computed once over n_calib rows of exogenous noise from the (model_seed, 0xCA11B)
stream). `scm.sample(n, sample_seed)` draws fresh exogenous noise for any n with the
model fixed. Model, sample, and pathology randomness are separate, with a dedicated
stream for model calibration:

* Model: default_rng(model_seed) calls sample_graph (graph and discrete nodes match
0.5.0 for the same seed), then draws X–X weights by node, μ₀, τ, gate, observational e,
and mediator parameters.

* Calibration: default_rng((model_seed, 0xCA11B)) supplies one stream shared across
noise kinds.

* Sample: default_rng((sample_seed, 0x5A3E, code(kind))) gives each noise kind its own
stream. Kinds are x, y, ln, t, med, coll, y1, ln1, z0, z1, and pre. y1 and ln1 are the
second outcome noise under independent coupling (0.8.0). z0 and z1 implement the
lognormal point mass at zero, and pre supplies outcome lags (0.11.0). Each block is
drawn in one call for n rows. Row-wise arithmetic uses mechanisms.rowdot without BLAS,
so for n₂ > n₁ the first n₁ rows of sample(n₂, s) match sample(n₁, s) bit for bit on any
given CPU.

* Pathologies: default_rng((pathology_seed, 0x9A70, i, code(kind))), as in 0.5.0
(truth.new_axis_rng).

The SCM is immutable. It is a frozen dataclass: params and calib are read-only
mappings over read-only arrays, with lists stored as tuples. graph is a frozen
SampledGraph (0.7.1). Its frozen flag makes attribute assignment, deletion, and
add_edges_to_y raise AttributeError. It holds read-only matrices, tuples of roles and
names, and a MappingProxyType over tuples for discrete data. The config property returns
a deep copy of the snapshot, so editing it leaves the model intact. Dataset.graph
provides a mutable copy via thaw_graph.

Because of this, sample_scm(config, model_seed, n_calib) can return a cached instance.
The cache key uses canonical config JSON, model_seed, and n_calib, avoiding
recalibration when repeated generate calls vary n. SCM.to_json and SCM.from_json
serialize the model artifact, scm.json in a dataset directory. scmgen.io.replay(dir)
reproduces data bit for bit from the recipe in the reference environment.
"""

import copy
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd

from .config import ScenarioConfig
from .graph import (
    ROLE_COLLIDER,
    ROLE_CONFOUNDER,
    ROLE_MED_CONFOUNDER,
    ROLE_MED_PREDICTOR,
    ROLE_MODIFIER,
    ROLE_PREDICTOR,
    SampledGraph,
    sample_graph,
)
from .mechanisms import (
    apply_covariates,
    apply_mediators,
    apply_mu0,
    apply_propensity,
    apply_tau,
    draw_mediators,
    draw_mu0,
    draw_tau,
    draw_treatment,
    draw_xx_weights,
    mediator_contributions,
)
from .outcomes import outcome_scale, realized_outcome
from .segments import compact_segments, grid_shape, raw_cells, segment_features

SCM_SCHEMA = 1
CALIB_STREAM = 0xCA11B
SAMPLE_STREAM = 0x5A3E
NOISE_KINDS = ("x", "y", "ln", "t", "med", "coll", "y1", "ln1", "z0", "z1", "pre")
DEFAULT_N_CALIB = 200_000
_CACHE_SIZE = 16
_BLOCK = 20_000  # Rows per calibration aggregation block.


def stream_code(kind: str) -> int:
    """Return the noise or axis stream code: the first four bytes of SHA-256 of its name."""
    return int.from_bytes(hashlib.sha256(kind.encode()).digest()[:4], "little")


def sample_rng(sample_seed: int, kind: str) -> np.random.Generator:
    """Return the sampling stream for exogenous noise of the given kind."""
    return np.random.default_rng((int(sample_seed), SAMPLE_STREAM, stream_code(kind)))


def config_sha256(config: ScenarioConfig) -> str:
    """Hash canonical config JSON (sorted keys) with SHA-256 to identify the scenario in the manifest."""
    return hashlib.sha256(_canonical_json(config).encode()).hexdigest()


def _canonical_json(config: ScenarioConfig) -> str:
    return json.dumps(
        config.to_dict(), sort_keys=True, ensure_ascii=False, default=_json_default
    )


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Mapping):
        return dict(o)
    msg = f"cannot serialize {type(o).__name__} to JSON"
    raise TypeError(msg)


# Freeze and thaw data.


def _freeze(obj):
    """Return a recursively immutable representation of nested data.

    Copy arrays and mark them read-only, convert lists to tuples, and wrap dictionaries
    in MappingProxyType. Scalars and None pass through unchanged.
    """
    if isinstance(obj, np.ndarray):
        a = np.array(obj, copy=True)
        a.flags.writeable = False
        return a
    if isinstance(obj, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(v) for v in obj)
    return obj


def _thaw(obj):
    """Return mutable copies: writable arrays, lists for tuples, and dictionaries for mappings."""
    if isinstance(obj, np.ndarray):
        return np.array(obj, copy=True)
    if isinstance(obj, Mapping):
        return {k: _thaw(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_thaw(v) for v in obj]
    return obj


def freeze_graph(g: SampledGraph) -> SampledGraph:
    """Return a new, immutable copy of a graph for SCM.graph.

    The frozen SampledGraph has read-only adjacency and xx_edges, tuples of roles and
    names, and discrete data transformed by _freeze. The _frozen flag protects
    attributes and edges (see SampledGraph.__setattr__).
    """
    d = {f.name: getattr(g, f.name) for f in dataclasses.fields(g)}
    for k in ("adjacency", "xx_edges"):
        d[k] = _freeze(np.asarray(d[k]))
    for k in (
        "roles",
        "node_names",
        "roles_effective",
        "blocks",
        "layers",
        "prototype",
    ):
        if d[k] is not None:
            d[k] = tuple(d[k])
    d["discrete"] = _freeze(d["discrete"])  # None passes through unchanged.
    out = SampledGraph(**d)
    object.__setattr__(out, "_frozen", True)
    return out


def thaw_graph(g: SampledGraph) -> SampledGraph:
    """Return a mutable deep copy for Dataset.graph, with writable arrays, lists, and dict/list discrete data."""
    d = {f.name: getattr(g, f.name) for f in dataclasses.fields(g)}
    for k in ("adjacency", "xx_edges"):
        d[k] = np.array(d[k], copy=True)
    for k in (
        "roles",
        "node_names",
        "roles_effective",
        "blocks",
        "layers",
        "prototype",
    ):
        if d[k] is not None:
            d[k] = list(d[k])
    d["discrete"] = _thaw(d["discrete"])
    return SampledGraph(**d)


@dataclass(frozen=True, init=False)
class SCM:
    """Define a scenario's structural causal model: graph, equation parameters, and calibration constants.

    Instances are immutable (0.7.0). For example, assigning scm.params["tau_w"][0]
    = 1 raises ValueError because the array is read-only, and scm.config returns a copy
    of the snapshot.
    """

    model_seed: int
    n_calib: int  # Calibration population size.
    graph: SampledGraph  # Read-only roles, adjacency, discrete data (Φ⁻¹ thresholds), and roles_effective.
    params: Mapping  # Read-only coefficients.
    calib: Mapping  # Read-only calibration constants.
    scmgen_version: str
    _config: ScenarioConfig = field(repr=False)

    def __init__(
        self,
        config: ScenarioConfig,
        model_seed: int,
        n_calib: int,
        graph: SampledGraph,
        params: Mapping,
        calib: Mapping,
        scmgen_version: str,
    ):
        s = object.__setattr__
        s(self, "_config", copy.deepcopy(config))
        s(self, "model_seed", int(model_seed))
        s(self, "n_calib", int(n_calib))
        s(self, "graph", freeze_graph(graph))
        s(self, "params", _freeze(params))
        s(self, "calib", _freeze(calib))
        s(self, "scmgen_version", str(scmgen_version))

    @property
    def config(self) -> ScenarioConfig:
        """Return a deep copy of the configuration. Edits to the copy leave the SCM intact."""
        return copy.deepcopy(self._config)

    # Equations on arbitrary true X inputs, with discrete nodes encoded as level indices (n × d).

    def x_from_noise(self, eps: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Compute X from exogenous eps (n × d) using the SEM.

        Colliders equal noise. Return (x, x_latent).
        """
        return apply_covariates(
            self.graph, self._config.mechanism, self.params["xx_w"], eps, self.calib
        )

    def segments(self, x: np.ndarray) -> tuple[np.ndarray, int]:
        """Return segment IDs 0..K−1 and the number of rows in cells unoccupied during calibration."""
        return compact_segments(self._raw_cells(x), self.calib)

    def _raw_cells(self, x: np.ndarray) -> np.ndarray:
        return raw_cells(
            x,
            self.graph,
            self.params["seg_features"],
            tuple(self.params["seg_shape"]),
            self.calib,
        )

    def mu0(self, x: np.ndarray) -> np.ndarray:
        """Return the standardized baseline outcome mu0_x."""
        return apply_mu0(x, self.graph, self._config.mechanism, self.params, self.calib)

    def tau(self, x: np.ndarray) -> np.ndarray:
        """Return structural tau_x.

        Segment effects use the cells of x itself. Return zeros when graph.direct_effect
        = False.
        """
        raw = self._raw_cells(x) if self._config.tau.form == "segment" else None
        return structural_tau(self._config, self.graph, self.params, x, raw, self.calib)

    def mediator_terms(self, x: np.ndarray) -> dict:
        """Compute the deterministic terms in the mediator equations at X.

        For gaussian mediators, med_f is empty if M has no X parents. For binary
        mediators, return med_logit0, med_effect, and med_theta (n × m).
        """
        return apply_mediators(
            x,
            self.graph,
            self._config.mediator,
            self._config.mechanism,
            self.params,
            self.calib,
        )

    def e_assign(self, x: np.ndarray) -> np.ndarray:
        """Return the SCM assignment propensity e_assign(x) before pathology axes contribute."""
        return apply_propensity(
            x, self.graph, self._config.treatment, self.params, self.calib
        )

    def e_logit(self, x: np.ndarray) -> np.ndarray:
        """Return the log odds of treatment, clipping extreme assignment probabilities."""
        e = np.clip(self.e_assign(x), 1e-12, 1 - 1e-12)
        return np.log(e / (1.0 - e))

    def dag(self) -> tuple[np.ndarray, list[str], list[str]]:
        """Return (adjacency, names, roles) for the clean graph, using the effective X roles (roles_effective)."""
        g = self.graph
        assert g.roles_effective is not None
        roles = (
            list(g.roles_effective)
            + ["treatment"]
            + ["mediator"] * self._config.graph.n_mediators
            + ["outcome"]
        )
        return np.array(g.adjacency, copy=True), list(g.node_names), roles

    # Sampling.

    def sample(self, n: int, sample_seed: int):
        """Draw a sample of n rows with the given seed.

        The clean Dataset uses the sample_seed stream for exogenous noise and
        includes the latent state, ground truth, and row_id = 0..n−1. The first n₁ rows
        match the corresponding smaller sample drawn with the same seed bit for bit.
        """
        if int(n) < 2:
            msg = f"n must be >= 2, got {n!r}"
            raise ValueError(msg)
        n = int(n)
        noise = draw_noise(
            lambda kind: sample_rng(sample_seed, kind), n, self.graph, self._config
        )
        dataset = _build(
            self._config,
            self.graph,
            self.params,
            self.calib,
            noise,
            False,
            int(sample_seed),
            self,
        )
        assert dataset is not None
        return dataset

    def _calibration_sample(self):
        """Return the calibration population as a Dataset for exact-equality tests."""
        noise = draw_noise(
            _calibration_rngs(self.model_seed), self.n_calib, self.graph, self._config
        )
        dataset = _build(
            self._config, self.graph, self.params, self.calib, noise, False, None, self
        )
        assert dataset is not None
        return dataset

    # Serialization.

    def to_dict(self) -> dict:
        """Return a dictionary containing the model, calibration and simulation parameters."""
        return {
            "scm_schema": SCM_SCHEMA,
            "scmgen_version": self.scmgen_version,
            "model_seed": int(self.model_seed),
            "n_calib": int(self.n_calib),
            "config": self._config.to_dict(),
            "graph": graph_to_dict(self.graph),
            "params": _jsonable(self.params),
            "calib": _jsonable(self.calib),
        }

    def to_json(self) -> str:
        """Serialize the structural causal model as JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=_json_default)

    @classmethod
    def from_dict(cls, d: dict) -> "SCM":
        """Reconstruct the structural causal model from its serialized fields."""
        import warnings

        from . import __version__

        if d.get("scm_schema") != SCM_SCHEMA:
            msg = (
                f"scm.json: scm_schema = {d.get('scm_schema')!r}, expected {SCM_SCHEMA}"
            )
            raise ValueError(msg)
        if d.get("scmgen_version") != __version__:
            warnings.warn(
                f"scm.json was written by scmgen {d.get('scmgen_version')}, current version {__version__}: "
                f"bitwise reproduction is not guaranteed",
                stacklevel=2,
            )
        return cls(
            config=ScenarioConfig.from_dict(d["config"]),
            model_seed=int(d["model_seed"]),
            n_calib=int(d["n_calib"]),
            graph=graph_from_dict(d["graph"]),
            params=_params_from_json(d["params"]),
            calib=_calib_from_json(d["calib"]),
            scmgen_version=d["scmgen_version"],
        )

    @classmethod
    def from_json(cls, text: str) -> "SCM":
        """Reconstruct the structural causal model from a JSON string."""
        return cls.from_dict(json.loads(text))


# Sample construction and calibration.


def structural_tau(
    cfg: ScenarioConfig,
    graph: SampledGraph,
    params: Mapping,
    x: np.ndarray,
    raw,
    calib,
    fit: bool = False,
) -> np.ndarray:
    """Evaluate the direct treatment effect at X.

    tau_x(x) is the τ equation for a direct T -> Y edge. Without that edge it is zero.
    With direct_effect = False, τ is neither sampled nor used (0.10.0).
    """
    if not cfg.graph.direct_effect:
        return np.zeros(x.shape[0])
    tau = apply_tau(
        x,
        graph,
        cfg.tau,
        params,
        raw if cfg.tau.form == "segment" else None,
        calib,
        fit,
    )
    kappa = float(cfg.tau.level_slope)
    if (
        kappa != 0.0
    ):  # Relative effect increases with baseline level (0.11.0, for positive level_slope).
        lvl = np.exp(
            kappa
            * cfg.outcome.level_sigma
            * apply_mu0(x, graph, cfg.mechanism, params, calib)
        )
        norm = _const(calib, "tau_level_norm", fit, lambda: float(lvl.mean()))
        tau = tau * (lvl / norm)
    return tau


def _build(
    cfg: ScenarioConfig,
    graph: SampledGraph,
    params: Mapping,
    calib,
    noise: dict,
    fit: bool,
    sample_seed,
    scm: SCM | None,
):
    """Construct a sample from exogenous noise with a fixed model.

    fit=True performs calibration: write constants to the mutable calib dictionary and
    return None without constructing a Dataset.
    """
    from .generate import Dataset  # Import lazily because generate imports scm.
    from .truth import refresh_truth

    n = noise["x"].shape[0]
    x, x_latent = apply_covariates(
        graph, cfg.mechanism, params["xx_w"], noise["x"], calib, fit
    )
    raw = raw_cells(
        x, graph, params["seg_features"], tuple(params["seg_shape"]), calib, fit
    )
    segments, unseen = compact_segments(raw, calib, fit)
    mu0 = apply_mu0(x, graph, cfg.mechanism, params, calib, fit)
    tau = structural_tau(cfg, graph, params, x, raw, calib, fit)
    e = apply_propensity(x, graph, cfg.treatment, params, calib, fit)

    latent = {"mu0_x": mu0, "tau_x": tau, "u_t": noise["t"], "e_assign": e}
    if x_latent is not None:
        latent["x_latent"] = x_latent
    m = cfg.graph.n_mediators
    if m:
        latent.update(
            apply_mediators(x, graph, cfg.mediator, cfg.mechanism, params, calib, fit)
        )
        if cfg.mediator.family == "gaussian":
            latent.update(
                med_a=np.array(params["med_a"], copy=True),
                med_w=np.array(params["med_w"], copy=True),
                med_eps=noise["med"],
            )
        else:
            latent["med_u"] = noise["med"]
    terms = mediator_contributions(latent)
    mu0_eff = mu0 if terms is None else mu0 + terms.mu0_row
    tau_eff = tau if terms is None else tau + terms.tau_row
    # Family-specific outcome noise: for gaussian, eps = ν·sd_eps, where sd_eps = sd(tau)/snr is a
    # calibration constant.
    if cfg.outcome.family == "gaussian":
        sd_eps = _const(calib, "sd_eps", fit, lambda: _sd_eps(tau_eff, cfg.outcome.snr))
        latent["sd_eps"] = float(sd_eps)
        latent["eps"] = noise["y"] * (
            sd_eps / np.sqrt(3.0) if cfg.outcome.heavy_tails else sd_eps
        )
    elif (
        cfg.outcome.family == "lognormal"
    ):  # Since 0.11.0, eps = σ_ε·ν on the log scale, with σ_ε from the config.
        sd_eps = float(cfg.outcome.noise_sigma)
        latent["sd_eps"] = sd_eps
        latent["eps"] = noise["y"] * sd_eps
    else:
        latent["u"] = noise["y"]
        if cfg.outcome.family == "ziln":
            latent["ln_z"] = noise["ln"]
    if (
        cfg.outcome.coupling == "independent"
    ):  # Independent outcome noise for Y(1) (0.8.0).
        if cfg.outcome.family in ("gaussian", "lognormal"):
            sd_eps = latent["sd_eps"]
            latent["eps1"] = noise["y1"] * (
                sd_eps / np.sqrt(3.0) if cfg.outcome.heavy_tails else sd_eps
            )
        else:
            latent["u1"] = noise["y1"]
            if cfg.outcome.family == "ziln":
                latent["ln_z1"] = noise["ln1"]
    if "z0" in noise:  # Lognormal point mass at zero (0.11.0).
        latent["u0"] = noise["z0"]
        if "z1" in noise:
            latent["u01"] = noise["z1"]
    if "pre" in noise:  # Pretreatment outcome lags (0.11.0).
        latent.update(pre_noise(noise["y"], noise["pre"], cfg.outcome.pre_rho))
    coll = graph.covariate_indices(ROLE_COLLIDER)
    if coll.size:
        latent["eps_c"] = noise["coll"]
    latent["x_true"] = np.ascontiguousarray(x, dtype=float)
    j = params.get("scale_feature")
    if j is not None:  # Natural outcome scale (0.9.0).
        latent["y_scale"] = outcome_scale(
            x[:, j],
            calib["x_raw_mean"][j],
            calib["x_raw_sd"][j],
            cfg.outcome.scale_sigma,
        )
    if fit:  # Calibration computes constants only, without constructing a Dataset.
        _fit_constants(cfg, graph, x, mu0, tau, e, mu0_eff, tau_eff, latent, calib)
        return None
    if coll.size:
        latent["coll_y_mean"], latent["coll_y_sd"] = calib["y_mean"], calib["y_sd"]

    x_df = pd.DataFrame(x, columns=[f"x{i}" for i in range(graph.d)])
    ds = Dataset(
        x=x_df,
        t=np.zeros(n, np.int8),
        y=np.zeros(n),
        y0=np.zeros(n),
        y1=np.zeros(n),
        tau=tau,
        mu0=mu0,
        mu1=mu0,
        e=e,
        segments=segments,
        graph=thaw_graph(graph),
        config=copy.deepcopy(cfg),
        model_seed=scm.model_seed if scm is not None else None,
        sample_seed=sample_seed,
        pathology_seed=None,
        latent=latent,
        n_requested=n,
        scm=scm,
        segments_unseen=unseen,
    )
    refresh_truth(ds)
    return ds


def _fit_constants(cfg, graph, x, mu0, tau, e, mu0_eff, tau_eff, latent, calib) -> None:
    """Compute calibration constants for the data-generating axes and colliders.

    Use calibration rows in blocks and aggregate with einsum without BLAS (0.7.0),
    so the resulting bits are independent of the OpenBLAS kernel.
    """
    n, d = x.shape
    idx = np.arange(d)

    def blocks():
        for lo in range(0, n, _BLOCK):
            # Copy this block because level_view(inplace) must leave x intact. For d = 1,
            # ascontiguousarray may return its input.
            yield lo, graph.level_view(x[lo : lo + _BLOCK].copy(), idx, inplace=True)

    # Mean and covariance of X as seen by descendants, including level effects.
    s1, s2 = np.zeros(d), np.zeros((d, d))
    for _, xv in blocks():
        s1 += xv.sum(axis=0)
        s2 += np.einsum("ni,nj->ij", xv, xv, optimize=False)
    mean = s1 / n
    calib["x_mean"] = mean
    calib["x_cov"] = s2 / n - np.outer(mean, mean)
    # Level proportions for discrete nodes.
    disc = graph.discrete or {}
    calib["discrete_shares"] = [
        np.bincount(x[:, i].astype(int), minlength=int(disc[i]["levels"])) / n
        for i in graph.discrete_nodes()
    ]
    # Standard deviation of structural tau including mediators for hidden_modifier and
    # effect_drift, plus its covariance with X to orthogonalize rotation.
    calib["tau_c_sd"] = _sd_or_zero(tau_eff)
    calib["tau_x_mean"], calib["tau_x_sd"] = float(tau.mean()), _sd_or_zero(tau)
    cov_tx = np.zeros(d)
    for lo, xv in blocks():
        cov_tx += np.einsum(
            "ni,n->i",
            xv - mean,
            tau[lo : lo + _BLOCK] - calib["tau_x_mean"],
            optimize=False,
        )
    calib["tau_x_cov"] = cov_tx / n
    calib["mu0_x_mean"], calib["mu0_x_sd"] = float(mu0.mean()), _sd_or_one(mu0)
    # Sum of confounders and its association with T ~ Bernoulli(e) for overlap and selection.
    conf = graph.covariate_indices(ROLE_CONFOUNDER)
    base = (
        graph.level_view(x[:, conf].copy(), conf, inplace=True).sum(axis=1)
        if conf.size
        else np.zeros(n)
    )
    t = (latent["u_t"] < e).astype(float)
    calib["conf_sum_mean"], calib["conf_sum_sd"] = float(base.mean()), _sd_or_one(base)
    calib["t_mean"], calib["t_var"] = float(t.mean()), float(t.var())
    calib["conf_t_cov"] = float(((base - base.mean()) * (t - t.mean())).mean())
    calib["e_assign_mean"], calib["e_assign_sd"] = float(e.mean()), _sd_or_one(e)
    ec = np.clip(e, 1e-12, 1 - 1e-12)
    calib["e_logit_var"] = float(np.log(ec / (1 - ec)).var())
    # Use clean calibration y_true to standardize colliders.
    y_true = realized_outcome(
        cfg.outcome.family, mu0_eff, tau_eff, cfg.outcome, dict(latent), t
    )
    calib["y_mean"], calib["y_sd"] = float(y_true.mean()), _sd_or_one(y_true)


def pre_noise(nu: np.ndarray, z: np.ndarray, rho: float) -> dict:
    """Generate outcome noise for earlier periods.

    For K lags (0.11.0), combine standardized current outcome noise ν with the pre
    stream z of shape n × (K + 1). Draw the unit's persistent component u = ρ·ν + √(1 −
    ρ²)·z₀ conditional on the existing ν, preserving Y and all ground truth available
    without lags. Period k uses ν_k = ρ·u + √(1 − ρ²)·z_k, so corr(ν, ν_k) = corr(ν_j,
    ν_k) = ρ² for distinct periods. The posterior ν | ν₁…ν_K is N(m, v), with m =
    ρ²·Σν_k / (1 + (K − 1)ρ²) and v = 1 − K·ρ⁴ / (1 + (K − 1)ρ²). Sum over lags with
    einsum without BLAS to preserve bitwise nested samples.
    """
    k = z.shape[1] - 1
    c = float(np.sqrt(1.0 - rho * rho))
    u = rho * nu + c * z[:, 0]
    pre_nu = rho * u[:, None] + c * z[:, 1:]
    r2 = rho * rho
    den = 1.0 + (k - 1) * r2
    return {
        "pre_nu": pre_nu,
        "pre_nu_m": r2 * np.einsum("nk->n", pre_nu, optimize=False) / den,
        "pre_nu_v": float(1.0 - k * r2 * r2 / den),
    }


# Model construction.


def _calibration_rngs(model_seed: int):
    rng = np.random.default_rng((int(model_seed), CALIB_STREAM))
    return lambda kind: rng


def draw_noise(rngs, n: int, graph: SampledGraph, cfg: ScenarioConfig) -> dict:
    """Generate exogenous noise for n observations.

    Draw x (n × d), y (standard normal or t₃ for gaussian, standard normal for
    lognormal, uniform for binary/ziln/beta), ln for ziln, uniform t, med (n × m), and
    coll (n × k). With independent outcome coupling draw a second outcome noise y1, and
    ln1 for ziln, after the other original noise kinds. Calibration shares one stream
    across all kinds, so those original draws match shared coupling. Since 0.11.0, the
    subsequent draws are uniform z0/z1 for lognormal zero mass and pre (n × (K + 1)) for
    outcome lags. rngs(kind) returns the Generator for that kind.
    """
    noise = {"x": rngs("x").standard_normal((n, graph.d))}
    fam = cfg.outcome.family
    if fam == "lognormal":
        noise["y"] = rngs("y").standard_normal(n)
    elif fam == "gaussian":
        noise["y"] = (
            rngs("y").standard_t(df=3, size=n)
            if cfg.outcome.heavy_tails
            else rngs("y").standard_normal(n)
        )
    else:
        noise["y"] = rngs("y").random(n)
        if fam == "ziln":
            noise["ln"] = rngs("ln").standard_normal(n)
    noise["t"] = rngs("t").random(n)
    m = cfg.graph.n_mediators
    if m:  # Binary mediators use uniform noise from the med stream (0.10.0).
        noise["med"] = (
            rngs("med").random((n, m))
            if cfg.mediator.family == "binary"
            else rngs("med").standard_normal((n, m))
        )
    k = graph.covariate_indices(ROLE_COLLIDER).size
    if k:
        noise["coll"] = rngs("coll").standard_normal((n, k))
    if cfg.outcome.coupling == "independent":
        if fam in ("gaussian", "lognormal"):
            noise["y1"] = (
                rngs("y1").standard_t(df=3, size=n)
                if cfg.outcome.heavy_tails
                else rngs("y1").standard_normal(n)
            )
        else:
            noise["y1"] = rngs("y1").random(n)
            if fam == "ziln":
                noise["ln1"] = rngs("ln1").standard_normal(n)
    oc = cfg.outcome
    if fam == "lognormal" and oc.zero_mass_ln > 0:
        noise["z0"] = rngs("z0").random(n)
        if oc.coupling == "independent":
            noise["z1"] = rngs("z1").random(n)
    if oc.pre_periods > 0:
        noise["pre"] = rngs("pre").standard_normal((n, int(oc.pre_periods) + 1))
    return noise


def _build_scm(config: ScenarioConfig, model_seed: int, n_calib: int) -> SCM:
    from . import __version__

    config.validate()
    if int(n_calib) < 100:
        msg = f"n_calib must be >= 100, got {n_calib!r}"
        raise ValueError(msg)
    config = copy.deepcopy(config)
    rng = np.random.default_rng(int(model_seed))
    rct = config.treatment.mode == "rct"
    graph = sample_graph(
        config.graph, rng, rct=rct
    )  # Draw the graph first to match 0.5.0.
    segment_tau = config.tau.form == "segment"
    feats = segment_features(graph, config.segments, segment_tau=segment_tau)
    shape = grid_shape(graph, config.segments, feats)
    if segment_tau:
        graph.add_edges_to_y(
            feats
        )  # The Y equation uses the grid features through tau.
    params: dict = {
        "seg_features": [int(i) for i in feats],
        "seg_shape": [int(shape[0]), int(shape[1])],
    }
    j = scale_feature(config, graph)
    if j is not None:
        params["scale_feature"] = j
    params["xx_w"] = draw_xx_weights(graph, rng)
    params.update(draw_mu0(graph, config.mechanism, rng))
    if (
        config.graph.direct_effect
    ):  # Without the direct edge tau is never drawn (0.10.0).
        params.update(draw_tau(graph, config.tau, shape[0] * shape[1], rng))
    params.update(draw_treatment(graph, config.treatment, rng))
    params.update(draw_mediators(graph, config.mediator, rng))
    calib: dict = {}
    noise = draw_noise(_calibration_rngs(model_seed), int(n_calib), graph, config)
    _build(
        config, graph, params, calib, noise, True, None, None
    )  # Calibration writes its constants to calib.
    return SCM(
        config=config,
        model_seed=int(model_seed),
        n_calib=int(n_calib),
        graph=graph,
        params=params,
        calib=calib,
        scmgen_version=__version__,
    )


def scale_feature(config: ScenarioConfig, graph: SampledGraph) -> int | None:
    """Select the node that sets the natural outcome scale.

    Return None if scale_sigma = 0 (0.9.0). An explicit index must
    name a continuous non-collider. Otherwise choose the first continuous node with
    declared role confounder, then predictor, then modifier, in that priority order.
    """
    oc = config.outcome
    if oc.scale_sigma <= 0:
        return None
    disc = set(graph.discrete_nodes())
    j = oc.scale_feature
    if j is not None:
        j = int(j)
        if graph.roles[j] == ROLE_COLLIDER:
            msg = f"outcome.scale_feature={j}: a collider (descendant of Y) cannot set the outcome scale"
            raise ValueError(msg)
        if j in disc:
            msg = f"outcome.scale_feature={j}: this node is discrete; the outcome scale requires a continuous node"
            raise ValueError(msg)
        return j
    for role in (
        ROLE_CONFOUNDER,
        ROLE_PREDICTOR,
        ROLE_MODIFIER,
        ROLE_MED_CONFOUNDER,
        ROLE_MED_PREDICTOR,
    ):
        for i in graph.covariate_indices(role):
            if int(i) not in disc:
                return int(i)
    msg = (
        "outcome.scale_sigma > 0: the graph has no continuous confounder, predictor, modifier, "
        "med_confounder or med_predictor to set the outcome scale. Set outcome.scale_feature or add nodes."
    )
    raise ValueError(msg)


_SCM_CACHE: dict = {}


def sample_scm(
    config: ScenarioConfig, model_seed: int, n_calib: int = DEFAULT_N_CALIB
) -> SCM:
    """Draw the scenario SCM from model_seed using n_calib rows for calibration.

    Cache by (configuration, seed, n_calib) and return the immutable cached instance
    directly.
    """
    key = (_canonical_json(config), int(model_seed), int(n_calib))
    scm = _SCM_CACHE.get(key)
    if scm is None:
        scm = _build_scm(config, model_seed, n_calib)
        if len(_SCM_CACHE) >= _CACHE_SIZE:
            del _SCM_CACHE[next(iter(_SCM_CACHE))]
        _SCM_CACHE[key] = scm
    return scm


def clear_cache() -> None:
    """Clear the cache of calibrated structural causal models."""
    _SCM_CACHE.clear()


# Helpers.


def _const(calib, key, fit, compute):
    if fit:
        calib[key] = compute()
    return calib[key]


def _sd_or_one(v) -> float:
    sd = float(np.asarray(v).std())
    return sd if sd > 1e-12 else 1.0


def _sd_or_zero(v) -> float:
    return float(np.asarray(v).std())


def _sd_eps(tau: np.ndarray, snr: float) -> float:
    sd_tau = float(tau.std())
    if sd_tau > 1e-12:
        return sd_tau / snr
    return float(np.abs(tau).mean()) / max(snr, 1e-12) or 1.0


def graph_to_dict(g: SampledGraph) -> dict:
    """Convert the graph fields to JSON-compatible values."""
    d = {f.name: _jsonable(getattr(g, f.name)) for f in dataclasses.fields(g)}
    if g.discrete:
        d["discrete"] = {str(k): _jsonable(v) for k, v in g.discrete.items()}
    return d


def graph_from_dict(d: dict) -> SampledGraph:
    """Reconstruct a sampled graph and its arrays from serialized fields."""
    d = dict(d)
    d["adjacency"] = np.asarray(d["adjacency"], dtype=np.int8)
    d["xx_edges"] = np.asarray(d["xx_edges"], dtype=np.int8)
    if d.get("discrete"):
        d["discrete"] = {int(k): v for k, v in d["discrete"].items()}
    return SampledGraph(**d)


def _jsonable(obj):
    if isinstance(obj, Mapping):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


_PARAM_ARRAYS = (
    "mu0_w",
    "mu0_pair_w",
    "tau_w",
    "tau_w1",
    "tau_w2",
    "seg_levels",
    "gate_w",
    "e_w",
    "med_a",
    "med_w",
    "med_f_w",
    "med_g_w",
    "med_gate_w",
    "med_h_w",
)
_CALIB_FLOAT_ARRAYS = ("x_mean", "x_cov", "tau_x_cov")


def _params_from_json(p: dict) -> dict:
    out = dict(p)
    out["xx_w"] = [None if w is None else np.asarray(w, dtype=float) for w in p["xx_w"]]
    for k in _PARAM_ARRAYS:
        if k in out:
            out[k] = np.asarray(out[k], dtype=float)
    return out


def _calib_from_json(c: dict) -> dict:
    out = dict(c)
    for k in _CALIB_FLOAT_ARRAYS:
        if k in out:
            out[k] = np.asarray(out[k], dtype=float)
    out["seg_ids"] = np.asarray(out["seg_ids"], dtype=int)
    out["seg_edges"] = [
        None if e is None else np.asarray(e, dtype=float) for e in out["seg_edges"]
    ]
    out["discrete_shares"] = [
        np.asarray(s, dtype=float) for s in out.get("discrete_shares", [])
    ]
    return out
