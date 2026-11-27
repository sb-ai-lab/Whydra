"""Mechanisms for DAG covariates X, mu0(x), tau(x), and treatment assignment T (M-1 to M-3).

Since 0.6.0, each mechanism has two steps. The draw_* functions draw model parameters
(weights, interaction pairs, and segment levels) from default_rng(model_seed). They do
not draw exogenous noise. The apply_* functions apply the parameters to X using
constants from calib. These constants standardize the discrete-node latent values, mu0,
and the treatment propensity logit. They also center nonlinear tau, set the sparsity
threshold, and scale tau to the requested amplitude.

With fit=True, the SCM computes and stores the constants using the calibration
population. Targets such as standard deviation(tau) = amplitude before the offset and
mean(mu0) = 0 are exact there when the variation can be scaled. They are approximate in
other samples.

The model draws parameters in this order: xx_w by node, mu0_*, tau_*, gate_w, e_w
(observational mode only), then med_* (see scm.sample_scm). Since 0.10.0,
direct_effect=False skips the tau_* draws. MediatorConfig supports Gaussian mediators
with X parents and binary mediators through draw_mediators and apply_mediators.

Since 0.7.0, rowdot computes each row's dot product with einsum without BLAS. The
resulting sample bits are independent of the row count n and the OpenBLAS kernel chosen
for the CPU.
"""

from collections.abc import Mapping, MutableMapping
from typing import NamedTuple

import numpy as np

from .config import MechanismConfig, MediatorConfig, TauConfig, TreatmentConfig
from .graph import (
    ROLE_COLLIDER,
    ROLE_CONFOUNDER,
    ROLE_INSTRUMENT,
    ROLE_MED_CONFOUNDER,
    ROLE_MED_PREDICTOR,
    ROLE_MODIFIER,
    ROLE_PREDICTOR,
    SampledGraph,
    level_codes,
)
from .outcomes import _sigmoid


def rowdot(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute the dot product of each row with the weight vector.

    For X of shape (n, k) and w of length k, summing over k in a fixed order without
    BLAS makes the result independent of n. For n1 <= n2, sample(n1) is an exact prefix
    of sample(n2), independent of the OpenBLAS kernel (0.7.0, section 6.2).
    """
    return np.einsum("ij,j->i", x, w, optimize=False)


def _parent_contribution(
    x_parents: np.ndarray, weights: np.ndarray, kind: str
) -> np.ndarray:
    """Compute the linear or smooth nonlinear contribution of a node's parents (M-1)."""
    lin = rowdot(x_parents, weights)
    if kind == "linear":
        return lin
    return np.sin(lin) + 0.5 * np.tanh(rowdot(x_parents, np.roll(weights, 1))) * lin


def const(calib: Mapping, key: str, fit: bool, compute):
    """Store compute() as a calibration constant during fitting, or read the stored value."""
    if fit:
        assert isinstance(calib, MutableMapping)
        calib[key] = compute()
    return calib[key]


def _sd_or_one(v: np.ndarray) -> float:
    sd = float(v.std())
    return sd if sd > 1e-12 else 1.0


# --- Covariates X along the DAG (M-1) ---


def draw_xx_weights(graph: SampledGraph, rng: np.random.Generator) -> list:
    """Draw parent weights in X-node order, using None for roots and colliders."""
    out: list[np.ndarray | None] = []
    for i in range(graph.d):
        parents = np.flatnonzero(graph.xx_edges[:, i])
        if graph.roles[i] == ROLE_COLLIDER or parents.size == 0:
            out.append(None)
        else:
            out.append(rng.normal(size=parents.size) / np.sqrt(parents.size))
    return out


def apply_covariates(
    graph: SampledGraph,
    mech: MechanismConfig,
    xx_w: list,
    eps: np.ndarray,
    calib: Mapping,
    fit: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute covariates in topological order along the X-to-X DAG.

    Start from the exogenous noise eps of shape (n, d). Collider values initially
    contain noise; truth.refresh_truth later updates them using T and Y.

    For a discrete node, add its parents' contribution to eps to get its latent value v.
    Standardize v with x_raw_mean[i] and x_raw_sd[i], then apply inverse-normal
    thresholds with graph.level_codes. Descendants read the effects of the resulting
    levels through graph.child_view. Return (x, x_latent), with one latent column per
    discrete node in graph.discrete_nodes() order, or None as the second value if there
    are none.
    """
    n, d = eps.shape
    x = np.empty(
        (n, d), order="F"
    )  # Contiguous columns make selecting parents with x[:, parents] much faster.
    disc = graph.discrete or {}
    if fit:
        assert isinstance(calib, MutableMapping)
        calib["x_raw_mean"], calib["x_raw_sd"] = [0.0] * d, [1.0] * d
    means, sds = calib["x_raw_mean"], calib["x_raw_sd"]
    x_lat = {}
    for i in range(d):
        parents = np.flatnonzero(graph.xx_edges[:, i])
        if xx_w[i] is None:
            v = eps[:, i]
        else:
            v = (
                _parent_contribution(graph.child_view(x, parents), xx_w[i], mech.kind)
                + eps[:, i]
            )
        if fit:
            means[i], sds[i] = float(v.mean()), _sd_or_one(v)
        if i in disc:
            x_lat[i] = v
            v = level_codes((v - means[i]) / sds[i], disc[i])
        x[:, i] = v
    x_latent = np.column_stack([x_lat[i] for i in sorted(x_lat)]) if x_lat else None
    return x, x_latent


# --- mu0 (M-1) ---


def _mu0_indices(graph: SampledGraph) -> np.ndarray:
    """List mu0 nodes: confounders, predictors, then mediator confounders.

    The last group was added in 0.10.0. With zero mediator confounders, the list is
    unchanged.
    """
    return np.concatenate(
        [
            graph.covariate_indices(ROLE_CONFOUNDER),
            graph.covariate_indices(ROLE_PREDICTOR),
            graph.covariate_indices(ROLE_MED_CONFOUNDER),
        ]
    )


def draw_mu0(
    graph: SampledGraph, mech: MechanismConfig, rng: np.random.Generator
) -> dict:
    """Draw mu0 weights for confounders, predictors, and mediator confounders.

    Also draw the interacting pairs and their weights.
    """
    idx = _mu0_indices(graph)
    if idx.size == 0:
        return {"mu0_w": np.zeros(0), "mu0_pairs": [], "mu0_pair_w": np.zeros(0)}
    w = rng.normal(size=idx.size) / np.sqrt(idx.size)
    pairs, pw = [], []
    if mech.interaction > 0 and idx.size >= 2:
        n_pairs = max(1, int(mech.interaction * idx.size))
        for _ in range(n_pairs):
            a, b = rng.choice(idx.size, size=2, replace=False)
            pairs.append([int(a), int(b)])
            pw.append(rng.normal() / np.sqrt(n_pairs))
    return {"mu0_w": w, "mu0_pairs": pairs, "mu0_pair_w": np.asarray(pw, dtype=float)}


def apply_mu0(
    x: np.ndarray,
    graph: SampledGraph,
    mech: MechanismConfig,
    params: Mapping,
    calib: Mapping,
    fit: bool = False,
) -> np.ndarray:
    """Standardize the baseline outcome mu0(x) using mu0_mean and mu0_sd."""
    idx = _mu0_indices(graph)
    if idx.size == 0:
        mu0 = np.zeros(x.shape[0])
    else:
        z = graph.child_view(x, idx)
        mu0 = _parent_contribution(z, params["mu0_w"], mech.kind)
        for (a, b), w in zip(params["mu0_pairs"], params["mu0_pair_w"], strict=False):
            mu0 = mu0 + w * z[:, a] * z[:, b]
    mean = const(calib, "mu0_mean", fit, lambda: float(mu0.mean()))
    sd = const(calib, "mu0_sd", fit, lambda: _sd_or_one(mu0))
    return (mu0 - mean) / sd


# --- tau (M-2) ---


def draw_tau(
    graph: SampledGraph, cfg: TauConfig, n_cells: int, rng: np.random.Generator
) -> dict:
    """Draw the parameters of the heterogeneous treatment effect.

    For the segment form, draw one tau level per raw grid cell (n_cells). For the linear
    or nonlinear form, draw its weights. Then draw any sparsity gate weights, preserving
    the draw order from 0.5.0.
    """
    idx = graph.covariate_indices(ROLE_MODIFIER)
    out: dict = {}
    if cfg.form == "segment":
        out["seg_levels"] = rng.normal(size=int(n_cells))
    elif idx.size == 0:
        return out  # A constant effect needs no parameters.
    elif cfg.form == "linear":
        out["tau_w"] = rng.normal(size=idx.size) / np.sqrt(idx.size)
    elif cfg.form == "nonlinear":
        out["tau_w1"] = rng.normal(size=idx.size) / np.sqrt(idx.size)
        out["tau_w2"] = rng.normal(size=idx.size) / np.sqrt(idx.size)
    else:
        msg = f"Unknown tau form: {cfg.form}"
        raise ValueError(msg)
    if cfg.amplitude == 0:
        return out
    if cfg.sparsity > 0 and idx.size > 0:
        out["gate_w"] = rng.normal(size=idx.size) / np.sqrt(idx.size)
    return out


def apply_tau(
    x: np.ndarray,
    graph: SampledGraph,
    cfg: TauConfig,
    params: Mapping,
    raw_cells: np.ndarray | None,
    calib: Mapping,
    fit: bool = False,
) -> np.ndarray:
    """Compute the conditional average treatment effect tau(x) from the modifiers (M-2).

    The segment form uses the raw grid cells. Compute tau for the chosen form, center
    the nonlinear form with tau_center, and set gated rows to zero using gate_thr. Scale
    by tau_sd to reach the target standard deviation (amplitude) when possible. Add the
    offset last, keeping gated rows at zero.
    """
    n = x.shape[0]
    idx = graph.covariate_indices(ROLE_MODIFIER)
    if cfg.form == "segment":
        tau = params["seg_levels"][raw_cells]
    elif idx.size == 0:
        return np.full(
            n, cfg.amplitude + cfg.offset
        )  # Without modifiers, the effect is constant.
    else:
        xm = graph.child_view(x, idx)
        if cfg.form == "linear":
            tau = rowdot(xm, params["tau_w"])
        else:
            tau = (
                np.sin(2.0 * rowdot(xm, params["tau_w1"]))
                + 0.5 * rowdot(xm, params["tau_w2"]) ** 2
            )
            tau = tau - const(calib, "tau_center", fit, lambda: float(tau.mean()))
    if (
        cfg.amplitude == 0
    ):  # With zero amplitude the varying term vanishes, so tau equals the offset.
        return np.full(n, float(cfg.offset))
    gate = None
    if "gate_w" in params:
        gate_score = rowdot(graph.child_view(x, idx), params["gate_w"])
        thr = const(
            calib, "gate_thr", fit, lambda: float(np.quantile(gate_score, cfg.sparsity))
        )
        gate = gate_score <= thr
        tau = np.where(gate, 0.0, tau)
    sd = const(calib, "tau_sd", fit, lambda: float(tau.std()))
    if sd > 1e-12:
        tau = tau * (cfg.amplitude / sd)
    return _apply_offset(tau, cfg.offset, gate)


def _apply_offset(
    tau: np.ndarray, offset: float, gate: np.ndarray | None
) -> np.ndarray:
    """Add the offset after scaling tau and keep sparsity-gated rows at zero."""
    if not offset:
        return tau
    tau = tau + offset
    if gate is not None:
        tau = np.where(gate, 0.0, tau)
    return tau


# --- Treatment assignment (M-3) ---


def _e_indices(graph: SampledGraph) -> np.ndarray:
    return np.concatenate(
        [
            graph.covariate_indices(ROLE_CONFOUNDER),
            graph.covariate_indices(ROLE_INSTRUMENT),
        ]
    )


def draw_treatment(
    graph: SampledGraph, cfg: TreatmentConfig, rng: np.random.Generator
) -> dict:
    """Draw propensity logit weights for observational confounders and instruments.

    A randomized trial uses no assignment parameters and makes no random draws here.
    """
    if cfg.mode == "rct":
        return {}
    if cfg.mode != "obs":
        msg = f"Unknown treatment assignment mode: {cfg.mode}"
        raise ValueError(msg)
    idx = _e_indices(graph)
    if idx.size == 0:
        msg = "Observational assignment requires confounders or instruments"
        raise ValueError(msg)
    return {"e_w": rng.normal(size=idx.size) / np.sqrt(idx.size)}


def apply_propensity(
    x: np.ndarray,
    graph: SampledGraph,
    cfg: TreatmentConfig,
    params: Mapping,
    calib: Mapping,
    fit: bool = False,
) -> np.ndarray:
    """Compute the true probability of treatment assignment.

    The propensity e(x) is constant in a randomized trial. In observational mode it is
    sigmoid(logit(p) + confounding * z), where z is the weighted logit standardized
    using e_logit_mean and e_logit_sd.
    """
    n = x.shape[0]
    if cfg.mode == "rct":
        return np.full(n, cfg.propensity)
    logit = rowdot(graph.child_view(x, _e_indices(graph)), params["e_w"])
    mean = const(calib, "e_logit_mean", fit, lambda: float(logit.mean()))
    sd = const(calib, "e_logit_sd", fit, lambda: _sd_or_one(logit))
    base = np.log(cfg.propensity / (1 - cfg.propensity))
    return 1.0 / (1.0 + np.exp(-(base + cfg.confounding * (logit - mean) / sd)))


# --- Mediators (0.10.0, section 3.6) ---


def _med_parent_indices(graph: SampledGraph) -> np.ndarray:
    """List mediator X parents by index, med_confounder first and med_predictor next."""
    return np.concatenate(
        [
            graph.covariate_indices(ROLE_MED_CONFOUNDER),
            graph.covariate_indices(ROLE_MED_PREDICTOR),
        ]
    )


def _theta_indices(graph: SampledGraph) -> np.ndarray:
    """List modifiers, then mediator confounders, for variation in the M-to-Y effect."""
    return np.concatenate(
        [
            graph.covariate_indices(ROLE_MODIFIER),
            graph.covariate_indices(ROLE_MED_CONFOUNDER),
        ]
    )


def draw_mediators(
    graph: SampledGraph, cfg: MediatorConfig, rng: np.random.Generator
) -> dict:
    """Draw the mediator coefficients.

    For Gaussian mediator k, M_k = f_k(x) + a_k T + eps_k and its outcome contribution
    is w_k M_k. Draw a_k and w_k as before 0.10.0, then draw weights for f_k if the
    mediator has X parents.

    For binary mediators, use their X parents for the baseline logit f_k, the variable
    treatment effect g_k (T to M), and the effect_sparsity gate. For variation in theta,
    h_k uses the modifiers and mediator confounders. Draw covariate weights from N(0, 1)
    divided by the square root of their input count. A zero input count skips the
    corresponding draws.
    """
    m = graph.n_mediators
    if not m:
        return {}
    p, q = _med_parent_indices(graph).size, _theta_indices(graph).size

    def weights(k_cols: int) -> np.ndarray:
        return rng.normal(size=(m, k_cols)) / np.sqrt(k_cols)

    out: dict = {}
    if cfg.family == "gaussian":
        coef = np.empty((m, 2))
        for k in range(m):
            coef[k] = rng.normal(), rng.normal()
        out.update(med_a=coef[:, 0].copy(), med_w=coef[:, 1].copy())
    if p:
        out["med_f_w"] = weights(p)
    if cfg.family == "binary":
        if p and cfg.effect_amplitude > 0:
            out["med_g_w"] = weights(p)
        if p and cfg.effect_sparsity > 0:
            out["med_gate_w"] = weights(p)
        if q and cfg.theta_amplitude > 0:
            out["med_h_w"] = weights(q)
    return out


def _list_const(calib: Mapping, key: str, k: int, m: int, fit: bool, compute) -> float:
    """Get or fit mediator k's calibration constant in the m-element list calib[key]."""
    if fit:
        assert isinstance(calib, MutableMapping)
        calib.setdefault(key, [0.0] * m)
        calib[key][k] = float(compute())
    return float(calib[key][k])


def _solve_intercept(f: np.ndarray, rate: float) -> float:
    """Find c such that mean(sigmoid(c + f)) = rate.

    The mean increases with c. Bisect the interval [-40, 40] until its width is below
    1e-12.
    """
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(np.mean(_sigmoid(mid + f))) < rate:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def apply_mediators(
    x: np.ndarray,
    graph: SampledGraph,
    cfg: MediatorConfig,
    mech: MechanismConfig,
    params: Mapping,
    calib: Mapping,
    fit: bool = False,
) -> dict:
    """Compute the X-dependent terms in the mediator equations.

    The deterministic terms given X have shape (n, m). Gaussian mediators supply med_f
    only when they have X parents. For binary mediators, med_logit0 = c_k + f_k(x),
    med_effect = effect + g_k(x) on ungated rows and zero on gated rows, and med_theta =
    theta_k(x).

    Calibration constants such as med_f_mean and med_f_sd center f, g, and h and scale
    each to its own amplitude. The intercept med_c sets mean(p0) to base_rate on the
    calibration population. med_gate_thr sets the sparsity threshold and med_theta_norm
    sets mean(theta_k) to theta on that same population.
    """
    m = graph.n_mediators
    n = x.shape[0]
    if not m:
        return {}
    # Compute the parents' values as seen by the child only when an equation uses them. child_view
    # is deterministic.
    xp = (
        graph.child_view(x, _med_parent_indices(graph))
        if any(k in params for k in ("med_f_w", "med_g_w", "med_gate_w"))
        else None
    )
    xq = graph.child_view(x, _theta_indices(graph)) if "med_h_w" in params else None

    def term(key: str, xv, amp: float) -> np.ndarray | None:
        w = params.get(f"{key}_w")
        if w is None:
            return None
        out = np.empty((n, m))
        for k in range(m):
            raw = _parent_contribution(xv, np.asarray(w[k]), mech.kind)
            mean = _list_const(calib, f"{key}_mean", k, m, fit, raw.mean)
            sd = _list_const(
                calib, f"{key}_sd", k, m, fit, lambda raw=raw: _sd_or_one(raw)
            )
            out[:, k] = (raw - mean) / sd * amp
        return out

    f = term("med_f", xp, cfg.base_amplitude)
    if cfg.family == "gaussian":
        return {} if f is None else {"med_f": f}
    logit0 = np.zeros((n, m)) if f is None else f
    for k in range(m):
        c = _list_const(
            calib,
            "med_c",
            k,
            m,
            fit,
            lambda k=k: _solve_intercept(logit0[:, k], cfg.base_rate),
        )
        logit0[:, k] = logit0[:, k] + c
    g = term("med_g", xp, cfg.effect_amplitude)
    effect = np.full((n, m), float(cfg.effect)) if g is None else g + float(cfg.effect)
    gate_w = params.get("med_gate_w")
    if gate_w is not None:
        assert xp is not None
        for k in range(m):
            score = rowdot(xp, np.asarray(gate_w[k]))
            thr = _list_const(
                calib,
                "med_gate_thr",
                k,
                m,
                fit,
                lambda score=score: np.quantile(score, cfg.effect_sparsity),
            )
            effect[:, k] = np.where(score <= thr, 0.0, effect[:, k])
    h = term("med_h", xq, cfg.theta_amplitude)
    theta = np.full((n, m), float(cfg.theta))
    if h is not None:
        eh = np.exp(h)
        for k in range(m):
            norm = _list_const(calib, "med_theta_norm", k, m, fit, eh[:, k].mean)
            theta[:, k] = float(cfg.theta) * eh[:, k] / norm
    return {"med_logit0": logit0, "med_effect": effect, "med_theta": theta}


def mediator_family(lat: dict) -> str | None:
    """Identify the mediator family from its latent state.

    This is the single place that interprets the latent keys as family markers: med_u
    means binary, med_eps means Gaussian, and neither means there are no mediators
    (None).
    """
    return "binary" if "med_u" in lat else "gaussian" if "med_eps" in lat else None


def mediator_probs(lat: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return binary mediator probabilities (p0, p1) as two (n, m) arrays.

    Each entry is P(M_k = 1 | x, t), with t=0 for p0 and t=1 for p1.
    """
    logit0 = lat["med_logit0"]
    return _sigmoid(logit0), _sigmoid(logit0 + lat["med_effect"])


def mediator_potentials(lat: dict, probs=None) -> tuple[np.ndarray, np.ndarray]:
    """Return binary mediator outcomes (m0, m1) using the shared uniform noise med_u.

    Both arrays have shape (n, m) and dtype int8. Pass precomputed (p0, p1) as probs to
    reuse them.
    """
    p0, p1 = mediator_probs(lat) if probs is None else probs
    u = lat["med_u"]
    return (u < p0).astype(np.int8), (u < p1).astype(np.int8)


def _rowsum(v: np.ndarray) -> np.ndarray:
    return np.einsum("nk->n", v, optimize=False)


class MediatorTerms(NamedTuple):
    """Mediator contributions to the outcome equation, per row and conditional on W.

    The fields give mu0 and tau for the realized latent variables in each row, and their
    expected values given W. None means there is no term to add, preserving bit-for-bit
    results when mediators are absent.
    """

    mu0_row: np.ndarray | None
    tau_row: np.ndarray | float
    mu0_w: np.ndarray | None
    tau_w: np.ndarray | float


def mediator_contributions(lat: dict) -> MediatorTerms | None:
    """Return the mediators' contributions to the control outcome and treatment effect.

    Gaussian mediators contribute Σ w_k (f_k + ε_k) and Σ w_k a_k per row.
    Given W, their contributions are Σ w_k f_k and Σ w_k a_k.
    For binary mediators the corresponding terms are Σ θ_k m0_k and
    Σ θ_k (m1_k − m0_k) per row, and Σ θ_k p0_k and Σ θ_k (p1_k − p0_k)
    given W. Return None when there are no mediators.
    """
    fam = mediator_family(lat)
    if fam == "binary":
        p0, p1 = mediator_probs(lat)
        m0, m1 = mediator_potentials(lat, (p0, p1))
        th = lat["med_theta"]
        return MediatorTerms(
            _rowsum(th * m0),
            _rowsum(th * (m1 - m0)),
            _rowsum(th * p0),
            _rowsum(th * (p1 - p0)),
        )
    if fam == "gaussian":
        w, a = lat["med_w"], lat["med_a"]
        ate = float(np.einsum("k,k->", w, a))
        f = lat.get("med_f")
        if f is None:
            return MediatorTerms(rowdot(lat["med_eps"], w), ate, None, ate)
        base = rowdot(f, w)
        return MediatorTerms(base + rowdot(lat["med_eps"], w), ate, base, ate)
    return None
