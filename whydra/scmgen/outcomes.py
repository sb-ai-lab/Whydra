"""Generate potential outcomes for the M-4 outcome families using stored exogenous noise.

By default, both potential outcomes are computed in one pass with shared noise, giving a
paired ITE for each row and comparable S-2 paired samples. Each dataset stores its draws
in latent. DGP changes P-17, P-20, and P-21 update latent mu0 or tau and recompute the
outcomes from the same draws. Since 0.6.0, SCM.sample stores both the noise and the SCM
calibration constant sd_eps. A random generator is needed here only outside the SCM when
the necessary draws have not been stored.

The coupling of potential outcomes (0.8.0) is an assumption of the generator. The data
do not identify it. By default outcome.coupling = "shared": Y(0) and Y(1) share ε for
Gaussian outcomes, or a uniform u for binary, ZILN, and beta outcomes, giving
comonotonic draws. With "independent" coupling, Y(1) uses its own noise (eps1 or u1, and
ln_z1 as appropriate) with the same marginals. Coupling affects y1, ite, vte_w, and the
noncompliance fields that integrate over outcome noise. The structural definitions of
tau, tau_cate, and cvte are unchanged. See the lagged-outcome qualifications below.

For positive lognormal outcomes (0.11.0), log Y(t) = log b + σ_μ·mu0 + t·tau +
σ_ε·ε. The effect is on the log scale, so the relative effect exp(tau) − 1 exceeds −1. A
separate uniform draw (latent u0 or u01) controls the zero_mass_ln point mass at zero,
independently of x and T. With pretreatment outcome lags (outcome.pre_periods),
integrate with respect to the posterior ν given the lags: N(pre_nu_m, pre_nu_v) in
latent. This adjusts the Gaussian mean and gives the lognormal factor E[exp(σ_ε·ν) | W].
"""

import numpy as np

from .betafn import betaincinv
from .config import OutcomeConfig


def _sigmoid(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-v))


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _noise(latent: dict, key: str, draw):
    """Reuse the exogenous noise stored under this key, or draw and store it once.

    This lets truth.refresh_truth recompute the outcomes from the latent state
    using the same noise draws.
    """
    if key not in latent:
        latent[key] = draw()
    return latent[key]


def _outcomes_gaussian(mu0, tau, cfg: OutcomeConfig, latent: dict, rng):
    if (
        "sd_eps" not in latent
    ):  # Outside the SCM. Since 0.6.0 the SCM supplies sd_eps from calibration.
        sd_tau = tau.std()
        sd_eps = (
            sd_tau / cfg.snr
            if sd_tau > 1e-12
            else abs(tau).mean() / max(cfg.snr, 1e-12) or 1.0
        )
        latent["sd_eps"] = float(sd_eps)
    sd_eps = latent["sd_eps"]
    if cfg.heavy_tails:

        def draw():
            return rng.standard_t(df=3, size=mu0.size) * sd_eps / np.sqrt(3.0)
    else:

        def draw():
            return rng.normal(scale=sd_eps, size=mu0.size)

    eps = _noise(latent, "eps", draw)
    eps1 = _noise(latent, "eps1", draw) if cfg.coupling == "independent" else eps
    y0, y1 = mu0 + eps, mu0 + tau + eps1
    m = latent.get(
        "pre_nu_m"
    )  # With lags, E[ν₀ | W] shifts the conditional mean of Y(0), and also of Y(1) under shared coupling (0.11.0).
    if m is None:
        mu0_o, mu1_o, tau_o = mu0, mu0 + tau, tau
    else:
        adj = sd_eps * m
        mu0_o = mu0 + adj
        if (
            cfg.coupling == "shared"
        ):  # Under shared noise the ITE equals tau. Keep the structural effect bit for bit.
            mu1_o, tau_o = mu0 + tau + adj, tau
        else:
            mu1_o, tau_o = mu0 + tau, tau - adj
    s = latent.get(
        "y_scale"
    )  # Natural scale s(x) (0.9.0): multiply the entire outcome by s.
    if s is None:
        return y0, y1, tau_o, mu0_o, mu1_o
    return s * y0, s * y1, s * tau_o, s * mu0_o, s * mu1_o


def outcome_scale(x_j, mean: float, sd: float, sigma: float) -> np.ndarray:
    """Compute the natural outcome scale from one observed covariate.

    s(x) = exp(σ·z), where z = (x_j − mean)/sd uses the SCM calibration constants
    (0.9.0). It depends only on the observed X node and consumes no random draws.
    """
    return np.exp(sigma * (np.asarray(x_j, dtype=float) - mean) / sd)


def _binary_probs(mu0, tau, cfg: OutcomeConfig):
    logit0 = _logit(cfg.base_rate) + mu0
    return _sigmoid(logit0), _sigmoid(logit0 + tau)  # tau is on the logit scale.


def _outcomes_binary(mu0, tau, cfg: OutcomeConfig, latent: dict, rng):
    p0, p1 = _binary_probs(mu0, tau, cfg)
    u = _noise(latent, "u", lambda: rng.random(mu0.size))  # Shared noise.
    u1 = (
        _noise(latent, "u1", lambda: rng.random(mu0.size))
        if cfg.coupling == "independent"
        else u
    )
    y0, y1 = (u < p0).astype(np.int8), (u1 < p1).astype(np.int8)
    return y0, y1, p1 - p0, p0, p1


_ZILN_SIGMA = 0.5


def _ziln_probs(mu0, tau, cfg: OutcomeConfig):
    """Return the control and treated probabilities of a nonzero response, and its mean size when positive."""
    logit_pos = _logit(1.0 - cfg.zero_mass) + 0.5 * mu0
    mean_ln = np.exp(0.3 * mu0 + 0.5 * _ZILN_SIGMA**2)
    return (
        _sigmoid(logit_pos),
        _sigmoid(logit_pos + tau),
        mean_ln,
    )  # tau is on the logit scale.


def _outcomes_ziln(mu0, tau, cfg: OutcomeConfig, latent: dict, rng):
    """Generate zero-inflated lognormal outcomes with a treatment effect on the probability of a nonzero response.

    For each row, the positive response is exp(0.3 mu0 + sigma z). The latent
    state stores the normal draw z under ``ln_z`` so the size can be recomputed
    if mu0 changes. The treatment arms share the draw under shared coupling
    and use separate draws under independent coupling.
    """
    p0, p1, mean_ln = _ziln_probs(mu0, tau, cfg)
    u = _noise(latent, "u", lambda: rng.random(mu0.size))
    z = _noise(latent, "ln_z", lambda: rng.standard_normal(mu0.size))
    ln = np.exp(0.3 * mu0 + _ZILN_SIGMA * z)
    if cfg.coupling == "independent":
        u1 = _noise(latent, "u1", lambda: rng.random(mu0.size))
        ln1 = np.exp(
            0.3 * mu0
            + _ZILN_SIGMA
            * _noise(latent, "ln_z1", lambda: rng.standard_normal(mu0.size))
        )
    else:
        u1, ln1 = u, ln
    y0, y1 = np.where(u < p0, ln, 0.0), np.where(u1 < p1, ln1, 0.0)
    mu0_out, mu1_out = p0 * mean_ln, p1 * mean_ln
    return y0, y1, mu1_out - mu0_out, mu0_out, mu1_out


def ziln_second_moment(mu0):
    """Return the expected squared size of a positive lognormal response given mu0."""
    return np.exp(0.6 * mu0 + 2.0 * _ZILN_SIGMA**2)


# Lognormal family: positive outcomes with multiplicative structure (0.11.0).


def lognormal_log_level(mu0, cfg: OutcomeConfig):
    """Return the noiseless log level, log b + σ_μ·mu0."""
    return np.log(cfg.base_level) + cfg.level_sigma * mu0


def nu_posterior(latent: dict):
    """Compute conditional moments of the outcome noise given pretreatment lags.

    Return (m, v), the posterior mean by row and variance of the standardized Y(0) noise
    ν given the lags. Without lags, return (0, 1).
    """
    m = latent.get("pre_nu_m")
    return (0.0, 1.0) if m is None else (m, float(latent["pre_nu_v"]))


def lognormal_noise_factors(cfg: OutcomeConfig, latent: dict):
    """Return (g₀, g₁, h₀), where g_t = E[exp(σ_ε·ν_t) | W] and h₀ = E[exp(2σ_ε·ν₀) | W].

    Without lags, g = exp(σ_ε²/2) and h₀ = exp(2σ_ε²). With lags, ν₀ | W ~ N(m, v).
    Shared coupling sets ν₁ = ν₀. Under independent coupling ν₁ is independent of the
    lags.
    """
    se = float(cfg.noise_sigma)
    m, v = nu_posterior(latent)
    prior = float(np.exp(0.5 * se * se))
    if latent.get("pre_nu_m") is None:
        g0, h0 = prior, float(np.exp(2.0 * se * se))
    else:
        g0, h0 = (
            np.exp(se * m + 0.5 * se * se * v),
            np.exp(2.0 * se * m + 2.0 * se * se * v),
        )
    g1 = g0 if cfg.coupling == "shared" else prior
    return g0, g1, h0


def _outcomes_lognormal(mu0, tau, cfg: OutcomeConfig, latent: dict, rng):
    """Generate potential outcomes for the lognormal family.

    The positive outcome is Y(t) = exp(log b + σ_μ·mu0 + t·tau + ε), where ε = σ_ε·ν
    (latent eps and sd_eps = σ_ε). Set Y to zero when u₀ < π. The arms share u₀ under
    shared coupling. Under independent coupling Y(1) uses u01. The arm mean given W and
    the latents is mu_t = (1 − π)·exp(level + t·tau)·g_t.
    """
    se = float(cfg.noise_sigma)
    if "sd_eps" not in latent:
        latent["sd_eps"] = se

    def draw():
        return rng.standard_normal(mu0.size) * se

    eps = _noise(latent, "eps", draw)
    eps1 = _noise(latent, "eps1", draw) if cfg.coupling == "independent" else eps
    level = lognormal_log_level(mu0, cfg)
    y0, y1 = np.exp(level + eps), np.exp(level + tau + eps1)
    pi = float(cfg.zero_mass_ln)
    if pi > 0:
        u0 = _noise(latent, "u0", lambda: rng.random(mu0.size))
        u01 = (
            _noise(latent, "u01", lambda: rng.random(mu0.size))
            if cfg.coupling == "independent"
            else u0
        )
        y0, y1 = np.where(u0 < pi, 0.0, y0), np.where(u01 < pi, 0.0, y1)
    g0, g1, _ = lognormal_noise_factors(cfg, latent)
    base = (1.0 - pi) * np.exp(level)
    mu0_out, mu1_out = base * g0, base * np.exp(tau) * g1
    return y0, y1, mu1_out - mu0_out, mu0_out, mu1_out


def lognormal_cate(cfg: OutcomeConfig, latent: dict, mu0_w, tau_c, var_h):
    """Compute the conditional mean and variance of the structural lognormal effect.

    Obtain tau_cate and cvte in closed form by integrating over U ~ N(0, var_h) and over
    ν (its posterior when lags are present). On the log scale the effect is tau_c + U.
    With A = (1 − π)·exp(log b + σ_μ·mu0_w), tau_cate = A·(g₁·exp(tau_c + var_h/2) − g₀)
    and cvte = Var_U = (A·g₁)²·exp(2·tau_c)·(exp(2·var_h) − exp(var_h)).
    """
    g0, g1, _ = lognormal_noise_factors(cfg, latent)
    a = (1.0 - float(cfg.zero_mass_ln)) * np.exp(lognormal_log_level(mu0_w, cfg))
    var_h = np.asarray(var_h, dtype=float)
    tau_cate = a * (g1 * np.exp(tau_c + 0.5 * var_h) - g0)
    cvte = (a * g1) ** 2 * np.exp(2.0 * tau_c) * (np.exp(2.0 * var_h) - np.exp(var_h))
    return tau_cate, np.maximum(cvte, 0.0) * np.ones_like(tau_cate)


def lognormal_ite_second_moment(cfg: OutcomeConfig, latent: dict, mu0_w, tau_c, var_h):
    """Compute E[(Y(1) − Y(0))² | W] for the lognormal family, used by vte_w.

    Set L = log b + σ_μ·mu0_w. With shared noise, the second moment is (1 −
    π)·exp(2L)·h₀·E_U[(exp(tau+U) − 1)²]. With independent coupling it is E[Y₁²] +
    E[Y₀²] − 2·E[Y₁]·E[Y₀]: both the noise and the zero-mass draws are independent
    between arms.
    """
    se, pi = float(cfg.noise_sigma), float(cfg.zero_mass_ln)
    g0, g1, h0 = lognormal_noise_factors(cfg, latent)
    e2l = np.exp(2.0 * lognormal_log_level(mu0_w, cfg))
    var_h = np.asarray(var_h, dtype=float)
    e1, e2 = np.exp(tau_c + 0.5 * var_h), np.exp(2.0 * tau_c + 2.0 * var_h)
    if cfg.coupling == "shared":
        return (1.0 - pi) * e2l * h0 * (e2 - 2.0 * e1 + 1.0)
    return (1.0 - pi) * e2l * (np.exp(2.0 * se * se) * e2 + h0) - 2.0 * (
        1.0 - pi
    ) ** 2 * e2l * g1 * g0 * e1


# Beta family: proportions with point masses at zero and one (0.8.0).

BETA_EDGE = 1e-9  # Keep the continuous part in [BETA_EDGE, 1 − BETA_EDGE] so only the point masses produce exact zeros or ones.


def beta_clip(y):
    """Clip the continuous component to [BETA_EDGE, 1 − BETA_EDGE]."""
    return np.clip(y, BETA_EDGE, 1.0 - BETA_EDGE)


def beta_arm(
    mu0, tau_t, cfg: OutcomeConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an arm's masses at zero and one and its continuous-component mean (π₀, π₁, m).

    tau_t is zero for the control arm and tau for the treated arm. beta_tau_on selects
    the channel. "mean" shifts the logit of m. "mass" shifts the mass logits, decreasing
    π₀ and increasing π₁ for positive tau. "both" shifts both channels. A zero point
    mass stays zero for every tau.
    """
    on = cfg.beta_tau_on
    c_mean = 1.0 if on in ("mean", "both") else 0.0
    c_mass = float(cfg.beta_mass_scale) if on in ("mass", "both") else 0.0
    shift = c_mass * tau_t
    m = _sigmoid(_logit(cfg.base_mean) + mu0 + c_mean * tau_t)
    shape = np.broadcast(mu0, tau_t).shape
    pi0 = (
        _sigmoid(_logit(cfg.zero_mass_beta) - shift) if cfg.zero_mass_beta > 0 else 0.0
    )
    pi1 = _sigmoid(_logit(cfg.one_mass_beta) + shift) if cfg.one_mass_beta > 0 else 0.0
    return (
        np.broadcast_to(pi0, shape),
        np.broadcast_to(pi1, shape),
        np.broadcast_to(m, shape),
    )


def beta_mean(pi0, pi1, m):
    """E[y] = π₁ + (1 − π₀ − π₁)·m."""
    return pi1 + (1.0 - pi0 - pi1) * m


def beta_second_moment(pi0, pi1, m, phi: float):
    """E[y²] = π₁ + (1 − π₀ − π₁)·(m² + m(1 − m)/(1 + φ))."""
    return pi1 + (1.0 - pi0 - pi1) * (m * m + m * (1.0 - m) / (1.0 + phi))


BETA_AB_MIN = 1e-3  # Lower bound for beta parameters in betafn. config.BETA_PHI_MAX sets the upper bound.
BETA_KAPPA_MIN = (
    0.01  # Lower bound on κ for mean-preserving noise: Var ≤ y(1 − y)/(1 + κ_min).
)


def beta_mean_preserving_noise(y, kappa, v, kappa_max: float = 1e4):
    """Add beta-distributed measurement noise to a proportion.

    For a uniform noise draw v, the beta measurement model uses the quantile
    y′ = Q(v; κy, κ(1 − y)). Before clipping, this beta distribution has
    E[y′ | y] = y and Var[y′ | y] = y(1 − y)/(κ + 1).
    Since version 0.9.0, κ is clipped to [BETA_KAPPA_MIN, kappa_max] and each
    beta shape parameter is bounded below by BETA_AB_MIN. When
    κ·min(y, 1 − y) < BETA_AB_MIN, the floor changes the mean. Near a boundary
    the shift can be comparable to the distance from that boundary.
    """
    kappa = np.clip(np.asarray(kappa, dtype=float), BETA_KAPPA_MIN, kappa_max)
    a = np.maximum(kappa * y, BETA_AB_MIN)
    b = np.maximum(kappa * (1.0 - y), BETA_AB_MIN)
    return beta_clip(betaincinv(a, b, v))


def beta_noise_kappa(y_cont, added_sd, kappa_max: float = 1e4):
    """Choose the beta noise concentration for a target amount of added variance.

    As for Gaussian noise, the target added variance is added_sd² = (strength·sd)².
    Solve E_c[y(1 − y)]/(κ + 1) = added_sd² for κ using moments of the
    continuous-component rows y_cont in this batch.
    """
    m2 = float((y_cont * (1.0 - y_cont)).mean())
    with np.errstate(divide="ignore"):
        kappa = m2 / np.square(np.asarray(added_sd, dtype=float)) - 1.0
    return np.clip(
        np.nan_to_num(kappa, nan=kappa_max, posinf=kappa_max), BETA_KAPPA_MIN, kappa_max
    )


def beta_quantile(u, pi0, pi1, m, phi: float):
    """Compute the quantile of a beta distribution with point masses at zero and one.

    The mixture places mass π₀ at zero and π₁ at one around Beta(mφ, (1 − m)φ). Return
    zero if u < π₀ and one if u ≥ 1 − π₁. Otherwise, evaluate the beta quantile at (u −
    π₀)/(1 − π₀ − π₁) and clip to [BETA_EDGE, 1 − BETA_EDGE].
    """
    u, pi0, pi1, m = np.broadcast_arrays(u, pi0, pi1, m)
    zero, one = u < pi0, u >= 1.0 - pi1
    out = np.where(one, 1.0, 0.0)
    cont = ~zero & ~one
    if cont.any():
        mc = m[cont]
        v = np.clip((u[cont] - pi0[cont]) / (1.0 - pi0[cont] - pi1[cont]), 0.0, 1.0)
        out[cont] = beta_clip(betaincinv(mc * phi, (1.0 - mc) * phi, v))
    return out


def _outcomes_beta(mu0, tau, cfg: OutcomeConfig, latent: dict, rng):
    """Generate proportions under both arms.

    By default a shared uniform draw couples their full distributions comonotonically.
    """
    arm0, arm1 = beta_arm(mu0, 0.0, cfg), beta_arm(mu0, tau, cfg)
    u = _noise(latent, "u", lambda: rng.random(mu0.size))
    u1 = (
        _noise(latent, "u1", lambda: rng.random(mu0.size))
        if cfg.coupling == "independent"
        else u
    )
    y0, y1 = (
        beta_quantile(u, *arm0, cfg.beta_phi),
        beta_quantile(u1, *arm1, cfg.beta_phi),
    )
    mu0_out, mu1_out = beta_mean(*arm0), beta_mean(*arm1)
    return y0, y1, mu1_out - mu0_out, mu0_out, mu1_out


_OUTCOME_FAMILIES = {
    "gaussian": _outcomes_gaussian,
    "binary": _outcomes_binary,
    "ziln": _outcomes_ziln,
    "beta": _outcomes_beta,
    "lognormal": _outcomes_lognormal,
}


def family_outcomes(
    family: str, mu0_latent, tau_latent, cfg: OutcomeConfig, latent: dict, rng
):
    """Generate the selected family's potential outcomes from latent mu0, tau, and stored noise.

    Return (y0, y1, tau_scale, mu0_out, mu1_out). If generation needs new draws, store
    them in latent. truth.refresh_truth reuses those same draws and can pass rng=None.
    This keeps S-2 paired samples comparable.
    """
    fn = _OUTCOME_FAMILIES.get(family)
    if fn is None:
        msg = f"Unknown outcome family: {family}"
        raise ValueError(msg)
    return fn(mu0_latent, tau_latent, cfg, latent, rng)


def realized_outcome(
    family: str, mu0_latent, tau_latent, cfg: OutcomeConfig, latent: dict, t
) -> np.ndarray:
    """Return the outcome for the received treatment using the stored noise: where(t == 1, y1, y0).

    For beta outcomes, evaluate only the arm each row receives. One quantile per row
    gives the same value as computing both arms and then selecting one.
    """
    if family != "beta":
        y0, y1, *_ = family_outcomes(family, mu0_latent, tau_latent, cfg, latent, None)
        return np.where(t == 1, y1, y0).astype(float)
    treated = np.asarray(t) == 1
    u = (
        np.where(treated, latent["u1"], latent["u"])
        if cfg.coupling == "independent"
        else latent["u"]
    )
    return beta_quantile(
        u, *beta_arm(mu0_latent, np.where(treated, tau_latent, 0.0), cfg), cfg.beta_phi
    )


def tau_scale_mean(family: str, mu0_latent, tau_latent, cfg: OutcomeConfig):
    """Compute the noise-free row effect on the outcome scale analytically, without sampling."""
    if family == "gaussian":
        return tau_latent
    if family == "beta":
        return beta_mean(*beta_arm(mu0_latent, tau_latent, cfg)) - beta_mean(
            *beta_arm(mu0_latent, 0.0, cfg)
        )
    if family == "lognormal":  # Without lags: (1 − π)·exp(L + σ_ε²/2)·(exp(tau) − 1).
        return (
            (1.0 - cfg.zero_mass_ln)
            * np.exp(lognormal_log_level(mu0_latent, cfg) + 0.5 * cfg.noise_sigma**2)
            * (np.exp(tau_latent) - 1.0)
        )
    p0, p1, scale = family_probs(family, mu0_latent, tau_latent, cfg)
    return (p1 - p0) * scale


def arm_moments(
    family: str, mu0_latent, tau_latent, cfg: OutcomeConfig, sd_eps: float = 0.0
):
    """Compute the first and second moments of each potential outcome.

    Return (m₀, m₁, s₀, s₁) at fixed latent mu0 and tau, integrating only over outcome
    noise. vte_w uses the moments under independent coupling. Its Gaussian path uses a
    closed form. The Gaussian branch here lets tests check the identification formula.
    """
    if family == "gaussian":
        m0, m1 = mu0_latent, mu0_latent + tau_latent
        return m0, m1, m0 * m0 + sd_eps**2, m1 * m1 + sd_eps**2
    if family == "beta":
        a0, a1 = beta_arm(mu0_latent, 0.0, cfg), beta_arm(mu0_latent, tau_latent, cfg)
        return (
            beta_mean(*a0),
            beta_mean(*a1),
            beta_second_moment(*a0, cfg.beta_phi),
            beta_second_moment(*a1, cfg.beta_phi),
        )
    if family == "lognormal":  # Without lags.
        se2, q = cfg.noise_sigma**2, 1.0 - cfg.zero_mass_ln
        level = lognormal_log_level(mu0_latent, cfg)
        m0, s0 = q * np.exp(level + 0.5 * se2), q * np.exp(2.0 * level + 2.0 * se2)
        return m0, m0 * np.exp(tau_latent), s0, s0 * np.exp(2.0 * tau_latent)
    p0, p1, scale = family_probs(family, mu0_latent, tau_latent, cfg)
    scale2 = ziln_second_moment(mu0_latent) if family == "ziln" else 1.0
    return p0 * scale, p1 * scale, p0 * scale2, p1 * scale2


def family_probs(family: str, mu0_latent, tau_latent, cfg: OutcomeConfig):
    """Return the positive-outcome probabilities and scale factor for binary or ZILN outcomes.

    In (p0, p1, scale), p0 and p1 are the probabilities under T=0 and T=1. The scale is
    1 for binary outcomes and E[ln] for ZILN. With a shared uniform draw u, the signed
    interval indicator ±1[p_lo ≤ u < p_hi] times scale gives the ITE for binary outcomes
    and its mean over the positive component for ZILN.
    """
    if family == "binary":
        p0, p1 = _binary_probs(mu0_latent, tau_latent, cfg)
        return p0, p1, np.ones_like(p0)
    if family == "ziln":
        return _ziln_probs(mu0_latent, tau_latent, cfg)
    msg = f"family_probs: family {family!r} has no probability representation"
    raise ValueError(msg)
