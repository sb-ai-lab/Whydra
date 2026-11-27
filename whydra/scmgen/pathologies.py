"""Model distortions in observed data and changes to the data-generating process.

The module implements data pathology axes P-1 through P-9, the extended set P-10
through P-21, and the 0.6.0 axes P-19, P-24, and P-26, with support for targeting
segments.

Since 0.5.0, ScenarioConfig.validate requires axes to follow three phases in
nondecreasing order:

1. Data generation: hidden_modifier, targeted_selection, overlap, noncompliance,
cluster, covariate_shift, effect_drift, and policy_drift change Dataset.latent and call
truth.refresh_truth. That one function recomputes tau, y0/y1, tau_cate, cvte, tau_itt,
e, and colliders from the same exogenous noise, preserving S-2 pairing. These axes read
true X from latent["x_true"] and standardize model quantities using Dataset.scm.calib,
independent of the sample (0.6.0). Assignment axes (overlap,
targeted_selection, policy_drift, cluster.icc_e) add to latent["e_logit_extra"] on the
logit scale. covariate_shift recomputes only the SCM propensity, retaining these
contributions.

2. Selection: selection chooses rows. sel_q1 and sel_q0 are products of stage-specific
retention probabilities under T = 1 and 0. e becomes P(T = 1 | W, S = 1). The first
stage saves the original e in e_source, together with ate_source and n_source.
truth.selection_truth updates the selected-population fields. duplicate_rows creates
copies identified by dup_index and may occur only once.

3. Measurement: noise_y, noise_t, noise_x, missing (including sentinel),
heteroscedastic, hidden_confounder, bad_controls, and measurement_drift change observed
X/T/Y or the column set while leaving ground truth intact. Standardization is
batch-specific, a property of the measurement process. Address nodes by column name
(x{i}). Dataset.observed_nodes() supplies candidates.

Each axis has its own RNG stream, default_rng((pathology_seed, 0x9A70, i, code(kind))),
keyed by its scenario index. Axis-wide quantities such as projection weights and cluster
effects are drawn before row-specific quantities. Thus generate_split(test_dgp="replay")
uses the same data-generating process on test rows: the same clusters and directions
with fresh rows. Each log entry records catalog numbers pathology and pathology_extra,
phase, index, and the actual impact: active, affected_columns, affected_rows_share
measured between input and output, and affected_fields for data-generating axes. Zero
strength normally leaves the dataset intact and active=False. Exceptions are
bad_controls with expose=[mediator] and cluster with params.emit.

Before measurement, y equals y_true. Afterward, y is the outcome available to the
learner.
"""

import copy
import dataclasses

import numpy as np
import pandas as pd

from .config import (
    PHASE_DGP,
    PHASE_MEASUREMENT,
    PHASE_SELECTION,
    QUANTIZE_Y_STEP_SD,
    PathologyConfig,
    drift_share,
    pathology_number,
    pathology_phase,
)
from .graph import (
    ROLE_COLLIDER,
    ROLE_CONFOUNDER,
    ROLE_MODIFIER,
    ROLE_PREDICTOR,
    level_codes,
)
from .graph import standardize as _standardize
from .mechanisms import rowdot
from .outcomes import beta_mean_preserving_noise, beta_noise_kappa
from .truth import (
    e_assign_effective,
    make_nc_state,
    new_axis_rng,
    outcome_noise_normal,
    refresh_truth,
    selection_truth,
)

_ROW_FIELDS = (
    "t",
    "y",
    "y0",
    "y1",
    "tau",
    "mu0",
    "mu1",
    "e",
    "segments",
    "t_true",
    "tau_cate",
    "cvte",
    "z",
    "tau_itt",
    "compliance",
    "mediators",
    "e_assign",
    "e_source",
    "row_id",
    "period",
    "cluster",
    "dup_index",
    "tau_itt_cate",
    "tau_cate_selected",
    "tau_itt_selected",
    "e_assign_selected",
    "tau_cate_x",
)
# Latent fields used to report each data-generating axis's impact in affected_fields.
_DGP_FIELDS = (
    "x_true",
    "mu0_x",
    "tau_x",
    "e_assign",
    "e_logit_extra",
    "mu0_cluster",
    "tau_hidden",
    "var_hidden",
    "u_t",
    "sel_q1",
    "sel_q0",
    "med_f",
    "med_logit0",
    "med_effect",
    "med_theta",
)


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def _row_mask(p: PathologyConfig, segments: np.ndarray, rng: np.random.Generator):
    """Return the pathology's target row mask and the list of affected segments."""
    if p.target == "global":
        return np.ones(segments.size, dtype=bool), None
    if p.target != "segments":
        msg = f"unknown pathology target: {p.target}"
        raise ValueError(msg)
    uniq = np.unique(segments)
    if p.segments is not None:
        chosen = [s for s in p.segments if s in uniq]
    else:
        k = int(np.ceil(p.segment_share * uniq.size))
        chosen = sorted(rng.choice(uniq, size=k, replace=False).tolist()) if k else []
    return np.isin(segments, chosen), chosen


def _require_global(p: PathologyConfig):
    if p.target != "global":
        msg = f"pathology {p.kind} is inherently global: target must be 'global'"
        raise ValueError(msg)


def _calib(ds) -> dict:
    """Return the SCM calibration constants required by data-generating axes (0.6.0)."""
    if ds.scm is None:
        msg = "this data-generating axis requires Dataset.scm (the dataset was built outside an SCM or the SCM was not loaded)"
        raise ValueError(msg)
    return ds.scm.calib


# Node addressing: data-generating axes use true X for all graph nodes. Measurement axes use
# observed nodes.


def _col(ds, i: int) -> str:
    return ds.graph.node_names[int(i)]


def _nodes(ds, roles=None, observed: bool = False) -> list[int]:
    """Choose covariate indices by role and observability.

    Select X nodes with one of the requested roles. None includes all roles except
    colliders. If observed is true, keep only nodes whose columns exist in ds.x.
    """
    pool = ds.observed_nodes() if observed else range(ds.graph.d)
    if roles:
        return [i for i in pool if ds.graph.roles[i] in roles]
    return [i for i in pool if ds.graph.roles[i] != ROLE_COLLIDER]


def _candidates(ds, kind: str, roles=None) -> list[int]:
    """Return candidate observed nodes for a measurement axis.

    If the set is empty, raise ValueError naming the axis.
    """
    cand = _nodes(ds, roles, observed=True)
    if not cand:
        raise ValueError(
            f"{kind}: no observed covariates"
            + (f" with roles {list(roles)}" if roles else "")
            + " (are all nodes hidden?)"
        )
    return cand


def _x_true(ds, idx) -> np.ndarray:
    """Return the true X values for idx as seen by mechanisms, using level effects for discrete nodes.

    Used by data-generating axes.
    """
    idx = list(idx)
    return ds.graph.level_view(ds.latent["x_true"][:, idx].copy(), idx, inplace=True)


def _obs_col(ds, i: int) -> np.ndarray:
    """Copy node i's observed column as floats.

    Raise KeyError naming the node if it is hidden.
    """
    col = _col(ds, i)
    if col not in ds.x.columns:
        msg = f"node {col} is absent from observed X (hidden by hidden_confounder)"
        raise KeyError(msg)
    return ds.x[col].to_numpy(dtype=float, copy=True)


def _x_obs(ds, idx) -> np.ndarray:
    """Return the observed X columns for idx by name.

    Raise KeyError naming an absent node if hidden.
    """
    idx = list(idx)
    names = [_col(ds, i) for i in idx]
    missing = [c for c in names if c not in ds.x.columns]
    if missing:
        msg = (
            f"node {missing[0]} is absent from observed X (hidden by hidden_confounder)"
        )
        raise KeyError(msg)
    return ds.graph.level_view(ds.x[names].to_numpy(dtype=float), idx)


def _projection_parts(ds, rng, roles=None):
    """Draw a random projection of true covariates.

    Standardize using calibration constants: (x·w − x_mean·w) / sqrt(wᵀ Σ w). Return
    (proj, idx, w, sd_proj). Draw the direction w first because it is an axis-level
    parameter. The arithmetic uses no BLAS.
    """
    idx = _nodes(ds, roles)
    if not idx:
        return np.zeros(ds.n), idx, np.zeros(0), 1.0
    calib = _calib(ds)
    x = np.nan_to_num(_x_true(ds, idx), nan=0.0)
    w = rng.normal(size=len(idx)) / np.sqrt(len(idx))
    mean = float(np.einsum("k,k->", np.asarray(calib["x_mean"])[idx], w))
    cov = np.asarray(calib["x_cov"])[np.ix_(idx, idx)]
    sd = float(
        np.sqrt(max(float(np.einsum("i,ij,j->", w, cov, w, optimize=False)), 0.0))
    )
    sd = sd if sd > 1e-12 else 1.0
    return (rowdot(x, w) - mean) / sd, idx, w, sd


def _projection(ds, rng, roles=None) -> np.ndarray:
    return _projection_parts(ds, rng, roles)[0]


def _conf_score(ds) -> np.ndarray:
    """Return the sum of true confounder values, standardized with calibration constants."""
    calib = _calib(ds)
    idx = ds.graph.covariate_indices(ROLE_CONFOUNDER)
    if idx.size == 0:
        return np.zeros(ds.n)
    return (_x_true(ds, idx).sum(axis=1) - calib["conf_sum_mean"]) / calib[
        "conf_sum_sd"
    ]


def _add_e_logit(ds, rows: np.ndarray, delta: np.ndarray) -> None:
    """Add this axis's contribution to the log odds of treatment assignment.

    Add delta for the selected rows to latent e_logit_extra on the e_assign logit scale.
    refresh_truth reassigns T using the same u_t. A zero contribution leaves the
    result bitwise unchanged.
    """
    lat = ds.latent
    extra = lat.get("e_logit_extra")
    if extra is None:
        extra = np.zeros(ds.n)
    lat["e_logit_extra"] = extra + np.where(rows, delta, 0.0)
    refresh_truth(ds)


# Measurement phase.


def _noise_y(ds, p, mask, rng):
    """Add measurement noise appropriate to the outcome family.

    P-1 preserves E[y′ | y] = y except for binary outcomes (0.9.0).
    Gaussian adds strength·sd(y)·ξ. For ZILN and lognormal outcomes, multiply positive
    values by exp(aξ − a²/2), where a = strength. Before 0.9.0 the missing −a²/2 term inflated the mean by exp(a²/2).
    Binary flips labels. For beta outcomes the continuous rows use y′ ~ Beta(κy, κ(1 −
    y)), taking the quantile at the axis's own uniform noise. κ is chosen for added
    variance (strength·sd_c)². Before 0.9.0, logit-scale noise pulled values toward 0.5.
    Point masses at 0 and 1 are preserved.

    P-14 (params.by_arm={0: k0, 1: k1}) scales noise strength separately by true
    treatment arm. Unequal k0 and k1 can introduce differential measurement bias even in
    an RCT.
    """
    by_arm = p.params.get("by_arm")
    if by_arm:
        by_arm = {int(k): float(v) for k, v in by_arm.items()}
        strength = np.where(
            ds.t_true == 1,
            p.strength * by_arm.get(1, 1.0),
            p.strength * by_arm.get(0, 1.0),
        )
    else:
        strength = np.full(ds.n, p.strength)
    fam = ds.config.outcome.family
    if fam == "binary":
        flip = mask & (rng.random(ds.n) < 0.3 * strength)
        ds.y = np.where(flip, 1 - ds.y, ds.y)
    elif fam in (
        "ziln",
        "lognormal",
    ):  # Lognormal (0.11.0) uses the ZILN approach, keeping positive outcomes positive.
        pos = mask & (ds.y > 0)
        xi = rng.normal(0.0, 1.0, ds.n)
        ds.y = np.where(
            pos, ds.y * np.exp(xi * strength - 0.5 * strength * strength), ds.y
        )
    elif fam == "beta":
        cont = mask & (ds.y > 0) & (ds.y < 1) & (strength > 0)
        v = rng.random(ds.n)
        if cont.any():
            y = ds.y.astype(float)
            yc = y[cont]
            kappa = beta_noise_kappa(yc, strength[cont] * (yc.std() + 1e-12))
            y[cont] = beta_mean_preserving_noise(yc, kappa, v[cont])
            ds.y = y
    else:
        sd = np.nanstd(ds.y) + 1e-12
        ds.y = ds.y + mask * (rng.normal(0.0, 1.0, ds.n) * (strength * sd))
    return {"by_arm": by_arm} if by_arm else {}


def _outliers(ds, p, mask, rng):
    """Add gross recording errors to a fraction of outcomes without changing truth (P-18, 0.9.0).

    For affected rows set y += δ. share is params.share or 0.01·strength. Zero strength
    has no effect. |δ| = magnitude·sd(y)·exp(spread·ζ), with median magnitude·sd(y),
    where sd is the standard deviation ignoring NaNs before this axis. params.sides
    controls the sign (both gives ±1 with equal probability). params.by_arm scales the
    probability of an error by true T arm (also P-14). Gaussian, ziln, and lognormal are
    supported. For ziln and lognormal, validation allows only upward shifts and only
    positive rows are eligible. Binary and beta are rejected. The RNG makes three
    length-n draws regardless of the data.
    """
    fam = ds.config.outcome.family
    if fam in ("binary", "beta"):
        msg = (
            f"outliers is unsupported for {fam}: additive recording errors can put "
            "values outside the outcome's allowed range"
        )
        raise ValueError(msg)
    sides = p.params.get("sides", "both")
    if fam in ("ziln", "lognormal") and sides != "up":
        msg = f"outliers for {fam} only supports sides='up' for positive outcomes"
        raise ValueError(msg)
    share = p.params.get("share")
    share = 0.01 * p.strength if share is None else float(share)
    if p.strength <= 0:
        share = 0.0
    magnitude, spread = (
        float(p.params.get("magnitude", 20.0)),
        float(p.params.get("spread", 1.0)),
    )
    by_arm = p.params.get("by_arm")
    prob = np.full(ds.n, share)
    if by_arm:
        by_arm = {int(k): float(v) for k, v in by_arm.items()}
        prob = np.where(
            ds.t_true == 1, share * by_arm.get(1, 1.0), share * by_arm.get(0, 1.0)
        )
    prob = np.clip(prob, 0.0, 1.0)
    y = ds.y.astype(float)
    sd = float(np.nanstd(y)) + 1e-12
    elig = mask & ~np.isnan(y)
    if fam in ("ziln", "lognormal"):
        elig &= y > 0
    u, zeta, su = rng.random(ds.n), rng.normal(0.0, 1.0, ds.n), rng.random(ds.n)
    hit = elig & (u < prob)
    delta = magnitude * sd * np.exp(spread * zeta)
    if sides == "down":
        delta = -delta
    elif sides == "both":
        delta = np.where(su < 0.5, -delta, delta)
    y[hit] = y[hit] + delta[hit]
    ds.y = y
    out = {
        "n_rows": int(hit.sum()),
        "share_achieved": float(hit.mean()) if ds.n else 0.0,
        "median_abs_delta": float(np.median(np.abs(delta[hit]))) if hit.any() else 0.0,
        "sd_unit": sd,
        "sides": sides,
        "magnitude": magnitude,
        "spread": spread,
    }
    if by_arm:
        out["by_arm"] = by_arm
    return out


def _quantize_y(ds, p, mask, rng):
    """Round the selected rows' outcomes to the configured step.

    This coarsening axis (0.11.0, P-16) sets y ← step·round(y / step) on its mask.
    Use params.step in outcome units or params.step_sd·sd(y), with sd computed over all
    rows before the axis, ignoring NaNs. When both parameters are absent, step_sd
    defaults to config.QUANTIZE_Y_STEP_SD. Ground truth is unchanged, and no random
    draws are used. Zero strength has no effect. As with bins in noise_x.mode=quantize,
    strength does not set the step. Gaussian, ziln, and lognormal are supported; small
    positive values may round to zero. Binary and beta are rejected. Pretreatment lags
    y_pre are untouched.
    """
    fam = ds.config.outcome.family
    if fam in ("binary", "beta"):
        msg = f"quantize_y does not apply to family {fam}"
        raise ValueError(msg)
    y = ds.y.astype(float)
    sd = float(np.nanstd(y))
    if p.params.get("step") is not None:
        step = float(p.params["step"])
    else:
        step = float(p.params.get("step_sd") or QUANTIZE_Y_STEP_SD) * sd
    valid = mask & ~np.isnan(y)
    before = int(np.unique(y[valid]).size)
    if p.strength > 0 and step > 0:
        y[valid] = step * np.round(y[valid] / step)
        ds.y = y
    return {
        "step": step,
        "n_distinct_before": before,
        "n_distinct_after": int(np.unique(y[valid]).size),
    }


def _noise_t(ds, p, mask, rng):
    """Introduce treatment logging error (P-2).

    The outcome stays consistent with the true treatment.
    """
    flip = mask & (rng.random(ds.n) < 0.5 * p.strength)
    ds.t = np.where(flip, 1 - ds.t, ds.t).astype(
        ds.t.dtype
    )  # Only t is recorded with error. Preserve t_true.
    return {}


def _noise_x(ds, p, mask, rng):
    """Introduce measurement error in a subset of covariates (P-3).

    With params.roles including confounder (P-13), only the specified roles receive
    noise, producing residual confounding. With params.mode="quantize" (P-16), replace
    values with their bin means in k quantile bins (params.bins, default 4). The bin
    count is independent of strength. Zero strength skips the axis.

    Since 0.4.0, gaussian mode changes a discrete node's level to another random level
    with probability strength·0.5 (logged in flip_features and flipped_share). Quantize
    mode leaves discrete nodes alone and logs skipped_discrete.
    """
    feats = p.params.get("features")
    if feats is None:
        roles = p.params.get("roles")
        cand = _candidates(ds, "noise_x", roles)
        k = len(cand) if roles else max(1, len(cand) // 2)
        feats = sorted(rng.choice(cand, size=k, replace=False).tolist())
    feats = [int(i) for i in feats]
    mode = p.params.get("mode", "gaussian")
    bins = int(p.params.get("bins", 4))
    disc = ds.graph.discrete or {}
    flip_feats, flipped, skipped = [], 0, []
    for i in feats:
        v = _obs_col(ds, i)  # Discrete nodes contain level codes.
        spec = disc.get(i)
        if spec is not None and mode == "quantize":
            skipped.append(i)  # Already discrete.
            continue
        if spec is not None:  # Flip discrete levels.
            k = int(spec["levels"])
            flip = mask & ~np.isnan(v) & (rng.random(v.size) < 0.5 * p.strength)
            shift = rng.integers(
                1, k, size=v.size
            )  # Choose a different level: (x + shift) mod k.
            v[flip] = np.mod(v[flip] + shift[flip], k)
            flip_feats.append(i)
            flipped += int(flip.sum())
        elif mode == "quantize":
            if p.strength == 0:
                continue
            edges = np.nanquantile(v, np.linspace(0, 1, bins + 1)[1:-1])
            b = np.searchsorted(edges, v)
            means = np.array(
                [
                    np.nanmean(v[b == j]) if np.any(b == j) else np.nan
                    for j in range(bins)
                ]
            )
            q = means[b]
            v[mask] = np.where(np.isnan(v[mask]), np.nan, q[mask])
        elif mode == "gaussian":
            v[mask] += rng.normal(0.0, p.strength * (np.nanstd(v) + 1e-12), mask.sum())
        else:
            msg = f"noise_x: unknown mode {mode!r}"
            raise ValueError(msg)
        ds.x[_col(ds, i)] = v
    out = {"features": feats, "mode": mode}
    if mode == "quantize":
        out["bins"] = bins
    if flip_feats:
        out["flip_features"] = flip_feats
        out["flipped_share"] = float(flipped / max(mask.sum() * len(flip_feats), 1))
    if skipped:
        out["skipped_discrete"] = skipped
    return out


def _missing(ds, p, mask, rng):
    """Introduce missing X values (MCAR/MAR, P-4), missing t or y, or MNAR values (P-15).

    params.columns is a subset of ["x", "t", "y"], default ["x"]. params.mode accepts
    mcar, mar (driven by the first covariate in features, logged as driver), or mnar
    (the missingness probability depends on the missing value itself).

    Since 0.6.0, params.sentinel (P-26) encodes values made missing by this specific
    axis within its mask. Use a single number for all features, a dictionary such as
    {"x3": -1}, or "auto": round(min − 3·sd) by column, or the out-of-range code k for a
    discrete node. Preexisting NaNs remain. To encode all newly missing values, set
    sentinel on the first missing axis (0.7.0). Dataset.x_missing records the true
    mask for sentinels, exported as miss_<col> in ground_truth. The log records
    sentinel_values.
    """
    mode = p.params.get("mode", "mcar")
    columns = list(p.params.get("columns", ["x"]))
    feats = p.params.get("features")
    feats = [
        int(i) for i in (feats if feats is not None else _candidates(ds, "missing"))
    ]
    driver_node = feats[0]
    driver = _standardize(np.nan_to_num(_x_obs(ds, [driver_node])[:, 0]))
    n = ds.n

    def prob(own: np.ndarray):
        if mode == "mcar":
            return np.full(n, p.strength)
        if mode == "mar":
            return np.clip(2 * p.strength * _sigmoid(driver), 0, 1)
        if mode == "mnar":
            return np.clip(
                2 * p.strength * _sigmoid(_standardize(np.nan_to_num(own))), 0, 1
            )
        msg = f"missing: unknown mode {mode!r}"
        raise ValueError(msg)

    x_feats = (feats[1:] if mode == "mar" else feats) if "x" in columns else []
    holes: dict[str, np.ndarray] = {}
    for i in x_feats:
        v = _obs_col(ds, i)
        hole = mask & (rng.random(n) < prob(v))
        v[hole] = np.nan
        ds.x[_col(ds, i)] = v
        holes[_col(ds, i)] = hole
    out = {}
    if mode == "mar":
        out["driver"] = driver_node
    sentinel = p.params.get("sentinel")
    if sentinel is not None and x_feats:
        values = _apply_sentinel(ds, holes, sentinel)
        if values:
            out["sentinel_values"] = values
    if "t" in columns:
        t = ds.t.astype(float)
        hole = mask & (rng.random(n) < prob(ds.t_true.astype(float)))
        t[hole] = np.nan
        ds.t = t
    if "y" in columns:
        y = ds.y.astype(float)
        hole = mask & (rng.random(n) < prob(ds.y_true.astype(float)))
        y[hole] = np.nan
        ds.y = y
    return {"mode": mode, "features": list(feats), "columns": columns, **out}


def _apply_sentinel(ds, holes: dict, sentinel) -> dict:
    """Replace only this axis's missing values in holes[col] with a sentinel and save their mask in ds.x_missing."""
    disc = ds.graph.discrete or {}
    values: dict = {}
    masks = {}
    for col, hole in holes.items():
        if not hole.any():  # The axis introduced no missing values in this column.
            continue
        i = int(col[1:])
        v = ds.x[col].to_numpy(dtype=float, copy=True)
        if isinstance(sentinel, dict):
            if col not in sentinel:
                continue
            val = float(sentinel[col])
        elif sentinel == "auto":
            if i in disc:
                val = float(disc[i]["levels"])
            else:
                ok = v[~np.isnan(v)]
                val = float(np.round(ok.min() - 3.0 * ok.std())) if ok.size else -1.0
        else:
            val = float(sentinel)
        v[hole] = val
        ds.x[col] = v
        values[col] = val
        masks[col] = hole
    if masks:
        new = pd.DataFrame(masks, index=ds.x.index)
        if ds.x_missing is None:
            ds.x_missing = new
        else:
            xm = ds.x_missing.copy()
            for col, m in masks.items():
                xm[col] = (xm[col].to_numpy() | m) if col in xm.columns else m
            ds.x_missing = xm
    return values


def _heteroscedastic(ds, p, mask, rng):
    """Increase outcome noise variance with a covariate for continuous outcome families (P-5).

    Compute standard deviations and standardization only for valid rows (inside mask and
    with nonmissing y and feature). Modify only those rows so existing NaNs do not
    spread. Every supported family preserves the conditional mean (0.9.0). For ZILN
    and lognormal outcomes, multiply the positive component by
    exp(aξ − a²/2), where a =
    2·strength·σ_ln·scale, increasing the lognormal tail variance with y >= 0. For beta
    outcomes and continuous rows (0 < y < 1), draw y′ ~ Beta(κy, κ(1 − y)) at the axis's
    uniform quantile, with κ from the added variance (2·strength·sd_c·scale)². The
    masses at 0 and 1 are preserved.
    """
    if ds.config.outcome.family == "binary":
        msg = "heteroscedastic does not apply to binary outcomes"
        raise ValueError(msg)
    i = int(p.params.get("feature", _candidates(ds, "heteroscedastic")[0]))
    xv = _x_obs(ds, [i])[:, 0]
    y = ds.y.astype(float)
    valid = mask & ~np.isnan(y) & ~np.isnan(xv)
    if not valid.any():
        return {"feature": i}
    scale = _sigmoid(_standardize(xv[valid]))
    if ds.config.outcome.family in (
        "ziln",
        "lognormal",
    ):  # lognormal (0.11.0): σ_ln = noise_sigma
        from .outcomes import _ZILN_SIGMA

        sigma_ln = (
            _ZILN_SIGMA
            if ds.config.outcome.family == "ziln"
            else float(ds.config.outcome.noise_sigma)
        )
        xi = rng.normal(0.0, 1.0, int(valid.sum()))
        pos = y[valid] > 0
        y_v = y[valid]
        a = 2 * p.strength * sigma_ln * scale[pos]
        y_v[pos] = y_v[pos] * np.exp(a * xi[pos] - 0.5 * a * a)
        y[valid] = y_v
    elif ds.config.outcome.family == "beta":
        v = rng.random(int(valid.sum()))
        y_v = y[valid]
        cont = (y_v > 0) & (y_v < 1)
        if cont.any() and p.strength > 0:
            yc = y_v[cont]
            kappa = beta_noise_kappa(
                yc, 2 * p.strength * (yc.std() + 1e-12) * scale[cont]
            )
            y_v[cont] = beta_mean_preserving_noise(yc, kappa, v[cont])
            y[valid] = y_v
    else:
        xi = rng.normal(0.0, 1.0, int(valid.sum()))
        sd = y[valid].std() + 1e-12
        y[valid] = y[valid] + xi * (2 * p.strength * sd * scale)
    ds.y = y
    return {"feature": i}


def _hidden_confounder(ds, p, mask, rng):
    """Hide some confounders from the observed covariates.

    P-9 is a global structural change to observed X, with confounding as its
    consequence. Hide k = ceil(strength · n_conf) nodes, so zero strength hides none.
    """
    _require_global(p)
    idx = [
        i
        for i in ds.graph.covariate_indices(ROLE_CONFOUNDER)
        if _col(ds, i) in ds.x.columns
    ]
    # Keep the outcome scale node observable (0.9.0).
    keep = _scale_feature(ds)
    if keep is not None and keep in idx:
        idx.remove(keep)
    if not idx:
        raise ValueError(
            "hidden_confounder requires observed confounders in the graph"
            + (
                " (excluding the outcome.scale_feature node)"
                if keep is not None
                else ""
            )
        )
    k = int(np.ceil(p.strength * len(idx)))
    hidden = (
        sorted(rng.choice(idx, size=min(k, len(idx)), replace=False).tolist())
        if k
        else []
    )
    cols = [_col(ds, i) for i in hidden]
    if cols:
        ds.x = ds.x.drop(columns=cols)
    out: dict = {
        "hidden_columns": cols,
        "dag_effect": [
            f"hides {c} ({ds.graph.roles_effective[i]})"
            for c, i in zip(cols, hidden, strict=False)
        ],
    }
    if keep is not None:
        out["kept_scale_feature"] = _col(ds, keep)
    return out


def _scale_feature(ds) -> int | None:
    """Return the SCM's natural outcome scale node index, or None for scale_sigma = 0 or an absent SCM."""
    if ds.config.outcome.scale_sigma <= 0 or ds.scm is None:
        return None
    return int(ds.scm.params["scale_feature"])


def _bad_controls(ds, p, mask, rng):
    """Expose mediators or descendants of Y as X, creating bad controls (P-10).

    params.expose is a subset of ["mediator", "y_descendant"] and defaults to both.
    params.n_y_descendants sets the number of Y-descendant columns (default 1). Strength
    controls the descendant's correlation with Y, introducing outcome leakage.
    graph.n_colliders already supplies colliders in X. This axis is global because it
    changes the column set. Requesting mediators exposes them at any strength, including
    zero, as an on/off switch.
    """
    _require_global(p)
    expose = list(p.params.get("expose", ["mediator", "y_descendant"]))
    added, dag = {}, []
    if "mediator" in expose and ds.mediators is not None:
        for k in range(ds.mediators.shape[1]):
            ds.x[f"m{k}"] = ds.mediators[:, k]
            added[f"m{k}"] = "mediator"
            dag.append(f"exposes m{k} (t -> m{k} -> y)")
    if "y_descendant" in expose:
        rho = float(np.clip(p.strength, 0.0, 0.999))
        y_std = _standardize(ds.y_true.astype(float))
        for k in range(int(p.params.get("n_y_descendants", 1))):
            ds.x[f"d{k}"] = rho * y_std + np.sqrt(1 - rho**2) * rng.standard_normal(
                ds.n
            )
            added[f"d{k}"] = "y_descendant"
            dag.append(f"adds column d{k} <- y")
    if not added:
        msg = "bad_controls: nothing to expose (the graph contains no mediators and y_descendant was not requested)"
        raise ValueError(msg)
    return {"added_columns": added, "dag_effect": dag}


def _measurement_drift(ds, p, mask, rng):
    """Shift observed X in period two by strength·sd (P-8, measurement).

    Descendants, μ₀, τ, and e stay unchanged. For discrete nodes, shift the standardized
    latent value by strength using calibration constants, then assign levels with the
    original thresholds. The first drift axis creates period from params.share. The
    observed table includes the period column.
    """
    if p.strength == 0:
        return {
            "features": []
        }  # Without a shift there is no change and no period column.
    period = _ensure_period(ds, p)
    rows = mask & (period == 1)
    feats = p.params.get("features")
    if feats is None:
        cand = _candidates(ds, "measurement_drift")
        feats = sorted(
            rng.choice(cand, size=max(1, len(cand) // 3), replace=False).tolist()
        )
    feats = [int(i) for i in feats]
    disc = ds.graph.discrete or {}
    disc_nodes = ds.graph.discrete_nodes()
    for i in feats:
        v = _obs_col(ds, i)
        if i in disc:
            calib = _calib(ds)
            z = (
                ds.latent["x_latent"][:, disc_nodes.index(i)] - calib["x_raw_mean"][i]
            ) / calib["x_raw_sd"][i]
            codes = level_codes(z + p.strength, disc[i])
            v[rows] = np.where(np.isnan(v[rows]), np.nan, codes[rows])
        else:
            v[rows] += p.strength * (np.nanstd(v) + 1e-12)
        ds.x[_col(ds, i)] = v
    ds.x["period"] = period
    return {"features": feats}


# Data-generating phase.


def _ensure_period(ds, p) -> np.ndarray:
    """Create period = 1[row_id >= n_requested·(1 − share)] for the first drift axis."""
    if ds.period is None:
        n2 = round(drift_share(p) * ds.n_requested)
        ds.period = (ds.row_id >= ds.n_requested - n2).astype(np.int8)
    return ds.period


def _covariate_shift(ds, p, mask, rng):
    """Draw period two from a shifted P(X) using the same structural equations (P-8, data generation).

    Add strength·sd(latent) to the selected nodes' exogenous noise in period-two rows.
    Recompute X via the SEM using the sample_seed noise stream. Recompute mu0_x, tau_x,
    the SCM propensity, and segments from the model equations. Preserve contributions
    from other axes: e_logit_extra, mu0_cluster, and tau_hidden. Descendants of
    shifted nodes change here, unlike measurement_drift.
    """
    if p.strength == 0:
        return {"features": []}
    scm = ds.scm
    if scm is None or ds.sample_seed is None:
        msg = "covariate_shift requires Dataset.scm and sample_seed to recompute X from the noise stream"
        raise ValueError(msg)
    from .scm import sample_rng

    period = _ensure_period(ds, p)
    rows = mask & (period == 1)
    feats = p.params.get("features")
    if feats is None:
        cand = _nodes(ds)
        feats = sorted(
            rng.choice(cand, size=max(1, len(cand) // 3), replace=False).tolist()
        )
    feats = [int(i) for i in feats]
    eps = sample_rng(ds.sample_seed, "x").standard_normal((ds.n_requested, ds.graph.d))[
        ds.row_id
    ]
    sds = scm.calib["x_raw_sd"]
    for i in feats:
        eps[rows, i] += p.strength * sds[i]
    x, x_lat = scm.x_from_noise(eps)
    lat = ds.latent
    lat["x_true"] = np.ascontiguousarray(x, dtype=float)
    if x_lat is not None:
        lat["x_latent"] = x_lat
    ds.segments, ds.segments_unseen = scm.segments(x)
    lat["mu0_x"], lat["tau_x"], lat["e_assign"] = (
        scm.mu0(x),
        scm.tau(x),
        scm.e_assign(x),
    )
    lat.update(
        scm.mediator_terms(x)
    )  # Recompute mediator equation terms using the new X (0.10.0).
    for i in range(ds.graph.d):  # During data generation, observed X equals true X.
        ds.x[_col(ds, i)] = x[:, i]
    refresh_truth(ds)  # Recompute colliders with the new T and Y.
    ds.x["period"] = period
    return {"features": feats}


def _effect_drift(ds, p, mask, rng):
    """Change the effect in period two (P-21, data generation).

    params.shift_tau rotates tau_x toward a new projection of X, orthogonalized using
    calibration covariance. Variance stays fixed and the mean shrinks by sqrt(1 − s²).
    params.scale_tau multiplies the result and params.shift_ate adds a constant. Apply
    shift_tau, scale_tau, then shift_ate. This axis ignores strength.

    Check the orthogonal residual. Rotation is undefined when resid_var = 1 −
    corr² < 1e-6 (one modifier or collinear X), when there are no projection candidates,
    or when tau_x is constant. Raise ValueError in these cases; use scale_tau or
    shift_ate instead. Log the projection's node indices and weights.
    """
    period = _ensure_period(ds, p)
    rows = mask & (period == 1)
    lat, calib = ds.latent, _calib(ds)
    out: dict = {}
    s = float(p.params.get("shift_tau", 0.0))
    if s > 0:
        s = min(s, 1.0)
        tau_x = lat["tau_x"]
        proj, idx, w, sd_p = _projection_parts(
            ds, rng, roles=(ROLE_MODIFIER, ROLE_PREDICTOR, ROLE_CONFOUNDER)
        )
        tau_sd = float(calib["tau_x_sd"])
        if tau_sd <= 1e-12 or not idx:
            msg = (
                "effect_drift.shift_tau: no direction orthogonal to tau (tau_x is constant or there are "
                "no modifiers, predictors, or confounders); use scale_tau or shift_ate"
            )
            raise ValueError(msg)
        corr = float(np.einsum("k,k->", np.asarray(calib["tau_x_cov"])[idx], w)) / (
            sd_p * tau_sd
        )
        resid_var = 1.0 - corr**2
        if resid_var < 1e-6:
            msg = (
                "effect_drift.shift_tau: no direction orthogonal to tau (one modifier or "
                "collinear X); use scale_tau or shift_ate"
            )
            raise ValueError(msg)
        tau_std = (tau_x - calib["tau_x_mean"]) / tau_sd
        direction = (proj - corr * tau_std) / np.sqrt(resid_var)
        tau_new = np.sqrt(1 - s**2) * tau_x + s * tau_sd * direction
        lat["tau_x"] = np.where(rows, tau_new, tau_x)
        out.update(
            shift_tau=s,
            direction_features=[int(i) for i in idx],
            direction_w=[float(v) for v in w],
            direction_corr=corr,
        )
    scale_tau = float(p.params.get("scale_tau", 1.0))
    if scale_tau != 1.0:
        lat["tau_x"] = np.where(rows, scale_tau * lat["tau_x"], lat["tau_x"])
        out["scale_tau"] = scale_tau
    shift_ate = float(p.params.get("shift_ate", 0.0))
    if shift_ate != 0.0:
        lat["tau_x"] = np.where(rows, lat["tau_x"] + shift_ate, lat["tau_x"])
        out["shift_ate"] = shift_ate
    refresh_truth(ds)
    ds.x["period"] = period
    return out


def _policy_drift(ds, p, mask, rng):
    """Change treatment assignment probabilities in period two.

    P-21 (data generation) adds strength · proj(confounders) to the e_assign logit in
    period two (formerly drift.shift_e). The log records the nodes used under features.
    """
    if p.strength == 0:
        return {}
    period = _ensure_period(ds, p)
    rows = mask & (period == 1)
    proj, idx, _, _ = _projection_parts(ds, rng, roles=(ROLE_CONFOUNDER,))
    _add_e_logit(ds, rows, p.strength * proj)
    ds.x["period"] = period
    return {"shift_e": float(p.strength), "features": [int(i) for i in idx]}


def _overlap(ds, p, mask, rng):
    """Push e toward 0 or 1 in subregions by changing the assignment mechanism (P-6).

    Update ground-truth e. The target is e_new = (1 − s)·e + s·where(score > 0, 1 − m,
    m), m = 0.005. Store logit(e_new) − logit(e) in the latent state. This contribution
    is exactly zero at s = 0. score sums the confounders and standardizes them with
    calibration constants.
    """
    if ds.config.treatment.mode != "obs":
        msg = "the overlap pathology requires treatment.mode='obs'"
        raise ValueError(msg)
    m = 0.005
    calib = _calib(ds)
    e = e_assign_effective(ds)
    idx = ds.graph.covariate_indices(ROLE_CONFOUNDER)
    if idx.size:
        score = _conf_score(ds)
    else:
        score = (e - calib["e_assign_mean"]) / calib["e_assign_sd"]
    extreme = np.where(score > 0, 1.0 - m, m)
    e_new = (1.0 - p.strength) * e + p.strength * extreme
    _add_e_logit(ds, mask, _logit(e_new) - _logit(e))
    return {"features": [int(i) for i in idx]}


def _targeted_selection(ds, p, mask, rng):
    """Make assignment depend on the predicted control outcome.

    P-12 (targeted selection) adds the predicted mu0(x) to the assignment logit after
    standardization with calibration constants. features in the log lists the nodes in
    the mu0 equation. With direction=-1 (default), a poorer prognosis (lower mu0)
    increases the chance of treatment. This generates regularization-induced confounding
    for flexible learners (Hahn et al. 2020).
    """
    if ds.config.treatment.mode != "obs":
        msg = "targeted_selection requires treatment.mode='obs'"
        raise ValueError(msg)
    direction = float(p.params.get("direction", -1.0))
    coef = 3.0 * p.strength * direction
    idx = [
        int(i)
        for i in np.concatenate(
            [
                ds.graph.covariate_indices(ROLE_CONFOUNDER),
                ds.graph.covariate_indices(ROLE_PREDICTOR),
            ]
        )
    ]
    if coef != 0.0:  # Leave the latent state unchanged at zero strength.
        calib = _calib(ds)
        mu0_std = (ds.latent["mu0_x"] - calib["mu0_x_mean"]) / calib["mu0_x_sd"]
        _add_e_logit(ds, mask, coef * mu0_std)
    return {"direction": direction, "features": idx}


def _noncompliance(ds, p, mask, rng):
    """Model noncompliance: assigned treatment z may differ from the treatment received t, which determines y (P-17).

    Draw each unit's type once: complier, never-taker, or, with params.side="two",
    possibly always-taker. Use score = sqrt(1−h)·proj(X) + sqrt(h)·ν, where ν is the
    unit's standardized outcome noise. A hidden variable can confound received treatment
    while the assignment effect remains identifiable. Draw the projection direction
    first as an axis parameter, and record its nodes under score_features. latent["nc"]
    stores the parameters. The truth module computes tau_itt, tau_itt_cate, and e =
    P(t_received = 1 | W).
    """
    if ds.compliance is not None:
        msg = "noncompliance: this axis has already been applied and cannot be repeated"
        raise ValueError(msg)
    side = p.params.get("side", "one")
    if side not in ("one", "two"):
        msg = "noncompliance: side must be 'one' or 'two'"
        raise ValueError(msg)
    proj, idx, _, _ = _projection_parts(ds, rng)
    nc = make_nc_state(p, proj, mask)
    nu = outcome_noise_normal(ds)
    score = np.sqrt(1 - nc["hidden"]) * nc["score_x"] + np.sqrt(nc["hidden"]) * nu
    p_nt = np.clip(
        2 * p.strength * _sigmoid(-score), 0, 1
    )  # A poorer prognosis makes refusal more likely.
    compliance = np.zeros(ds.n, dtype=np.int8)
    nt = mask & (rng.random(ds.n) < p_nt)
    compliance[nt] = 1
    if side == "two":
        p_at = np.clip(p.strength * _sigmoid(score), 0, 1)
        at = mask & ~nt & (rng.random(ds.n) < p_at)
        compliance[at] = 2
    ds.compliance = compliance
    ds.latent["nc"] = nc
    refresh_truth(ds)
    return {
        "side": side,
        "hidden": nc["hidden"],
        "score_features": [int(i) for i in idx],
        "never_taker_share": float((compliance == 1).mean()),
        "always_taker_share": float((compliance == 2).mean()),
        "dag_effect": [
            "adds z -> t (z = prior t)",
            "adds compliance -> t (type of unit)",
        ]
        + (
            ["adds nu_y -> compliance (hidden outcome noise)"]
            if nc["hidden"] > 0
            else []
        ),
    }


def _hidden_modifier(ds, p, mask, rng):
    """Add an unobserved component U to tau as a hidden effect modifier (P-20).

    Strength sets the fraction of ITE variance unexplained by X: var_hidden = s/(1-s) ·
    var(tau_c), with var(tau_c) given by the calibration constant tau_c_sd². tau is the
    per-row effect including U. tau_cate = E[tau | W] and cvte = Var[tau | W] integrate
    over the combined U from every applied hidden_modifier and mediator noise. cvte
    gives a lower bound on expected squared ITE prediction error (PEHE) for affected rows.
    Replay draws fresh row-level U in the test population, keeping the modifier hidden.
    """
    s = float(np.clip(p.strength, 0.0, 0.9))
    lat = ds.latent
    sd_h = np.sqrt(s / (1 - s)) * float(_calib(ds)["tau_c_sd"])
    u = rng.standard_normal(ds.n)
    add = np.where(mask, sd_h * u, 0.0)
    lat["tau_hidden"] = lat.get("tau_hidden", np.zeros(ds.n)) + add
    lat["var_hidden"] = lat.get("var_hidden", np.zeros(ds.n)) + np.where(
        mask, sd_h**2, 0.0
    )
    refresh_truth(ds)
    return {"hidden_var_share": s, "pehe_floor": float(np.sqrt(ds.cvte.mean()))}


def _cluster(ds, p, mask, rng):
    """Add cluster-level variation and assignment (P-19).

    Assign rows to params.n_clusters unequal-sized clusters with weights
    exp(size_sigma·N(0,1)). The observed cluster column belongs to W. Strength is the
    latent-scale outcome ICC: mu0 += σ_b·u_c[cluster], with σ_b² = icc/(1 −
    icc)·Var_within. The calibration constants give Var_within = Var(mu0_x) + sd_eps²
    for gaussian and Var(mu0_x) for binary/ziln. In observational designs, params.icc_e
    uses a cluster effect correlated with u_c by params.rho to shift the e_assign logit.
    params.assign=cluster randomizes by cluster with a cluster-level u_t. Validation
    requires a pure RCT. Check that Z is constant within clusters in the data and log
    z_constant_within_cluster.

    First draw size weights, u_c, u_c', and cluster u_t (axis-level parameters), then
    assign rows. This order gives the test replay the same clusters. Zero strength has
    no effect unless params.emit is set.
    """
    _require_global(p)
    q = p.params
    icc, icc_e = float(p.strength), float(q.get("icc_e", 0.0))
    if icc == 0 and not q.get("emit", False):
        return {}
    calib = _calib(ds)
    k = int(q.get("n_clusters", 20))
    size_sigma, rho, assign = (
        float(q.get("size_sigma", 0.7)),
        float(q.get("rho", 1.0)),
        q.get("assign", "individual"),
    )
    weights = np.exp(size_sigma * rng.standard_normal(k))
    u_c, u_c2, u_t_c = rng.standard_normal(k), rng.standard_normal(k), rng.random(k)
    bp = q.get("block_propensity")
    z_b = (
        rng.standard_normal(k)
        if assign == "block" and bp is not None and not isinstance(bp, list)
        else None
    )
    cluster = rng.choice(k, size=ds.n, p=weights / weights.sum())
    lat = ds.latent
    oc = ds.config.outcome
    var_within = calib["mu0_x_sd"] ** 2 + (
        calib["sd_eps"] ** 2 if oc.family == "gaussian" else 0.0
    )
    if (
        oc.family == "lognormal" and oc.level_sigma > 0
    ):  # Since 0.11.0, calculate ICC on the log scale with σ_ε noise in σ_μ·mu0 units.
        var_within = calib["mu0_x_sd"] ** 2 + (oc.noise_sigma / oc.level_sigma) ** 2
    sigma_b = np.sqrt(icc / (1.0 - icc) * var_within)
    lat["u_cluster"] = u_c[cluster]
    lat["mu0_cluster"] = sigma_b * u_c[cluster]
    if assign == "cluster":
        lat["u_t"] = u_t_c[cluster]
    block = _block_assignment(ds, cluster, k, bp, z_b) if assign == "block" else None
    ds.cluster = cluster.astype(np.int64)
    ds.x["cluster"] = ds.cluster
    if icc_e > 0:
        u_e = rho * u_c + np.sqrt(max(1.0 - rho**2, 0.0)) * u_c2
        sigma_e = np.sqrt(icc_e / (1.0 - icc_e) * calib["e_logit_var"])
        _add_e_logit(ds, np.ones(ds.n, dtype=bool), sigma_e * u_e[cluster])
    else:
        refresh_truth(ds)
    z_const = None
    if assign == "cluster":
        z = ds.z if ds.z is not None else ds.t_true
        z_const = bool(
            all(np.unique(z[cluster == g]).size <= 1 for g in np.unique(cluster))
        )
        if not z_const:
            msg = "cluster.assign='cluster': assignment Z varies within a cluster"
            raise AssertionError(msg)
    if block is not None:
        z = ds.z if ds.z is not None else ds.t_true
        counts = np.bincount(cluster, weights=z.astype(float), minlength=k)
        if not np.array_equal(counts.astype(np.int64), block["k_b"]):
            msg = "cluster.assign='block': assigned count in a block differs from ⌊e_b·n_b + 0.5⌋"
            raise AssertionError(msg)
    sizes = np.bincount(cluster, minlength=k)
    sizes_nz = sizes[sizes > 0]
    out = {
        "icc_y": icc,
        "icc_e": icc_e,
        "n_clusters": k,
        "rho": rho,
        "assign": assign,
        "size_min": int(np.min(sizes_nz)),
        "size_median": float(np.median(sizes_nz)),
        "size_max": int(np.max(sizes_nz)),
        "icc_y_achieved": icc_anova(ds.y_true.astype(float), cluster),
    }
    if z_const is not None:
        out["z_constant_within_cluster"] = z_const
    if block is not None:
        nz = block["n_b"] > 0
        share = block["k_b"][nz] / block["n_b"][nz]
        out.update(
            block_share_min=float(share.min()),
            block_share_max=float(share.max()),
            blocks_without_treatment=int((block["k_b"][nz] == 0).sum()),
            block_propensity=(
                "treatment.propensity"
                if bp is None
                else "list"
                if isinstance(bp, list)
                else float(bp)
            ),
        )
    return out


def _block_assignment(ds, cluster: np.ndarray, k: int, bp, z_b) -> dict:
    """Assign treatment to a fixed number of rows in each block.

    With cluster.assign=block (0.11.0), complete randomization assigns Z = 1 to
    exactly k_b = ⌊e_b·n_b + 0.5⌋ rows with the smallest u_t in block b. Replace each
    row's latent u_t with (within-block rank of u_t + 0.5) / n_b and P(Z = 1 | W) with
    the block's k_b / n_b in latent["e_block"]. Exactly k_b rows then satisfy u_t < e.
    For target block proportions e_b, None uses treatment.propensity, a list supplies
    the proportions, and a scalar s uses σ(logit(e) + s·z_b).
    """
    lat = ds.latent
    e = float(ds.config.treatment.propensity)
    if bp is None:
        e_b = np.full(k, e)
    elif isinstance(bp, list):
        e_b = np.asarray(bp, dtype=float)
    else:
        e_b = _sigmoid(_logit(e) + float(bp) * z_b)
    n_b = np.bincount(cluster, minlength=k).astype(np.int64)
    k_b = np.floor(e_b * n_b + 0.5).astype(np.int64)
    order = np.lexsort(
        (lat["u_t"], cluster)
    )  # Sort by block, then by u_t within each block.
    rank = np.empty(ds.n, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(n_b)[:-1]])
    rank[order] = np.arange(ds.n) - starts[cluster[order]]
    nb_row = n_b[cluster].astype(float)
    lat["u_t"] = (rank + 0.5) / nb_row
    lat["e_block"] = k_b[cluster] / nb_row
    return {"n_b": n_b, "k_b": k_b}


def icc_anova(y: np.ndarray, groups: np.ndarray) -> float:
    """Estimate ICC by ANOVA for unequal group sizes: (MSB − MSW) / (MSB + (n0 − 1)·MSW)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    k, n = uniq.size, y.size
    if k < 2 or n <= k:
        return float("nan")
    sizes = np.bincount(inv)
    means = np.bincount(inv, weights=y) / sizes
    grand = y.mean()
    ssb = float((sizes * (means - grand) ** 2).sum())
    ssw = float(((y - means[inv]) ** 2).sum())
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - (sizes**2).sum() / n) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


# Selection phase.


def _selection(ds, p, mask, rng):
    """Introduce selection bias by excluding rows as a function of (X, T) (P-7).

    params.t_weight (default 1.0) weights treatment relative to the confounder sum in
    the score. Standardize using calibration moments of that sum and T ~ Bernoulli(e).
    Multiply the stage retention probabilities q_t(x) = 1 − strength·σ(score(x, t)) into
    sel_q1 and sel_q0 (product over stages). After selection truth.refresh_truth
    uses Bayes' rule with e_source to update e = P(T = 1 | W, S = 1) and the
    selected-population fields. The first stage saves ate_source and n_source.
    Each stage logs stage, n_before, n_after, kept_share, and t_share_shift.
    """
    t_weight = float(p.params.get("t_weight", 1.0))
    calib = _calib(ds)
    idx = ds.graph.covariate_indices(ROLE_CONFOUNDER)
    base = _x_true(ds, idx).sum(axis=1) if idx.size else np.zeros(ds.n)
    if idx.size:
        m = calib["conf_sum_mean"] + t_weight * calib["t_mean"]
        var = (
            calib["conf_sum_sd"] ** 2
            + t_weight**2 * calib["t_var"]
            + 2 * t_weight * calib["conf_t_cov"]
        )
    else:  # When base ≡ 0, only the variance of T remains.
        m, var = t_weight * calib["t_mean"], t_weight**2 * calib["t_var"]
    sd = np.sqrt(var) if var > 1e-24 else 1.0

    def keep_prob(t):
        return np.where(
            mask, 1.0 - p.strength * _sigmoid((base + t_weight * t - m) / sd), 1.0
        )

    drop_prob = 1.0 - keep_prob(ds.t_true.astype(float))
    keep = ~(mask & (rng.random(ds.n) < drop_prob))
    t_before, t_before_aff = (
        ds.t_true.mean(),
        ds.t_true[mask].mean() if mask.any() else float("nan"),
    )
    lat = ds.latent
    stage = 1 + sum(1 for e in ds.pathology_log if e.get("kind") == "selection")
    if "sel_q1" not in lat:  # The first stage records the source population.
        lat["sel_q1"], lat["sel_q0"] = np.ones(ds.n), np.ones(ds.n)
        ds.ate_source, ds.n_source = ds.ate, int(ds.n)
    lat["sel_q1"] = lat["sel_q1"] * keep_prob(1.0)
    lat["sel_q0"] = lat["sel_q0"] * keep_prob(0.0)
    n_before = int(ds.n)
    kept_mask = mask[keep]
    _take_rows(ds, np.flatnonzero(keep))
    selection_truth(
        ds
    )  # Read e_source, e, and *_selected from the latent state without recomputing the data-generating process.
    return {
        "stage": stage,
        "n_before": n_before,
        "n_after": int(ds.n),
        "kept_share": float(keep.mean()),
        "t_share_shift": float(ds.t_true.mean() - t_before),
        "t_share_shift_affected": float(ds.t_true[kept_mask].mean() - t_before_aff)
        if kept_mask.any()
        else 0.0,
        "affected_rows_share": float(1.0 - keep.mean()),
    }


def _duplicate_rows(ds, p, mask, rng):
    """Duplicate rows as repeated observations without an observed unit identifier (P-24, 0.6.0).

    Strength is the fraction of eligible rows that receive copies. params.k sets the
    number of copies (default 1). params.noise sets the standard deviation of noise
    added to continuous X in copies, as a fraction of the column standard deviation.
    Discrete values stay intact. params.position is end or shuffle. Each copy represents
    the same observation: t, y, ground truth, and row_id agree. dup_index is 0 for the
    original and 1..k for copies, so (row_id, dup_index) uniquely identifies
    observations. Duplicating rows increases n and can take it above n_requested. With
    noise > 0, observed X in the copies differs from true X. Log affected_columns
    (continuous features) and affected_rows_share (fraction of copies) and set
    targets.cate.observed_equivalent = False.
    """
    k = int(p.params.get("k", 1))
    noise = float(p.params.get("noise", 0.0))
    position = p.params.get("position", "end")
    chosen = np.flatnonzero(mask & (rng.random(ds.n) < p.strength))
    if chosen.size == 0:
        return {
            "dup_rows_share": 0.0,
            "n_added": 0,
            "noise": noise,
            "affected_rows_share": 0.0,
        }
    n = ds.n
    idx = np.concatenate([np.arange(n)] + [chosen] * k)
    dup = np.concatenate(
        [np.zeros(n, dtype=np.int64)]
        + [np.full(chosen.size, j, dtype=np.int64) for j in range(1, k + 1)]
    )
    if position == "shuffle":
        perm = rng.permutation(idx.size)
        idx, dup = idx[perm], dup[perm]
    _take_rows(ds, idx)
    ds.dup_index = dup if ds.dup_index is None else np.where(dup > 0, dup, ds.dup_index)
    copies = dup > 0
    affected = []
    if noise > 0:
        disc = ds.graph.discrete or {}
        for i in ds.observed_nodes():
            if i in disc:
                continue
            v = _obs_col(ds, i)
            v[copies] += rng.normal(
                0.0, noise * (np.nanstd(v) + 1e-12), int(copies.sum())
            )
            ds.x[_col(ds, i)] = v
            affected.append(_col(ds, i))
    return {
        "dup_rows_share": float(copies.mean()),
        "n_added": int(copies.sum()),
        "noise": noise,
        "affected_columns": affected,
        "affected_rows_share": float(copies.mean()),
    }


def _take_rows(ds, idx: np.ndarray):
    """Select rows by idx, allowing subsets and repetitions across row fields, the latent state, x, and x_missing."""
    n = ds.n
    for name in _ROW_FIELDS:
        v = getattr(ds, name)
        if v is not None:
            setattr(ds, name, v[idx])
    ds.latent = {k: _take_value(v, idx, n) for k, v in ds.latent.items()}
    ds.x = ds.x.iloc[idx].reset_index(drop=True)
    if ds.x_missing is not None:
        ds.x_missing = ds.x_missing.iloc[idx].reset_index(drop=True)


def _take_value(v, idx, n: int):
    """Take arrays with first dimension n by idx, including n × k arrays.

    Traverse nested dictionaries and retain scalars.
    """
    if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
        return v[idx]
    if isinstance(v, dict):
        return {k: _take_value(u, idx, n) for k, u in v.items()}
    return v


_KINDS = {
    "noise_y": _noise_y,
    "noise_t": _noise_t,
    "noise_x": _noise_x,
    "outliers": _outliers,
    "missing": _missing,
    "heteroscedastic": _heteroscedastic,
    "overlap": _overlap,
    "selection": _selection,
    "hidden_confounder": _hidden_confounder,
    "bad_controls": _bad_controls,
    "targeted_selection": _targeted_selection,
    "noncompliance": _noncompliance,
    "hidden_modifier": _hidden_modifier,
    "measurement_drift": _measurement_drift,
    "covariate_shift": _covariate_shift,
    "effect_drift": _effect_drift,
    "policy_drift": _policy_drift,
    "cluster": _cluster,
    "duplicate_rows": _duplicate_rows,
    "quantize_y": _quantize_y,
}
_PHASE_LABEL = {
    PHASE_DGP: "dgp",
    PHASE_SELECTION: "selection",
    PHASE_MEASUREMENT: "measurement",
}


# Measure actual impact.


def _snapshot(ds) -> dict:
    lat = {
        k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
        for k, v in ds.latent.items()
        if k in _DGP_FIELDS
    }
    return {
        "n": ds.n,
        "x": ds.x.copy(),
        "t": np.array(ds.t, copy=True),
        "y": np.array(ds.y, copy=True),
        "latent": lat,
        "compliance": None if ds.compliance is None else ds.compliance.copy(),
        "cluster": None if ds.cluster is None else ds.cluster.copy(),
        "period": None if ds.period is None else ds.period.copy(),
    }


def _changed(a, b) -> np.ndarray:
    """Mark rows with differing values, treating pairs of NaNs as equal.

    If b is None, mark nonzero values of the new field.
    """
    a = np.asarray(a)
    if b is None:  # A new field affects rows where its value is nonzero.
        nz = np.nan_to_num(a.astype(float), nan=0.0) != 0.0
        return nz if nz.ndim == 1 else np.any(nz, axis=tuple(range(1, nz.ndim)))
    b = np.asarray(b)
    if a.shape != b.shape:
        return np.ones(a.shape[0], dtype=bool)
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        af, bf = a.astype(float), b.astype(float)
        diff = ~((af == bf) | (np.isnan(af) & np.isnan(bf)))
    else:
        diff = a != b
    return diff if diff.ndim == 1 else np.any(diff, axis=tuple(range(1, diff.ndim)))


def _impact(ds, before: dict, phase: int, extra: dict) -> dict:
    """Compare the dataset before and after applying one axis.

    Report active, affected_columns, affected_rows_share, columns_added,
    columns_removed, and affected_fields for data-generating axes. Axes that change the
    row set report their own affected_rows_share.
    """
    out: dict = {"active": False, "affected_columns": [], "affected_rows_share": 0.0}
    cols_before, cols_after = (
        list(map(str, before["x"].columns)),
        list(map(str, ds.x.columns)),
    )
    added = [c for c in cols_after if c not in cols_before]
    removed = [c for c in cols_before if c not in cols_after]
    if added:
        out["columns_added"] = added
    if removed:
        out["columns_removed"] = removed
    same_rows = ds.n == before["n"] and phase != PHASE_SELECTION
    if same_rows:
        rows = np.zeros(ds.n, dtype=bool)
        for c in cols_after:
            if c in cols_before:
                d = _changed(ds.x[c].to_numpy(), before["x"][c].to_numpy())
                if d.any():
                    out["affected_columns"].append(c)
                    rows |= d
        for name, b in (("t", before["t"]), ("y", before["y"])):
            d = _changed(getattr(ds, name), b)
            if d.any():
                out["affected_columns"].append(name)
                rows |= d
        if phase == PHASE_DGP:
            fields = []
            for k in _DGP_FIELDS:
                v = ds.latent.get(k)
                if v is None or not isinstance(v, np.ndarray):
                    continue
                d = _changed(v, before["latent"].get(k))
                if d.any():
                    fields.append(k)
                    rows |= d
            for k in ("compliance", "cluster", "period"):
                v = getattr(ds, k)
                if v is None:
                    continue
                d = _changed(v, before[k])
                if (
                    k == "cluster" and before[k] is None
                ):  # A newly added cluster field affects every row, including rows whose cluster ID is zero.
                    d = np.ones(ds.n, dtype=bool)
                if d.any():
                    fields.append(k)
                    rows |= d
            out["affected_fields"] = fields
        out["affected_rows_share"] = float(rows.mean()) if ds.n else 0.0
    else:
        out["affected_rows_share"] = float(extra.get("affected_rows_share", 1.0))
        out["affected_columns"] = list(extra.get("affected_columns", []))
    out["active"] = bool(
        out["affected_columns"]
        or added
        or removed
        or out.get("affected_fields")
        or (not same_rows and (ds.n != before["n"] or out["affected_rows_share"] > 0))
    )
    return out


def _overrides(p: PathologyConfig, base: PathologyConfig | None) -> dict:
    """Return recipe overrides that distinguish a plan axis from the scenario axis with the same index."""
    if base is None:
        return {
            "strength": p.strength,
            "target": p.target,
            "segments": p.segments,
            "segment_share": p.segment_share,
            "params": dict(p.params),
        }
    out: dict = {}
    for f in ("strength", "target", "segments", "segment_share"):
        if getattr(p, f) != getattr(base, f):
            out[f] = getattr(p, f)
    params = {
        k: v for k, v in p.params.items() if k not in base.params or base.params[k] != v
    }
    params.update({k: None for k in base.params if k not in p.params})
    if params:
        out["params"] = params
    return out


def apply_pathologies(clean, seed: int, plan=None):
    """Return a corrupted copy, keeping clean and ds.config intact (S-2).

    A given dataset, seed, and plan yield reproducible paired runs. Configuration
    validation checks phase order. Each axis uses its index to obtain an independent
    stream from truth.new_axis_rng. A plan lists (index, PathologyConfig) entries in
    place of config.pathologies for generate_split(test_dgp="replay") or
    scmgen.io.replay. The index selects the stream, while the config gives effective
    parameters. The recipe records differences from the scenario axis in
    applied_pathologies[].overrides.
    """
    clean.config.validate()
    copies = {
        name: (
            getattr(clean, name).copy() if getattr(clean, name) is not None else None
        )
        for name in _ROW_FIELDS
    }
    ds = dataclasses.replace(
        clean,
        x=clean.x.copy(),
        pathology_log=[],
        graph=copy.deepcopy(clean.graph),
        config=copy.deepcopy(clean.config),
        latent=copy.deepcopy(clean.latent),
        x_missing=None if clean.x_missing is None else clean.x_missing.copy(),
        applied_pathologies=[],
        pathology_seed=int(seed),
        **copies,
    )
    if plan is None:
        plan = list(enumerate(ds.config.pathologies))
    for i, p in plan:
        i = int(i)
        fn = _KINDS.get(p.kind)
        if fn is None:
            msg = f"unknown pathology: {p.kind}"
            raise ValueError(msg)
        phase = pathology_phase(p)
        rng = new_axis_rng(seed, i, p.kind)
        mask, seg_list = _row_mask(p, ds.segments, rng)
        before = _snapshot(ds)
        extra = fn(ds, p, mask, rng) or {}
        impact = _impact(ds, before, phase, extra)
        number, number_extra = pathology_number(p)
        entry = {
            "kind": p.kind,
            "pathology": number,
            "pathology_extra": number_extra,
            "phase": _PHASE_LABEL[phase],
            "index": i,
            "strength": p.strength,
            "target": p.target,
            "segments": seg_list,
            "affected_share": float(mask.mean()),
        }
        entry.update(extra)
        entry.update(impact)
        ds.pathology_log.append(entry)
        base = ds.config.pathologies[i] if i < len(ds.config.pathologies) else None
        ds.applied_pathologies.append(
            {
                "index": i,
                "kind": p.kind,
                "strength": p.strength,
                "target": p.target,
                "segments": None if p.segments is None else list(p.segments),
                "segment_share": p.segment_share,
                "params": copy.deepcopy(dict(p.params)),
                "overrides": _overrides(p, base),
            }
        )
    return ds
