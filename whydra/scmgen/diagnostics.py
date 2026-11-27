"""Summarize the generated data (B.3).

`data_properties(x)` counts significant modes of the X correlation matrix using the
Marchenko–Pastur bound. A data owner can share this single summary of feature structure
without disclosing the observations. For example, arXiv:2607.10297 reports 10–16
structural modes for 140–425 financial variables. The function only reads data and uses
no random draws.

`outcome_profile(ds, level=None)` (0.11.0) summarizes a continuous outcome using
measures also available from real randomized experiments. These include baseline
predictability, tails of the outcome, baseline and residual, noise growth with the
baseline, effects by baseline quintile, and quantile shifts. They make it possible to
compare a scenario quantitatively with a target. The default baseline is the
ground-truth E[Y(0) | W, latent variables] from `ds.mu0`, which needs no model fit. Pass
an external model's prediction as `level` to calculate the same measures with it.
"""

import numpy as np
import pandas as pd


def data_properties(x, n: int | None = None) -> dict:
    """Summarize the spectrum of the X correlation matrix.

    x is a DataFrame or an n × d array. Replace NaNs from missingness pathologies with
    column means and exclude constant columns. n is the observation count for the
    Marchenko–Pastur bound and defaults to the row count of x. Return
    n_significant_modes (the number of correlation eigenvalues above λ₊ = (1 +
    sqrt(d/n))²), mp_upper (λ₊), top_eigenvalues (the three largest), d, and n.
    """
    arr = (
        x.to_numpy(dtype=float)
        if isinstance(x, pd.DataFrame)
        else np.asarray(x, dtype=float)
    )
    if arr.ndim != 2:
        msg = "data_properties expects an n-by-d matrix"
        raise ValueError(msg)
    n_rows = arr.shape[0]
    n_eff = int(n) if n is not None else n_rows
    col_mean = np.nanmean(arr, axis=0)
    arr = np.where(np.isnan(arr), col_mean, arr)
    keep = arr.std(axis=0) > 1e-12
    arr = arr[:, keep]
    d = arr.shape[1]
    if d == 0 or n_eff < 2:
        return {
            "n_significant_modes": 0,
            "mp_upper": float("nan"),
            "top_eigenvalues": [],
            "d": d,
            "n": n_eff,
        }
    corr = np.corrcoef(arr, rowvar=False) if d > 1 else np.ones((1, 1))
    ev = np.sort(np.linalg.eigvalsh(corr))[::-1]
    mp_upper = (1.0 + np.sqrt(d / n_eff)) ** 2
    return {
        "n_significant_modes": int((ev > mp_upper).sum()),
        "mp_upper": float(mp_upper),
        "top_eigenvalues": [float(v) for v in ev[:3]],
        "d": d,
        "n": n_eff,
    }


PROFILE_QUANTILES = (0.5, 0.9, 0.99, 0.999)


def _tail_share(v: np.ndarray, top: float = 0.01) -> float:
    """Return the share of total squared deviation due to the rows farthest from the mean.

    Use the fraction ``top`` of rows, rounded down to at least one row.
    Return zero if all values are equal.
    """
    d2 = (v - v.mean()) ** 2
    total = float(d2.sum())
    if total <= 0:
        return 0.0
    k = max(int(v.size * top), 1)
    return float(np.sort(d2)[-k:].sum() / total)


def outcome_profile(ds, level=None) -> dict:
    """Profile a dataset's continuous outcome (0.11.0).

    Use recorded `ds.t` to identify control rows. The observed outcome `ds.y` includes
    measurement effects, so outliers and rounding affect the tails as they would in real
    data.

    ``level`` predicts the control outcome for each row. It defaults to the
    generator's reference baseline, ``ds.mu0``. The returned dictionary contains:

    * `n` counts all rows, `d` counts observed X columns, and `treated_share`
      is the fraction of rows treated. Among nonmissing outcomes, `zero_share`
      is the fraction equal to zero and `distinct_share` is the number of
      distinct values divided by the number of observations.

    * `r2_level` is 1 − Var(y − level) / Var(y) in controls. `max_corr_feature` is the
      largest absolute correlation of y with a single X column in controls.

    * `tail_y`, `tail_level`, and `tail_resid` are the shares of total squared
      deviations in the top 1% of control rows, calculated for y, level, and y − level
      respectively.

    * `g` measures how noise grows with level. It is half the slope of log(residual²)
      against log(level). It uses control rows with a positive baseline above
      the fifth control percentile and a nonzero residual. It is NaN if 100 or
      fewer rows qualify.

    * `ate_over_sd` is the difference between arm means divided by sd(y).

    * `effect_by_level` gives the difference in arm means for each baseline quintile in
      units of |ATE|. `effect_by_level_rel` divides that arm difference by the
      absolute value of the quintile's mean control outcome. `tau_cate_by_level_rel`
      uses the generator's ground truth, mean(tau_cate) / abs(mean(mu0)), within
      each quintile. A zero denominator gives NaN.

    * `corr_tau_level` is the correlation of tau_cate with level.

    * `quantile_shift` maps each q in PROFILE_QUANTILES to (Q_y1(q) − Q_y0(q)) /
      |Q_y0(q)| using the observed arm outcomes.

    Only NumPy is required. No random draws are used.
    """
    y = np.asarray(ds.y, dtype=float)
    t = np.asarray(ds.t)
    lvl = np.asarray(ds.mu0 if level is None else level, dtype=float)
    if lvl.shape != y.shape:
        msg = f"outcome_profile: expected level with shape {y.shape}, got {lvl.shape}"
        raise ValueError(msg)
    ok = ~np.isnan(y)
    c, tr = ok & (t == 0), ok & (t == 1)
    yc, lc = y[c], lvl[c]
    res = yc - lc
    out: dict = {
        "n": int(y.size),
        "d": int(ds.x.shape[1]),
        "treated_share": float((t == 1).mean()),
        "zero_share": float((y[ok] == 0).mean()),
        "distinct_share": float(np.unique(y[ok]).size / ok.sum()),
    }
    out["r2_level"] = (
        float(1.0 - res.var() / yc.var()) if yc.var() > 0 else float("nan")
    )
    xs = ds.x.to_numpy(dtype=float)[c]
    corr = []
    for j in range(xs.shape[1]):
        col = xs[:, j]
        m = ~np.isnan(col)
        if m.sum() > 2 and col[m].std() > 0 and yc[m].std() > 0:
            corr.append(abs(float(np.corrcoef(col[m], yc[m])[0, 1])))
    out["max_corr_feature"] = max(corr) if corr else float("nan")
    out["tail_y"], out["tail_level"], out["tail_resid"] = (
        _tail_share(yc),
        _tail_share(lc),
        _tail_share(res),
    )
    pos = (lc > np.quantile(lc, 0.05)) & (res != 0) & (lc > 0)
    out["g"] = (
        float(np.polyfit(np.log(lc[pos]), np.log(res[pos] ** 2), 1)[0] / 2.0)
        if pos.sum() > 100
        else float("nan")
    )
    ate = float(y[tr].mean() - y[c].mean())
    out["ate_over_sd"] = ate / float(y[ok].std())
    edges = np.quantile(lvl[ok], [0.2, 0.4, 0.6, 0.8])
    b = np.searchsorted(edges, lvl)
    eff, rel, rel_true = [], [], []
    tau_cate, mu0 = (
        np.asarray(ds.tau_cate, dtype=float),
        np.asarray(ds.mu0, dtype=float),
    )
    for k in range(5):
        m = b == k
        if not (
            (m & tr).any() and (m & c).any()
        ):  # An empty quintile can occur when the baseline is degenerate.
            eff.append(float("nan"))
            rel.append(float("nan"))
            rel_true.append(float("nan"))
            continue
        d = float(y[m & tr].mean() - y[m & c].mean())
        eff.append(d / abs(ate) if ate != 0 else float("nan"))
        base = float(y[m & c].mean())
        rel.append(d / abs(base) if base != 0 else float("nan"))
        mu = float(mu0[m].mean())
        rel_true.append(
            float(tau_cate[m].mean()) / abs(mu) if mu != 0 else float("nan")
        )
    out["effect_by_level"], out["effect_by_level_rel"], out["tau_cate_by_level_rel"] = (
        eff,
        rel,
        rel_true,
    )
    out["corr_tau_level"] = (
        float(np.corrcoef(tau_cate[ok], lvl[ok])[0, 1])
        if tau_cate[ok].std() > 0 and lvl[ok].std() > 0
        else 0.0
    )
    q1, q0 = np.quantile(y[tr], PROFILE_QUANTILES), np.quantile(y[c], PROFILE_QUANTILES)
    out["quantile_shift"] = {
        float(q): float((a - b0) / abs(b0)) if b0 != 0 else float("nan")
        for q, a, b0 in zip(PROFILE_QUANTILES, q1, q0, strict=False)
    }
    return out
