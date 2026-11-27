"""Local evidence operations for native momentary conditional independence.

Benjamini--Hochberg uses monotone step-up adjusted probabilities; the tested
family can be restricted to supplied candidate assumptions.
"""

import numpy as np


def symmetrize_evidence(p_matrix, val_matrix, link_assumptions, conf_matrix=None):
    """Merge zero-lag directions by p-value, respecting fixed arrowheads.

    An undirected pair uses the less significant direction. Equal p-values
    select the direction from the smaller-index source, independent of the
    statistic's magnitude. A supplied arrow uses its forward direction.
    """
    probabilities = np.array(p_matrix, dtype=float, copy=True)
    values = np.array(val_matrix, dtype=float, copy=True)
    confidence = None if conf_matrix is None else np.array(conf_matrix, copy=True)
    for i in range(probabilities.shape[0]):
        for j in range(i + 1, probabilities.shape[1]):
            forward = link_assumptions.get(j, {}).get((i, 0), "")
            reverse = link_assumptions.get(i, {}).get((j, 0), "")
            if not forward and not reverse:
                continue
            if forward in {"-->", "-?>"}:
                source, target = i, j
            elif reverse in {"-->", "-?>"}:
                source, target = j, i
            elif probabilities[i, j, 0] >= probabilities[j, i, 0]:
                source, target = i, j
            else:
                source, target = j, i
            probabilities[i, j, 0] = probabilities[j, i, 0] = probabilities[
                source, target, 0
            ]
            values[i, j, 0] = values[j, i, 0] = values[source, target, 0]
            if confidence is not None:
                interval = confidence[source, target, 0].copy()
                confidence[i, j, 0] = confidence[j, i, 0] = interval
    return {"p_matrix": probabilities, "val_matrix": values, "conf_matrix": confidence}


def adjust_pvalues(
    p_matrix,
    *,
    fdr_method="fdr_bh",
    exclude_contemporaneous=True,
    link_assumptions=None,
):
    """Adjust configured directed-array entries, including self-lags.

    Reverse entries of known contemporaneous arrows are not hypotheses.
    Undirected contemporaneous entries count in both directions when enabled,
    preserving the existing matrix-based tested-family convention.
    Entries outside the family are returned unchanged.
    """
    probabilities = np.asarray(p_matrix, dtype=float)
    if probabilities.ndim != 3 or probabilities.shape[0] != probabilities.shape[1]:
        msg = "p_matrix must have shape (N, N, lags)"
        raise ValueError(msg)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        msg = "p_matrix must contain finite nonnegative probabilities"
        raise ValueError(msg)
    corrected = probabilities.copy()
    if fdr_method in {None, "none"}:
        return corrected
    if fdr_method != "fdr_bh":
        msg = "Supported FDR methods are none and fdr_bh"
        raise ValueError(msg)
    family = np.zeros(probabilities.shape, dtype=bool)
    if link_assumptions is None:
        family[:] = True
    else:
        for j, candidates in link_assumptions.items():
            for (i, lag), mark in candidates.items():
                if mark not in {"<?-", "<--"}:
                    family[i, j, -lag] = True
    for node in range(probabilities.shape[0]):
        family[node, node, 0] = False
    if exclude_contemporaneous:
        family[:, :, 0] = False
    samples = probabilities[family]
    if not len(samples):
        return corrected
    order = np.argsort(samples, kind="stable")
    adjusted = samples[order] * len(samples) / np.arange(1, len(samples) + 1)
    adjusted = np.minimum(1.0, np.minimum.accumulate(adjusted[::-1])[::-1])
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    corrected[family] = restored
    return corrected
