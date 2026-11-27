"""Define true segments (S-1) on a quantile grid over one or two covariates.

Segments are deterministic given X and use no random draws. The same segmentation
defines segment-specific tau (M-2), localizes pathologies, and validates segment
curation.

Since 0.6.0, SCM calibration supplies both quantile boundaries (`calib["seg_edges"]`)
and cell numbering (`calib["seg_ids"]`). The grid belongs to the model, so a row's label
is independent of other rows in its batch. A raw cell (`raw_cells`) is an index in the
full K0 × K1 grid used to draw the segment's tau level. Segments are numbered 0 through
K−1 over cells occupied during calibration. Empty cells receive the nearest ID in grid
order.
"""

from collections.abc import Mapping, MutableMapping

import numpy as np

from .config import SegmentConfig
from .graph import (
    ROLE_COLLIDER,
    ROLE_CONFOUNDER,
    ROLE_INSTRUMENT,
    ROLE_MED_CONFOUNDER,
    ROLE_MED_PREDICTOR,
    ROLE_MODIFIER,
    ROLE_NOISE,
    ROLE_PREDICTOR,
    ROLE_SIDE,
    ROLE_UPSTREAM,
    SampledGraph,
)

# Role priority when choosing grid covariates. Favor features related to the outcome or effect,
# use noise last, and always exclude colliders and instruments. Including an instrument in a
# segment-tau grid would introduce an edge to Y and invalidate it as an instrument. Since
# 0.10.0, med_confounder and med_predictor follow predictor, and upstream and side precede noise.
_FEATURE_PRIORITY = (
    ROLE_CONFOUNDER,
    ROLE_MODIFIER,
    ROLE_PREDICTOR,
    ROLE_MED_CONFOUNDER,
    ROLE_MED_PREDICTOR,
    ROLE_UPSTREAM,
    ROLE_SIDE,
    ROLE_NOISE,
)
# These roles lack the edge to Y that a segment-tau grid would introduce. Such an edge would
# invalidate the upstream and side roles.
_SEGMENT_TAU_FORBIDDEN = (
    ROLE_NOISE,
    ROLE_INSTRUMENT,
    ROLE_COLLIDER,
    ROLE_UPSTREAM,
    ROLE_SIDE,
)


def _default_features(graph: SampledGraph, segment_tau: bool = False) -> list[int]:
    """Choose up to two grid covariates by role rather than topological position.

    When segment_tau=True, exclude roles that would acquire an edge to Y through the
    grid.
    """
    feats: list[int] = []
    for role in _FEATURE_PRIORITY:
        if segment_tau and role in _SEGMENT_TAU_FORBIDDEN:
            continue
        feats.extend(int(i) for i in graph.covariate_indices(role))
        if len(feats) >= 2:
            break
    return feats[:2]


def segment_features(
    graph: SampledGraph, cfg: SegmentConfig, segment_tau: bool = False
) -> list[int]:
    """Return the grid node indices, chosen explicitly or by role priority.

    For segment-specific tau, noise, instrument, collider, upstream, and side roles
    cannot supply grid features. They would gain an edge to Y, so selection raises
    ValueError.
    """
    feats = (
        _resolve_features(graph, cfg.features)
        if cfg.features is not None
        else _default_features(graph, segment_tau)
    )
    if not feats:
        msg = "no covariates available for segmentation"
        raise ValueError(msg)
    if segment_tau:
        bad = [i for i in feats if graph.roles[i] in _SEGMENT_TAU_FORBIDDEN]
        if bad:
            msg = (
                f"segments.features: for tau.form='segment', grid feature x{bad[0]} has unsupported role "
                f"{graph.roles[bad[0]]!r}. The outcome equation does not use noise, instrument, collider, "
                "upstream or side covariates."
            )
            raise ValueError(msg)
    return feats


def _resolve_features(graph: SampledGraph, features) -> list[int]:
    """Return grid indices.

    Since 0.4.0, 'categorical' and 'binary' select the first discrete node of that kind.
    """
    out: list[int] = []
    for f in features:
        if isinstance(f, str):
            if f not in ("categorical", "binary"):
                msg = f"segments.features: unknown token {f!r} (expected categorical or binary)"
                raise ValueError(msg)
            nodes = graph.discrete_nodes(f)
            if not nodes:
                msg = f"segments.features: the graph has no nodes of kind {f!r}"
                raise ValueError(msg)
            out.append(nodes[0])
        else:
            out.append(int(f))
    return out


def grid_shape(
    graph: SampledGraph, cfg: SegmentConfig, feats: list[int]
) -> tuple[int, int]:
    """Return the grid dimensions (K0, K1), with K1 = 1 for a single feature.

    A discrete feature uses its k levels as bins in place of quantile bins.
    """
    disc = graph.discrete or {}
    if len(feats) == 1:
        k0, k1 = cfg.n_segments, 1
    else:
        k0 = int(np.ceil(np.sqrt(cfg.n_segments)))
        k1 = int(np.ceil(cfg.n_segments / k0))
    ks = [k0, k1]
    for j, i in enumerate(feats):
        if i in disc:
            ks[j] = int(disc[i]["levels"])
    return ks[0], ks[1]


def raw_cells(
    x: np.ndarray,
    graph: SampledGraph,
    feats: list[int],
    shape: tuple[int, int],
    calib: Mapping,
    fit: bool = False,
) -> np.ndarray:
    """Return each row's raw grid cell index in 0..K0·K1−1.

    Quantile boundaries come from `calib["seg_edges"]`, one entry per feature. The entry
    for a discrete feature is None.
    """
    disc = graph.discrete or {}
    if fit:
        assert isinstance(calib, MutableMapping)
        edges: list[np.ndarray | None] = [None] * len(feats)
        calib["seg_edges"] = edges
    labels = np.zeros(x.shape[0], dtype=int)
    for j, (i, k) in enumerate(zip(feats, shape, strict=False)):
        v = x[:, i]
        if i in disc:
            b = v.astype(int)
        elif k <= 1:
            b = np.zeros(v.size, dtype=int)
        else:
            if fit:
                calib["seg_edges"][j] = np.quantile(v, np.linspace(0, 1, k + 1)[1:-1])
            b = np.searchsorted(np.asarray(calib["seg_edges"][j], dtype=float), v)
        labels = labels * k + b
    return labels


def compact_segments(
    raw: np.ndarray, calib: Mapping, fit: bool = False
) -> tuple[np.ndarray, int]:
    """Map rows to segments on the calibration grid.

    Segments 0 through K−1 correspond to occupied calibration cells. `calib["seg_ids"]`
    stores their sorted raw indices. A cell outside the list gets the nearest ID in cell
    order. Return the segment labels and the number of rows whose cells were empty
    during calibration.
    """
    if fit:
        assert isinstance(calib, MutableMapping)
        calib["seg_ids"] = np.unique(raw)
    ids = np.asarray(calib["seg_ids"], dtype=int)
    pos = np.searchsorted(ids, raw)
    hi = np.minimum(pos, ids.size - 1)
    lo = np.maximum(pos - 1, 0)
    nearest = np.where(np.abs(ids[hi] - raw) <= np.abs(ids[lo] - raw), hi, lo)
    unseen = int((ids[nearest] != raw).sum())
    return nearest.astype(int), unseen


def build_segments(
    x: np.ndarray,
    graph: SampledGraph,
    cfg: SegmentConfig,
    feats: list[int] | None = None,
    calib: dict | None = None,
) -> np.ndarray:
    """Return a segment label in 0..K-1 for each row, with K approximately cfg.n_segments.

    calib contains the SCM calibration boundaries and cell numbering. When it is None,
    calculate them from x, as before 0.6.0. This mode is used outside the SCM. feats can
    supply grid indices from segment_features. When it is None, choose indices without
    the restrictions for segment-specific tau.
    """
    if feats is None:
        feats = segment_features(graph, cfg)
    fit = calib is None
    if calib is None:
        calib = {}
    raw = raw_cells(x, graph, feats, grid_shape(graph, cfg, feats), calib, fit)
    seg, _ = compact_segments(raw, calib, fit)
    return seg
