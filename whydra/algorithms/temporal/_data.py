"""Lag alignment and context preparation for temporal causal discovery."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...evaluation.graph_utils import default_node_names


@dataclass
class Dataset:
    name: str
    data: pd.DataFrame
    time: pd.Index | None = None


def _time_info(time):
    if time is None:
        return {"explicit": False, "regular": None, "unique": None, "increasing": None}
    unique = bool(time.is_unique)
    increasing = bool(time.is_monotonic_increasing)
    try:
        values = (
            time.asi8
            if isinstance(time, pd.DatetimeIndex)
            else np.asarray(time, dtype=float)
        )
        steps = np.diff(values)
        regular = bool(len(steps) == 0 or np.allclose(steps, steps[0]))
    except (TypeError, ValueError):
        regular = False
    return {
        "explicit": True,
        "regular": regular,
        "unique": unique,
        "increasing": increasing,
    }


def coerce_dataset(data, index=0):
    if isinstance(data, Dataset):
        result = deepcopy(data)
    elif isinstance(data, pd.DataFrame):
        result = Dataset(
            str(index),
            data.copy(),
            data.index.copy() if isinstance(data.index, pd.DatetimeIndex) else None,
        )
    else:
        array = np.asarray(data, dtype=float)
        if array.ndim != 2:
            msg = "Expected a 2D table or a Dataset"
            raise ValueError(msg)
        result = Dataset(
            str(index), pd.DataFrame(array, columns=default_node_names(array.shape[1]))
        )
    result.data.columns = result.data.columns.map(str)
    if not result.data.columns.is_unique or result.data.shape[1] < 1:
        msg = "Variable names must be unique and nonempty"
        raise ValueError(msg)
    result.data = result.data.astype(float)
    if np.isinf(result.data.to_numpy()).any():
        msg = "Infinite values are not supported; use NaN for missing samples"
        raise ValueError(msg)
    if len(result.data) < 4 or result.data.isna().all().any():
        msg = "Need at least four rows and an observation for each variable"
        raise ValueError(msg)
    return result


def validate_time(dataset):
    info = _time_info(dataset.time)
    if dataset.time is not None and (
        len(dataset.time) != len(dataset.data)
        or not all(info[k] for k in ["regular", "unique", "increasing"])
    ):
        msg = f"{dataset.name}: time must be unique, increasing and regularly spaced; resample explicitly first"
        raise ValueError(msg)


def make_frame(datasets, tau_max, *, vector_vars=None, names=None, joint=False):
    from ._engine._joint_data import JointDataFrame as DataFrame

    values, masks = {}, {}
    for i, dataset in enumerate(datasets):
        validate_time(dataset)
        if len(dataset.data) <= 2 * tau_max + 3:
            msg = f"{dataset.name}: too few rows for tau_max={tau_max}"
            raise ValueError(msg)
        a = dataset.data.to_numpy(dtype=float)
        masks[i] = ~np.isfinite(a)
        values[i] = np.where(masks[i], 0.0, a)
    var_names = names or list(datasets[0].data.columns)
    if len(datasets) == 1:
        return DataFrame(
            values[0], mask=masks[0], var_names=var_names, vector_vars=vector_vars
        )
    return DataFrame(
        values,
        mask=masks,
        analysis_mode="multiple",
        var_names=var_names,
        vector_vars=vector_vars,
    )


def prepare_joint(
    datasets,
    *,
    temporal_context=(),
    spatial_context=(),
    dataset_dummy=True,
    time_dummy=False,
):
    if not isinstance(datasets, (list, tuple)) or len(datasets) < 2:
        msg = "J-PCMCI+ requires a list of at least two separate datasets"
        raise ValueError(msg)
    datasets = [coerce_dataset(d, i) for i, d in enumerate(datasets)]
    names = list(datasets[0].data.columns)
    for d in datasets:
        if set(d.data.columns) != set(names):
            msg = "All environments must contain the same named variables"
            raise ValueError(msg)
        d.data = d.data.reindex(columns=names)
        validate_time(d)
    temporal_context, spatial_context = set(temporal_context), set(spatial_context)
    if temporal_context & spatial_context or (temporal_context | spatial_context) - set(
        names
    ):
        msg = "Context names must exist and temporal/spatial roles must be disjoint"
        raise ValueError(msg)
    system = set(names) - temporal_context - spatial_context
    if not system:
        msg = "At least one system variable is required"
        raise ValueError(msg)
    if temporal_context or time_dummy:
        first = datasets[0]
        for d in datasets[1:]:
            if len(d.data) != len(first.data):
                msg = (
                    "Temporal contexts/dummy require aligned, equal-length environments"
                )
                raise ValueError(msg)
            if (d.time is None) != (first.time is None) or (
                d.time is not None and not d.time.equals(first.time)
            ):
                msg = "Temporal contexts/dummy require identical time coordinates"
                raise ValueError(msg)
        for c in temporal_context:
            if any(
                not np.allclose(d.data[c], first.data[c], equal_nan=True)
                for d in datasets[1:]
            ):
                msg = f"Temporal context {c} must be shared across aligned environments"
                raise ValueError(msg)
    for c in spatial_context:
        if any(d.data[c].isna().any() or d.data[c].nunique() != 1 for d in datasets):
            msg = f"Spatial context {c} must be constant within every environment"
            raise ValueError(msg)
    for c in names:
        if (
            len(set(np.concatenate([d.data[c].dropna().to_numpy() for d in datasets])))
            <= 1
        ):
            msg = f"Globally constant variable: {c}"
            raise ValueError(msg)
        if c in system and any(d.data[c].nunique() <= 1 for d in datasets):
            msg = f"System variable {c} is constant within an environment; declare its context role if appropriate"
            raise ValueError(msg)
    roles = {
        i: (
            "time_context"
            if c in temporal_context
            else "space_context"
            if c in spatial_context
            else "system"
        )
        for i, c in enumerate(names)
    }
    vectors = {i: [(i, 0)] for i in range(len(names))}
    blocks = []
    physical = len(names)
    if time_dummy:
        blocks.append(("__time_dummy__", "time_dummy", len(datasets[0].data)))
    if dataset_dummy:
        blocks.append(("__dataset_dummy__", "space_dummy", len(datasets)))
    for name, role, width in blocks:
        if name in names:
            msg = f"Reserved dummy name in input: {name}"
            raise ValueError(msg)
        node = len(names)
        roles[node] = role
        names.append(name)
        vectors[node] = [(physical + i, 0) for i in range(width)]
        physical += width
    for i, d in enumerate(datasets):
        pieces = [d.data.to_numpy()]
        for _, role, width in blocks:
            if role == "time_dummy":
                pieces.append(np.eye(width))
            else:
                block = np.zeros((len(d.data), width))
                block[:, i] = 1.0
                pieces.append(block)
        d.data = pd.DataFrame(np.column_stack(pieces))
    return datasets, names, roles, vectors
