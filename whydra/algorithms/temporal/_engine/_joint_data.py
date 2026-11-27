"""Independent environment-aware lag alignment for time-series discovery.

Logical nodes may expand into several physical columns, as with one-hot
context dummies. Every environment is aligned and masked separately before
samples are pooled, so lag windows never cross an environment boundary.
"""

import functools
import operator
from typing import cast

import numpy as np


class JointDataFrame:
    """Numeric observations and optional masks for one or more environments."""

    def __init__(
        self,
        data,
        *,
        mask=None,
        vector_vars=None,
        var_names=None,
        analysis_mode=None,
        missing_flag=None,
        remove_missing_upto_maxlag=False,
        data_type=None,
        time_offsets=None,
        **kwargs,
    ):
        if time_offsets is not None and any(time_offsets.values()):
            msg = "JointDataFrame requires environments with local row coordinates"
            raise ValueError(msg)
        if kwargs:
            raise TypeError("Unsupported JointDataFrame options: " + ", ".join(kwargs))
        if isinstance(data, dict):
            values = data
        else:
            block = np.asarray(data)
            values = dict(enumerate(block)) if block.ndim == 3 else {0: block}
        if not values:
            msg = "At least one environment is required"
            raise ValueError(msg)
        self.values = {
            key: np.asarray(block, dtype=float) for key, block in values.items()
        }
        if any(
            block.ndim != 2 or not block.shape[0] or not block.shape[1]
            for block in self.values.values()
        ):
            msg = "Every environment must be a nonempty observations-by-columns array"
            raise ValueError(msg)
        widths = {block.shape[1] for block in self.values.values()}
        if len(widths) != 1:
            msg = "All environments must have the same physical columns"
            raise ValueError(msg)
        self.Ndata = widths.pop()
        self.vector_vars = (
            {i: [(i, 0)] for i in range(self.Ndata)}
            if vector_vars is None
            else {
                int(i): [tuple(node) for node in nodes]
                for i, nodes in vector_vars.items()
            }
        )
        if set(self.vector_vars) != set(range(len(self.vector_vars))):
            msg = "Logical node indices must be consecutive integers starting at zero"
            raise ValueError(msg)
        self.N = len(self.vector_vars)
        for nodes in self.vector_vars.values():
            if not nodes:
                msg = "A logical node must contain at least one physical component"
                raise ValueError(msg)
            for col, lag in nodes:
                self._validate_node(col, lag, self.Ndata)
        self.var_names = list(range(self.N)) if var_names is None else list(var_names)
        if len(self.var_names) != self.N:
            msg = "var_names must describe the logical nodes"
            raise ValueError(msg)
        self.T = {key: block.shape[0] for key, block in self.values.items()}
        self.datasets = list(self.values)
        self.M = len(self.datasets)
        self.analysis_mode = analysis_mode or ("multiple" if self.M > 1 else "single")
        if self.analysis_mode not in ("single", "multiple"):
            msg = "analysis_mode must be 'single' or 'multiple'"
            raise ValueError(msg)
        if self.analysis_mode == "single" and self.M != 1:
            msg = "single mode requires one environment"
            raise ValueError(msg)
        self.mask = (
            self._blocks(mask, dtype=bool)
            if mask is not None
            else {
                key: np.zeros_like(block, dtype=bool)
                for key, block in self.values.items()
            }
        )
        self.data_type = (
            self._blocks(data_type, dtype=int) if data_type is not None else None
        )
        self.missing_flag = missing_flag
        self.remove_missing_upto_maxlag = bool(remove_missing_upto_maxlag)
        self.time_offsets = dict.fromkeys(self.values, 0)
        self.largest_time_step = max(self.T.values())
        self.use_indices_dataset_dict = {}

    @staticmethod
    def _validate_node(col, lag, width):
        if (
            not isinstance(col, (int, np.integer))
            or isinstance(col, (bool, np.bool_))
            or not 0 <= col < width
        ):
            msg = "Node column index is out of bounds"
            raise ValueError(msg)
        if (
            not isinstance(lag, (int, np.integer))
            or isinstance(lag, (bool, np.bool_))
            or lag > 0
        ):
            msg = "Node lag must be a nonpositive integer"
            raise ValueError(msg)

    def _blocks(self, value, dtype):
        if isinstance(value, dict):
            blocks = value
        else:
            array = np.asarray(value)
            if len(self.values) == 1:
                blocks = {next(iter(self.values)): array}
            elif array.ndim == 3:
                blocks = dict(zip(self.values, array, strict=False))
            else:
                msg = "Multiple environments require a dictionary or a 3D mask"
                raise ValueError(msg)
        if set(blocks) != set(self.values):
            msg = "Mask/type environments must match the observation environments"
            raise ValueError(msg)
        result = {key: np.asarray(blocks[key], dtype=dtype) for key in self.values}
        if any(result[key].shape != self.values[key].shape for key in self.values):
            msg = "Mask/type arrays must match their observation arrays"
            raise ValueError(msg)
        return result

    def _expand(self, nodes):
        expanded = []
        for col, lag in nodes or []:
            self._validate_node(col, lag, self.N)
            for physical_col, component_lag in self.vector_vars[int(col)]:
                node = (int(physical_col), int(lag + component_lag))
                if node not in expanded:
                    expanded.append(node)
        return expanded

    def construct_array(
        self,
        X,
        Y,
        Z=None,
        tau_max=0,
        *,
        extraZ=None,
        mask=None,
        mask_type=None,
        data_type=None,
        return_cleaned_xyz=False,
        do_checks=True,
        remove_overlaps=True,
        cut_off="2xtau_max",
        verbosity=0,
    ):
        """Return feature rows with independently aligned, pooled samples.

        Groups are marked 0 (X), 1 (Y), 2 (Z). Cleaned groups refer to physical
        columns after vector expansion. The last return item is an optional
        aligned type array, retained for callers that use the common CI API.
        """
        if not isinstance(tau_max, (int, np.integer)) or tau_max < 0:
            msg = "tau_max must be a nonnegative integer"
            raise ValueError(msg)
        if mask_type is not None and (
            not isinstance(mask_type, str) or set(mask_type) - set("xyz")
        ):
            msg = "mask_type must select x, y, and/or z"
            raise ValueError(msg)
        groups = [self._expand(X), self._expand(Y), self._expand(Z)]
        if not groups[0] or not groups[1]:
            msg = "X and Y must contain at least one node"
            raise ValueError(msg)
        if remove_overlaps:
            endpoints = set(groups[0] + groups[1])
            groups[2] = [node for node in groups[2] if node not in endpoints]
        additional = self._expand(extraZ)
        if remove_overlaps:
            occupied = set(functools.reduce(operator.iadd, groups, []))
            additional = [node for node in additional if node not in occupied]
        all_groups = groups + ([additional] if additional else [])
        nodes: list[tuple[int, int]] = functools.reduce(operator.iadd, all_groups, [])
        xyz = np.asarray(
            [group for group, members in enumerate(all_groups) for _ in members],
            dtype=int,
        )
        max_lag = max(-lag for _, lag in nodes)
        starts = {
            "2xtau_max": 2 * tau_max,
            "tau_max": tau_max,
            "max_lag": max_lag,
            "max_lag_or_tau_max": max(max_lag, cast("int", tau_max)),
            "2xtau_max_future": max_lag,
        }
        if cut_off not in starts:
            raise ValueError("Unknown lag cutoff: " + str(cut_off))
        start = int(starts[cut_off])
        if start < max_lag:
            msg = "The lag cutoff is smaller than a requested node lag"
            raise ValueError(msg)
        masks = self.mask if mask is None else self._blocks(mask, dtype=bool)
        types = (
            self.data_type if data_type is None else self._blocks(data_type, dtype=int)
        )
        samples, type_samples = [], []
        self.use_indices_dataset_dict = {}
        selected_groups = {
            index for index, name in enumerate("xyz") if mask_type and name in mask_type
        }
        for key, block in self.values.items():
            stop = len(block)
            if cut_off == "2xtau_max_future":
                stop -= max(0, 2 * tau_max - max_lag)
            times = np.arange(start, max(start, stop), dtype=int)
            aligned = np.asarray([block[times + lag, col] for col, lag in nodes])
            keep = np.isfinite(aligned).all(axis=0)
            if self.missing_flag is not None and not (
                isinstance(self.missing_flag, float) and np.isnan(self.missing_flag)
            ):
                keep &= (aligned != self.missing_flag).all(axis=0)
            if selected_groups:
                aligned_mask = np.asarray(
                    [masks[key][times + lag, col] for col, lag in nodes]
                )
                keep &= ~aligned_mask[np.isin(xyz, list(selected_groups))].any(axis=0)
            if self.remove_missing_upto_maxlag and len(times):
                bad_rows = ~np.isfinite(block).all(axis=1)
                if self.missing_flag is not None:
                    bad_rows |= (block == self.missing_flag).any(axis=1)
                contaminated = np.zeros(len(block), dtype=bool)
                for lag in range(start + 1):
                    contaminated[lag:] |= bad_rows[: len(block) - lag]
                keep &= ~contaminated[times]
            self.use_indices_dataset_dict[key] = times[keep]
            samples.append(aligned[:, keep])
            if types is not None:
                aligned_type = np.asarray(
                    [types[key][times + lag, col] for col, lag in nodes]
                )
                type_samples.append(aligned_type[:, keep])
        array = np.concatenate(samples, axis=1)
        if not array.shape[1]:
            msg = "No complete lag-aligned samples remain"
            raise ValueError(msg)
        type_array = np.concatenate(type_samples, axis=1) if types is not None else None
        if return_cleaned_xyz:
            cleaned = tuple(all_groups)
            return array, xyz, cleaned, type_array
        return array, xyz, type_array
