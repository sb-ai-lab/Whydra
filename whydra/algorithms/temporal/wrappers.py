"""Native PCMCI family exposed through the common Whydra algorithm contract."""

from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from numbers import Integral, Real
from typing import Any, cast

import numpy as np
import pandas as pd

from ...background_knowledge import BKPhase
from ..base import CausalAlgorithm
from ..graph_core import GeneralGraph
from ._bk import TemporalKnowledge
from ._ci import make_ci
from ._data import coerce_dataset, make_frame, prepare_joint
from ._engine.j_pcmci_plus import JPCMCIplus
from ._engine.pcmci import PCMCIPipeline
from ._engine.pcmci_plus import PCMCIPlusPipeline
from .graphs import lagged_node_names, temporal_to_general_graph


def _datasets(data, node_names):
    if isinstance(data, dict):
        blocks = list(data.values())
        labels = list(map(str, data))
    elif (isinstance(data, np.ndarray) and data.ndim == 3) or (
        isinstance(data, (list, tuple))
        and len(data)
        and (isinstance(data[0], pd.DataFrame) or np.ndim(data[0]) == 2)
    ):
        blocks, labels = list(data), None
    else:
        blocks, labels = [data], None
    if not blocks:
        msg = "At least one time series is required"
        raise ValueError(msg)
    datasets = [coerce_dataset(block, i) for i, block in enumerate(blocks)]
    if labels is not None:
        for dataset, label in zip(datasets, labels, strict=False):
            dataset.name = label
    names = list(datasets[0].data.columns)
    for dataset in datasets:
        if set(dataset.data.columns) != set(names):
            msg = "All environments must contain the same variables"
            raise ValueError(msg)
        dataset.data = dataset.data.reindex(columns=names)
    if node_names is not None:
        node_names = list(map(str, node_names))
        if (
            len(node_names) != len(names)
            or len(set(node_names)) != len(names)
            or any(not name for name in node_names)
        ):
            msg = "node_names must contain one unique, nonempty name per feature"
            raise ValueError(msg)
        names = node_names
        for dataset in datasets:
            dataset.data.columns = names
    return datasets, names


class _TemporalAlgorithm(CausalAlgorithm):
    """Shared adapter using analytic partial-correlation tests by default.

    ``run`` accepts an observations-by-features array or a DataFrame. J-PCMCI+
    also accepts (environments, time, features), a list, or a mapping of 2-D
    arrays/DataFrames. Lengths may differ when both temporal context and time
    dummies are disabled. Each environment is aligned separately.

    The default return value is a GeneralGraph over the input variables.
    ``lag_graph_``, ``p_matrix_`` and ``val_matrix_`` retain the full temporal
    tensors; ``lagged_graph_`` is a GeneralGraph with explicit lagged nodes.
    All three algorithms use joblib processes when ``n_jobs`` differs from 1;
    graph orientation and global evidence correction remain sequential.
    """

    algorithm: str | None = None
    default_tau_min = 0

    def __init__(
        self,
        alpha=0.05,
        indep_test="parcorr",
        n_jobs=1,
        verbose=False,
        show_progress=False,
        background_knowledge=None,
        *,
        tau_min=None,
        tau_max=1,
        pc_alpha=None,
        node_names=None,
        graph_mode="summary",
        **kwargs,
    ):
        self.alpha = alpha
        self.indep_test = indep_test
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.show_progress = show_progress
        self.background_knowledge = background_knowledge
        self.tau_min = self.default_tau_min if tau_min is None else tau_min
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.node_names = node_names
        self.graph_mode = graph_mode
        self.kwargs = dict(kwargs)

    def _options(self, overrides):
        keys = (
            "alpha",
            "indep_test",
            "n_jobs",
            "verbose",
            "show_progress",
            "background_knowledge",
            "tau_min",
            "tau_max",
            "pc_alpha",
            "node_names",
            "graph_mode",
        )
        params = {key: getattr(self, key) for key in keys}
        params.update(self.kwargs)
        params.update(overrides)
        PCMCIPipeline._check_tau_limits(params["tau_min"], params["tau_max"])
        alpha = params["alpha"]
        if (
            isinstance(alpha, (bool, np.bool_))
            or not isinstance(alpha, Real)
            or not 0 < cast("float", alpha) < 1
        ):
            msg = "alpha must be between zero and one"
            raise ValueError(msg)
        if params["pc_alpha"] is None:
            params["pc_alpha"] = 0.2 if self.algorithm == "pcmci" else alpha
        pc_alpha = params["pc_alpha"]
        if (
            isinstance(pc_alpha, (bool, np.bool_))
            or not isinstance(pc_alpha, Real)
            or not 0 < cast("float", pc_alpha) < 1
        ):
            msg = "pc_alpha must be a scalar between zero and one"
            raise ValueError(msg)
        if self.algorithm != "pcmci" and pc_alpha != alpha:
            msg = "PCMCI+ and J-PCMCI+ use one alpha; omit pc_alpha or set it equal to alpha"
            raise ValueError(msg)
        jobs = params["n_jobs"]
        if (
            isinstance(jobs, (bool, np.bool_))
            or not isinstance(jobs, Integral)
            or (jobs != -1 and jobs < 1)
        ):
            msg = "n_jobs must be a positive integer or -1"
            raise ValueError(msg)
        if params["graph_mode"] not in {"summary", "lagged"}:
            msg = "graph_mode must be 'summary' or 'lagged'"
            raise ValueError(msg)
        return params

    def run(self, data: np.ndarray | pd.DataFrame, **kwargs) -> GeneralGraph:
        params = self._options(kwargs)
        datasets, names = _datasets(data, params.pop("node_names"))
        observed_count = len(names)
        joint = self.algorithm == "j_pcmci_plus"
        roles = dict.fromkeys(range(len(names)), "system")
        vectors = None
        if not joint and len(datasets) != 1:
            msg = "PCMCI and PCMCI+ expect one 2D series; use J-PCMCI+ for multiple environments"
            raise ValueError(msg)
        if joint:
            if params["tau_min"] != 0:
                msg = "J-PCMCI+ requires tau_min=0"
                raise ValueError(msg)
            context_options = {
                key: params.pop(key, default)
                for key, default in (
                    ("temporal_context", ()),
                    ("spatial_context", ()),
                    ("dataset_dummy", True),
                    ("time_dummy", False),
                )
            }
            if len(datasets) > 1:
                datasets, names, roles, vectors = prepare_joint(
                    datasets, **context_options
                )
            elif (
                context_options["temporal_context"]
                or context_options["spatial_context"]
                or context_options["time_dummy"]
            ):
                msg = (
                    "Joint contexts and time dummies require at least two environments"
                )
                raise ValueError(msg)
            # With one 2-D series, use the context-free joint controller. A
            # dataset dummy would be constant and carries no information.
        if vectors is None and any(
            (dataset.data.nunique() <= 1).any() for dataset in datasets
        ):
            msg = "Constant system variables cannot be tested"
            raise ValueError(msg)
        lagged_node_names(names, params["tau_max"])
        frame = make_frame(
            datasets, params["tau_max"], vector_vars=vectors, names=names
        )
        ci = make_ci(
            params["indep_test"],
            multivariate=any(role.endswith("dummy") for role in roles.values()),
        )
        engine_kwargs: dict[str, Any] = {
            "dataframe": frame,
            "cond_ind_test": ci,
            "verbosity": int(params["verbose"]),
        }
        engine: JPCMCIplus | PCMCIPipeline | PCMCIPlusPipeline
        run: Callable[..., Any]
        if joint:
            if params["n_jobs"] == 1:
                engine = JPCMCIplus(node_classification=roles, **engine_kwargs)
            else:
                from ._engine.parallel import ParallelJPCMCIplus

                engine = ParallelJPCMCIplus(
                    node_classification=roles, n_jobs=params["n_jobs"], **engine_kwargs
                )
            run = engine.run_jpcmciplus
        elif self.algorithm == "pcmci":
            if params["n_jobs"] == 1:
                engine = PCMCIPipeline(**engine_kwargs)
            else:
                from ._engine.parallel import ParallelPCMCIPipeline

                engine = ParallelPCMCIPipeline(n_jobs=params["n_jobs"], **engine_kwargs)
            run = engine.run_pcmci
        else:
            if params["n_jobs"] == 1:
                engine = PCMCIPlusPipeline(**engine_kwargs)
            else:
                from ._engine.parallel import ParallelPCMCIPlusPipeline

                engine = ParallelPCMCIPlusPipeline(
                    n_jobs=params["n_jobs"], **engine_kwargs
                )
            run = engine.run_pcmciplus
        self.engine_ = engine
        knowledge = params.pop("background_knowledge")
        temporal_bk = (
            TemporalKnowledge(
                knowledge,
                names,
                params["tau_min"],
                params["tau_max"],
                roles=roles if joint else None,
            )
            if knowledge is not None
            else None
        )
        links = params.pop("link_assumptions", None)
        include_dummies = params.pop("include_context_dummies", False)
        run_options = {
            key: params.pop(key) for key in ("tau_min", "tau_max", "pc_alpha")
        }
        if self.algorithm == "pcmci":
            run_options["alpha_level"] = params["alpha"]
            allowed = {
                "max_conds_dim",
                "max_combinations",
                "max_conds_py",
                "max_conds_px",
                "fdr_method",
                "save_iterations",
            }
        else:
            run_options["contemp_collider_rule"] = params.pop(
                "collider_rule", "majority"
            )
            allowed = {
                "max_conds_dim",
                "max_combinations",
                "max_conds_py",
                "max_conds_px",
                "max_conds_px_lagged",
                "fdr_method",
                "reset_lagged_links",
                "conflict_resolution",
            }
        common = {
            "alpha",
            "indep_test",
            "n_jobs",
            "verbose",
            "show_progress",
            "graph_mode",
        }
        unknown = set(params) - allowed - common
        if unknown:
            raise TypeError(
                "Unsupported temporal algorithm parameters: "
                + ", ".join(sorted(unknown))
            )
        run_options.update(
            {key: value for key, value in params.items() if key in allowed}
        )
        with knowledge.run_scope() if knowledge is not None else nullcontext():
            if temporal_bk is not None:
                engine._background_knowledge = temporal_bk
                links = temporal_bk.assumptions(engine, links)
            raw = run(link_assumptions=links, **run_options)
            if temporal_bk is not None:
                raw["graph"] = temporal_bk.before_orientation(raw["graph"])
                raw["graph"] = temporal_bk.checkpoint(
                    raw["graph"], BKPhase.POST_ORIENTATION
                )
        if (
            not np.isfinite(raw["p_matrix"]).all()
            or not np.isfinite(raw["val_matrix"]).all()
        ):
            msg = "Undefined CI result: insufficient or degenerate aligned samples"
            raise ValueError(msg)
        self.results_ = deepcopy(raw)
        self.lag_graph_ = self.results_["graph"]
        self.p_matrix_ = self.results_["p_matrix"] = np.minimum(raw["p_matrix"], 1.0)
        self.val_matrix_ = self.results_["val_matrix"]
        self.sepsets_ = self.results_.get("sepsets", {})
        self.parents_ = deepcopy(
            getattr(engine, "all_lagged_parents", getattr(engine, "all_parents", {}))
        )
        self.node_names_ = tuple(names)
        self.node_roles_ = dict(roles)
        self.observed_context_parents_ = deepcopy(
            getattr(engine, "observed_context_parents", {})
        )
        self.dummy_parents_ = deepcopy(getattr(engine, "dummy_parents", {}))
        self.lagged_graph_ = temporal_to_general_graph(
            self.lag_graph_, names, graph_mode="lagged"
        )
        count = len(names) if include_dummies else observed_count
        self.graph_ = temporal_to_general_graph(
            self.lag_graph_[:count, :count],
            names[:count],
            graph_mode=params["graph_mode"],
        )
        return self.graph_


class StandalonePCMCI(_TemporalAlgorithm):
    """Parallel PC1 parent selection and MCI, with separate pc_alpha/alpha."""

    algorithm = "pcmci"
    default_tau_min = 1


class StandalonePCMCIPlus(_TemporalAlgorithm):
    """Parallel PCMCI+ search with contemporaneous colliders and temporal Meek rules."""

    algorithm = "pcmci_plus"


class StandaloneJPCMCIPlus(_TemporalAlgorithm):
    """J-PCMCI+ with observed contexts, vector dummies and joblib parallel search."""

    algorithm = "j_pcmci_plus"


PCMCI = StandalonePCMCI
PCMCIPlus = StandalonePCMCIPlus
JPCMCIPlus = StandaloneJPCMCIPlus

__all__ = [
    "PCMCI",
    "JPCMCIPlus",
    "PCMCIPlus",
    "StandaloneJPCMCIPlus",
    "StandalonePCMCI",
    "StandalonePCMCIPlus",
]
