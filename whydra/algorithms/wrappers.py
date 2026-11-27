"""Public algorithm wrappers and the shared parameter-resolution facade."""

import inspect
import os
import pickle
import sys
import warnings
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, TypeGuard

import numpy as np

from ..background_knowledge import MatrixEncoding
from ..evaluation.graph_utils import default_node_names, pag_matrix_to_general_graph
from .base import CausalAlgorithm
from .graph_core import (
    CIT,
    CausalGraph,
    Edge,
    Endpoint,
    GeneralGraph,
    _optional_imports,
)
from .graph_core import Node as GraphNode
from .graph_core.conditioning import (
    validate_cuda_background_knowledge,
    validate_max_cond_set_size,
)
from .graph_core.validation import (
    effective_n_jobs,
    validate_alpha,
    validate_data,
    validate_node_names,
)

if TYPE_CHECKING:
    from .temporal import StandaloneJPCMCIPlus, StandalonePCMCI, StandalonePCMCIPlus


def _is_dataframe(value) -> TypeGuard["_optional_imports.DataFrame"]:
    # Creating a real DataFrame (including a subclass) already imports pandas.
    # NumPy-only callers must not pay for importing an unrelated tabular runtime.
    pandas = sys.modules.get("pandas")
    return pandas is not None and isinstance(value, getattr(pandas, "DataFrame", ()))


# Some standalone modules import their siblings by unqualified name.
current_dir = os.path.dirname(os.path.abspath(__file__))
standalone_dir = os.path.join(current_dir, "standalone")
if standalone_dir not in sys.path:
    sys.path.append(standalone_dir)

_ALGORITHM_MODULES = frozenset(
    {
        "pc_algo",
        "parallel_pc_stable",
        "fci_algo_v2",
        "fci_algo_cash",
        "fci_algo_paralel_cash",
        "cfci_algo",
        "cfci_algo_paralel_cash",
        "fci_plus_algo",
        "fci_plus_algo_paralel_cash",
        "rai_algo",
        "rfci_algo",
        "gfci_algo",
    }
)


_TEMPORAL_ALGORITHM_ALIASES = {
    "StandalonePCMCI": ("pcmci", "pcmci_stable"),
    "StandalonePCMCIPlus": ("pcmci_plus", "pcmci+", "pcmciplus"),
    "StandaloneJPCMCIPlus": (
        "j_pcmci_plus",
        "j_pcmci+",
        "jpcmciplus",
        "native_jpcmci_plus",
    ),
}


def _algorithm_module(name):
    module = globals().get(name)
    if module is None:
        module = import_module(f".standalone.{name}", __package__)
        globals()[name] = module
    return module


def _temporal_algorithm_class(name):
    cls = getattr(import_module(".temporal", __package__), name)
    globals()[name] = cls
    return cls


def __getattr__(name):
    # Retain direct module access used by integrations and monkeypatching,
    # while a normal wrapper import loads no algorithm implementations.
    if name in _ALGORITHM_MODULES:
        return _algorithm_module(name)
    if name in _TEMPORAL_ALGORITHM_ALIASES:
        return _temporal_algorithm_class(name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__():
    return sorted(
        set(globals()) | _ALGORITHM_MODULES | _TEMPORAL_ALGORITHM_ALIASES.keys()
    )


def _publish_execution_metadata(graph, cg, cit, jobs):
    """Expose observed dispatch separately from the configured worker count."""
    from copy import deepcopy

    from .graph_core import ci_batch

    snapshot = getattr(ci_batch, "batch_execution_stats", None)
    numeric = snapshot(cit) if snapshot is not None else {}
    graph.metadata["ci_batch_execution"] = numeric
    actual = set()
    for backend, count in numeric.get("batches", {}).items():
        if count:
            actual.add(backend)
    coarse = getattr(cg, "pc_execution_stats", None)
    if coarse is not None:
        graph.metadata["pc_execution"] = deepcopy(coarse)
        actual.update(
            backend for backend, count in coarse["coarse_batches"].items() if count
        )
    skeleton = getattr(cg, "fci_skeleton_execution_stats", None)
    if skeleton is not None:
        graph.metadata["fci_skeleton_execution"] = deepcopy(skeleton)
        actual.update(
            backend for backend, count in skeleton["coarse_batches"].items() if count
        )
    stats = getattr(cg, "ci_stats", {})
    policy = []
    for source in (numeric, coarse, skeleton, stats):
        if source:
            policy.extend(deepcopy(source.get("process_policy", ())))
    graph.metadata["process_policy"] = policy[-128:]
    if stats.get("pdsep_process_pairs", 0) or stats.get("pdsep_process_rounds", 0):
        actual.add("processes")
    if stats.get("pdsep_thread_rounds", 0):
        actual.add("threads")
    for stage in ("augment", "dsep", "triple"):
        if stats.get(f"{stage}_process_batches", 0) or stats.get(
            f"{stage}_process_tasks", 0
        ):
            actual.add("processes")
        if stats.get(f"{stage}_thread_batches", 0):
            actual.add("threads")
    parallel = actual - {"serial"}
    graph.metadata["execution"] = {
        **graph.metadata.get("execution", {}),
        "scope": "observed_ci_dispatch",
        "configured_n_jobs": jobs,
        "actual_backend": (
            "mixed" if len(parallel) > 1 else next(iter(parallel), "serial")
        ),
        "actual_backends": sorted(actual or {"serial"}),
    }


def _run_causal(
    algorithm, data, run_kwargs, kernel, *, graph_kind, pc=False, rai=False
):
    """Resolve all public parameters once, before any graph or CI search."""
    defaults = {
        name: getattr(algorithm, name)
        for name in (
            "alpha",
            "indep_test",
            "n_jobs",
            "verbose",
            "maj_rule",
            "score_func",
            "use_max_p",
            "depth",
            "start_from_complete_graph",
            "use_ci_cache",
            "k",
            "device",
        )
        if hasattr(algorithm, name)
    }
    params = {**defaults, **algorithm.kwargs, **run_kwargs}
    device = params.get("device")
    cuda = device is not None and str(device).startswith("cuda")
    if cuda:
        validate_cuda_background_knowledge(
            params.get("background_knowledge"), type(algorithm).__name__
        )
    if device is not None and not cuda:
        # ``device="cpu"`` is the default backend, exactly like ``device=None``.
        params["device"] = None
    if "n_procs" in params:
        warnings.warn(
            "n_procs is deprecated; use n_jobs", DeprecationWarning, stacklevel=3
        )
        params["n_jobs"] = params.pop("n_procs")
    signature = inspect.signature(kernel).parameters
    supported = {
        name
        for name, p in signature.items()
        if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }
    supported |= {
        "indep_test",
        "node_names",
        "strict_degenerate",
        "data_type",
        "ci_cache_size",
    }
    if rai:
        supported |= {"n_jobs", "verbose", "show_progress", "background_knowledge"}
    for optional in ("device", "dtype"):
        if params.get(optional) is None:
            params.pop(optional, None)
    unknown = set(params) - supported
    if unknown:
        msg = f"{type(algorithm).__name__}: unsupported parameter(s): {', '.join(sorted(unknown))}"
        raise TypeError(msg)
    alpha = validate_alpha(params.get("alpha", 0.05))
    limit = validate_max_cond_set_size(params.get("max_cond_set_size"))
    params["max_cond_set_size"] = limit
    effective_limit = limit
    if "depth" in signature and params.get("depth", -1) >= 0:
        effective_limit = (
            params["depth"] if limit is None else min(limit, params["depth"])
        )
    params["alpha"] = alpha
    names = params.pop("node_names", None)
    if names is None:
        names = (
            list(data.columns)
            if _is_dataframe(data)
            else default_node_names(
                np.asarray(data).shape[1] if np.asarray(data).ndim == 2 else 0
            )
        )
    method = params.pop("indep_test", "fisherz")
    data = validate_data(data, numeric=method == "fisherz")
    names = validate_node_names(names, data.shape[1])
    cit = CIT(
        data,
        method=method,
        max_cond_set_size=limit,
        strict_degenerate=params.pop("strict_degenerate", False),
        data_type=params.pop("data_type", None),
        cache_size=params.pop(
            "ci_cache_size", 100000 if params.get("use_ci_cache", True) else 0
        ),
    )
    # The same CIT can be hidden behind cached/legacy algorithm closures.
    # Preserve reference decisions near alpha at that shared numeric boundary.
    cit._decision_alpha = alpha
    from .graph_core.ci_batch import configure_fisher_alpha

    configure_fisher_alpha(cit, alpha)
    jobs = 1 if cuda else effective_n_jobs(params.get("n_jobs", 1))
    if "n_jobs" in signature:
        params["n_jobs"] = jobs
    if params.get("stable") is False and jobs != 1:
        msg = "stable=False requires n_jobs=1"
        raise ValueError(msg)
    if params.get("uc_rule", 0) != 0 or params.get("uc_priority", 2) != 2:
        msg = "Only uc_rule=0 and uc_priority=2 are implemented"
        raise ValueError(msg)
    if "show_progress" in signature:
        params.setdefault("show_progress", False)
    cg = None
    if rai:
        if params.pop("background_knowledge", None) is not None:
            msg = "RAI does not support background_knowledge"
            raise NotImplementedError(msg)
        if params.pop("show_progress", False):
            msg = "RAI does not support show_progress=True"
            raise NotImplementedError(msg)
        warnings.warn(
            "RAI is experimental. certify_skeleton=True enables optional stable adjacency refinement, changing statistical behavior and cost",
            RuntimeWarning,
            stacklevel=3,
        )
        params = {k: v for k, v in params.items() if k in signature}
        result = kernel(
            data, indep_test=method, node_names=names, ci_test=cit, **params
        )
        graph = result.G if hasattr(result, "G") else result
    else:
        bk = params.get("background_knowledge")
        if bk is not None and graph_kind == "pag":
            from ..background_knowledge import BKPhase

            # PRE_SEARCH is a checkpoint, not a persistent direction constraint.
            # Reject a one-off PAG prohibition rather than silently erasing it
            # when the skeleton is reset to circles.
            pre = bk._directions_at_phase("forbidden", BKPhase.PRE_SEARCH)
            later = bk._directions_at_phase(
                "forbidden", BKPhase.BETWEEN_ORIENTATION
            ) | bk._directions_at_phase("forbidden", BKPhase.POST_ORIENTATION)
            if pre - later:
                msg = "PAG PRE_SEARCH direction prohibitions must also be enabled during or after orientation"
                raise ValueError(msg)
        cg = CausalGraph(data.shape[1], names, device=device)
        cg.set_ind_test(cit)
        if not pc:
            params["indep_test"] = method
        if "node_names" in signature:
            params["node_names"] = names
        if pc and cuda:
            params["_return_local_graph"] = True
        result = kernel(cg, **params) if pc else kernel(cg, data, **params)
        graph = result.G if hasattr(result, "G") else result
        if not pc:
            graph = pag_matrix_to_general_graph(
                graph, names, encoding=algorithm.OUTPUT_ENCODING
            )
    graph.graph_kind = graph_kind
    graph.metadata = {
        **getattr(graph, "metadata", {}),
        "graph_kind": graph_kind,
        "effective_n_jobs": jobs,
        "alpha": alpha,
        "indep_test": method,
        "max_cond_set_size": limit,
        "effective_max_cond_set_size": effective_limit,
        "depth_limited": effective_limit is not None,
        "experimental": rai,
        "ci_cache_hits": cit.cache_hits,
        "ci_computations": cit.cache_misses,
    }
    if cg is not None and hasattr(cg, "ci_stats"):
        graph.metadata["ci_stats"] = cg.ci_stats
    if cg is not None and hasattr(cg, "execution_stats"):
        # Preserve the per-pass dispatch policy after PAG matrix conversion.
        # effective_n_jobs above retains its existing normalized-option meaning.
        graph.metadata["execution"] = cg.execution_stats
    from ..background_knowledge import find_ancestral_violation, find_directed_cycle

    violation = (
        find_ancestral_violation(graph.graph)
        if graph_kind == "pag"
        else find_directed_cycle(graph.graph, -1, 1)
    )
    if violation is not None and graph_kind == "pag":
        # Return the PAG so callers can inspect the ancestry violation.
        graph.metadata["ancestral_violation"] = violation
        warnings.warn(
            f"{type(algorithm).__name__} produced an invalid {graph_kind}: {violation}",
            RuntimeWarning,
            stacklevel=3,
        )
    elif violation is not None:
        msg = (
            f"{type(algorithm).__name__} produced an invalid {graph_kind}: {violation}"
        )
        raise ValueError(msg)
    if cg is not None:
        _publish_execution_metadata(graph, cg, cit, jobs)
    return graph


class _PagMatrixResult:
    """Turns an algorithm's endpoint matrix into a local graph.

    Each wrapper declares the encoding of its result in ``OUTPUT_ENCODING``.
    The two encodings assign different numbers to arrowheads and circles.
    """

    #: Endpoint encoding this algorithm's matrix comes back in.
    OUTPUT_ENCODING: MatrixEncoding | None = None

    def _matrix_to_graph(self, matrix, n_nodes):
        if self.OUTPUT_ENCODING is None:
            msg = f"{type(self).__name__} must declare OUTPUT_ENCODING"
            raise NotImplementedError(msg)
        return pag_matrix_to_general_graph(
            matrix, default_node_names(n_nodes), encoding=self.OUTPUT_ENCODING
        )


class StandalonePCStable(CausalAlgorithm):
    """Run PC-stable: use CI tests to find the skeleton, orient colliders, then apply Meek's rules.

    The sequential implementation (``pc_algo``) returns a CPDAG.

    ``device="cuda"`` selects PC-stable-cuda and sets ``n_jobs`` to 1.
    """

    def __init__(
        self, alpha=0.05, indep_test="fisherz", n_jobs=1, device=None, **kwargs
    ):
        self.alpha = alpha
        self.indep_test = indep_test
        self.n_jobs = n_jobs
        self.device = device
        self.kwargs = kwargs

    def run(
        self, data: "np.ndarray | _optional_imports.DataFrame", **kwargs
    ) -> GeneralGraph:
        """Run the configured algorithm on ``data`` and return its graph."""
        return _run_causal(
            self,
            data,
            kwargs,
            _algorithm_module("pc_algo").pc_stable,
            graph_kind="cpdag",
            pc=True,
            rai=False,
        )


class ParallelPCStable(StandalonePCStable):
    """Run PC-stable with parallel skeleton search (``parallel_pc_stable``).

    With the same data and background knowledge, the result must match
    the CPDAG from :class:`StandalonePCStable`.

    ``device="cuda"`` selects PC-stable-cuda (``pc_algo.pc_stable`` with fused
    skeleton search, collider orientation and Meek rules). It sets ``n_jobs``
    to 1 and is expected to produce the same CPDAG as the CPU path.
    """

    def run(
        self, data: "np.ndarray | _optional_imports.DataFrame", **kwargs
    ) -> GeneralGraph:
        """Run the configured algorithm on ``data`` and return its graph."""
        device = kwargs.get("device", getattr(self, "device", None))
        cuda = device is not None and str(device).startswith("cuda")
        kernel = (
            _algorithm_module("pc_algo").pc_stable
            if cuda
            else _algorithm_module("parallel_pc_stable").pc_stable
        )
        return _run_causal(
            self,
            data,
            kwargs,
            kernel,
            graph_kind="cpdag",
            pc=True,
            rai=False,
        )


class StandaloneFCIStable(_PagMatrixResult, CausalAlgorithm):
    """Run FCI with edge-based Possible-D-SEP and R0–R10 orientation.

    FCI finds a skeleton, tests additional separating sets with Possible-D-SEP,
    then orients the PAG. It allows latent variables. The ``fci_algo_v2`` backend
    implements the orientation rules, including R1, R2, R5 and R10.
    ``m_max`` bounds the Possible-D-SEP phase. ``max_cond_set_size`` bounds
    conditioning sets in every CI phase. Other keyword arguments are passed
    to ``fci_stable_v2``.

    ``device="cuda"`` selects FCI-stable-cuda (``n_jobs=1``).
    """

    OUTPUT_ENCODING = MatrixEncoding.STANDARD

    def __init__(
        self,
        alpha=0.05,
        indep_test="fisherz",
        n_jobs=1,
        verbose=False,
        device=None,
        **kwargs,
    ):
        self.alpha = alpha
        self.indep_test = indep_test
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.device = device
        self.kwargs = kwargs

    def run(
        self, data: "np.ndarray | _optional_imports.DataFrame", **kwargs
    ) -> GeneralGraph:
        """Run the configured algorithm on ``data`` and return its graph."""
        return _run_causal(
            self,
            data,
            kwargs,
            _algorithm_module("fci_algo_v2").fci_stable_v2,
            graph_kind="pag",
            pc=False,
            rai=False,
        )


class StandaloneFCIStableCached(StandaloneFCIStable):
    """Run FCI with cached CI tests (``fci_algo_cash``).

    Caching changes execution time while preserving the result of
    :class:`StandaloneFCIStable`.
    """

    def run(
        self, data: "np.ndarray | _optional_imports.DataFrame", **kwargs
    ) -> GeneralGraph:
        """Run the configured algorithm on ``data`` and return its graph."""
        return _run_causal(
            self,
            data,
            kwargs,
            _algorithm_module("fci_algo_cash").fci_stable,
            graph_kind="pag",
            pc=False,
            rai=False,
        )


class StandaloneFCIStableParallelCached(StandaloneFCIStable):
    """Run FCI with caching and parallel execution (``fci_algo_paralel_cash``).

    ``n_jobs`` sets the number of threads or processes. The result matches
    :class:`StandaloneFCIStable`.
    """

    def run(
        self, data: "np.ndarray | _optional_imports.DataFrame", **kwargs
    ) -> GeneralGraph:
        """Run the configured algorithm on ``data`` and return its graph."""
        return _run_causal(
            self,
            data,
            kwargs,
            _algorithm_module("fci_algo_paralel_cash").fci_stable,
            graph_kind="pag",
            pc=False,
            rai=False,
        )


class StandaloneCFCIStable(StandaloneFCIStable):
    """Run conservative FCI, marking ambiguous triples and leaving their collider status unresolved.

    This trades fewer false arrowheads for more circle endpoints
    in the PAG.
    """

    def __init__(
        self,
        alpha=0.05,
        indep_test="fisherz",
        n_jobs=1,
        verbose=False,
        maj_rule=False,
        **kwargs,
    ):
        super().__init__(
            alpha=alpha, indep_test=indep_test, n_jobs=n_jobs, verbose=verbose, **kwargs
        )
        self.maj_rule = maj_rule

    def run(
        self, data: "np.ndarray | _optional_imports.DataFrame", **kwargs
    ) -> GeneralGraph:
        """Run the configured algorithm on ``data`` and return its graph."""
        return _run_causal(
            self,
            data,
            kwargs,
            _algorithm_module("cfci_algo").cfci_stable,
            graph_kind="pag",
            pc=False,
            rai=False,
        )


class StandaloneCFCIStableParallelCached(StandaloneCFCIStable):
    """Run conservative FCI with caching and parallel execution.

    The result matches :class:`StandaloneCFCIStable`.
    """

    def run(
        self, data: "np.ndarray | _optional_imports.DataFrame", **kwargs
    ) -> GeneralGraph:
        """Run the configured algorithm on ``data`` and return its graph."""
        return _run_causal(
            self,
            data,
            kwargs,
            _algorithm_module("cfci_algo_paralel_cash").cfci_stable,
            graph_kind="pag",
            pc=False,
            rai=False,
        )


class StandaloneRFCIStable(_PagMatrixResult, CausalAlgorithm):
    """Run RFCI, which tests discriminating paths in place of a full Possible-D-SEP search.

    On large graphs, RFCI is much cheaper than FCI, but its PAG may contain less orientation information.
    """

    OUTPUT_ENCODING = MatrixEncoding.STANDARD

    def __init__(
        self, alpha=0.05, indep_test="fisherz", n_jobs=1, verbose=False, **kwargs
    ):
        self.alpha = alpha
        self.indep_test = indep_test
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.kwargs = kwargs

    def run(
        self, data: "np.ndarray | _optional_imports.DataFrame", **kwargs
    ) -> GeneralGraph:
        """Run the configured algorithm on ``data`` and return its graph."""
        return _run_causal(
            self,
            data,
            kwargs,
            _algorithm_module("rfci_algo").rfci_stable,
            graph_kind="pag",
            pc=False,
            rai=False,
        )


class StandaloneGFCIStable(StandaloneRFCIStable):
    """Run GFCI with an initial CPDAG from score-based GES, then apply FCI rules.

    ``start_from_complete_graph=True`` replaces the GES result with a complete o-o graph.
    """

    def __init__(
        self,
        alpha=0.05,
        indep_test="fisherz",
        n_jobs=1,
        verbose=False,
        score_func="auto",
        use_max_p=True,
        depth=-1,
        start_from_complete_graph=False,
        use_ci_cache=True,
        **kwargs,
    ):
        super().__init__(
            alpha=alpha,
            indep_test=indep_test,
            n_jobs=n_jobs,
            verbose=verbose,
            **kwargs,
        )
        self.score_func = score_func
        self.use_max_p = use_max_p
        self.depth = depth
        self.start_from_complete_graph = start_from_complete_graph
        self.use_ci_cache = use_ci_cache

    def run(
        self, data: "np.ndarray | _optional_imports.DataFrame", **kwargs
    ) -> GeneralGraph:
        """Run the configured algorithm on ``data`` and return its graph."""
        return _run_causal(
            self,
            data,
            kwargs,
            _algorithm_module("gfci_algo").gfci_stable,
            graph_kind="pag",
            pc=False,
            rai=False,
        )


class StandaloneFCIPlusStable(StandaloneRFCIStable):
    """Build the FCI+ skeleton using graph augmentation.

    FCI+ searches for the skeleton in order of increasing conditioning set size.
    Graph augmentation reduces the number of CI tests on sparse graphs.
    """

    def __init__(
        self,
        alpha=0.05,
        indep_test="fisherz",
        n_jobs=1,
        verbose=False,
        k=None,
        **kwargs,
    ):
        super().__init__(
            alpha=alpha, indep_test=indep_test, n_jobs=n_jobs, verbose=verbose, **kwargs
        )
        self.k = k

    def run(
        self, data: "np.ndarray | _optional_imports.DataFrame", **kwargs
    ) -> GeneralGraph:
        """Run the configured algorithm on ``data`` and return its graph."""
        return _run_causal(
            self,
            data,
            kwargs,
            _algorithm_module("fci_plus_algo").fci_plus_stable,
            graph_kind="pag",
            pc=False,
            rai=False,
        )


class StandaloneFCIPlusStableParallelCached(StandaloneFCIPlusStable):
    """Run FCI+ with caching and parallel execution.

    The result matches
    :class:`StandaloneFCIPlusStable`.
    """

    def run(
        self, data: "np.ndarray | _optional_imports.DataFrame", **kwargs
    ) -> GeneralGraph:
        """Run the configured algorithm on ``data`` and return its graph."""
        return _run_causal(
            self,
            data,
            kwargs,
            _algorithm_module("fci_plus_algo_paralel_cash").fci_plus_stable,
            graph_kind="pag",
            pc=False,
            rai=False,
        )


class StandaloneRAIStable(CausalAlgorithm):
    """Run the reference implementation of RAI (recursive autonomy identification).

    Background knowledge is unsupported.

    ``device="cuda"`` selects RAI-stable-cuda and sets ``n_jobs`` to 1.
    """

    def __init__(
        self,
        alpha=0.05,
        indep_test="fisherz",
        n_jobs=1,
        verbose=False,
        device=None,
        **kwargs,
    ):
        self.alpha = alpha
        self.indep_test = indep_test
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.verbose = verbose
        self.device = device

    def run(
        self, data: "np.ndarray | _optional_imports.DataFrame", **kwargs
    ) -> GeneralGraph:
        """Run the configured algorithm on ``data`` and return its graph."""
        return _run_causal(
            self,
            data,
            kwargs,
            _algorithm_module("rai_algo").rai_stable,
            graph_kind="cpdag",
            pc=False,
            rai=True,
        )


dagnn_runner = None
DAG_GNN_IMPORT_ERROR = None


def _load_dagnn():
    global dagnn_runner, DAG_GNN_IMPORT_ERROR
    if dagnn_runner is None and DAG_GNN_IMPORT_ERROR is None:
        try:
            from .standalone.dag_gnn import runner
        except ModuleNotFoundError as exc:
            DAG_GNN_IMPORT_ERROR = exc
        else:
            dagnn_runner = runner


class DagGNN(CausalAlgorithm):
    """DAG-GNN wrapper (requires PyTorch)."""

    _RUNNER_PARAMETERS = frozenset(
        {
            "lambda_A",
            "c_A",
            "h_tol",
            "tau_A",
            "k_max_iter",
            "l2_reg_diag",
            "device",
        }
    )

    def __init__(
        self,
        epochs=300,
        batch_size=100,
        lr=3e-3,
        graph_threshold=0.3,
        encoder_hidden=64,
        decoder_hidden=64,
        indep_test="fisherz",
        alpha=0.05,
        n_jobs=1,
        verbose=False,
        show_progress=False,
        **kwargs,
    ):
        unknown = sorted(set(kwargs) - self._RUNNER_PARAMETERS)
        if unknown:
            raise TypeError("Unsupported DagGNN parameters: " + ", ".join(unknown))

        _load_dagnn()
        if DAG_GNN_IMPORT_ERROR is not None:
            msg = "DagGNN requires PyTorch. Install the PyTorch extra with pip install 'whydra[torch]'."
            raise ImportError(msg) from DAG_GNN_IMPORT_ERROR

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.graph_threshold = graph_threshold
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.indep_test = indep_test
        # These parameters belong to the shared public wrapper interface. DagGNN
        # does not use them and must not pass them to the runner.
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.show_progress = show_progress
        self.runner_kwargs = dict(kwargs)

    def run(
        self, data: "np.ndarray | _optional_imports.DataFrame", **kwargs
    ) -> GeneralGraph:
        """Run the configured algorithm on ``data`` and return its graph."""
        # Infer the data type from the independence test.
        data = np.asarray(data)
        if self.indep_test in ("chisq", "gsq"):
            data_type = "discrete"
            # Discrete data must be integers.
            data = data.astype(int)
        else:  # fisherz, parcorr, etc.
            data_type = "continuous"
            # Normalize continuous data.
            data_mean = np.mean(data, axis=0)
            data_std = np.std(data, axis=0)
            data_std[data_std == 0] = 1.0
            data = (data - data_mean) / data_std

        actual_batch_size = min(self.batch_size, data.shape[0])

        assert dagnn_runner is not None
        adj_matrix = dagnn_runner.train_dag_gnn(
            data,
            data_type=data_type,
            epochs=self.epochs,
            batch_size=actual_batch_size,
            lr=self.lr,
            graph_threshold=self.graph_threshold,
            encoder_hidden=self.encoder_hidden,
            decoder_hidden=self.decoder_hidden,
            **self.runner_kwargs,
        )

        return self._matrix_to_graph(adj_matrix, data.shape[1])

    def _matrix_to_graph(self, matrix, n_nodes):
        node_names = default_node_names(n_nodes)
        nodes = [GraphNode(name) for name in node_names]
        g = GeneralGraph(nodes)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue
                if matrix[i, j] != 0:
                    edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW)
                    g.add_edge(edge)
        return g


#: Public wrappers for standalone algorithms, mapping classes to short aliases.
#: The canonical name (``ClassName.lower()``) is registered automatically.
_STANDALONE_ALGORITHM_ALIASES: list[tuple[type[CausalAlgorithm], tuple[str, ...]]] = [
    (StandalonePCStable, ("pc_stable", "pcstable", "pc_stable_seq")),
    (ParallelPCStable, ("pc_stable_parallel", "parallel_pc_stable", "pc_stable_par")),
    (StandaloneFCIStable, ("fci_stable", "fcistable")),
    (StandaloneFCIStableCached, ("fci_stable_cached", "fci_stable_cash")),
    (
        StandaloneFCIStableParallelCached,
        ("fci_stable_parallel_cached", "fci_stable_paralel_cash"),
    ),
    (StandaloneCFCIStable, ("cfci_stable",)),
    (
        StandaloneCFCIStableParallelCached,
        ("cfci_stable_parallel_cached", "cfci_stable_paralel_cash"),
    ),
    (StandaloneRFCIStable, ("rfci_stable",)),
    (StandaloneGFCIStable, ("gfci_stable",)),
    (StandaloneFCIPlusStable, ("fci_plus_stable",)),
    (
        StandaloneFCIPlusStableParallelCached,
        ("fci_plus_stable_parallel_cached", "fci_plus_stable_paralel_cash"),
    ),
    (StandaloneRAIStable, ("rai_stable",)),
    (DagGNN, ("dag_gnn",)),
]


def _build_standalone_registry(
    pairs: list[tuple[type[CausalAlgorithm], tuple[str, ...]]],
) -> dict[str, type[CausalAlgorithm]]:
    registry: dict[str, type[CausalAlgorithm]] = {}
    for algo_cls, aliases in pairs:
        registry[algo_cls.__name__.lower()] = algo_cls
        for alias in aliases:
            registry[alias] = algo_cls
    return registry


_STANDALONE_ALGORITHM_REGISTRY: dict[str, type[CausalAlgorithm]] = (
    _build_standalone_registry(_STANDALONE_ALGORITHM_ALIASES)
)
_TEMPORAL_ALGORITHM_REGISTRY = {
    alias: name
    for name, aliases in _TEMPORAL_ALGORITHM_ALIASES.items()
    for alias in (name.lower(), *aliases)
}


def _normalize_algorithm_name(name: str) -> str:
    """Normalize algorithm name to make lookups more user-friendly."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def available_standalone_algorithms() -> list[str]:
    """Return canonical standalone algorithm names supported by the unified API."""
    return [
        *(algo_cls.__name__ for algo_cls, _ in _STANDALONE_ALGORITHM_ALIASES),
        "StandalonePCMCI",
        "StandalonePCMCIPlus",
        "StandaloneJPCMCIPlus",
    ]


def create_standalone_algorithm(name: str, **kwargs) -> CausalAlgorithm:
    """
    Create a standalone algorithm wrapper by name.

    Examples
    --------
        create_standalone_algorithm("pc_stable", alpha=0.01)
        create_standalone_algorithm("StandaloneRAIStable", n_jobs=8)
    """
    normalized_name = _normalize_algorithm_name(name)
    if "paralel" in normalized_name or normalized_name.endswith("_cash"):
        warnings.warn(
            "The paralel/cash alias is deprecated; use parallel_cached/cached (retained through 0.x)",
            DeprecationWarning,
            stacklevel=2,
        )
    temporal_class_name = _TEMPORAL_ALGORITHM_REGISTRY.get(normalized_name)
    if temporal_class_name is not None:
        return _temporal_algorithm_class(temporal_class_name)(**kwargs)
    algo_cls = _STANDALONE_ALGORITHM_REGISTRY.get(normalized_name)
    if algo_cls is None:
        available = ", ".join(available_standalone_algorithms())
        msg = (
            f"Unknown standalone algorithm '{name}'. Available algorithms: {available}"
        )
        raise ValueError(msg)
    return algo_cls(**kwargs)


def _text_table_has_header(path: Path) -> bool:
    """Infer a header if the first nonempty line contains a nonnumeric field.

    Split the line on commas, semicolons, and whitespace. A row of numbers
    counts as an observation. An empty file has no header.
    """
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                first_line = line
                break
        else:
            return False

    tokens = first_line.replace(",", " ").replace(";", " ").replace("	", " ").split()
    if not tokens:
        return False
    for token in tokens:
        try:
            float(token)
        except ValueError:
            return True
    return False


def _load_data_like_main(
    data_or_path: "np.ndarray | _optional_imports.DataFrame | str | Path",
    header: bool | None = None,
    *,
    sep=None,
    delimiter=None,
    trusted_pickle=False,
    preserve_names=False,
    validate_table=True,
) -> "np.ndarray | _optional_imports.DataFrame":
    """
    Load observations from a NumPy array, a pandas DataFrame, or a file.

    Return a NumPy array unless ``preserve_names`` is set and the input is a
    DataFrame or a text file with a header. For text, ``header`` overrides
    header detection and ``validate_table`` enables data validation. For
    ``.npy`` files, load the array with pickling disabled. Legacy pickle
    files require explicit opt-in. See :func:`run_standalone_algorithm`.
    """
    if _is_dataframe(data_or_path):
        return data_or_path if preserve_names else data_or_path.to_numpy()
    if isinstance(data_or_path, np.ndarray):
        return data_or_path

    if isinstance(data_or_path, (str, Path)):
        path = Path(data_or_path)
        if not path.exists():
            msg = f"Data file not found: {path}"
            raise FileNotFoundError(msg)

        suffix = path.suffix.lower()
        if suffix in {".pickle", ".pkl"}:
            if not trusted_pickle:
                msg = "Legacy pickle loading requires explicit trusted_pickle=True; migrate trusted data to NPY"
                raise ValueError(msg)
            warnings.warn(
                "trusted_pickle permits arbitrary code execution; convert this trusted file to NPY",
                DeprecationWarning,
                stacklevel=2,
            )
            with path.open("rb") as f:
                loaded = pickle.load(f)
            if not isinstance(loaded, np.ndarray):
                msg = (
                    f"Pickle file '{path}' does not contain np.ndarray "
                    f"(got {type(loaded).__name__})."
                )
                raise ValueError(msg)
            return loaded

        if suffix == ".npy":
            return np.load(path, allow_pickle=False)

        if sep is not None and delimiter is not None and sep != delimiter:
            msg = "sep and delimiter disagree"
            raise ValueError(msg)
        separator = sep if sep is not None else delimiter
        if separator is None:
            import csv

            sample = path.read_text(encoding="utf-8-sig")[:8192]
            try:
                separator = csv.Sniffer().sniff(sample, delimiters=",;\t ").delimiter
                if separator == " ":
                    separator = r"\s+"
            except csv.Error:
                separator = r"\s+"
        has_header = _text_table_has_header(path) if header is None else bool(header)
        import pandas as pd

        df = pd.read_csv(
            path, sep=separator, header=0 if has_header else None, encoding="utf-8-sig"
        )
        if df.shape[1] == 1:
            first_line = path.read_text(encoding="utf-8-sig").splitlines()[0]
            if any(d in first_line for d in ",;\t"):
                msg = "Wrong delimiter produced a single column"
                raise ValueError(msg)
        if validate_table:
            validate_data(df)
        return df if preserve_names and has_header else df.to_numpy()

    msg = (
        "data_or_path must be np.ndarray, str, or pathlib.Path. "
        f"Got {type(data_or_path).__name__}."
    )
    raise TypeError(msg)


def run_standalone_algorithm(
    name: str,
    data_or_path: "np.ndarray | _optional_imports.DataFrame | str | Path",
    header: bool | None = None,
    *,
    sep=None,
    delimiter=None,
    trusted_pickle=False,
    **kwargs,
) -> GeneralGraph:
    """
    Run a standalone causal graph discovery algorithm.

    Choose an algorithm by name and run it on in-memory observations or a
    data file. Pass its constructor options as keyword arguments.

    Parameters
    ----------
    name : str
        Algorithm name (canonical class name or alias). Several forms
        are accepted, for example ``"StandalonePCStable"``,
        ``"pc_stable"``, ``"pcstable"``, ``"fci_stable"``,
        ``"rai_stable"``.
    data_or_path : numpy.ndarray, DataFrame, str, pathlib.Path, or collection of environments
        Input data for the algorithm:

        - ``numpy.ndarray``: an in-memory array of observations.
        - ``DataFrame``: a table with variable names.
        - J-PCMCI+ also accepts a list, tuple, dictionary or 3D array of environments.
        - ``str`` / ``Path``: path to a data file.

        Supported file formats:

        - ``.pickle`` / ``.pkl``: the stored object must be a ``numpy.ndarray``.
        - ``.npy``: loaded with ``numpy.load``.
        - Text tables are read with ``pandas.read_csv``.

    header : bool, optional
        Override automatic header detection for text files.
        With ``True``, the first row is a header. With ``False``, it contains data.
        The default, ``None``, treats the first nonempty line as a header if
        any field on that line is nonnumeric.
        Ignored for ``.npy`` and pickle files.
    sep, delimiter : str, optional
        Separator for text files. These are aliases and must agree if both
        are given. By default, detect commas, semicolons, tabs, or spaces,
        using whitespace if detection fails.
    trusted_pickle : bool, optional
        Allow loading a legacy pickle file. Enable this only for a file you
        trust, because unpickling can execute arbitrary code. Defaults to False.
    **kwargs
        Constructor parameters for the chosen algorithm wrapper.
        Passed to :func:`create_standalone_algorithm` (for example,
        ``alpha``, ``indep_test``, ``n_jobs``, ``verbose``).

    Returns
    -------
    GeneralGraph
        Graph in Whydra's internal format. The algorithm determines the graph
        type: DAG, CPDAG/PDAG or PAG. For example, FCI returns a PAG.

    Raises
    ------
    ValueError
        If the algorithm name is unknown, the separators disagree, or the text
        table fails validation. Also raised if pickle loading is disabled or
        the stored object is not a NumPy array.
    FileNotFoundError
        If the data file does not exist.
    TypeError
        If the type of ``data_or_path`` is unsupported.
    Exception
        Any exception raised by the chosen wrapper's constructor or
        its ``run`` method.

    Notes
    -----
    Use ``.npy`` or a text table for data from an untrusted source.

    Examples
    --------
    Run on a NumPy array:

    >>> graph = run_standalone_algorithm(
    ...     "pc_stable",
    ...     data,
    ...     alpha=0.01,
    ...     indep_test="fisherz",
    ...     n_jobs=8
    ... )

    Run on data from a file:

    >>> graph = run_standalone_algorithm(
    ...     "StandaloneFCIStable",
    ...     "data/observations.npy",
    ...     alpha=0.05,
    ...     verbose=True
    ... )
    """
    algorithm = create_standalone_algorithm(name, **kwargs)
    is_temporal = _normalize_algorithm_name(name) in _TEMPORAL_ALGORITHM_REGISTRY
    if is_temporal and not isinstance(data_or_path, (str, Path)):
        return algorithm.run(data_or_path)
    data = _load_data_like_main(
        data_or_path,
        header=header,
        sep=sep,
        delimiter=delimiter,
        trusted_pickle=trusted_pickle,
        preserve_names=True,
        validate_table=not is_temporal,
    )
    return algorithm.run(data)


__all__ = [
    "DagGNN",
    "ParallelPCStable",
    "StandaloneCFCIStable",
    "StandaloneCFCIStableParallelCached",
    "StandaloneFCIPlusStable",
    "StandaloneFCIPlusStableParallelCached",
    "StandaloneFCIStable",
    "StandaloneFCIStableCached",
    "StandaloneFCIStableParallelCached",
    "StandaloneGFCIStable",
    "StandaloneJPCMCIPlus",
    "StandalonePCMCI",
    "StandalonePCMCIPlus",
    "StandalonePCStable",
    "StandaloneRAIStable",
    "StandaloneRFCIStable",
    "available_standalone_algorithms",
    "create_standalone_algorithm",
    "run_standalone_algorithm",
]
