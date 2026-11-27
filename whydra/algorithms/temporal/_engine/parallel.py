"""Process parallelism for the native PCMCI, PCMCI+ and J-PCMCI+ engines.

Lagged parent selection and PCMCI's MCI tests are independent across targets.
Skeleton tests are independent across edges at one stable conditioning depth,
with both directions of a contemporaneous edge kept together. Graph updates and
CI observations are replayed in deterministic search order in the parent process;
orientation stays sequential.
"""

import itertools
import math
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs, parallel_config

from ._joint_search import _pc_alpha_candidates
from .j_pcmci_plus import JPCMCIplus
from .pcmci import PCMCIPipeline
from .pcmci_plus import PCMCIPlusPipeline

if TYPE_CHECKING:
    from ._joint_search import JointSearch as _ParallelSearchBase
else:
    _ParallelSearchBase = object


def _engine_snapshot(engine):
    """Remove live process resources and parent-only observer closures."""
    snapshot = copy(engine)
    for name in list(vars(snapshot)):
        if name.startswith("_parallel_") or name == "_ci_observer":
            delattr(snapshot, name)
    # Unwrap the application's observer proxy, while retaining any genuine
    # instance override supplied by a custom CI test.
    ci = getattr(engine.cond_ind_test, "_jpcmci_raw_ci_test", engine.cond_ind_test)
    snapshot.cond_ind_test = copy(ci)
    return snapshot


def _worker_engine(snapshot):
    """Isolate mutable CI/frame state while sharing read-only input arrays."""
    engine = copy(snapshot)
    engine.diagnostics = list(getattr(snapshot, "diagnostics", []))
    engine.dataframe = copy(snapshot.dataframe)
    engine.dataframe.use_indices_dataset_dict = {}
    engine.cond_ind_test = deepcopy(
        snapshot.cond_ind_test,
        memo={id(snapshot.cond_ind_test.dataframe): engine.dataframe},
    )
    engine.cond_ind_test.set_dataframe(engine.dataframe)
    engine.verbosity = 0
    engine.cond_ind_test.verbosity = 0
    events = []
    run_test = engine.cond_ind_test.run_test

    def recorded_test(
        X, Y, Z=None, tau_max=0, cut_off="2xtau_max", alpha_or_thres=None
    ):
        event = {
            "X": list(X),
            "Y": list(Y),
            "Z": list(Z or []),
            "alpha_or_thres": alpha_or_thres,
            "stage": getattr(engine, "mode", "skeleton"),
        }
        outcome = run_test(
            X, Y, Z=Z, tau_max=tau_max, cut_off=cut_off, alpha_or_thres=alpha_or_thres
        )
        event["outcome"] = outcome
        events.append(event)
        return outcome

    engine.cond_ind_test.run_test = recorded_test
    return engine, events


def _cache_checkpoint(engine):
    ci = engine.cond_ind_test
    return (
        set(getattr(ci, "cached_ci_results", {})),
        dict(getattr(ci, "ci_results", {})),
        set(getattr(ci, "residuals", {})),
        len(getattr(engine, "diagnostics", [])),
    )


def _cache_changes(engine, checkpoint):
    ci = engine.cond_ind_test
    cached, results, residuals, diagnostic_count = checkpoint
    return {
        "cached_ci_results": {
            key: value
            for key, value in getattr(ci, "cached_ci_results", {}).items()
            if key not in cached
        },
        "ci_results": {
            key: value
            for key, value in getattr(ci, "ci_results", {}).items()
            if key not in results or value != results[key]
        },
        "residuals": {
            key: value
            for key, value in getattr(ci, "residuals", {}).items()
            if key not in residuals
        },
        "use_indices_dataset_dict": engine.dataframe.use_indices_dataset_dict,
        "bk_diagnostics": list(getattr(engine, "diagnostics", [])[diagnostic_count:]),
    }


def _pc_target(snapshot, j, assumptions, alphas, optimize, parameters):
    engine, events = _worker_engine(snapshot)
    checkpoint = _cache_checkpoint(engine)
    scores = np.zeros_like(alphas)
    results = {}
    for index, alpha in enumerate(alphas):
        results[alpha] = engine._run_pc_stable_single(
            j, link_assumptions_j=assumptions, pc_alpha=alpha, **parameters
        )
        if optimize:
            scores[index] = engine.cond_ind_test.get_model_selection_criterion(
                j, results[alpha]["parents"], parameters["tau_max"]
            )
    optimal_alpha = alphas[scores.argmin()]
    return (
        j,
        results[optimal_alpha],
        optimal_alpha,
        scores,
        results,
        events,
        _cache_changes(engine, checkpoint),
        os.getpid(),
    )


def _mci_target(snapshot, j, candidates, parents, parameters):
    engine, events = _worker_engine(snapshot)
    checkpoint = _cache_checkpoint(engine)
    evidence = engine._run_mci_target(j, candidates, parents, **parameters)
    return j, evidence, events, _cache_changes(engine, checkpoint), os.getpid()


def _edge_chunks(groups, adjacency, depth, max_combinations, n_jobs):
    """Balance estimated CI work without serializing a snapshot per edge."""
    count = min(len(groups), max(1, 2 * effective_n_jobs(n_jobs)))
    chunks: list[list[Any]] = [[] for _ in range(count)]
    loads = [0] * count
    weighted = []
    for directions in groups:
        weight = 0
        for _, (i, j, lag), _, _ in directions:
            candidates = sum(node != (i, -lag) for node in adjacency[j])
            combinations = math.comb(candidates, depth)
            if math.isfinite(max_combinations):
                combinations = min(combinations, math.floor(max_combinations) + 1)
            weight += combinations
        weighted.append((weight, directions))
    for weight, directions in sorted(
        weighted, key=lambda item: (-item[0], item[1][0][0])
    ):
        index = min(range(count), key=lambda item: (loads[item], item))
        chunks[index].append(directions)
        loads[index] += weight
    # Keep the original order within each chunk, including each reciprocal pair.
    for chunk in chunks:
        chunk.sort(key=lambda directions: directions[0][0])
    return chunks


class _EdgeEvidence(NamedTuple):
    minimum: float
    probability: float
    statistic: float | None
    separator: tuple | None


def _skeleton_chunk(
    snapshot, groups, graph, adjacency, depth, lagged_parents, parameters
):
    engine, events = _worker_engine(snapshot)
    # Both directions see reciprocal removal exactly as in the serial loop.
    local_graph = np.array(graph, copy=True)
    traces = []
    for index, (i, j, lag), minimum, probability in itertools.chain.from_iterable(
        groups
    ):
        if local_graph[i, j, lag] == "":
            continue
        checkpoint = _cache_checkpoint(engine)
        first_event = len(events)
        statistic = None
        separator = None
        conditions = itertools.combinations(
            [node for node in adjacency[j] if node != (i, -lag)], depth
        )
        for q, S in enumerate(conditions):
            # Retain the skeleton's inclusive boundary. PC1 uses
            # its own, different >= boundary in _run_pc_stable_single.
            if q > parameters["max_combinations"]:
                break
            value, p_value, _, dependent = engine._run_pcalg_test(
                graph=local_graph,
                i=i,
                abstau=lag,
                j=j,
                S=S,
                lagged_parents=lagged_parents,
                max_conds_py=parameters["max_conds_py"],
                max_conds_px=parameters["max_conds_px"],
                max_conds_px_lagged=parameters["max_conds_px_lagged"],
                tau_max=parameters["tau_max"],
                alpha_or_thres=parameters["pc_alpha"],
            )
            minimum = min(np.abs(value), minimum)
            if p_value >= probability:
                probability, statistic = float(p_value), value
            if not dependent:
                separator = S
                local_graph[i, j, lag] = ""
                if lag == 0:
                    local_graph[j, i, 0] = ""
                break
        traces.append(
            (
                index,
                (i, j, lag),
                _EdgeEvidence(minimum, probability, statistic, separator),
                events[first_event:],
                _cache_changes(engine, checkpoint),
            )
        )
    return traces, os.getpid()


class _ParallelSearch(_ParallelSearchBase):
    """Share one loky pool and deterministic PC1/skeleton replay across engines."""

    def __init__(self, *args, n_jobs=1, backend="loky", **kwargs):
        if (
            isinstance(n_jobs, (bool, np.bool_))
            or not isinstance(n_jobs, (int, np.integer))
            or (n_jobs < 1 and n_jobs != -1)
        ):
            msg = "n_jobs must be a positive integer or -1"
            raise ValueError(msg)
        if backend != "loky":
            msg = "Temporal parallel backend must be 'loky'"
            raise ValueError(msg)
        self.n_jobs = int(n_jobs)
        self.backend = backend
        self._parallel_executor = None
        self._parallel_worker_pids = set()
        super().__init__(*args, **kwargs)
        if (
            self.n_jobs != 1
            and getattr(self.cond_ind_test, "significance", None) != "analytic"
        ):
            msg = (
                "Parallel temporal search requires analytic CI significance; "
                "use n_jobs=1 for stochastic CI tests"
            )
            raise ValueError(msg)

    @contextmanager
    def _parallel_context(self) -> Iterator[Callable[[Iterable], list[tuple]]]:
        if self._parallel_executor is not None:
            yield self._parallel_executor
            return
        with (
            parallel_config(backend="loky", inner_max_num_threads=1),
            Parallel(n_jobs=self.n_jobs, max_nbytes="1M", mmap_mode="r") as pool,
        ):
            # Joblib's default return mode collects one result tuple per worker task.
            executor = cast("Callable[[Iterable], list[tuple]]", pool)
            self._parallel_executor = executor
            try:
                yield executor
            finally:
                self._parallel_executor = None

    def _replay_ci(self, events, changes):
        ci = self.cond_ind_test
        self.diagnostics.extend(changes["bk_diagnostics"])
        if hasattr(ci, "cached_ci_results"):
            ci.cached_ci_results.update(changes["cached_ci_results"])
        if hasattr(ci, "ci_results"):
            ci.ci_results.update(changes["ci_results"])
        if hasattr(ci, "residuals"):
            ci.residuals.update(changes["residuals"])
        self.dataframe.use_indices_dataset_dict = changes["use_indices_dataset_dict"]
        observer = getattr(self, "_ci_observer", None)
        if observer is not None:
            for event in events:
                observer(event)

    def _parallel_run(self, run, *args, **kwargs):
        if self.n_jobs == 1:
            return run(*args, **kwargs)
        if self._parallel_executor is None:
            self._parallel_worker_pids.clear()
        with self._parallel_context():
            return run(*args, **kwargs)

    def run_pc_stable(
        self,
        selected_links=None,
        link_assumptions=None,
        tau_min=1,
        tau_max=1,
        save_iterations=False,
        pc_alpha=0.2,
        max_conds_dim=None,
        max_combinations=1,
    ):
        if self.n_jobs == 1:
            return super().run_pc_stable(
                selected_links=selected_links,
                link_assumptions=link_assumptions,
                tau_min=tau_min,
                tau_max=tau_max,
                save_iterations=save_iterations,
                pc_alpha=pc_alpha,
                max_conds_dim=max_conds_dim,
                max_combinations=max_combinations,
            )
        if selected_links is not None:
            msg = "Use link_assumptions instead of selected_links"
            raise ValueError(msg)
        self._check_tau_limits(tau_min, tau_max)
        if max_combinations <= 0:
            msg = "max_combinations must be > 0"
            raise ValueError(msg)
        alphas, optimize = _pc_alpha_candidates(pc_alpha)
        if not alphas or any(not 0 <= alpha <= 1 for alpha in alphas):
            msg = "pc_alpha must be in [0, 1]"
            raise ValueError(msg)
        tau_min = max(1, tau_min)
        if self.verbosity > 0:
            self._print_pc_params(
                link_assumptions,
                tau_min,
                tau_max,
                alphas,
                max_conds_dim,
                max_combinations,
            )
        assumptions = self._set_link_assumptions(
            link_assumptions, tau_min, tau_max, remove_contemp=True
        )
        parameters = {
            "tau_min": tau_min,
            "tau_max": tau_max,
            "save_iterations": save_iterations,
            "max_conds_dim": self._set_max_condition_dim(
                max_conds_dim, tau_min, tau_max
            ),
            "max_combinations": max_combinations,
        }
        snapshot = _engine_snapshot(self)
        with self._parallel_context() as executor:
            targets = executor(
                delayed(_pc_target)(
                    snapshot, j, assumptions[j], alphas, optimize, parameters
                )
                for j in range(self.N)
            )

        def selected_targets():
            for j, result, optimal, scores, candidates, events, changes, pid in targets:
                self._parallel_worker_pids.add(pid)
                self._replay_ci(events, changes)
                if self.verbosity > 1 and optimize:
                    self._print_pc_sel_results(alphas, candidates, j, scores, optimal)
                yield j, result, optimal if optimize else None

        all_parents = self._collect_pc1_results(
            selected_targets(),
            tau_max,
            val_min=defaultdict(dict),
            pval_max=defaultdict(dict),
            iterations=defaultdict(dict),
        )
        if self.verbosity > 0:
            self._print_parents(all_parents, self.val_min, self.pval_max)
        return all_parents

    def _pcalg_skeleton(
        self,
        initial_graph,
        lagged_parents,
        mode,
        pc_alpha,
        tau_min,
        tau_max,
        max_conds_dim,
        max_combinations,
        max_conds_py,
        max_conds_px,
        max_conds_px_lagged,
    ):
        if self.n_jobs == 1:
            return super()._pcalg_skeleton(
                initial_graph,
                lagged_parents,
                mode,
                pc_alpha,
                tau_min,
                tau_max,
                max_conds_dim,
                max_combinations,
                max_conds_py,
                max_conds_px,
                max_conds_px_lagged,
            )
        graph, p_matrix, val_matrix, val_min = self._initial_skeleton(
            initial_graph, tau_max
        )
        adjacency_fn = (
            self._get_adj_time_series_contemp
            if mode == "contemp_conds"
            else self._get_adj_time_series
        )
        adjacency = adjacency_fn(graph)
        sepsets = self._get_sepsets(tau_min, tau_max)
        parameters = {
            "pc_alpha": pc_alpha,
            "tau_max": tau_max,
            "max_combinations": np.inf
            if max_combinations is None
            else max_combinations,
            "max_conds_py": max_conds_py,
            "max_conds_px": max_conds_px,
            "max_conds_px_lagged": max_conds_px_lagged,
        }
        max_conds_dim = self._set_max_condition_dim(max_conds_dim, tau_min, tau_max)
        max_combinations = parameters["max_combinations"]
        depth = 0
        with self._parallel_context() as executor:
            while (
                self._any_tests_remaining(graph, adjacency, tau_min, tau_max, depth)
                and depth <= max_conds_dim
            ):
                pairs = self._remaining_pairs(graph, adjacency, tau_min, tau_max, depth)
                groups: dict[
                    tuple[int, int, int],
                    list[tuple[int, tuple[int, int, int], float, float]],
                ] = {}
                for index, (i, j, lag) in enumerate(pairs):
                    key = (min(i, j), max(i, j), 0) if lag == 0 else (i, j, lag)
                    edge = (i, j, lag)
                    groups.setdefault(key, []).append(
                        (index, edge, val_min[j][(i, -lag)], float(p_matrix[edge]))
                    )
                snapshot = _engine_snapshot(self)
                chunks = _edge_chunks(
                    list(groups.values()),
                    adjacency,
                    depth,
                    max_combinations,
                    self.n_jobs,
                )
                batches = executor(
                    delayed(_skeleton_chunk)(
                        snapshot,
                        chunk,
                        graph,
                        adjacency,
                        depth,
                        lagged_parents,
                        parameters,
                    )
                    for chunk in chunks
                )
                traces = []
                for batch, pid in batches:
                    self._parallel_worker_pids.add(pid)
                    traces.extend(batch)
                for _, edge, evidence, events, changes in sorted(traces):
                    self._replay_ci(events, changes)
                    i, j, lag = edge
                    val_min[j][(i, -lag)] = evidence.minimum
                    if evidence.statistic is not None:
                        p_matrix[edge] = evidence.probability
                        val_matrix[edge] = evidence.statistic
                    if evidence.separator is None:
                        continue
                    separator = list(evidence.separator)
                    directions = ((i, j), (j, i)) if lag == 0 else ((i, j),)
                    for source, target in directions:
                        graph[source, target, lag] = ""
                        sepsets[((source, -lag), target)] = separator
                    if lag == 0:
                        p_matrix[j, i, 0] = p_matrix[edge]
                depth += 1
                adjacency = adjacency_fn(graph, sort_by=val_min)
        return {
            "graph": graph,
            "sepsets": sepsets,
            "p_matrix": p_matrix,
            "val_matrix": val_matrix,
        }


class ParallelPCMCIPipeline(_ParallelSearch, PCMCIPipeline):
    """Native PCMCI with target-parallel PC1 and MCI tests."""

    def run_pcmci(self, *args, **kwargs):
        return self._parallel_run(super().run_pcmci, *args, **kwargs)

    def run_mci(self, *args, **kwargs):
        return self._parallel_run(super().run_mci, *args, **kwargs)

    def _iter_mci_results(self, assumptions, parents, **parameters):
        if self.n_jobs == 1:
            yield from super()._iter_mci_results(assumptions, parents, **parameters)
            return
        snapshot = _engine_snapshot(self)
        with self._parallel_context() as executor:
            targets = executor(
                delayed(_mci_target)(snapshot, j, candidates, parents, parameters)
                for j, candidates in assumptions.items()
            )
        for j, evidence, events, changes, pid in targets:
            self._parallel_worker_pids.add(pid)
            self._replay_ci(events, changes)
            yield j, evidence


class ParallelPCMCIPlusPipeline(_ParallelSearch, PCMCIPlusPipeline):
    """Native PCMCI+ with parallel PC1/skeleton and sequential orientation."""

    def run_pcmciplus(self, *args, **kwargs):
        return self._parallel_run(super().run_pcmciplus, *args, **kwargs)


class ParallelJPCMCIplus(_ParallelSearch, JPCMCIplus):
    """Native J-PCMCI+ with deterministic loky workers for CI-heavy phases."""

    def run_jpcmciplus(self, *args, **kwargs):
        return self._parallel_run(super().run_jpcmciplus, *args, **kwargs)
