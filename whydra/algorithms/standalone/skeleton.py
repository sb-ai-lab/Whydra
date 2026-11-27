"""Skeleton discovery for PC-stable and FCI-stable, CPU and CUDA."""

import math
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from itertools import chain, combinations, islice
from multiprocessing import cpu_count
from time import perf_counter
from typing import TYPE_CHECKING, overload

import numpy as np

from ..graph_core import CIT, CausalGraph, fisher_z_from_corr
from ..graph_core._optional_imports import Parallel, delayed, tqdm
from ..graph_core.ci_batch import (
    _merge_execution_stats,
    batch_execution_stats,
    configure_fisher_alpha,
    evaluate_ci_batch,
    fisher_z_batch_from_corr,
    supports_fisher_batch,
)
from ..graph_core.conditioning import bounded_size
from ..graph_core.cpu_graph_convert import causal_graph_to_causallearn
from ..graph_core.endpoints import Endpoint
from ..graph_core.independence_tests import (
    _builtin_fisherz,
    _fisher_z_reference,
    _threshold_pvalue,
    fisher_z_from_corr_batch,
)
from ..graph_core.native_fisher import NATIVE_AVAILABLE, native_level
from ..graph_core.process_execution import plan_process_work, process_map
from ..graph_core.validation import effective_n_jobs
from ._counted_ci import is_counted_fisherz
from .fisher_skeleton import fisher_level_chunk, pair_chunks

if TYPE_CHECKING:
    from causallearn.graph.GeneralGraph import GeneralGraph as CLGeneralGraph
import sys

from .profiler import profiler

_DEFAULT_FISHER_BATCH = fisher_z_batch_from_corr


class _PCChunkResult(tuple):
    """Keep the six-field worker protocol while attaching measured dispatch."""

    native_execution: dict | None

    def __new__(cls, values, native_execution=None):
        obj = super().__new__(cls, values)
        obj.native_execution = native_execution
        return obj


def _record_native_chunks(execution, results):
    native = execution.setdefault(
        "native_kernel",
        {
            "cpp_chunks": 0,
            "python_chunks": 0,
            "compiled_low_order_queries": 0,
            "python_numeric_queries": 0,
            "windows": 0,
            "worker_seconds": 0.0,
        },
    )
    for result in results:
        details = (
            result.get("native_execution")
            if isinstance(result, dict)
            else getattr(result, "native_execution", None)
        )
        if details is None:
            native["python_chunks"] += 1
        else:
            native["cpp_chunks"] += 1
            for key in (
                "compiled_low_order_queries",
                "python_numeric_queries",
                "windows",
            ):
                native[key] += details[key]
            # Sum of worker durations, not the fit's wall-clock duration.
            native["worker_seconds"] += details["seconds"]


def _pc_execution_stats(cg, backend, n_jobs):
    """Describe submitted work separately from the requested worker count."""
    stats = getattr(cg, "pc_execution_stats", None)
    if stats is None:
        stats = {
            "requested_backend": backend,
            "configured_n_jobs": n_jobs,
            "coarse_batches": dict.fromkeys(("serial", "threads", "processes"), 0),
            "coarse_tasks": dict.fromkeys(("serial", "threads", "processes"), 0),
            "depths": [],
            "task_build_seconds": 0.0,
            "cache_publication_seconds": 0.0,
            "cache_seed_entries": 0,
            "cache_updates_returned": 0,
        }
        cg.pc_execution_stats = stats
        cg.G.metadata["pc_execution"] = stats
    return stats


def _record_pc_dispatch(cg, backend, n_jobs, actual, depth, tasks):
    stats = _pc_execution_stats(cg, backend, n_jobs)
    if tasks:
        chosen = (
            stats.get("process_policy", [{}])[-1].get("selected_jobs", n_jobs)
            if actual == "processes"
            else 1
        )
        _record_execution(cg, "skeleton_pc", depth, 0, n_jobs, chosen, actual)
        stats["coarse_batches"][actual] += 1
        stats["coarse_tasks"][actual] += tasks
        stats["depths"].append({"depth": depth, "backend": actual, "tasks": tasks})
    return stats


def _record_fci_dispatch(cg, backend, n_jobs, actual=None, depth=None, tasks=0):
    stats = getattr(cg, "fci_skeleton_execution_stats", None)
    if stats is None:
        stats = {
            "requested_backend": backend,
            "configured_n_jobs": n_jobs,
            "coarse_batches": dict.fromkeys(("serial", "threads", "processes"), 0),
            "coarse_tasks": dict.fromkeys(("serial", "threads", "processes"), 0),
            "depths": [],
        }
        cg.fci_skeleton_execution_stats = stats
    if tasks:
        chosen = (
            stats.get("process_policy", [{}])[-1].get("selected_jobs", n_jobs)
            if actual == "processes"
            else 1
        )
        _record_execution(cg, "skeleton_fci", depth, 0, n_jobs, chosen, actual)
        stats["coarse_batches"][actual] += 1
        stats["coarse_tasks"][actual] += tasks
        stats["depths"].append({"depth": depth, "backend": actual, "tasks": tasks})
    return stats


def _pc_fisher_edge_chunk(
    edges, neighbors, corr_matrix, n_samples, depth, alpha, cache_size, cache_seed
):
    """Enumerate and reduce a frozen PC depth without transporting a CIT.

    Both directions of an unordered pair stay in one process. Common subsets
    are recognized by membership in the first pool, so symmetry needs no
    growing set of all previously visited CI keys. The second pool retains
    its original argument order for subsets absent from the first pool.
    """
    native = native_level(
        edges,
        neighbors,
        corr_matrix,
        n_samples,
        depth,
        alpha,
        {key: (value, False) for key, value in cache_seed.items()},
        cache_size > 0,
        False,
        cache_size,
        False,
        batch=fisher_z_batch_from_corr,
        window=512,
        native_math=fisher_z_batch_from_corr is _DEFAULT_FISHER_BATCH,
    )
    if native is not None:
        reductions = [
            (a, b, members)
            for a, b, _, separators, members in native["pairs"]
            if separators
        ]
        return _PCChunkResult(
            (
                reductions,
                native["computed"],
                native["cit_hits"],
                native["updates"],
                native["errors"],
                native["discarded"],
            ),
            native["native_execution"],
        )
    cache_enabled = cache_size > 0
    reductions = []
    updates = []
    misses = hits = cacheable = 0
    errors = []
    requests = []
    destinations = []
    found = [False] * len(edges)
    members = [set() for _ in edges]

    def consume(edge_index, x, y, condition, value):
        if isinstance(value, BaseException):
            errors.append(((x, y, condition), value))
        elif value > alpha:
            found[edge_index] = True
            members[edge_index].update(condition)

    def flush():
        nonlocal misses, cacheable
        if not requests:
            return
        values = fisher_z_batch_from_corr(
            corr_matrix,
            n_samples,
            requests,
            n_jobs=1,
            batch_size=256,
            return_exceptions=True,
            decision_alpha=alpha,
        )
        misses += len(requests)
        for (edge_index, x, y, condition), value in zip(
            destinations, values, strict=False
        ):
            consume(edge_index, x, y, condition, value)
            if cache_enabled and not isinstance(value, BaseException):
                cacheable += 1
                a, b = edges[edge_index]
                updates.append(((a, b, condition), value))
                # Returning at most the coordinator's cache capacity bounds
                # IPC even for an exhaustive, high-order conditioning level.
                if len(updates) >= max(512, 2 * cache_size):
                    del updates[:-cache_size]
        requests.clear()
        destinations.clear()

    for edge_index, (a, b) in enumerate(edges):
        first_pool = tuple(v for v in neighbors[a] if v != b) if depth else ()
        second_pool = tuple(v for v in neighbors[b] if v != a) if depth else ()
        first_members = frozenset(first_pool)
        for direction, (x, y, pool) in enumerate(
            ((a, b, first_pool), (b, a, second_pool))
        ):
            for condition in combinations(pool, depth):
                if (
                    cache_enabled
                    and direction
                    and all(v in first_members for v in condition)
                ):
                    # The first direction's result already contributes to the
                    # symmetric separating-member union, including depth zero.
                    hits += 1
                    continue
                key = (a, b, condition)
                if cache_enabled and key in cache_seed:
                    hits += 1
                    consume(edge_index, x, y, condition, cache_seed[key])
                    continue
                requests.append((x, y, condition))
                destinations.append((edge_index, x, y, condition))
                if len(requests) >= 512:
                    flush()
                    if errors:
                        break
            if errors:
                break
        if errors:
            break
    flush()
    for (a, b), separated, union in zip(edges, found, members, strict=False):
        if separated:
            reductions.append((a, b, tuple(sorted(union))))
    if cache_enabled and len(updates) > cache_size:
        updates = updates[-cache_size:]
    return reductions, misses, hits, updates, errors, cacheable - len(updates)


def _pc_weighted_edge_chunks(neighbors, depth, n_chunks):
    """Balance compact pair jobs by both directions' conditioning counts."""
    import heapq

    counts = [
        math.comb(len(row) - 1, depth) if len(row) > depth else 0 for row in neighbors
    ]
    sets = [frozenset(row) for row in neighbors]
    weighted = [
        (counts[x] + counts[y] - math.comb(len(sets[x] & sets[y]), depth), x, y)
        for x, row in enumerate(neighbors)
        for y in row
        if x < y
    ]
    weighted = [item for item in weighted if item[0]]
    weighted.sort(reverse=True)
    chunks = [[] for _ in range(min(n_chunks, len(weighted)))]
    loads = [(0, index) for index in range(len(chunks))]
    heapq.heapify(loads)
    for cost, x, y in weighted:
        load, index = heapq.heappop(loads)
        chunks[index].append((x, y))
        heapq.heappush(loads, (load + cost, index))
    # Restore scalar pair priority within each independent process chunk.
    for chunk in chunks:
        chunk.sort()
    return chunks


# ------------------------------
# Keep the worker function at module scope, outside the class.
# ------------------------------
def _skeleton_worker(
    x,
    cg,
    depth,
    alpha,
    stable=True,
    protected_edges: set[frozenset[int]] | frozenset = frozenset(),
):
    """
    Worker function to process a single variable x.

    Checks for independence of x with its neighbours at the given depth.

    Parameters
    ----------
    x : int
        Node index to process.
    cg : CausalGraph
        Causal graph carrying the CI test.
    depth : int
        Size of the conditioning sets.
    alpha : float
        Significance level.
    stable : bool
        Whether to enumerate all separating sets (PC-stable).
    protected_edges : collection of node pairs
        Pairs ``frozenset((x, y))`` kept by background knowledge.

    Returns
    -------
    tuple
        Tuple (edge_removals, sepset_updates), with one (x, y, union)
        record per separated pair; union is a sorted tuple of Python ints.
    """
    edge_removals = []
    sepset_updates = []

    neigh_x = cg.neighbors(x)
    if len(neigh_x) < depth:
        return edge_removals, sepset_updates

    for y in neigh_x:
        neigh_x_noy = np.delete(neigh_x, np.where(neigh_x == y))
        protected = frozenset((int(x), int(y))) in protected_edges

        found = False
        members = set()
        for S in combinations(neigh_x_noy, depth):
            p_val = cg.ci_test(x, y, S)
            if p_val > alpha:
                if not found:
                    edge_removals.append((x, y))
                    found = True
                members.update(int(v) for v in S)
                if not stable or protected:
                    break

        if found:
            sepset_updates.append((x, y, tuple(sorted(members))))

    return edge_removals, sepset_updates


def ci_worker(args):
    """Unpack one worker task and evaluate its conditional-independence test."""
    x, y, S, ci_func = args
    return x, y, S, ci_func(x, y, S)


def _skeleton_worker_fci(x, cg, G1, depth, alpha, pMax_row, stable=True):
    """Worker function for FCI parallel processing."""
    edge_removals = []
    sepset_updates = []
    pmax_updates = []
    tests_performed = 0

    # Neighbors in current graph G
    curr_neighs = cg.neighbors(x)

    for y in curr_neighs:
        # Neighbors of x in G1 (stable reference)
        nbrs = np.where(G1[x, :] != 0)[0]
        nbrs = nbrs[nbrs != y]

        if len(nbrs) >= depth:
            current_pmax = pMax_row[y]

            for S in combinations(nbrs, depth):
                tests_performed += 1
                pval = cg.ci_test(x, y, S)

                # Update pMax if pval is larger
                if pval > current_pmax:
                    current_pmax = pval
                    pmax_updates.append((x, y, pval))

                if pval >= alpha:
                    edge_removals.append((x, y))
                    sepset_updates.append((x, y, S))
                    if not stable:
                        break

    return edge_removals, sepset_updates, pmax_updates, tests_performed


def _ci_cache_key_pair(i, j, S):
    a, b = (int(i), int(j)) if i <= j else (int(j), int(i))
    return a, b, tuple(sorted(int(v) for v in S))


def _cached_fisher_z(local_cache, corr_matrix, n_samples, x, y, S, alpha=None):
    key = _ci_cache_key_pair(x, y, S)
    cached = local_cache.get(key)
    if cached is not None:
        return cached
    pval = fisher_z_from_corr(corr_matrix, n_samples, x, y, S, decision_alpha=alpha)
    local_cache[key] = pval
    return pval


def _process_fci_direction_fisherz(
    x, y, G1, corr_matrix, n_samples, depth, alpha, pmax_value, local_cache, stable=True
):
    edge_removals = []
    sepset_updates = []
    pmax_updates = []
    tests_performed = 0

    nbrs = np.where(G1[x, :] != 0)[0]
    nbrs = nbrs[nbrs != y]

    if len(nbrs) >= depth:
        current_pmax = pmax_value
        for S in combinations(nbrs, depth):
            tests_performed += 1
            pval = _cached_fisher_z(local_cache, corr_matrix, n_samples, x, y, S, alpha)
            if pval > current_pmax:
                current_pmax = pval
                pmax_updates.append((x, y, pval))
            if pval >= alpha:
                edge_removals.append((x, y))
                sepset_updates.append((x, y, S))
                if not stable:
                    break

    return edge_removals, sepset_updates, pmax_updates, tests_performed


def _skeleton_worker_fci_fisherz_edge_chunk(
    edges, G1, pMax, corr_matrix, n_samples, depth, alpha, stable=True
):
    edge_removals = []
    sepset_updates = []
    pmax_updates = []
    tests_performed = 0
    local_cache = {}

    for x, y in edges:
        for a, b in ((x, y), (y, x)):
            out = _process_fci_direction_fisherz(
                a,
                b,
                G1,
                corr_matrix,
                n_samples,
                depth,
                alpha,
                pMax[a, b],
                local_cache,
                stable,
            )
            dir_edge_removals, dir_sepset_updates, dir_pmax_updates, dir_tests = out
            edge_removals.extend(dir_edge_removals)
            sepset_updates.extend(dir_sepset_updates)
            pmax_updates.extend(dir_pmax_updates)
            tests_performed += dir_tests

    return edge_removals, sepset_updates, pmax_updates, tests_performed


def _fci_edge_cost(G1, x, y, depth):
    cost = 0
    nbrs_x = np.where(G1[x, :] != 0)[0]
    nbrs_x = nbrs_x[nbrs_x != y]
    if len(nbrs_x) >= depth:
        cost += math.comb(len(nbrs_x), depth)

    nbrs_y = np.where(G1[y, :] != 0)[0]
    nbrs_y = nbrs_y[nbrs_y != x]
    if len(nbrs_y) >= depth:
        cost += math.comb(len(nbrs_y), depth)

    return cost


def _balanced_edge_chunks(edges, G1, depth, n_chunks):
    weighted_edges = [(_fci_edge_cost(G1, x, y, depth), (x, y)) for x, y in edges]
    weighted_edges.sort(key=lambda item: item[0], reverse=True)

    chunks = [[] for _ in range(n_chunks)]
    loads = [0 for _ in range(n_chunks)]
    for cost, edge in weighted_edges:
        idx = min(range(n_chunks), key=lambda i: loads[i])
        chunks[idx].append(edge)
        loads[idx] += cost

    return [chunk for chunk in chunks if chunk]


class SkeletonDiscovery:
    """Discover a PC/FCI skeleton with CPU batch backends or a resident CUDA path."""

    def __init__(
        self,
        cg: CausalGraph,
        alpha: float = 0.05,
        stable: bool = True,
        n_jobs: int = 1,
        show_progress: bool = False,
        verbose: bool = False,
        parallel_backend: str = "auto",
        device: str | None = None,
        dtype: str | None = None,
        max_depth: int | None = None,
        process_ci_info=None,
        protected_edges=(),
    ):
        cuda = device is not None and str(device).startswith("cuda")
        protected_edges = {frozenset((int(x), int(y))) for x, y in protected_edges}
        if protected_edges and device is not None and str(device).startswith("cuda"):
            msg = (
                "CUDA skeleton does not support protected_edges; "
                "use the CPU backend to preserve required adjacencies"
            )
            raise NotImplementedError(msg)
        if cuda:
            from ..graph_core.cuda_contracts import validate_cuda_ci_test

            validate_cuda_ci_test(cg.test, "CUDA skeleton")
        self.cg = cg
        self.alpha = alpha
        alpha_test = cg.test
        seen = set()
        while id(alpha_test) not in seen:
            seen.add(id(alpha_test))
            inner = getattr(alpha_test, "_batch_cit", None)
            if inner is None or inner is alpha_test:
                break
            alpha_test = inner
        if not getattr(alpha_test, "_inflight", None):
            configure_fisher_alpha(cg.test, alpha)
        self.stable = stable
        self.n_jobs = effective_n_jobs(n_jobs)
        self.show_progress = show_progress
        self.verbose = verbose
        if parallel_backend not in ("auto", "threads", "processes"):
            msg = "skeleton backend must be 'auto', 'threads', or 'processes'"
            raise ValueError(msg)
        self.device = device
        self._ci_batcher = None
        if cuda:
            assert device is not None
            test = self.cg.test
            assert isinstance(
                test, CIT
            )  # validate_cuda_ci_test verified the built-in implementation.
            method = test.method
            if method == "fisherz":
                from ..graph_core.cpu_cuda_fisher_bridge import CudaFisherZ

                data = getattr(test, "data", None)
                if test._corr_matrix is not None or data is None:
                    corr = test.corr_matrix
                    assert corr is not None
                    self._ci_batcher = CudaFisherZ(
                        corr, test.n_samples, device=device, dtype=dtype
                    )
                else:
                    self._ci_batcher = CudaFisherZ.from_data(
                        data,
                        n_samples=test.n_samples,
                        device=device,
                        dtype=dtype,
                    )
                self._ci_batcher.cpu_test = test
                self._ci_batcher.decision_alpha = alpha
            elif method == "chisq":
                from ..graph_core.cuda_chisq import CudaChisqHost

                self._ci_batcher = CudaChisqHost(test, device=device)
            else:
                msg = f"device={device!r} requires fisherz or chisq CIT, got {method!r}"
                raise ValueError(msg)
        self.parallel_backend = parallel_backend
        self.process_ci_info = process_ci_info
        self.protected_edges = protected_edges
        self._parallel_executor = None
        self.ci_tests_performed = 0

        self.no_of_var = len(self.cg.nodes)
        # We work directly on cg.G.graph
        self.G = self.cg.G.graph
        self.G1 = np.copy(self.G)
        self.max_depth = max_depth

    def _epilogue_from_device(
        self,
        G_t,
        hits,
        *,
        fci: bool = False,
        sepset=None,
        copy_endpoints: bool = False,
        materialize_host_sepsets: bool = False,
    ):
        """One D2H of ``G`` plus compacted sepsets.

        Skeleton-only: write ``Endpoint.TAIL`` wherever device adjacency is
        nonzero. After PC orientation: ``copy_endpoints=True`` copies the
        endpoint matrix as-is. ``materialize_host_sepsets=True`` also expands
        the packed CSR into ``cg.sepset`` so the host orientation fallback can
        read it.
        """
        import torch

        from ..graph_core.cuda_graph_batch import _cuda_step

        with _cuda_step("d2h_G"):
            g = G_t.detach().cpu().numpy()
        with _cuda_step("apply_G"):
            if copy_endpoints:
                np.copyto(self.G, g)
            else:
                self.G.fill(0)
                self.G[g != 0] = Endpoint.TAIL.value
        if isinstance(hits, tuple) and hits and hits[0] == "csr":
            S, ranges = hits[1], hits[2]
            if torch.is_tensor(S):
                S = S.detach().cpu().numpy()
            if torch.is_tensor(ranges):
                ranges = ranges.detach().cpu().numpy()
            with _cuda_step("sepsets"):
                self.cg.sepset_S = np.asarray(S, dtype=np.int64)
                self.cg.sepset_ranges = np.asarray(ranges, dtype=np.int64)
                if materialize_host_sepsets:
                    self._materialize_packed_sepsets(
                        self.cg.sepset_S, self.cg.sepset_ranges
                    )
            return
        with _cuda_step("sepsets"):
            for rx, ry, S_hit in hits:
                for i in range(len(rx)):
                    x, y = int(rx[i]), int(ry[i])
                    S = tuple(int(v) for v in S_hit[i])
                    if fci and sepset is not None:
                        sepset[(x, y)] = S
                        sepset[(y, x)] = S
                    self._append_sepset(x, y, S)

    def _materialize_packed_sepsets(self, S, ranges):
        """Expand packed CSR sepsets into the legacy ``cg.sepset`` lists.

        The host PC orientation reads ``cg.sepset[x, z]`` directly. When the
        CUDA skeleton hands over at ``k > 32``, those pairs were committed to
        the device CSR only, so without this step every collider found in the
        fused depths would be skipped.
        """
        p = int(self.no_of_var)
        S = np.asarray(S, dtype=np.int64)
        ranges = np.asarray(ranges, dtype=np.int64)
        n = int(S.size)
        for x in range(p):
            for y in range(x + 1, p):
                start, end = int(ranges[x, y, 0]), int(ranges[x, y, 1])
                if start < 0 or end < start or end > n:
                    continue
                members = tuple(sorted({int(v) for v in S[start:end]}))
                self.cg.sepset[x, y] = [members]
                self.cg.sepset[y, x] = [members]

    def _resident_complete_G(self):
        """Return the resident ``int8`` complete adjacency (create if missing)."""
        G_t = getattr(self.cg, "_G_cuda", None)
        if G_t is not None:
            return G_t
        from ..graph_core.cuda_graph_batch import complete_undirected

        assert self._ci_batcher is not None
        G_t = complete_undirected(self.no_of_var, self._ci_batcher.device)
        self.cg._G_cuda = G_t
        return G_t

    def _run_cuda_skeleton(self, *, fci: bool = False, to_host: bool = True):
        """Run fused skeleton discovery on the device.

        With ``to_host=False``, keep ``G``, the CSR data and ``pMax`` on the device for
        later PC or FCI phases.
        """
        from ..graph_core.cuda_graph_batch import run_skeleton

        if not self.stable:
            msg = (
                "CUDA skeleton discovery supports only stable=True; "
                "use the CPU backend for stable=False"
            )
            raise ValueError(msg)
        G_t = self._resident_complete_G()
        G1_t = None
        pmax = None
        if fci:
            import torch

            G1_t = G_t.clone()
            p = self.no_of_var
            if not hasattr(self.cg, "pMax") or self.cg.pMax is None:
                pmax = torch.full(
                    (p, p),
                    -float("inf"),
                    dtype=torch.float64,
                    device=G_t.device,
                )
                pmax.fill_diagonal_(1.0)
            else:
                pmax = self.cg.pMax
        max_depth = self.max_depth
        cond_limit = getattr(self.cg, "max_cond_set_size", None)
        if cond_limit is not None:
            max_depth = cond_limit if max_depth is None else min(max_depth, cond_limit)
        assert self._ci_batcher is not None
        hits, pmax_out, fallback = run_skeleton(
            self._ci_batcher,
            G_t,
            self.alpha,
            max_depth=max_depth,
            k_max=self._ci_batcher._k_max,
            fci=fci,
            G1_t=G1_t,
            pmax=pmax,
            to_host=to_host,
        )
        return G_t, hits, pmax_out, fallback

    def _run_cuda_resident(self) -> "CLGeneralGraph":
        """PC-stable-cuda skeleton: fused ``run_skeleton`` then host TAIL epilogue.

        ``G`` stays on device across depths. Fused Fisher-Z covers ``k ≤ 32``.
        ``k = 33..96`` stays on device (fill + ``CudaFisherZ``). ``k > K_MAX``
        continues with the Python combinadic wavefront on ``CudaFisherZ.batch``.
        ``chisq`` uses the same skeleton; tiled k=0/k=1 histograms and
        packed-Z / LSD CSR ``k≥2``.
        """
        G_t, hits, _, fallback = self._run_cuda_skeleton(fci=False, to_host=True)
        self._epilogue_from_device(G_t, hits)
        if fallback is not None:
            pbar = self._init_progress_bar(self.show_progress)
            print(
                f"[cuda skeleton] k={fallback} > k_max; "
                "Python combinadic wavefront on CudaFisherZ.batch",
                file=sys.stderr,
                flush=True,
            )
            depth = fallback - 1
            while self._continue_search(depth):
                depth += 1
                if self.show_progress:
                    self._reset_progress_bar(pbar, depth)
                edge_removal = self._process_depth_cuda_wavefront(depth, pbar)
                if self.stable:
                    self._apply_edge_removals(edge_removal)
            self._close_progress_bar(pbar)
        return self.convert_to_causallearn_graph()

    def _run_fci_cuda_resident(self, *, to_host: bool = True):
        """FCI-stable-cuda skeleton: ``run_skeleton(fci=True)`` plus ``pMax`` / CSR.

        Freezes ``G1`` each depth, independence ``p ≥ α``, union sepsets per
        directed pair mirrored last-writer-wins and packed into ``SepsetStore``
        for ``pc_cons`` / pdsep.
        ``to_host=False`` keeps ``G`` / ``pMax`` on device. Sepsets are still
        packed into a host ``SepsetStore`` so ``pc_cons`` can vote.
        """
        from ..graph_core.cuda_graph_batch import run_skeleton
        from ..graph_core.cuda_pdsep import SepsetStore

        pbar = self._init_progress_bar(self.show_progress)
        G_t, hits, pmax, fallback = self._run_cuda_skeleton(fci=True, to_host=to_host)
        if pmax is not None:
            self.cg.pMax = pmax
        depth = getattr(run_skeleton, "last_depth", -1)
        store = None
        S, ranges = hits[1], hits[2]
        if fallback is not None:
            self._epilogue_from_device(G_t, hits, fci=True)
            assert self.cg.sepset_S is not None
            assert self.cg.sepset_ranges is not None
            store = SepsetStore(self.cg.sepset_S, self.cg.sepset_ranges)
            print(
                f"[cuda skeleton] FCI k={fallback} > k_max; "
                "Python combinadic wavefront on CudaFisherZ.batch",
                file=sys.stderr,
                flush=True,
            )
            sepset = {}
            self.G1 = np.copy(self.G)
            depth = fallback - 1
            while self._continue_search(depth):
                depth += 1
                if self.show_progress:
                    self._reset_progress_bar(pbar, depth)
                edge_removal, sepset = self._process_depth_fci_cuda_wavefront(
                    depth, pbar, sepset
                )
                if self.stable:
                    self._apply_edge_removals(edge_removal)
                self.G1 = np.copy(self.G)
            for (x, y), members in sepset.items():
                if members:
                    store.write(int(x), int(y), members, mirror=False)
            to_host = True
        elif to_host:
            self._epilogue_from_device(G_t, hits, fci=True)
            assert self.cg.sepset_S is not None
            assert self.cg.sepset_ranges is not None
            store = SepsetStore(self.cg.sepset_S, self.cg.sepset_ranges)
            self.G1 = np.copy(self.G)
        else:
            import torch

            S_h = (
                S.detach().cpu().numpy()
                if isinstance(S, torch.Tensor)
                else np.asarray(S)
            )
            R_h = (
                ranges.detach().cpu().numpy()
                if isinstance(ranges, torch.Tensor)
                else np.asarray(ranges)
            )
            store = SepsetStore(S_h, R_h)
        self._close_progress_bar(pbar)
        if to_host:
            S, ranges = store.packed()
            self.cg.sepset_S = S
            self.cg.sepset_ranges = ranges
            return {
                "sk": self.cg.G.graph,
                "pMax": self.cg.pMax,
                "sepset": {},
                "sepset_S": S,
                "sepset_ranges": ranges,
                "sepset_store": store,
                "unfTriples": set(),
                "max_ord": depth - 1,
            }
        return {
            "sk": G_t,
            "pMax": self.cg.pMax,
            "sepset": {},
            "sepset_S": S,
            "sepset_ranges": ranges,
            "sepset_store": store,
            "unfTriples": set(),
            "max_ord": depth - 1,
        }

    def _validate_fci_parallel_config(self):
        if self.n_jobs == 1 or self.parallel_backend != "processes":
            return

        info = self.process_ci_info
        if not isinstance(info, Mapping) or info.get("method") != "fisherz":
            msg = (
                "run_fci with parallel_backend='processes' requires "
                "process_ci_info with method='fisherz'"
            )
            raise ValueError(msg)

        missing = sorted({"corr_matrix", "n_samples"}.difference(info))
        if missing:
            msg = (
                "run_fci with parallel_backend='processes' has incomplete "
                f"process_ci_info; missing required keys: {missing}"
            )
            raise ValueError(msg)

    def run(self):
        """Stage 1: Skeleton discovery."""
        if self._ci_batcher is not None:
            return self._run_cuda_resident()

        _pc_execution_stats(self.cg, self.parallel_backend, self.n_jobs)
        depth = -1
        pbar = self._init_progress_bar(self.show_progress)

        with profiler.time_block(
            "skeleton_discovery_total", tags={"algo": "sequential"}
        ):
            while self._continue_search(depth):
                depth += 1
                if self.show_progress:
                    self._reset_progress_bar(pbar, depth)

                edge_removal = self._process_depth(depth=depth, pbar=pbar)

                if self.stable:
                    self._apply_edge_removals(edge_removal)

                if pbar is not None:
                    pbar.refresh()

        self._close_progress_bar(pbar)

        return self.convert_to_general_graph()

    def run_fci(self):
        """Run FCI skeleton discovery and return the skeleton result dictionary."""
        if self._ci_batcher is not None:
            return self._run_fci_cuda_resident()
        self._validate_fci_parallel_config()
        _record_fci_dispatch(self.cg, self.parallel_backend, self.n_jobs)

        sepset = {}

        pbar = self._init_progress_bar(self.show_progress)

        # Initialize pMax if it doesn't exist
        if not hasattr(self.cg, "pMax"):
            self.cg.pMax = np.full((self.no_of_var, self.no_of_var), -np.inf)
            for i in range(self.no_of_var):
                self.cg.pMax[i, i] = 1

        depth = -1

        if (
            self.n_jobs != 1
            and self.parallel_backend != "auto"
            and not (self.stable and supports_fisher_batch(self.cg.test))
        ):
            if self.parallel_backend == "processes":
                self._parallel_executor = Parallel(
                    n_jobs=self.n_jobs,
                    backend="loky",
                    max_nbytes="1M",
                )
                self._parallel_executor.__enter__()
            else:
                max_workers = self.n_jobs if self.n_jobs > 0 else cpu_count()
                self._parallel_executor = ThreadPoolExecutor(max_workers=max_workers)

        try:
            # while not done and depth <= self.m_max:
            while self._continue_search(depth):
                depth += 1
                if self.show_progress:
                    self._reset_progress_bar(pbar, depth)

                edge_removal, sepset = self._process_depth_fci(
                    depth=depth, pbar=pbar, sepset=sepset
                )

                if self.stable:
                    self._apply_edge_removals(edge_removal)

                self.G1 = np.copy(self.G)

                if pbar is not None:
                    pbar.refresh()
        finally:
            if self._parallel_executor is not None:
                if isinstance(self._parallel_executor, ThreadPoolExecutor):
                    self._parallel_executor.shutdown(wait=True)
                else:
                    self._parallel_executor.__exit__(None, None, None)
                self._parallel_executor = None

        self._close_progress_bar(pbar)

        return {
            "sk": self.cg.G.graph,
            "pMax": self.cg.pMax,
            "sepset": sepset,
            "unfTriples": set(),
            "max_ord": depth - 1,
        }

    def _bulk_allowed(self, depth):
        return supports_fisher_batch(self.cg.test)

    def _process_depth_fci(
        self,
        depth: int,
        pbar,
        sepset,
    ):
        if (
            self.stable
            and self._bulk_allowed(depth)
            and supports_fisher_batch(self.cg.test)
        ):
            if self.n_jobs == 1 or self.parallel_backend in ("auto", "processes"):
                result = self._process_depth_fci_native(depth, pbar, sepset)
                if result is not None:
                    return result
            return self._process_depth_batch(depth, pbar, sepset=sepset)

        # print(f'depth = {depth}')

        if self.n_jobs != 1 and self.parallel_backend != "auto":
            # print(f'Going to parallel analysis with n_jobs = {self.n_jobs}')
            return self._process_depth_parallel_fci(depth, sepset, pbar)

        edge_removal = []

        # Get all edges. G contains Endpoint values. != 0 means edge exists.
        # We only need one direction per pair, but skeleton logic iterates both directions
        # because neighbors depend on x.
        # However, for undirected graph, G[x,y] == G[y,x].
        # We iterate over all edges (x, y) where G[x,y] != 0
        # Snapshot for stable algorithm
        _record_fci_dispatch(
            self.cg,
            self.parallel_backend,
            self.n_jobs,
            "serial",
            depth,
            sum(len(self.cg.neighbors(x)) > depth for x in range(self.no_of_var)),
        )

        for x in range(self.no_of_var):
            if self.show_progress:
                pbar.update()

            # Fast-path guard on the LIVE graph: self.G aliases cg.G.graph, so
            # when stable=False it already reflects removals made earlier in this
            # depth. Inside _process_node_at_depth_fci the edges are filtered on
            # self.G too, but conditioning sets come from the self.G1 snapshot,
            # so this check is not redundant with the len(nbrs) >= depth one there.
            if len(self.cg.neighbors(x)) < depth:
                continue

            removals_for_x, sepset = self._process_node_at_depth_fci(
                x=x,
                depth=depth,
                sepset=sepset,
            )
            if removals_for_x:
                edge_removal.extend(removals_for_x)

        return edge_removal, sepset

    def _process_depth_fci_cuda_wavefront(self, depth: int, pbar, sepset):
        """FCI-stable depth: G1-frozen neighbors, wavefront batch, pMax updates."""
        if depth == 0:
            return self._process_depth_fci_cuda_k0(pbar, sepset)
        active = []
        for x in range(self.no_of_var):
            if self.show_progress:
                pbar.update()
            neigh_x = self.cg.neighbors(x)
            if len(neigh_x) < depth:
                continue
            for y in range(self.no_of_var):
                if x == y or self.G[x, y] == 0:
                    continue
                nbrs = np.where(self.G1[x, :] != 0)[0]
                nbrs = nbrs[nbrs != y]
                if len(nbrs) < depth:
                    continue
                active.append((int(x), int(y), combinations(nbrs, depth)))

        edge_removal = []
        while active:
            xs, ys, Ss, still = [], [], [], []
            for x, y, it in active:
                try:
                    S = next(it)
                except StopIteration:
                    continue
                xs.append(x)
                ys.append(y)
                Ss.append(S)
                still.append((x, y, it))
            if not xs:
                break
            S_arr = np.array([list(S) for S in Ss], dtype=np.int64).reshape(
                len(Ss), depth
            )
            assert self._ci_batcher is not None
            pvals = self._ci_batcher.batch(
                np.asarray(xs, dtype=np.int64), np.asarray(ys, dtype=np.int64), S_arr
            )
            next_active = []
            for i, (x, y, it) in enumerate(still):
                pval = float(pvals[i])
                self.cg.pMax[x, y] = max(self.cg.pMax[x, y], pval)
                if pval >= self.alpha:
                    if not self.stable:
                        self._remove_edge_immediately(x, y)
                        sepset[(x, y)] = set(Ss[i])
                        sepset[(y, x)] = set(Ss[i])
                        self._append_sepset(x, y, Ss[i])
                    else:
                        # stable FAS: keep testing this pair (no early exit) and
                        # union every separating set into both keys
                        self._register_edge_removal(edge_removal, x, y)
                        members = {int(v) for v in Ss[i]}
                        prev = sepset.get((x, y))
                        base = members if prev is None else members | set(prev)
                        sepset[(x, y)] = base
                        sepset[(y, x)] = base
                        self._append_sepset(x, y, Ss[i])
                    if not self.stable:
                        continue
                    next_active.append((x, y, it))
                else:
                    next_active.append((x, y, it))
            active = next_active
        return edge_removal, sepset

    def _process_depth_fci_cuda_k0(self, pbar, sepset):
        """Depth 0: unique ``x < y`` via CUDA ``ldlt_batch`` (k=0 kernel)."""
        if self.show_progress:
            pbar.update(self.no_of_var)
        n = self.no_of_var
        xs, ys = np.triu_indices(n, k=1)
        S = np.zeros((xs.size, 0), dtype=np.int64)
        assert self._ci_batcher is not None
        pvals = self._ci_batcher.batch(xs, ys, S)
        empty = ()
        edge_removal = []
        for i in range(len(pvals)):
            x, y = int(xs[i]), int(ys[i])
            pval = float(pvals[i])
            self.cg.pMax[x, y] = max(self.cg.pMax[x, y], pval)
            self.cg.pMax[y, x] = max(self.cg.pMax[y, x], pval)
            if pval >= self.alpha:
                if not self.stable:
                    self._remove_edge_immediately(x, y)
                else:
                    self._register_edge_removal(edge_removal, x, y)
                sepset[(x, y)] = empty
                sepset[(y, x)] = empty
                self._append_sepset(x, y, empty)
        return edge_removal, sepset

    def _process_node_at_depth_fci(
        self,
        x: int,
        depth: int,
        sepset,
    ):
        edge_removal = []
        for y in range(self.no_of_var):
            if x == y:
                continue
            if self.G[x, y] == 0:
                continue

            # Neighbors of x in G1
            # In undirected graph check != 0
            nbrs = np.where(self.G1[x, :] != 0)[0]
            nbrs = nbrs[nbrs != y]
            # nbrs = self.cg.neighbors(x)

            if len(nbrs) >= depth:
                for S in combinations(nbrs, depth):
                    pval = self.cg.ci_test(x, y, S)
                    self.cg.pMax[x, y] = max(self.cg.pMax[x, y], pval)

                    if pval >= self.alpha:
                        # Remove edge (set to NULL=0) now, or at the end of the
                        # depth when running stable.
                        if not self.stable:
                            self._remove_edge_immediately(x, y)
                        else:
                            self._register_edge_removal(edge_removal, x, y)

                        self._record_sepset(sepset, x, y, S)
                        if not self.stable:
                            break
        return edge_removal, sepset

    def convert_to_causallearn_graph(self) -> "CLGeneralGraph":
        """Return the current graph in causal-learn's matrix encoding."""
        from ..graph_core.cuda_graph_batch import _cuda_step

        with _cuda_step("cl_graph"):
            return causal_graph_to_causallearn(self.cg)

    def convert_to_general_graph(self):
        """Return the local graph maintained by the algorithm."""
        return self.cg.G

    def _init_progress_bar(self, show_progress: bool):
        return tqdm(total=self.no_of_var) if show_progress else None

    def _reset_progress_bar(self, pbar, depth: int):
        pbar.reset()
        pbar.set_description(f"Depth={depth}, max_degree={self.cg.max_degree()}")

    def _close_progress_bar(self, pbar):
        if self.show_progress and pbar is not None:
            pbar.close()

    def _continue_search(self, depth: int) -> bool:
        """Continue the outer depth loop while max_degree(cg) - 1 exceeds depth."""
        if self.max_depth is not None and depth >= self.max_depth:
            return False
        return bounded_size(self.cg, self.cg.max_degree() - 1) > depth

    def _process_depth(
        self,
        depth: int,
        pbar,
    ):
        """Process all variables at this depth and return the edges to remove in stable mode."""
        if (
            self.stable
            and self._bulk_allowed(depth)
            and not self.protected_edges
            and supports_fisher_batch(self.cg.test)
        ):
            if (
                (self.n_jobs > 1 or NATIVE_AVAILABLE)
                and type(self.cg.test) is CIT
                and self.parallel_backend in ("auto", "processes")
            ):
                degrees = np.count_nonzero(self.G, axis=1)
                estimated = sum(
                    int(degree) * math.comb(int(degree) - 1, depth)
                    for degree in degrees
                    if degree > depth
                )
                if NATIVE_AVAILABLE or estimated >= 1024:
                    result = self._process_depth_pc_processes(depth, pbar)
                    if result is not None:
                        return result
            # Record the pass with the backend the numeric batches actually used.
            before = batch_execution_stats(self.cg.test)["batches"]
            edge_removal = self._process_depth_batch(depth, pbar)
            after = batch_execution_stats(self.cg.test)["batches"]
            used = [
                name
                for name in ("processes", "threads", "serial")
                if after.get(name, 0) > before.get(name, 0)
            ]
            _record_pc_dispatch(
                self.cg,
                self.parallel_backend,
                self.n_jobs,
                used[0] if used else "serial",
                depth,
                sum(len(self.cg.neighbors(x)) > depth for x in range(self.no_of_var)),
            )
            return edge_removal

        if self.n_jobs != 1 and self.parallel_backend != "auto":
            # print(f'Going to parallel analysis with n_jobs = {self.n_jobs}')
            return self._process_depth_parallel(depth, pbar)

        if depth == 0 and self.stable:
            return self._process_depth_cpu_k0(pbar)

        edge_removal = []

        _record_pc_dispatch(
            self.cg,
            self.parallel_backend,
            self.n_jobs,
            "serial",
            depth,
            sum(len(self.cg.neighbors(x)) > depth for x in range(self.no_of_var)),
        )

        for x in range(self.no_of_var):
            if self.show_progress:
                pbar.update()

            neigh_x = self.cg.neighbors(x)
            if len(neigh_x) < depth:
                continue

            removals_for_x = self._process_node_at_depth(
                x=x,
                neigh_x=neigh_x,
                depth=depth,
            )
            edge_removal.extend(removals_for_x)

        return edge_removal

    def _process_depth_pc_processes(self, depth, pbar):
        """Move PC enumeration, numeric work and reduction into coarse jobs."""
        cit = self.cg.test
        from ..graph_core.independence_tests import _BUILTIN_CIT_BATCH

        if (
            not isinstance(cit, CIT)
            or getattr(cit.batch, "__func__", None) is not _BUILTIN_CIT_BATCH
        ):
            # A public batch override may instrument or replace CI evaluation.
            return None
        if (
            (cit.max_cond_set_size is not None and depth > cit.max_cond_set_size)
            or self.no_of_var != cit.data.shape[1]
            or depth >= cit.n_samples - 3
        ):
            # Keep CIT's public conditioning/index validation authoritative
            # even for direct SkeletonDiscovery callers without graph limits.
            return None
        build_started = perf_counter()
        neighbors = [tuple(map(int, np.flatnonzero(row))) for row in self.G]
        execution = _pc_execution_stats(self.cg, self.parallel_backend, self.n_jobs)
        work = sum(
            len(row) * math.comb(len(row) - 1, depth)
            for row in neighbors
            if len(row) > depth
        )
        dimension_factor = (depth + 2) ** 3 / 8
        plan = plan_process_work(
            work * dimension_factor,
            self.n_jobs,
            family=f"pc_skeleton_{'cpp' if NATIVE_AVAILABLE else 'python'}_d{depth}",
            backend=self.parallel_backend,
            min_work=1024 * dimension_factor,
            tasks=sum(len(row) for row in neighbors) // 2,
            stats=execution,
        )
        jobs = plan.jobs
        chunks = _pc_weighted_edge_chunks(neighbors, depth, max(1, jobs * 2))
        if jobs == 1:
            chunks = (
                [sorted(pair for chunk in chunks for pair in chunk)] if chunks else []
            )
        if not chunks:
            return []
        with cit._lock:
            # A concurrently used CIT needs its normal single-flight path.
            if cit._inflight:
                return None
            cache_size = cit.cache_size
            seeds = [{} for _ in chunks]
            if cache_size:
                pair_owner = {
                    pair: index for index, chunk in enumerate(chunks) for pair in chunk
                }
                for key, value in cit._cache.items():
                    if len(key[2]) == depth:
                        owner = pair_owner.get(key[:2])
                        if owner is not None:
                            seeds[owner][key] = value
        execution = _record_pc_dispatch(
            self.cg,
            self.parallel_backend,
            self.n_jobs,
            "processes" if jobs > 1 else "serial",
            depth,
            len(chunks),
        )
        execution["task_build_seconds"] += perf_counter() - build_started
        execution["cache_seed_entries"] += sum(map(len, seeds))
        results = process_map(
            _pc_fisher_edge_chunk,
            (
                (
                    chunk,
                    neighbors,
                    cit.corr_matrix,
                    cit.n_samples,
                    depth,
                    self.alpha,
                    cache_size,
                    seed,
                )
                for chunk, seed in zip(chunks, seeds, strict=False)
            ),
            n_jobs=jobs,
            plan=plan,
            shared_arg_indices=(2,),
        )
        _record_native_chunks(execution, results)
        edge_removal = []
        errors = []
        computations = cache_hits = 0
        publication_started = perf_counter()
        with cit._lock:
            for _reductions, misses, hits, updates, failures, discarded in results:
                execution["cache_updates_returned"] += len(updates)
                computations += misses
                cache_hits += hits
                errors.extend(failures)
                if cit.cache_size:
                    # Values before a chunk's retained cache tail would have
                    # been evicted within that chunk. Count those virtual
                    # insertions without transmitting the discarded entries.
                    cit.cache_evictions += discarded
                    for key, value in updates:
                        cit._cache[key] = value
                        cit._cache.move_to_end(key)
                        if len(cit._cache) > cit.cache_size:
                            cit._cache.popitem(last=False)
                            cit.cache_evictions += 1
            cit.cache_misses += computations
            cit.cache_hits += cache_hits
        execution["cache_publication_seconds"] += perf_counter() - publication_started
        if errors:
            # Error batches stop early rather than enumerating the remaining
            # combinatorial space. Replay the coordinator's ordered stream to
            # expose the same first visited error as the normal CI path.
            return self._process_depth_batch(depth, pbar)
        for reductions, _, _, _, _, _ in results:
            for x, y, union in reductions:
                self._register_edge_removal(edge_removal, x, y)
                self._merge_sepset_union(x, y, union)
        if self.show_progress and pbar is not None:
            pbar.update(self.no_of_var)
        stats = self.cg.G.metadata.setdefault(
            "pc_process_batches",
            {
                "depths": [],
                "computations": 0,
                "cache_hits": 0,
            },
        )
        stats["depths"].append(depth)
        stats["computations"] += computations
        stats["cache_hits"] += cache_hits
        return edge_removal

    def _process_depth_fci_native(self, depth, pbar, sepset):
        """Use one compact pair kernel in both serial and process execution."""
        test = self.cg.test
        context = getattr(test, "_pdsep_batch_context", None)
        if type(test) is CIT:
            cit, outer, stats, phase, lock = test, None, None, None, None
        elif context is not None:
            cit, outer, stats, phase, lock = context
        else:
            return None
        from ..graph_core.independence_tests import _BUILTIN_CIT_BATCH

        if (
            not isinstance(cit, CIT)
            or getattr(cit.batch, "__func__", None) is not _BUILTIN_CIT_BATCH
        ):
            return None
        if (
            self.no_of_var != cit.data.shape[1]
            or depth >= cit.n_samples - 3
            or (cit.max_cond_set_size is not None and depth > cit.max_cond_set_size)
        ):
            return None
        adjacency = [tuple(map(int, np.flatnonzero(row))) for row in self.G]
        reference = [tuple(map(int, np.flatnonzero(row))) for row in self.G1]
        retain = max(cit.cache_size, getattr(outer, "max_entries", 0))
        cache_enabled = bool(retain)
        chunks, work = pair_chunks(
            adjacency, reference, depth, max(1, self.n_jobs * 2), cache_enabled
        )
        if not chunks:
            return [], sepset
        # The whole search, rather than small matrix calls, is the process task.
        execution = _record_fci_dispatch(self.cg, self.parallel_backend, self.n_jobs)
        dimension_factor = (depth + 2) ** 3 / 8
        plan = plan_process_work(
            work * dimension_factor,
            self.n_jobs,
            family=f"fci_skeleton_{'cpp' if NATIVE_AVAILABLE else 'python'}_d{depth}",
            backend=self.parallel_backend,
            min_work=512 * dimension_factor,
            tasks=sum(map(len, chunks)),
            stats=execution,
        )
        jobs = plan.jobs
        if jobs == 1:
            chunks = [sorted(pair for chunk in chunks for pair in chunk)]
        elif jobs != self.n_jobs:
            chunks, _ = pair_chunks(
                adjacency, reference, depth, jobs * 2, cache_enabled
            )
        outer_enabled = outer is not None and getattr(outer, "max_entries", 0) > 0
        with lock if lock is not None else nullcontext(), cit._lock:
            if cit._inflight:
                return None
            seeds = [{} for _ in chunks]
            owner = {pair: i for i, chunk in enumerate(chunks) for pair in chunk}
            for key, value in cit._cache.items():
                if len(key[2]) == depth and key[:2] in owner:
                    seeds[owner[key[:2]]][key] = (value, False)
            if outer_enabled and outer is not None:
                for key, value in outer.items():
                    if len(key[2]) == depth and key[:2] in owner:
                        seeds[owner[key[:2]]][key] = (value, True)
        results = process_map(
            fisher_level_chunk,
            (
                (
                    chunk,
                    reference,
                    cit.corr_matrix,
                    cit.n_samples,
                    depth,
                    self.alpha,
                    seed,
                    cache_enabled,
                    outer_enabled,
                    retain,
                    True,
                )
                for chunk, seed in zip(chunks, seeds, strict=False)
            ),
            n_jobs=jobs,
            plan=plan,
            shared_arg_indices=(2,),
        )
        execution = _record_fci_dispatch(
            self.cg,
            self.parallel_backend,
            self.n_jobs,
            "processes" if jobs > 1 else "serial",
            depth,
            len(chunks),
        )
        _record_native_chunks(execution, results)
        if any(result["errors"] for result in results):
            # Preserve the first scalar exception and validation path. Do not
            # publish speculative graph edits from a failed exhaustive level.
            abandoned = sum(result["computed"] for result in results)
            execution["error_replay_computations"] = (
                execution.get("error_replay_computations", 0) + abandoned
            )
            with cit._lock:
                cit.cache_misses += abandoned
            return self._process_depth_batch(depth, pbar, sepset=sepset)
        edge_removal = []
        with lock if lock is not None else nullcontext(), cit._lock:
            for result in results:
                self.ci_tests_performed += result["requests"]
                cit.cache_misses += result["computed"]
                cit.cache_hits += result["cit_hits"]
                if stats is not None and phase is not None:
                    stats["total"] += result["requests"]
                    stats[phase["name"]] += result["requests"]
                    if outer is not None:
                        stats["ci_cache_hits"] += result["outer_hits"]
                        stats["ci_cache_misses"] += (
                            result["requests"] - result["outer_hits"]
                        )
                if jobs == 1 and result["numeric"]:
                    if not hasattr(cit, "_batch_execution_stats"):
                        cit._batch_execution_stats = {}
                    _merge_execution_stats(
                        cit._batch_execution_stats, result["numeric"]
                    )
                if cit.cache_size:
                    cit.cache_evictions += result["discarded"]
                for key, value in result["updates"]:
                    if cit.cache_size:
                        cit._cache[key] = value
                        cit._cache.move_to_end(key)
                        if len(cit._cache) > cit.cache_size:
                            cit._cache.popitem(last=False)
                            cit.cache_evictions += 1
                if outer_enabled and outer is not None:
                    # The worker reserves these events before numeric
                    # flushes, preserving miss/promotion interleaving.
                    # Dropped events are new outer-cache writes, not hits
                    # to entries already present in that worker's seed.
                    outer.evictions += result["outer_discarded"]
                    for key, value in result["outer_updates"]:
                        outer[key] = value
        for result in results:
            for a, b, maxima, separators, _union in result["pairs"]:
                self.cg.pMax[a, b] = max(self.cg.pMax[a, b], maxima[0])
                self.cg.pMax[b, a] = max(self.cg.pMax[b, a], maxima[1])
                if separators:
                    self._register_edge_removal(edge_removal, a, b)
                    self._record_fci_pair_evidence(sepset, a, b, separators, _union)
        if self.show_progress and pbar is not None:
            pbar.update(self.no_of_var)
        return edge_removal, sepset

    @overload
    def _process_depth_batch(
        self, depth, pbar, sepset: None = None
    ) -> list[tuple[int, int]]: ...

    @overload
    def _process_depth_batch(
        self, depth, pbar, sepset: dict
    ) -> tuple[list[tuple[int, int]], dict]: ...

    def _process_depth_batch(self, depth, pbar, sepset=None):
        """Stream a stable depth through native CI batches, then merge in order.

        The coordinator owns the CI cache, avoiding reversed-pair duplicates
        and serialization of a growing CIT/graph for every process task.
        """
        fci = sepset is not None
        adjacency = [tuple(map(int, np.flatnonzero(row))) for row in self.G]
        reference = (
            [tuple(map(int, np.flatnonzero(row))) for row in self.G1]
            if fci
            else adjacency
        )
        edge_removal = []
        separating_members = {}
        requests = []
        window_size = max(512, min(8192, self.n_jobs * 1024))

        def flush():
            values = evaluate_ci_batch(
                self.cg.test,
                requests,
                n_jobs=self.n_jobs,
                backend=self.parallel_backend,
                batch_size=256,
            )
            if fci:
                self.ci_tests_performed += len(requests)
            for (x, y, conditions), pval in zip(requests, values, strict=False):
                if isinstance(pval, Exception):
                    raise pval
                if fci and self.cg.pMax[x, y] < pval:
                    self.cg.pMax[x, y] = pval
                if pval >= self.alpha if fci else pval > self.alpha:
                    if fci:
                        self._register_edge_removal(edge_removal, x, y)
                        self._record_sepset(sepset, x, y, conditions)
                    else:
                        key = (x, y)
                        if key not in separating_members:
                            separating_members[key] = set()
                            self._register_edge_removal(edge_removal, x, y)
                        separating_members[key].update(conditions)
            requests.clear()

        for x, neighbors in enumerate(adjacency):
            if len(neighbors) < depth:
                continue
            for y in neighbors:
                candidates = tuple(v for v in reference[x] if v != y)
                for conditions in combinations(candidates, depth):
                    requests.append((x, y, conditions))
                    if len(requests) >= window_size:
                        flush()
            if self.show_progress and pbar is not None:
                pbar.update()
        if requests:
            flush()
        if not fci:
            for (x, y), members in separating_members.items():
                self._merge_sepset_union(x, y, members)
        return (edge_removal, sepset) if fci else edge_removal

    def _process_depth_cpu_k0(self, pbar):
        """PC-stable depth 0 on CPU: generate directed pairs, then graph updates."""
        empty = ()
        pairs = []
        for x in range(self.no_of_var):
            if self.show_progress:
                pbar.update()
            neigh_x = self.cg.neighbors(x)
            for y in neigh_x:
                pairs.append((int(x), int(y)))

        pvals = [self.cg.ci_test(x, y, empty) for x, y in pairs]
        edge_removal = []
        for (x, y), p_val in zip(pairs, pvals, strict=False):
            if p_val > self.alpha:
                self._register_edge_removal(edge_removal, x, y)
                self._append_sepset(x, y, empty)
        return edge_removal

    def _process_depth_cuda_wavefront(self, depth: int, pbar):
        """PC-stable depth with one-S-per-pair host wavefront (k > K_MAX)."""
        neigh = [self.cg.neighbors(x) for x in range(self.no_of_var)]
        active = []
        for x in range(self.no_of_var):
            nx = neigh[x]
            if len(nx) < depth:
                continue
            for y in nx:
                cand = np.delete(nx, np.where(nx == y))
                if len(cand) < depth:
                    continue
                active.append((int(x), int(y), combinations(cand, depth)))
        assert self._ci_batcher is not None
        return self._wavefront_pc(active, depth, self._ci_batcher.batch)

    def _wavefront_pc(self, active, depth, batch_fn):
        edge_removal = []
        while active:
            xs, ys, Ss, still = [], [], [], []
            for x, y, it in active:
                try:
                    S = next(it)
                except StopIteration:
                    continue
                xs.append(x)
                ys.append(y)
                Ss.append(S)
                still.append((x, y, it))
            if not xs:
                break
            S_arr = np.array([list(S) for S in Ss], dtype=np.int64).reshape(
                len(Ss), depth
            )
            pvals = batch_fn(
                np.asarray(xs, dtype=np.int64),
                np.asarray(ys, dtype=np.int64),
                S_arr,
            )
            next_active = []
            for i, (x, y, it) in enumerate(still):
                if pvals[i] > self.alpha:
                    if not self.stable:
                        self._remove_edge_immediately(x, y)
                    else:
                        self._register_edge_removal(edge_removal, x, y)
                    self._append_sepset(x, y, Ss[i])
                else:
                    next_active.append((x, y, it))
            active = next_active
        return edge_removal

    def _process_depth_parallel_fci(self, depth, sepset, pbar):
        """Parallel version of _process_depth for FCI algo."""
        if self.verbose:
            # Every directed neighbour pair at a node has degree - 1 possible
            # conditioning variables. Count the row once instead of scanning
            # it again for every neighbour, and skip diagnostics in quiet runs.
            degrees = np.count_nonzero(self.G, axis=1)
            total_ops_estimate = sum(
                int(degree) * math.comb(int(degree) - 1, depth)
                for degree in degrees
                if degree > depth
            )
            print(f"Depth {depth}: Estimated max CI tests: {total_ops_estimate}")
            print(f"Jobs in parallel: {self.n_jobs}")
        if (
            self.parallel_backend == "processes"
            and self.process_ci_info is not None
            and self.process_ci_info.get("method") == "fisherz"
        ):
            effective_n_jobs = self.n_jobs if self.n_jobs > 0 else cpu_count()
            edges = [
                (x, y)
                for x in range(self.no_of_var)
                for y in range(x + 1, self.no_of_var)
                if self.G[x, y] != 0 or self.G[y, x] != 0
            ]
            edge_chunks = _balanced_edge_chunks(edges, self.G1, depth, effective_n_jobs)
            assert not isinstance(self._parallel_executor, ThreadPoolExecutor)
            executor = self._parallel_executor or Parallel(
                n_jobs=effective_n_jobs,
                backend="loky",
                max_nbytes="1M",
            )
            _record_fci_dispatch(
                self.cg,
                self.parallel_backend,
                self.n_jobs,
                "processes",
                depth,
                len(edge_chunks),
            )
            results_list = executor(
                delayed(_skeleton_worker_fci_fisherz_edge_chunk)(
                    edge_chunk,
                    self.G1,
                    self.cg.pMax,
                    self.process_ci_info["corr_matrix"],
                    self.process_ci_info["n_samples"],
                    depth,
                    self.alpha,
                    self.stable,
                )
                for edge_chunk in edge_chunks
            )
        else:
            owns_executor = self._parallel_executor is None
            assert self._parallel_executor is None or isinstance(
                self._parallel_executor, ThreadPoolExecutor
            )
            executor = self._parallel_executor or ThreadPoolExecutor(
                max_workers=self.n_jobs if self.n_jobs > 0 else cpu_count()
            )
            _record_fci_dispatch(
                self.cg,
                self.parallel_backend,
                self.n_jobs,
                "threads",
                depth,
                self.no_of_var,
            )
            try:
                futures = [
                    executor.submit(
                        _skeleton_worker_fci,
                        x,
                        self.cg,
                        self.G1,
                        depth,
                        self.alpha,
                        self.cg.pMax[x],
                        self.stable,
                    )
                    for x in range(self.no_of_var)
                ]
                results_list = [future.result() for future in futures]
            finally:
                if owns_executor:
                    executor.shutdown(wait=True)

        if self.show_progress and pbar is not None:
            pbar.update(self.no_of_var)

        aggregated_edge_removals = []

        total_tests_actual = 0
        for edge_removals, sepset_updates, pmax_updates, tests_count in results_list:
            total_tests_actual += tests_count

            # Apply pMax updates
            for x, y, pval in pmax_updates:
                self.cg.pMax[x, y] = max(self.cg.pMax[x, y], pval)

            # Apply edge removals
            for x, y in edge_removals:
                if self.stable:
                    self._register_edge_removal(aggregated_edge_removals, x, y)
                else:
                    self._remove_edge_immediately(x, y)

            # Apply sepset updates
            for x, y, S in sepset_updates:
                self._record_sepset(sepset, x, y, S)

        if self.verbose:
            print(f"Depth {depth}: Actual CI tests performed: {total_tests_actual}")
        self.ci_tests_performed += total_tests_actual
        return aggregated_edge_removals, sepset

    def _process_depth_parallel(self, depth, pbar):
        """
        Process a PC depth in parallel and merge separating-member unions.

        Parameters
        ----------
        depth : int
            Size of the conditioning sets.
        pbar : progress bar or None
            Optional progress bar.

        Returns
        -------
        list
            Registered edge removals for stable mode.
        """
        effective_n_jobs = self.n_jobs if self.n_jobs > 0 else cpu_count()

        _record_pc_dispatch(
            self.cg,
            self.parallel_backend,
            self.n_jobs,
            "processes" if self.parallel_backend == "processes" else "threads",
            depth,
            self.no_of_var,
        )

        # Run parallel jobs for each variable x
        if self.parallel_backend == "processes":
            assert not isinstance(self._parallel_executor, ThreadPoolExecutor)
            executor = self._parallel_executor or Parallel(
                n_jobs=effective_n_jobs,
                backend="loky",
            )
            results_list = executor(
                delayed(_skeleton_worker)(
                    x, self.cg, depth, self.alpha, self.stable, self.protected_edges
                )
                for x in range(self.no_of_var)
            )
        else:
            owns_executor = self._parallel_executor is None
            assert self._parallel_executor is None or isinstance(
                self._parallel_executor, ThreadPoolExecutor
            )
            executor = self._parallel_executor or ThreadPoolExecutor(
                max_workers=effective_n_jobs
            )
            try:
                futures = [
                    executor.submit(
                        _skeleton_worker,
                        x,
                        self.cg,
                        depth,
                        self.alpha,
                        self.stable,
                        self.protected_edges,
                    )
                    for x in range(self.no_of_var)
                ]
                results_list = [future.result() for future in futures]
            finally:
                if owns_executor:
                    executor.shutdown(wait=True)

        # Update progress bar
        if self.show_progress and pbar is not None:
            pbar.update(self.no_of_var)

        # Aggregate results
        aggregated_edge_removals = []

        for edge_removals, sepsets in results_list:
            for x, y in edge_removals:
                if self.stable:
                    self._register_edge_removal(aggregated_edge_removals, x, y)
                else:
                    self._remove_edge_immediately(x, y)

            for x, y, S in sepsets:
                self._merge_sepset_union(x, y, S)

        return aggregated_edge_removals

    def _process_node_at_depth(
        self,
        x: int,
        neigh_x: np.ndarray,
        depth: int,
    ):
        """
        Process one node x at the given depth.

        For every neighbour y, the union of separating-set members at the
        current depth is collected (stable mode) and the (x, y) edge removal
        is registered once if at least one separating set exists.

        Parameters
        ----------
        x : int
            Node index to process.
        neigh_x : numpy.ndarray
            Stable snapshot of the neighbours of x.
        depth : int
            Size of the conditioning sets.

        Returns
        -------
        list
            Local edge_removal list (used only when stable=True).
        """
        local_edge_removal = []

        for y in neigh_x:
            # IMPORTANT: do not skip a pair just because we already added (x, y)
            # to edge_removal; needed to union all separating members in stable mode.
            neigh_x_noy = np.delete(neigh_x, np.where(neigh_x == y))

            members = self._find_separating_union(x, y, neigh_x_noy, depth)

            if members is not None:
                if not self.stable:
                    self._remove_edge_immediately(x, y)
                else:
                    self._register_edge_removal(local_edge_removal, x, y)

                self._merge_sepset_union(x, y, members)

        return local_edge_removal

    def _find_separating_union(
        self,
        x,
        y,
        candidates,
        depth,
    ) -> tuple | None:
        """Find the union of separating-set members for a pair at one depth.

        Stable mode tests every conditioning set. Non-stable mode and
        protected pairs stop at the first separating set.

        Parameters
        ----------
        x : int
            First node of the pair.
        y : int
            Second node of the pair.
        candidates : iterable of int
            Neighbours of x excluding y.
        depth : int
            Size of the conditioning sets.

        Returns
        -------
        tuple | None
            Sorted Python int members, or None if no set
            separates the pair. An empty tuple denotes depth-zero separation.
        """
        found = False
        members = set()
        for S in combinations(candidates, depth):
            p_val = self.cg.ci_test(x, y, S)
            if p_val > self.alpha:
                found = True
                members.update(int(v) for v in S)
                if not self.stable or self._is_protected(x, y):
                    break
        return tuple(sorted(members)) if found else None

    def _merge_sepset_union(
        self,
        x,
        y,
        members,
    ):
        """Merge PC separating members into one tuple in both directions.

        Protected pairs keep their edge and receive no separating-set entry.

        Parameters
        ----------
        x : int
            First node index.
        y : int
            Second node index.
        members : iterable of int
            Separating-set members to merge as Python integers.

        Returns
        -------
        None
            Updates both symmetric cells in place, preserving empty sets.
        """
        if self._is_protected(x, y):
            return
        members = {int(v) for v in members}
        for i, j in ((x, y), (y, x)):
            current = self.cg.sepset[i, j]
            union = members if current is None else members.union(current[0])
            self.cg.sepset[i, j] = [tuple(sorted(union))]

    def _is_protected(self, x: int, y: int) -> bool:
        """Check whether background knowledge requires the edge to survive the search.

        A protected pair is excluded from BOTH halves of a deletion: the edge
        itself and the separating set that justified deleting it. Recording the
        sepset of an edge that stays is not harmless bookkeeping — collider
        orientation later asks whether the middle node is in the separating set
        of the outer pair, so a sepset left behind for a surviving edge silently
        changes the orientation of triples around it.
        """
        return frozenset((int(x), int(y))) in self.protected_edges

    def _remove_edge_immediately(self, x: int, y: int):
        """Remove edge between x and y from cg.G if it exists."""
        if self._is_protected(x, y):
            return
        edge = self.cg.G.get_edge(self.cg.nodes[x], self.cg.nodes[y])
        if edge:
            self.cg.G.remove_edge(edge)

    def _register_edge_removal(self, edge_removal_acc: list, x: int, y: int):
        """Register (x, y) and (y, x) for later removal in stable mode."""
        if self._is_protected(x, y):
            return
        edge_removal_acc.append((x, y))
        edge_removal_acc.append((y, x))

    def _apply_edge_removals(self, edge_removal: list[tuple[int, int]]):
        """Apply all edge removals collected in stable mode after finishing one depth."""
        from ..graph_core.general_graph import GeneralGraph

        if type(self.cg.G) is GeneralGraph:
            # GeneralGraph.remove_edge only clears these two matrix entries.
            # Avoid constructing Edge/Endpoint objects for a whole stable level.
            pairs = (
                edge_removal
                if not self.protected_edges
                else [(x, y) for x, y in edge_removal if not self._is_protected(x, y)]
            )
            if pairs:
                left, right = np.asarray(pairs, dtype=np.intp).T
                self.cg.G.graph[left, right] = 0
                self.cg.G.graph[right, left] = 0
            return
        for x, y in {(min(x, y), max(x, y)) for x, y in edge_removal}:
            self._remove_edge_immediately(x, y)

    def _record_sepset(self, sepset, x, y, S):
        """Single entry point for the flat `sepset` map.

        Both storage modes record a set of node indices. ``stable`` selects the
        mode independently of ``n_jobs``. ``_append_sepset`` retains the full
        list of separating sets for each pair in ``cg.sepset``.

        Under ``stable=True`` (what every algorithm entry point uses) skeleton
        and sepsets are identical for any `n_jobs`. Under ``stable=False`` edges
        disappear mid-sweep, so which separating set is reached still depends on
        traversal order — that order dependence is what `stable` exists to
        remove, and it is present in causal-learn for the same reason.
        """
        if self._is_protected(x, y):
            return
        members = {int(v) for v in S}
        actual = tuple(sorted(members))
        for key in ((x, y), (y, x)):
            old = self.cg.separators.get(key)
            if old is None or (len(actual), actual) < (len(old), tuple(sorted(old))):
                self.cg.separators[key] = set(actual)
        if self.stable:
            sepset[(x, y)] = set(sepset.get((x, y), set())) | members
            sepset[(y, x)] = set(sepset.get((y, x), set())) | members
        else:
            sepset[(x, y)] = set(members)
            sepset[(y, x)] = set(members)
        self._append_sepset(x, y, S)
        self.cg.separator_members[(x, y)] = set(sepset[(x, y)])
        self.cg.separator_members[(y, x)] = set(sepset[(y, x)])

    def _record_fci_pair_evidence(self, sepset, x, y, separators, members):
        """Publish a complete stable pair reduction without repeated unions."""
        if self._is_protected(x, y):
            return
        actual = min(separators, key=lambda values: (len(values), values))
        for key in ((x, y), (y, x)):
            old = self.cg.separators.get(key)
            if old is None or (len(actual), actual) < (len(old), tuple(sorted(old))):
                self.cg.separators[key] = set(actual)
            union = set(sepset.get(key, ()))
            union.update(members)
            sepset[key] = union
            self.cg.separator_members[key] = set(union)
            current = self.cg.sepset.item(*key)
            if current is None:
                self.cg.sepset[key] = list(dict.fromkeys(separators))
            else:
                for condition in separators:
                    if condition not in current:
                        current.append(condition)

    # Store each separating set only once for a pair.
    def _append_sepset(self, x, y, S):
        if self._is_protected(x, y):
            return
        if self.cg.sepset[x, y] is None:
            self.cg.sepset[x, y] = [S]
        elif S not in self.cg.sepset[x, y]:
            self.cg.sepset.item(x, y).append(S)
        if self.cg.sepset[y, x] is None:
            self.cg.sepset[y, x] = [S]
        elif S not in self.cg.sepset[y, x]:
            self.cg.sepset.item(y, x).append(S)


# Shared helpers retained for the branch's standalone public PC entry point.

_GRAPH_CI_TEST = CausalGraph.ci_test

_FISHER_KERNEL = fisher_z_from_corr

_SKELETON_BATCH = 1024

_PARALLEL_MIN_QUERIES = 8192


def _depth_work(graph, depth):
    degrees = np.count_nonzero(graph, axis=1)
    return sum(
        int(degree) * math.comb(int(degree) - 1, depth)
        for degree in degrees
        if degree and degree - 1 >= depth
    )


def _kernel_work(queries, depth, test=None, backend="threads"):
    """Relative work estimate; policy units, not predicted elapsed seconds.

    Low-order batches cost less than deeper inverses. Historical cache reuse
    discounts arithmetic conservatively; process startup has a higher floor.
    No queries or conditioning orders are removed by this estimate.
    """
    hits, misses = getattr(test, "cache_hits", 0), getattr(test, "cache_misses", 0)
    fraction = max(0.25, misses / (hits + misses)) if hits + misses else 1.0
    return (
        queries
        * ((depth + 2) ** 3 / 64)
        * fraction
        / (4 if backend == "processes" else 1)
    )


def _record_execution(
    cg, phase, depth, work, requested, effective, backend, kernel_work=None
):
    stats = getattr(cg, "execution_stats", None)
    if stats is None:
        stats = cg.execution_stats = {"passes": []}
    stats["passes"].append(
        {
            "phase": phase,
            "depth": int(depth),
            "estimated_queries": int(work),
            "requested_workers": int(requested),
            "effective_workers": int(effective),
            "backend": backend,
            "estimated_kernel_work": kernel_work,
        }
    )


def _batch_graph(cg):
    return getattr(cg.ci_test, "__func__", None) is _GRAPH_CI_TEST and (
        _builtin_fisherz(cg.test) or is_counted_fisherz(cg.test)
    )


def _test_values(test, ci_call, x, y, subsets, alpha, allow_batch=False):
    """Preserve query order and custom CI calls; batch only built-in Fisher-Z."""
    if allow_batch and _builtin_fisherz(test):
        iterator = iter(subsets)
        while chunk := list(islice(iterator, _SKELETON_BATCH)):
            values = (
                test.batch((x, y, conditions) for conditions in chunk)
                if len(chunk) >= 8
                else [ci_call(x, y, conditions) for conditions in chunk]
            )
            for conditions, value in zip(chunk, values, strict=False):
                yield (
                    conditions,
                    _threshold_pvalue(test, x, y, conditions, value, alpha),
                )
    else:
        for conditions in subsets:
            value = ci_call(x, y, conditions)
            yield conditions, _threshold_pvalue(test, x, y, conditions, value, alpha)


def _ci_values(cg, x, y, subsets, alpha, allow_batch=False):
    # An overridden graph method is itself a custom test, even if cg.test is CIT.
    builtin_graph = getattr(cg.ci_test, "__func__", None) is _GRAPH_CI_TEST
    if not builtin_graph:
        for conditions in subsets:
            yield conditions, cg.ci_test(x, y, conditions)
        return
    # The graph may impose a stricter conditioning budget than its CIT.
    from ..graph_core.conditioning import conditioning_allowed

    iterator = iter(subsets)
    for conditions in iterator:
        if not conditioning_allowed(cg, conditions):
            cg.ci_test(x, y, conditions)  # retain the graph's own diagnostic
        yield from _test_values(
            cg.test,
            cg.ci_test,
            x,
            y,
            chain((conditions,), iterator),
            alpha,
            allow_batch,
        )
        break


def _marginal_test_values(test, ci_call, x, neighbors, alpha):
    """Batch consecutive marginal pairs, replaying their original node order."""
    if not _builtin_fisherz(test):
        for y in neighbors:
            yield y, ci_call(x, y, ())
        return
    iterator = iter(neighbors)
    while chunk := list(islice(iterator, _SKELETON_BATCH)):
        values = (
            test.batch((x, y, ()) for y in chunk)
            if len(chunk) > 1
            else [ci_call(x, chunk[0], ())]
        )
        for y, value in zip(chunk, values, strict=False):
            yield y, _threshold_pvalue(test, x, y, (), value, alpha)


def _marginal_values(cg, x, neighbors, alpha):
    if _batch_graph(cg):
        yield from _marginal_test_values(cg.test, cg.ci_test, x, neighbors, alpha)
    else:
        for y in neighbors:
            yield y, cg.ci_test(x, y, ())


def _process_fisher_values(cache, corr, n_samples, x, y, subsets, alpha, stable):
    if not stable or fisher_z_from_corr is not _FISHER_KERNEL:
        for conditions in subsets:
            value = _cached_fisher_z(cache, corr, n_samples, x, y, conditions)
            if fisher_z_from_corr is _FISHER_KERNEL and math.isclose(
                value, alpha, rel_tol=1e-10, abs_tol=1e-14
            ):
                value = _fisher_z_reference(corr, n_samples, x, y, conditions)
            yield conditions, value
        return
    iterator = iter(subsets)
    while chunk := list(islice(iterator, _SKELETON_BATCH)):
        missing = [
            (x, y, conditions)
            for conditions in chunk
            if _ci_cache_key_pair(x, y, conditions) not in cache
        ]
        values = fisher_z_from_corr_batch(corr, n_samples, missing)
        for request, value in zip(missing, values, strict=False):
            cache[_ci_cache_key_pair(*request)] = value
        for conditions in chunk:
            value = cache[_ci_cache_key_pair(x, y, conditions)]
            if math.isclose(value, alpha, rel_tol=1e-10, abs_tol=1e-14):
                value = _fisher_z_reference(corr, n_samples, x, y, conditions)
            yield conditions, value
