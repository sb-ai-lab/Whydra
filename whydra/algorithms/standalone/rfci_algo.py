import copy
import hashlib
import math
import multiprocessing as mp
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, pairwise
from multiprocessing import shared_memory

import numpy as np

from ...background_knowledge import (
    BKOrientationGuard,
    BKPhase,
    MatrixEncoding,
    apply_bk_checkpoint,
)
from ..graph_core import CIT, CausalGraph

# Import the shared test class from the neighboring module.
from ..graph_core import cache as _cache_exports
from ..graph_core._optional_imports import Parallel, delayed
from ..graph_core.cache import ci_cache_for
from ..graph_core.ci_batch import (
    PROCESS_POOL_OPTIONS,
    batch_execution_stats,
    evaluate_ci_batch,
    supports_fisher_batch,
)
from ..graph_core.conditioning import (
    bounded_size,
    conditioning_allowed,
    configure_conditioning,
)
from ..graph_core.endpoints import Endpoint
from ..graph_core.validation import effective_n_jobs
from . import pag_rules as _pag_rules_exports
from ._counted_ci import counted_fisherz
from .ci_adapters import attach_counted_batch, observed_phase_backend
from .orientation_queue import OrientationQueue
from .pag_init import to_circle_adjacency
from .pag_rules import _clear_triple, _discriminating, _Orientations

# --- Helper functions ---
from .skeleton import SkeletonDiscovery

BoundedCache = _cache_exports.BoundedCache
faith_check = _pag_rules_exports.faith_check
minUncovCircPath = _pag_rules_exports.minUncovCircPath
minUncovPdPath = _pag_rules_exports.minUncovPdPath


class _SharedRFCICache:
    """Shared-memory hash table cache for CI p-values."""

    def __init__(
        self,
        capacity=1_000_003,
        n_locks=64,
        *,
        create=True,
        names=None,
        locks=None,
        owner=False,
    ):
        self.capacity = int(capacity)
        self.n_locks = int(n_locks)
        self.owner = bool(owner)
        self._names = names
        self._locks = locks
        self._manager = None

        self._shm_hi = None
        self._shm_lo = None
        self._shm_val = None
        self._shm_state = None
        self._keys_hi = None
        self._keys_lo = None
        self._values = None
        self._state = None

        if create:
            self._shm_hi = shared_memory.SharedMemory(
                create=True, size=self.capacity * 8
            )
            self._shm_lo = shared_memory.SharedMemory(
                create=True, size=self.capacity * 8
            )
            self._shm_val = shared_memory.SharedMemory(
                create=True, size=self.capacity * 8
            )
            self._shm_state = shared_memory.SharedMemory(
                create=True, size=self.capacity
            )
            self._names = (
                self._shm_hi.name,
                self._shm_lo.name,
                self._shm_val.name,
                self._shm_state.name,
            )
            if locks is None:
                # Picklable lock proxies for loky workers.
                self._manager = mp.Manager()
                self._locks = [self._manager.Lock() for _ in range(self.n_locks)]
            else:
                self._locks = locks
            self.owner = True
            self._attach_arrays()
            assert self._state is not None
            self._state[:] = 0

    def __getstate__(self):
        return {
            "capacity": self.capacity,
            "n_locks": self.n_locks,
            "owner": False,
            "names": self._names,
            "locks": self._locks,
        }

    def __setstate__(self, state):
        self.capacity = state["capacity"]
        self.n_locks = state["n_locks"]
        self.owner = state["owner"]
        self._names = state["names"]
        self._locks = state["locks"]
        self._manager = None

        self._shm_hi = None
        self._shm_lo = None
        self._shm_val = None
        self._shm_state = None
        self._keys_hi = None
        self._keys_lo = None
        self._values = None
        self._state = None

    def _ensure_attached(self):
        if self._keys_hi is None:
            assert self._names is not None
            n_hi, n_lo, n_val, n_state = self._names
            self._shm_hi = shared_memory.SharedMemory(name=n_hi)
            self._shm_lo = shared_memory.SharedMemory(name=n_lo)
            self._shm_val = shared_memory.SharedMemory(name=n_val)
            self._shm_state = shared_memory.SharedMemory(name=n_state)
            self._attach_arrays()

        assert self._keys_hi is not None
        assert self._keys_lo is not None
        assert self._values is not None
        assert self._state is not None
        return self._keys_hi, self._keys_lo, self._values, self._state

    def _attach_arrays(self):
        assert self._shm_hi is not None
        assert self._shm_lo is not None
        assert self._shm_val is not None
        assert self._shm_state is not None
        self._keys_hi = np.ndarray(
            (self.capacity,), dtype=np.uint64, buffer=self._shm_hi.buf
        )
        self._keys_lo = np.ndarray(
            (self.capacity,), dtype=np.uint64, buffer=self._shm_lo.buf
        )
        self._values = np.ndarray(
            (self.capacity,), dtype=np.float64, buffer=self._shm_val.buf
        )
        self._state = np.ndarray(
            (self.capacity,), dtype=np.uint8, buffer=self._shm_state.buf
        )

    @staticmethod
    def _fingerprint(key_obj):
        a, b, cond = key_obj
        h = hashlib.blake2b(digest_size=16)
        h.update(int(a).to_bytes(4, "little", signed=False))
        h.update(int(b).to_bytes(4, "little", signed=False))
        h.update(len(cond).to_bytes(4, "little", signed=False))
        for x in cond:
            h.update(int(x).to_bytes(4, "little", signed=False))
        digest = h.digest()
        hi = np.uint64(int.from_bytes(digest[:8], "little", signed=False))
        lo = np.uint64(int.from_bytes(digest[8:], "little", signed=False))
        return hi, lo

    def _start_idx(self, hi, lo):
        return int((int(hi) ^ int(lo)) % self.capacity)

    def get(self, key_obj):
        keys_hi, keys_lo, values, state = self._ensure_attached()
        hi, lo = self._fingerprint(key_obj)
        idx = self._start_idx(hi, lo)
        for _ in range(self.capacity):
            st = state[idx]
            if st == 0:
                return None
            if st == 1 and keys_hi[idx] == hi and keys_lo[idx] == lo:
                return float(values[idx])
            idx = (idx + 1) % self.capacity
        return None

    def set(self, key_obj, val):
        keys_hi, keys_lo, values, state = self._ensure_attached()
        hi, lo = self._fingerprint(key_obj)
        if self._locks is None:
            return
        # Choose the lock from the same value used to choose the slot.
        # Keys with different hi values but equal hi^lo share a slot. Locking on
        # int(hi) used different locks and allowed partial writes across the four arrays.
        lock = self._locks[self._start_idx(hi, lo) % self.n_locks]
        with lock:
            idx = self._start_idx(hi, lo)
            for _ in range(self.capacity):
                st = state[idx]
                if st == 0 or (keys_hi[idx] == hi and keys_lo[idx] == lo):
                    keys_hi[idx] = hi
                    keys_lo[idx] = lo
                    values[idx] = float(val)
                    state[idx] = 1
                    return
                idx = (idx + 1) % self.capacity

    def close(self):
        for shm in (self._shm_hi, self._shm_lo, self._shm_val, self._shm_state):
            if shm is not None:
                shm.close()
        if self.owner and self._manager is not None:
            self._manager.shutdown()

    def unlink(self):
        if not self.owner:
            return
        for shm in (self._shm_hi, self._shm_lo, self._shm_val, self._shm_state):
            if shm is not None:
                shm.unlink()


def _ci_cache_key(i: int, j: int, S):
    """Canonical cache key for CI test symmetry in (i, j, S)."""
    a, b = (int(i), int(j)) if i <= j else (int(j), int(i))
    if S is None:
        cond = ()
    elif isinstance(S, (set, frozenset, list, tuple, np.ndarray)):
        cond = tuple(sorted(int(x) for x in S))
    else:
        cond = (int(S),)
    return a, b, cond


def _process_ci_eval(cit, shared_cache, x, y, S):
    key = _ci_cache_key(x, y, S)
    if shared_cache is not None:
        cached = shared_cache.get(key)
        if cached is not None:
            return cached, True
        pval = cit(x, y, list(key[2]))
        shared_cache.set(key, pval)
        return pval, False
    return cit(x, y, list(key[2])), None


def _process_ci_chunk(cit, shared_cache, x, y, cond_sets):
    if shared_cache is None and supports_fisher_batch(cit):
        return [
            (value, None)
            for value in evaluate_ci_batch(cit, [(x, y, S) for S in cond_sets])
        ]
    return [_process_ci_eval(cit, shared_cache, x, y, S) for S in cond_sets]


def _record_process_ci_stats(cg, total, hits=0, misses=0):
    ci_stats = getattr(cg, "_parallel_ci_stats", None)
    phase = getattr(cg, "_parallel_phase", None)
    if ci_stats is None or phase is None:
        return
    ci_stats["total"] += int(total)
    ci_stats[phase["name"]] += int(total)
    key = f"{phase['name']}_process_tasks"
    ci_stats[key] = ci_stats.get(key, 0) + int(total)
    if "ci_cache_hits" in ci_stats:
        ci_stats["ci_cache_hits"] += int(hits)
    if "ci_cache_misses" in ci_stats:
        ci_stats["ci_cache_misses"] += int(misses)


def _ci_batch(cg: CausalGraph, x: int, y: int, cond_sets, n_jobs: int = 1):
    """Evaluate CI tests for fixed (x, y) over many conditioning sets."""
    cond_sets = list(cond_sets)
    if not cond_sets:
        return []
    parallel_backend = getattr(cg, "_parallel_backend", "processes")
    if supports_fisher_batch(cg.test):
        return evaluate_ci_batch(
            cg.test,
            [(x, y, S) for S in cond_sets],
            n_jobs=n_jobs,
            backend=parallel_backend,
        )
    if n_jobs in (None, 0, 1) or (
        parallel_backend != "threads" and len(cond_sets) < 256
    ):
        return [cg.ci_test(x, y, list(S)) for S in cond_sets]
    if parallel_backend != "threads" and not hasattr(cg, "_parallel_cit"):
        return [cg.ci_test(x, y, list(S)) for S in cond_sets]

    parallel_executor = getattr(cg, "_parallel_executor", None)
    n_jobs_eff = effective_n_jobs(n_jobs)
    if parallel_backend != "processes":
        owns_executor = parallel_executor is None
        executor = parallel_executor or ThreadPoolExecutor(max_workers=n_jobs_eff)
        try:
            futures = [executor.submit(cg.ci_test, x, y, list(S)) for S in cond_sets]
            values = [future.result() for future in futures]
            stats = getattr(cg, "_parallel_ci_stats", None)
            if stats is not None:
                stats["rfci_orient_thread_tasks"] = stats.get(
                    "rfci_orient_thread_tasks", 0
                ) + len(values)
            return values
        finally:
            if owns_executor:
                executor.shutdown(wait=True)

    # rfci_stable sets _parallel_backend="processes" and _parallel_cit under
    # the same use_process_orient condition, so cit is always available here.
    parallel_cit = cg._parallel_cit

    shared_cache = getattr(cg, "_parallel_shared_cache", None)
    chunk_size = max(1, math.ceil(len(cond_sets) / n_jobs_eff))
    chunks = [
        cond_sets[i : i + chunk_size] for i in range(0, len(cond_sets), chunk_size)
    ]
    if parallel_executor is None:
        # Most fits never dispatch orientation work. Create its executor only
        # on demand so the numerical skeleton pool stays reusable.
        parallel_executor = Parallel(n_jobs=n_jobs_eff, **PROCESS_POOL_OPTIONS)
        parallel_executor.__enter__()
        cg._parallel_executor = parallel_executor
    executor = parallel_executor
    chunk_results = executor(
        delayed(_process_ci_chunk)(parallel_cit, shared_cache, x, y, chunk)
        for chunk in chunks
    )
    results = [item for chunk in chunk_results for item in chunk]
    pvals = [pval for pval, _hit in results]
    if shared_cache is None:
        _record_process_ci_stats(cg, len(results), misses=len(results))
    else:
        hits = sum(1 for _pval, hit in results if hit)
        misses = len(results) - hits
        _record_process_ci_stats(cg, len(results), hits=hits, misses=misses)
    return pvals


def _resolve_rfci_backend(backend, n_vars, process_min_vars):
    requested = "auto" if backend in (None, "auto") else str(backend)
    if requested == "auto":
        # Workload thresholds belong to the dispatcher, not the variable count.
        effective = "processes"
    elif requested in {"threads", "processes"}:
        effective = requested
    else:
        msg = "RFCI backend must be 'auto', 'threads', or 'processes'"
        raise ValueError(msg)
    return requested, effective


def _observed_phase_backend(test, before, coarse=None, extra=()):
    """Describe actual dispatch, independently of a requested backend label."""
    return observed_phase_backend(batch_execution_stats(test), before, coarse, extra)


def _sepset_to_set(seps):
    """Normalize sepset payload into set[int]."""
    if seps is None:
        return set()
    if isinstance(seps, (set, frozenset)):
        return {int(x) for x in seps}
    if isinstance(seps, (np.integer, int)):
        return {int(seps)}
    if isinstance(seps, (list, tuple, np.ndarray)):
        if len(seps) == 0:
            return set()
        if all(isinstance(x, (int, np.integer)) for x in seps):
            return {int(x) for x in seps}
        candidates = []
        for S in seps:
            if S is None:
                continue
            if isinstance(S, (set, frozenset, list, tuple, np.ndarray)):
                candidates.append({int(x) for x in S})
            else:
                candidates.append({int(S)})
        # RFCI selects one shortest separator. For [(1, 4), (5,)] it uses {5}.
        # The skeleton's flat sepset map stores the union {1, 4, 5}, but this
        # code reads the individual sets. A smaller separator can make R0
        # orient extra colliders when the middle node appears only in another
        # set. FCI+ also uses a smallest separator for collider orientation.
        # In a comparison with causal-learn on 100 configurations, using the
        # union produced 6 differing graphs (23 cells), versus 5 (21 cells)
        # with one shortest set. Whydra issue #23 tracks this R0 choice.
        return min(candidates, key=len) if candidates else set()
    return {int(seps)}


def _sep_union(cg: CausalGraph, i: int, k: int):
    return _sepset_to_set(cg.sepset[i][k]) | _sepset_to_set(cg.sepset[k][i])


def _unshielded_triples(cg: CausalGraph):
    """Return a list of unshielded triples (k, j, m) with k < m.

    The middle node j is adjacent to both k and m, and k and m are not adjacent.

    The middle node must have at least two neighbors. Earlier code tested the end node k with
    ``if len(cg.neighbors(k)) < 2: continue``. It missed triples with a degree-one
    endpoint, including simple colliders such as X1 *-> X3 <-* X2. Iterating over
    middle nodes also avoids an extra scan of all node pairs.
    """
    triples = []
    adj = cg.G.graph
    for j in range(len(cg.nodes)):
        neighbors_j = sorted(int(x) for x in cg.neighbors(j))
        if len(neighbors_j) < 2:
            continue
        for k, m in combinations(neighbors_j, 2):
            if not adj[k, m]:
                triples.append((k, j, m))
    return triples


def _find_minimal_set_from_candidates(
    cg: CausalGraph, x: int, y: int, candidates_set, alpha: float
):
    cand = list(candidates_set)
    for size in range(bounded_size(cg, len(cand)) + 1):
        for subset in combinations(cand, size):
            if cg.ci_test(x, y, list(subset)) >= alpha:
                S = list(subset)
                cg.sepset[x][y] = S
                cg.sepset[y][x] = S
                return S
    return None


def _as_pairs(protected_edges):
    """Build a set of ``frozenset({x, y})`` values from arbitrary pairs of indices."""
    return {frozenset((int(x), int(y))) for x, y in protected_edges}


def _empty_guard() -> BKOrientationGuard:
    """Allow all orientations without background knowledge and leave rule hooks inactive."""
    return BKOrientationGuard(None, MatrixEncoding.STANDARD)


def _orient_v_structures_rfci(
    cg: CausalGraph, alpha, M, unfVect=None, protected_edges=(), guard=None
):
    """RFCI unshielded-triple refinement (Colombo et al., 2012, §3.2)."""
    guard = _empty_guard() if guard is None else guard
    protected = _as_pairs(protected_edges)
    groups = deque(M)
    candidates = []
    deleted = set()

    def invalid(triple):
        a, b, c = triple
        return frozenset((a, b)) in deleted or frozenset((b, c)) in deleted

    while groups:
        triple = groups.popleft()
        if invalid(triple):
            continue
        a, b, c = triple
        separator = _sep_union(cg, a, c)
        conditioning = list(separator - {b})
        if not conditioning_allowed(cg, conditioning):
            continue
        left = cg.ci_test(a, b, list(conditioning))
        right = cg.ci_test(b, c, list(conditioning))
        if left < alpha and right < alpha:
            if _clear_triple(a, b, c, unfVect):
                candidates.append(triple)
            continue
        for outside, other, score in ((a, c, left), (c, a, right)):
            edge = frozenset((outside, b))
            if not score >= alpha or edge in protected:
                continue
            pool = separator.copy()
            pool.discard(other)
            _find_minimal_set_from_candidates(cg, outside, b, pool, alpha)
            lo, hi = sorted(edge)
            common = set(cg.neighbors(outside)) & set(cg.neighbors(b))
            groups.extend((lo, middle, hi) for middle in common if middle not in edge)
            cg.remove_edge(cg.nodes[outside], cg.nodes[b])
            deleted.add(edge)

    graph = cg.G.graph
    for a, b, c in candidates:
        if invalid((a, b, c)) or b in _sep_union(cg, a, c):
            continue
        if all((graph[a, b], graph[b, a], graph[c, b], graph[b, c])):
            guard.set_mark(graph, a, b, ARROW)
            guard.set_mark(graph, c, b, ARROW)


# Rule predicates and returned PAGs use the standard Endpoint marks.
CIRCLE_MARK = Endpoint.CIRCLE.value  # 2: open endpoint (circle) in rule predicates.
ARROW = Endpoint.ARROW.value  # 1
TAIL_MARK = Endpoint.TAIL.value  # -1: tail in rule predicates.
NULL = Endpoint.NULL.value  # 0


def _apply_r1_r3(pag, unfVect=None, guard=None, work=None):
    restrictions = _empty_guard() if guard is None else guard
    implications = _Orientations(pag, restrictions, unfVect, work)
    for rule in (1, 2, 3):
        implications.run(rule)
    return pag


def _apply_r5_r10(pag, p, unfVect=None, guard=None, work=None):
    restrictions = _empty_guard() if guard is None else guard
    implications = _Orientations(pag, restrictions, unfVect, work)
    for rule in range(5, 11):
        implications.run(rule)
    return pag


def _is_adj(pag, u, v):
    return not (pag[u, v] == NULL and pag[v, u] == NULL)


def _min_discriminating_path(pag, a, b, c):
    return _discriminating(pag, a, b, c, reciprocal=False) or None


def _check_edges_on_disc_path(
    cg: CausalGraph, alpha, path, unfVect=None, max_k=None, n_jobs=1, protected_edges=()
):
    """Return True if a path edge was separated, False if the search was exhausted, or None if it was truncated."""
    protected = _as_pairs(protected_edges)
    evidence = _sep_union(cg, path[0], path[-1])
    exhausted = True
    for u, v in pairwise(path):
        # Preserve the established enumeration order inside a CI batch.
        pool = evidence.copy()
        pool.discard(u)
        pool.discard(v)
        limit = bounded_size(
            cg, len(pool) if max_k is None else min(len(pool), int(max_k))
        )
        exhausted = exhausted and limit >= len(pool)
        edge = frozenset((u, v))
        for depth in range(limit + 1):
            trial = list(combinations(pool, depth))
            scores = _ci_batch(cg, u, v, trial, n_jobs=n_jobs)
            winner = next((k for k, score in enumerate(scores) if score >= alpha), None)
            if winner is not None and edge not in protected:
                graph = cg.G.graph
                lo, hi = sorted(edge)
                triples = [
                    (lo, m, hi)
                    for m in range(len(graph))
                    if m not in edge
                    and _is_adj(graph, u, v)
                    and _is_adj(graph, u, m)
                    and _is_adj(graph, v, m)
                ]
                cg.sepset[u][v] = list(trial[winner])
                cg.sepset[v][u] = list(trial[winner])
                cg.remove_edge(cg.nodes[u], cg.nodes[v])
                _orient_v_structures_rfci(
                    cg, alpha, triples, unfVect, protected_edges=protected_edges
                )
                return True
            if not _is_adj(cg.G.graph, u, v):
                break
    return False if exhausted else None


def _apply_discriminating_path_rules(
    cg,
    pag,
    p,
    alpha,
    unfVect=None,
    n_jobs=1,
    max_k=None,
    guard=None,
    protected_edges=(),
    work=None,
):
    """Refine an R4 witness with extra CI tests before inferring its marks."""
    guard = _empty_guard() if guard is None else guard

    def witnesses(graph):
        circle = graph.T == CIRCLE_MARK
        np.fill_diagonal(circle, False)
        candidates = (
            zip(*circle.nonzero(), strict=False)
            if work is None
            else work.candidates("R4", graph, circle, paths=True, row_order=True)
        )
        for b, c in candidates:
            parents = (graph[:, c] == ARROW) & (graph[c] == TAIL_MARK)
            for a in np.flatnonzero(parents & (graph[b] == ARROW) & (graph[:, b] != 0)):
                if graph[c, b] == CIRCLE_MARK and a != b and a != c:
                    path = _min_discriminating_path(graph, a, b, c)
                    if path:
                        yield path

    while True:
        graph = cg.G.graph
        for path in witnesses(graph):
            separated = _check_edges_on_disc_path(
                cg,
                alpha,
                path,
                unfVect=unfVect,
                max_k=max_k,
                n_jobs=n_jobs,
                protected_edges=protected_edges,
            )
            if separated:
                break
            if separated is None:
                continue
            a, b, c = path[-3:]
            if b in _sep_union(cg, path[0], c):
                conclusions = ((b, c, ARROW), (c, b, TAIL_MARK))
            else:
                conclusions = (
                    (a, b, ARROW),
                    (b, a, ARROW),
                    (b, c, ARROW),
                    (c, b, ARROW),
                )
            original = graph.copy()
            for x, y, mark in conclusions:
                guard.set_mark(graph, x, y, mark)
            if not np.array_equal(original, graph):
                break
        else:
            return graph


def _orient_edges_rfci(
    cg,
    p,
    alpha,
    unfVect=None,
    n_jobs=1,
    max_k=None,
    background_knowledge=None,
    node_names=None,
    guard=None,
    protected_edges=(),
):
    if guard is None:
        guard = _empty_guard()
    pag = copy.deepcopy(cg.G.graph)
    work = OrientationQueue(
        pag,
        enabled=supports_fisher_batch(cg.test) and type(guard) is BKOrientationGuard,
    )
    from inspect import signature

    rule_kwargs = (
        {"work": work}
        if all(
            "work" in signature(fn).parameters for fn in (_apply_r1_r3, _apply_r5_r10)
        )
        else {}
    )
    if not rule_kwargs:
        work.enabled = False
    guard = work.guard(guard)
    prev_pag = np.zeros_like(pag)
    while not np.array_equal(pag, prev_pag):
        prev_pag = copy.deepcopy(pag)
        pag = _apply_r1_r3(pag, unfVect, guard=guard, **rule_kwargs)
        cg.G.graph = pag

        # pag = np.array([
        #     [ 0, -1,  1,  0,  0,  2,  0,  0,  0,  0],
        #     [ 1,  0,  2,  2,  0,  0,  0,  0,  0,  0],
        #     [-1,  1,  0,  2,  0,  0,  0,  0,  0,  0],
        #     [ 0,  1,  1,  0,  2,  0,  0,  0,  0,  0],
        #     [ 0,  0,  0,  1,  0,  2,  0,  0,  0,  0],
        #     [ 2,  0,  0,  0,  1,  0,  2,  0,  0,  0],
        #     [ 0,  0,  0,  0,  0,  1,  0,  2,  0,  0],
        #     [ 0,  0,  0,  0,  0,  0,  1,  0,  2,  0],
        #     [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  2],
        #     [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0],
        # ], dtype=int)
        # cg.G.graph = pag

        pag = _apply_discriminating_path_rules(
            cg,
            pag,
            p,
            alpha,
            unfVect=unfVect,
            n_jobs=n_jobs,
            max_k=max_k,
            guard=guard,
            protected_edges=protected_edges,
            work=work,
        )
        pag = _apply_r5_r10(pag, p, unfVect, guard=guard, **rule_kwargs)
        cg.G.graph = pag
        apply_bk_checkpoint(
            background_knowledge,
            cg.G.graph,
            node_names if node_names is not None else [node.name for node in cg.nodes],
            phase=BKPhase.BETWEEN_ORIENTATION,
            checkpoint="after_orientation_rule_pass",
            encoding=MatrixEncoding.STANDARD,
            graph_kind="pag",
            previous=prev_pag,
        )
        pag = cg.G.graph

    cg.G.graph = pag
    # print("After R5-R10:")
    # print(cg.G.graph)


def rfci_stable(
    cg,
    data,
    alpha=0.05,
    indep_test="fisherz",
    max_k=None,
    n_jobs=1,
    verbose=False,
    show_progress=False,
    use_ci_cache=True,
    use_shared_ci_cache=False,
    shared_ci_cache_capacity=1_000_003,
    skeleton_backend=None,
    orient_backend=None,
    ci_backend=None,
    process_min_vars=150,
    background_knowledge=None,
    max_cond_set_size=None,
) -> np.ndarray:
    """Test conditional independences with stable RFCI and return the endpoint matrix."""
    configure_conditioning(cg, max_cond_set_size)
    total_t0 = time.perf_counter()
    _n, p = data.shape
    n_jobs_eff = effective_n_jobs(n_jobs)
    skeleton_backend_requested, skeleton_backend_eff = _resolve_rfci_backend(
        skeleton_backend,
        p,
        process_min_vars,
    )
    orient_backend_requested, orient_backend_eff = _resolve_rfci_backend(
        orient_backend if orient_backend is not None else ci_backend,
        p,
        process_min_vars,
    )
    use_process_skeleton = (
        n_jobs_eff != 1
        and indep_test == "fisherz"
        and skeleton_backend_eff == "processes"
    )
    use_process_orient = n_jobs_eff != 1 and orient_backend_eff == "processes"
    cit = (
        cg.test
        if isinstance(cg.test, CIT)
        and cg.test.method == indep_test
        and np.array_equal(cg.test.data, data)
        else CIT(data, method=indep_test)
    )
    cit.max_cond_set_size = max_cond_set_size
    native_fisher = supports_fisher_batch(cit)
    use_process_skeleton = use_process_skeleton and native_fisher
    ci_stats = {
        "total": 0,
        "skeleton": 0,
        "rfci_orient": 0,
        "skeleton_backend": "serial",
        "orient_backend": "serial",
        "skeleton_backend_requested": skeleton_backend_requested,
        "orient_backend_requested": orient_backend_requested,
        "process_min_vars": int(process_min_vars),
        "n_vars": int(p),
        "n_jobs": int(n_jobs_eff),
    }
    use_shared_cache = bool(use_ci_cache and use_shared_ci_cache and use_process_orient)
    use_local_cache = bool(use_ci_cache and not use_shared_cache)
    if use_local_cache or use_shared_cache:
        ci_stats["ci_cache_hits"] = 0
        ci_stats["ci_cache_misses"] = 0
    phase = {"name": "skeleton"}
    ci_cache = ci_cache_for(cit) if use_local_cache else None
    ci_inflight = {}
    stats_lock = threading.Lock()
    shared_ci_cache = (
        _SharedRFCICache(capacity=shared_ci_cache_capacity, create=True)
        if use_shared_cache
        else None
    )

    def scalar_cit(i, j, S):
        key = _ci_cache_key(i, j, S)
        with stats_lock:
            ci_stats["total"] += 1
            ci_stats[phase["name"]] += 1

        if shared_ci_cache is not None:
            cached = shared_ci_cache.get(key)
            if cached is not None:
                with stats_lock:
                    ci_stats["ci_cache_hits"] += 1
                return cached

            with stats_lock:
                ci_stats["ci_cache_misses"] += 1
            pval = cit(i, j, list(key[2]))
            shared_ci_cache.set(key, pval)
            return pval

        if ci_cache is None:
            return cit(i, j, S)

        while True:
            with stats_lock:
                cached = ci_cache.get(key)
                if cached is not None:
                    ci_stats["ci_cache_hits"] += 1
                    return cached
                event = ci_inflight.get(key)
                if event is None:
                    event = threading.Event()
                    ci_inflight[key] = event
                    ci_stats["ci_cache_misses"] += 1
                    owner = True
                else:
                    owner = False
            if owner:
                break
            event.wait()

        try:
            pval = cit(i, j, list(key[2]))
            with stats_lock:
                ci_cache[key] = pval
            return pval
        finally:
            with stats_lock:
                ci_inflight.pop(key, None)
                event.set()

    counted_cit = scalar_cit
    if shared_ci_cache is None:
        counted_cit = counted_fisherz(
            scalar_cit, cit, ci_cache, ci_stats, phase, stats_lock
        )
        attach_counted_batch(counted_cit, cit, ci_stats, phase, ci_cache, stats_lock)

    try:
        """"""
        node_names = [node.name for node in cg.nodes]
        apply_bk_checkpoint(
            background_knowledge,
            cg.G.graph,
            node_names,
            phase=BKPhase.PRE_SEARCH,
            checkpoint="before_skeleton",
            encoding=MatrixEncoding.STANDARD,
            graph_kind="pag",
        )
        protected_edges = ()
        if background_knowledge is not None:
            protected_edges = background_knowledge.required_pairs(
                node_names, phase=BKPhase.PRE_SEARCH
            )
        cg.set_ind_test(cit if use_process_skeleton else counted_cit)
        skeleton_before = batch_execution_stats(cit)
        skeleton_t0 = time.perf_counter()
        skeleton = SkeletonDiscovery(
            cg,
            alpha,
            n_jobs=n_jobs_eff
            if native_fisher or skeleton_backend_eff == "threads"
            else 1,
            verbose=verbose,
            show_progress=show_progress,
            parallel_backend=skeleton_backend_requested,
            process_ci_info=(
                {
                    "method": indep_test,
                    "corr_matrix": cit.corr_matrix,
                    "n_samples": cit.n_samples,
                }
                if use_process_skeleton
                else None
            ),
            protected_edges=protected_edges,
        )
        skeleton.run_fci()
        ci_stats["skeleton_backend"] = _observed_phase_backend(
            cit, skeleton_before, getattr(cg, "fci_skeleton_execution_stats", None)
        )
        ci_stats["time_skeleton_sec"] = time.perf_counter() - skeleton_t0
        if use_process_skeleton:
            ci_stats["total"] += int(skeleton.ci_tests_performed)
            ci_stats["skeleton"] += int(skeleton.ci_tests_performed)
            if "ci_cache_misses" in ci_stats:
                ci_stats["ci_cache_misses"] += int(skeleton.ci_tests_performed)
            cg.set_ind_test(counted_cit)

        cg._parallel_backend = orient_backend_requested
        cg._parallel_ci_stats = ci_stats
        cg._parallel_phase = phase
        if use_process_orient:
            cg._parallel_cit = cit
            cg._parallel_shared_cache = shared_ci_cache
            cg._parallel_ci_stats = ci_stats
            cg._parallel_phase = phase
        # Explicit o-o initialization right after skeleton.
        cg.G.graph = to_circle_adjacency(cg.G.graph, MatrixEncoding.STANDARD)
        # print(cg.G.graph)

        # print("Starting RFCI orientation rules...")
        phase["name"] = "rfci_orient"
        orient_before = batch_execution_stats(cit)
        orient_t0 = time.perf_counter()
        # Rules write marks only through the guard, keeping background knowledge
        # constraints in force throughout orientation. Checkpoints alone are
        # insufficient: R0–R10 can overwrite an endpoint of a required edge. If
        # restoring its direction creates a cycle, the checkpoint rolls back the
        # entire transaction, leaving the constraint violated.
        orientation_guard = (
            background_knowledge.orientation_guard(
                node_names,
                phase=BKPhase.BETWEEN_ORIENTATION,
                encoding=MatrixEncoding.STANDARD,
            )
            if background_knowledge is not None
            else _empty_guard()
        )

        # Apply background knowledge before R0 so the rules start from a graph
        # that already satisfies the constraints.
        apply_bk_checkpoint(
            background_knowledge,
            cg.G.graph,
            node_names,
            phase=BKPhase.BETWEEN_ORIENTATION,
            checkpoint="before_orientation",
            encoding=MatrixEncoding.STANDARD,
            graph_kind="pag",
        )

        M = _unshielded_triples(cg)
        before_colliders = cg.G.graph.copy()
        _orient_v_structures_rfci(
            cg,
            alpha=alpha,
            M=M,
            protected_edges=protected_edges,
            guard=orientation_guard,
        )
        apply_bk_checkpoint(
            background_knowledge,
            cg.G.graph,
            node_names,
            phase=BKPhase.BETWEEN_ORIENTATION,
            checkpoint="after_colliders",
            encoding=MatrixEncoding.STANDARD,
            graph_kind="pag",
            previous=before_colliders,
        )
        # print(cg.G.graph)
        _orient_edges_rfci(
            cg,
            p,
            alpha,
            n_jobs=n_jobs_eff,
            max_k=max_k,
            background_knowledge=background_knowledge,
            node_names=node_names,
            guard=orientation_guard,
            protected_edges=protected_edges,
        )
        apply_bk_checkpoint(
            background_knowledge,
            cg.G.graph,
            node_names,
            phase=BKPhase.POST_ORIENTATION,
            checkpoint="after_orientation",
            encoding=MatrixEncoding.STANDARD,
            graph_kind="pag",
        )
        ci_stats["time_orient_sec"] = time.perf_counter() - orient_t0
        ci_stats["orient_backend"] = _observed_phase_backend(
            cit,
            orient_before,
            extra=(
                backend
                for backend, key in (
                    ("processes", "rfci_orient_process_tasks"),
                    ("threads", "rfci_orient_thread_tasks"),
                )
                if ci_stats.get(key, 0)
            ),
        )
        if ci_cache is not None:
            ci_stats["ci_cache_size"] = len(ci_cache)

        executor = getattr(cg, "_parallel_executor", None)
        if executor is not None:
            if isinstance(executor, ThreadPoolExecutor):
                executor.shutdown(wait=True)
            else:
                executor.__exit__(None, None, None)
            delattr(cg, "_parallel_executor")

        ci_stats["time_total_sec"] = time.perf_counter() - total_t0
        cg.ci_stats = ci_stats

        return cg.G.graph
    finally:
        executor = getattr(cg, "_parallel_executor", None)
        if executor is not None:
            if isinstance(executor, ThreadPoolExecutor):
                executor.shutdown(wait=True)
            else:
                executor.__exit__(None, None, None)
        for attr in (
            "_parallel_backend",
            "_parallel_executor",
            "_parallel_cit",
            "_parallel_shared_cache",
            "_parallel_ci_stats",
            "_parallel_phase",
        ):
            if hasattr(cg, attr):
                delattr(cg, attr)
        if shared_ci_cache is not None:
            shared_ci_cache.close()
            shared_ci_cache.unlink()


# For local tests (numpy and time are already imported at the top of this module).


def make_collider_data(n=3000, seed=0):
    """Generate seven synthetic variables for a reproducible collider example."""
    rng = np.random.default_rng(seed)

    L1 = rng.normal(size=n)
    L2 = rng.normal(size=n)

    X = L1 + rng.normal(scale=0.2, size=n)
    Y = L1 + rng.normal(scale=0.2, size=n)
    Z = X + rng.normal(scale=0.2, size=n)
    W = Y + rng.normal(scale=0.2, size=n)

    T = Z + W + rng.normal(scale=0.2, size=n)
    U = T + L2 + rng.normal(scale=0.2, size=n)
    V = U + L2 + rng.normal(scale=0.2, size=n)

    return np.column_stack([X, Y, Z, W, T, U, V])
