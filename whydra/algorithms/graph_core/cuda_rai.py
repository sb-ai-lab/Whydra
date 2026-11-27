"""RAI-stable-cuda: reference control flow with GPU-batched CI tests.

The CPU reference is Whydra's port of IntelLabs ``LearnStructRAI``
(``whydra/algorithms/standalone/rai_algo.py``). It makes sequential decisions
on a live ``_PDAG``. Each task refines exogenous and endogenous edges, orients
v-structures in two passes using the separating sets, and applies Meek-style
rules R1–R4 with bidirected-to-undirected conversion. After recursion, the
learner applies R1–R3 and reorients the final skeleton. Batching all graph
decisions at one recursion level can change the result because later tasks
depend on earlier edge changes.

The CUDA learner subclasses the CPU reference to preserve set iteration,
task order and separating-set bookkeeping.
Each refinement phase snapshots the potential
parents of its connected pairs (exactly like the reference's joblib
speculation layer, ``RAIStableLearner._speculate``), enumerates their
conditioning sets in reference order, tests them in batched GPU calls
(``CudaFisherZ`` LDLᵀ / ``CudaChisqHost``), and hands the p-values to the
reference replay loop, which still makes every graph decision in reference
order. Memo misses (the snapshot went stale after an edge deletion)
re-synchronize on the CPU tester. A p-value within ``_P_RECHECK_EPS`` of
``alpha`` is recomputed with the CPU tester so that GPU-vs-CPU rounding next
to the decision boundary cannot flip a decision.
"""

import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from scipy.special import ndtr as _ndtr

from ..standalone.rai_algo import (
    _PDAG,
    RAIStableLearner,
    _CIOracle,
    _pot_parents,
)
from ._timings import records_timings
from .cuda_graph_batch import compact_adj
from .endpoints import Endpoint

_TAIL = int(Endpoint.TAIL.value)
_ARROW = int(Endpoint.ARROW.value)

#: Marks "speculate on the GPU" inside ``RAIStableCudaLearner``; the base class
#: gates its speculation layer on ``self._parallel is not None``.
_GPU_SPECULATIVE = object()

#: Maximum tests for GPU speculation. The scheduler counts after enumeration
#: and replays the whole phase on CPU when the count exceeds the limit.
_SPEC_MAX_TESTS = 8_000_000
#: Below this many tests a phase is not worth a GPU round trip.
_SPEC_MIN_TESTS = 48
#: GPU batch slice (tests per kernel launch).
_SPEC_CHUNK = 1_000_000
#: ``|p_gpu - alpha|`` within this band is re-computed with the CPU tester.
#: GPU (LDLᵀ + GPU corrcoef) and CPU (LAPACK inv + np.corrcoef) agree to
#: ~1e-12 on p-values; the band makes a boundary flip impossible in practice.
_P_RECHECK_EPS = 1e-7

#: Rule order for the native ``maximally_orient_pattern`` call in
#: ``maximally_orient_edges`` (the reference applies R1–R4).
_RULES_1234 = [1, 2, 3, 4]


class _FastPDAG(_PDAG):
    """``_PDAG`` with an O(1) adjacency bitset backing ``is_connected``.

    ``is_connected`` is the single hottest primitive (~1e6 calls at p=100); the
    reference walks the parents/undirected sets. Here each node keeps an int
    bitset of its neighbours, maintained by the *only* primitives that change
    adjacency (``create_empty_graph`` / ``create_complete_graph`` /
    ``add_edges`` / ``delete_edge``): ``orient_edge`` keeps an existing edge,
    ``convert_bidirected_to_undirected`` keeps it too, and the directed /
    undirected deletes run only inside ``delete_edge``. The predicate and its
    result are identical to the reference; only the lookup is faster.
    """

    def create_empty_graph(self, nodes_set=None) -> None:
        super().create_empty_graph(nodes_set)
        ns = self.nodes_set if nodes_set is None else nodes_set
        self._adj = dict.fromkeys(ns, 0)

    def create_complete_graph(self, nodes_set=None) -> None:
        ns = self.nodes_set if nodes_set is None else nodes_set
        super().create_complete_graph(ns)
        full = 0
        for n in ns:
            full |= 1 << n
        self._adj = {n: full & ~(1 << n) for n in ns}

    def add_edges(self, parents_set, target_node, arrowhead_type="---") -> None:
        super().add_edges(parents_set, target_node, arrowhead_type)
        t = target_node
        tbit = 1 << t
        for p in parents_set:
            self._adj[p] |= tbit
            self._adj[t] |= 1 << p

    def delete_edge(self, node_i, node_j) -> None:
        super().delete_edge(node_i, node_j)
        self._adj[node_i] &= ~(1 << node_j)
        self._adj[node_j] &= ~(1 << node_i)

    def is_connected(self, node_i: int, node_j: int) -> bool:
        assert node_i != node_j
        return ((self._adj[node_i] >> node_j) & 1) == 1

    def rebuild_adj(self) -> None:
        """Recompute the ``is_connected`` bitsets from the live sets.

        The native hot loops (orient_v / Meek / replay) mutate ``_graph``'s
        sets through the CPython C-API without touching ``_adj``; this restores
        the invariant before any Python-side ``is_connected`` call. Native, so
        it is cheap even when called once per phase.
        """
        self._adj = _host().adjacency_native(self._graph, len(self._graph))

    def maximally_orient_pattern(self, rules_set, en_nodes=None) -> None:
        """Restore adjacency before the final Python orientation step.

        At the end of ``_learn``, ``maximally_orient_pattern((1,2,3))`` reads
        ``is_connected``. The native learning loop leaves ``_adj`` untouched. Per-phase
        Meek orientation runs natively elsewhere.
        """
        self.rebuild_adj()
        return super().maximally_orient_pattern(rules_set, en_nodes)


_HOST = None


def _host():
    """Lazily import the host-side RAI accelerator (``_whydra_rai_host``)."""
    global _HOST
    if _HOST is None:
        from . import _whydra_rai_host as _H

        _HOST = _H
    return _HOST


def _edge_endpoint_count(graph) -> int:
    """Cheap exact invariant of the PDAG edge set.

    ``_PDAG._graph`` is ``{node: {head_type: set(neighbors)}}``. Orientation and
    the bidirected flip keep every edge (they only change its type), while a
    refinement phase only deletes edges, so an unchanged count means an
    unchanged edge set within the learn loop.
    """
    return sum(len(neigh) for heads in graph.values() for neigh in heads.values())


def _speculation_jobs(
    graph,
    pairs: Sequence[tuple[int, int]],
    order: int,
) -> list[tuple[int, int, tuple[int, ...], tuple[int, ...]]]:
    """Snapshot potential parents of connected pairs (reference snapshot).

    Identical to ``RAIStableLearner._speculate``'s job assembly: iterate the
    phase's pairs in order, keep connected ones, tuple the live sets so the
    enumeration order matches what a worker (or the GPU) would see.
    """
    jobs: list[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = []
    for a, b in pairs:
        if not graph.is_connected(a, b):
            continue
        pot_a: tuple[int, ...]
        pot_b: tuple[int, ...]
        if order == 0:
            pot_a = pot_b = ()
        else:
            pot_a = tuple(_pot_parents(graph, a, b))
            pot_b = tuple(_pot_parents(graph, b, a))
        jobs.append((a, b, pot_a, pot_b))
    return jobs


def _build_gpu_batcher(cit, dtype=None):
    """GPU CI batcher for the CIT method (fisherz LDLᵀ or chisq)."""
    method = getattr(cit, "method", None)
    if method == "fisherz":
        from .cuda_fisher_z import CudaFisherZ

        data = getattr(cit, "data", None)
        if data is None:
            return CudaFisherZ(cit.corr_matrix, cit.n_samples, dtype=dtype)
        # NOTE: does not touch cit.corr_matrix — sync fallbacks must keep
        # computing on the CPU's own np.corrcoef matrix, like the reference.
        return CudaFisherZ.from_data(data, n_samples=cit.n_samples, dtype=dtype)
    if method == "chisq":
        from .cuda_chisq import CudaChisqHost

        return CudaChisqHost(cit)
    msg = f"device='cuda' requires fisherz or chisq CIT, got {method!r}"
    raise ValueError(msg)


_CUT_AT = 0.9999999


def _fast_fisher_z_corr(corr_matrix, n_samples, x, y, condition_set):
    """Compute a scalar Fisher-Z p-value for the upstream RAI fallback.

    Select the correlation block for the node pair and conditioning set
    with ``np.ix_``. Return 1.0 if the degrees of freedom are nonpositive.
    Otherwise invert the block, using a pseudoinverse if inversion fails.
    A nonfinite partial correlation yields NaN. Clip a finite correlation
    before applying the Fisher-Z transform.

    The two-sided p-value is ``2 * (1 - ndtr(stat))``. Calling SciPy's normal
    CDF primitive directly avoids the scalar argument processing in
    ``scipy.stats.norm.cdf`` (including masking and broadcasting). Earlier
    measurements found about 150 µs of wrapper overhead per call and a
    roughly 15-fold speedup in the fallback that used that wrapper.
    """
    var = [x, y, *condition_set]
    sub_corr = corr_matrix[np.ix_(var, var)]
    dof = n_samples - len(condition_set) - 3
    if dof <= 0:
        return 1.0
    try:
        inv = np.linalg.inv(sub_corr)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(sub_corr)
    denominator = inv[0, 0] * inv[1, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (
            -inv[0, 1] / math.sqrt(abs(denominator))
            if denominator != 0
            else float("nan")
        )
    if not np.isfinite(r):
        return float("nan")
    r = min(_CUT_AT, max(-_CUT_AT, r))
    z = 0.5 * math.log((1 + r) / (1 - r))
    stat = math.sqrt(dof) * abs(z)
    return 2.0 * (1.0 - _ndtr(stat))


class RAIStableCudaLearner(RAIStableLearner):
    """Run RAI with candidate conditional-independence tests precomputed on the GPU.

    The base class sets the order of the recursive search. This subclass
    batches CI tests on the GPU, replays their results in order using the
    host extension, and applies the orientation rules there. Missing or
    borderline p-values are computed on the CPU.
    """

    def __init__(
        self,
        data: np.ndarray,
        alpha: float,
        indep_test: str,
        node_names: Sequence[str],
        verbose: bool = False,
        dtype=None,
        ci_test=None,
    ):
        super().__init__(
            data=data,
            alpha=alpha,
            indep_test=indep_test,
            node_names=node_names,
            verbose=verbose,
        )
        if ci_test is not None:
            if getattr(ci_test, "method", None) not in ("fisherz", "chisq"):
                msg = (
                    "RAI-stable-cuda supports a custom ci_test only when it is "
                    "a CIT with method='fisherz' or method='chisq'; use the "
                    "CPU path for arbitrary CI callables"
                )
                raise NotImplementedError(msg)
            self.ci_test = ci_test
            self.oracle = _CIOracle(ci_test, alpha)
        self._gpu = _build_gpu_batcher(self.ci_test, dtype=dtype)
        # Swap in the fast-endpoint PDAG (same semantics, inlined is_connected).
        self.graph: _FastPDAG = _FastPDAG(self.nodes_set)
        self.graph.create_complete_graph(self.nodes_set)
        self._last_orient_key: Any = None
        self._parallel: Any = None
        self.max_cond_set_size: int | None = None
        self.stats.update(gpu_phases=0, gpu_tests=0, gpu_borderline=0)
        self.stats["gpu_seconds"] = 0.0  # type: ignore[assignment]

    def learn_structure(self, n_jobs: int = 1):
        """Run RAI-stable with GPU CI speculation; ``n_jobs`` is ignored."""
        self.n_jobs = 1
        self._parallel = _GPU_SPECULATIVE
        # The reference PDAG/sepset structures are acyclic, so reference
        # counting reclaims them; the cyclic GC only walks the live set on
        # every collection. Turning it off for the duration removes a
        # measurable slice of host time without touching any decision.
        import gc

        was_enabled = gc.isenabled()
        if was_enabled:
            gc.disable()
        try:
            self._learn()
            final = self._to_general_graph()
        finally:
            self._parallel = None
            if was_enabled:
                gc.enable()
        return final

    # -- GPU speculation layer -------------------------------------------------

    def _cpu_pvalue(self, a, b, key):
        """Compute a missing or borderline p-value during native replay.

        The conditioning variables in ``key`` are sorted. Fisher-Z uses the
        scalar normal CDF helper. Other methods call the configured tester
        directly.
        """
        cit = self.ci_test
        if getattr(cit, "method", None) == "fisherz":
            return _fast_fisher_z_corr(cit.corr_matrix, cit.n_samples, a, b, tuple(key))
        return cit(a, b, tuple(key))

    def maximally_orient_edges(self, en_nodes) -> None:
        """Run the reference orientation sequence natively: orient_v, R1–R4, then the bidirected flip.

        Iteration orders are CPython's set orders by construction (the native
        code iterates the real sets / their real ``copy()``/``intersection``
        results), so this is bit-identical to the reference; only the per-item
        interpreter overhead and the connectivity lookups are removed.

        Redundant-call skip: within the learn loop the graph only loses edges,
        orientation keeps every edge, and the native sequence is idempotent. If
        the edge-endpoint count and ``en_nodes`` are unchanged since the last
        call, re-running it on its own output cannot change anything, so it is
        skipped. Exactness is gated by the parity harness (final CPDAG + the
        full CI-decision trace).
        """
        H = _host()
        graph = self.graph._graph
        key = (_edge_endpoint_count(graph), frozenset(en_nodes))
        if key == self._last_orient_key:
            self.stats["orient_skipped"] = self.stats.get("orient_skipped", 0) + 1
            return
        # Reference sequence, one native call: orient_v(en) -> Meek over all
        # nodes (the reference passes no en_nodes) -> bidirected flip(en).
        # No adj rebuild: orientation/convert keep every edge (they only change
        # its type), so the adjacency bitset stays valid.
        H.maximally_orient_edges_native(
            graph,
            en_nodes,
            self.sepset._sepset,
            self.graph.nodes_set,
            _RULES_1234,
            self.n_vars,
        )
        self._last_orient_key = key

    def _re_orient_skeleton(self) -> None:
        """Restore adjacency before running the reference final skeleton pass.

        The native learning loop leaves the bitset untouched to avoid rebuilding it at
        each phase.
        """
        self.graph.rebuild_adj()
        return RAIStableLearner._re_orient_skeleton(self)

    def _refine_pairs(self, pairs, order, phase, n_en, n_ex) -> None:
        """Run the reference replay loop natively.

        The C++ kernel rebuilds each pair's ``pot_parents`` with the reference's
        own set operations (so the conditioning-set order is identical),
        enumerates ``_cond_sets`` in order, looks each p-value up in the memo,
        re-checks boundary/missing values on the CPU, records the trace, and
        deletes+stamps on the first independent set. The graph sets and the
        sepset store are mutated in place through the CPython C-API.
        """
        if self._parallel is None:
            return RAIStableLearner._refine_pairs(self, pairs, order, phase, n_en, n_ex)
        pmap = self._speculate_fast(pairs, order)
        H = _host()
        n_ci, n_sync, n_border = H.refine_pairs_native(
            self.graph._graph,
            pairs,
            int(order),
            self.sepset._sepset,
            pmap,
            float(self.oracle.alpha),
            float(_P_RECHECK_EPS),
            self.trace,
            self._cpu_pvalue,
        )
        # No adj rebuild here: with ``_speculate_fast`` (native pool assembly)
        # nothing in the learn loop consults ``_adj`` any more; it is restored
        # once, in ``_re_orient_skeleton``, before the final skeleton pass.
        self.n_ci_tests += int(n_ci)
        self.stats["ci_total"] += int(n_ci)
        self.stats["ci_sync"] += int(n_sync)
        self.stats["gpu_borderline"] += int(n_border)
        if pmap is not None:
            self.stats["ci_worker"] += int(pmap.size())
            self.stats["parallel_phases"] += 1
        self.stats["phases"] += 1

    def _speculate(self, pairs, order):
        """Batch CI tests on the GPU, returning None outside the supported batch sizes.

        The snapshot enumeration runs in the host C++ accelerator (fast
        combination generation + dedupe), the p-values come from one (or a few)
        batched GPU calls, and the lookup table is built in C++. The replay
        then resolves every conditioning set against that table by value.
        """
        jobs = _speculation_jobs(self.graph, pairs, order)
        if not jobs:
            return None
        H = _host()
        import numpy as np

        pr = np.asarray([(a, b) for (a, b, _pa, _pb) in jobs], dtype=np.int64)
        pa_list = [list(pa) for (_a, _b, pa, _pb) in jobs]
        pb_list = [list(pb) for (_a, _b, pa, pb) in jobs]
        X, Y, S = H.enumerate_phase(
            torch.from_numpy(pr), pa_list, pb_list, order, self.n_vars
        )
        n = int(X.numel())
        if n < _SPEC_MIN_TESTS or n > _SPEC_MAX_TESTS:
            return None
        return self._gpu_table(H, X, Y, S, n)

    def _speculate_fast(self, pairs, order):
        """Assemble pools natively and otherwise follow ``_speculate``.

        The p-value table is value-keyed, so the snapshot pools only need the
        same *contents* as the reference (not the same iteration order); the
        C++ enumerator builds them straight from the graph and drops
        disconnected pairs, replacing the per-pair Python set/job assembly.
        """
        H = _host()
        import numpy as np

        if not pairs:
            return None
        pr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        X, Y, S = H.enumerate_phase_graph(
            torch.from_numpy(pr), self.graph._graph, int(order), self.n_vars
        )
        n = int(X.numel())
        if n < _SPEC_MIN_TESTS or n > _SPEC_MAX_TESTS:
            return None
        return self._gpu_table(H, X, Y, S, n)

    def _gpu_table(self, H, X, Y, S, n):
        """Run the batched GPU CI tests and build the C++ p-value table."""
        import numpy as np

        t0 = time.perf_counter()
        p = np.empty(n, dtype=np.float64)
        xn = X.numpy()
        yn = Y.numpy()
        Sn = S.numpy()
        for lo in range(0, n, _SPEC_CHUNK):
            hi = min(lo + _SPEC_CHUNK, n)
            p[lo:hi] = np.asarray(
                self._gpu.batch(xn[lo:hi], yn[lo:hi], Sn[lo:hi]),
                dtype=np.float64,
            )
        pmap = H.PValMap(X, Y, S, torch.from_numpy(p), self.n_vars)
        self.stats["gpu_phases"] += 1
        self.stats["gpu_tests"] += n
        self.stats["gpu_seconds"] += time.perf_counter() - t0  # type: ignore[assignment]
        return pmap

    def _ci_memo(self, a, b, cset, memo) -> bool:
        """Apply the reference independence decision with a boundary check.

        Recompute GPU p-values within ``_P_RECHECK_EPS`` of ``alpha`` using the CPU
        tester's exact reference arithmetic before comparing ``p > alpha``.
        """
        key = tuple(sorted(cset))
        self.n_ci_tests += 1
        self.stats["ci_total"] += 1
        p = memo.get(a, b, key) if memo is not None else None
        if p is not None and abs(p - self.oracle.alpha) <= _P_RECHECK_EPS:
            p = None
            self.stats["gpu_borderline"] += 1
        if p is None:
            p = self.ci_test(a, b, key)
            if memo is not None:
                self.stats["ci_sync"] += 1
        independent = p > self.oracle.alpha
        if self.trace is not None:
            self.trace.append((a, b, key, independent))
        return independent


# ---------------------------------------------------------------------------
# Host-side helpers (kept from the endpoint-matrix era; still unit-tested).
# ---------------------------------------------------------------------------


def complete_undirected_endpoints(p: int, device) -> torch.Tensor:
    """Complete undirected CPDAG on CUDA: ``TAIL`` off-diagonal, 0 diagonal."""
    G = torch.full((int(p), int(p)), _TAIL, dtype=torch.int8, device=device)
    G.fill_diagonal_(0)
    return G.contiguous()


def compact_potential_parents(G_ep: torch.Tensor):
    """Packed potential parents: parents, then undirected neighbors.

    ``j`` is a parent of ``i`` iff ``j → i`` (``G[j,i]==ARROW`` and
    ``G[i,j]==TAIL``). Undirected is both tails. Children are excluded.
    Returns ``(nbrs, deg, G_pp)`` with ``G_pp`` int8 0/1 adjacency.
    """
    is_par = ((G_ep.mT == _ARROW) & (G_ep == _TAIL)).to(torch.int8)
    is_und = ((G_ep == _TAIL) & (G_ep.mT == _TAIL)).to(torch.int8)
    nbrs_p, deg_p = compact_adj(is_par)
    nbrs_u, deg_u = compact_adj(is_und)
    deg = (deg_p + deg_u).to(torch.int32)
    device = deg.device
    max_a = int(deg_p.max().item()) if int(deg_p.numel()) else 0
    max_b = int(deg_u.max().item()) if int(deg_u.numel()) else 0
    max_d = max(int(deg.max().item()) if int(deg.numel()) else 0, 1)
    out = torch.zeros((int(deg.numel()), max_d), dtype=torch.int32, device=device)
    for t in range(max_a):
        sel = deg_p > t
        if bool(sel.any()):
            out[sel, t] = nbrs_p[sel, t]
    for t in range(max_b):
        sel = deg_u > t
        if not bool(sel.any()):
            continue
        rows = sel.nonzero(as_tuple=False).reshape(-1)
        cols = (deg_p.index_select(0, rows) + t).to(torch.int64)
        out[rows, cols] = nbrs_u[sel, t]
    G_pp = ((is_par != 0) | (is_und != 0)).to(torch.int8)
    return out, deg, G_pp


def _parents_mask(G_ep: torch.Tensor) -> torch.Tensor:
    """``[i, j]`` iff ``i → j`` (i is a parent of j: ARROW at j, TAIL at i)."""
    return (G_ep == _ARROW) & (G_ep.mT == _TAIL)


def partial_topological_layers(
    G_ep: torch.Tensor, en_mask: torch.Tensor
) -> list[torch.Tensor]:
    """Layers of ``rai_algo._PDAG.find_partial_topological_order`` on endpoints.

    Layer 0 holds nodes with no child in ``en``; each next layer strips the
    previous one's parents (within ``en``). A cyclic remainder collapses into
    one final layer, like the reference.
    """
    is_par = _parents_mask(G_ep)
    layers: list[torch.Tensor] = []
    remaining = en_mask.clone()
    while bool(remaining.any()):
        has_child_in_en = (is_par & remaining.unsqueeze(0)).any(dim=1)
        layer = remaining & ~has_child_in_en
        if not bool(layer.any()):
            layers.append(remaining)
            return layers
        layers.append(layer)
        remaining = remaining & ~layer
    return layers


def lowest_topo(G_ep: torch.Tensor, en_mask: torch.Tensor) -> torch.Tensor:
    """Descendants per the IntelLabs reference ``_get_lowest_topological_set``.

    With several topological layers, return ``en`` minus the topmost layer.
    With a single layer, return all of ``en``.
    """
    layers = partial_topological_layers(G_ep, en_mask)
    if len(layers) > 1:
        return en_mask & ~layers[-1]
    return en_mask.clone()


def exit_cond(deg: "torch.Tensor", en_idx: "torch.Tensor", k: int) -> bool:
    """Check whether every node in ``en`` has ``fan_in = deg ≤ k``."""
    if int(en_idx.numel()) == 0:
        return True
    return int(deg.index_select(0, en_idx).max().item()) <= int(k)


def pair_mask_endogenous(
    tasks: Sequence[tuple[set[int], set[int]]],
    p: int,
    device,
) -> torch.Tensor:
    """``allowed[i,j]`` if some task has both endpoints in that task's ``en``."""
    allowed = torch.zeros((p, p), dtype=torch.bool, device=device)
    for en_nodes, _ex in tasks:
        if len(en_nodes) < 2:
            continue
        m = torch.zeros(p, dtype=torch.bool, device=device)
        idx = torch.as_tensor(sorted(en_nodes), dtype=torch.int64, device=device)
        m.index_fill_(0, idx, True)
        allowed |= m.unsqueeze(1) & m.unsqueeze(0)
    allowed.fill_diagonal_(False)
    return allowed


def pair_mask_exogenous(
    tasks: Sequence[tuple[set[int], set[int]]],
    p: int,
    device,
) -> torch.Tensor:
    """``allowed[en, ex]`` if some task has ``en`` endogenous and ``ex`` exogenous."""
    allowed = torch.zeros((p, p), dtype=torch.bool, device=device)
    for en_nodes, ex_nodes in tasks:
        if not en_nodes or not ex_nodes:
            continue
        men = torch.zeros(p, dtype=torch.bool, device=device)
        mex = torch.zeros(p, dtype=torch.bool, device=device)
        men.index_fill_(
            0,
            torch.as_tensor(sorted(en_nodes), dtype=torch.int64, device=device),
            True,
        )
        mex.index_fill_(
            0,
            torch.as_tensor(sorted(ex_nodes), dtype=torch.int64, device=device),
            True,
        )
        allowed |= men.unsqueeze(1) & mex.unsqueeze(0)
    return allowed


def filter_pairs(
    px: torch.Tensor,
    py: torch.Tensor,
    allowed: torch.Tensor,
) -> tuple:
    """Keep directed pairs whose ``allowed[x, y]`` bit is set."""
    if int(px.numel()) == 0:
        return px, py
    keep = allowed[px.to(torch.int64), py.to(torch.int64)]
    if bool(keep.all()):
        return px, py
    idx = keep.nonzero(as_tuple=False).reshape(-1)
    return px.index_select(0, idx), py.index_select(0, idx)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _sepset_csr(learner: RAIStableLearner, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Return reference separating sets as the CSR pair ``(sepset_S, sepset_ranges)``.

    Only non-empty separators are recorded: ``rai_algo._SeparationSet`` cannot
    distinguish "separated by ∅ at order 0" from "never tested apart", so the
    CSR here is a diagnostic view (empty-S hits read as "no sepset").
    """
    parts: list[tuple[int, int, tuple[int, ...]]] = []
    for (i, j), S in learner.sepset._sepset.items():
        if S:
            parts.append((int(i), int(j), tuple(sorted(int(v) for v in S))))
    parts.sort()
    ranges = np.full((p, p, 2), -1, dtype=np.int64)
    chunks: list[np.ndarray] = []
    offset = 0
    for i, j, S in parts:
        arr = np.asarray(S, dtype=np.int64)
        for u, v in ((i, j), (j, i)):
            if ranges[u, v, 0] >= 0:
                continue
            if arr.size == 0:
                ranges[u, v, 0] = 0
                ranges[u, v, 1] = 0
            else:
                ranges[u, v, 0] = offset
                ranges[u, v, 1] = offset + arr.size
        if arr.size:
            chunks.append(arr)
            offset += int(arr.size)
    S_all = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
    return S_all, ranges


def _write_back_causal_graph(cg, learner: RAIStableCudaLearner) -> None:
    """Endpoint matrix + sepset CSR from the learner into a host CausalGraph."""
    final = learner._to_general_graph()
    np.copyto(cg.G.graph, np.asarray(final.graph))
    S_all, ranges = _sepset_csr(learner, int(cg.G.num_vars))
    cg.sepset_S = S_all
    cg.sepset_ranges = ranges


@records_timings
def run_rai_cuda(
    cg, alpha: float = 0.05, *, dtype=None, max_cond_set_size=None
) -> None:
    """RAI-stable-cuda on ``cg``: reference parity path with GPU CI tests.

    ``cg`` must carry a ``CIT`` holding the raw data (``cg.set_ind_test``).
    ``max_cond_set_size`` overrides the limit already configured on ``cg``.
    On return ``cg.G.graph`` holds the final endpoint matrix (TAIL=-1,
    ARROW=1) and ``cg.sepset_S`` / ``cg.sepset_ranges`` the reference
    sepsets (non-empty ones) in CSR form. Wall-time stats land in
    ``run_rai_cuda.last_times``.
    """
    test = cg.test
    data = getattr(test, "data", None)
    if data is None:
        msg = "run_rai_cuda needs cg.test to hold the raw data matrix (CIT)"
        raise ValueError(msg)
    names = [f"X{i + 1}" for i in range(data.shape[1])]
    learner = RAIStableCudaLearner(
        data=data,
        alpha=float(alpha),
        indep_test=test.method,
        node_names=names,
        verbose=False,
        dtype=dtype,
        ci_test=test,
    )
    limit = max_cond_set_size
    if limit is None:
        limit = getattr(cg, "max_cond_set_size", None)
    learner.max_cond_set_size = limit
    learner.ci_test.max_cond_set_size = limit
    t0 = time.perf_counter()
    final = learner.learn_structure()
    total = time.perf_counter() - t0
    # Reuse the graph learn_structure() already produced (no second pass).
    np.copyto(cg.G.graph, np.asarray(final.graph))
    S_all, ranges = _sepset_csr(learner, int(cg.G.num_vars))
    cg.sepset_S = S_all
    cg.sepset_ranges = ranges
    run_rai_cuda.last_times = {
        "total": total,
        "n_ci_tests": learner.n_ci_tests,
        **{k: learner.stats[k] for k in learner.stats if k.startswith("gpu_")},
    }


__all__ = [
    "RAIStableCudaLearner",
    "compact_potential_parents",
    "complete_undirected_endpoints",
    "exit_cond",
    "filter_pairs",
    "lowest_topo",
    "pair_mask_endogenous",
    "pair_mask_exogenous",
    "partial_topological_layers",
    "run_rai_cuda",
]
