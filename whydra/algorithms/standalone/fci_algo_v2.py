import itertools
import math
import time
from itertools import combinations, islice
from math import comb
from typing import TYPE_CHECKING, cast

import numpy as np

from ... import background_knowledge as _background_knowledge_exports
from ...background_knowledge import (
    BKPhase,
    MatrixEncoding,
    apply_bk_checkpoint,
)
from ..graph_core import CIT
from ..graph_core._optional_imports import Parallel, delayed
from ..graph_core._timings import records_timings
from ..graph_core.conditioning import (
    bounded_size,
    configure_conditioning,
    validate_cuda_background_knowledge,
)
from ..graph_core.independence_tests import _builtin_fisherz, _threshold_pvalue
from ..graph_core.validation import effective_n_jobs
from . import pag_rules
from . import pag_rules as _pag_rules_exports
from .ci_adapters import LegacyCITest
from .fci_search import orient_colliders
from .pag_rules import (
    PAG_ARROW,
    PAG_CIRCLE,
    PAG_TAIL,
    _clear_triple,
    _orient_skeleton,
    _orientation_guard,
)
from .skeleton import SkeletonDiscovery

if TYPE_CHECKING:
    from ..graph_core.cpu_cuda_fisher_bridge import CudaFisherZ

BKOrientationGuard = _background_knowledge_exports.BKOrientationGuard
faith_check = _pag_rules_exports.faith_check
minDiscPath = _pag_rules_exports.minDiscPath
minUncovCircPath = _pag_rules_exports.minUncovCircPath
minUncovPdPath = _pag_rules_exports.minUncovPdPath

# Avoid process startup for a PDS pass smaller than a modest numerical batch.
# This is an execution policy, never a cap on tested conditioning sets.
_PDS_PARALLEL_MIN_TESTS = 8192


# =============================================================================
#  Biconnected components for the pds_path optimization.
# =============================================================================


def _find_biconnected_components(adj_matrix):
    """Run an iterative low-link depth-first search using canonical (min, max) edge pairs."""
    skeleton = (adj_matrix != 0) | (adj_matrix.T != 0)
    np.fill_diagonal(skeleton, False)
    p = len(skeleton)
    discovered = [-1] * p
    low = [0] * p
    parent = [-1] * p
    timer = 0
    edges, components = [], []
    for root in range(p):
        if discovered[root] >= 0:
            continue
        discovered[root] = low[root] = timer
        timer += 1
        frames = [(root, iter(np.flatnonzero(skeleton[root])))]
        while frames:
            u, neighbors = frames[-1]
            v = next(neighbors, None)
            if v is None:
                frames.pop()
                ancestor = parent[u]
                if ancestor >= 0:
                    low[ancestor] = min(low[ancestor], low[u])
                    if low[u] >= discovered[ancestor]:
                        component = set()
                        while edges:
                            edge = edges.pop()
                            component.add(tuple(sorted(edge)))
                            if edge == (ancestor, u):
                                break
                        if component:
                            components.append(component)
                continue
            v = int(v)
            if discovered[v] < 0:
                parent[v] = u
                edges.append((u, v))
                discovered[v] = low[v] = timer
                timer += 1
                frames.append((v, iter(np.flatnonzero(skeleton[v]))))
            elif v != parent[u] and discovered[v] < discovered[u]:
                edges.append((u, v))
                low[u] = min(low[u], discovered[v])
    return components


def _build_edge_to_component_map(components):
    """Map each edge (i, j) to its component index."""
    edge_map = {}
    for idx, comp in enumerate(components):
        for edge in comp:
            edge_map[edge] = idx
    return edge_map


def _nodes_in_component(component):
    """Return the set of nodes in the component."""
    nodes = set()
    for i, j in component:
        nodes.add(i)
        nodes.add(j)
    return nodes


# =============================================================================
#  Legacy external-endpoint adapters for single-source Possible-D-SEP
# =============================================================================


def _legacy_endpoints(amat):
    # This public helper predates the switch to Whydra's internal PAG encoding.
    values = np.asarray(amat)
    return np.where(values == 2, PAG_ARROW, np.where(values != 0, PAG_CIRCLE, 0))


def legal_path(a, b, c, amat):
    """Check whether a path may continue through the triple in the legacy endpoint matrix."""
    return pag_rules.legal_path(a, b, c, _legacy_endpoints(amat))


def qreach(x, amat):
    """Return the nodes reachable by legal triples in the legacy endpoint matrix."""
    return pag_rules.qreach(x, _legacy_endpoints(amat))


def _restrict_pdsep_mask_pds_path(amat, pdsep_mask):
    """Restrict Possible-D-SEP rows to the v2 ``pds_path`` union.

    For each ``x``, union ``qreach(x) ∩ component(x, y)`` over neighbours ``y``.
    """
    x_uses_c, node_in_comp, has_missing = _pds_path_gemm_factors(np.asarray(amat) != 0)
    union = (x_uses_c.T.astype(np.int32) @ node_in_comp.astype(np.int32)) > 0
    mask = np.asarray(pdsep_mask) != 0
    out = mask & union
    if np.any(has_missing):
        out[has_missing] = mask[has_missing]
    return out.astype(np.asarray(pdsep_mask).dtype, copy=False)


def _biconnected_components_skeleton(skeleton):
    """Use adjacency lists to obtain the same partition as scanning ``for v in range(p)``."""
    p = int(skeleton.shape[0])
    nbrs = [np.flatnonzero(skeleton[u]).tolist() for u in range(p)]
    visited = [False] * p
    disc = [0] * p
    low = [0] * p
    parent = [-1] * p
    timer = [0]
    stack = []
    components = []

    def _dfs(u):
        children = 0
        visited[u] = True
        disc[u] = low[u] = timer[0]
        timer[0] += 1

        for v in nbrs[u]:
            if not visited[v]:
                children += 1
                parent[v] = u
                stack.append((u, v))
                _dfs(v)
                low[u] = min(low[u], low[v])
                if (parent[u] == -1 and children > 1) or (
                    parent[u] != -1 and low[v] >= disc[u]
                ):
                    component = set()
                    while stack and stack[-1] != (u, v):
                        edge = stack.pop()
                        component.add((min(edge), max(edge)))
                    if stack:
                        edge = stack.pop()
                        component.add((min(edge), max(edge)))
                    components.append(component)
            elif v != parent[u] and disc[v] < disc[u]:
                stack.append((u, v))
                low[u] = min(low[u], disc[v])

    for i in range(p):
        if not visited[i]:
            _dfs(i)
            if stack:
                component = set()
                while stack:
                    edge = stack.pop()
                    component.add((min(edge), max(edge)))
                components.append(component)

    return components


def _pds_path_gemm_factors(adj):
    """Factors for ``union = (x_uses_c.T @ node_in_comp) > 0``.

    ``x_uses_c[c, x]`` if some directed neighbour ``y`` of ``x`` lies on
    biconnected component ``c``. ``has_missing[x]`` if some neighbour edge
    is absent from the component map (then row ``x`` keeps the full Possible-D-SEP).
    """
    adj = np.asarray(adj, dtype=bool)
    p = int(adj.shape[0])
    skeleton = adj | adj.T
    np.fill_diagonal(skeleton, False)
    components = _biconnected_components_skeleton(skeleton)
    n_comp = len(components)
    node_in_comp = np.zeros((n_comp, p), dtype=np.uint8)
    edge_to_c = {}
    for ci, edges in enumerate(components):
        for i, j in edges:
            i = int(i)
            j = int(j)
            edge_to_c[(i, j)] = ci
            node_in_comp[ci, i] = 1
            node_in_comp[ci, j] = 1
    x_uses_c = np.zeros((n_comp, p), dtype=np.uint8)
    has_missing = np.zeros(p, dtype=bool)
    xs, ys = np.nonzero(adj)
    for x, y in zip(xs.tolist(), ys.tolist(), strict=False):
        if x == y:
            continue
        ci = edge_to_c.get((x, y) if x < y else (y, x))
        if ci is None:
            has_missing[x] = True
        else:
            x_uses_c[ci, x] = 1
    return x_uses_c, node_in_comp, has_missing


def _restrict_pdsep_mask_pds_path_cuda(amat, pdsep_mask):
    """In-place Possible-D-SEP restrict: CPU BCC + device GEMM.

    Hopcroft–Tarjan on one GPU thread was slower than amat D2H (~1 MB at
    ``p=1000``), CPU biconnected components, then
    ``union = (x_uses_c.T @ node_in_comp) > 0`` on device.
    """
    import torch

    adj_np = (amat != 0).detach().cpu().numpy()
    x_uses_c, node_in_comp, has_missing = _pds_path_gemm_factors(adj_np)
    mask = pdsep_mask != 0
    a = torch.as_tensor(
        np.ascontiguousarray(x_uses_c),
        device=pdsep_mask.device,
        dtype=torch.float32,
    )
    b = torch.as_tensor(
        np.ascontiguousarray(node_in_comp),
        device=pdsep_mask.device,
        dtype=torch.float32,
    )
    union = (a.T @ b) > 0
    out = mask & union
    if np.any(has_missing):
        hm = torch.as_tensor(has_missing, device=pdsep_mask.device)
        out = torch.where(hm.unsqueeze(1), mask, out)
    pdsep_mask.copy_(out.to(dtype=pdsep_mask.dtype))
    torch.cuda.synchronize(pdsep_mask.device)


def _sepset_as_pairs(sepset) -> dict:
    """``(i, j) → sorted S`` for test equality (one sepset per pair)."""
    if callable(getattr(sepset, "packed", None)):
        from ..graph_core.cuda_pdsep import sepset_csr_to_dict

        S, ranges = sepset.packed()
        d = sepset_csr_to_dict(S, ranges, int(np.asarray(ranges).shape[0]))
        return {k: tuple(sorted(v)) for k, v in d.items() if v}
    out = {}
    for k, S in sepset.items():
        if not S:
            continue
        out[k] = tuple(sorted(int(v) for v in S))
    return out


def _cl_node_name_key(i):
    """Return the node name ``X{i+1}`` for the zero-based node index i."""
    return f"X{i + 1}"


def _adjacency_lists(amat):
    """Build adjacency lists for all nodes in ascending index order.

    Parameters
    ----------
    amat : array-like
        PAG matrix in Whydra's layout. On edge u—v, amat[u][v] is the
        mark at v. A nonzero value means the edge exists.

    Returns
    -------
    list of lists of int
        A list of p lists. adj[i] contains the indices of nodes adjacent to i.
    """
    p = len(amat)
    return [[j for j in range(p) if amat[i][j] != 0] for i in range(p)]


def _semi_directed_reachable(amat, adj, src):
    """Destinations of nonempty walks leaving only tail or circle endpoints."""
    visited = set()
    fringe = {src}
    while fringe:
        next_layer = {
            target
            for start in fringe
            for target in adj[start]
            if amat[target][start] in (PAG_TAIL, PAG_CIRCLE)
        }
        fringe = next_layer.difference(visited)
        visited.update(fringe)
    return visited


def _exists_semi_directed_path(amat, src, dst, adj=None):
    """Check whether a semi-directed path leads from src to dst.

    Parameters
    ----------
    amat : array-like
        PAG matrix in Whydra's layout.
    src : int
        Index of the starting node.
    dst : int
        Index of the target node.
    adj : sequence of sequences or None
        Adjacency lists. If None, build them from amat.

    Returns
    -------
    bool
        True if a semi-directed path leads from src to dst.
        A path of length zero does not count.
    """
    if adj is None:
        adj = _adjacency_lists(amat)
    return dst in _semi_directed_reachable(amat, adj, src)


def _possible_dsep(amat, x, y, adj=None):
    """Build Whydra v2's directional candidate pool on the current PAG.

    Direct neighbors other than y are admissible. An additional vertex
    must share an arrowheaded neighbor with x. Thus the rooted expansion
    needs only two layers. For the compatibility ancestor filter, a vertex
    may use any adjacent vertex in an earlier discovery position, including
    a previously considered vertex that failed its own ancestor filter.
    """
    if adj is None:
        adj = _adjacency_lists(amat)
    neighbors = [node for node in adj[x] if node != y and node != x]
    discovered = set(neighbors)
    choices = discovered.copy()
    second_layer = {}
    for middle in neighbors:
        if amat[x][middle] == PAG_ARROW:
            for candidate in adj[middle]:
                if (
                    candidate not in discovered
                    and candidate not in (x, y)
                    and amat[candidate][middle] == PAG_ARROW
                ):
                    second_layer[candidate] = None

    closure = {}
    for target in second_layer:
        for support in adj[target]:
            if support not in discovered:
                continue
            if support not in closure:
                closure[support] = _semi_directed_reachable(amat, adj, support)
            if target in closure[support] or x in closure[support]:
                choices.add(target)
                break
        discovered.add(target)
    return sorted(choices, key=_cl_node_name_key, reverse=True)


def _side_order(amat, i, j):
    """Order the endpoints of edge (i, j), where i < j, as causal-learn's Edge does.

    The causal-learn Edge constructor swaps node1 and node2 if the mark at
    node1 (the lower-indexed node) is ARROW and the mark at node2 is TAIL or
    CIRCLE. In Whydra's layout, the mark at node i is amat[j][i].

    Parameters
    ----------
    amat : array-like
        PAG matrix in Whydra's layout.
    i : int
        Lower node index of the edge.
    j : int
        Higher node index of the edge.

    Returns
    -------
    tuple of int
        A tuple (a, b) giving the endpoint order after any swap.
    """
    if amat[j][i] == PAG_ARROW and amat[i][j] in (PAG_CIRCLE, PAG_TAIL):
        return (j, i)
    return (i, j)


def _restrict_to_component(
    pds,
    a,
    b,
    components,
    edge_to_comp,
):
    """Restrict Possible-D-SEP to the biconnected component of edge (a, b).

    Used only with use_pds_path=True (Colombo et al. 2012, Def. 3.4).
    Nodes outside the biconnected component of edge (a, b) cannot lie
    on a simple path between a and b, so discard them.

    Parameters
    ----------
    pds : sequence of int
        List of Possible-D-SEP node indices, in the order to preserve.
    a : int
        First node of the edge.
    b : int
        Second node of the edge.
    components : list
        List of biconnected components
        (see _find_biconnected_components).
    edge_to_comp : dict
        Mapping from an edge to its component index
        (see _build_edge_to_component_map).

    Returns
    -------
    list of int
        The indices in pds that belong to the component of edge (a, b).
        If no component contains the edge, return pds unchanged.
    """
    comp_idx = edge_to_comp.get((min(a, b), max(a, b)))
    if comp_idx is None:
        return list(pds)
    allowed = _nodes_in_component(components[comp_idx])
    return [node for node in pds if node in allowed]


def _edge_context(
    amat,
    a,
    b,
    adj,
    components=None,
    edge_to_comp=None,
    pds_cache=None,
):
    """Build the context for checking edge a—b with _test_edge.

    The context captures the graph inputs to the CI search for this edge:
    PDS on both sides and the adjacency lists of both endpoints. With the
    same test inputs, matching contexts give the same _test_edge result.

    Parameters
    ----------
    amat : array-like
        PAG matrix in Whydra's layout.
    a : int
        First node returned by _side_order for this edge.
    b : int
        Second node of the edge.
    adj : sequence of sequences
        Adjacency lists for amat.
    components : list or None
        List of biconnected components or None
        (use_pds_path=False).
    edge_to_comp : dict or None
        Mapping from an edge to its component, or None.
    pds_cache : dict, optional
        Precomputed Possible-D-SEP lists for both directions, keyed by (a, b).

    Returns
    -------
    dict
        A dictionary with keys 'a', 'b', 'pds_a', 'pds_b', 'adj_a', and 'adj_b'.
    """
    if pds_cache is not None and (a, b) in pds_cache:
        pds_a, pds_b = pds_cache[(a, b)]
    else:
        pds_a = _possible_dsep(amat, a, b, adj)
        pds_b = _possible_dsep(amat, b, a, adj)
    if components is not None:
        pds_a = _restrict_to_component(pds_a, a, b, components, edge_to_comp)
        pds_b = _restrict_to_component(pds_b, a, b, components, edge_to_comp)
    return {
        "a": a,
        "b": b,
        "pds_a": pds_a,
        "pds_b": pds_b,
        "adj_a": list(adj[a]),
        "adj_b": list(adj[b]),
    }


def _pds_levels(pds, adj_x, adj_y, limit):
    """Yield fresh CI requests after the skeleton exhausted both neighborhoods."""
    seen_x, seen_y = set(adj_x), set(adj_y)
    for size in range(2, int(min(limit, len(pds))) + 1):
        yield (
            size,
            (
                group
                for group in combinations(pds, size)
                if not seen_x.issuperset(group) and not seen_y.issuperset(group)
            ),
        )


def _search_separator(
    x, y, pds, adj_x, adj_y, indepTest, suffStat, alpha, m_max, pmax_start
):
    peak, count, depth = pmax_start, 0, 0
    for size, candidates in _pds_levels(
        pds, adj_x, adj_y, bounded_size(indepTest, m_max)
    ):
        for group in candidates:
            value = indepTest(suffStat, x, y, list(group))
            peak = max(peak, value)
            count += 1
            depth = size
            if value >= alpha:
                return set(group), count, peak, depth
    return None, count, peak, depth


def _directional_search(ctx, search, options, initial_peak):
    """Commit the first accepted witness in the edge's declared side order."""
    count, peak, depth, witness = 0, initial_peak, 0, None
    for left, right in (("a", "b"), ("b", "a")):
        witness, calls, peak, size = search(
            ctx[left],
            ctx[right],
            ctx["pds_" + left],
            ctx["adj_" + left],
            ctx["adj_" + right],
            *options,
            peak,
        )
        count += calls
        depth = max(depth, size)
        if witness is not None:
            break
    return {
        "removed": witness is not None,
        "sepset": witness,
        "tests": count,
        "pmax": peak,
        "max_ord": depth,
    }


def _test_edge(ctx, indepTest, suffStat, alpha, m_max, pmax_start):
    return _directional_search(
        ctx, _search_separator, (indepTest, suffStat, alpha, m_max), pmax_start
    )


def _pdsep_edge_worker(
    amat_snapshot,
    i,
    j,
    indepTest,
    suffStat,
    alpha,
    m_max,
    pmax_start,
    components,
    edge_to_comp,
):
    """Run a speculative check of one edge on a snapshot as one joblib task.

    Parameters
    ----------
    amat_snapshot : array-like
        Copy of the PAG matrix at the start of the removal phase.
    i : int
        Lower node index of the edge.
    j : int
        Higher node index of the edge.
    indepTest : callable
        Independence test function.
    suffStat : object
        Sufficient statistics.
    alpha : float
        Significance level.
    m_max : int or float
        Maximum separating set size.
    pmax_start : float
        Initial maximum p-value for pair (i, j).
    components : list or None
        Biconnected components or None.
    edge_to_comp : dict or None
        Mapping from an edge to its component, or None.

    Returns
    -------
    tuple
        A tuple (ctx, result) with the edge context and the _test_edge result.
    """
    adj = _adjacency_lists(amat_snapshot)
    a, b = _side_order(amat_snapshot, i, j)
    ctx = _edge_context(amat_snapshot, a, b, adj, components, edge_to_comp)
    result = _test_edge(ctx, indepTest, suffStat, alpha, m_max, pmax_start)
    return ctx, result


def _context_test_bound(ctx, m_max, stop_at, *, weighted=False):
    """Bound eligible PDS tests, accounting for the two adjacency exclusions.

    Inclusion/exclusion counts subsets without materializing them. Early
    separators can only reduce work; stop once the dispatch threshold is met.
    This bound selects execution policy and never prunes the actual search.
    """
    total = 0
    adj_a, adj_b = set(ctx["adj_a"]), set(ctx["adj_b"])
    for key in ("pds_a", "pds_b"):
        pool = set(ctx[key])
        count = len(pool)
        upper = count if m_max == float("inf") else min(count, int(m_max))
        a, b = len(pool & adj_a), len(pool & adj_b)
        both = len(pool & adj_a & adj_b)
        for size in range(2, upper + 1):
            queries = (
                comb(count, size) - comb(a, size) - comb(b, size) + comb(both, size)
            )
            total += queries * ((size + 2) ** 3 / 256) if weighted else queries
            if total >= stop_at:
                return total
    return total


# =============================================================================
#  Batched separator search (FCI-stable-cuda; semantics of _search_separator)
# =============================================================================

_PDSEP_BATCH_CHUNK = 1 << 16
# Per-side cap on speculative Possible-D-SEP enumeration: an edge whose subset
# count exceeds this is not speculated and is recomputed on the live graph.
_PDSEP_SPEC_CAP = 1 << 16
# Total speculative edge-work as a multiple of the edge count: each wave
# re-speculates the remaining edges after a removal, so this bounds the wasted
# re-enumeration when removals are frequent.
_PDSEP_SPEC_FACTOR = 16
# Context-keyed cache of the per-side PDS combinatorics. A wave re-speculates
# its whole suffix after every removal, but most edges' contexts are unchanged
# from the previous wave; keying on ``(p, m_max, pds, adj_x, adj_y)`` lets those
# edges skip the (dominant) Python/numpy enumeration. Cleared per pdsep call.
_PDSEP_ENUM_CACHE: dict = {}


def _search_separator_batched(
    x,
    y,
    pds,
    adj_x,
    adj_y,
    batch_pvals,
    alpha,
    m_max,
    pmax_start,
    chunk=_PDSEP_BATCH_CHUNK,
):
    """Evaluate one cardinality per GPU batch, accounting for its full cost."""
    peak, count, depth = pmax_start, 0, 0
    for size, candidates in _pds_levels(pds, adj_x, adj_y, m_max):
        while group := list(islice(candidates, max(1, chunk))):
            values = np.asarray(
                batch_pvals(x, y, np.asarray(group, dtype=np.int32)), dtype=np.float64
            )
            valid = np.isfinite(values)
            count += len(group)
            depth = size
            if valid.any():
                peak = max(peak, float(values[valid].max()))
                accepted = np.flatnonzero(valid & (values >= alpha))
                if len(accepted):
                    return set(group[accepted[0]]), count, peak, depth
    return None, count, peak, depth


def _comb_index_table(n, k):
    """Cache the (C(n, k), k) int32 table of lexicographic combinations of ``range(n)``.

    Materializing combinations with ``itertools`` is the hot Python loop of the
    pdsep host side; the *shape* of that loop depends only on ``(n, k)``, which
    takes a handful of distinct values per pdsep step. Precompute the index
    table once per ``(n, k)`` and gather the conditioned indices with a single
    vectorized ``numpy`` take, so an edge with a PDS of the same length reuses
    the table instead of re-generating every combination.
    """
    key = (n, k)
    table = _COMB_INDEX_CACHE.get(key)
    if table is None:
        m = math.comb(n, k)
        table = np.fromiter(
            itertools.chain.from_iterable(combinations(range(n), k)),
            dtype=np.int32,
            count=m * k,
        ).reshape(m, k)
        if len(_COMB_INDEX_CACHE) < 4096:
            _COMB_INDEX_CACHE[key] = table
    return table


_COMB_INDEX_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _enumerate_pds_blocks_py(pds, adj_x, adj_y, m_max, p, cap):
    """Per-size non-adjacent subsets of ``pds``, ``(size, lex)`` order, or ``None``.

    Returns a list ``[(size, arr_mk)]`` with ``arr_mk`` the kept combinations of
    that size as an ``int32`` ``(m, size)`` array, in the exact order
    ``_search_separator`` tests them (sizes ascending, lexicographic within a
    size, skipping ``S ⊆ adj_x`` / ``S ⊆ adj_y``). ``None`` as soon as more than
    ``cap`` subsets are collected so the caller can fall back to the bounded
    per-size flush search.

    The skip filter is vectorized over the whole size block with ``numpy``
    instead of a Python loop per combination, and the combinations themselves
    come from a cached ``(len(pds), size)`` index table (see
    ``_comb_index_table``) gathered through ``pds`` — an edge whose PDS length
    repeats therefore never re-runs ``itertools``. The whole result is also
    memoized on the context ``(p, m_max, pds, adj_x, adj_y)`` so a wave that
    re-speculates an unchanged edge reuses the previous blocks.
    """
    key = (p, m_max, tuple(pds), tuple(adj_x), tuple(adj_y), cap)
    cached = _PDSEP_ENUM_CACHE.get(key)
    if cached is not None:
        return cached if cached is not False else None
    in_x = np.zeros(p, dtype=bool)
    in_x[np.asarray(list(adj_x), dtype=np.int64)] = True
    in_y = np.zeros(p, dtype=bool)
    in_y[np.asarray(list(adj_y), dtype=np.int64)] = True
    upper = len(pds) if m_max == float("inf") else min(len(pds), int(m_max))
    n_pds = len(pds)
    pds_arr = np.asarray(pds, dtype=np.int32)
    blocks: list[tuple[int, np.ndarray]] = []
    total = 0
    for size in range(2, upper + 1):
        if size > n_pds:
            break
        arr = pds_arr[_comb_index_table(n_pds, size)]
        keep = ~(in_x[arr].all(axis=1) | in_y[arr].all(axis=1))
        arr = np.ascontiguousarray(arr[keep])
        if arr.shape[0] == 0:
            continue
        total += arr.shape[0]
        if total > cap:
            _PDSEP_ENUM_CACHE[key] = False
            return None
        blocks.append((size, arr))
    if len(_PDSEP_ENUM_CACHE) < 1 << 16:
        _PDSEP_ENUM_CACHE[key] = blocks
    return blocks


def _enumerate_pds_blocks(pds, adj_x, adj_y, m_max, p, cap):
    """PDS conditioning blocks, generated by the C++ bitset enumerator.

    The C++ path counts the allowed subsets first (all k-subsets of PDS minus
    those entirely in adj(a) or adj(b)) and returns immediately when the count
    exceeds ``cap``; it never materializes the forbidden C(|PDS|, k) blocks.
    The Python implementation is kept as a fallback for a CPU-only build.
    """
    try:
        from ..graph_core import _cuda

        upper = len(pds) if m_max == float("inf") else min(len(pds), int(m_max))
        ok, sizes, flat, offs = _cuda.fci_host.pds_blocks(
            int(p),
            [int(v) for v in pds],
            [int(v) for v in adj_x],
            [int(v) for v in adj_y],
            upper,
            int(cap),
        )
        if not ok:
            return None
        if not sizes:
            return []
        arr = np.asarray(flat, dtype=np.int32)
        return [
            (
                int(sizes[i]),
                arr[int(offs[i]) : int(offs[i + 1])].reshape(-1, int(sizes[i])),
            )
            for i in range(len(sizes))
        ]
    except (ImportError, AttributeError):
        return _enumerate_pds_blocks_py(pds, adj_x, adj_y, m_max, p, cap)


def _merged_side_search(blocks, alpha, pmax_start):
    """Replay ``_search_separator``'s selection on precomputed p-value blocks.

    ``blocks`` is ``[(size, S_mk, pvals_m)]`` in ``(size, lex)`` order (from
    ``_enumerate_pds_blocks``). Returns ``(sepset, tests, pmax, max_size)`` with
    the exact values the sequential search would report: the first subset with
    ``pval >= alpha`` in ``(size, lex)`` order, ``tests`` counted up to the end
    of that size's buffer (``_flush`` semantics), ``pmax`` the maximum finite
    p-value over that prefix.
    """
    pmax = pmax_start
    total = sum(b[2].size for b in blocks)
    if total == 0:
        return None, 0, pmax, 0
    pvals = np.concatenate([b[2] for b in blocks]) if len(blocks) > 1 else blocks[0][2]
    sizes = (
        np.concatenate([np.full(b[2].size, b[0], dtype=np.int64) for b in blocks])
        if len(blocks) > 1
        else np.full(blocks[0][2].size, blocks[0][0], dtype=np.int64)
    )
    finite = np.isfinite(pvals)
    hit = finite & (pvals >= float(alpha))
    h = int(np.argmax(hit)) if hit.any() else -1
    if h < 0:
        end = total - 1
        size = int(sizes[end])
    else:
        size = int(sizes[h])
        end = h
        while end + 1 < total and sizes[end + 1] == size:
            end += 1
    fin = finite[: end + 1]
    if fin.any():
        m = float(pvals[: end + 1][fin].max())
        pmax = max(pmax, m)
    if h < 0:
        return None, end + 1, pmax, size
    # locate the hit's block and local row to recover the node set
    off = 0
    for sz, arr, _pv in blocks:
        if sz == size and h < off + arr.shape[0]:
            sepset = {int(v) for v in arr[h - off]}
            return sepset, end + 1, pmax, size
        off += arr.shape[0]
    msg = "hit block not found"
    raise AssertionError(msg)


def _test_edge_batched(ctx, batch_pvals, alpha, m_max, pmax_start):
    return _directional_search(
        ctx, _search_separator_batched, (batch_pvals, alpha, m_max), pmax_start
    )


def _pdsep_speculate(
    edges_list,
    amat,
    components,
    edge_to_comp,
    p,
    m_max,
    pmax_host,
    batch_pvals,
    alpha,
):
    """One Possible-D-SEP CI pass on the frozen ``amat`` for ``edges_list``.

    Builds every edge's Possible-D-SEP combinatorics on the snapshot, groups the
    conditioning subsets by size across ALL edges/sides, and issues one
    ``batch_pvals`` call per size (instead of one call per edge/size — the
    dominant cost at p=50 was hundreds of ~0.25 ms launches).

    Returns ``dict[(i, j)] -> {'ctx': ctx, 'result': result}`` where ``ctx`` is
    the snapshot edge context and ``result`` has the same keys as
    ``_test_edge_batched`` and is bit-identical to running ``_test_edge`` on the
    snapshot. ``result`` is ``None`` for an edge whose combinatorics exceed
    ``_PDSEP_SPEC_CAP`` (the caller recomputes that edge on the live graph).

    The caller reuses a snapshot result only while the live edge context still
    equals the snapshot ``ctx`` — exactly the ``n_jobs != 1`` reuse rule of the
    reference ``pdsep_v2`` (``speculative[(i, j)][0] == ctx_live``), which the
    reference documents as producing the same graph/sepset/pMax as the
    sequential ``n_jobs == 1`` run.
    """
    adj_snap = _adjacency_lists(amat)
    pds_cache = None
    try:
        import torch

        from ..graph_core import _cuda

        amat32 = np.ascontiguousarray(amat, dtype=np.int32)
        pairs, keys = [], []
        for i, j in edges_list:
            a, b = _side_order(amat, i, j)
            pairs.extend(((a, b), (b, a)))
            keys.append((a, b))
        flat = _cuda.fci_host.pds_batch(torch.from_numpy(amat32), pairs)
        pds_cache = {key: (flat[2 * n], flat[2 * n + 1]) for n, key in enumerate(keys)}
    except (ImportError, AttributeError):
        pds_cache = None
    plans: dict[tuple[int, int], dict] = {}
    by_k: dict[int, list] = {}
    for i, j in edges_list:
        a, b = _side_order(amat, i, j)
        ctx = _edge_context(
            amat, a, b, adj_snap, components, edge_to_comp, pds_cache=pds_cache
        )
        entry: dict = {"ctx": ctx, "result": None}
        plans[(i, j)] = entry
        ba = _enumerate_pds_blocks(
            ctx["pds_a"], ctx["adj_a"], ctx["adj_b"], m_max, p, _PDSEP_SPEC_CAP
        )
        if ba is None:
            continue  # too large to speculate; recomputed on the live graph
        bb = _enumerate_pds_blocks(
            ctx["pds_b"], ctx["adj_b"], ctx["adj_a"], m_max, p, _PDSEP_SPEC_CAP
        )
        if bb is None:
            continue
        blk_a, blk_b = [], []
        for size, arr in ba:
            blk = [size, arr, None]
            blk_a.append(blk)
            by_k.setdefault(size, []).append((blk, a, b))
        for size, arr in bb:
            blk = [size, arr, None]
            blk_b.append(blk)
            by_k.setdefault(size, []).append((blk, b, a))
        entry["ba"] = blk_a
        entry["bb"] = blk_b
    for _size, items in by_k.items():
        # One GPU call per conditioning-set size across ALL edges/sides. The
        # per-edge p-values are zero-copy views into this size's output, laid
        # out in the caller's per-edge block order, so the group/scatter step
        # costs one ``np.repeat`` per side instead of one ``np.full`` per block.
        S_arr = np.concatenate([it[0][1] for it in items], axis=0)
        n_items = len(items)
        counts = np.fromiter(
            (it[0][1].shape[0] for it in items), dtype=np.int64, count=n_items
        )
        xs = np.repeat(
            np.fromiter((it[1] for it in items), dtype=np.int32, count=n_items),
            counts,
        )
        ys = np.repeat(
            np.fromiter((it[2] for it in items), dtype=np.int32, count=n_items),
            counts,
        )
        pv = np.asarray(batch_pvals(xs, ys, S_arr), dtype=np.float64)
        off = 0
        for idx in range(n_items):
            n = int(counts[idx])
            items[idx][0][2] = pv[off : off + n]
            off += n
    for (i, j), entry in plans.items():
        if entry["result"] is not None or "ba" not in entry:
            continue
        pmax_start = max(pmax_host[i, j], pmax_host[j, i])
        sepset, tests, pmax, mx = _merged_side_search(entry["ba"], alpha, pmax_start)
        if sepset is None:
            sepset, tb, pmax_b, mxb = _merged_side_search(
                entry["bb"], alpha, pmax_start
            )
            tests += tb
            pmax = max(pmax, pmax_b)
            mx = max(mx, mxb)
        entry["result"] = {
            "removed": sepset is not None,
            "sepset": sepset,
            "tests": tests,
            "pmax": pmax,
            "max_ord": mx,
        }
        del entry["ba"], entry["bb"]
    return plans


def _cuda_batch_pvals(ci_batcher):
    """Return one p-value per row of ``S_arr``.

    Possible-D-SEP can pass scalar endpoints or one endpoint pair per row.
    """
    lean = getattr(ci_batcher, "batch_pvalues", None)

    def batch_pvals(x, y, S_arr):
        if lean is not None:
            return np.asarray(lean(x, y, S_arr), dtype=np.float64)
        B = int(S_arr.shape[0])
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        xs = (
            np.full(B, int(x_arr), dtype=np.int64)
            if x_arr.ndim == 0
            else x_arr.astype(np.int64)
        )
        ys = (
            np.full(B, int(y_arr), dtype=np.int64)
            if y_arr.ndim == 0
            else y_arr.astype(np.int64)
        )
        return np.asarray(ci_batcher.batch(xs, ys, S_arr), dtype=np.float64)

    return batch_pvals


def _pdsep_v2_cuda(
    skel,
    p,
    sepSet,
    alpha,
    pMax,
    m_max,
    unfVect,
    use_pds_path=True,
    ci_batcher=None,
    protected_edges=(),
):
    """Run Possible-D-SEP for FCI-stable-cuda with CPU ``pdsep_v2`` semantics and GPU CI tests.

    Returns 'G', a host-side binary adjacency matrix; 'sepset', a mirrored
    CSR ``SepsetStore``; and 'pMax', the updated input p-value matrix.
    """
    import torch

    from ..graph_core.cuda_pdsep import SepsetStore, sepset_dict_to_csr

    _PDSEP_ENUM_CACHE.clear()
    t_imp0 = time.perf_counter()
    sk_in = skel["sk"]
    if torch.is_tensor(sk_in):
        G = (sk_in.detach().cpu().numpy() != 0).astype(int)
    else:
        G = (np.asarray(sk_in) != 0).astype(int)
    if torch.is_tensor(pMax):
        pmax_host = pMax.detach().cpu().numpy().copy()
    else:
        pmax_host = np.array(pMax, dtype=float, copy=True)
    sep_store = skel.get("sepset_store")
    S0 = skel.get("sepset_S")
    R0 = skel.get("sepset_ranges")
    if sep_store is None:
        if S0 is None:
            S0, R0 = sepset_dict_to_csr(sepSet, p)
        if isinstance(S0, torch.Tensor):
            S0 = S0.detach().cpu().numpy()
        if isinstance(R0, torch.Tensor):
            R0 = R0.detach().cpu().numpy()
        sep_store = SepsetStore(S0, R0)
    t_import = time.perf_counter() - t_imp0

    protected = {frozenset((int(x), int(y))) for x, y in protected_edges}

    amat = np.where(G != 0, PAG_CIRCLE, 0).astype(int)

    # --- Orient v-structures as in rule0, matching pdsep_v2 ---
    t_col0 = time.perf_counter()
    done = False
    if len(unfVect) == 0:
        try:
            from ..graph_core import _cuda

            flat, rng = sep_store.packed()
            amat = np.ascontiguousarray(amat, dtype=np.int32)
            _cuda.fci_host.pds_colliders(
                torch.from_numpy(amat),
                torch.from_numpy(np.ascontiguousarray(flat, dtype=np.int64)),
                torch.from_numpy(np.ascontiguousarray(rng, dtype=np.int32)),
            )
            done = True
        except (ImportError, AttributeError):
            done = False
    if not done:
        # Packed storage supplies an empty separator for missing pairs.
        orient_colliders(amat, sep_store, missing=())
    t_colliders = time.perf_counter() - t_col0

    # --- Visit edges in causal-learn order and skip protected edges ---
    edges = [
        (i, j)
        for i in range(p)
        for j in range(i + 1, p)
        if amat[i][j] != 0 and frozenset((i, j)) not in protected
    ]

    components = None
    edge_to_comp = None
    if use_pds_path:
        components = _find_biconnected_components(amat)
        edge_to_comp = _build_edge_to_component_map(components)

    t_ci0 = time.perf_counter()
    batch_pvals = _cuda_batch_pvals(ci_batcher)

    # --- Run Possible-D-SEP CI tests in waves on a graph snapshot ---
    # The graph stays fixed within a wave, so each edge can use its speculative
    # result: its context still matches the snapshot. The first removal ends
    # the wave. The next wave takes a fresh snapshot and handles the remaining
    # edges. This batches CI tests by separating set size instead of launching
    # one test per edge and size. With p=50, that avoids hundreds of launches
    # of about 0.25 ms each. ``_PDSEP_SPEC_FACTOR`` limits speculative work.
    # When edges are removed often, finish one edge at a time to avoid
    # recomputing waves that are likely to end early.
    # The cache in ``_enumerate_pds_blocks`` reuses the enumerated combinations
    # when an edge's context has not changed between waves.
    total_tests = 0
    max_ord = 0
    n_removed = 0
    n_spec_hits = 0
    n_recomputed = 0
    t_spec_build = 0.0
    t_live = 0.0
    adj = _adjacency_lists(amat)
    n_edges = len(edges)
    pos = 0
    speculated = 0
    spec_budget = _PDSEP_SPEC_FACTOR * max(n_edges, 1)
    while pos < n_edges:
        t0 = time.perf_counter()
        if speculated + (n_edges - pos) <= spec_budget:
            spec = _pdsep_speculate(
                edges[pos:],
                amat.tolist(),
                components,
                edge_to_comp,
                p,
                m_max,
                pmax_host,
                batch_pvals,
                alpha,
            )
            speculated += n_edges - pos
        else:
            spec = None
        t_spec_build += time.perf_counter() - t0
        t0 = time.perf_counter()
        restarted = False
        for kk in range(pos, n_edges):
            i, j = edges[kk]
            a, b = _side_order(amat, i, j)
            entry = spec.get((i, j)) if spec is not None else None
            result = entry["result"] if entry is not None else None
            if result is not None:
                n_spec_hits += 1
            else:
                ctx_live = _edge_context(amat, a, b, adj, components, edge_to_comp)
                result = _test_edge_batched(
                    ctx_live,
                    batch_pvals,
                    alpha,
                    m_max,
                    max(pmax_host[i, j], pmax_host[j, i]),
                )
                n_recomputed += 1
            if result["removed"]:
                amat[a][b] = 0
                amat[b][a] = 0
                adj[a] = [k for k in adj[a] if k != b]
                adj[b] = [k for k in adj[b] if k != a]
                sep_store.write(a, b, result["sepset"], mirror=True)
                n_removed += 1
                pos = kk + 1
                restarted = True
            pmax_host[a, b] = max(pmax_host[a, b], pmax_host[b, a], result["pmax"])
            pmax_host[b, a] = pmax_host[a, b]
            total_tests += result["tests"]
            max_ord = max(max_ord, result["max_ord"])
            if restarted:
                break
        t_live += time.perf_counter() - t0
        if not restarted:
            pos = n_edges
    t_ci = time.perf_counter() - t_ci0
    if torch.is_tensor(pMax):
        pMax.copy_(torch.as_tensor(pmax_host, device=pMax.device))

    pdsep_v2.last_times = {
        "colliders": t_colliders,
        "qreach": 0.0,
        "ci": t_ci,
        "spec_build": t_spec_build,
        "live": t_live,
        "n_removed": n_removed,
        "n_spec_hits": n_spec_hits,
        "n_recomputed": n_recomputed,
        "tests": total_tests,
        "max_ord": max_ord,
        "n_edges": n_edges,
        "sepset_import": t_import,
        "sepset_export": 0.0,
    }
    G_out = (amat != 0).astype(int)
    return {
        "G": G_out,
        "sepset": sep_store,
        "pMax": pMax,
        "allPdsep": [[] for _ in range(p)],
        "max_ord": max_ord,
    }


@records_timings
def pdsep_v2(
    skel,
    suffStat,
    indepTest,
    p,
    sepSet,
    alpha,
    pMax,
    m_max=float("inf"),
    pdsep_max=float("inf"),
    unfVect=None,
    n_jobs=1,
    verbose=False,
    use_pds_path=False,
    protected_edges=(),
    parallel_min_tests=0,
    ci_batcher=None,
):
    """Refine an oriented skeleton using ordered, per-edge PDS candidates.

    A successful CI request updates the live graph before the next edge.
    Parallel predictions are reused only if both current directional pools
    still agree with their snapshot. The optional path restriction limits
    candidates to the initial biconnected component (Colombo et al., 2012).
    """
    m_max = bounded_size(indepTest, m_max)
    if unfVect is None:
        unfVect = set()
    protected = {frozenset((int(x), int(y))) for x, y in protected_edges}

    if ci_batcher is not None:
        return _pdsep_v2_cuda(
            skel,
            p,
            sepSet,
            alpha,
            pMax,
            m_max,
            unfVect,
            use_pds_path=use_pds_path,
            ci_batcher=ci_batcher,
            protected_edges=protected_edges,
        )

    amat = np.where(np.asarray(skel["sk"]) != 0, PAG_CIRCLE, 0).astype(int)

    # --- Orient v-structures as in causal-learn's rule0 ---
    orient_colliders(amat, sepSet, unfVect)

    # --- Snapshot edges in causal-learn order and skip protected edges ---
    edges = [
        (i, j)
        for i in range(p)
        for j in range(i + 1, p)
        if amat[i][j] != 0 and frozenset((i, j)) not in protected
    ]

    components = None
    edge_to_comp = None
    if use_pds_path:
        components = _find_biconnected_components(amat)
        edge_to_comp = _build_edge_to_component_map(components)

    # Decide before constructing joblib workers. Preflight uses exactly the
    # same oriented snapshot as speculation, not a different PDS algorithm.
    if (
        isinstance(parallel_min_tests, bool)
        or not isinstance(parallel_min_tests, int)
        or parallel_min_tests < 0
    ):
        msg = "parallel_min_tests must be a nonnegative integer"
        raise ValueError(msg)
    parallel_jobs = n_jobs
    estimated_tests = None
    estimated_kernel_work = None
    preflight = {}
    if n_jobs != 1 and edges and parallel_min_tests:
        estimated_tests = 0
        estimated_kernel_work = 0
        initial_adj = _adjacency_lists(amat)
        for i, j in edges:
            a, b = _side_order(amat, i, j)
            ctx = _edge_context(amat, a, b, initial_adj, components, edge_to_comp)
            preflight[i, j] = ctx
            estimated_tests += _context_test_bound(ctx, m_max, 4 * parallel_min_tests)
            estimated_kernel_work += _context_test_bound(
                ctx, m_max, parallel_min_tests - estimated_kernel_work, weighted=True
            )
            if estimated_kernel_work >= parallel_min_tests:
                break
        if estimated_kernel_work < parallel_min_tests:
            parallel_jobs = 1

    # --- Run speculative checks in parallel only when n_jobs != 1 ---
    t_spec = time.perf_counter()
    speculative = {}
    if parallel_jobs != 1 and edges:
        amat_snapshot = np.copy(amat)
        parallel_results = Parallel(n_jobs=parallel_jobs)(
            delayed(_pdsep_edge_worker)(
                amat_snapshot,
                i,
                j,
                indepTest,
                suffStat,
                alpha,
                m_max,
                max(pMax[i, j], pMax[j, i]),
                components,
                edge_to_comp,
            )
            for (i, j) in edges
        )
        speculative = dict(zip(edges, parallel_results, strict=False))

    # --- Check and apply results sequentially ---
    t_valid = time.perf_counter()
    total_tests = 0
    max_ord = 0
    n_removed = 0
    n_speculative_hits = 0
    n_recomputed = 0
    live_ci_requests = 0
    invalidated_speculative_edges = 0
    adj = _adjacency_lists(amat)

    for i, j in edges:
        a, b = _side_order(amat, i, j)
        # The initial preflight remains valid until the first live deletion.
        ctx_live = preflight.get((i, j)) if n_removed == 0 else None
        if ctx_live is None:
            ctx_live = _edge_context(amat, a, b, adj, components, edge_to_comp)
        if (i, j) in speculative and speculative[(i, j)][0] == ctx_live:
            result = speculative[(i, j)][1]
            n_speculative_hits += 1
        else:
            result = _test_edge(
                ctx_live,
                indepTest,
                suffStat,
                alpha,
                m_max,
                max(pMax[i, j], pMax[j, i]),
            )
            n_recomputed += 1
            live_ci_requests += result["tests"]
            invalidated_speculative_edges += int((i, j) in speculative)

        if result["removed"]:
            amat[a][b] = 0
            amat[b][a] = 0
            adj[a] = [k for k in adj[a] if k != b]
            adj[b] = [k for k in adj[b] if k != a]
            sepSet[(a, b)] = set(result["sepset"])
            sepSet[(b, a)] = set(result["sepset"])
            n_removed += 1
        pMax[a, b] = max(pMax[a, b], pMax[b, a], result["pmax"])
        pMax[b, a] = pMax[a, b]
        total_tests += result["tests"]
        max_ord = max(max_ord, result["max_ord"])

    if verbose:
        t_end = time.perf_counter()
        print(
            f"[FCI v2] pdsep: edges={len(edges)} removed={n_removed} "
            f"ci_tests={total_tests} speculative_hits={n_speculative_hits} "
            f"recomputed={n_recomputed} n_jobs={n_jobs} "
            f"t_speculative={t_valid - t_spec:.1f}s t_validate={t_end - t_valid:.1f}s"
        )

    G = (amat != 0).astype(int)
    speculative_ci_requests = sum(
        result["tests"] for _ctx, result in speculative.values()
    )
    execution = {
        "phase": "pdsep_fci",
        "depth": None,
        "estimated_queries": estimated_tests,
        "estimated_kernel_work": estimated_kernel_work,
        "requested_workers": n_jobs,
        "effective_workers": effective_n_jobs(parallel_jobs) if speculative else 1,
        "backend": "processes" if speculative else "serial",
        "parallel_min_tests": parallel_min_tests,
        "selected_ci_requests": total_tests,
        "live_ci_requests": live_ci_requests,
        "speculative_ci_requests": speculative_ci_requests,
        "total_ci_requests": live_ci_requests + speculative_ci_requests,
        "speculative_hits": n_speculative_hits,
        "live_evaluated_edges": n_recomputed,
        "invalidated_speculative_edges": invalidated_speculative_edges,
    }
    return {
        "G": G,
        "sepset": sepSet,
        "pMax": pMax,
        "allPdsep": [],
        "max_ord": max_ord,
        "execution": execution,
    }


def _arrow_allowed(pag, x, y):
    """Check whether an arrowhead at y is compatible with the current mark.

    In Whydra's PAG matrix, ``pag[u][v]`` is the mark at v. For an arrow from x toward
    y, return True if the mark at y on edge x–y is not TAIL.
    """
    return pag[x][y] != PAG_TAIL


def _rule_r1(pag, bk, a, b, c, unfVect):
    """R1: a *-> b o-* c with a and c nonadjacent implies b -> c.

    Args:
        pag: PAG matrix in Whydra's layout.
        bk: BKOrientationGuard used to write endpoint marks.
        a: node with an arrowhead at b on edge a—b.
        b: middle node of the triple.
        c: node joined to b by an edge with a circle at b.
        unfVect: triples that must remain unoriented (empty in v2).
    """
    if pag[a][c] != 0 or pag[a][b] != PAG_ARROW or pag[c][b] != PAG_CIRCLE:
        return
    if not _arrow_allowed(pag, b, c):
        return
    if unfVect and ((a, b, c) in unfVect or (c, b, a) in unfVect):
        return
    bk.set_mark(pag, b, c, PAG_ARROW)
    bk.set_mark(pag, c, b, PAG_TAIL)


def _rule_r2(pag, bk, a, b, c):
    """R2: (a -> b *-> c or a *-> b -> c) and a *-o c imply a *-> c.

    Args:
        pag: PAG matrix in Whydra's layout.
        bk: BKOrientationGuard used to write endpoint marks.
        a: start of the chain.
        b: middle of the chain.
        c: node with a circle at its end of edge a—c.
    """
    if pag[a][c] != PAG_CIRCLE:
        return
    if (
        pag[a][b] == PAG_ARROW
        and pag[b][c] == PAG_ARROW
        and (pag[b][a] == PAG_TAIL or pag[c][b] == PAG_TAIL)
    ):
        bk.set_mark(pag, a, c, PAG_ARROW)


def _rules_r1_r2_cycle(pag, bk, unfVect, apply_r1, apply_r2, adjacency=None):
    """Visit wedges incident to an arrow at the center, in stable vertex order.

    A missing chord can trigger R1. A circle on an existing chord can trigger
    R2. Only R1 modifies a spoke, and it never introduces an arrow at the
    current center, so the set of possible incoming spokes is fixed.
    """
    if not (apply_r1 or apply_r2):
        return
    bk = _orientation_guard(None, None) if bk is None else bk
    for center in range(len(pag)):
        inbound = set(np.flatnonzero(pag[:, center] == PAG_ARROW).tolist())
        if not inbound:
            continue
        spokes = (
            np.flatnonzero(pag[center]).tolist()
            if adjacency is None
            else [x for x in adjacency[center] if pag[center, x]]
        )
        for offset, x in enumerate(spokes):
            for y in spokes[offset + 1 :]:
                for left, right in ((x, y), (y, x)):
                    if left not in inbound:
                        continue
                    chord = pag[left, right]
                    if apply_r1 and chord == 0 and pag[right, center] == PAG_CIRCLE:
                        if pag[center, right] != PAG_TAIL and _clear_triple(
                            left, center, right, unfVect
                        ):
                            bk.set_mark(pag, center, right, PAG_ARROW)
                            bk.set_mark(pag, right, center, PAG_TAIL)
                    elif (
                        apply_r2
                        and chord == PAG_CIRCLE
                        and pag[center, right] == PAG_ARROW
                    ) and PAG_TAIL in (pag[center, left], pag[right, center]):
                        bk.set_mark(pag, left, right, PAG_ARROW)


# =============================================================================
#  udag2pag v2 with the orientation bugs fixed.
# =============================================================================


@records_timings
def udag2pag_v2(
    pag,
    sepset,
    p,
    unfVect=None,
    rules=None,
    orientCollider=True,
    background_knowledge=None,
    node_names=None,
):
    """Orient with bilateral separating evidence and a center-ordered R1/R2 pass."""
    enabled = [True] * 10 if rules is None else rules
    ambiguous = set() if unfVect is None else unfVect
    if callable(getattr(sepset, "packed", None)):
        import torch

        from ..graph_core.cuda_udag2pag import apply_udag2pag, apply_udag2pag_tensor

        names = list(range(p)) if node_names is None else node_names
        guard = _orientation_guard(background_knowledge, names)

        def local_rules(host):
            _rules_r1_r2_cycle(host, guard, ambiguous, enabled[0], enabled[1])

        if isinstance(pag, torch.Tensor):
            graph = (pag != 0).to(dtype=torch.int8).mul_(PAG_CIRCLE)
            times = apply_udag2pag_tensor(
                graph,
                sepset,
                ambiguous,
                enabled,
                orientCollider=orientCollider,
                r1r2_host=local_rules,
                whydra_marks=True,
            )
        else:
            graph = np.where(np.asarray(pag) != 0, PAG_CIRCLE, 0)
            times = apply_udag2pag(
                graph,
                sepset,
                ambiguous,
                enabled,
                orientCollider=orientCollider,
                r1r2_host=local_rules,
                whydra_marks=True,
            )
        udag2pag_v2.last_times = {**times, "n_iters": times.get("n_iters", 1.0)}
        return graph
    udag2pag_v2.last_times = {}
    return _orient_skeleton(
        pag,
        sepset,
        ambiguous,
        enabled,
        orientCollider,
        background_knowledge,
        node_names,
        evidence="both",
        wedge_cycle=_rules_r1_r2_cycle,
    )


# =============================================================================
#  Entry point: fci_stable_v2
# =============================================================================


def fci_stable_v2(
    cg,
    data,
    alpha=0.05,
    indep_test="fisherz",
    n_jobs=1,
    verbose=False,
    use_pds_path=False,
    m_max=float("inf"),
    background_knowledge=None,
    show_progress: bool = False,
    max_cond_set_size=None,
    use_ci_cache=True,
    device=None,
    dtype=None,
):
    """Run FCI Stable v2: find the skeleton, run Possible-D-SEP, then apply R0 and R1–R10.

    The Possible-D-SEP step (``pdsep_v2``) follows
    ``removeByPossibleDsep`` in causal-learn 0.1.4.5. It computes PDS per edge
    (Tetrad's ``getPossibleDsep``), tries separating sets of size 2 or larger,
    skips subsets of adj(a) or adj(b), removes edges immediately, and visits
    sides in causal-learn Edge order. ``udag2pag_v2`` handles orientation.

    Parameters
    ----------
    cg : CausalGraph
        All phases use the same CI test. It is taken from ``cg.test`` or
        constructed from ``data`` and ``indep_test``.
    data : np.ndarray
        Data matrix (n_rows x n_features).
    alpha : float
        Significance level for independence tests.
    indep_test : str
        Independence test method (e.g. 'fisherz', 'chisq').
    n_jobs : int
        Number of parallel jobs. Use 1 for sequential execution or -1 for all
        cores. The result is independent of ``n_jobs``.
    verbose : bool
        Print progress information.
    use_pds_path : bool
        Restrict Possible-D-SEP to the biconnected component of the edge under
        test (Colombo et al. 2012, Def. 3.4). This differs from causal-learn,
        so the option is off by default.
    m_max : float
        Maximum separating set size in Possible-D-SEP. Sets start at size 2,
        so m_max < 2 disables CI tests in this step. The default is infinity,
        which preserves the original result by leaving the size unrestricted.
    background_knowledge : BackgroundKnowledge, optional
        Background knowledge for CPU runs. It applies in three phases:
        PRE_SEARCH, before skeleton search, also supplies the required edges
        that the skeleton and pdsep steps must retain;
        BETWEEN_ORIENTATION runs after collider orientation and each rule pass;
        POST_ORIENTATION runs after convergence. CUDA supports only ``None``.
    show_progress : bool
        Show a progress bar during the skeleton stage.

    Returns
    -------
    pag : np.ndarray
        PAG adjacency matrix.
    """
    cuda = device is not None and str(device).startswith("cuda")
    if cuda:
        validate_cuda_background_knowledge(background_knowledge, "fci_stable_cuda")
        n_jobs = 1
    max_cond_set_size = configure_conditioning(cg, max_cond_set_size)

    cit = cg.test or CIT(
        data,
        method=indep_test,
        max_cond_set_size=max_cond_set_size,
        cache_size=100000 if use_ci_cache else 0,
    )
    if cuda:
        from ..graph_core.cuda_contracts import validate_cuda_ci_test

        validate_cuda_ci_test(cit, "FCI-stable-cuda")
    cit.max_cond_set_size = max_cond_set_size
    cg.set_ind_test(cit)

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

    # Step 1: Skeleton discovery (parallelized in SkeletonDiscovery).
    # FCI-stable-cuda: fused skeleton stays resident on the GPU.
    if verbose:
        print("[FCI v2] Step 1: Skeleton discovery...")
    disc = SkeletonDiscovery(
        cg,
        alpha,
        n_jobs=n_jobs,
        verbose=verbose,
        show_progress=show_progress,
        protected_edges=protected_edges,
        device=device,
        dtype=dtype,
    )
    t0 = time.perf_counter()
    if cuda:
        import torch

        graphDict = disc._run_fci_cuda_resident(to_host=False)
        sk = graphDict["sk"]
        if torch.is_tensor(sk):
            from ..graph_core.endpoints import Endpoint

            sk.masked_fill_(sk != 0, Endpoint.TAIL.value)
    else:
        graphDict = disc.run_fci()
    if cuda:
        import torch

        torch.cuda.synchronize()
    t_skel = time.perf_counter() - t0

    _n, p = data.shape

    ci_call = type(cit).__call__

    def threshold_test(x, y, S):
        value = ci_call(cit, x, y, S)
        return _threshold_pvalue(cit, x, y, S, value, alpha)

    test_func_wrapper = LegacyCITest(threshold_test, max_cond_set_size)

    suffStat = None
    ci_batcher = disc._ci_batcher if cuda else None
    if cuda and ci_batcher is None:
        assert device is not None
        method = getattr(cit, "method", indep_test)
        if method == "chisq":
            from ..graph_core.cuda_chisq import CudaChisqHost

            ci_batcher = CudaChisqHost(cit, device=device)
        else:
            from ..graph_core.cpu_cuda_fisher_bridge import CudaFisherZ

            if cit._corr_matrix is not None:
                corr = cit.corr_matrix
                assert corr is not None
                ci_batcher = CudaFisherZ(
                    corr, cit.n_samples, device=device, dtype=dtype
                )
            else:
                ci_batcher = CudaFisherZ.from_data(
                    cit.data, n_samples=cit.n_samples, device=device, dtype=dtype
                )
    if cuda and getattr(cit, "method", None) == "fisherz":
        # Both construction paths above chose the Fisher adapter for this method.
        fisher_batcher = cast("CudaFisherZ", ci_batcher)
        fisher_batcher.cpu_test = cit
        fisher_batcher.decision_alpha = alpha

    # Step 2: Possible-D-SEP (semantics of causal-learn, parallel via joblib)
    if verbose:
        print(f"[FCI v2] Step 2: Possible-D-SEP (n_jobs={n_jobs})...")
    t0 = time.perf_counter()
    pdsep_res = pdsep_v2(
        graphDict,
        suffStat,
        test_func_wrapper,
        p,
        graphDict["sepset"],
        alpha,
        graphDict["pMax"],
        unfVect=graphDict["unfTriples"],
        m_max=m_max,
        n_jobs=n_jobs,
        verbose=verbose,
        use_pds_path=use_pds_path,
        protected_edges=protected_edges,
        parallel_min_tests=_PDS_PARALLEL_MIN_TESTS if _builtin_fisherz(cit) else 0,
        ci_batcher=ci_batcher,
    )
    if "execution" in pdsep_res:
        execution = getattr(cg, "execution_stats", None)
        if execution is None:
            execution = cg.execution_stats = {"passes": []}
        execution["passes"].append(pdsep_res["execution"])
    t_pdsep = time.perf_counter() - t0
    pd_times = dict(getattr(pdsep_v2, "last_times", {}))

    # Step 3: Orient v-structures + R1-R10
    if verbose:
        print("[FCI v2] Step 3: Orientation rules...")
    t0 = time.perf_counter()
    pag = udag2pag_v2(
        pdsep_res["G"],
        pdsep_res["sepset"],
        p,
        unfVect=graphDict["unfTriples"],
        background_knowledge=None if cuda else background_knowledge,
        node_names=node_names,
    )
    t_pag = time.perf_counter() - t0
    pag_times = dict(getattr(udag2pag_v2, "last_times", {}))

    sep_out = pdsep_res["sepset"]
    if cuda:
        import torch

        if isinstance(pag, torch.Tensor):
            pag = pag.detach().cpu().numpy()
        packed = getattr(sep_out, "packed", None)
        if callable(packed):
            S, R = sep_out.packed()
            if torch.is_tensor(S):
                from ..graph_core.cuda_pdsep import SepsetStore

                sep_out = SepsetStore(
                    S.detach().cpu().numpy(), R.detach().cpu().numpy()
                )
    fci_stable_v2.__dict__.update(
        last_sepset=sep_out,
        last_times={
            "skeleton": t_skel,
            "pdsep": t_pdsep,
            "pdsep_colliders": pd_times.get("colliders", 0.0),
            "pdsep_qreach": pd_times.get("qreach", 0.0),
            "pdsep_ci": pd_times.get("ci", 0.0),
            "pdsep_spec_build": pd_times.get("spec_build", 0.0),
            "pdsep_live": pd_times.get("live", 0.0),
            "pdsep_tests": pd_times.get("tests", 0),
            "pdsep_max_ord": pd_times.get("max_ord", 0),
            "pdsep_n_removed": pd_times.get("n_removed", 0),
            "pdsep_n_spec_hits": pd_times.get("n_spec_hits", 0),
            "pdsep_n_recomputed": pd_times.get("n_recomputed", 0),
            "pdsep_sepset_import": pd_times.get("sepset_import", 0.0),
            "pdsep_sepset_export": pd_times.get("sepset_export", 0.0),
            "udag2pag": t_pag,
            "udag2pag_n_iters": pag_times.get("n_iters", 0.0),
            "udag2pag_colliders": pag_times.get("colliders", 0.0),
            **{f"udag2pag_R{i}": pag_times.get(f"R{i}", 0.0) for i in range(1, 11)},
        },
    )

    return pag
