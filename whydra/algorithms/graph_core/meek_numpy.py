"""Meek R1–R3 on a dense endpoint matrix (no Node/Edge objects).

``G[i, j]`` uses :class:`Endpoint` values: ``0`` none, ``-1`` tail, ``+1`` arrow.
``i → j`` is ``G[i, j] == ARROW`` and ``G[j, i] == TAIL``. Undirected is both
tails. The adjacency pattern (nonzero) is frozen; only endpoints change.

Pattern order is the same as ``CausalGraph.find_unshielded_triples`` /
``find_triangles`` / ``find_kites``. Orientations are applied immediately,
same as ``_orient_edge_conflict_aware``, without allocating Edge objects or
kite lists.
"""

from itertools import combinations, permutations

import numpy as np

from .endpoints import Endpoint

_TAIL = int(Endpoint.TAIL.value)
_ARROW = int(Endpoint.ARROW.value)


def meek_orient(G: np.ndarray) -> dict:
    """In-place Meek R1–R3 closure. Returns ``meek_iters`` / ``meek_orients``."""
    G = np.asarray(G)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        msg = "G must be square"
        raise ValueError(msg)
    n_iters = 0
    n_orients = 0
    p = int(G.shape[0])
    limit = max(p * p, 1)
    while n_iters < limit:
        n_iters += 1
        n = _round_r1(G) + _round_r2(G) + _round_r3(G)
        n_orients += n
        if n == 0:
            break
    return {"meek_iters": n_iters, "meek_orients": n_orients}


def _dir(G, i, j) -> bool:
    return G[i, j] == _ARROW and G[j, i] == _TAIL


def _und(G, i, j) -> bool:
    return G[i, j] == _TAIL and G[j, i] == _TAIL


def _bi(G, i, j) -> bool:
    return G[i, j] == _ARROW and G[j, i] == _ARROW


def _reaches_directed(G, src, tgt) -> bool:
    """Whether a directed ``src -> ... -> tgt`` path exists."""
    p = int(G.shape[0])
    stack = [src]
    seen = np.zeros(p, dtype=bool)
    seen[src] = True
    while stack:
        i = stack.pop()
        if i == tgt:
            return True
        for j in np.flatnonzero(G[i, :] != 0):
            if not seen[j] and _dir(G, i, j):
                seen[j] = True
                stack.append(int(j))
    return False


def _orient(G, src, tgt) -> int:
    if _dir(G, src, tgt) or _bi(G, src, tgt):
        return 0
    if _dir(G, tgt, src):
        G[src, tgt] = _ARROW
        G[tgt, src] = _ARROW
        return 1
    if _und(G, src, tgt):
        # Same guard as the sequential Meek loop: never orient src -> tgt
        # when tgt is already an ancestor of src.
        if _reaches_directed(G, tgt, src):
            return 0
        G[src, tgt] = _ARROW
        G[tgt, src] = _TAIL
        return 1
    return 0


def _round_r1(G) -> int:
    p = int(G.shape[0])
    n = 0
    for j in range(p):
        neighbors = np.where(G[j, :] != 0)[0]
        if len(neighbors) < 2:
            continue
        for i, k in combinations(neighbors, 2):
            if G[i, k] != 0:
                continue
            if _dir(G, i, j) and _und(G, j, k):
                n += _orient(G, j, k)
            elif _dir(G, k, j) and _und(G, j, i):
                n += _orient(G, j, i)
    return n


def _round_r2(G) -> int:
    p = int(G.shape[0])
    n = 0
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if G[i, j] != 0:
                edges.append((i, j))
    for i, j in edges:
        neigh_i = set(np.where(G[i, :] != 0)[0])
        neigh_j = set(np.where(G[j, :] != 0)[0])
        common = neigh_i.intersection(neigh_j)
        for k in common:
            if k <= j:
                continue
            for a, b, c in permutations((i, j, int(k)), 3):
                if _dir(G, a, b) and _dir(G, b, c) and _und(G, a, c):
                    n += _orient(G, a, c)
    return n


def _round_r3(G) -> int:
    p = int(G.shape[0])
    n = 0
    for v in range(p):
        neighbors_v = np.where(G[v, :] != 0)[0]
        if len(neighbors_v) < 2:
            continue
        neighbors_v_set = set(neighbors_v)
        for j, k in combinations(neighbors_v, 2):
            if G[j, k] != 0:
                continue
            neigh_j = set(np.where(G[j, :] != 0)[0])
            neigh_k = set(np.where(G[k, :] != 0)[0])
            common = neigh_j.intersection(neigh_k).intersection(neighbors_v_set)
            for i in common:
                if (
                    _dir(G, j, v)
                    and _dir(G, k, v)
                    and _und(G, i, v)
                    and _und(G, i, j)
                    and _und(G, i, k)
                ):
                    n += _orient(G, i, v)
    return n


__all__ = ["meek_orient"]
