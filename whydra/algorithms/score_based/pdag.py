"""PDAG/CPDAG primitives for the equivalence-class search in :mod:`ges`.

Representation
--------------
All functions here work on a square ``int8`` adjacency matrix ``C`` whose
semantics are *adjacency*, not endpoints (the PAG/endpoint encoding used
elsewhere in the package is deliberately kept out of this module):

* ``C[x, y] == 1`` and ``C[y, x] == 0``  ->  ``x -> y``
* ``C[x, y] == 1`` and ``C[y, x] == 1``  ->  ``x -- y`` (undirected)
* both zero                              ->  no edge

Conversion to the package-wide endpoint encoding happens only at the module
boundary, in :func:`cpdag_matrix_to_general_graph`.

The GES orientation step applies Meek's rules R1–R4 to this adjacency matrix.
"""

from collections import deque

import numpy as np

from ..graph_core import Edge, Endpoint, GeneralGraph, Node

# ---------------------------------------------------------------------------
#  Basic relations
# ---------------------------------------------------------------------------


def is_directed(C, a, b) -> bool:
    """Return whether the graph contains the directed edge ``a -> b``."""
    return bool(C[a, b]) and not bool(C[b, a])


def is_undirected(C, a, b) -> bool:
    """Return whether the graph contains the undirected edge ``a -- b``."""
    return bool(C[a, b]) and bool(C[b, a])


def is_adjacent(C, a, b) -> bool:
    """Return whether either direction connects the two nodes."""
    return bool(C[a, b]) or bool(C[b, a])


def neighbours(C, x) -> set:
    """Nodes joined to ``x`` by an undirected edge."""
    row = C[x].astype(bool)
    col = C[:, x].astype(bool)
    return {int(y) for y in np.flatnonzero(row & col)}


def parents(C, x) -> set:
    """Nodes with a directed edge into ``x``."""
    row = C[x].astype(bool)
    col = C[:, x].astype(bool)
    return {int(y) for y in np.flatnonzero(col & ~row)}


def adjacent(C, x) -> set:
    """Return all nodes adjacent to x, regardless of edge direction."""
    row = C[x].astype(bool)
    col = C[:, x].astype(bool)
    return {int(y) for y in np.flatnonzero(row | col)}


def na_yx(C, y, x) -> set:
    """``NA_yx`` — neighbours of ``y`` that are adjacent to ``x``."""
    return {t for t in neighbours(C, y) if is_adjacent(C, t, x)}


def is_clique(C, nodes) -> bool:
    """Whether every pair in ``nodes`` is adjacent (orientation irrelevant)."""
    nodes = list(nodes)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            if not is_adjacent(C, a, b):
                return False
    return True


def exists_semidirected_path(C, source, target, block=()) -> bool:
    """Whether a semi-directed path ``source ~> target`` avoids ``block``.

    A semi-directed path may traverse ``u -> v`` and ``u -- v`` but never
    runs against an arrowhead. Nodes in ``block`` are not visited, which is
    exactly the "every semi-directed path contains a node of NA_yx u T"
    condition of the Insert operator.
    """
    block = set(block)
    if source in block:
        return False
    seen = {source}
    queue = deque([source])
    while queue:
        current = queue.popleft()
        if current == target:
            return True
        for nxt in np.flatnonzero(C[current]):
            nxt = int(nxt)
            if nxt in seen or nxt in block:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return False


# ---------------------------------------------------------------------------
#  Meek rules R1-R4
# ---------------------------------------------------------------------------


def _meek_rules_reference(C, inplace: bool = False):
    """Orient a PDAG as far as Meek's four rules allow.

    R4 matters here even though DAG -> CPDAG conversion gets by with R1-R3:
    completing an arbitrary PDAG (which is what Insert/Delete leave behind)
    without R4 can stop short of the maximally oriented graph, and the next
    search step would then read the wrong ``NA_yx``.
    """
    # GES needs R4 to complete partially directed graphs after Insert/Delete.
    # The PC rules operate on CausalGraph with background-knowledge and
    # conflict checks. The evaluation converter accepts a DAG and only needs
    # R1–R3 to construct its ground-truth CPDAG.
    if not inplace:
        C = C.copy()
    p = C.shape[0]

    def orient(a, b):
        """Turn ``a -- b`` into ``a -> b``; report whether anything changed."""
        if is_undirected(C, a, b):
            C[b, a] = 0
            return True
        return False

    changed = True
    while changed:
        changed = False

        # R1: a -> b, b -- c, a and c non-adjacent  =>  b -> c
        for a in range(p):
            for b in range(p):
                if not is_directed(C, a, b):
                    continue
                for c in range(p):
                    if c in (a, b) or not is_undirected(C, b, c):
                        continue
                    if not is_adjacent(C, a, c):
                        changed |= orient(b, c)

        # R2: a -> b -> c and a -- c  =>  a -> c
        for a in range(p):
            for b in range(p):
                if not is_directed(C, a, b):
                    continue
                for c in range(p):
                    if c in (a, b) or not is_directed(C, b, c):
                        continue
                    changed |= orient(a, c)

        # R3: a -- b, a -- c, a -- d, c -> b, d -> b, c and d non-adjacent
        #     =>  a -> b
        for a in range(p):
            for b in range(p):
                if not is_undirected(C, a, b):
                    continue
                cands = [
                    c
                    for c in range(p)
                    if c not in (a, b)
                    and is_undirected(C, a, c)
                    and is_directed(C, c, b)
                ]
                found = False
                for i, c in enumerate(cands):
                    for d in cands[i + 1 :]:
                        if not is_adjacent(C, c, d):
                            found = True
                            break
                    if found:
                        break
                if found:
                    changed |= orient(a, b)

        # R4: a -- b, a -- c, c -> d, d -> b, b and c non-adjacent,
        #     a adjacent to d  =>  a -> b
        for a in range(p):
            for b in range(p):
                if not is_undirected(C, a, b):
                    continue
                hit = False
                for c in range(p):
                    if c in (a, b) or not is_undirected(C, a, c):
                        continue
                    if is_adjacent(C, b, c):
                        continue
                    for d in range(p):
                        if d in (a, b, c):
                            continue
                        if (
                            is_directed(C, c, d)
                            and is_directed(C, d, b)
                            and is_adjacent(C, a, d)
                        ):
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    changed |= orient(a, b)

    return C


# ---------------------------------------------------------------------------
#  Extension and completion
# ---------------------------------------------------------------------------


def _pdag_to_dag_reference(C):
    """Extend a PDAG to a consistent DAG (Dor & Tarsi); ``None`` if impossible.

    Repeatedly removes a node ``x`` that has no outgoing directed edge and
    whose undirected neighbours are adjacent to all of ``x``'s other
    neighbours, orienting every ``y -- x`` as ``y -> x`` on the way out.
    """
    p = C.shape[0]
    work = C.copy()
    dag = np.zeros((p, p), dtype=np.int8)
    # Directed edges carry over unchanged; undirected ones get oriented below.
    for a in range(p):
        for b in range(p):
            if is_directed(work, a, b):
                dag[a, b] = 1

    remaining = set(range(p))
    while remaining:
        for x in sorted(remaining):
            has_outgoing = any(is_directed(work, x, y) for y in remaining if y != x)
            if has_outgoing:
                continue
            ne_x = {y for y in neighbours(work, x) if y in remaining}
            adj_x = {y for y in adjacent(work, x) if y in remaining}
            if all(is_adjacent(work, y, z) for y in ne_x for z in adj_x if y != z):
                for y in ne_x:
                    dag[y, x] = 1
                for y in list(remaining):
                    work[x, y] = 0
                    work[y, x] = 0
                remaining.discard(x)
                break
        else:
            return None
    return dag


def meek_rules(C, inplace: bool = False):
    """Apply the ordered R1–R4 closure using sparse graph relations."""
    if np.any(np.diag(C)):
        return _meek_rules_reference(C, inplace=inplace)
    from ._pdag_masks import meek

    return meek(C, inplace=inplace)


def pdag_to_dag(C):
    """Find the same smallest-sink consistent extension using adjacency masks."""
    if np.any(np.diag(C)):
        return _pdag_to_dag_reference(C)
    from ._pdag_masks import extension

    return extension(C)


def dag_to_cpdag(dag_adj):
    """CPDAG of a binary DAG matrix (``dag[i, j] == 1`` means ``i -> j``)."""
    dag = np.asarray(dag_adj)
    p = dag.shape[0]
    skeleton = (dag != 0) | (dag.T != 0)
    C = np.zeros((p, p), dtype=np.int8)
    C[skeleton] = 1
    np.fill_diagonal(C, 0)

    # Unshielded colliders are invariant across the equivalence class.
    for y in range(p):
        pa = [x for x in range(p) if dag[x, y]]
        for i, x in enumerate(pa):
            for z in pa[i + 1 :]:
                if not skeleton[x, z]:
                    C[y, x] = 0
                    C[y, z] = 0

    return meek_rules(C, inplace=True)


def complete_pdag(C, *, affected_nodes=None):
    """Re-derive the CPDAG of the equivalence class a PDAG represents.

    Insert/Delete change which v-structures exist, so the CPDAG cannot be
    patched up by running Meek on the modified graph — it has to go through a
    consistent DAG extension.
    """
    if affected_nodes is not None:
        # GES supplies the changed endpoints of a move on a known CPDAG.
        # Include the WHOLE connected region, including both parts after a
        # deletion. Other components are still CPDAGs and cannot be affected.
        from ._pdag_masks import affected_component

        nodes = affected_component(C, affected_nodes)
        if len(nodes) < len(C):
            selection = np.ix_(nodes, nodes)
            completed = complete_pdag(C[selection])
            if completed is None:
                return None
            result = C.copy()
            result[selection] = completed
            return result
    dag = pdag_to_dag(C)
    if dag is None:
        return None
    return dag_to_cpdag(dag)


# ---------------------------------------------------------------------------
#  Conversion to the package endpoint encoding
# ---------------------------------------------------------------------------


def cpdag_matrix_to_general_graph(C, node_names) -> GeneralGraph:
    """Convert an adjacency-encoded CPDAG into a :class:`GeneralGraph`."""
    p = C.shape[0]
    graph = GeneralGraph([Node(name) for name in node_names])
    for i in range(p):
        for j in range(i + 1, p):
            if not is_adjacent(C, i, j):
                continue
            if is_undirected(C, i, j):
                end_i, end_j = Endpoint.TAIL, Endpoint.TAIL
            elif is_directed(C, i, j):
                end_i, end_j = Endpoint.TAIL, Endpoint.ARROW
            else:
                end_i, end_j = Endpoint.ARROW, Endpoint.TAIL
            graph.add_edge(Edge(graph.nodes[i], graph.nodes[j], end_i, end_j))
    return graph
