"""Sparse, ordered implementations of the existing PDAG operations.

The ascending visitation order and immediate updates match the scalar oracle.
Only impossible triples are omitted; all four Meek rules still run to a fixed
point. Masks represent adjacency, not the package's PAG endpoint encoding.
"""

import numpy as np

from ._ges_scan import _nodes, _row_masks


class Relations:
    def __init__(self, matrix):
        present = np.asarray(matrix, dtype=bool)
        self.adjacent = _row_masks(present | present.T)
        self.undirected = list(_row_masks(present & present.T))
        self.outgoing = list(_row_masks(present & ~present.T))
        self.parents = list(_row_masks(present.T & ~present))


def meek(matrix, inplace=False):
    result = matrix if inplace else matrix.copy()
    graph = Relations(result)
    adjacent, undirected = graph.adjacent, graph.undirected
    outgoing, parents = graph.outgoing, graph.parents
    size = len(result)

    def orient(a, b):
        if not (undirected[a] & (1 << b)):
            return False
        result[b, a] = 0
        undirected[a] &= ~(1 << b)
        undirected[b] &= ~(1 << a)
        outgoing[a] |= 1 << b
        parents[b] |= 1 << a
        return True

    changed = True
    while changed:
        changed = False
        # R1 cannot create another outgoing edge from its outer vertex a.
        for a in range(size):
            for b in _nodes(outgoing[a]):
                for c in _nodes(undirected[b] & ~adjacent[a] & ~(1 << a)):
                    changed |= orient(b, c)
        # R2 CAN create an outgoing edge from a. Refresh the unvisited suffix
        # after every b, as the scalar range loop would observe those updates.
        for a in range(size):
            remaining = outgoing[a]
            while remaining:
                bit = remaining & -remaining
                b = bit.bit_length() - 1
                for c in _nodes(outgoing[b] & undirected[a] & ~(1 << a)):
                    changed |= orient(a, c)
                remaining = outgoing[a] & ~((bit << 1) - 1)
        for a in range(size):
            for b in _nodes(undirected[a]):
                candidates = undirected[a] & parents[b] & ~(1 << b)
                remaining = candidates
                while remaining:
                    bit = remaining & -remaining
                    remaining ^= bit
                    if remaining & ~adjacent[bit.bit_length() - 1]:
                        changed |= orient(a, b)
                        break
        for a in range(size):
            for b in _nodes(undirected[a]):
                for c in _nodes(undirected[a] & ~adjacent[b] & ~(1 << b)):
                    if (
                        outgoing[c]
                        & parents[b]
                        & adjacent[a]
                        & ~((1 << a) | (1 << b) | (1 << c))
                    ):
                        changed |= orient(a, b)
                        break
    return result


def extension(matrix):
    graph = Relations(matrix)
    present = np.asarray(matrix, dtype=bool)
    dag = (present & ~present.T).astype(np.int8)
    remaining = (1 << len(matrix)) - 1
    while remaining:
        for x in _nodes(remaining):
            if graph.outgoing[x] & remaining:
                continue
            neighbours = graph.undirected[x] & remaining
            adjacent = graph.adjacent[x] & remaining
            if any(
                (adjacent & ~(1 << y)) & ~graph.adjacent[y] for y in _nodes(neighbours)
            ):
                continue
            for y in _nodes(neighbours):
                dag[y, x] = 1
            remaining &= ~(1 << x)
            break
        else:
            return None
    return dag


def affected_component(matrix, nodes):
    """Union of complete post-move components containing the changed ends."""
    adjacent = Relations(matrix).adjacent
    reached = sum(1 << int(node) for node in set(nodes))
    frontier = reached
    while frontier:
        following = 0
        for node in _nodes(frontier):
            following |= adjacent[node]
        frontier = following & ~reached
        reached |= frontier
    return tuple(_nodes(reached))
