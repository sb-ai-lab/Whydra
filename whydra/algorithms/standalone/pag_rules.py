"""Orientation of partial ancestral graphs.

The edge implications follow J. Zhang (2008), Artificial Intelligence 172,
1873–1896, doi:10.1016/j.artint.2008.08.001. Matrix entries are marks at the
column endpoint. Rules infer marks through the caller's background-knowledge
guard, on a fixed skeleton between checkpoints.
"""

from collections import deque
from itertools import chain, combinations, pairwise, product

import numpy as np

from ...background_knowledge import (
    BKOrientationGuard,
    BKPhase,
    MatrixEncoding,
    apply_bk_checkpoint,
)
from ..graph_core.endpoints import Endpoint
from .orientation_queue import OrientationQueue

PAG_NULL = Endpoint.NULL.value
PAG_TAIL = Endpoint.TAIL.value
PAG_ARROW = Endpoint.ARROW.value
PAG_CIRCLE = Endpoint.CIRCLE.value


def updateList(path, set_nodes, old_list):
    """Append independent path extensions, preserving the public list contract."""
    extensions = old_list or []
    extensions.extend([*path, node] for node in set_nodes)
    return extensions


def powerset(iterable):
    """Yield every subset of the input, in increasing order of size."""
    xs = list(iterable)
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))


def legal_path(a, b, c, amat):
    """Check whether a path may extend from a through b to c under the endpoint marks."""
    if a == c or not amat[b][c]:
        return False
    if amat[a][c]:
        return bool(amat[a][b])
    return bool(amat[a][b] == PAG_ARROW and amat[c][b] == PAG_ARROW)


def qreach(x, amat):
    """Return nodes reachable from x along legal triples of adjacent nodes."""
    graph = np.asarray(amat)
    origin = 1 << int(x)
    n = len(graph)
    degree = np.count_nonzero(graph[x]) - int(bool(graph[x, x]))
    if degree == 0:
        return []
    if degree == n - 1:
        return [v for v in range(n) if v != x]

    def masks(rows):
        packed = np.packbits(rows, axis=1, bitorder="little")
        data = packed.tobytes()
        width = packed.shape[1]
        return [
            int.from_bytes(data[i : i + width], "little")
            for i in range(0, len(data), width)
        ]

    completed = ((1 << n) - 1) ^ origin
    links = masks(graph != 0)
    heads = masks(graph.T == PAG_ARROW)
    available = links.copy()
    visited = links[x] & ~origin
    frontier = [(int(x), visited)]
    while frontier:
        previous, centers = frontier.pop()
        previous_bit = 1 << previous
        allowed_targets = ~(previous_bit | origin)
        triangle_ends = links[previous]
        while centers:
            center_bit = centers & -centers
            centers ^= center_bit
            center = center_bit.bit_length() - 1
            open_edges = available[center] & allowed_targets
            if not open_edges:
                continue
            valid = triangle_ends
            if heads[center] & previous_bit:
                valid |= heads[center]
            next_centers = open_edges & valid
            if next_centers:
                available[center] ^= next_centers
                visited |= next_centers
                if visited == completed:
                    return [v for v in range(n) if v != x]
                frontier.append((center, next_centers))
    bits = visited.to_bytes((n + 7) // 8, "little")
    flags = np.unpackbits(np.frombuffer(bits, dtype=np.uint8), bitorder="little")
    return np.flatnonzero(flags).tolist()


def _clear_triple(left, middle, right, ambiguous):
    return not ambiguous or (
        (left, middle, right) not in ambiguous
        and (right, middle, left) not in ambiguous
    )


def faith_check(cp, unfVect, p, boolean=True):
    """Check ambiguity at every vertex of the closed walk, in either direction."""
    if not boolean:
        return 0 if boolean == 0 else None
    if not len(cp) or not unfVect:
        return True
    vertices = iter(cp)
    first = middle = next(vertices)
    previous = cp[-1]
    for following in chain(vertices, (first,)):
        if not _clear_triple(previous, middle, following, unfVect):
            return False
        previous, middle = middle, following
    return True


def _discriminating(pag, a, b, c, reciprocal=True):
    """Find an R4 witness by tracing incoming arrowheads through parents of c.

    Paths of equal length use vertex order. A blocked arrival closes admission
    at its queue position, but another arrival already waiting may expand.
    """
    closed = {a, b, c}
    successor: dict[int, int] = {}
    waiting = deque([(a, True)])
    discovered = {a}
    while waiting:
        node, can_expand = waiting.popleft()
        closed.add(node)
        if not can_expand:
            continue
        arrows = pag[:, node] == PAG_ARROW
        if reciprocal:
            arrows &= pag[node] != PAG_NULL
        for predecessor in arrows.nonzero()[0].tolist():
            if predecessor in closed:
                continue
            if _gap(pag, predecessor, c):
                path = [predecessor, node]
                while path[-1] != a:
                    path.append(successor[path[-1]])
                return [*path, b, c]
            through_collider = (
                pag[node, predecessor] == PAG_ARROW
                and pag[predecessor, c] == PAG_ARROW
                and pag[c, predecessor] == PAG_TAIL
            )
            if through_collider and predecessor not in successor:
                successor[predecessor] = node
                waiting.append((predecessor, True))
            elif predecessor not in discovered:
                waiting.append((predecessor, False))
            discovered.add(predecessor)
    return []


def minDiscPath(pag, a, b, c):
    """Return a discriminating path ending in (a, b, c), or an empty list if none exists."""
    return _discriminating(pag, a, b, c)


def _gap(matrix, x, y):
    return not (matrix[x, y] or matrix[y, x])


def _can_advance(matrix, x, y):
    return matrix[x, y] in (PAG_CIRCLE, PAG_ARROW) and matrix[y, x] in (
        PAG_TAIL,
        PAG_CIRCLE,
    )


def minUncovCircPath(p, pag, path, unfVect):
    """Complete the prescribed ends of an uncovered circle path for R5."""
    start, first, last, end = path
    if len({start, first, last, end}) < 4:
        return []
    if (pag[start, first], pag[first, start]) != (PAG_CIRCLE, PAG_CIRCLE):
        return []
    if (pag[last, end], pag[end, last]) != (PAG_CIRCLE, PAG_CIRCLE):
        return []
    if not _clear_triple(last, end, start, unfVect) or not _clear_triple(
        end, start, first, unfVect
    ):
        return []

    neighbors = {}
    paths: deque[tuple[int, ...]] = deque([(start, first)])
    while paths:
        prefix = paths.popleft()
        previous, node = prefix[-2:]
        if node not in neighbors:
            circles = (pag[node] == PAG_CIRCLE) & (pag[:, node] == PAG_CIRCLE)
            neighbors[node] = circles.nonzero()[0].tolist()
        for nxt in neighbors[node]:
            if nxt == end or nxt in prefix or not _gap(pag, previous, nxt):
                continue
            if not _clear_triple(previous, node, nxt, unfVect):
                continue
            if nxt == last:
                if _gap(pag, node, end) and _clear_triple(node, last, end, unfVect):
                    return [*prefix, last, end]
            else:
                paths.append((*prefix, nxt))
    return []


def minUncovPdPath(p, pag, a, b, c, unfVect):
    """Complete a potential directed path with fixed first hop and an interior vertex.

    Every arrival limits later admissions. An inadmissible prefix still takes
    its turn, but it can never supply a witness or need a second expansion.
    """
    if not _clear_triple(c, a, b, unfVect):
        return []

    children = {}

    def continuations(node):
        if node not in children:
            front, back = pag[node], pag[:, node]
            outgoing = ((front == PAG_ARROW) | (front == PAG_CIRCLE)) & (
                (back == PAG_TAIL) | (back == PAG_CIRCLE)
            )
            children[node] = outgoing.nonzero()[0].tolist()
        return children[node]

    closed = {a, b, c}
    rejected = set()
    pending: deque[tuple[int, tuple[int, ...] | None]] = deque([(b, (a, b))])
    while pending:
        node, prefix = pending.popleft()
        closed.add(node)
        if node != b and _can_advance(pag, node, c):
            if (
                prefix is not None
                and _gap(pag, prefix[-2], c)
                and _clear_triple(prefix[-2], node, c, unfVect)
                and _clear_triple(node, c, a, unfVect)
            ):
                return [*prefix, c]
            continue
        for nxt in continuations(node):
            if nxt in closed or (node == b and pag[nxt, a] != 0):
                continue
            if (
                prefix is not None
                and _gap(pag, prefix[-2], nxt)
                and _clear_triple(prefix[-2], node, nxt, unfVect)
            ):
                pending.append((nxt, (*prefix, nxt)))
            elif nxt not in rejected:
                pending.append((nxt, None))
                rejected.add(nxt)
    return []


def _ordered_mark_pairs(mask):
    targets, sources = np.nonzero(mask.T)
    return zip(sources.tolist(), targets.tolist(), strict=False)


class _Orientations:
    """Evaluate each local or path implication on the live endpoint matrix."""

    def __init__(self, pag, guard, ambiguous=(), work=None):
        self.pag = pag
        self.guard = guard
        self.ambiguous = ambiguous
        self.work = work
        self.neighbors = [
            np.flatnonzero(row).tolist() for row in ((pag != 0) | (pag.T != 0))
        ]

    def edit(self, x, y, mark):
        self.guard.set_mark(self.pag, x, y, mark)

    def allowed(self, a, b, c):
        return _clear_triple(a, b, c, self.ambiguous)

    def run(self, number, sepset=None):
        g = self.pag
        if number == 1:
            mask = (g == PAG_ARROW) & (g.T != 0)
        elif number == 2:
            mask = (g == PAG_CIRCLE) & (g.T != 0)
        elif number == 5:
            mask = (g == PAG_CIRCLE) & (g.T == PAG_CIRCLE)
        elif number >= 8:
            mask = (g == PAG_ARROW) & (g.T == PAG_CIRCLE)
        else:
            mask = (g != 0) & (g.T == PAG_CIRCLE)
        pairs = (
            _ordered_mark_pairs(mask)
            if self.work is None
            else self.work.candidates(
                f"R{number}", g, mask, paths=number in (4, 5, 9, 10)
            )
        )
        apply = getattr(self, f"r{number}")
        for x, y in pairs:
            if number == 4:
                apply(x, y, sepset)
            else:
                apply(x, y)

    def r1(self, a, b):
        g = self.pag
        for c in self.neighbors[b]:
            if (
                c != a
                and g[b, c]
                and g[c, b] == PAG_CIRCLE
                and _gap(g, a, c)
                and self.allowed(a, b, c)
            ):
                self.edit(b, c, PAG_ARROW)
                self.edit(c, b, PAG_TAIL)

    def r2(self, a, c):
        g = self.pag
        for b in self.neighbors[a]:
            if (
                g[a, b] == g[b, c] == PAG_ARROW
                and g[b, a]
                and g[c, b]
                and PAG_TAIL in (g[b, a], g[c, b])
            ):
                self.edit(a, c, PAG_ARROW)
                return

    def r3(self, b, d):
        g = self.pag
        witnesses = [
            x
            for x in self.neighbors[b]
            if g[x, b] == PAG_ARROW and g[x, d] == PAG_CIRCLE and g[b, x] and g[d, x]
        ]
        if any(
            _gap(g, x, y) and self.allowed(x, b, y)
            for x, y in combinations(witnesses, 2)
        ):
            self.edit(d, b, PAG_ARROW)

    def r4(self, b, c, sepset):
        g = self.pag
        for a in self.neighbors[b]:
            if g[c, b] != PAG_CIRCLE:
                return
            if not (
                g[a, b]
                and g[b, c]
                and g[b, a] == g[a, c] == PAG_ARROW
                and g[c, a] == PAG_TAIL
            ):
                continue
            witness = minDiscPath(g, a, b, c)
            if witness:
                x = witness[0]
                noncollider = b in sepset.get((x, c), ()) or b in sepset.get((c, x), ())
                if not noncollider:
                    self.edit(a, b, PAG_ARROW)
                self.edit(b, c, PAG_ARROW)
                self.edit(c, b, PAG_TAIL if noncollider else PAG_ARROW)

    def r5(self, a, b):
        g = self.pag
        if g[a, b] != PAG_CIRCLE:
            return

        def outside(x, y):
            return [
                v
                for v in self.neighbors[x]
                if v != y and _gap(g, v, y) and g[x, v] == g[v, x] == PAG_CIRCLE
            ]

        for left, right in product(outside(a, b), outside(b, a)):
            if g[a, b] != PAG_CIRCLE:
                return
            four = (a, left, right, b)
            if g[left, right] == g[right, left] == PAG_CIRCLE:
                if not faith_check(four, self.ambiguous, len(g)):
                    continue
                witness = four
            else:
                witness = minUncovCircPath(len(g), g, four, self.ambiguous)
            if witness:
                for x, y in chain(((a, b),), pairwise(witness)):
                    self.edit(x, y, PAG_TAIL)
                    self.edit(y, x, PAG_TAIL)

    def r6(self, b, c):
        g = self.pag
        if any(g[b, a] == g[a, b] == PAG_TAIL for a in self.neighbors[b]):
            self.edit(c, b, PAG_TAIL)

    def r7(self, b, c):
        g = self.pag
        if any(
            a != c
            and g[b, a] == PAG_TAIL
            and g[a, b] == PAG_CIRCLE
            and _gap(g, a, c)
            and self.allowed(a, b, c)
            for a in self.neighbors[b]
        ):
            self.edit(c, b, PAG_TAIL)

    def r8(self, a, c):
        g = self.pag
        if any(
            g[b, a] == g[c, b] == PAG_TAIL
            and g[b, c] == PAG_ARROW
            and g[a, b] in (PAG_CIRCLE, PAG_ARROW)
            for b in self.neighbors[a]
        ):
            self.edit(c, a, PAG_TAIL)

    def r9(self, a, c):
        g = self.pag
        if g[c, a] == PAG_CIRCLE and any(
            b != c
            and _gap(g, b, c)
            and _can_advance(g, a, b)
            and minUncovPdPath(len(g), g, a, b, c, self.ambiguous)
            for b in self.neighbors[a]
        ):
            self.edit(c, a, PAG_TAIL)

    def r10(self, a, c):
        g = self.pag
        if g[c, a] != PAG_CIRCLE:
            return
        parents = [
            b for b in self.neighbors[c] if (g[b, c], g[c, b]) == (PAG_ARROW, PAG_TAIL)
        ]
        if len(parents) < 2:
            return
        first = [v for v in self.neighbors[a] if v != c and _can_advance(g, a, v)]
        # Store first hops by the parent they can reach. This turns the
        # two-witness condition into a join of previously computed path sets.
        reachable = {}
        for b, d in combinations(parents, 2):
            if b in first and d in first and _gap(g, b, d):
                if self.allowed(b, a, d):
                    self.edit(c, a, PAG_TAIL)
                    return
                continue
            for parent in (b, d):
                if parent not in reachable:
                    reachable[parent] = [
                        v
                        for v in first
                        if minUncovPdPath(len(g), g, a, v, parent, self.ambiguous)
                    ]
            if any(
                x != y and not (g[x, y] and g[y, x]) and self.allowed(x, a, y)
                for x, y in product(reachable[b], reachable[d])
            ):
                self.edit(c, a, PAG_TAIL)
                return


def _orientation_guard(knowledge, names):
    if knowledge is None:
        return BKOrientationGuard(None, MatrixEncoding.STANDARD)
    return knowledge.orientation_guard(
        names, phase=BKPhase.BETWEEN_ORIENTATION, encoding=MatrixEncoding.STANDARD
    )


def _orient_skeleton(
    pag,
    sepset,
    unf,
    enabled,
    orient_colliders,
    knowledge,
    names,
    *,
    stats=None,
    evidence="recorded",
    colliders=None,
    wedge_cycle=None,
):
    graph = np.where(np.asarray(pag) != 0, PAG_CIRCLE, 0)
    names = list(range(len(graph))) if names is None else names
    guard = _orientation_guard(knowledge, names)

    def checkpoint(label, previous=None, phase=BKPhase.BETWEEN_ORIENTATION):
        apply_bk_checkpoint(
            knowledge,
            graph,
            names,
            phase=phase,
            checkpoint=label,
            encoding=MatrixEncoding.STANDARD,
            graph_kind="pag",
            previous=previous,
        )

    def record(rule, before):
        if stats is not None:
            stats[f"R{rule}"] += int(np.count_nonzero(graph != before))

    checkpoint("before_orientation")
    initial = graph.copy() if orient_colliders else None
    if orient_colliders:
        for center in range(len(graph)):
            spokes = np.flatnonzero(
                (graph[:, center] != 0) | (graph[center] != 0)
            ).tolist()
            circle = {v for v in spokes if graph[v, center] == PAG_CIRCLE}
            for x in spokes:
                if x not in circle:
                    continue
                for z in spokes:
                    if (
                        x == z
                        or not graph[center, z]
                        or graph[x, z]
                        or not _clear_triple(x, center, z, unf)
                    ):
                        continue
                    if colliders is not None:
                        accept = (x, center, z) in colliders or (
                            z,
                            center,
                            x,
                        ) in colliders
                    else:
                        left, right = sepset.get((x, z)), sepset.get((z, x))
                        if evidence == "both":
                            accept = (
                                left is not None
                                and right is not None
                                and center not in left
                                and center not in right
                            )
                        elif evidence == "recorded":
                            sep = right if left is None else left
                            accept = sep is not None and center not in sep
                        else:
                            accept = (left is None or center not in left) and (
                                right is None or center not in right
                            )
                    if accept:
                        guard.set_mark(graph, x, center, PAG_ARROW)
                        guard.set_mark(graph, z, center, PAG_ARROW)
        record(0, initial)
    checkpoint("after_colliders", initial)
    queue = (
        None
        if wedge_cycle is not None
        else OrientationQueue(graph, enabled=type(guard) is BKOrientationGuard)
    )
    writer = guard if queue is None else queue.guard(guard)
    while True:
        previous = graph.copy()
        engine = _Orientations(graph, writer, unf, queue)
        first_rule = 1
        if wedge_cycle is not None:
            wedge_cycle(graph, guard, unf, enabled[0], enabled[1], engine.neighbors)
            first_rule = 3
        for rule in range(first_rule, 11):
            if enabled[rule - 1]:
                before = graph.copy() if stats is not None else None
                engine.run(rule, sepset)
                record(rule, before)
        checkpoint("after_orientation_rule_pass", previous)
        if np.array_equal(previous, graph):
            break
    checkpoint("after_orientation", phase=BKPhase.POST_ORIENTATION)
    return graph


def udag2pag(
    pag,
    sepset,
    p,
    unfVect=None,
    rules=None,
    orientCollider=True,
    rule_stats=None,
    background_knowledge=None,
    node_names=None,
    require_recorded_sepset=True,
    collider_triples=None,
):
    """Orient a skeleton with optional conservative evidence and hard constraints."""
    if rule_stats is not None:
        for number in range(11):
            rule_stats.setdefault(f"R{number}", 0)
    return _orient_skeleton(
        pag,
        sepset,
        unfVect,
        [True] * 10 if rules is None else rules,
        orientCollider,
        background_knowledge,
        node_names,
        stats=rule_stats,
        evidence="recorded" if require_recorded_sepset else "optional",
        colliders=collider_triples,
    )
