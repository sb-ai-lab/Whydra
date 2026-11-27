"""Immutable graph views and local candidate reuse for GES.

The view owns Python values rather than the mutable adjacency array. Insert
and Delete mutate that array before completion, so object identity is never
used to decide whether a candidate can be retained.
"""

import heapq

import numpy as np


def _nodes(mask):
    """Yield the indices of set bits in a Python integer, in increasing order."""
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _mask(nodes):
    value = 0
    for node in nodes:
        value |= 1 << int(node)
    return value


def _row_masks(matrix):
    packed = np.packbits(matrix, axis=1, bitorder="little")
    return tuple(int.from_bytes(row.tobytes(), "little") for row in packed)


class GraphView:
    """Relations and path checks for exactly one immutable graph scan."""

    def __init__(self, matrix):
        present = np.asarray(matrix, dtype=bool)
        self.size = len(present)
        self.adjacent = _row_masks(present | present.T)
        self.neighbours = _row_masks(present & present.T)
        self.parents = _row_masks(present.T & ~present)
        self.successors = _row_masks(present)
        self._reach = {}

    def clique(self, mask):
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            remaining ^= bit
            if remaining & ~self.adjacent[bit.bit_length() - 1]:
                return False
        return True

    def reachable(self, source, block):
        """All semidirected destinations avoiding ``block``, including source."""
        key = source, block
        if key not in self._reach:
            if block & (1 << source):
                self._reach[key] = 0
            else:
                seen = frontier = 1 << source
                while frontier:
                    next_nodes = 0
                    for node in _nodes(frontier):
                        next_nodes |= self.successors[node]
                    frontier = next_nodes & ~block & ~seen
                    seen |= frontier
                self._reach[key] = seen
        return self._reach[key]

    def insert_valid(self, x, y, subset):
        block = (self.neighbours[y] & self.adjacent[x]) | _mask(subset)
        return self.clique(block) and not (self.reachable(y, block) & (1 << x))

    def delete_valid(self, x, y, subset):
        block = (self.neighbours[y] & self.adjacent[x]) & ~_mask(subset)
        return self.clique(block)

    def forward_signature(self, y):
        """Every graph relation that affects target y's candidates and scores.

        Sources depend on adjacency(y), bases on Pa(y) and Ne(y), and each
        NA_yx/pool on adjacency(x) intersect Ne(y). The latter is fully fixed
        by the adjacency rows of Ne(y), because adjacency is symmetric.
        Global semidirected reachability is deliberately absent: validity is
        rechecked against the new view on every scan, including reused rows.
        """
        return (
            self.parents[y],
            self.neighbours[y],
            self.adjacent[y],
            tuple((t, self.adjacent[t]) for t in _nodes(self.neighbours[y])),
        )


class MaskedScores:
    """Reuse exact local scores without repeatedly canonicalizing parent sets.

    The underlying score implementation, including its numerical policy, is
    unchanged. Integer keys merely avoid repeated frozenset construction and
    lookup through LocalScoreCache for families already used by this search.
    """

    def __init__(self, cache, size):
        self.cache = cache
        self.by_target = [{} for _ in range(size)]
        self.reuses = 0

    def score(self, target, parent_mask):
        families = self.by_target[target]
        value = families.get(parent_mask)
        if value is None:
            value = self.cache.score(target, tuple(_nodes(parent_mask)))
            families[parent_mask] = value
        else:
            self.reuses += 1
        return value


def best_candidate(rows, view, *, forward):
    """Merge sorted target rows in the original (-delta, x, y, subset) order."""
    heap = []
    for row_index, candidates in enumerate(rows):
        if candidates:
            delta, x, y, subset = candidates[0]
            heap.append((-delta, x, y, subset, row_index, 0))
    heapq.heapify(heap)
    validity = view.insert_valid if forward else view.delete_valid
    while heap:
        _, x, y, subset, row_index, position = heapq.heappop(heap)
        candidate = rows[row_index][position]
        if validity(x, y, subset):
            return candidate
        position += 1
        if position < len(rows[row_index]):
            delta, x, y, subset = rows[row_index][position]
            heapq.heappush(heap, (-delta, x, y, subset, row_index, position))
    return None


class ForwardCandidates:
    """Retain target rows only when their complete local dependency is equal."""

    def __init__(self, scores, size, max_operator_set, max_parents, min_delta):
        self.scores = scores
        self.size = size
        self.max_operator_set = max_operator_set
        self.max_parents = max_parents
        self.min_delta = min_delta
        self.entries = {}
        self.rebuilt_targets = 0
        self.reused_targets = 0

    def rows(self, view, subsets, record_pool, statistics, record_summary=None):
        rows = []
        everyone = (1 << self.size) - 1
        for y in range(self.size):
            signature = view.forward_signature(y)
            previous = self.entries.get(y)
            if previous is not None and signature == previous[0]:
                candidates, pool_sizes, largest = previous[1:]
                # Approximation diagnostics describe the same logical search
                # sweeps as before, even when their score work is reused.
                if self.max_operator_set is None and record_summary is not None:
                    record_summary(len(pool_sizes), largest, statistics=statistics)
                else:
                    for size in pool_sizes:
                        record_pool(size, self.max_operator_set, statistics=statistics)
                self.reused_targets += 1
            else:
                candidates, pool_sizes = [], []
                pa_y, ne_y = view.parents[y], view.neighbours[y]
                sources = everyone & ~view.adjacent[y] & ~(1 << y)
                for x in _nodes(sources):
                    na = ne_y & view.adjacent[x]
                    pool = ne_y & ~view.adjacent[x] & ~(1 << x)
                    pool_sizes.append(pool.bit_count())
                    for subset in subsets(
                        _nodes(pool), self.max_operator_set, statistics=statistics
                    ):
                        base = pa_y | na | _mask(subset)
                        with_x = base | (1 << x)
                        if (
                            self.max_parents is not None
                            and with_x.bit_count() > self.max_parents
                        ):
                            continue
                        delta = self.scores.score(y, with_x) - self.scores.score(
                            y, base
                        )
                        if delta > self.min_delta:
                            candidates.append((delta, x, y, subset))
                candidates.sort(key=lambda c: (-c[0], c[1], c[2], c[3]))
                self.entries[y] = (
                    signature,
                    candidates,
                    pool_sizes,
                    max(pool_sizes, default=0),
                )
                self.rebuilt_targets += 1
            rows.append(candidates)
        return rows


def backward_rows(view, scores, subsets, max_operator_set, min_delta, statistics):
    rows = []
    for y in range(view.size):
        candidates = []
        for x in _nodes(view.parents[y] | view.neighbours[y]):
            na = view.neighbours[y] & view.adjacent[x]
            for subset in subsets(_nodes(na), max_operator_set, statistics=statistics):
                base = view.parents[y] | (na & ~_mask(subset))
                delta = scores.score(y, base & ~(1 << x)) - scores.score(
                    y, base | (1 << x)
                )
                if delta > min_delta:
                    candidates.append((delta, x, y, subset))
        candidates.sort(key=lambda c: (-c[0], c[1], c[2], c[3]))
        rows.append(candidates)
    return rows
