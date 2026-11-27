"""Bounded FCI+ closures invalidated by the separator pairs they read."""

from collections import OrderedDict, defaultdict, deque

import numpy as np


class HierarchyCache:
    """Cache closures reached by following the graph's separating sets."""

    def __init__(self, cg, decode, capacity=1024):
        self.decode = decode
        self.capacity = capacity
        self.entries = OrderedDict()
        self.readers = defaultdict(set)
        self.separators = {}
        self.pair_versions = defaultdict(int)
        self.graph_snapshot = None
        self.graph_version = 0
        self.hits = self.misses = self.invalidations = 0
        self.sync(cg)

    def _drop(self, seed):
        entry = self.entries.pop(seed, None)
        if entry is not None:
            for pair in entry[1]:
                self.readers[pair].discard(seed)
                if not self.readers[pair]:
                    del self.readers[pair]

    def update_pairs(self, cg, pairs):
        """Invalidate affected closures after an internal separating-set write and retain unrelated closures."""
        for pair in pairs:
            x, y = pair
            value = frozenset(self.decode(cg.sepset[x][y]))
            if self.separators.get(pair) == value:
                continue
            self.separators[pair] = value
            self.pair_versions[pair] += 1
            for seed in tuple(self.readers.get(pair, ())):
                self._drop(seed)
                self.invalidations += 1

    def sync(self, cg):
        """Catch arbitrary external/in-place edits once at each DSEP entry.

        Closures depend on sepsets, not endpoint marks. Graph versions are kept
        separately: changed topology affects fresh candidate walks but must not
        throw away closures whose separator dependencies remain valid.
        """
        pag = cg.G.graph
        if self.graph_snapshot is not None and self.graph_snapshot.shape != pag.shape:
            self.invalidations += len(self.entries)
            self.entries.clear()
            self.readers.clear()
            self.separators.clear()
            self.pair_versions.clear()
        if self.graph_snapshot is None or not np.array_equal(self.graph_snapshot, pag):
            self.graph_snapshot = np.array(pag, copy=True)
            self.graph_version += 1
        pairs = [(x, y) for x in range(len(pag)) for y in range(len(pag)) if x != y]
        self.update_pairs(cg, pairs)

    def closure(self, seeds):
        """Return the separating-set closure of the seeds, reusing cached results."""
        seed = frozenset(seeds)
        cached = self.entries.get(seed)
        if cached is not None:
            self.hits += 1
            self.entries.move_to_end(seed)
            return set(cached[0])
        self.misses += 1
        reached, pending, dependencies = set(seed), deque(seed), set()
        while pending:
            node = pending.popleft()
            for other in tuple(reached):
                if node == other:
                    continue
                dependencies.update(((node, other), (other, node)))
                added = (
                    self.separators[node, other] | self.separators[other, node]
                ) - reached
                reached.update(added)
                pending.extend(added)
        self.entries[seed] = frozenset(reached), frozenset(dependencies)
        for pair in dependencies:
            self.readers[pair].add(seed)
        if len(self.entries) > self.capacity:
            self._drop(next(iter(self.entries)))
        return reached


def hierarchy_cache(cg, decode):
    """Return the graph's hierarchy cache for the supplied separating-set decoder."""
    cache = getattr(cg, "_fci_plus_hierarchy_cache", None)
    if cache is None or cache.decode is not decode:
        cache = HierarchyCache(cg, decode)
        cg._fci_plus_hierarchy_cache = cache
    else:
        cache.sync(cg)
    return cache


def hierarchy_changed(cg, pairs=(), *, graph=False):
    """Notify the hierarchy cache of changes to separating sets and optionally graph marks."""
    cache = getattr(cg, "_fci_plus_hierarchy_cache", None)
    if cache is not None:
        cache.update_pairs(cg, pairs)
        if graph and not np.array_equal(cache.graph_snapshot, cg.G.graph):
            cache.graph_snapshot = np.array(cg.G.graph, copy=True)
            cache.graph_version += 1
