"""Candidate sets for the classical FCI and CFCI refinement step."""

from itertools import combinations, filterfalse


class PossibleDsepRow:
    """Candidate targets and conditioning variables for one Possible-D-SEP row."""

    def __init__(self, source, reachable, neighbors, protected):
        self.possible = tuple(node for node in reachable if node != source)
        self.neighbors = frozenset(neighbors)
        self.outside = tuple(
            node for node in self.possible if node not in self.neighbors
        )
        self.targets = (
            [node for node in neighbors if frozenset((source, node)) not in protected]
            if protected
            else neighbors
        )

    def pool(self, target):
        """Return the possible conditioning nodes other than the target."""
        return tuple(node for node in self.possible if node != target)

    def levels(self, target, cap):
        """Yield conditioning-set sizes and candidate combinations up to the cap."""
        limit = int(min(cap, len(self.possible)))
        if limit > 0:
            yield 1, combinations(self.outside, 1)
        if limit > 1:
            pool = self.pool(target)
            for size in range(2, min(limit, len(pool)) + 1):
                yield size, pdsep_conditions(pool, self.outside, size)

    def history_entry(self):
        """Return the conditioning pool for the final target followed by that target."""
        if not self.targets:
            return []
        last = self.targets[-1]
        return [*self.pool(last), last]


def pdsep_conditions(pool, outside, size):
    """Generate candidate sets of the requested size that include an outside node."""
    if size == 1:
        return combinations(outside, 1)
    subsets = combinations(pool, size)
    if size > len(pool) - len(outside):
        return subsets
    return filterfalse(frozenset(outside).isdisjoint, subsets)
