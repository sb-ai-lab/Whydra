"""Reusable GES candidate records with conservative numerical refinement.

Structural rows depend on the immutable graph signature. Numeric rows also
depend on their child's score revision, so refining a shared local family
cannot leave a stale delta in a retained candidate row.
"""

import heapq

from ._ges_scan import _mask, _nodes


class GramCandidates:
    def __init__(
        self, cache, size, max_operator_set, max_parents, min_delta, *, forward
    ):
        self.cache, self.size = cache, size
        self.max_operator_set, self.max_parents = max_operator_set, max_parents
        self.min_delta, self.forward = min_delta, forward
        self.entries = {}
        self.keys = [{} for _ in range(size)]
        self.rebuilt_targets = self.reused_targets = 0

    def _key(self, child, mask):
        keys = self.keys[child]
        if mask not in keys:
            keys[mask] = child, frozenset(_nodes(mask))
        return keys[mask]

    def prepare(self, view, subsets, record_pool, statistics, record_summary):
        missing = []
        everyone = (1 << self.size) - 1
        for y in range(self.size):
            signature = view.forward_signature(y)
            previous = self.entries.get(y)
            if previous is not None and signature == previous["signature"]:
                pools = previous["pools"]
                if self.max_operator_set is None:
                    record_summary(
                        len(pools), max(pools, default=0), statistics=statistics
                    )
                else:
                    for count in pools:
                        record_pool(count, self.max_operator_set, statistics=statistics)
                self.reused_targets += 1
                continue
            records, pools = [], []
            pa_y, ne_y = view.parents[y], view.neighbours[y]
            sources = (
                (everyone & ~view.adjacent[y] & ~(1 << y))
                if self.forward
                else pa_y | ne_y
            )
            for x in _nodes(sources):
                na = ne_y & view.adjacent[x]
                pool = (ne_y & ~view.adjacent[x] & ~(1 << x)) if self.forward else na
                pools.append(pool.bit_count())
                for subset in subsets(
                    _nodes(pool), self.max_operator_set, statistics=statistics
                ):
                    if self.forward:
                        negative = pa_y | na | _mask(subset)
                        positive = negative | (1 << x)
                        if (
                            self.max_parents is not None
                            and positive.bit_count() > self.max_parents
                        ):
                            continue
                    else:
                        base = pa_y | (na & ~_mask(subset))
                        positive, negative = base & ~(1 << x), base | (1 << x)
                    pos, neg = self._key(y, positive), self._key(y, negative)
                    records.append((x, y, subset, pos, neg))
                    missing.extend((pos, neg))
            self.entries[y] = {
                "signature": signature,
                "records": records,
                "pools": pools,
                "revision": -1,
            }
            self.rebuilt_targets += 1
        self.cache.prefetch(missing)

    def _row(self, y):
        entry = self.entries[y]
        revision = self.cache._revisions[y]
        if entry["revision"] != revision:
            row = []
            for x, child, subset, pos, neg in entry["records"]:
                delta = self.cache._cache[pos] - self.cache._cache[neg]
                error = self.cache._errors[pos] + self.cache._errors[neg]
                self.cache.hits += 2
                if delta + error > self.min_delta:
                    row.append((delta, x, child, subset, error, pos, neg))
            row.sort(key=lambda c: (-(c[0] + c[4]), c[1], c[2], c[3]))
            entry["row"], entry["revision"] = row, revision
        return entry["row"]

    def best(self, view):
        validity = view.insert_valid if self.forward else view.delete_valid
        checked = {}
        while True:
            rows = [self._row(y) for y in range(self.size)]
            heap = []
            for y, row in enumerate(rows):
                if row:
                    delta, x, child, subset, error, _, _ = row[0]
                    heap.append((-(delta + error), x, child, subset, y, 0))
            heapq.heapify(heap)
            refined = False
            while heap:
                _, x, y, subset, row_index, position = heapq.heappop(heap)
                candidate = rows[row_index][position]
                key = x, y, subset
                if key not in checked:
                    checked[key] = validity(x, y, subset)
                if checked[key]:
                    delta, _, _, _, error, pos, neg = candidate
                    if error:
                        self.cache.refine((pos, neg))
                        refined = True
                        break
                    # This exact candidate outranks every remaining upper
                    # bound, including the original lexicographic tie-break.
                    return candidate[:4]
                position += 1
                if position < len(rows[row_index]):
                    delta, x, y, subset, error, _, _ = rows[row_index][position]
                    heapq.heappush(
                        heap, (-(delta + error), x, y, subset, row_index, position)
                    )
            if not refined:
                return None
