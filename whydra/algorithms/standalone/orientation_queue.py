"""Ordered incremental rule candidates with conservative path dependencies."""

from collections import defaultdict
from heapq import heapify, heappop, heappush
from typing import TypedDict

import numpy as np

_Pair = tuple[int, int]


class _ActiveRule(TypedDict):
    state: dict
    heap: list[tuple[_Pair, _Pair]]
    queued: set[_Pair]
    allowed: set[_Pair]
    cursor: _Pair
    row_order: bool


class OrientationQueue:
    """Keep rule order and each rule's original candidate-mask snapshot.

    Local mark rules read endpoint rows/columns. Their other adjacency tests
    are invariant while only marks change; any topology change invalidates all
    candidates. Path rules depend on the entire undirected component, including
    nodes far from the candidate endpoints. Guard writes enqueue dependencies
    immediately, so a later candidate in the same rule sees earlier writes.
    """

    def __init__(self, pag, *, enabled=True):
        self.enabled = enabled
        self.snapshot = np.array(pag, copy=True)
        self.states = {}
        self.active = None
        self.stats = {
            "candidate_snapshots": 0,
            "candidates_visited": 0,
            "candidates_skipped": 0,
            "mark_events": 0,
            "topology_events": 0,
        }
        self._components()

    def _components(self):
        adjacent = (self.snapshot != 0) | (self.snapshot.T != 0)
        self.components = {}
        remaining = set(range(len(adjacent)))
        while remaining:
            first = remaining.pop()
            reached, pending = {first}, [first]
            while pending:
                node = pending.pop()
                added = set(map(int, np.flatnonzero(adjacent[node]))) - reached
                reached.update(added)
                remaining.difference_update(added)
                pending.extend(added)
            component = frozenset(reached)
            for node in component:
                self.components[node] = component

    def _enqueue(self, state, pairs):
        for pair in pairs:
            state["pending"].add(pair)
            active = self.active
            if (
                active is not None
                and active["state"] is state
                and pair in active["allowed"]
            ):
                key = pair if active["row_order"] else pair[::-1]
                if key > active["cursor"] and pair not in active["queued"]:
                    heappush(active["heap"], (key, pair))
                    active["queued"].add(pair)

    def _changed(self, source, target):
        self.stats["mark_events"] += 1
        for state in self.states.values():
            nodes = (
                self.components[source] | self.components[target]
                if state["paths"]
                else (source, target)
            )
            affected = set()
            for node in nodes:
                affected.update(state["by_node"].get(node, ()))
            self._enqueue(state, affected)

    def sync(self, pag):
        """Notice external writes, RFCI removals and BK checkpoint rollbacks."""
        changed = np.argwhere(self.snapshot != pag)
        if not len(changed):
            return
        topology = any(
            (self.snapshot[a, b] == 0) != (pag[a, b] == 0) for a, b in changed
        )
        self.snapshot = np.array(pag, copy=True)
        if topology:
            self.stats["topology_events"] += 1
            self._components()
            for state in self.states.values():
                self._enqueue(state, state["known"])
        else:
            for source, target in changed:
                self._changed(int(source), int(target))

    def guard(self, original):
        """Wrap an orientation guard so successful mark changes notify this queue."""
        queue = self

        class NotifyingGuard:
            def __getattr__(self, name):
                return getattr(original, name)

            def set_mark(self, pag, source, target, mark):
                before = pag[source, target]
                result = original.set_mark(pag, source, target, mark)
                if pag[source, target] != before:
                    # Standard guards write exactly this endpoint. sync() at
                    # rule boundaries also covers external/custom guard edits.
                    if (before == 0) != (pag[source, target] == 0):
                        queue.sync(pag)
                    else:
                        queue.snapshot[source, target] = pag[source, target]
                        queue._changed(source, target)
                return result

        return NotifyingGuard()

    def candidates(self, rule, pag, mask, *, paths=False, row_order=False):
        """Yield candidate pairs in scan order and track updates made during orientation."""
        self.sync(pag)
        rows, columns = np.nonzero(mask if row_order else mask.T)
        pairs = list(zip(rows.tolist(), columns.tolist(), strict=False))
        if not row_order:
            pairs = [(column, row) for row, column in pairs]
        self.stats["candidate_snapshots"] += len(pairs)
        if not self.enabled:
            yield from pairs
            return
        state = self.states.setdefault(
            rule,
            {
                "paths": paths,
                "known": set(),
                "by_node": defaultdict(set),
                "pending": set(),
            },
        )
        allowed = set(pairs)
        for pair in allowed - state["known"]:
            state["known"].add(pair)
            state["by_node"][pair[0]].add(pair)
            state["by_node"][pair[1]].add(pair)
            state["pending"].add(pair)
        # A newly eligible candidate must be checked even if an external edit
        # changed its eligibility without going through our notifying guard.
        state["pending"].update(allowed - state.get("previous_allowed", set()))
        state["previous_allowed"] = allowed
        state["pending"].intersection_update(allowed)
        pending = state["pending"].copy()
        heap = [(pair if row_order else pair[::-1], pair) for pair in pending]
        heapify(heap)
        active: _ActiveRule = {
            "state": state,
            "heap": heap,
            "queued": pending,
            "allowed": allowed,
            "cursor": (-1, -1),
            "row_order": row_order,
        }
        self.active = active
        visited = 0
        try:
            while heap:
                key, pair = heappop(heap)
                active["queued"].remove(pair)
                active["cursor"] = key
                state["pending"].discard(pair)
                visited += 1
                self.stats["candidates_visited"] += 1
                yield pair
        finally:
            self.stats["candidates_skipped"] += len(pairs) - visited
            self.active = None
