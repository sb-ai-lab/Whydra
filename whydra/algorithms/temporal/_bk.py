"""Use Whydra's phased background knowledge on temporal endpoint matrices."""

import numpy as np

from ...background_knowledge import (
    BackgroundKnowledge,
    BKEvent,
    BKPhase,
    MatrixEncoding,
    apply_bk_checkpoint,
)
from .graphs import lagged_node_names, matrix_to_temporal, temporal_to_matrix


class TemporalKnowledge:
    """Translate current/lagged node references for the native search engines.

    Bare variable names denote t. ``X1[t-2] -> X2`` constrains lag two.
    PRE_SEARCH protects adjacencies; later requirements only orient surviving
    edges, exactly as in the other Whydra algorithm families.
    """

    def __init__(self, knowledge, names, tau_min, tau_max, roles=None):
        if not isinstance(knowledge, BackgroundKnowledge):
            msg = "background_knowledge must be a Whydra BackgroundKnowledge instance"
            raise TypeError(msg)
        self.knowledge = knowledge
        self.n = len(names)
        self.names = lagged_node_names(names, tau_max)
        self.tau_min, self.tau_max = tau_min, tau_max
        self.roles = roles or {}
        knowledge.validate(self.names)
        self.guards = {
            phase: knowledge.orientation_guard(self.names, phase=phase)
            for phase in BKPhase
        }
        for guard in self.guards.values():
            for source, target in guard.required | guard.forbidden:
                if source >= self.n and target >= self.n:
                    msg = "Temporal BK edges must touch a current-time node"
                    raise ValueError(msg)
            for source, target in guard.required:
                if target >= self.n:
                    msg = "Required temporal edges cannot point into the past"
                    raise ValueError(msg)
                lag, i = divmod(source, self.n)
                if lag < tau_min:
                    msg = "Required edge lies outside tau_min/tau_max"
                    raise ValueError(msg)
                source_role, target_role = (
                    self.roles.get(i, "system"),
                    self.roles.get(target, "system"),
                )
                if (
                    target_role.endswith("dummy")
                    or (
                        lag
                        and source_role
                        in {"time_dummy", "space_dummy", "space_context"}
                    )
                    or (source_role == "system" and target_role != "system")
                    or {source_role, target_role} == {"time_context", "space_context"}
                ):
                    msg = "Required edge conflicts with J-PCMCI+ context roles"
                    raise ValueError(msg)
        self._protected = self.guards[BKPhase.PRE_SEARCH].required
        self.before_orientation_applied = False

    def protected(self, i, j, lag):
        return (lag * self.n + i, j) in self._protected or (
            lag == 0 and (j, i) in self._protected
        )

    def permits(self, i, j, lag):
        source = lag * self.n + i
        guard = self.guards[BKPhase.BETWEEN_ORIENTATION]
        return (source, j) not in guard.forbidden and (j, source) not in guard.required

    def adjacency_allowed(self, i, j, lag):
        guard = self.guards[BKPhase.PRE_SEARCH]
        source = lag * self.n + i
        return (
            (source, j) not in guard.forbidden
            if lag
            else not {(i, j), (j, i)} <= guard.forbidden
        )

    def checkpoint(self, graph, phase):
        matrix = temporal_to_matrix(graph)
        before = matrix.copy()
        # Time order already excludes the reverse direction of a lagged edge.
        # Forbidding its forward direction therefore removes the adjacency;
        # the generic CPDAG checkpoint must not reverse it into the past.
        removed = []
        for source, target in self.guards[phase].forbidden:
            if source >= self.n and target < self.n and matrix[source, target]:
                matrix[source, target] = matrix[target, source] = 0
                removed.append((source, target))
        checkpoint = f"temporal.{phase.value}"
        report = apply_bk_checkpoint(
            self.knowledge,
            matrix,
            self.names,
            phase=phase,
            checkpoint=checkpoint,
            encoding=MatrixEncoding.STANDARD,
            graph_kind="cpdag",
        )
        if report is not None:
            for source, target in removed:
                report.events.append(
                    BKEvent(
                        phase,
                        checkpoint,
                        "apply",
                        source,
                        target,
                        "forbidden lagged edge removed; time order excludes reversal",
                    )
                )
            report.changed |= not np.array_equal(before, matrix)
        return matrix_to_temporal(matrix, self.n, self.tau_max)

    def assumptions(self, engine, supplied):
        links = engine._set_link_assumptions(supplied, self.tau_min, self.tau_max)
        marks = np.full((self.n, self.n, self.tau_max + 1), "", dtype="<U3")
        for j, incoming in links.items():
            for (i, lag), mark in incoming.items():
                marks[i, j, -lag] = mark
        # Sparse link assumptions may exclude a required edge. Fail clearly
        # instead of silently inventing an adjacency outside the search space.
        for source, target in self._protected:
            lag, i = divmod(source, self.n)
            if not marks[i, target, lag]:
                msg = "PRE_SEARCH required edge is absent from link_assumptions"
                raise ValueError(msg)
        constrained = self.checkpoint(marks, BKPhase.PRE_SEARCH)
        result: dict[int, dict[tuple[int, int], str]] = {j: {} for j in range(self.n)}
        for i, j, lag in np.ndindex(marks.shape):
            mark = str(constrained[i, j, lag])
            if not mark:
                continue
            fixed = self.protected(i, j, lag) or (
                marks[i, j, lag] and "?" not in marks[i, j, lag]
            )
            result[j][(i, -lag)] = mark if fixed else mark[0] + "?" + mark[2]
        return result

    def before_orientation(self, graph):
        if self.before_orientation_applied:
            return graph
        self.before_orientation_applied = True
        return self.checkpoint(graph, BKPhase.BETWEEN_ORIENTATION)
