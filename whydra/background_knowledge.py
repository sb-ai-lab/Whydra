"""Background-knowledge constraints shared by causal discovery algorithms.

Every constraint can be assigned independently to one or more execution phases:
before skeleton search, between orientation passes, or after orientation.
Unscoped constraints use ``enabled_phases`` for backwards compatibility with
older configurations.
"""

from __future__ import annotations

import warnings
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Hashable, Iterable, Iterator, Sequence

import numpy as np


NodeRef = int | str | Hashable
Direction = tuple[NodeRef, NodeRef]
PhasedDirection = tuple[NodeRef, NodeRef, "BKPhase | str | Iterable[BKPhase | str]"]


class BKPhase(str, Enum):
    PRE_SEARCH = "pre_search"
    BETWEEN_ORIENTATION = "between_orientation"
    POST_ORIENTATION = "post_orientation"


class MatrixEncoding(str, Enum):
    """Endpoint codes used by this project."""

    STANDARD = "standard"  # NULL=0, TAIL=-1, ARROW=1, CIRCLE=2
    PCALG = "pcalg"        # NULL=0, CIRCLE=1, ARROW=2, TAIL=3


class BKConflictWarning(UserWarning):
    """A background-knowledge constraint could not be satisfied.

    Emitted instead of failing: the graph the algorithm returns is still a
    valid one, it just does not carry the constraint the user asked for.
    Silence it with ``warnings.simplefilter("ignore", BKConflictWarning)``.
    """


@dataclass(frozen=True)
class BKEvent:
    phase: BKPhase
    checkpoint: str
    action: str
    source: int
    target: int
    detail: str


@dataclass
class BKApplicationReport:
    phase: BKPhase
    checkpoint: str
    active: bool
    run_id: object = 0
    changed: bool = False
    events: list[BKEvent] = field(default_factory=list)

    @property
    def conflicts(self) -> list[BKEvent]:
        return [event for event in self.events if event.action == "conflict"]


@dataclass(frozen=True)
class _CompiledKnowledge:
    forbidden: frozenset[tuple[int, int]]
    required: frozenset[tuple[int, int]]
    allowed: frozenset[tuple[int, int]]


def endpoint_codes(encoding: MatrixEncoding | str) -> tuple[int, int, int, int]:
    """``(null, tail, arrow, circle)`` endpoint codes for ``encoding``.

    The two encodings differ in every non-zero code, so any function reading or
    writing endpoint marks must be told which one it is looking at.
    """

    encoding = MatrixEncoding(encoding)
    if encoding is MatrixEncoding.STANDARD:
        return 0, -1, 1, 2
    return 0, 3, 2, 1


def _find_cycle(edges: Iterable[tuple[object, object]]) -> list | None:
    """Return one directed cycle as ``[n0, n1, ..., n0]``, or ``None``.

    Iterative DFS with grey/black colouring: the constraint sets are small, but
    a long required chain should not be able to blow the recursion limit.
    """

    adjacency: dict[object, list] = {}
    for source, target in edges:
        adjacency.setdefault(source, []).append(target)
        adjacency.setdefault(target, [])

    WHITE, GREY, BLACK = 0, 1, 2
    colour = dict.fromkeys(adjacency, WHITE)

    for root in sorted(adjacency, key=repr):
        if colour[root] != WHITE:
            continue
        path: list = []
        # Each frame is (node, iterator over its successors).
        stack = [(root, iter(adjacency[root]))]
        colour[root] = GREY
        path.append(root)
        while stack:
            node, successors = stack[-1]
            advanced = False
            for nxt in successors:
                if colour[nxt] == GREY:
                    return path[path.index(nxt):] + [nxt]
                if colour[nxt] == WHITE:
                    colour[nxt] = GREY
                    path.append(nxt)
                    stack.append((nxt, iter(adjacency[nxt])))
                    advanced = True
                    break
            if not advanced:
                colour[node] = BLACK
                path.pop()
                stack.pop()
    return None


def _directed_edges(matrix: np.ndarray, tail: int, arrow: int) -> list[tuple[int, int]]:
    """Fully directed edges ``u -> v`` of an endpoint matrix."""

    size = matrix.shape[0]
    return [
        (u, v)
        for u in range(size)
        for v in range(size)
        if u != v and matrix[v, u] == tail and matrix[u, v] == arrow
    ]


def find_directed_cycle(matrix: np.ndarray, tail: int, arrow: int) -> list | None:
    """One directed cycle of an endpoint matrix, or ``None`` if it is acyclic."""

    return _find_cycle(_directed_edges(matrix, tail, arrow))


class BKOrientationGuard:
    """Проверяет, не нарушает ли отдельная запись метки ограничения BK.

    Однократного применения BK до правил ориентации недостаточно: Meek и
    R1–R10 могут переписать конец ребра позже. Правила проводят каждую запись
    через этот guard, и тогда BK остаётся жёстким инвариантом всего поиска,
    а не подсказкой в одной точке.

    Пустой guard (BK нет или на этой фазе ничего не активно) ничего не
    запрещает — обвязка правил в этом случае полностью прозрачна.
    """

    __slots__ = ("required", "forbidden", "null", "tail", "arrow")

    def __init__(self, compiled: "_CompiledKnowledge | None", encoding: MatrixEncoding | str):
        self.required = frozenset(compiled.required) if compiled else frozenset()
        self.forbidden = frozenset(compiled.forbidden) if compiled else frozenset()
        self.null, self.tail, self.arrow, _circle = endpoint_codes(encoding)

    def __bool__(self) -> bool:
        return bool(self.required or self.forbidden)

    def allows_mark(self, matrix: np.ndarray, source: int, target: int, mark: int) -> bool:
        """Можно ли записать ``mark`` как метку у ``target`` на ребре (source, target)."""

        if not self:
            return True
        mark = int(mark)

        for a, b in self.required:
            if (source, target) == (a, b):
                # Метка у b обязана остаться стрелкой; ноль стёр бы ребро.
                if mark != self.arrow:
                    return False
            elif (source, target) == (b, a):
                # Метка у a обязана остаться хвостом.
                if mark != self.tail:
                    return False

        for a, b in self.forbidden:
            if (source, target) == (a, b):
                # Запись даст a -> b, если у b встанет стрелка, а у a уже хвост.
                if mark == self.arrow and int(matrix[b, a]) == self.tail:
                    return False
            elif (source, target) == (b, a):
                if mark == self.tail and int(matrix[a, b]) == self.arrow:
                    return False
        return True

    def set_mark(self, matrix: np.ndarray, source: int, target: int, mark: int) -> bool:
        """Записать метку, если BK не против. Возвращает, была ли запись выполнена."""

        if not self.allows_mark(matrix, source, target, mark):
            return False
        matrix[source][target] = mark
        return True


def orientation_guard_for(background_knowledge, node_names, *,
                         phase: "BKPhase | str" = BKPhase.BETWEEN_ORIENTATION,
                         encoding: "MatrixEncoding | str" = MatrixEncoding.STANDARD
                         ) -> "BKOrientationGuard":
    """Guard для правил ориентации; без BK — пустой и полностью прозрачный."""

    if background_knowledge is None:
        return BKOrientationGuard(None, encoding)
    return background_knowledge.orientation_guard(
        node_names, phase=phase, encoding=encoding
    )


def bk_allows_edge(guard, matrix, source: int, target: int,
                   at_source: int, at_target: int) -> bool:
    """Разрешает ли BK записать пару меток ребра ``source``-``target``.

    Одна проверка на все реализации PC: раньше её имел только
    pc_algo.PCAlgorithm, а функциональные модули писали метку напрямую и
    полагались на чекпоинт после фазы — тот направление чинил, но уже
    потерянный коллайдер не возвращал.
    """

    if guard is None or not guard:
        return True
    return (guard.allows_mark(matrix, target, source, int(at_source))
            and guard.allows_mark(matrix, source, target, int(at_target)))


def _format_cycle(cycle: Iterable, names: Sequence[object] | None = None) -> str:
    def label(node):
        if names is None:
            return str(node)
        try:
            return str(names[node])
        except (TypeError, IndexError):
            return str(node)

    return " -> ".join(label(n) for n in cycle)


class BackgroundKnowledge:
    """Direction constraints with explicit activation phases.

    ``allowed`` has deliberately weak semantics: it only cancels the matching
    ``forbidden`` direction and never modifies a graph by itself.
    """

    def __init__(
        self,
        *,
        forbidden: Iterable[Direction | PhasedDirection] = (),
        required: Iterable[Direction | PhasedDirection] = (),
        allowed: Iterable[Direction | PhasedDirection] = (),
        enabled_phases: Iterable[BKPhase | str] = (BKPhase.BETWEEN_ORIENTATION,),
        strict_required_adjacency: bool = False,
        history_limit: int | None = 1000,
    ) -> None:
        self.enabled_phases = {BKPhase(phase) for phase in enabled_phases}
        #: Если required-ребро защищалось на PRE_SEARCH, а к моменту ориентации
        #: его нет — это нарушенный инвариант, а не конфликт с данными. По
        #: умолчанию сообщается предупреждением; ``True`` превращает его в
        #: ValueError для тех, кому нужен жёсткий контракт.
        self.strict_required_adjacency = bool(strict_required_adjacency)
        self.forbidden: set[Direction] = set()
        self.required: set[Direction] = set()
        self.allowed: set[Direction] = set()
        self._phase_overrides: dict[str, dict[Direction, frozenset[BKPhase]]] = {
            "forbidden": {},
            "required": {},
            "allowed": {},
        }
        #: Кольцевой буфер: объект переиспользуется между прогонами, и без
        #: границы отчёты копятся неограниченно. ``history_limit=None`` снимает
        #: ограничение, самые старые отчёты вытесняются молча.
        self.history: deque[BKApplicationReport] = deque(maxlen=history_limit)
        self._run_counter = 0
        self._run_id: object = 0
        self._compiled_cache: dict[tuple[object, ...], _CompiledKnowledge] = {}
        self._add_initial("forbidden", forbidden)
        self._add_initial("required", required)
        self._add_initial("allowed", allowed)

    @staticmethod
    def _normalise_phases(
        phases: BKPhase | str | Iterable[BKPhase | str],
    ) -> frozenset[BKPhase]:
        if isinstance(phases, (BKPhase, str)):
            phases = (phases,)
        result = frozenset(BKPhase(phase) for phase in phases)
        if not result:
            raise ValueError("A phase-scoped BK constraint needs at least one phase")
        return result

    def _add_initial(self, kind: str, items: Iterable[Direction | PhasedDirection]) -> None:
        for item in items:
            values = tuple(item)
            if len(values) == 2:
                self._add(kind, values[0], values[1], phases=None)
            elif len(values) == 3:
                self._add(kind, values[0], values[1], phases=values[2])
            else:
                raise ValueError(
                    f"{kind} constraints must be (source, target) or "
                    "(source, target, phase_or_phases)"
                )

    def _add(
        self,
        kind: str,
        source: NodeRef,
        target: NodeRef,
        *,
        phases: BKPhase | str | Iterable[BKPhase | str] | None,
    ) -> "BackgroundKnowledge":
        """Add one constraint transactionally.

        The candidate constraint set is validated BEFORE it is committed, so a
        rejected constraint leaves the object exactly as it was. Otherwise a
        caller that catches the ValueError would carry on with a half-applied,
        already-invalid set of background knowledge.
        """

        direction = (source, target)
        scoped = self._normalise_phases(phases) if phases is not None else None

        # --- строим кандидата, не трогая объект ---
        candidate_sets = {k: set(getattr(self, k)) for k in ("forbidden", "required", "allowed")}
        candidate_overrides = {k: dict(v) for k, v in self._phase_overrides.items()}
        candidate_sets[kind].add(direction)
        if scoped is not None:
            previous = candidate_overrides[kind].get(direction, frozenset())
            candidate_overrides[kind][direction] = previous | scoped

        self._validate_sets(candidate_sets, candidate_overrides)

        # --- фиксируем ---
        for k, v in candidate_sets.items():
            setattr(self, k, v)
        self._phase_overrides = candidate_overrides
        self._compiled_cache.clear()
        return self

    # ------------------------------------------------------------------
    #  Статическая валидация
    # ------------------------------------------------------------------

    @staticmethod
    def _phases_of(direction, kind, overrides, enabled_phases):
        explicit = overrides[kind].get(direction)
        return explicit if explicit is not None else frozenset(enabled_phases)

    def _validate_sets(self, sets, overrides, enabled_phases=None) -> None:
        """Reject a constraint set that no graph could ever satisfy.

        These are errors in the background knowledge itself, not conflicts with
        some particular graph: no data and no algorithm can satisfy them, so the
        right moment to complain is when they are written, not halfway through a
        search.
        """

        enabled = self.enabled_phases if enabled_phases is None else enabled_phases

        for kind in ("forbidden", "required", "allowed"):
            for source, target in sets[kind]:
                if source == target:
                    raise ValueError(
                        f"A BK direction cannot be a self-loop: {kind} {source!r} -> {target!r}"
                    )

        # Дальше всё считается ПОФАЗОВО. Запретить направление на одной фазе и
        # потребовать его на другой — легальный сценарий (например, «не
        # ориентируй так во время поиска, но выставь в конце»), поэтому
        # глобальное пересечение required и forbidden было бы ложной тревогой.
        def at(kind: str, phase: BKPhase) -> set:
            return {
                d for d in sets[kind]
                if phase in self._phases_of(d, kind, overrides, enabled)
            }

        PHASES = (BKPhase.PRE_SEARCH, BKPhase.BETWEEN_ORIENTATION, BKPhase.POST_ORIENTATION)
        cumulative: set = set()
        for phase in PHASES:
            required = at("required", phase)
            # `allowed` снимает ровно совпадающий запрет — вычитаем до сравнения.
            forbidden = at("forbidden", phase) - at("allowed", phase)

            contradiction = forbidden & required
            if contradiction:
                pretty = ", ".join(f"{a!r} -> {b!r}" for a, b in sorted(contradiction, key=repr))
                raise ValueError(
                    f"BK directions are both forbidden and required at phase "
                    f"{phase.value!r}: {pretty}"
                )

            for source, target in required:
                if (target, source) in required:
                    raise ValueError(
                        f"Opposite directions are both required at phase {phase.value!r}: "
                        f"{source!r} -> {target!r} and {target!r} -> {source!r}"
                    )

            # Требуемые направления должны складываться в DAG — и на каждой фазе
            # по отдельности, и накопительно: required описывает то, что обязано
            # остаться в графе, поэтому ограничения разных фаз способны замкнуть
            # цикл совместно.
            cycle = _find_cycle(required)
            if cycle is not None:
                raise ValueError(
                    f"Background knowledge contains directed cycle at phase "
                    f"{phase.value!r}: {_format_cycle(cycle)}"
                )
            cumulative |= required
            cycle = _find_cycle(cumulative)
            if cycle is not None:
                raise ValueError(
                    f"Background knowledge contains directed cycle across phases "
                    f"up to {phase.value!r}: {_format_cycle(cycle)}"
                )

    def validate(self, node_names: Sequence[object] | None = None) -> "BackgroundKnowledge":
        """Check the constraint set; raise ``ValueError`` if it is unsatisfiable.

        Without ``node_names`` the symbolic constraints are checked. With them
        the constraints are additionally resolved to indices and re-checked:
        two different references can turn out to be the same node only after
        resolution, so ``require("X1", 0)`` is a self-loop that is invisible
        until then.
        """

        sets = {k: set(getattr(self, k)) for k in ("forbidden", "required", "allowed")}
        self._validate_sets(sets, self._phase_overrides)
        if node_names is not None:
            names = tuple(self._name_of(node) for node in node_names)
            cumulative: set = set()
            for phase in (BKPhase.PRE_SEARCH, BKPhase.BETWEEN_ORIENTATION,
                          BKPhase.POST_ORIENTATION):
                cumulative |= set(self._compile(node_names, phase).required)
                cycle = _find_cycle(cumulative)
                if cycle is not None:
                    raise ValueError(
                        f"Background knowledge contains directed cycle across phases "
                        f"up to {phase.value!r}: {_format_cycle(cycle, names)}"
                    )
        return self

    def forbid(
        self,
        source: NodeRef,
        target: NodeRef,
        *,
        phases: BKPhase | str | Iterable[BKPhase | str] | None = None,
    ) -> "BackgroundKnowledge":
        """Forbid ``source -> target`` at the selected execution phases."""

        return self._add("forbidden", source, target, phases=phases)

    def require(
        self,
        source: NodeRef,
        target: NodeRef,
        *,
        phases: BKPhase | str | Iterable[BKPhase | str] | None = None,
    ) -> "BackgroundKnowledge":
        """Require ``source -> target`` at the selected execution phases."""

        return self._add("required", source, target, phases=phases)

    def allow(
        self,
        source: NodeRef,
        target: NodeRef,
        *,
        phases: BKPhase | str | Iterable[BKPhase | str] | None = None,
    ) -> "BackgroundKnowledge":
        """Cancel the matching prohibition at the selected phases."""

        return self._add("allowed", source, target, phases=phases)

    def enable_phase(self, phase: BKPhase | str) -> "BackgroundKnowledge":
        """Расширить область действия unscoped-ограничений на ещё одну фазу.

        Валидируется так же, как `_add`: расширение задним числом способно
        собрать набор, который конструктор отверг бы (например required на
        between_orientation плюс forbidden на post_orientation), а без
        проверки ValueError прилетал бы только из `_compile` — посреди поиска.
        """

        previous = self.enabled_phases
        self.enabled_phases = set(previous) | {BKPhase(phase)}
        try:
            self.validate()
        except ValueError:
            self.enabled_phases = previous
            raise
        self._compiled_cache.clear()
        return self

    def start_run(self, run_id: object | None = None) -> "BackgroundKnowledge":
        """Open a new independent run (bootstrap replicate, dataset, algorithm).

        The same ``BackgroundKnowledge`` object is meant to be reused across
        runs, so reports produced from here on carry a fresh ``run_id`` and
        conflicts coming from different runs stay separable in :attr:`history`.

        Switching runs is deliberately left to the caller rather than done on
        entry to an algorithm: only the caller knows where a run begins (one
        bootstrap replicate may span several algorithm invocations), and
        ``_run_id`` is plain instance state, so an implicit bump would race
        once one object is shared by concurrently running algorithms.
        Prefer :meth:`run_scope` over calling this directly.

        The compiled-constraint cache is deliberately NOT dropped here: its key
        already carries the node names and the per-phase direction sets, so a
        new node set simply misses. Use :meth:`reset_cache` to reclaim memory.
        """

        self._run_counter += 1
        self._run_id = self._run_counter if run_id is None else run_id
        return self

    @contextmanager
    def run_scope(self, run_id: object | None = None) -> Iterator["BackgroundKnowledge"]:
        """Scope reports to one run, restoring the previous ``run_id`` on exit.

        >>> for _ in range(n_replicates):          # doctest: +SKIP
        ...     with bk.run_scope() as scoped:
        ...         run_algorithm(data, bk=scoped)

        Reports stay in :attr:`history`; :meth:`reports_for_run` splits them.
        """

        previous = self._run_id
        self.start_run(run_id)
        try:
            yield self
        finally:
            self._run_id = previous

    def reset_cache(self) -> "BackgroundKnowledge":
        """Drop the compiled-constraint cache.

        Purely a memory-reclaim knob: entries are keyed by node names and
        per-phase direction sets, so stale ones are never served, they only
        take up space once a node set is out of use.
        """

        self._compiled_cache.clear()
        return self

    def clear_history(self) -> "BackgroundKnowledge":
        """Drop accumulated reports; nothing inside the algorithms reads them back."""

        self.history.clear()
        return self

    def reports_for_run(self, run_id: object) -> list[BKApplicationReport]:
        """Reports recorded for a single run, in application order."""

        return [report for report in self.history if report.run_id == run_id]

    def is_active(self, phase: BKPhase | str) -> bool:
        """True when at least one constraint is actually scheduled for ``phase``.

        Membership of ``phase`` in ``enabled_phases`` is not enough on its own:
        an active report has to mean that something could apply here, otherwise
        the flag says nothing about phases that only other constraints target.
        """

        phase = BKPhase(phase)
        return any(
            self._directions_at_phase(kind, phase)
            for kind in ("forbidden", "required", "allowed")
        )

    def _directions_at_phase(self, kind: str, phase: BKPhase) -> set[Direction]:
        result = set()
        for direction in getattr(self, kind):
            explicit = self._phase_overrides[kind].get(direction)
            if (explicit is not None and phase in explicit) or (
                explicit is None and phase in self.enabled_phases
            ):
                result.add(direction)
        return result

    @staticmethod
    def _name_of(node: object) -> str:
        if isinstance(node, str):
            return node
        name = getattr(node, "name", None)
        if name is not None:
            return str(name)
        get_name = getattr(node, "get_name", None)
        if callable(get_name):
            return str(get_name())
        return str(node)

    def _compile(
        self,
        node_names: Sequence[object],
        phase: BKPhase | str = BKPhase.BETWEEN_ORIENTATION,
    ) -> _CompiledKnowledge:
        phase = BKPhase(phase)
        names = tuple(self._name_of(node) for node in node_names)
        cache_key = (
            names,
            phase,
            frozenset(self._directions_at_phase("forbidden", phase)),
            frozenset(self._directions_at_phase("required", phase)),
            frozenset(self._directions_at_phase("allowed", phase)),
        )
        cached = self._compiled_cache.get(cache_key)
        if cached is not None:
            return cached

        name_to_index = {name: index for index, name in enumerate(names)}

        def resolve(ref: NodeRef) -> int:
            if isinstance(ref, (int, np.integer)):
                index = int(ref)
                if 0 <= index < len(names):
                    return index
                raise ValueError(f"BK node index {index} is outside graph size {len(names)}")
            name = self._name_of(ref)
            if name not in name_to_index:
                raise ValueError(f"Unknown BK node {name!r}; available nodes: {list(names)!r}")
            return name_to_index[name]

        def resolve_all(items: Iterable[Direction]) -> frozenset[tuple[int, int]]:
            result = set()
            for source, target in items:
                source_i, target_i = resolve(source), resolve(target)
                if source_i == target_i:
                    # Две разные ссылки могут указывать на один узел, и видно
                    # это только здесь: require("X1", 0) — скрытая петля, если
                    # "X1" стоит нулевым в node_names.
                    raise ValueError(
                        f"A BK direction cannot be a self-loop: {source!r} -> {target!r} "
                        f"both resolve to node {source_i} ({names[source_i]!r})"
                    )
                result.add((source_i, target_i))
            return frozenset(result)

        allowed = resolve_all(self._directions_at_phase("allowed", phase))
        forbidden = resolve_all(self._directions_at_phase("forbidden", phase)) - allowed
        required = resolve_all(self._directions_at_phase("required", phase))

        contradiction = forbidden & required
        if contradiction:
            raise ValueError(f"BK directions are both forbidden and required: {sorted(contradiction)!r}")
        for source, target in required:
            if (target, source) in required:
                raise ValueError(f"Opposite directions are both required: {source}->{target} and {target}->{source}")
        cycle = _find_cycle(required)
        if cycle is not None:
            raise ValueError(
                f"Background knowledge contains directed cycle at phase {phase.value!r}: "
                f"{_format_cycle(cycle, names)}"
            )

        compiled = _CompiledKnowledge(forbidden, required, allowed)
        self._compiled_cache[cache_key] = compiled
        return compiled

    def required_pairs(
        self,
        node_names: Sequence[object],
        *,
        phase: BKPhase | str,
    ) -> frozenset[tuple[int, int]]:
        """Resolved required pairs active at ``phase`` (for search protection)."""

        return self._compile(node_names, phase).required

    def orientation_guard(
        self,
        node_names: Sequence[object],
        *,
        phase: BKPhase | str = BKPhase.BETWEEN_ORIENTATION,
        encoding: MatrixEncoding | str = MatrixEncoding.STANDARD,
    ) -> "BKOrientationGuard":
        """Guard, через который правила ориентации проводят каждую запись метки."""

        return BKOrientationGuard(self._compile(node_names, phase), encoding)


    _codes = staticmethod(endpoint_codes)

    def apply_matrix(
        self,
        matrix: np.ndarray,
        node_names: Sequence[object],
        *,
        phase: BKPhase | str = BKPhase.BETWEEN_ORIENTATION,
        checkpoint: str,
        encoding: MatrixEncoding | str,
        graph_kind: str,
        previous: np.ndarray | None = None,
    ) -> BKApplicationReport:
        """Apply active constraints to a CPDAG/DAG or PAG endpoint matrix.

        Matrix convention: ``matrix[x, y]`` is the endpoint mark at ``y``.
        A required edge removed during the current transition is restored from
        ``previous``. If it was already absent at the selected checkpoint, a
        conflict is reported instead of inventing a new adjacency. Required
        constraints assigned to ``PRE_SEARCH`` are additionally exposed through
        :meth:`required_pairs` so skeleton and Possible-D-SEP deletion can protect
        them throughout the search phase.
        """

        phase = BKPhase(phase)
        report = BKApplicationReport(phase, checkpoint, self.is_active(phase), self._run_id)
        if not report.active:
            # Nothing is scheduled here; recording the no-op would only pad history.
            return report
        self.history.append(report)

        if graph_kind not in {"cpdag", "dag", "pag"}:
            raise ValueError(f"Unsupported graph kind: {graph_kind!r}")
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("BK expects a square endpoint matrix")
        if len(node_names) != matrix.shape[0]:
            raise ValueError("node_names length must match the endpoint matrix")

        compiled = self._compile(node_names, phase)
        null, tail, arrow, _circle = self._codes(encoding)
        names = [self._name_of(node) for node in node_names]

        # Фаза применяется транзакционно: сначала целиком к копии, потом одна
        # проверка на цикл, и только потом фиксация. Пооперационная проверка
        # делала результат зависимым от порядка ограничений — первые успевали
        # примениться, а последнее, замыкающее цикл, отклонялось.
        snapshot = matrix.copy()
        candidate = matrix.copy()

        def adjacent(source: int, target: int, graph: np.ndarray | None = None) -> bool:
            graph = candidate if graph is None else graph
            return graph[source, target] != null or graph[target, source] != null

        def emit(action: str, source: int, target: int, detail: str) -> None:
            report.events.append(BKEvent(phase, checkpoint, action, source, target, detail))

        def set_pair(source: int, target: int, at_source: int, at_target: int, detail: str) -> None:
            old_pair = (int(candidate[target, source]), int(candidate[source, target]))
            new_pair = (int(at_source), int(at_target))
            if old_pair != new_pair:
                candidate[target, source] = at_source
                candidate[source, target] = at_target
                emit("apply", source, target, detail)

        # Required adjacency is protected before endpoint constraints are applied.
        for source, target in sorted(compiled.required):
            if adjacent(source, target):
                continue
            if previous is not None and adjacent(source, target, previous):
                candidate[source, target] = previous[source, target]
                candidate[target, source] = previous[target, source]
                emit("restore", source, target, "required edge deletion was blocked")
            else:
                protected_earlier = (source, target) in self._compile(
                    node_names, BKPhase.PRE_SEARCH
                ).required
                if protected_earlier and phase is not BKPhase.POST_ORIENTATION:
                    detail = (
                        f"required edge was protected at {BKPhase.PRE_SEARCH.value!r} "
                        f"but is absent at phase {phase.value!r} — broken invariant: "
                        "the search deleted an edge it was told to keep"
                    )
                    if self.strict_required_adjacency:
                        raise ValueError(
                            f"{names[source]} -> {names[target]}: {detail}"
                        )
                else:
                    detail = f"required edge was absent before phase {phase.value!r}; "                              "BK never invents an adjacency that the search did not find"
                emit("conflict", source, target, detail)

        processed: set[frozenset[int]] = set()
        for source, target in sorted(compiled.forbidden):
            pair = frozenset((source, target))
            if pair in processed or not adjacent(source, target):
                continue
            processed.add(pair)
            reverse_forbidden = (target, source) in compiled.forbidden

            if reverse_forbidden:
                required_on_pair = (source, target) in compiled.required or (target, source) in compiled.required
                if required_on_pair:
                    emit("conflict", source, target, "both directions are forbidden on a required edge")
                elif graph_kind in {"cpdag", "dag"}:
                    set_pair(source, target, null, null, "both causal directions are forbidden; edge removed")
                else:
                    set_pair(source, target, arrow, arrow, "both causal directions are forbidden; retained as bidirected")
                continue

            # For CPDAG/DAG, a single prohibition resolves an existing adjacency
            # in the opposite direction.  In PAG, an arrowhead at the forbidden
            # source excludes source as an ancestor while preserving the other mark.
            if graph_kind in {"cpdag", "dag"}:
                # Проверку на цикл делает общая транзакционная проверка ниже.
                set_pair(source, target, arrow, tail, "forbidden direction oriented oppositely")
            else:
                at_target = int(candidate[source, target])
                if at_target == null:
                    at_target = tail
                set_pair(source, target, arrow, at_target, "forbidden causal direction blocked at source endpoint")

        # Required has the strongest dynamic effect: preserve adjacency and force
        # the exact tail/arrow endpoint pair. Allowed never reaches this section.
        for source, target in sorted(compiled.required):
            if not adjacent(source, target):
                continue
            set_pair(source, target, tail, arrow, "required direction enforced")

        # --- транзакция: коммит или полный откат ---
        cycle = find_directed_cycle(candidate, tail, arrow)
        if cycle is None:
            matrix[...] = candidate
            report.changed = not np.array_equal(snapshot, candidate)
        else:
            # Ограничения сами по себе ацикличны — это проверено статически.
            # Различаем два случая, потому что виноваты разные стороны.
            pre_existing = find_directed_cycle(snapshot, tail, arrow)
            report.events = [event for event in report.events if event.action == "conflict"]
            report.changed = False
            if pre_existing is not None:
                # Цикл был в графе ДО применения BK: его замкнули правила
                # ориентации на рёбрах, которые BK не ограничивает. BK тут не
                # причина и не может это починить — но обязан сказать вслух,
                # что результат невалиден.
                emit(
                    "conflict",
                    int(pre_existing[0]),
                    int(pre_existing[1]),
                    "graph already contained the directed cycle "
                    f"{_format_cycle(pre_existing, names)} before background "
                    "knowledge was applied; the orientation rules produced it, "
                    "and BK was left unapplied",
                )
            else:
                emit(
                    "conflict",
                    int(cycle[0]),
                    int(cycle[1]),
                    "background knowledge was not applied: it would create the "
                    f"directed cycle {_format_cycle(cycle, names)}",
                )

        if report.conflicts:
            details = "; ".join(
                f"{names[event.source]}->{names[event.target]}: {event.detail}"
                for event in report.conflicts
            )
            warnings.warn(
                f"Background knowledge was not satisfied at checkpoint "
                f"{checkpoint!r} (phase {phase.value!r}): {details}",
                BKConflictWarning,
                stacklevel=2,
            )

        return report


def apply_bk_checkpoint(
    background_knowledge: BackgroundKnowledge | None,
    matrix: np.ndarray,
    node_names: Sequence[object],
    *,
    phase: BKPhase | str,
    checkpoint: str,
    encoding: MatrixEncoding | str,
    graph_kind: str,
    previous: np.ndarray | None = None,
) -> BKApplicationReport | None:
    """Small integration helper used by all algorithm families."""

    if background_knowledge is None:
        return None
    if not isinstance(background_knowledge, BackgroundKnowledge):
        raise TypeError("background_knowledge must be a BackgroundKnowledge instance")
    return background_knowledge.apply_matrix(
        matrix,
        node_names,
        phase=phase,
        checkpoint=checkpoint,
        encoding=encoding,
        graph_kind=graph_kind,
        previous=previous,
    )


__all__ = [
    "BKApplicationReport",
    "BKConflictWarning",
    "BKEvent",
    "BKPhase",
    "BackgroundKnowledge",
    "MatrixEncoding",
    "apply_bk_checkpoint",
    "endpoint_codes",
    "find_directed_cycle",
    "BKOrientationGuard",
    "orientation_guard_for",
    "bk_allows_edge",
]
