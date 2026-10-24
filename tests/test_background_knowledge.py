"""
Юнит-тесты фоновых знаний (background knowledge).

Перенесены из SCL-v0 (tests/test_background_knowledge.py) и перенацелены
на fci_algo_v2: прежняя реализация v1 (fci_algo.py) удалена вместе с этим
PR, осталась единственная — с исправленными правилами R1/R2/R5/R10.
"""

import warnings
from itertools import permutations

import numpy as np

from whydra.background_knowledge import (
    BKPhase,
    BackgroundKnowledge,
    MatrixEncoding,
)
from whydra.algorithms.standalone.fci_algo_v2 import (
    fci_stable_v2 as fci_stable,
    pdsep_v2 as pdsep,
    udag2pag_v2 as udag2pag,
)
from whydra.algorithms.graph_core import CausalGraph
from whydra.algorithms.standalone.pc_algo import PCAlgorithm, pc_stable
from whydra.algorithms.standalone.skeleton import SkeletonDiscovery


NAMES = ["X", "Y"]


def test_allowed_only_cancels_matching_forbidden_direction():
    pag = np.array([[0, 1], [1, 0]], dtype=int)
    bk = BackgroundKnowledge(forbidden=[("X", "Y")], allowed=[("X", "Y")])

    report = bk.apply_matrix(
        pag,
        NAMES,
        phase=BKPhase.BETWEEN_ORIENTATION,
        checkpoint="test",
        encoding=MatrixEncoding.PCALG,
        graph_kind="pag",
    )

    assert np.array_equal(pag, np.array([[0, 1], [1, 0]]))
    assert not report.changed
    assert not report.events


def test_forbidden_direction_adds_arrowhead_at_forbidden_pag_source():
    # X o-o Y -> X <-o Y: matrix[row, col] stores the mark at col.
    pag = np.array([[0, 1], [1, 0]], dtype=int)
    bk = BackgroundKnowledge(forbidden=[("X", "Y")])

    bk.apply_matrix(
        pag,
        NAMES,
        phase=BKPhase.BETWEEN_ORIENTATION,
        checkpoint="test",
        encoding=MatrixEncoding.PCALG,
        graph_kind="pag",
    )

    assert pag[1, 0] == 2  # arrowhead at X
    assert pag[0, 1] == 1  # circle at Y is preserved


def test_both_forbidden_directions_keep_pag_edge_as_bidirected():
    pag = np.array([[0, 1], [1, 0]], dtype=int)
    bk = BackgroundKnowledge(forbidden=[("X", "Y"), ("Y", "X")])

    bk.apply_matrix(
        pag,
        NAMES,
        phase=BKPhase.BETWEEN_ORIENTATION,
        checkpoint="test",
        encoding=MatrixEncoding.PCALG,
        graph_kind="pag",
    )

    assert np.array_equal(pag, np.array([[0, 2], [2, 0]]))


def test_both_forbidden_directions_remove_cpdag_edge():
    cpdag = np.array([[0, -1], [-1, 0]], dtype=int)
    bk = BackgroundKnowledge(forbidden=[("X", "Y"), ("Y", "X")])

    bk.apply_matrix(
        cpdag,
        NAMES,
        phase=BKPhase.BETWEEN_ORIENTATION,
        checkpoint="test",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="cpdag",
    )

    assert not np.any(cpdag)


def test_required_direction_restores_edge_deleted_during_rule_pass():
    previous = np.array([[0, 2], [1, 0]], dtype=int)
    pag = np.zeros((2, 2), dtype=int)
    bk = BackgroundKnowledge(required=[("X", "Y")])

    report = bk.apply_matrix(
        pag,
        NAMES,
        phase=BKPhase.BETWEEN_ORIENTATION,
        checkpoint="test",
        encoding=MatrixEncoding.PCALG,
        graph_kind="pag",
        previous=previous,
    )

    # Здесь матрица объявлена в кодировке pcalg явным параметром encoding,
    # поэтому и коды ожидаются pcalg: tail=3, arrow=2.
    assert pag[1, 0] == 3  # tail at X
    assert pag[0, 1] == 2  # arrowhead at Y
    assert any(event.action == "restore" for event in report.events)


def test_phase_hooks_are_inactive_until_enabled():
    graph = np.array([[0, -1], [-1, 0]], dtype=int)
    bk = BackgroundKnowledge(required=[("X", "Y")])

    report = bk.apply_matrix(
        graph,
        NAMES,
        phase=BKPhase.PRE_SEARCH,
        checkpoint="future_hook",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="cpdag",
    )

    assert not report.active
    assert np.array_equal(graph, np.array([[0, -1], [-1, 0]]))


def test_fci_kernel_applies_bk_after_colliders_and_rule_passes():
    skeleton = np.array([[0, 1], [1, 0]], dtype=int)
    bk = BackgroundKnowledge(required=[("X", "Y")])

    pag = udag2pag(
        skeleton,
        sepset={},
        p=2,
        rules=[False] * 10,
        background_knowledge=bk,
        node_names=NAMES,
    )

    assert pag[1, 0] == -1
    assert pag[0, 1] == 1
    active_checkpoints = [report.checkpoint for report in bk.history if report.active]
    assert "after_colliders" in active_checkpoints
    assert "after_orientation_rule_pass" in active_checkpoints


def test_pc_kernel_applies_required_direction_between_orientation_stages():
    graph = CausalGraph(2, NAMES)
    bk = BackgroundKnowledge(required=[("X", "Y")])

    PCAlgorithm(graph, background_knowledge=bk, show_progress=False).run()

    assert graph.G.graph[1, 0] == -1  # tail at X
    assert graph.G.graph[0, 1] == 1   # arrowhead at Y
    active_checkpoints = [report.checkpoint for report in bk.history if report.active]
    # BK применяется ДО ориентации коллайдеров: правила должны стартовать с
    # графа, который уже уважает ограничения.
    assert active_checkpoints[0] == "before_orientation"
    assert "after_colliders" in active_checkpoints
    assert "after_meek_pass" in active_checkpoints


def test_each_constraint_can_target_a_different_phase():
    names = ["X", "Y", "Z"]
    graph = np.array(
        [[0, -1, -1], [-1, 0, -1], [-1, -1, 0]], dtype=int
    )
    bk = BackgroundKnowledge(enabled_phases=())
    bk.forbid("X", "Y", phases=BKPhase.PRE_SEARCH)
    bk.require("Y", "Z", phases=BKPhase.POST_ORIENTATION)

    pre = bk.apply_matrix(
        graph,
        names,
        phase=BKPhase.PRE_SEARCH,
        checkpoint="before_skeleton",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="cpdag",
    )
    assert pre.active and pre.changed
    assert graph[1, 0] == 1 and graph[0, 1] == -1  # Y -> X
    assert graph[2, 1] == -1 and graph[1, 2] == -1  # Y - Z unchanged

    between = bk.apply_matrix(
        graph,
        names,
        phase=BKPhase.BETWEEN_ORIENTATION,
        checkpoint="after_colliders",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="cpdag",
    )
    assert not between.active
    assert not between.changed

    post = bk.apply_matrix(
        graph,
        names,
        phase=BKPhase.POST_ORIENTATION,
        checkpoint="after_orientation",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="cpdag",
    )
    assert post.active and post.changed
    assert graph[2, 1] == -1 and graph[1, 2] == 1  # Y -> Z


def test_constructor_accepts_phase_scoped_constraints():
    graph = np.array([[0, -1], [-1, 0]], dtype=int)
    bk = BackgroundKnowledge(
        forbidden=[("X", "Y", BKPhase.POST_ORIENTATION)],
        enabled_phases=(),
    )

    bk.apply_matrix(
        graph,
        NAMES,
        phase=BKPhase.PRE_SEARCH,
        checkpoint="before_skeleton",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="cpdag",
    )
    assert np.array_equal(graph, np.array([[0, -1], [-1, 0]]))

    bk.apply_matrix(
        graph,
        NAMES,
        phase=BKPhase.POST_ORIENTATION,
        checkpoint="after_orientation",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="cpdag",
    )
    assert graph[1, 0] == 1 and graph[0, 1] == -1  # Y -> X


def test_pre_search_required_edge_is_protected_from_skeleton_deletion():
    graph = CausalGraph(2, NAMES)
    graph.set_ind_test(lambda x, y, conditioning_set: 1.0)
    bk = BackgroundKnowledge(enabled_phases=()).require(
        "X", "Y", phases=BKPhase.PRE_SEARCH
    )
    bk.apply_matrix(
        graph.G.graph,
        NAMES,
        phase=BKPhase.PRE_SEARCH,
        checkpoint="before_skeleton",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="cpdag",
    )

    SkeletonDiscovery(
        graph,
        alpha=0.05,
        show_progress=False,
        protected_edges=bk.required_pairs(NAMES, phase=BKPhase.PRE_SEARCH),
    ).run()

    assert graph.G.graph[1, 0] == -1
    assert graph.G.graph[0, 1] == 1


def test_fci_applies_post_only_constraint_only_at_final_checkpoint():
    skeleton = np.array([[0, 1], [1, 0]], dtype=int)
    bk = BackgroundKnowledge(enabled_phases=()).require(
        "X", "Y", phases=BKPhase.POST_ORIENTATION
    )

    pag = udag2pag(
        skeleton,
        sepset={},
        p=2,
        rules=[False] * 10,
        background_knowledge=bk,
        node_names=NAMES,
    )

    assert pag[1, 0] == -1 and pag[0, 1] == 1
    active_checkpoints = [report.checkpoint for report in bk.history if report.active]
    assert active_checkpoints == ["after_orientation"]


def test_pc_entrypoint_routes_different_constraints_to_pre_and_post():
    graph = CausalGraph(2, NAMES)
    graph.set_ind_test(lambda x, y, conditioning_set: 0.0)
    bk = BackgroundKnowledge(enabled_phases=())
    bk.require("X", "Y", phases=BKPhase.PRE_SEARCH)
    bk.forbid("Y", "X", phases=BKPhase.POST_ORIENTATION)

    pc_stable(
        graph,
        background_knowledge=bk,
        show_progress=False,
        n_jobs=1,
    )

    assert graph.G.graph[1, 0] == -1 and graph.G.graph[0, 1] == 1
    active_checkpoints = [report.checkpoint for report in bk.history if report.active]
    assert active_checkpoints == ["before_skeleton", "after_orientation"]


def test_fci_entrypoint_routes_different_constraints_to_pre_and_post():
    rng = np.random.default_rng(7)
    x = rng.normal(size=300)
    data = np.column_stack((x, x + rng.normal(scale=0.05, size=x.size)))
    graph = CausalGraph(2, NAMES)
    bk = BackgroundKnowledge(enabled_phases=())
    bk.require("X", "Y", phases=BKPhase.PRE_SEARCH)
    bk.forbid("Y", "X", phases=BKPhase.POST_ORIENTATION)

    pag = fci_stable(
        graph,
        data,
        background_knowledge=bk,
        n_jobs=1,
    )

    assert pag[1, 0] != 0 and pag[0, 1] != 0
    assert pag[0, 1] == 1  # arrowhead at Y blocks Y -> X
    active_checkpoints = [report.checkpoint for report in bk.history if report.active]
    assert active_checkpoints == ["before_skeleton", "after_orientation"]


def test_pre_search_required_edge_is_also_protected_in_fci_pdsep():
    skeleton = np.array(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int
    )
    sepset = {pair: set() for pair in permutations(range(3), 2)}

    result = pdsep(
        {"sk": skeleton},
        None,
        lambda suffstat, x, y, conditioning_set: 1.0,
        3,
        sepset,
        0.05,
        np.full((3, 3), -np.inf),
        unfVect=set(),
        protected_edges={(0, 1)},
    )

    assert result["G"][0, 1] != 0 and result["G"][1, 0] != 0


def test_allowed_cancels_forbidden_only_in_the_same_phase():
    bk = BackgroundKnowledge(
        forbidden=[(
            "X",
            "Y",
            [BKPhase.BETWEEN_ORIENTATION, BKPhase.POST_ORIENTATION],
        )],
        allowed=[("X", "Y", BKPhase.POST_ORIENTATION)],
        enabled_phases=(),
    )
    between_graph = np.array([[0, -1], [-1, 0]], dtype=int)
    post_graph = between_graph.copy()

    bk.apply_matrix(
        between_graph,
        NAMES,
        phase=BKPhase.BETWEEN_ORIENTATION,
        checkpoint="between",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="cpdag",
    )
    bk.apply_matrix(
        post_graph,
        NAMES,
        phase=BKPhase.POST_ORIENTATION,
        checkpoint="post",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="cpdag",
    )

    assert between_graph[1, 0] == 1 and between_graph[0, 1] == -1
    assert np.array_equal(post_graph, np.array([[0, -1], [-1, 0]]))


def test_opposite_phase_constraints_do_not_form_a_static_conflict():
    graph = np.array([[0, -1], [-1, 0]], dtype=int)
    bk = BackgroundKnowledge(
        forbidden=[("X", "Y", BKPhase.PRE_SEARCH)],
        required=[("X", "Y", BKPhase.POST_ORIENTATION)],
        enabled_phases=(),
    )

    bk.apply_matrix(
        graph,
        NAMES,
        phase=BKPhase.PRE_SEARCH,
        checkpoint="pre",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="cpdag",
    )
    assert graph[1, 0] == 1 and graph[0, 1] == -1  # Y -> X at pre

    bk.apply_matrix(
        graph,
        NAMES,
        phase=BKPhase.POST_ORIENTATION,
        checkpoint="post",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="cpdag",
    )
    assert graph[1, 0] == -1 and graph[0, 1] == 1  # X -> Y at post


def test_post_required_missing_edge_reports_conflict_without_inventing_adjacency():
    graph = np.zeros((2, 2), dtype=int)
    bk = BackgroundKnowledge(enabled_phases=()).require(
        "X", "Y", phases=BKPhase.POST_ORIENTATION
    )

    report = bk.apply_matrix(
        graph,
        NAMES,
        phase=BKPhase.POST_ORIENTATION,
        checkpoint="after_orientation",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="cpdag",
    )

    assert not np.any(graph)
    assert len(report.conflicts) == 1
    assert "post_orientation" in report.conflicts[0].detail


def _apply(bk, checkpoint="cp"):
    """Один активный чекпоинт — ровно один отчёт в history.

    Граф пустой, поэтому required-ребро ожидаемо конфликтует; для этих тестов
    важен только сам факт записи отчёта, так что предупреждение глушится.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bk.apply_matrix(
            np.zeros((2, 2), dtype=int),
            NAMES,
            phase=BKPhase.BETWEEN_ORIENTATION,
            checkpoint=checkpoint,
            encoding=MatrixEncoding.STANDARD,
            graph_kind="cpdag",
        )


def test_start_run_keeps_compiled_cache():
    """Кеш переживает смену прогона: ключ содержит имена узлов и фазу."""
    bk = BackgroundKnowledge(required=[("X", "Y")])
    _apply(bk)
    assert bk._compiled_cache

    bk.start_run()

    assert bk._compiled_cache, "start_run не должен ронять кеш скомпилированных ограничений"


def test_reset_cache_drops_compiled_entries():
    bk = BackgroundKnowledge(required=[("X", "Y")])
    _apply(bk)
    assert bk._compiled_cache

    assert bk.reset_cache() is bk
    assert not bk._compiled_cache


def test_run_scope_restores_previous_run_id():
    bk = BackgroundKnowledge(required=[("X", "Y")])

    _apply(bk, "outer_before")
    with bk.run_scope("inner"):
        _apply(bk, "inner")
    _apply(bk, "outer_after")

    assert [r.checkpoint for r in bk.reports_for_run("inner")] == ["inner"]
    assert [r.checkpoint for r in bk.reports_for_run(0)] == ["outer_before", "outer_after"]


def test_run_scope_restores_run_id_after_exception():
    bk = BackgroundKnowledge(required=[("X", "Y")])

    try:
        with bk.run_scope("boom"):
            raise RuntimeError("algorithm blew up")
    except RuntimeError:
        pass

    _apply(bk, "after")
    assert [r.checkpoint for r in bk.reports_for_run(0)] == ["after"]


def test_history_is_bounded_ring_buffer():
    """Переиспользуемый объект не копит отчёты бесконечно."""
    bk = BackgroundKnowledge(required=[("X", "Y")], history_limit=3)

    for i in range(10):
        _apply(bk, f"cp{i}")

    assert len(bk.history) == 3
    assert [r.checkpoint for r in bk.history] == ["cp7", "cp8", "cp9"]
