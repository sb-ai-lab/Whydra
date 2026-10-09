import numpy as np
from itertools import permutations

from ..graph_core import CausalGraph, Edge, Endpoint, GeneralGraph
from ...background_knowledge import (
    BKOrientationGuard,
    BKPhase,
    MatrixEncoding,
    apply_bk_checkpoint,
)
from .profiler import profiler

from .skeleton import SkeletonDiscovery

class PCAlgorithm:
    def __init__(
        self,
        cg: CausalGraph,
        alpha: float = 0.05,
        stable: bool = True,
        uc_rule: int = 0,
        uc_priority: int = 2,
        background_knowledge=None,
        verbose: bool = False,
        show_progress: bool = True,
        n_jobs: int = -1,
        **kwargs
    ):
        self.cg = cg
        self.alpha = alpha
        self.stable = stable
        self.uc_rule = uc_rule
        self.uc_priority = uc_priority
        self.background_knowledge = background_knowledge
        self.verbose = verbose
        self.show_progress = show_progress
        self.n_jobs = n_jobs
        self.extra_kwargs = kwargs
        if self.verbose:
            print(f'n_jobs: {n_jobs}')

    def _guard(self) -> BKOrientationGuard:
        """Guard ориентации; без BK ничего не запрещает."""
        if self.background_knowledge is None:
            return BKOrientationGuard(None, MatrixEncoding.STANDARD)
        return self.background_knowledge.orientation_guard(
            [node.name for node in self.cg.nodes],
            phase=BKPhase.BETWEEN_ORIENTATION,
            encoding=MatrixEncoding.STANDARD,
        )

    def run(self) -> GeneralGraph:
        self.bk_guard = self._guard()
        with profiler.time_block("full_pc_algorithm", tags={"algo": "sequential"}):
            # BK применяется после скелета, но ДО правил ориентации: коллайдеры
            # и Meek должны работать на графе, который уже уважает ограничения,
            # а не исправляться задним числом.
            # SkeletonDiscovery(
            #     cg=self.cg,
            #     alpha=self.alpha,
            #     stable=self.stable,
            #     n_jobs=self.n_jobs,
            #     show_progress=self.show_progress,
            #     verbose=self.verbose
            # ).run()
            self._apply_bk("before_orientation")
            self.orient_colliders()
            self._apply_bk("after_colliders")
            self.apply_meek_rules()
            self._apply_bk("after_orientation", phase=BKPhase.POST_ORIENTATION)

        return self.convert_to_general_graph()

    def _apply_bk(self, checkpoint, *, previous=None, phase=BKPhase.BETWEEN_ORIENTATION):
        return apply_bk_checkpoint(
            self.background_knowledge,
            self.cg.G.graph,
            [node.name for node in self.cg.nodes],
            phase=phase,
            checkpoint=checkpoint,
            encoding=MatrixEncoding.STANDARD,
            graph_kind="cpdag",
            previous=previous,
        )

    def orient_colliders(self):
        """
        Stage 2: Orientation of V-structures (Unshielded Colliders).
        Priority 2: Prioritize Existing Colliders (Conservative).
        """
        triples = self.cg.find_unshielded_triples()

        for (x, y, z) in triples:
            sepsets = self.cg.sepset[x, z]
            if sepsets is None: continue

            is_collider = True
            for S in sepsets:
                if y in S:
                    is_collider = False
                    break

            if is_collider:
                # Priority 2: Do not create collider if it contradicts existing edges
                conflict_x = self.cg.is_fully_directed(y, x)
                conflict_z = self.cg.is_fully_directed(y, z)

                if not conflict_x and not conflict_z:
                    self._orient_edge(x, y)
                    self._orient_edge(z, y)

    def _bk_allows(self, source, target, at_source, at_target) -> bool:
        """Разрешает ли BK выставить эту пару меток. Без BK — всегда да."""
        guard = getattr(self, "bk_guard", None)
        if guard is None or not guard:
            return True
        m = self.cg.G.graph
        return (guard.allows_mark(m, target, source, at_source.value)
                and guard.allows_mark(m, source, target, at_target.value))

    def _orient_edge(self, source, target):
        if not self._bk_allows(source, target, Endpoint.TAIL, Endpoint.ARROW):
            return False
        edge = self.cg.G.get_edge(self.cg.nodes[source], self.cg.nodes[target])
        if edge:
            self.cg.G.remove_edge(edge)
            self.cg.G.add_edge(Edge(self.cg.nodes[source], self.cg.nodes[target], Endpoint.TAIL, Endpoint.ARROW))
            return True
        return False

    def _orient_edge_conflict_aware(self, source, target):
        """
        Tries to orient source -> target.
        Used in Meek rules.
        """
        if self.cg.is_fully_directed(source, target): return False
        if self.cg.is_bidirected(source, target): return False

        if self.cg.is_fully_directed(target, source):
            # Conflict -> Bi-directed
            if not self._bk_allows(source, target, Endpoint.ARROW, Endpoint.ARROW):
                return False
            edge = self.cg.G.get_edge(self.cg.nodes[source], self.cg.nodes[target])
            if edge:
                self.cg.G.remove_edge(edge)
                self.cg.G.add_edge(Edge(self.cg.nodes[source], self.cg.nodes[target], Endpoint.ARROW, Endpoint.ARROW))
                return True
            return False

        if self.cg.is_undirected(source, target):
            if not self._bk_allows(source, target, Endpoint.TAIL, Endpoint.ARROW):
                return False
            edge = self.cg.G.get_edge(self.cg.nodes[source], self.cg.nodes[target])
            if edge:
                self.cg.G.remove_edge(edge)
                self.cg.G.add_edge(Edge(self.cg.nodes[source], self.cg.nodes[target], Endpoint.TAIL, Endpoint.ARROW))
                return True
            return False
        return False

    def apply_meek_rules(self):
        """
        Stage 3: Meek rules.
        """
        loop = True
        while loop:
            before_pass = self.cg.G.graph.copy()

            # R1
            triples = self.cg.find_unshielded_triples()
            for (i, j, k) in triples:
                if self.cg.is_fully_directed(i, j) and self.cg.is_undirected(j, k):
                    self._orient_edge_conflict_aware(j, k)
                elif self.cg.is_fully_directed(k, j) and self.cg.is_undirected(j, i):
                    self._orient_edge_conflict_aware(j, i)

            # R2
            triangles = self.cg.find_triangles()
            for (i, j, k) in triangles:
                nodes_tri = [i, j, k]
                for a, b, c in permutations(nodes_tri, 3):
                    if self.cg.is_fully_directed(a, b) and self.cg.is_fully_directed(b, c) and self.cg.is_undirected(a, c):
                        self._orient_edge_conflict_aware(a, c)

            # R3
            kites = self.cg.find_kites()
            for (i, j, k, l) in kites:
                if self.cg.is_fully_directed(j, l) and self.cg.is_fully_directed(k, l) and self.cg.is_undirected(i, l):
                    if self.cg.is_undirected(i, j) and self.cg.is_undirected(i, k):
                        self._orient_edge_conflict_aware(i, l)

            self._apply_bk("after_meek_pass", previous=before_pass)
            loop = not np.array_equal(before_pass, self.cg.G.graph)

        return self.cg

    def convert_to_general_graph(self) -> GeneralGraph:
        """Return the graph already maintained by the local algorithm."""
        return self.cg.G


def pc_stable(
        cg: CausalGraph,
        alpha=0.05,
        stable: bool = True,
        uc_rule: int = 0,
        uc_priority: int = 2,
        background_knowledge=None,
        verbose: bool = False,
        show_progress: bool = True,
        n_jobs: int = 1,
        **kwargs
):
    node_names = [node.name for node in cg.nodes]
    apply_bk_checkpoint(
        background_knowledge,
        cg.G.graph,
        node_names,
        phase=BKPhase.PRE_SEARCH,
        checkpoint="before_skeleton",
        encoding=MatrixEncoding.STANDARD,
        graph_kind="cpdag",
    )
    protected_edges = ()
    if background_knowledge is not None:
        protected_edges = background_knowledge.required_pairs(
            node_names, phase=BKPhase.PRE_SEARCH
        )
    SkeletonDiscovery(
        cg=cg,
        alpha=alpha,
        stable=stable,
        n_jobs=n_jobs,
        show_progress=show_progress,
        verbose=verbose,
        protected_edges=protected_edges,
    ).run()

    cg_res = PCAlgorithm(
        cg=cg,
        alpha=alpha,
        stable=stable,
        uc_rule=uc_rule,
        uc_priority=uc_priority,
        background_knowledge=background_knowledge,
        verbose=verbose,
        show_progress=show_progress,
        n_jobs=n_jobs,
        **kwargs
    ).run()

    return cg_res
