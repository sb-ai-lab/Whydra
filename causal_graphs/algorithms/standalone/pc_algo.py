import numpy as np
import os
from itertools import combinations, permutations

from causallearn.graph.GraphNode import GraphNode as CLGraphNode
from causallearn.graph.Edge import Edge as CLEdge
from causallearn.graph.Endpoint import Endpoint as CLEndpoint
from causallearn.graph.GeneralGraph import GeneralGraph as CLGeneralGraph

from ..graph_core import CausalGraph, Edge, Endpoint
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
        print(f'n_jobs: {n_jobs}')

    def run(self) -> CLGeneralGraph:
        with profiler.time_block("full_pc_algorithm", tags={"algo": "sequential"}):
            # SkeletonDiscovery(
            #     cg=self.cg,
            #     alpha=self.alpha,
            #     stable=self.stable,
            #     n_jobs=self.n_jobs,
            #     show_progress=self.show_progress,
            #     verbose=self.verbose
            # ).run()
            self.orient_colliders()
            self.apply_meek_rules()

        return self.convert_to_causallearn_graph()

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

    def _orient_edge(self, source, target):
        edge = self.cg.G.get_edge(self.cg.nodes[source], self.cg.nodes[target])
        if edge:
            self.cg.G.remove_edge(edge)
            self.cg.G.add_edge(Edge(self.cg.nodes[source], self.cg.nodes[target], Endpoint.TAIL, Endpoint.ARROW))

    def _orient_edge_conflict_aware(self, source, target):
        """
        Tries to orient source -> target.
        Used in Meek rules.
        """
        if self.cg.is_fully_directed(source, target): return
        if self.cg.is_bidirected(source, target): return

        if self.cg.is_fully_directed(target, source):
            # Conflict -> Bi-directed
            edge = self.cg.G.get_edge(self.cg.nodes[source], self.cg.nodes[target])
            if edge:
                self.cg.G.remove_edge(edge)
                self.cg.G.add_edge(Edge(self.cg.nodes[source], self.cg.nodes[target], Endpoint.ARROW, Endpoint.ARROW))
            return

        if self.cg.is_undirected(source, target):
            edge = self.cg.G.get_edge(self.cg.nodes[source], self.cg.nodes[target])
            if edge:
                self.cg.G.remove_edge(edge)
                self.cg.G.add_edge(Edge(self.cg.nodes[source], self.cg.nodes[target], Endpoint.TAIL, Endpoint.ARROW))
            return

    def apply_meek_rules(self):
        """
        Stage 3: Meek rules.
        """
        loop = True
        while loop:
            loop = False

            # R1
            triples = self.cg.find_unshielded_triples()
            for (i, j, k) in triples:
                if self.cg.is_fully_directed(i, j) and self.cg.is_undirected(j, k):
                    self._orient_edge_conflict_aware(j, k)
                    loop = True
                elif self.cg.is_fully_directed(k, j) and self.cg.is_undirected(j, i):
                    self._orient_edge_conflict_aware(j, i)
                    loop = True

            # R2
            triangles = self.cg.find_triangles()
            for (i, j, k) in triangles:
                nodes_tri = [i, j, k]
                for a, b, c in permutations(nodes_tri, 3):
                    if self.cg.is_fully_directed(a, b) and self.cg.is_fully_directed(b, c) and self.cg.is_undirected(a, c):
                        self._orient_edge_conflict_aware(a, c)
                        loop = True

            # R3
            kites = self.cg.find_kites()
            for (i, j, k, l) in kites:
                if self.cg.is_fully_directed(j, l) and self.cg.is_fully_directed(k, l) and self.cg.is_undirected(i, l):
                    if self.cg.is_undirected(i, j) and self.cg.is_undirected(i, k):
                        self._orient_edge_conflict_aware(i, l)
                        loop = True

        return self.cg

    def convert_to_causallearn_graph(self) -> CLGeneralGraph:
        cl_nodes = [CLGraphNode(node.name) for node in self.cg.nodes]
        cl_graph = CLGeneralGraph(cl_nodes)

        for i in range(self.cg.G.num_vars):
            for j in range(i + 1, self.cg.G.num_vars):
                end_j_val = self.cg.G.graph[i, j]
                end_i_val = self.cg.G.graph[j, i]

                if end_j_val != 0 or end_i_val != 0:
                    def map_end(val):
                        if val == 1: return CLEndpoint.ARROW
                        if val == -1: return CLEndpoint.TAIL
                        if val == 2: return CLEndpoint.CIRCLE
                        return CLEndpoint.NULL

                    cl_edge = CLEdge(cl_nodes[i], cl_nodes[j], map_end(end_i_val), map_end(end_j_val))
                    cl_graph.add_edge(cl_edge)

        return cl_graph


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
    os.makedirs("results", exist_ok=True)
    ini_file: str = 'results/cg_matrix_ini.csv'
    np.savetxt(ini_file, cg.G.graph, delimiter=',')

    SkeletonDiscovery(
        cg=cg,
        alpha=alpha,
        stable=stable,
        n_jobs=n_jobs,
        show_progress=show_progress,
        verbose=verbose
    ).run()
    
    skeleton_file: str = 'results/cg_matrix_skeleton.csv'
    np.savetxt(skeleton_file, cg.G.graph, delimiter=',')

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

    algo_file: str = 'results/cg_matrix_algo.csv'
    np.savetxt(algo_file, cg.G.graph, delimiter=',')

    return cg_res
