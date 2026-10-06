import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint

from .base import CausalAlgorithm
from .graph_core import CausalGraph, CIT
from .standalone import pc_algo
from .standalone import fci_algo
from .standalone import rai_algo
from .standalone import standalone_raioptimized


class StandalonePCStable(CausalAlgorithm):
    def __init__(self, alpha=0.05, indep_test='fisherz', n_jobs=4, **kwargs):
        self.alpha = alpha
        self.indep_test = indep_test
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        node_names = [f"X{i + 1}" for i in range(data.shape[1])]
        cg = CausalGraph(data.shape[1], node_names)
        cg.set_ind_test(CIT(data, method=self.indep_test))

        params = dict(self.kwargs)
        params.update(kwargs)
        result = pc_algo.pc_stable(
            cg,
            alpha=self.alpha,
            n_jobs=self.n_jobs,
            **params
        )
        return self._ensure_causallearn_graph(result)

    def _ensure_causallearn_graph(self, result):
        if hasattr(result, 'G') and isinstance(result.G, GeneralGraph):
            return result.G
        return result


class StandaloneFCIStable(CausalAlgorithm):
    def __init__(self, alpha=0.05, indep_test='fisherz', n_jobs=4, verbose=False, **kwargs):
        self.alpha = alpha
        self.indep_test = indep_test
        self.n_jobs = n_jobs
        self.verbose = verbose

    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        n_nodes = data.shape[1]

        node_names = [f"X{i + 1}" for i in range(data.shape[1])]
        cg = CausalGraph(data.shape[1], node_names)
        cg.set_ind_test(CIT(data, method=self.indep_test))

        pag_matrix = fci_algo.fci_stable(
            cg,
            data,
            alpha=self.alpha,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )

        return self._matrix_to_graph(pag_matrix, n_nodes)

    def _matrix_to_graph(self, matrix, n_nodes):
        node_names = [f"X{i + 1}" for i in range(n_nodes)]
        nodes = [GraphNode(name) for name in node_names]
        g = GeneralGraph(nodes)

        endpoint_map = {
            0: Endpoint.NULL, 1: Endpoint.TAIL,
            2: Endpoint.ARROW, 3: Endpoint.TAIL
        }

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                mark_j = int(matrix[i, j])
                mark_i = int(matrix[j, i])
                if mark_j != 0 or mark_i != 0:
                    edge = Edge(nodes[i], nodes[j], endpoint_map[mark_i], endpoint_map[mark_j])
                    g.add_edge(edge)
        return g

class StandaloneRAIOptimized(CausalAlgorithm):
    def __init__(self, alpha=0.05, indep_test='fisherz', **kwargs):
        self.alpha = alpha
        self.indep_test = indep_test
        self.kwargs = kwargs

    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        node_names = [f"X{i + 1}" for i in range(data.shape[1])]
        cg = standalone_raioptimized.rai_optimized(
            data,
            alpha=self.alpha,
            indep_test=self.indep_test,
            node_names=node_names,
            **self.kwargs
        )
        if hasattr(cg, 'G') and isinstance(cg.G, GeneralGraph):
            return cg.G
        return cg

class StandaloneRAIStable(CausalAlgorithm):
    def __init__(self, alpha=0.05, indep_test='fisherz', n_jobs=4, **kwargs):
        self.alpha = alpha
        self.indep_test = indep_test
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.verbose: bool = False,

    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        node_names = [f"X{i + 1}" for i in range(data.shape[1])]
        cg = rai_algo.rai_stable(
            data,
            alpha=self.alpha,
            indep_test=self.indep_test,
            node_names=node_names,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **self.kwargs
        )
        if hasattr(cg, 'G') and isinstance(cg.G, GeneralGraph):
            return cg.G
        return cg
