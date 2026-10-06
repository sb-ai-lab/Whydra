import numpy as np
from itertools import combinations, chain
from typing import Iterable, List, Sequence, Set, Tuple

from causallearn.graph.GeneralGraph import GeneralGraph as CLGeneralGraph

from ..graph_core import CausalGraph, CIT, Endpoint
from .profiler import profiler
from .standalone_pc_stable import (
    orient_colliders,
    apply_meek_rules,
    convert_to_causallearn_graph,
    _append_sepset,
)


def _unique_element_iterator(iterable: Iterable[Sequence[int]]) -> Iterable[Tuple[int, ...]]:
    """
    Generator that yields tuples only once (order-insensitive) while preserving the original order.
    """
    seen: Set[Tuple[int, ...]] = set()
    for item in iterable:
        key = tuple(sorted(item))
        if key in seen:
            continue
        seen.add(key)
        yield tuple(item)


class _CIOracle:
    def __init__(self, tester: CIT, alpha: float):
        self.tester = tester
        self.alpha = alpha

    def cond_indep(self, x: int, y: int, cond_set: Sequence[int]) -> bool:
        p_val = self.tester(x, y, cond_set)
        return p_val > self.alpha


class RAIStableLearner:
    """
    Minimal standalone adaptation of IntelLabs' RAI algorithm for the local pipeline.
    The implementation follows the recursive autonomy identification idea while
    reusing the local PC-style primitives (CIT, CausalGraph, and Meek orientation rules).
    """

    def __init__(
        self,
        data: np.ndarray,
        alpha: float,
        indep_test: str,
        node_names: Sequence[str],
    ):
        self.data = data
        self.alpha = alpha
        self.indep_test = indep_test
        self.node_names = list(node_names)

        self.cg = CausalGraph(data.shape[1], self.node_names)
        self.ci_test = CIT(data, method=indep_test)
        self.oracle = _CIOracle(self.ci_test, alpha)
        self.n_vars = data.shape[1]

    # --- Utility getters -------------------------------------------------
    def _parents(self, node: int) -> Set[int]:
        incoming = np.where(self.cg.G.graph[:, node] == Endpoint.ARROW.value)[0]
        parents = set()
        for cand in incoming:
            if self.cg.G.graph[node, cand] == Endpoint.TAIL.value:
                parents.add(cand)
        return parents

    def _children(self, node: int) -> Set[int]:
        outgoing = np.where(self.cg.G.graph[node, :] == Endpoint.ARROW.value)[0]
        children = set()
        for cand in outgoing:
            if self.cg.G.graph[cand, node] == Endpoint.TAIL.value:
                children.add(cand)
        return children

    def _undirected_neighbors(self, node: int) -> Set[int]:
        mask = (
            (self.cg.G.graph[node, :] == Endpoint.TAIL.value)
            & (self.cg.G.graph[:, node] == Endpoint.TAIL.value)
        )
        return set(np.where(mask)[0])

    def _adjacent_nodes(self, node: int) -> Set[int]:
        mask = (self.cg.G.graph[node, :] != 0) | (self.cg.G.graph[:, node] != 0)
        return set(np.where(mask)[0])

    def _is_connected(self, i: int, j: int) -> bool:
        return bool(self.cg.G.graph[i, j] != 0 or self.cg.G.graph[j, i] != 0)

    def _fan_in(self, node: int) -> int:
        return len(self._parents(node) | self._undirected_neighbors(node))

    def _delete_edge(self, i: int, j: int) -> None:
        edge = self.cg.G.get_edge(self.cg.nodes[i], self.cg.nodes[j])
        if edge:
            self.cg.G.remove_edge(edge)

    # --- Core steps ------------------------------------------------------
    def learn_structure(self) -> CLGeneralGraph:
        en_nodes = set(range(self.n_vars))
        ex_nodes: Set[int] = set()

        with profiler.time_block("standalone_rai_total", tags={"algo": "sequential"}):
            self._learn_recursively(en_nodes, ex_nodes, order=0)
            self._maximally_orient_edges()

        return convert_to_causallearn_graph(self.cg)

    def _learn_recursively(self, en_nodes: Set[int], ex_nodes: Set[int], order: int) -> None:
        if not en_nodes:
            return

        if self._exit_cond(en_nodes, order):
            return

        self._refine_and_orient(en_nodes, ex_nodes, order)

        d_nodes, list_of_ancestors_sets, a_nodes = self._split_ancestors_descendant(en_nodes)

        for ancestor_set in list_of_ancestors_sets:
            self._learn_recursively(ancestor_set, ex_nodes, order + 1)

        self._learn_recursively(d_nodes, a_nodes | ex_nodes, order + 1)

    def _exit_cond(self, en_nodes: Set[int], order: int) -> bool:
        for node in en_nodes:
            if self._fan_in(node) > order:
                return False
        return True

    def _refine_and_orient(self, en_nodes: Set[int], ex_nodes: Set[int], order: int) -> None:
        self._refine_exogenous_effect(en_nodes, ex_nodes, order)
        self._maximally_orient_edges()
        self._refine_endogenous(en_nodes, order)
        self._maximally_orient_edges()

    def _refine_exogenous_effect(self, en_nodes: Set[int], ex_nodes: Set[int], order: int) -> None:
        if order < 0:
            return

        for node in en_nodes:
            for ex in ex_nodes:
                if not self._is_connected(ex, node):
                    continue

                pot_parents_node = (self._parents(node) | self._undirected_neighbors(node)) - {ex}
                pot_parents_ex = (self._parents(ex) | self._undirected_neighbors(ex)) - {node}

                cond_sets_node = combinations(pot_parents_node, order)
                cond_sets_ex = combinations(pot_parents_ex, order)
                cond_sets = _unique_element_iterator(chain(cond_sets_node, cond_sets_ex))

                for cset in cond_sets:
                    if self.oracle.cond_indep(ex, node, cset):
                        self._delete_edge(ex, node)
                        _append_sepset(self.cg, ex, node, cset)
                        break

    def _refine_endogenous(self, en_nodes: Set[int], order: int) -> None:
        if len(en_nodes) < 2 or order < 0:
            return

        for node_i, node_j in combinations(en_nodes, 2):
            if not self._is_connected(node_i, node_j):
                continue

            pot_parents_i = (self._parents(node_i) | self._undirected_neighbors(node_i)) - {node_j}
            pot_parents_j = (self._parents(node_j) | self._undirected_neighbors(node_j)) - {node_i}

            cond_sets_i = combinations(pot_parents_i, order)
            cond_sets_j = combinations(pot_parents_j, order)
            cond_sets = _unique_element_iterator(chain(cond_sets_i, cond_sets_j))

            for cset in cond_sets:
                if self.oracle.cond_indep(node_i, node_j, cset):
                    self._delete_edge(node_i, node_j)
                    _append_sepset(self.cg, node_i, node_j, cset)
                    break

    def _split_ancestors_descendant(self, en_nodes: Set[int]) -> Tuple[Set[int], List[Set[int]], Set[int]]:
        if not en_nodes:
            return set(), [], set()

        d_nodes = self._get_lowest_topological_set(en_nodes)
        if not d_nodes:
            d_nodes = set(en_nodes)

        a_nodes = en_nodes - d_nodes
        ancestor_sets = self._get_unconnected_subgraphs(a_nodes)
        return d_nodes, ancestor_sets, a_nodes

    def _get_lowest_topological_set(self, en_nodes: Set[int]) -> Set[int]:
        lowest = set()
        for node in en_nodes:
            children = self._children(node) & en_nodes
            if not children:
                lowest.add(node)
        return lowest

    def _get_unconnected_subgraphs(self, nodes: Set[int]) -> List[Set[int]]:
        if not nodes:
            return []

        visited: Set[int] = set()
        components: List[Set[int]] = []
        for node in nodes:
            if node in visited:
                continue
            stack = [node]
            comp = set()
            while stack:
                current = stack.pop()
                if current in visited or current not in nodes:
                    continue
                visited.add(current)
                comp.add(current)
                for neighbor in self._adjacent_nodes(current):
                    if neighbor in nodes and neighbor not in visited:
                        stack.append(neighbor)
            if comp:
                components.append(comp)
        return components

    def _maximally_orient_edges(self) -> None:
        orient_colliders(self.cg, priority=2)
        apply_meek_rules(self.cg)


def rai_optimized(
    data: np.ndarray,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    node_names: List[str] | None = None,
    **kwargs,
):
    """
    Standalone RAI-Stable algorithm interface aligned with other local standalone algorithms.
    """
    if node_names is None:
        node_names = [f"X{i + 1}" for i in range(data.shape[1])]

    learner = RAIStableLearner(
        data=data,
        alpha=alpha,
        indep_test=indep_test,
        node_names=node_names,
    )
    final_graph = learner.learn_structure()

    class ResultWrapper:
        def __init__(self, g):
            self.G = g

    return ResultWrapper(final_graph)

