from __future__ import annotations
import numpy as np
from itertools import combinations, chain
from typing import Iterable, List, Sequence, Set, Tuple, Dict, Optional

from tqdm.auto import tqdm
from joblib import Parallel, delayed

from ..graph_core import CausalGraph, CIT, Endpoint
from .profiler import profiler
from .standalone_pc_stable import (
    orient_colliders,
    apply_meek_rules,
    convert_to_general_graph,
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


# ------------------------- Graph Helper Functions --------------------------

def _get_parents(cg: CausalGraph, node: int) -> Set[int]:
    """Get parents of a node (incoming arrows with outgoing tails)."""
    incoming = np.where(cg.G.graph[:, node] == Endpoint.ARROW.value)[0]
    parents = set()
    for cand in incoming:
        if cg.G.graph[node, cand] == Endpoint.TAIL.value:
            parents.add(cand)
    return parents


def _get_undirected_neighbors(cg: CausalGraph, node: int) -> Set[int]:
    """Get undirected neighbors of a node (both sides are tails)."""
    mask = (
        (cg.G.graph[node, :] == Endpoint.TAIL.value)
        & (cg.G.graph[:, node] == Endpoint.TAIL.value)
    )
    return set(np.where(mask)[0])


def _is_connected(cg: CausalGraph, i: int, j: int) -> bool:
    """Check if two nodes are connected by any edge."""
    return bool(cg.G.graph[i, j] != 0 or cg.G.graph[j, i] != 0)


# ------------------------- Node-level Workers --------------------------

def _rai_exogenous_worker(
    node: int,
    en_nodes: List[int],
    ex_nodes: List[int],
    cg: CausalGraph,
    order: int,
    oracle: _CIOracle
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, Tuple[int, ...]]]]:
    """
    Worker for exogenous phase: tests independence between exogenous and endogenous nodes.
    """
    edge_removals = []
    sepsets_local = []

    # Pre-fetch potential parents for node since it's used against all ex
    pot_parents_node = (_get_parents(cg, node) | _get_undirected_neighbors(cg, node))

    # Optimization: Only check ex nodes that are actually connected to node
    # This allows merging tasks with disjoint ex sets efficiently
    relevant_ex = [ex for ex in ex_nodes if _is_connected(cg, ex, node)]

    for ex in relevant_ex:
        pot_parents_node_ex = pot_parents_node - {ex}
        # For ex, we also look at its parents in the full graph (cg)
        pot_parents_ex = (_get_parents(cg, ex) | _get_undirected_neighbors(cg, ex)) - {node}

        cond_sets_node = combinations(pot_parents_node_ex, order)
        cond_sets_ex = combinations(pot_parents_ex, order)

        # In exogenous phase, we check both sides (node and ex) because only 'node' has a worker
        cond_sets = _unique_element_iterator(chain(cond_sets_node, cond_sets_ex))

        for cset in cond_sets:
            if oracle.cond_indep(ex, node, cset):
                edge_removals.append((ex, node))
                sepsets_local.append((ex, node, cset))
                break

    return edge_removals, sepsets_local


def _rai_endogenous_worker(
    node_i: int,
    en_nodes: List[int],
    cg: CausalGraph,
    order: int,
    oracle: _CIOracle
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, Tuple[int, ...]]]]:
    """
    Worker for endogenous phase: tests independence between endogenous nodes.
    """
    edge_removals = []
    sepsets_local = []

    # Potential parents of i
    pot_parents_i = (_get_parents(cg, node_i) | _get_undirected_neighbors(cg, node_i))

    # Optimization: Only check en nodes that are actually connected to node_i
    # Note: we filter en_nodes to finding relevant neighbors.
    # Since we check neighbors efficiently using graph, we iterate neighbors and check if they are in en_nodes.

    en_nodes_set = set(en_nodes)
    relevant_neighbors = [n for n in pot_parents_i if n in en_nodes_set]

    for node_j in relevant_neighbors:
        if node_j == node_i:
            continue

        # No need for _is_connected check as we derived it from pot_parents_i

        # In endogenous phase, we split the work.
        # Worker 'i' checks independence using neighbors of 'i'.
        # Worker 'j' will check using neighbors of 'j'.
        # This matches PC-Stable logic and covers all sets when results are merged.

        pot_parents_i_j = pot_parents_i - {node_j}
        cond_sets_i = combinations(pot_parents_i_j, order)

        for cset in cond_sets_i:
            if oracle.cond_indep(node_i, node_j, cset):
                edge_removals.append((node_i, node_j))
                sepsets_local.append((node_i, node_j, cset))
                break

    return edge_removals, sepsets_local


class RAIStableLearner:
    """
    Итеративная (без рекурсии) версия RAI-Stable адаптации с PC-like parallelization (node-level).
    """

    def __init__(
        self,
        data: np.ndarray,
        alpha: float,
        indep_test: str,
        node_names: Sequence[str],
        verbose: bool=False,
    ):
        self.data = data
        self.alpha = alpha
        self.indep_test = indep_test
        self.node_names = list(node_names)

        self.cg = CausalGraph(data.shape[1], self.node_names)
        self.ci_test = CIT(data, method=indep_test)
        self.oracle = _CIOracle(self.ci_test, alpha)
        self.n_vars = data.shape[1]
        self.verbose = verbose

    # --- Utility getters (wrappers for class methods usage if needed) ----------------
    def _parents(self, node: int) -> Set[int]:
        return _get_parents(self.cg, node)

    def _children(self, node: int) -> Set[int]:
        outgoing = np.where(self.cg.G.graph[node, :] == Endpoint.ARROW.value)[0]
        children = set()
        for cand in outgoing:
            if self.cg.G.graph[cand, node] == Endpoint.TAIL.value:
                children.add(cand)
        return children

    def _undirected_neighbors(self, node: int) -> Set[int]:
        return _get_undirected_neighbors(self.cg, node)

    def _adjacent_nodes(self, node: int) -> Set[int]:
        mask = (self.cg.G.graph[node, :] != 0) | (self.cg.G.graph[:, node] != 0)
        return set(np.where(mask)[0])

    def _is_connected(self, i: int, j: int) -> bool:
        return _is_connected(self.cg, i, j)

    def _fan_in(self, node: int) -> int:
        return len(self._parents(node) | self._undirected_neighbors(node))

    def _delete_edge(self, i: int, j: int) -> None:
        edge = self.cg.G.get_edge(self.cg.nodes[i], self.cg.nodes[j])
        if edge:
            self.cg.G.remove_edge(edge)

    # --- Core steps (iterative) -----------------------------------------
    def learn_structure(self, n_jobs: int = 1):
        """
        High-level entry.
        """
        en_nodes = set(range(self.n_vars))
        ex_nodes: Set[int] = set()

        if n_jobs < 0:
            import os
            n_jobs = os.cpu_count() or 1

        # Всегда используем Stable-логику (батчевую), даже для 1 ядра
        with profiler.time_block("standalone_rai_total", tags={"algo": "rai_stable"}):
            self._learn_iteratively(en_nodes, ex_nodes, 0, n_jobs=n_jobs)
            self._maximally_orient_edges()

        return convert_to_general_graph(self.cg)

    def _learn_iteratively_seq(self, en_nodes_initial: Set[int], ex_nodes_initial: Set[int]) -> None:

        stack: List[Tuple[Set[int], Set[int], int]] = [(set(en_nodes_initial), set(ex_nodes_initial), 0)]

        while stack:
            en_nodes, ex_nodes, order = stack.pop()

            if not en_nodes:
                continue

            if self._exit_cond(en_nodes, order):
                continue

            # эквивалент _refine_and_orient(en_nodes, ex_nodes, order)
            self._refine_exogenous_effect(en_nodes, ex_nodes, order)
            self._maximally_orient_edges()
            self._refine_endogenous(en_nodes, order)
            self._maximally_orient_edges()

            # split и пуш в стек в обратном порядке, чтобы сохранить порядок рекурсии
            d_nodes, a_nodes = self._split_ancestors_descendant(en_nodes)

            # Сначала пушим d_nodes, чтобы он обработался после a_nodes (как в рекурсии)
            if d_nodes:
                stack.append((set(d_nodes), set(a_nodes) | set(ex_nodes), order + 1))

            # Затем a_nodes
            # (они будут извлечены/обработаны до d_nodes, потому что добавлены позже)
            if a_nodes:
                stack.append((set(a_nodes), set(ex_nodes), order + 1))

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

    def _learn_iteratively(self, en_nodes_initial: Set[int], ex_nodes_initial: Set[int], start_order: int = 0,
                          n_jobs: int = 1) -> None:
        """
        RAI Loop efficiently processing depth by depth (order) but decomposing graph.
        Batched parallelization is applied ("Graph Alignment") to maximize utilization.
        """

        # Tasks grouped by order: order -> list of (en_nodes, ex_nodes)
        tasks_by_order: Dict[int, List[Tuple[Set[int], Set[int]]]] = {}
        tasks_by_order[start_order] = [(set(en_nodes_initial), set(ex_nodes_initial))]

        curr_order = start_order

        while tasks_by_order:
            if curr_order not in tasks_by_order or not tasks_by_order[curr_order]:
                curr_order += 1
                if curr_order > self.n_vars:
                    break
                continue

            raw_tasks = tasks_by_order[curr_order]
            del tasks_by_order[curr_order]

            # Filter active tasks based on exit condition
            active_tasks = []
            for en, ex in raw_tasks:
                if en and not self._exit_cond(en, curr_order):
                    active_tasks.append((en, ex))

            if not active_tasks:
                # Even if no active tasks for refinement, we might have had tasks that finished?
                # No, if finished, we stop.
                # But wait, checking exit_cond means "do we continue refinement?".
                # If yes, we refine AND split.
                # If no, we stop.
                curr_order += 1
                continue

            # 1. Parallel Exogenous Refinement for ALL tasks at this order
            self._batch_refine_exogenous_effect(active_tasks, curr_order, n_jobs)
            self._maximally_orient_edges()

            # 2. Parallel Endogenous Refinement for ALL tasks at this order
            self._batch_refine_endogenous(active_tasks, curr_order, n_jobs)
            self._maximally_orient_edges()

            # 3. Split and Prepare Next Order
            if curr_order + 1 not in tasks_by_order:
                tasks_by_order[curr_order + 1] = []

            for en_nodes, ex_nodes in active_tasks:
                d_nodes, a_nodes = self._split_ancestors_descendant(en_nodes)

                # Push Descendants (en=d_nodes, ex=a_nodes | ex_nodes)
                if d_nodes:
                     tasks_by_order[curr_order + 1].append((d_nodes, a_nodes | ex_nodes))

                # Push Ancestors (en=a_nodes, ex=ex_nodes)
                if a_nodes:
                     tasks_by_order[curr_order + 1].append((a_nodes, ex_nodes))

            curr_order += 1

    def _exit_cond(self, en_nodes: Set[int], order: int) -> bool:
        for node in en_nodes:
            if self._fan_in(node) > order:
                return False
        return True

    def _batch_refine_exogenous_effect(self, tasks: List[Tuple[Set[int], Set[int]]], order: int, n_jobs: int) -> None:
        if order < 0 or not tasks:
            return

        # Granularize tasks to edges for load balancing
        # Sort by node complexity (Fan-in)
        edge_jobs = []

        for en_nodes, ex_nodes in tasks:
            if not ex_nodes or not en_nodes:
                continue

            # We must iterate nodes to find connected ex_nodes
            # To avoid slow sequential graph checks, we can iterate:
            # For each en_node, get its neighbors, intersect with ex_nodes.

            for node_i in en_nodes:
                # Find connected ex nodes
                # Fast implementation: neighbors of i in CG
                all_neighbors = self._parents(node_i) | self._undirected_neighbors(node_i)
                # Intersect with ex_nodes for this specific task
                relevant_ex = all_neighbors.intersection(ex_nodes)

                if not relevant_ex:
                    continue

                cost = len(all_neighbors) # Heuristic for complexity

                for ex_node in relevant_ex:
                    # Create a definitive job for this pair
                    job = delayed(_rai_exogenous_worker)(node_i, [node_i], [ex_node], self.cg, order, self.oracle)
                    edge_jobs.append((cost, job))

        if not edge_jobs:
            return

        # Sort tasks by complexity descending (Longest Processing Time First) to prevent stragglers
        edge_jobs.sort(key=lambda x: x[0], reverse=True)
        jobs = [j[1] for j in edge_jobs]

        iterator = jobs
        if self.verbose:
            iterator = tqdm(jobs, desc=f"RAI Exog Edge-Batch (order={order})", leave=False)

        # Use efficient batch size to avoid too much overhead for tiny tasks
        # But for heavy tasks, we want them distributed. 'auto' is usually fine if sorted.
        results = Parallel(n_jobs=n_jobs)(iterator)

        # Synchronization
        edge_removals = []
        sepset_updates = []

        for rem, sep in results:
            edge_removals.extend(rem)
            sepset_updates.extend(sep)

        for (x, y) in edge_removals:
            self._delete_edge(x, y)

        for (x, y, S) in sepset_updates:
            _append_sepset(self.cg, x, y, S)

    def _batch_refine_endogenous(self, tasks: List[Tuple[Set[int], Set[int]]], order: int, n_jobs: int) -> None:
        if order < 0 or not tasks:
            return

        edge_jobs = []

        for en_nodes, _ in tasks:
            if len(en_nodes) < 2:
                continue

            for node_i in en_nodes:
                # Neighbors of i
                all_neighbors = self._parents(node_i) | self._undirected_neighbors(node_i)
                # Only check neighbors that are in the current en_nodes set
                relevant_neighbors = all_neighbors.intersection(en_nodes)

                if not relevant_neighbors:
                    continue

                cost = len(all_neighbors)

                for node_j in relevant_neighbors:
                     if node_i == node_j:
                         continue
                     # Create job for directed pair (i, j)
                     # Note: we pass en_nodes=[node_j] to the worker to restrict it to checking just this j
                     job = delayed(_rai_endogenous_worker)(node_i, [node_j], self.cg, order, self.oracle)
                     edge_jobs.append((cost, job))

        if not edge_jobs:
            return

        # Sort desc
        edge_jobs.sort(key=lambda x: x[0], reverse=True)
        jobs = [j[1] for j in edge_jobs]

        iterator = jobs
        if self.verbose:
            iterator = tqdm(jobs, desc=f"RAI Endo Edge-Batch (order={order})", leave=False)

        results = Parallel(n_jobs=n_jobs)(iterator)

        # Synchronization
        edge_removals = []
        sepset_updates = []

        for rem, sep in results:
            edge_removals.extend(rem)
            sepset_updates.extend(sep)

        for (x, y) in edge_removals:
            self._delete_edge(x, y)

        for (x, y, S) in sepset_updates:
            _append_sepset(self.cg, x, y, S)

    def _split_ancestors_descendant(self, en_nodes: Set[int]) -> Tuple[Set[int], Set[int]]:
        if not en_nodes:
            return set(), set()

        d_nodes = self._get_lowest_topological_set(en_nodes)
        if not d_nodes:
            d_nodes = set(en_nodes)

        a_nodes = en_nodes - d_nodes
        # We no longer split a_nodes into unconnected subgraphs to avoid task fragmentation
        return d_nodes, a_nodes

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


def rai_stable(
    data: np.ndarray,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    node_names: Optional[List[str]] = None,
    n_jobs: int = -1,
    verbose: bool = False,
    **kwargs,
):
    """
    Standalone RAI-Stable algorithm interface (parallelized on node-level).
    - n_jobs: number of jobs to run in parallel
    """
    if node_names is None:
        node_names = [f"X{i + 1}" for i in range(data.shape[1])]

    learner = RAIStableLearner(
        data=data,
        alpha=alpha,
        indep_test=indep_test,
        node_names=node_names,
        verbose=verbose
    )
    # Map n_procs from kwargs if present for backward compatibility
    if "n_procs" in kwargs:
        n_jobs = kwargs["n_procs"]

    if n_jobs>1 or n_jobs<0:
        final_graph = learner.learn_structure(n_jobs=n_jobs)
    else:
        final_graph = learner.learn_structure()

    class ResultWrapper:
        def __init__(self, g):
            self.G = g

    return ResultWrapper(final_graph)
