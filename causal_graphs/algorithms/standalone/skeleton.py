import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
from multiprocessing import cpu_count

from itertools import combinations, permutations
import math

from causallearn.graph.GraphNode import GraphNode as CLGraphNode
from causallearn.graph.Edge import Edge as CLEdge
from causallearn.graph.Endpoint import Endpoint as CLEndpoint
from causallearn.graph.GeneralGraph import GeneralGraph as CLGeneralGraph

from ..graph_core import CausalGraph
from .profiler import profiler

# ------------------------------
# Worker-функция (вынести НА УРОВЕНЬ МОДУЛЯ, не внутрь класса!)
# ------------------------------
def _skeleton_worker(x, cg, depth, alpha):
    """
    Worker function to process a single variable x.
    Checks for independence with neighbors.
    """
    edge_removals = []
    sepset_updates = []

    neigh_x = cg.neighbors(x)
    if len(neigh_x) < depth:
        return edge_removals, sepset_updates

    for y in neigh_x:
        neigh_x_noy = np.delete(neigh_x, np.where(neigh_x == y))

        for S in combinations(neigh_x_noy, depth):
            p_val = cg.ci_test(x, y, S)
            if p_val > alpha:
                edge_removals.append((x, y))
                sepset_updates.append((x, y, S))
                break

    return edge_removals, sepset_updates

def ci_worker(args):
    x, y, S, ci_func = args
    return x, y, S, ci_func(x, y, S)

def _skeleton_worker_fci(x, cg, G1, depth, alpha, pMax_row):
    """
    Worker function for FCI parallel processing.
    """
    edge_removals = []
    sepset_updates = []
    pmax_updates = []
    tests_performed = 0

    # Neighbors in current graph G
    curr_neighs = cg.neighbors(x)

    for y in curr_neighs:
        # Neighbors of x in G1 (stable reference)
        nbrs = np.where(G1[x, :] != 0)[0]
        nbrs = nbrs[nbrs != y]

        if len(nbrs) >= depth:
            current_pmax = pMax_row[y]

            for S in combinations(nbrs, depth):
                tests_performed += 1
                pval = cg.ci_test(x, y, S)

                # Update pMax if pval is larger
                if pval > current_pmax:
                    current_pmax = pval
                    pmax_updates.append((x, y, pval))

                if pval >= alpha:
                    edge_removals.append((x, y))
                    sepset_updates.append((x, y, S))
                    break

    return edge_removals, sepset_updates, pmax_updates, tests_performed

class SkeletonDiscovery:
    def __init__(
        self,
        cg: CausalGraph,
        alpha: float = 0.05,
        stable: bool = True,
        n_jobs: int = 1,
        show_progress: bool = True,
        verbose: bool = False
    ):
        self.cg = cg
        self.alpha = alpha
        self.stable = stable
        self.n_jobs = n_jobs
        self.show_progress = show_progress
        self.verbose = verbose

        self.no_of_var = len(self.cg.nodes)
        # We work directly on cg.G.graph
        self.G = self.cg.G.graph
        self.G1 = np.copy(self.G)

    def run(self) -> CLGeneralGraph:
        """
        Stage 1: Skeleton discovery.
        """
        depth = -1
        pbar = self._init_progress_bar(self.show_progress)

        with profiler.time_block("skeleton_discovery_total", tags={"algo": "sequential"}):
            while self._continue_search(depth):
                depth += 1
                if self.show_progress:
                    self._reset_progress_bar(pbar, depth)


                edge_removal = self._process_depth(
                    depth=depth,
                    pbar=pbar
                )

                if self.stable:
                    self._apply_edge_removals(edge_removal)

                if self.show_progress:
                    pbar.refresh()

        self._close_progress_bar(pbar)

        return self.convert_to_causallearn_graph()


    def run_fci(self):
        sepset = {}
        for i in permutations([i for i in range(len(self.cg.nodes))], 2):
            sepset[i] = set()

        pbar = self._init_progress_bar(self.show_progress)

        # Initialize pMax if it doesn't exist
        if not hasattr(self.cg, 'pMax'):
            self.cg.pMax = np.full((self.no_of_var, self.no_of_var), -np.inf)
            for i in range(self.no_of_var):
                self.cg.pMax[i, i] = 1

        depth = -1

        # while not done and depth <= self.m_max:
        while self._continue_search(depth):
            depth += 1
            if self.show_progress:
                self._reset_progress_bar(pbar, depth)

            edge_removal, sepset = self._process_depth_fci(
                depth=depth,
                pbar=pbar,
                sepset=sepset
            )

            # print(f'cg.G.graph skeleton {self.cg.G.graph}')
            # print(f'G skeleton {self.G}')
            # print(f'cg.sepset skeleton {sepset}')

            if self.stable:
                self._apply_edge_removals(edge_removal)

            self.G1 = np.copy(self.G)

            if self.show_progress:
                pbar.refresh()

        self._close_progress_bar(pbar)

        return {'sk': self.cg.G.graph, 'pMax': self.cg.pMax, 'sepset': sepset, "unfTriples": set(), "max_ord": depth - 1}

    def _process_depth_fci(
        self,
        depth: int,
        pbar,
        sepset,
    ):
        print(f'depth = {depth}')


        if self.n_jobs != 1:
            # print(f'Going to parallel analysis with n_jobs = {self.n_jobs}')
            return self._process_depth_parallel_fci(depth, sepset, pbar)

        edge_removal = []

        # Get all edges. G contains Endpoint values. != 0 means edge exists.
        # We only need one direction per pair, but skeleton logic iterates both directions
        # because neighbors depend on x.
        # However, for undirected graph, G[x,y] == G[y,x].
        # We iterate over all edges (x, y) where G[x,y] != 0
        # Snapshot for stable algorithm

        for x in range(self.no_of_var):

            if self.show_progress:
                pbar.update()

            neigh_x = self.cg.neighbors(x)
            if len(neigh_x) < depth:
                continue

            removals_for_x, sepset= self._process_node_at_depth_fci(
                x=x,
                neigh_x=neigh_x,
                depth=depth,
                sepset=sepset,
            )
            # print(removals_for_x)
            if removals_for_x:
                edge_removal.extend(removals_for_x)

        return edge_removal, sepset

    def _process_node_at_depth_fci(
        self,
        x: int,
        neigh_x: np.ndarray,
        depth: int,
        sepset,
    ):
        edge_removal = []
        for y in range(self.no_of_var):
            if x == y:
                continue
            if self.G[x, y] == 0:
                continue

            # Neighbors of x in G1
            # In undirected graph check != 0
            nbrs = np.where(self.G1[x, :] != 0)[0]
            nbrs = nbrs[nbrs != y]
            # nbrs = self.cg.neighbors(x)

            if len(nbrs) >= depth:
                if len(nbrs) > depth:
                    done = False

                for S in combinations(nbrs, depth):
                    pval = self.cg.ci_test(x, y, S)

                    if self.cg.pMax[x, y] < pval:
                        self.cg.pMax[x, y] = pval

                    if pval >= self.alpha:
                        # Remove edge (set to NULL=0)
                        # G[x, y] = 0
                        # G[y, x] = 0
                        if not self.stable:
                            self._remove_edge_immediately(x, y)
                        else:
                            self._register_edge_removal(edge_removal, x, y)

                        sepset[(x, y)] = S
                        sepset[(y, x)] = S

                        self._append_sepset(x, y, S)
                        break
        return edge_removal, sepset


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

    def _init_progress_bar(self, show_progress: bool):
        return tqdm(total=self.no_of_var) if show_progress else None

    def _reset_progress_bar(self, pbar, depth: int):
        pbar.reset()
        pbar.set_description(f'Depth={depth}, max_degree={self.cg.max_degree()}')

    def _close_progress_bar(self, pbar):
        if self.show_progress and pbar is not None:
            pbar.close()

    def _continue_search(self, depth: int) -> bool:
        """
        Condition of the outer while-loop:
        keep increasing depth while max_degree(cg) - 1 > depth.
        """
        return self.cg.max_degree() - 1 > depth

    def _process_depth(
        self,
        depth: int,
        pbar,
    ):
        """
        Process all variables at a given depth.
        Returns list of edges to remove (for stable mode).
        """
        if self.n_jobs != 1:
            # print(f'Going to parallel analysis with n_jobs = {self.n_jobs}')
            return self._process_depth_parallel(depth, pbar)

        edge_removal = []

        for x in range(self.no_of_var):
            if self.show_progress:
                pbar.update()

            neigh_x = self.cg.neighbors(x)
            if len(neigh_x) < depth:
                continue

            removals_for_x = self._process_node_at_depth(
                x=x,
                neigh_x=neigh_x,
                depth=depth,
            )
            edge_removal.extend(removals_for_x)

        return edge_removal

    def _process_depth_parallel_fci(self, depth, sepset, pbar):
        """
        Parallel version of _process_depth for FCI algo.
        """
        # Estimate total operations
        total_ops_estimate = 0
        for x in range(self.no_of_var):
            curr_neighs = self.cg.neighbors(x)
            for y in curr_neighs:
                 # Reconstruct nbrs logic from worker
                 nbrs = np.where(self.G[x, :] != 0)[0]
                 # print(f'x = {x}, y = {y}, nbrs: {nbrs}')
                 nbrs = nbrs[nbrs != y]
                 if len(nbrs) >= depth:
                     total_ops_estimate += math.comb(len(nbrs), depth)
        
        print(f"Depth {depth}: Estimated max CI tests: {total_ops_estimate}")

        print(f"Jobs in parallel: {self.n_jobs}")
        results_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_skeleton_worker_fci)(x, self.cg, self.G, depth, self.alpha, self.cg.pMax[x])
            for x in range(self.no_of_var)
        )

        if self.show_progress and pbar is not None:
            pbar.update(self.no_of_var)

        aggregated_edge_removals = []
        total_tests_actual = 0

        for edge_removals, sepset_updates, pmax_updates, tests_count in results_list:
            total_tests_actual += tests_count

            # Apply pMax updates
            for (x, y, pval) in pmax_updates:
                if self.cg.pMax[x, y] < pval:
                    self.cg.pMax[x, y] = pval

            # Apply edge removals
            for (x, y) in edge_removals:
                 if self.stable:
                     self._register_edge_removal(aggregated_edge_removals, x, y)
                 else:
                     self._remove_edge_immediately(x, y)

            # Apply sepset updates
            for (x, y, S) in sepset_updates:
                sepset[(x, y)] = S
                sepset[(y, x)] = S
                self._append_sepset(x, y, S)

        print(f"Depth {depth}: Actual CI tests performed: {total_tests_actual}")
        return aggregated_edge_removals, sepset

    def _process_depth_parallel(self, depth, pbar):
        """
        Parallel version of _process_depth.
        """
        effective_n_jobs = self.n_jobs if self.n_jobs > 0 else cpu_count()

        # Run parallel jobs for each variable x
        results_list = Parallel(n_jobs=effective_n_jobs)(
            delayed(_skeleton_worker)(x, self.cg, depth, self.alpha)
            for x in range(self.no_of_var)
        )

        # Update progress bar
        if self.show_progress and pbar is not None:
            pbar.update(self.no_of_var)

        # Aggregate results
        aggregated_edge_removals = []

        for edge_removals, sepsets in results_list:
            for (x, y) in edge_removals:
                 if self.stable:
                     self._register_edge_removal(aggregated_edge_removals, x, y)
                 else:
                     self._remove_edge_immediately(x, y)

            for (x, y, S) in sepsets:
                self._append_sepset(x, y, S)

        return aggregated_edge_removals

    def _process_node_at_depth(
        self,
        x: int,
        neigh_x: np.ndarray,
        depth: int,
    ):
        """
        Process one node x at the given depth.
        Returns local edge_removal list (used only when stable=True).
        """
        local_edge_removal = []

        for y in neigh_x:
            # IMPORTANT: do not skip a pair just because we already added (x, y)
            # to edge_removal; needed to collect ALL sepsets in stable mode.
            neigh_x_noy = np.delete(neigh_x, np.where(neigh_x == y))

            S = self._find_separating_set(x, y, neigh_x_noy, depth)

            if S is not None:
                if not self.stable:
                    self._remove_edge_immediately(x, y)
                else:
                    self._register_edge_removal(local_edge_removal, x, y)

                self._append_sepset(x, y, S)

        return local_edge_removal

    def _find_separating_set(self, x, y, candidates, depth):
        """
        Core logic to find a separating set S of size `depth` for (x, y).
        Returns S if found, None otherwise.
        """
        for S in combinations(candidates, depth):
            p_val = self.cg.ci_test(x, y, S)
            if p_val > self.alpha:
                return S
        return None


    def _test_pair_over_subsets(
        self,
        x: int,
        y: int,
        candidates: np.ndarray,
        depth: int,
        edge_removal_acc: list,
    ) -> bool:
        """
        For a given pair (x, y) and its candidate neighbors (without y),
        iterate over all conditioning sets S with |S| = depth.
        If we find p_val > alpha:
          - if not stable: immediately remove edge and record sepset.
          - if stable: record in edge_removal_acc for later removal and record sepset.
        Returns True if a separating set has been found (and loop should break),
        False otherwise.
        """
        for S in combinations(candidates, depth):
            p_val = self.cg.ci_test(x, y, S)
            if p_val > self.alpha:
                if not self.stable:
                    self._remove_edge_immediately(x, y)
                else:
                    self._register_edge_removal(edge_removal_acc, x, y)

                self._append_sepset(x, y, S)
                # As in the original: break after first sepset for (x, y) at this depth.
                return True

        return False

    def _remove_edge_immediately(self, x: int, y: int):
        """
        Remove edge between x and y from cg.G if it exists.
        """
        edge = self.cg.G.get_edge(self.cg.nodes[x], self.cg.nodes[y])
        if edge:
            self.cg.G.remove_edge(edge)

    def _register_edge_removal(self, edge_removal_acc: list, x: int, y: int):
        """
        Register (x, y) and (y, x) for later removal in stable mode.
        """
        edge_removal_acc.append((x, y))
        edge_removal_acc.append((y, x))

    def _apply_edge_removals(self, edge_removal: list[tuple[int, int]]):
        """
        Apply all edge removals collected in stable mode after finishing one depth.
        """
        for (x, y) in set(edge_removal):
            self._remove_edge_immediately(x, y)

    def _append_sepset(self, x, y, S):
        if self.cg.sepset[x, y] is None:
            self.cg.sepset[x, y] = [S]
        else:
            self.cg.sepset[x, y].append(S)
        if self.cg.sepset[y, x] is None:
            self.cg.sepset[y, x] = [S]
        else:
            self.cg.sepset[y, x].append(S)
