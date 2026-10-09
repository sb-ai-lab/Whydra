import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping

from itertools import combinations, permutations
import math

from ..graph_core import CausalGraph, fisher_z_from_corr
from .profiler import profiler

# ------------------------------
# Worker-функция (вынести НА УРОВЕНЬ МОДУЛЯ, не внутрь класса!)
# ------------------------------
def _skeleton_worker(
    x,
    cg,
    depth,
    alpha,
    stable=True,
    protected_edges=frozenset(),
):
    """
    Worker function to process a single variable x.

    Checks for independence of x with its neighbours at the given depth.
    In stable mode (causal-learn parity) every conditioning set of the
    current depth is tested and one union of separating-set members is
    recorded per separated pair for collider orientation;
    in non-stable mode the search stops at the first separating set.
    A pair protected by background knowledge also stops at the first
    separating set: its edge survives and its sepsets are discarded.

    Args:
        x: node index to process.
        cg: causal graph carrying the CI test.
        depth: size of the conditioning sets.
        alpha: significance level.
        stable: whether to enumerate all separating sets (PC-stable).
        protected_edges: pairs ``frozenset((x, y))`` kept by background knowledge.

    Returns:
        Tuple (edge_removals, sepset_updates), with one (x, y, union)
        record per separated pair; union is a sorted tuple of Python ints.
    """
    edge_removals = []
    sepset_updates = []

    neigh_x = cg.neighbors(x)
    if len(neigh_x) < depth:
        return edge_removals, sepset_updates

    for y in neigh_x:
        neigh_x_noy = np.delete(neigh_x, np.where(neigh_x == y))
        protected = frozenset((int(x), int(y))) in protected_edges

        found = False
        members = set()
        for S in combinations(neigh_x_noy, depth):
            p_val = cg.ci_test(x, y, S)
            if p_val > alpha:
                if not found:
                    edge_removals.append((x, y))
                    found = True
                members.update(int(v) for v in S)
                if not stable or protected:
                    break

        if found:
            sepset_updates.append((x, y, tuple(sorted(members))))

    return edge_removals, sepset_updates

def ci_worker(args):
    x, y, S, ci_func = args
    return x, y, S, ci_func(x, y, S)

def _skeleton_worker_fci(x, cg, G1, depth, alpha, pMax_row, stable=True):
    """
    Worker function for FCI parallel processing.

    ``stable`` governs the early exit exactly as it does in the sequential
    path and in causal-learn's FAS: without it the search stops at the first
    separating set, with it every conditioning set at this depth is examined
    because the stored separating set is their union.
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
                    if not stable:
                        break

    return edge_removals, sepset_updates, pmax_updates, tests_performed


def _ci_cache_key_pair(i, j, S):
    a, b = (int(i), int(j)) if i <= j else (int(j), int(i))
    return a, b, tuple(sorted(int(v) for v in S))


def _cached_fisher_z(local_cache, corr_matrix, n_samples, x, y, S):
    key = _ci_cache_key_pair(x, y, S)
    cached = local_cache.get(key)
    if cached is not None:
        return cached
    pval = fisher_z_from_corr(corr_matrix, n_samples, x, y, S)
    local_cache[key] = pval
    return pval


def _process_fci_direction_fisherz(x, y, G1, corr_matrix, n_samples, depth, alpha, pmax_value,
                                   local_cache, stable=True):
    edge_removals = []
    sepset_updates = []
    pmax_updates = []
    tests_performed = 0

    nbrs = np.where(G1[x, :] != 0)[0]
    nbrs = nbrs[nbrs != y]

    if len(nbrs) >= depth:
        current_pmax = pmax_value
        for S in combinations(nbrs, depth):
            tests_performed += 1
            pval = _cached_fisher_z(local_cache, corr_matrix, n_samples, x, y, S)
            if pval > current_pmax:
                current_pmax = pval
                pmax_updates.append((x, y, pval))
            if pval >= alpha:
                edge_removals.append((x, y))
                sepset_updates.append((x, y, S))
                if not stable:
                    break

    return edge_removals, sepset_updates, pmax_updates, tests_performed


def _skeleton_worker_fci_fisherz_edge_chunk(edges, G1, pMax, corr_matrix, n_samples, depth, alpha,
                                            stable=True):
    edge_removals = []
    sepset_updates = []
    pmax_updates = []
    tests_performed = 0
    local_cache = {}

    for x, y in edges:
        for a, b in ((x, y), (y, x)):
            out = _process_fci_direction_fisherz(
                a, b, G1, corr_matrix, n_samples, depth, alpha, pMax[a, b], local_cache, stable
            )
            dir_edge_removals, dir_sepset_updates, dir_pmax_updates, dir_tests = out
            edge_removals.extend(dir_edge_removals)
            sepset_updates.extend(dir_sepset_updates)
            pmax_updates.extend(dir_pmax_updates)
            tests_performed += dir_tests

    return edge_removals, sepset_updates, pmax_updates, tests_performed


def _fci_edge_cost(G1, x, y, depth):
    cost = 0
    nbrs_x = np.where(G1[x, :] != 0)[0]
    nbrs_x = nbrs_x[nbrs_x != y]
    if len(nbrs_x) >= depth:
        cost += math.comb(len(nbrs_x), depth)

    nbrs_y = np.where(G1[y, :] != 0)[0]
    nbrs_y = nbrs_y[nbrs_y != x]
    if len(nbrs_y) >= depth:
        cost += math.comb(len(nbrs_y), depth)

    return cost


def _balanced_edge_chunks(edges, G1, depth, n_chunks):
    weighted_edges = [(_fci_edge_cost(G1, x, y, depth), (x, y)) for x, y in edges]
    weighted_edges.sort(key=lambda item: item[0], reverse=True)

    chunks = [[] for _ in range(n_chunks)]
    loads = [0 for _ in range(n_chunks)]
    for cost, edge in weighted_edges:
        idx = min(range(n_chunks), key=lambda i: loads[i])
        chunks[idx].append(edge)
        loads[idx] += cost

    return [chunk for chunk in chunks if chunk]


class SkeletonDiscovery:
    def __init__(
        self,
        cg: CausalGraph,
        alpha: float = 0.05,
        stable: bool = True,
        n_jobs: int = 1,
        show_progress: bool = True,
        verbose: bool = False,
        parallel_backend: str = "threads",
        process_ci_info=None,
        protected_edges=(),
    ):
        self.cg = cg
        self.alpha = alpha
        self.stable = stable
        self.n_jobs = n_jobs
        self.show_progress = show_progress
        self.verbose = verbose
        self.parallel_backend = parallel_backend
        self.process_ci_info = process_ci_info
        self.protected_edges = {
            frozenset((int(x), int(y))) for x, y in protected_edges
        }
        self._parallel_executor = None
        self.ci_tests_performed = 0

        self.no_of_var = len(self.cg.nodes)
        # We work directly on cg.G.graph
        self.G = self.cg.G.graph
        self.G1 = np.copy(self.G)

    def _validate_fci_parallel_config(self):
        if self.n_jobs == 1 or self.parallel_backend != "processes":
            return

        info = self.process_ci_info
        if not isinstance(info, Mapping) or info.get("method") != "fisherz":
            raise ValueError(
                "run_fci with parallel_backend='processes' requires "
                "process_ci_info with method='fisherz'"
            )

        missing = sorted({"corr_matrix", "n_samples"}.difference(info))
        if missing:
            raise ValueError(
                "run_fci with parallel_backend='processes' has incomplete "
                f"process_ci_info; missing required keys: {missing}"
            )

    def run(self):
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

        return self.convert_to_general_graph()


    def run_fci(self):
        self._validate_fci_parallel_config()

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

        if self.n_jobs != 1:
            if self.parallel_backend == "processes":
                self._parallel_executor = Parallel(
                    n_jobs=self.n_jobs,
                    backend="loky",
                    max_nbytes="10K",
                )
                self._parallel_executor.__enter__()
            else:
                max_workers = self.n_jobs if self.n_jobs > 0 else cpu_count()
                self._parallel_executor = ThreadPoolExecutor(max_workers=max_workers)

        try:
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
        finally:
            if self._parallel_executor is not None:
                if isinstance(self._parallel_executor, ThreadPoolExecutor):
                    self._parallel_executor.shutdown(wait=True)
                else:
                    self._parallel_executor.__exit__(None, None, None)
                self._parallel_executor = None

        self._close_progress_bar(pbar)

        return {'sk': self.cg.G.graph, 'pMax': self.cg.pMax, 'sepset': sepset, "unfTriples": set(), "max_ord": depth - 1}

    def _process_depth_fci(
        self,
        depth: int,
        pbar,
        sepset,
    ):
        # print(f'depth = {depth}')


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

            # Fast-path guard on the LIVE graph: self.G aliases cg.G.graph, so
            # when stable=False it already reflects removals made earlier in this
            # depth. Inside _process_node_at_depth_fci the edges are filtered on
            # self.G too, but conditioning sets come from the self.G1 snapshot,
            # so this check is not redundant with the len(nbrs) >= depth one there.
            if len(self.cg.neighbors(x)) < depth:
                continue

            removals_for_x, sepset= self._process_node_at_depth_fci(
                x=x,
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
                for S in combinations(nbrs, depth):
                    pval = self.cg.ci_test(x, y, S)
                    if self.cg.pMax[x, y] < pval:
                        self.cg.pMax[x, y] = pval

                    if pval >= self.alpha:
                        # Remove edge (set to NULL=0) now, or at the end of the
                        # depth when running stable.
                        if not self.stable:
                            self._remove_edge_immediately(x, y)
                        else:
                            self._register_edge_removal(edge_removal, x, y)

                        self._record_sepset(sepset, x, y, S)
                        if not self.stable:
                            break
        return edge_removal, sepset


    def convert_to_general_graph(self):
        """Return the local graph maintained by the algorithm."""
        return self.cg.G

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

        if self.verbose:
            print(f"Depth {depth}: Estimated max CI tests: {total_ops_estimate}")
            print(f"Jobs in parallel: {self.n_jobs}")
        if (
            self.parallel_backend == "processes"
            and self.process_ci_info is not None
            and self.process_ci_info.get("method") == "fisherz"
        ):
            effective_n_jobs = self.n_jobs if self.n_jobs > 0 else cpu_count()
            edges = [
                (x, y)
                for x in range(self.no_of_var)
                for y in range(x + 1, self.no_of_var)
                if self.G[x, y] != 0 or self.G[y, x] != 0
            ]
            edge_chunks = _balanced_edge_chunks(edges, self.G1, depth, effective_n_jobs)
            executor = self._parallel_executor or Parallel(
                n_jobs=effective_n_jobs,
                backend="loky",
                max_nbytes="10K",
            )
            results_list = executor(
                delayed(_skeleton_worker_fci_fisherz_edge_chunk)(
                    edge_chunk,
                    self.G1,
                    self.cg.pMax,
                    self.process_ci_info["corr_matrix"],
                    self.process_ci_info["n_samples"],
                    depth,
                    self.alpha,
                    self.stable,
                )
                for edge_chunk in edge_chunks
            )
        else:
            owns_executor = self._parallel_executor is None
            executor = self._parallel_executor or ThreadPoolExecutor(
                max_workers=self.n_jobs if self.n_jobs > 0 else cpu_count()
            )
            try:
                futures = [
                    executor.submit(
                        _skeleton_worker_fci,
                        x,
                        self.cg,
                        self.G1,
                        depth,
                        self.alpha,
                        self.cg.pMax[x],
                        self.stable,
                    )
                    for x in range(self.no_of_var)
                ]
                results_list = [future.result() for future in futures]
            finally:
                if owns_executor:
                    executor.shutdown(wait=True)

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
                self._record_sepset(sepset, x, y, S)

        if self.verbose:
            print(f"Depth {depth}: Actual CI tests performed: {total_tests_actual}")
        self.ci_tests_performed += total_tests_actual
        return aggregated_edge_removals, sepset

    def _process_depth_parallel(self, depth, pbar):
        """
        Process a PC depth in parallel and merge separating-member unions.

        Args:
            depth: size of the conditioning sets.
            pbar: optional progress bar.

        Returns:
            list: registered edge removals for stable mode.
        """
        effective_n_jobs = self.n_jobs if self.n_jobs > 0 else cpu_count()

        # Run parallel jobs for each variable x
        if self.parallel_backend == "processes":
            executor = self._parallel_executor or Parallel(
                n_jobs=effective_n_jobs,
                backend="loky",
            )
            results_list = executor(
                delayed(_skeleton_worker)(
                    x, self.cg, depth, self.alpha, self.stable, self.protected_edges
                )
                for x in range(self.no_of_var)
            )
        else:
            owns_executor = self._parallel_executor is None
            executor = self._parallel_executor or ThreadPoolExecutor(
                max_workers=effective_n_jobs
            )
            try:
                futures = [
                    executor.submit(
                        _skeleton_worker, x, self.cg, depth, self.alpha, self.stable,
                        self.protected_edges,
                    )
                    for x in range(self.no_of_var)
                ]
                results_list = [future.result() for future in futures]
            finally:
                if owns_executor:
                    executor.shutdown(wait=True)

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
                self._merge_sepset_union(x, y, S)

        return aggregated_edge_removals

    def _process_node_at_depth(
        self,
        x: int,
        neigh_x: np.ndarray,
        depth: int,
    ):
        """
        Process one node x at the given depth.

        For every neighbour y, the union of separating-set members at the
        current depth is collected (stable mode) and the (x, y) edge removal
        is registered once if at least one separating set exists.

        Args:
            x: node index to process.
            neigh_x: stable snapshot of the neighbours of x.
            depth: size of the conditioning sets.

        Returns:
            list: local edge_removal list (used only when stable=True).
        """
        local_edge_removal = []

        for y in neigh_x:
            # IMPORTANT: do not skip a pair just because we already added (x, y)
            # to edge_removal; needed to union all separating members in stable mode.
            neigh_x_noy = np.delete(neigh_x, np.where(neigh_x == y))

            members = self._find_separating_union(x, y, neigh_x_noy, depth)

            if members is not None:
                if not self.stable:
                    self._remove_edge_immediately(x, y)
                else:
                    self._register_edge_removal(local_edge_removal, x, y)

                self._merge_sepset_union(x, y, members)

        return local_edge_removal

    def _find_separating_union(
        self,
        x,
        y,
        candidates,
        depth,
    ) -> tuple | None:
        """Find the union of separating-set members for a pair at one depth.

        Stable mode tests every conditioning set. Non-stable mode and
        protected pairs stop at the first separating set.

        Args:
            x: first node of the pair.
            y: second node of the pair.
            candidates: neighbours of x excluding y.
            depth: size of the conditioning sets.

        Returns:
            tuple | None: sorted Python int members, or None if no set
            separates the pair. An empty tuple denotes depth-zero separation.
        """
        found = False
        members = set()
        for S in combinations(candidates, depth):
            p_val = self.cg.ci_test(x, y, S)
            if p_val > self.alpha:
                found = True
                members.update(int(v) for v in S)
                if not self.stable or self._is_protected(x, y):
                    break
        return tuple(sorted(members)) if found else None

    def _merge_sepset_union(
        self,
        x,
        y,
        members,
    ):
        """Merge PC separating members into one tuple in both directions.

        Protected pairs keep their edge and receive no separating-set entry.

        Args:
            x: first node index.
            y: second node index.
            members: separating-set members to merge as Python integers.

        Returns:
            None: updates both symmetric cells in place, preserving empty sets.
        """
        if self._is_protected(x, y):
            return
        members = {int(v) for v in members}
        for i, j in ((x, y), (y, x)):
            current = self.cg.sepset[i, j]
            union = members if current is None else members.union(current[0])
            self.cg.sepset[i, j] = [tuple(sorted(union))]

    def _is_protected(self, x: int, y: int) -> bool:
        """True when background knowledge requires this edge to survive the search.

        A protected pair is excluded from BOTH halves of a deletion: the edge
        itself and the separating set that justified deleting it. Recording the
        sepset of an edge that stays is not harmless bookkeeping — collider
        orientation later asks whether the middle node is in the separating set
        of the outer pair, so a sepset left behind for a surviving edge silently
        changes the orientation of triples around it.
        """
        return frozenset((int(x), int(y))) in self.protected_edges

    def _remove_edge_immediately(self, x: int, y: int):
        """
        Remove edge between x and y from cg.G if it exists.
        """
        if self._is_protected(x, y):
            return
        edge = self.cg.G.get_edge(self.cg.nodes[x], self.cg.nodes[y])
        if edge:
            self.cg.G.remove_edge(edge)

    def _register_edge_removal(self, edge_removal_acc: list, x: int, y: int):
        """
        Register (x, y) and (y, x) for later removal in stable mode.
        """
        if self._is_protected(x, y):
            return
        edge_removal_acc.append((x, y))
        edge_removal_acc.append((y, x))

    def _apply_edge_removals(self, edge_removal: list[tuple[int, int]]):
        """
        Apply all edge removals collected in stable mode after finishing one depth.
        """
        for (x, y) in set(edge_removal):
            self._remove_edge_immediately(x, y)

    def _record_sepset(self, sepset, x, y, S):
        """Single entry point for the flat `sepset` map, mirroring causal-learn's FAS.

        causal-learn applies two different rules, and `udag2pag` is tuned to
        them, so both are reproduced here rather than picked from:

        * ``stable=False`` — the search stops at the first separating set and
          that one set is stored (``FAS.py``, the ``if not stable`` branch);
        * ``stable=True``  — every conditioning set that separates the pair is
          examined and the stored value is the UNION of their members
          (``FAS.py``: ``sepsets.add(s)``, then the ``origin_set`` rebuild over
          the whole of ``cg.sepset[x, y]``).

        The union is therefore deliberate, not an accident of a missing
        ``break``. R0 in `udag2pag` decides colliders by asking whether y is in
        the separating set of (x, z); answering that the same way causal-learn
        does is what keeps our PAGs identical to the reference implementation.
        Dropping to a single set under ``stable=True`` measurably diverges on
        bnlearn/child and bnlearn/water.

        What this method does fix is consistency: both branches store a `set`
        of node indices, and `stable` — not `n_jobs` — is the only thing that
        selects between them, so sequential and parallel backends now agree.
        The full per-pair list of sets stays in `cg.sepset` via `_append_sepset`.

        Under ``stable=True`` (what every algorithm entry point uses) skeleton
        and sepsets are identical for any `n_jobs`. Under ``stable=False`` edges
        disappear mid-sweep, so which separating set is reached still depends on
        traversal order — that order dependence is what `stable` exists to
        remove, and it is present in causal-learn for the same reason.
        """
        if self._is_protected(x, y):
            return
        members = {int(v) for v in S}
        if self.stable:
            sepset[(x, y)] = set(sepset.get((x, y), set())) | members
            sepset[(y, x)] = set(sepset.get((y, x), set())) | members
        else:
            sepset[(x, y)] = set(members)
            sepset[(y, x)] = set(members)
        self._append_sepset(x, y, S)

    #временно довавил проверку на дубликаты, так как у нас иначе будет 2 раза один и тот же сепсет
    def _append_sepset(self, x, y, S):
        if self._is_protected(x, y):
            return
        if self.cg.sepset[x, y] is None:
            self.cg.sepset[x, y] = [S]
        elif S not in self.cg.sepset[x, y]:
            self.cg.sepset[x, y].append(S)
        if self.cg.sepset[y, x] is None:
            self.cg.sepset[y, x] = [S]
        elif S not in self.cg.sepset[y, x]:
            self.cg.sepset[y, x].append(S)
