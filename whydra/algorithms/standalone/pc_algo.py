"""PC-stable search with CPU orientation and a CUDA entry point."""

import sys
from itertools import permutations
from typing import TYPE_CHECKING, cast

import numpy as np

from ...background_knowledge import (
    BKOrientationGuard,
    BKPhase,
    MatrixEncoding,
    apply_bk_checkpoint,
)
from ..graph_core import CausalGraph, Edge, Endpoint, GeneralGraph
from ..graph_core.conditioning import (
    configure_conditioning,
    validate_cuda_background_knowledge,
)
from ..graph_core.cpu_graph_convert import causal_graph_to_causallearn
from .profiler import profiler
from .skeleton import SkeletonDiscovery

if TYPE_CHECKING:
    from causallearn.graph.GeneralGraph import GeneralGraph as CLGeneralGraph
    from torch import Tensor


class PCAlgorithm:
    """PC-stable orientation: colliders and Meek rules on a discovered skeleton."""

    def __init__(
        self,
        cg: CausalGraph,
        alpha: float = 0.05,
        stable: bool = True,
        uc_rule: int = 0,
        uc_priority: int = 2,
        background_knowledge=None,
        verbose: bool = False,
        show_progress: bool = False,
        n_jobs: int = -1,
        **kwargs,
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
        self.bk_guard = self._guard()
        if self.verbose:
            print(f"n_jobs: {n_jobs}")

    def _guard(self) -> BKOrientationGuard:
        """Guard orientation writes. Without background knowledge, allow all writes."""
        if self.background_knowledge is None:
            return BKOrientationGuard(None, MatrixEncoding.STANDARD)
        return self.background_knowledge.orientation_guard(
            [node.name for node in self.cg.nodes],
            phase=BKPhase.BETWEEN_ORIENTATION,
            encoding=MatrixEncoding.STANDARD,
        )

    def run(self) -> GeneralGraph:
        """Orient the current skeleton and return the resulting CPDAG."""
        self.bk_guard = self._guard()
        with profiler.time_block("full_pc_algorithm", tags={"algo": "sequential"}):
            self._apply_bk("before_orientation")
            self.orient_colliders()
            self._apply_bk("after_colliders")
            self.apply_meek_rules()
            self._apply_bk("after_orientation", phase=BKPhase.POST_ORIENTATION)

        return self.convert_to_general_graph()

    def _apply_bk(
        self, checkpoint, *, previous=None, phase=BKPhase.BETWEEN_ORIENTATION
    ):
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
        """Orient unshielded colliders in the current graph."""
        from .standalone_pc_stable import orient_colliders

        self.cg.bk_guard = self.bk_guard
        orient_colliders(self.cg, priority=self.uc_priority)

    def _bk_allows(self, source, target, at_source, at_target) -> bool:
        """Check whether background knowledge allows these two marks. Without it, always allow them."""
        guard = getattr(self, "bk_guard", None)
        if guard is None or not guard:
            return True
        m = self.cg.G.graph
        return guard.allows_mark(
            m, target, source, at_source.value
        ) and guard.allows_mark(m, source, target, at_target.value)

    def _orient_edge(self, source, target):
        if not self._bk_allows(source, target, Endpoint.TAIL, Endpoint.ARROW):
            return False
        edge = self.cg.G.get_edge(self.cg.nodes[source], self.cg.nodes[target])
        if edge:
            self.cg.G.remove_edge(edge)
            self.cg.G.add_edge(
                Edge(
                    self.cg.nodes[source],
                    self.cg.nodes[target],
                    Endpoint.TAIL,
                    Endpoint.ARROW,
                )
            )
            return True
        return False

    def _orient_edge_conflict_aware(self, source, target):
        """Try to orient source -> target when applying Meek's rules."""
        if self.cg.is_fully_directed(source, target):
            return False
        if self.cg.is_bidirected(source, target):
            return False

        if self.cg.is_fully_directed(target, source):
            # Conflict -> Bi-directed
            if not self._bk_allows(source, target, Endpoint.ARROW, Endpoint.ARROW):
                return False
            edge = self.cg.G.get_edge(self.cg.nodes[source], self.cg.nodes[target])
            if edge:
                self.cg.G.remove_edge(edge)
                self.cg.G.add_edge(
                    Edge(
                        self.cg.nodes[source],
                        self.cg.nodes[target],
                        Endpoint.ARROW,
                        Endpoint.ARROW,
                    )
                )
                return True
            return False

        if self.cg.is_undirected(source, target):
            if self.cg.is_ancestor_of(target, source):
                return False
            if not self._bk_allows(source, target, Endpoint.TAIL, Endpoint.ARROW):
                return False
            edge = self.cg.G.get_edge(self.cg.nodes[source], self.cg.nodes[target])
            if edge:
                self.cg.G.remove_edge(edge)
                self.cg.G.add_edge(
                    Edge(
                        self.cg.nodes[source],
                        self.cg.nodes[target],
                        Endpoint.TAIL,
                        Endpoint.ARROW,
                    )
                )
                return True
            return False
        return False

    def apply_meek_rules(self):
        """Stage 3: original Node/Edge Meek R1–R3 loop."""
        return self.apply_meek_rules_python()

    def apply_meek_rules_python(self):
        """Apply the reference Meek R1–R3 loop using Node and Edge objects."""
        loop = True
        topology = None
        triples, triangles, kites = [], [], []
        while loop:
            before_pass = self.cg.G.graph.copy()
            adjacent = before_pass != 0
            if topology is None or not np.array_equal(topology, adjacent):
                # Endpoint orientation does not change these candidate sets.
                # Rebuild if a background-knowledge checkpoint changed topology.
                topology = adjacent
                triples = self.cg.find_unshielded_triples()
                triangles = self.cg.find_triangles()
                kites = self.cg.find_kites()

            # R1
            for i, j, k in triples:
                if self.cg.is_fully_directed(i, j) and self.cg.is_undirected(j, k):
                    self._orient_edge_conflict_aware(j, k)
                elif self.cg.is_fully_directed(k, j) and self.cg.is_undirected(j, i):
                    self._orient_edge_conflict_aware(j, i)

            # R2
            for i, j, k in triangles:
                nodes_tri = [i, j, k]
                for a, b, c in permutations(nodes_tri, 3):
                    if (
                        self.cg.is_fully_directed(a, b)
                        and self.cg.is_fully_directed(b, c)
                        and self.cg.is_undirected(a, c)
                    ):
                        self._orient_edge_conflict_aware(a, c)

            # R3
            for i, j, k, target in kites:
                if (
                    (
                        self.cg.is_fully_directed(j, target)
                        and self.cg.is_fully_directed(k, target)
                        and self.cg.is_undirected(i, target)
                    )
                    and self.cg.is_undirected(i, j)
                    and self.cg.is_undirected(i, k)
                ):
                    self._orient_edge_conflict_aware(i, target)

            self._apply_bk("after_meek_pass", previous=before_pass)
            loop = not np.array_equal(before_pass, self.cg.G.graph)

        return self.cg

    def convert_to_general_graph(self) -> GeneralGraph:
        """Return the graph already maintained by the local algorithm."""
        return self.cg.G

    def convert_to_causallearn_graph(self) -> "CLGeneralGraph":
        """Return the graph in causal-learn's matrix encoding."""
        return causal_graph_to_causallearn(self.cg)


def _run_pc_cuda(
    cg: CausalGraph,
    alpha=0.05,
    stable: bool = True,
    uc_rule: int = 0,
    uc_priority: int = 2,
    background_knowledge=None,
    verbose: bool = False,
    show_progress: bool = True,
    dtype: str | None = None,
    device: str = "cuda",
    _return_local_graph: bool = False,
    **kwargs,
):
    """PC-stable-cuda: skeleton, colliders, Meek, and dpath stay on device.

    One H2D of the data at ``CudaFisherZ.from_data``. One D2H of the endpoint
    CPDAG and CSR at the end. ``k > K_MAX`` falls back to the host wavefront
    plus host colliders / Meek.
    """
    validate_cuda_background_knowledge(background_knowledge, "pc_stable_cuda")
    from ..graph_core.cuda_contracts import validate_cuda_ci_test

    if cg.test is not None:
        validate_cuda_ci_test(cg.test, "PC-stable-cuda")
    from ..graph_core.endpoints import Endpoint
    from ..graph_core.meek_torch import meek_orient_tensor, orient_colliders_tensor

    disc = SkeletonDiscovery(
        cg=cg,
        alpha=alpha,
        stable=stable,
        n_jobs=1,
        show_progress=show_progress,
        verbose=verbose,
        device=device,
        dtype=dtype,
    )
    G_t, hits, _, fallback = disc._run_cuda_skeleton(fci=False, to_host=False)
    if fallback is not None:
        disc._epilogue_from_device(G_t, hits, materialize_host_sepsets=True)
        pbar = disc._init_progress_bar(show_progress)
        print(
            f"[cuda skeleton] k={fallback} > k_max; "
            "Python combinadic wavefront on CudaFisherZ.batch",
            file=sys.stderr,
            flush=True,
        )
        depth = fallback - 1
        while disc._continue_search(depth):
            depth += 1
            if show_progress:
                disc._reset_progress_bar(pbar, depth)
            edge_removal = disc._process_depth_cuda_wavefront(depth, pbar)
            if stable:
                disc._apply_edge_removals(edge_removal)
        disc._close_progress_bar(pbar)
        PCAlgorithm(
            cg=cg,
            alpha=alpha,
            stable=stable,
            uc_rule=uc_rule,
            uc_priority=uc_priority,
            background_knowledge=None,
            verbose=verbose,
            show_progress=show_progress,
            n_jobs=1,
            **kwargs,
        ).run()
        return cg.G if _return_local_graph else causal_graph_to_causallearn(cg)

    S, ranges = hits[1], hits[2]
    G_t.masked_fill_(G_t != 0, Endpoint.TAIL.value)
    # The resident, non-fallback path requested device tensors with to_host=False.
    orient_colliders_tensor(G_t, cast("Tensor", S), cast("Tensor", ranges))
    meek_orient_tensor(G_t)
    disc._epilogue_from_device(G_t, hits, copy_endpoints=True)
    cg._G_cuda = G_t
    cg._cuda_endpoints = True
    return cg.G if _return_local_graph else causal_graph_to_causallearn(cg)


def pc_stable(
    cg: CausalGraph,
    alpha=0.05,
    stable: bool = True,
    uc_rule: int = 0,
    uc_priority: int = 2,
    background_knowledge=None,
    verbose: bool = False,
    show_progress: bool = False,
    n_jobs: int = 1,
    max_cond_set_size=None,
    skeleton_backend: str = "auto",
    device: str | None = None,
    dtype: str | None = None,
    **kwargs,
):
    """PC-stable / PC-stable-cuda.

    Sequential (default): ``n_jobs=1``, one-at-a-time ``CIT`` tests, Python Meek.
    PC-stable-cuda: ``device="cuda"`` fused skeleton + CUDA colliders + CUDA Meek.
    ``n_jobs`` is 1. Fisher-Z: fused CI for ``k ≤ 32``, ``CudaFisherZ`` for
    ``k ≤ 96``. ``chisq``: same CUDA graph loop; tiled k=0/k=1 histograms,
    ``k≥2`` packed-Z / LSD CSR counting-sort reused as k=1.
    See ``docs/pc_stable_cuda.md``.
    """
    cuda = device is not None and str(device).startswith("cuda")
    if cuda:
        validate_cuda_background_knowledge(background_knowledge, "pc_stable_cuda")
    configure_conditioning(cg, max_cond_set_size)
    if cuda:
        n_jobs = 1
        if not stable:
            msg = (
                "PC-stable-cuda supports only stable=True; use the CPU "
                "backend for stable=False"
            )
            raise ValueError(msg)
        assert device is not None
        return _run_pc_cuda(
            cg,
            alpha=alpha,
            stable=stable,
            uc_rule=uc_rule,
            uc_priority=uc_priority,
            background_knowledge=background_knowledge,
            verbose=verbose,
            show_progress=show_progress,
            dtype=dtype,
            device=device,
            **kwargs,
        )
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
        parallel_backend=skeleton_backend,
        device=device,
        dtype=dtype,
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
        **kwargs,
    ).run()

    return cg_res
