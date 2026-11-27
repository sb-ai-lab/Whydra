"""Independent PCMCI+ controller over the local search primitives.

The four stages follow Runge, UAI 2020, Algorithms 1, 2 and S2–S3:
https://proceedings.mlr.press/v124/runge20a.html
The implementation shares our PC1, stable MCI skeleton and orientation code
with J-PCMCI+.
"""

from copy import deepcopy
from itertools import combinations
from typing import Any, cast

import numpy as np

from ._joint_search import JointSearch


class PCMCIPlusPipeline(JointSearch):
    """Lagged screening followed by contemporaneous MCI search and orientation."""

    implementation = "Native PCMCI+ (Runge 2020)"

    def run_pcmciplus(
        self,
        selected_links=None,
        link_assumptions=None,
        tau_min=0,
        tau_max=1,
        pc_alpha=0.01,
        contemp_collider_rule="majority",
        conflict_resolution=True,
        reset_lagged_links=False,
        max_conds_dim=None,
        max_combinations=1,
        max_conds_py=None,
        max_conds_px=None,
        max_conds_px_lagged=None,
        fdr_method="none",
    ):
        """Return a temporal CPDAG and the evidence from skeleton CI tests.

        ``pc_alpha=None`` or a list runs complete candidate searches and chooses
        the minimum CI-test model criterion on a consistent DAG extension.
        Optional ``fdr_bh`` adjusts the lagged evidence family, including
        self-dependencies, and removes optional lagged links above the same
        threshold before orientation. This is BH-adjusted link selection;
        it is not a claim of an FDR theorem for adaptive PCMCI+ orientations.
        """
        if selected_links is not None:
            msg = "Use link_assumptions instead of selected_links"
            raise ValueError(msg)
        self._check_tau_limits(tau_min, tau_max)
        if contemp_collider_rule not in {"majority", "conservative", "none"}:
            msg = "Unknown contemporaneous collider rule"
            raise ValueError(msg)
        if fdr_method not in {None, "none", "fdr_bh"}:
            msg = "Supported fdr_method: none, fdr_bh"
            raise ValueError(msg)
        if pc_alpha is None or isinstance(pc_alpha, (list, tuple, np.ndarray)):
            candidates = (
                [0.001, 0.005, 0.01, 0.025, 0.05]
                if pc_alpha is None
                else list(pc_alpha)
            )
            return self._select_alpha(
                candidates,
                {
                    "link_assumptions": link_assumptions,
                    "tau_min": tau_min,
                    "tau_max": tau_max,
                    "contemp_collider_rule": contemp_collider_rule,
                    "conflict_resolution": conflict_resolution,
                    "reset_lagged_links": reset_lagged_links,
                    "max_conds_dim": max_conds_dim,
                    "max_combinations": max_combinations,
                    "max_conds_py": max_conds_py,
                    "max_conds_px": max_conds_px,
                    "max_conds_px_lagged": max_conds_px_lagged,
                    "fdr_method": fdr_method,
                },
            )
        if (
            isinstance(pc_alpha, (bool, np.bool_))
            or not np.isscalar(pc_alpha)
            or not np.isfinite(pc_alpha)
            or not 0 <= cast("float", pc_alpha) <= 1
        ):
            msg = "pc_alpha must be a finite number in [0, 1]"
            raise ValueError(msg)

        assumptions = self._set_link_assumptions(link_assumptions, tau_min, tau_max)
        self._search_link_assumptions = deepcopy(assumptions)
        self.diagnostics = []
        self.mode = "pc1"
        parents = self.run_pc_stable(
            link_assumptions=assumptions,
            tau_min=tau_min,
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            max_conds_dim=max_conds_dim,
            max_combinations=max_combinations,
        )
        self.all_lagged_parents = deepcopy(parents)
        self.mode = "pcmci_plus_skeleton"
        skeleton = self._pcmciplus_mci_skeleton_phase(
            parents,
            assumptions,
            pc_alpha,
            tau_min,
            tau_max,
            max_conds_dim,
            None,
            max_conds_py,
            max_conds_px,
            max_conds_px_lagged,
            reset_lagged_links,
            "none",
            self.p_matrix,
            self.val_matrix,
        )
        if fdr_method == "fdr_bh":
            self._select_bh_lagged_links(skeleton, assumptions, pc_alpha)

        self.mode = "pcmci_plus_colliders"
        colliders = self._pcmciplus_collider_phase(
            skeleton["graph"],
            skeleton["sepsets"],
            parents,
            pc_alpha,
            tau_min,
            tau_max,
            max_conds_py,
            max_conds_px,
            max_conds_px_lagged,
            conflict_resolution,
            contemp_collider_rule,
        )
        self.mode = "pcmci_plus_orientation"
        graph = self._pcmciplus_rule_orientation_phase(
            colliders["graph"], colliders["ambiguous_triples"], conflict_resolution
        )
        self.results: dict[str, Any] = {
            "graph": graph,
            "p_matrix": skeleton["p_matrix"],
            "val_matrix": skeleton["val_matrix"],
            "sepsets": colliders["sepsets"],
            "ambiguous_triples": colliders["ambiguous_triples"],
            "conf_matrix": None,
        }
        if fdr_method == "fdr_bh":
            self.results["fdr_method"] = "fdr_bh"
        if self.verbosity:
            print(f"PCMCI+: {np.count_nonzero(graph)} temporal graph marks")
        return self.results

    def _select_bh_lagged_links(self, skeleton, assumptions, alpha):
        from ._mci import adjust_pvalues

        skeleton["p_matrix"] = adjust_pvalues(
            skeleton["p_matrix"],
            fdr_method="fdr_bh",
            exclude_contemporaneous=True,
            link_assumptions=assumptions,
        )
        knowledge = getattr(self, "_background_knowledge", None)
        for j, incoming in assumptions.items():
            for (i, lag), mark in incoming.items():
                if lag == 0 or skeleton["graph"][i, j, -lag] == "":
                    continue
                protected = "?" not in mark or (
                    knowledge is not None and knowledge.protected(i, j, -lag)
                )
                if not protected and skeleton["p_matrix"][i, j, -lag] > alpha:
                    skeleton["graph"][i, j, -lag] = ""
                    # An adjusted p-value is not an observed conditioning set.
                    # Collider tests may recover evidence, while the standard
                    # rule must treat this newly unshielded triple as ambiguous.
                    self._prior_absent[i, j, -lag] = True
        self.p_matrix = skeleton["p_matrix"]

    def _select_alpha(self, candidates, options):
        if not candidates:
            msg = "At least one candidate pc_alpha is required"
            raise ValueError(msg)
        best, scores = None, []
        for alpha in candidates:
            result = self.run_pcmciplus(pc_alpha=alpha, **options)
            parents = self._scoring_parents(result["graph"])
            criterion = sum(
                self.cond_ind_test.get_model_selection_criterion(
                    j, parents[j], options["tau_max"]
                )
                for j in range(self.N)
            )
            scores.append(float(criterion))
            if best is None or criterion < best[0]:
                state = {
                    name: deepcopy(getattr(self, name))
                    for name in (
                        "all_lagged_parents",
                        "all_parents",
                        "val_min",
                        "pval_max",
                        "iterations",
                        "p_matrix",
                        "val_matrix",
                        "diagnostics",
                        "_orientation_locks",
                        "_prior_absent",
                        "_search_link_assumptions",
                    )
                }
                best = criterion, deepcopy(result), state, alpha
        selected = cast("tuple[float, dict[str, Any], dict[str, Any], float]", best)
        for name, value in selected[2].items():
            setattr(self, name, value)
        self.results = selected[1]
        self.results["optimal_alpha"] = selected[3]
        self.results["alpha_scores"] = dict(zip(candidates, scores, strict=False))
        return self.results

    def _scoring_parents(self, graph):
        """Choose a DAG extension by successively eliminating admissible sinks."""
        zero = graph[:, :, 0]
        if np.any(zero == "x-x"):
            msg = "Cannot select pc_alpha: conflicting edges have no defined DAG extension"
            raise ValueError(msg)
        directed = {
            (i, j) for i in range(self.N) for j in range(self.N) if zero[i, j] == "-->"
        }
        unresolved = {
            frozenset((i, j))
            for i in range(self.N)
            for j in range(i + 1, self.N)
            if zero[i, j] == "o-o"
        }
        remaining = set(range(self.N))
        while remaining:
            sink = None
            for node in sorted(remaining):
                if any(a == node and b in remaining for a, b in directed):
                    continue
                neighbors = {
                    other
                    for other in remaining
                    if frozenset((node, other)) in unresolved
                }
                parents = {a for a, b in directed if b == node and a in remaining}
                if any(zero[a, b] == "" for a, b in combinations(neighbors, 2)):
                    continue
                if any(zero[a, b] == "" for a in neighbors for b in parents if a != b):
                    continue
                sink = node
                directed.update((neighbor, node) for neighbor in neighbors)
                break
            if sink is None:
                msg = "Cannot select pc_alpha: the discovered graph has no consistent DAG extension"
                raise ValueError(msg)
            remaining.remove(sink)
        result = {
            j: [
                (i, -lag)
                for i in range(self.N)
                for lag in range(1, graph.shape[2])
                if graph[i, j, lag] == "-->"
            ]
            for j in range(self.N)
        }
        for i, j in sorted(directed):
            result[j].append((i, 0))
        return result
