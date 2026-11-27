"""Native J-PCMCI+ phase controller over separate environments.

Implements Algorithm 2 of Gunther, Ninad and Runge (UAI 2023), with the
PC1/PCMCI+ search primitives from Runge (UAI 2020). The implementation
uses local sample alignment, CI tests and graph operations.
https://proceedings.mlr.press/v216/gunther23a.html
https://proceedings.mlr.press/v124/runge20a.html
"""

from copy import deepcopy
from typing import Any, cast

import numpy as np

from ._joint_search import JointSearch


class JPCMCIplus(JointSearch):
    """Joint discovery with observed contexts and vector categorical dummies."""

    implementation = "Native J-PCMCI+ (Gunther et al. 2023)"
    _roles = frozenset(
        {"system", "time_context", "space_context", "time_dummy", "space_dummy"}
    )
    system_nodes: list[int]
    time_context_nodes: list[int]
    space_context_nodes: list[int]
    time_dummy: list[int]
    space_dummy: list[int]
    _tau_max: int
    all_lagged_parents: dict[int, list[tuple[int, int]]]

    def __init__(self, node_classification, *, dataframe, cond_ind_test, verbosity=0):
        super().__init__(
            dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=verbosity
        )
        if set(node_classification) != set(range(self.N)):
            msg = "Specify a role for every logical node"
            raise ValueError(msg)
        if set(node_classification.values()) - self._roles:
            msg = "Unknown J-PCMCI+ node role"
            raise ValueError(msg)
        self.node_classification = dict(node_classification)
        for role, attribute in [
            ("system", "system_nodes"),
            ("time_context", "time_context_nodes"),
            ("space_context", "space_context_nodes"),
            ("time_dummy", "time_dummy"),
            ("space_dummy", "space_dummy"),
        ]:
            setattr(self, attribute, self.group_nodes(node_classification, role))
        if not self.system_nodes:
            msg = "At least one system node is required"
            raise ValueError(msg)
        self.observed_context_parents = self._empty_parents()
        self.dummy_parents = self._empty_parents()
        self.mode = "system_search"

    def group_nodes(self, node_types, node_type):
        return [i for i in range(self.N) if node_types[i] == node_type]

    def _empty_parents(self):
        return {i: [] for i in range(self.N)}

    @property
    def _contexts(self):
        return self.time_context_nodes + self.space_context_nodes

    @property
    def _dummies(self):
        return self.time_dummy + self.space_dummy

    def _joint_assumptions(self, links, tau_min, tau_max):
        """Apply exogeneity and the time/space structure of context variables."""
        result = self._set_link_assumptions(links, tau_min, tau_max)
        for target, incoming in result.items():
            role = self.node_classification[target]
            kept = {}
            for (source, lag), mark in incoming.items():
                source_role = self.node_classification[source]
                if role.endswith("dummy"):
                    continue
                if lag and source_role in {
                    "time_dummy",
                    "space_dummy",
                    "space_context",
                }:
                    continue
                if role != "system" and source_role == "system":
                    continue
                if {role, source_role} == {"time_context", "space_context"}:
                    continue
                if role == "system" and source_role != "system":
                    mark = "-" + mark[1] + ">"
                kept[source, lag] = mark
            result[target] = kept
        return result

    def _phase_assumptions(self, links, phase):
        """Freeze discovered context parents during the next search phase."""
        result = {}
        for target in range(self.N):
            incoming = {}
            for node, mark in links[target].items():
                source = node[0]
                if (
                    (phase == "context" and source not in self._dummies)
                    or (
                        phase == "dummy"
                        and target in self.system_nodes
                        and source not in self._contexts
                    )
                    or (
                        phase == "system"
                        and target in self.system_nodes
                        and source in self.system_nodes
                    )
                ):
                    incoming[node] = mark
            if phase == "dummy" or (phase == "system" and target in self.system_nodes):
                incoming.update(
                    dict.fromkeys(self.observed_context_parents[target], "-->")
                )
            if phase == "system" and target in self.system_nodes:
                incoming.update(dict.fromkeys(self.dummy_parents[target], "-->"))
            result[target] = incoming
        return self._set_link_assumptions(result, 0, self._tau_max)

    def _skeleton_phase(self, parents, assumptions, options):
        return self._pcmciplus_mci_skeleton_phase(
            parents,
            assumptions,
            options["pc_alpha"],
            options["tau_min"],
            options["tau_max"],
            options["max_conds_dim"],
            None,
            options["max_conds_py"],
            options["max_conds_px"],
            options["max_conds_px_lagged"],
            options["reset_lagged_links"],
            options["fdr_method"],
            self.p_matrix,
            self.val_matrix,
        )

    def _found_parents(self, graph, sources, targets):
        parents = self._empty_parents()
        for target in targets:
            for source in sources:
                for lag in range(graph.shape[2]):
                    if graph[source, target, lag] in {"o?o", "-?>", "o-o", "-->"}:
                        parents[target].append((source, -lag))
        return parents

    @staticmethod
    def _restore_role_statistics(
        destination, previous, nodes, targets, *, all_p_values=False
    ):
        for key in ("p_matrix", "val_matrix"):
            for node in nodes:
                if key == "p_matrix" and all_p_values:
                    destination[key][node, :, :] = previous[key][node, :, :]
                    destination[key][:, node, :] = previous[key][:, node, :]
                    continue
                for target in targets:
                    destination[key][node, target, :] = previous[key][node, target, :]
                    destination[key][target, node, :] = previous[key][target, node, :]

    def run_jpcmciplus(
        self,
        contemp_collider_rule="majority",
        link_assumptions=None,
        tau_min=0,
        tau_max=1,
        pc_alpha=0.01,
        conflict_resolution=True,
        reset_lagged_links=False,
        max_conds_dim=None,
        max_combinations=1,
        max_conds_py=None,
        max_conds_px=None,
        max_conds_px_lagged=None,
        fdr_method="none",
    ):
        """Run lagged screening, three MCI searches, colliders and orientation."""
        self._check_tau_limits(tau_min, tau_max)
        if tau_min != 0:
            msg = "J-PCMCI+ context discovery requires tau_min=0"
            raise ValueError(msg)
        if contemp_collider_rule not in {"majority", "conservative", "none"}:
            msg = "Invalid collider rule"
            raise ValueError(msg)
        if fdr_method not in {None, "none"}:
            msg = "Native J-PCMCI+ supports fdr_method=none; FDR-adjusted graph search is not implemented"
            raise ValueError(msg)
        options = {
            "contemp_collider_rule": contemp_collider_rule,
            "link_assumptions": link_assumptions,
            "tau_min": tau_min,
            "tau_max": tau_max,
            "pc_alpha": pc_alpha,
            "conflict_resolution": conflict_resolution,
            "reset_lagged_links": reset_lagged_links,
            "max_conds_dim": max_conds_dim,
            "max_combinations": max_combinations,
            "max_conds_py": max_conds_py,
            "max_conds_px": max_conds_px,
            "max_conds_px_lagged": max_conds_px_lagged,
            "fdr_method": fdr_method,
        }
        if pc_alpha is None or isinstance(pc_alpha, (list, tuple, np.ndarray)):
            candidates = (
                [0.001, 0.005, 0.01, 0.025, 0.05]
                if pc_alpha is None
                else list(pc_alpha)
            )
            if not candidates:
                msg = "At least one candidate alpha is required"
                raise ValueError(msg)
            best = None
            for alpha in candidates:
                result = self.run_jpcmciplus(**{**options, "pc_alpha": alpha})
                score = sum(
                    self.cond_ind_test.get_model_selection_criterion(
                        j,
                        self.all_lagged_parents[j]
                        + self.observed_context_parents[j]
                        + self.dummy_parents[j],
                        tau_max,
                    )
                    for j in self.system_nodes
                )
                if best is None or score < best[0]:
                    state = {
                        k: deepcopy(getattr(self, k))
                        for k in (
                            "all_lagged_parents",
                            "observed_context_parents",
                            "dummy_parents",
                            "all_parents",
                        )
                    }
                    best = score, deepcopy(result), state, alpha
            selected = cast("tuple[float, dict[str, Any], dict[str, Any], float]", best)
            for name, value in selected[2].items():
                setattr(self, name, value)
            self.results = selected[1]
            self.results["optimal_alpha"] = selected[3]
            return self.results
        if not np.isscalar(pc_alpha) or not 0 <= cast("float", pc_alpha) <= 1:
            msg = "Choose 0 <= pc_alpha <= 1"
            raise ValueError(msg)
        self._tau_max = tau_max
        self.observed_context_parents = self._empty_parents()
        self.dummy_parents = self._empty_parents()
        self.diagnostics = []
        self.mode = "system_search"
        links = self._joint_assumptions(link_assumptions, tau_min, tau_max)
        self._search_link_assumptions = deepcopy(links)
        lagged = self.run_pc_stable(
            link_assumptions=links,
            tau_min=tau_min,
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            max_conds_dim=max_conds_dim,
            max_combinations=max_combinations,
        )
        self.mode = "context_search"
        contexts = self._skeleton_phase(
            lagged, self._phase_assumptions(links, "context"), options
        )
        self.observed_context_parents = self._found_parents(
            contexts["graph"],
            self.space_context_nodes + self.time_context_nodes,
            self.system_nodes + self._contexts,
        )
        self.all_lagged_parents = {
            j: [node for node in lagged[j] if node[0] not in self.time_context_nodes]
            for j in range(self.N)
        }
        dummy_results = contexts
        if self._dummies:
            self.mode = "dummy_search"
            dummy_conditions = {
                j: list(
                    dict.fromkeys(
                        self.observed_context_parents[j] + self.all_lagged_parents[j]
                    )
                )
                for j in range(self.N)
            }
            dummy_results = self._skeleton_phase(
                dummy_conditions, self._phase_assumptions(links, "dummy"), options
            )
            self.dummy_parents = self._found_parents(
                dummy_results["graph"], self._dummies, self.system_nodes
            )
            self._restore_role_statistics(
                dummy_results, contexts, self._contexts, self.system_nodes
            )
        self.mode = "system_search"
        conditions = {
            j: list(
                dict.fromkeys(
                    self.all_lagged_parents[j]
                    + self.observed_context_parents[j]
                    + self.dummy_parents[j]
                )
            )
            if j in self.system_nodes
            else []
            for j in range(self.N)
        }
        system = self._skeleton_phase(
            conditions, self._phase_assumptions(links, "system"), options
        )
        colliders = self._pcmciplus_collider_phase(
            system["graph"],
            system["sepsets"],
            conditions,
            pc_alpha,
            tau_min,
            tau_max,
            max_conds_py,
            max_conds_px,
            max_conds_px_lagged,
            conflict_resolution,
            contemp_collider_rule,
        )
        graph = self._pcmciplus_rule_orientation_phase(
            colliders["graph"], colliders["ambiguous_triples"], conflict_resolution
        )
        self._restore_role_statistics(
            system,
            dummy_results,
            self._contexts + self._dummies,
            self.system_nodes,
            all_p_values=True,
        )
        self.results = {
            "graph": graph,
            "p_matrix": system["p_matrix"],
            "val_matrix": system["val_matrix"],
            "sepsets": colliders["sepsets"],
            "ambiguous_triples": colliders["ambiguous_triples"],
            "conf_matrix": None,
        }
        return self.results

    def _remaining_pairs(self, graph, adjt, tau_min, tau_max, p):
        pairs = super()._remaining_pairs(graph, adjt, tau_min, tau_max, p)
        if self.mode == "context_search":
            return [pair for pair in pairs if pair[0] in self._contexts]
        if self.mode == "dummy_search":
            return [pair for pair in pairs if pair[0] in self._dummies]
        return pairs

    def _run_pcalg_test(
        self,
        graph,
        i,
        abstau,
        j,
        S,
        lagged_parents,
        max_conds_py,
        max_conds_px,
        max_conds_px_lagged,
        tau_max,
        alpha_or_thres=None,
    ):
        conditions = list(S)
        if self.mode in {"dummy_search", "system_search"}:
            if self.mode == "system_search":
                conditions.extend(self.dummy_parents[j])
            conditions.extend(self.observed_context_parents[j])
        return super()._run_pcalg_test(
            graph,
            i,
            abstau,
            j,
            list(dict.fromkeys(conditions)),
            lagged_parents,
            max_conds_py,
            max_conds_px,
            max_conds_px_lagged,
            tau_max,
            alpha_or_thres,
        )
