"""Independent search primitives for the local J-PCMCI+ engine.

The algorithms here are written from the PC1 and PCMCI+ descriptions in
Runge (UAI 2020, Algorithms 1, 2 and the orientation rules) and the staged
J-PCMCI+ search in Günther et al. (UAI 2023).
A node is represented by ``(variable, lag)``;
lags in conditioning sets are nonpositive.
"""

from copy import deepcopy
from itertools import combinations
from numbers import Integral
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from .._bk import TemporalKnowledge


def _unique(nodes):
    """Preserve conditioning order while removing duplicates."""
    return list(dict.fromkeys(tuple(node) for node in nodes))


def _pc_alpha_candidates(pc_alpha):
    if pc_alpha is None:
        return [0.05, 0.1, 0.2, 0.3, 0.4, 0.5], True
    if isinstance(pc_alpha, (list, tuple, np.ndarray)):
        return list(pc_alpha), True
    return [pc_alpha], False


class JointSearch:
    """Stable parent selection, MCI skeleton search and CPDAG orientation."""

    _background_knowledge: "TemporalKnowledge"
    iterations: dict[int, Any]
    val_min: dict[int, Any]
    pval_max: dict[int, Any]

    def __init__(self, dataframe, cond_ind_test, verbosity=0):
        self.dataframe = dataframe
        self.N = dataframe.N
        self.var_names = dataframe.var_names
        self.verbosity = verbosity
        self.cond_ind_test = deepcopy(cond_ind_test)
        self.cond_ind_test.set_dataframe(dataframe)
        self._orientation_locks = set()
        self.diagnostics = []

    @staticmethod
    def _check_tau_limits(tau_min, tau_max):
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            for value in (tau_min, tau_max)
        ):
            msg = "tau_min and tau_max must be nonnegative integers"
            raise ValueError(msg)
        if tau_min < 0 or tau_max < tau_min:
            msg = "Require 0 <= tau_min <= tau_max"
            raise ValueError(msg)

    def _set_max_condition_dim(self, max_conds_dim, tau_min, tau_max):
        if max_conds_dim is None:
            return max(0, self.N * (tau_max - tau_min + 1))
        if (
            isinstance(max_conds_dim, (bool, np.bool_))
            or not isinstance(max_conds_dim, Integral)
            or max_conds_dim < 0
        ):
            msg = "Condition dimensions must be nonnegative integers or None"
            raise ValueError(msg)
        return int(max_conds_dim)

    @staticmethod
    def _reverse(mark):
        return mark[::-1].translate(str.maketrans("<>", "><"))

    def _set_link_assumptions(
        self, link_assumptions, tau_min, tau_max, remove_contemp=False
    ):
        """Normalize a sparse candidate map, including one-sided zero-lag links."""
        links: dict[int, dict[tuple[int, int], str]] = {j: {} for j in range(self.N)}
        if link_assumptions is None:
            for j in range(self.N):
                for i in range(self.N):
                    for lag in range(tau_min, tau_max + 1):
                        if lag or i != j:
                            links[j][(i, -lag)] = "-?>" if lag else "o?o"
        else:
            valid = {"o?o", "o-o", "-?>", "-->", "<?-", "<--"}
            for j, candidates in link_assumptions.items():
                if not isinstance(j, Integral) or j not in links:
                    msg = "Unknown target in link_assumptions"
                    raise ValueError(msg)
                for node, mark in candidates.items():
                    if len(node) != 2:
                        msg = "Link nodes must be (variable, nonpositive lag)"
                        raise ValueError(msg)
                    i, lag = node
                    if remove_contemp and lag == 0:
                        continue
                    if (
                        not isinstance(i, Integral)
                        or not 0 <= cast("int", i) < self.N
                        or not isinstance(lag, Integral)
                        or cast("int", lag) > 0
                        or not tau_min <= -cast("int", lag) <= tau_max
                        or (lag == 0 and i == j)
                        or mark not in valid
                    ):
                        msg = "Invalid link assumption"
                        raise ValueError(msg)
                    if lag and mark not in {"-?>", "-->"}:
                        msg = "Lagged candidates must point forward in time"
                        raise ValueError(msg)
                    links[cast("int", j)][(int(i), int(lag))] = mark
            for j in range(self.N):
                for (i, lag), mark in list(links[j].items()):
                    if lag:
                        continue
                    reverse = self._reverse(mark)
                    supplied = links[i].get((j, 0))
                    if supplied is not None and supplied != reverse:
                        msg = "Contradictory contemporaneous link assumptions"
                        raise ValueError(msg)
                    links[i][(j, 0)] = reverse
        if remove_contemp:
            links = {
                j: {node: mark for node, mark in parents.items() if node[1] < 0}
                for j, parents in links.items()
            }
        else:
            edges = {
                (i, j)
                for j, incoming in links.items()
                for (i, lag), mark in incoming.items()
                if lag == 0 and mark in {"-->", "-?>"}
            }
            remaining = set(range(self.N))
            while remaining:
                roots = {
                    node
                    for node in remaining
                    if not any(
                        target == node and source in remaining
                        for source, target in edges
                    )
                }
                if not roots:
                    msg = "Contemporaneous link assumptions contain a directed cycle"
                    raise ValueError(msg)
                remaining -= roots
        return links

    def _measure_required(self, i, j, lag):
        knowledge = getattr(self, "_background_knowledge", None)
        if knowledge is None or not knowledge.protected(i, j, lag):
            return False
        role = getattr(knowledge, "roles", {}).get(i, "system")
        stage = getattr(self, "mode", "")
        if stage == "system_search" and role != "system":
            return False
        return not (
            stage == "dummy_search" and role in {"time_context", "space_context"}
        )

    def _record_preserved(self, i, j, lag, pvalue):
        self.diagnostics.append(
            {
                "source": i,
                "target": j,
                "lag": lag,
                "stage": getattr(self, "mode", "pc1"),
                "p_value": float(pvalue),
                "reason": "required_link",
            }
        )

    def _run_pc_stable_single(
        self,
        j,
        link_assumptions_j,
        tau_min=1,
        tau_max=1,
        save_iterations=False,
        pc_alpha=0.2,
        max_conds_dim=None,
        max_combinations=1,
    ):
        """PC1: freeze candidates at each depth and rank by minimum effect size."""
        parents = [
            node
            for node, mark in link_assumptions_j.items()
            if node[1] < 0 and mark in {"-?>", "-->"}
        ]
        minimum = dict.fromkeys(parents, np.inf)
        strengths = dict.fromkeys(parents, 0.0)
        largest_p = dict.fromkeys(parents, 0.0)
        history: dict[int, dict[tuple[int, int], dict[int, dict[str, Any]]]] = {}
        limit = self._set_max_condition_dim(max_conds_dim, tau_min, tau_max)
        for depth in range(limit + 1):
            if len(parents) <= depth:
                break
            remove = set()
            frozen = list(parents)
            for node in frozen:
                others = [other for other in frozen if other != node]
                for q, S in enumerate(combinations(others, depth)):
                    if q >= max_combinations:
                        break
                    i, lag = node
                    required = link_assumptions_j[node] == "-->"
                    if required and not self._measure_required(i, j, -lag):
                        value, pvalue, dependent = 1.0, 0.0, True
                    else:
                        value, pvalue, dependent = self.cond_ind_test.run_test(
                            X=[node],
                            Y=[(j, 0)],
                            Z=list(S),
                            tau_max=tau_max,
                            alpha_or_thres=pc_alpha,
                        )
                    minimum[node] = min(minimum[node], abs(value))
                    if pvalue > largest_p[node]:
                        largest_p[node], strengths[node] = pvalue, value
                    elif q == 0 and depth == 0 and pvalue == 0:
                        strengths[node] = value
                    if save_iterations:
                        history.setdefault(depth, {}).setdefault(node, {})[q] = {
                            "conds": list(S),
                            "val": value,
                            "pval": pvalue,
                        }
                    if not dependent:
                        if required:
                            self._record_preserved(i, j, -lag, pvalue)
                        else:
                            remove.add(node)
                            break
            parents = [node for node in parents if node not in remove]
            parents.sort(key=lambda node: -minimum[node])
        return {
            "parents": parents,
            "val_min": {node: minimum[node] for node in parents},
            "val_dict": strengths,
            "pval_max": largest_p,
            "iterations": history,
        }

    def _collect_pc1_results(
        self, selections, tau_max, *, val_min, pval_max, iterations
    ):
        parents = {}
        strengths = np.zeros((self.N, self.N, tau_max + 1))
        probabilities = np.ones_like(strengths)
        for target, choice, optimal_alpha in selections:
            parents[target] = choice["parents"]
            val_min[target] = choice["val_min"]
            pval_max[target] = choice["pval_max"]
            iterations[target] = choice["iterations"]
            if optimal_alpha is not None:
                iterations[target]["optimal_pc_alpha"] = optimal_alpha
            for (source, lag), value in choice["val_dict"].items():
                strengths[source, target, -lag] = value
            for (source, lag), value in pval_max[target].items():
                probabilities[source, target, -lag] = value
        self.val_matrix, self.p_matrix = strengths, probabilities
        self.val_min, self.pval_max, self.iterations = val_min, pval_max, iterations
        self.all_parents = parents
        return parents

    def run_pc_stable(
        self,
        selected_links=None,
        link_assumptions=None,
        tau_min=1,
        tau_max=1,
        save_iterations=False,
        pc_alpha=0.2,
        max_conds_dim=None,
        max_combinations=1,
    ):
        if selected_links is not None:
            msg = "Use link_assumptions instead of selected_links"
            raise ValueError(msg)
        self._check_tau_limits(tau_min, tau_max)
        if max_combinations <= 0:
            msg = "max_combinations must be > 0"
            raise ValueError(msg)
        alphas, optimize = _pc_alpha_candidates(pc_alpha)
        if not alphas or any(not 0 <= alpha <= 1 for alpha in alphas):
            msg = "pc_alpha must be in [0, 1]"
            raise ValueError(msg)
        assumptions = self._set_link_assumptions(
            link_assumptions, max(1, tau_min), tau_max, remove_contemp=True
        )

        def selected_targets():
            for j in range(self.N):
                choices, scores = [], []
                for alpha in alphas:
                    result = self._run_pc_stable_single(
                        j,
                        assumptions[j],
                        max(1, tau_min),
                        tau_max,
                        save_iterations,
                        alpha,
                        max_conds_dim,
                        max_combinations,
                    )
                    choices.append(result)
                    scores.append(
                        self.cond_ind_test.get_model_selection_criterion(
                            j, result["parents"], tau_max
                        )
                        if optimize
                        else 0.0
                    )
                index = int(np.argmin(scores))
                yield j, choices[index], alphas[index] if optimize else None

        self.val_min, self.pval_max, self.iterations = {}, {}, {}
        return self._collect_pc1_results(
            selected_targets(),
            tau_max,
            val_min=self.val_min,
            pval_max=self.pval_max,
            iterations=self.iterations,
        )

    def _get_adj_time_series(self, graph, sort_by=None):
        adjacency = {}
        for j in range(self.N):
            nodes = [
                (i, -lag)
                for i in range(self.N)
                for lag in range(graph.shape[2])
                if graph[i, j, lag] != ""
            ]
            if sort_by is not None:
                nodes.sort(key=lambda node: -abs(sort_by[j].get(node, 0.0)))
            adjacency[j] = nodes
        return adjacency

    def _get_adj_time_series_contemp(self, graph, sort_by=None):
        adjacency = self._get_adj_time_series(graph[:, :, :1], sort_by)
        return adjacency

    def _get_sepsets(self, tau_min, tau_max):
        return {
            ((i, -lag), j): []
            for i in range(self.N)
            for j in range(self.N)
            for lag in range(tau_min, tau_max + 1)
        }

    def _remaining_pairs(self, graph, adjt, tau_min, tau_max, p):
        return [
            (i, j, lag)
            for i in range(self.N)
            for j in range(self.N)
            for lag in range(tau_min, tau_max + 1)
            if graph[i, j, lag] != ""
            and sum(node != (i, -lag) for node in adjt[j]) >= p
        ]

    def _any_tests_remaining(self, graph, adjt, tau_min, tau_max, p):
        return bool(self._remaining_pairs(graph, adjt, tau_min, tau_max, p))

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
        for limit in (max_conds_py, max_conds_px, max_conds_px_lagged):
            if limit is not None:
                self._set_max_condition_dim(limit, 0, 0)
        parents = lagged_parents or {}
        source = (i, -abstau)
        target_parents = list(parents.get(j, []))
        source_parents = list(parents.get(i, []))
        if max_conds_py is not None:
            target_parents = target_parents[:max_conds_py]
        source_limit = (
            max_conds_px_lagged
            if abstau and max_conds_px_lagged is not None
            else max_conds_px
        )
        if source_limit is not None:
            source_parents = source_parents[:source_limit]
        shifted = [(k, lag - abstau) for k, lag in source_parents]
        Z = _unique(
            list(S) + [node for node in target_parents if node != source] + shifted
        )
        Z = [node for node in Z if node not in {source, (j, 0)}]
        required = "?" not in graph[i, j, abstau]
        if required and not self._measure_required(i, j, abstau):
            return 1.0, 0.0, Z, True
        value, pvalue, dependent = self.cond_ind_test.run_test(
            X=[source], Y=[(j, 0)], Z=Z, tau_max=tau_max, alpha_or_thres=alpha_or_thres
        )
        if required and not dependent:
            self._record_preserved(i, j, abstau, pvalue)
            dependent = True
        return value, pvalue, Z, dependent

    def _initial_skeleton(self, initial_graph, tau_max):
        if initial_graph is None:
            graph = np.full((self.N, self.N, tau_max + 1), "-?>", dtype="<U3")
            graph[:, :, 0] = "o?o"
        else:
            graph = np.array(initial_graph, dtype="<U3", copy=True)
        graph[np.arange(self.N), np.arange(self.N), 0] = ""
        present = graph != ""
        minimum = {
            j: {
                (i, -lag): np.inf
                for i, lag in zip(*np.nonzero(present[:, j, :]), strict=False)
            }
            for j in range(self.N)
        }
        return graph, np.where(present, 0.0, 1.0), np.zeros(graph.shape), minimum

    def _pcalg_skeleton(
        self,
        initial_graph,
        lagged_parents,
        mode,
        pc_alpha,
        tau_min,
        tau_max,
        max_conds_dim,
        max_combinations,
        max_conds_py,
        max_conds_px,
        max_conds_px_lagged,
    ):
        """Stable skeleton: only adjacency is frozen, removals suppress reverse tests."""
        graph, probabilities, values, minimum = self._initial_skeleton(
            initial_graph, tau_max
        )
        adjacency_fn = (
            self._get_adj_time_series_contemp
            if mode == "contemp_conds"
            else self._get_adj_time_series
        )
        adjacency = adjacency_fn(graph)
        separators = self._get_sepsets(tau_min, tau_max)
        max_conds_dim = self._set_max_condition_dim(max_conds_dim, tau_min, tau_max)
        max_combinations = np.inf if max_combinations is None else max_combinations
        depth = 0
        while depth <= max_conds_dim and self._any_tests_remaining(
            graph, adjacency, tau_min, tau_max, depth
        ):
            for i, j, lag in self._remaining_pairs(
                graph, adjacency, tau_min, tau_max, depth
            ):
                if graph[i, j, lag] == "":
                    continue
                candidates = [node for node in adjacency[j] if node != (i, -lag)]
                for q, S in enumerate(combinations(candidates, depth)):
                    if q > max_combinations:
                        break
                    value, pvalue, _, dependent = self._run_pcalg_test(
                        graph,
                        i,
                        lag,
                        j,
                        S,
                        lagged_parents,
                        max_conds_py,
                        max_conds_px,
                        max_conds_px_lagged,
                        tau_max,
                        pc_alpha,
                    )
                    minimum[j][(i, -lag)] = min(minimum[j][(i, -lag)], abs(value))
                    if pvalue >= probabilities[i, j, lag]:
                        probabilities[i, j, lag], values[i, j, lag] = pvalue, value
                    if not dependent:
                        graph[i, j, lag] = ""
                        separators[((i, -lag), j)] = list(S)
                        if lag == 0:
                            graph[j, i, 0] = ""
                            separators[((j, 0), i)] = list(S)
                            probabilities[j, i, 0] = probabilities[i, j, 0]
                        break
            depth += 1
            adjacency = adjacency_fn(graph, sort_by=minimum)
        return {
            "graph": graph,
            "sepsets": separators,
            "p_matrix": probabilities,
            "val_matrix": values,
        }

    def _pcmciplus_mci_skeleton_phase(
        self,
        lagged_parents,
        link_assumptions,
        pc_alpha,
        tau_min,
        tau_max,
        max_conds_dim,
        max_combinations,
        max_conds_py,
        max_conds_px,
        max_conds_px_lagged,
        reset_lagged_links,
        fdr_method,
        p_matrix,
        val_matrix,
    ):
        if fdr_method not in {None, "none"}:
            msg = "The native J-PCMCI+ engine currently supports fdr_method=none only"
            raise ValueError(msg)
        for limit in (max_conds_py, max_conds_px, max_conds_px_lagged):
            if limit is not None:
                self._set_max_condition_dim(limit, 0, 0)
        graph = np.full((self.N, self.N, tau_max + 1), "", dtype="<U3")
        self._orientation_locks = set()
        for j, candidates in link_assumptions.items():
            for (i, lag), mark in candidates.items():
                if (
                    lag == 0
                    or reset_lagged_links
                    or (i, lag) in lagged_parents.get(j, [])
                    or mark == "-->"
                ):
                    graph[i, j, -lag] = mark
                if lag == 0 and mark in {"-->", "-?>"}:
                    self._orientation_locks.add((i, j))
        initial = graph.copy()
        # PC1-pruned lagged links have CI evidence; links excluded by input
        # assumptions do not. Only the latter must not act as empty separators.
        self._prior_absent = np.ones(graph.shape, dtype=bool)
        original_candidates = getattr(
            self, "_search_link_assumptions", link_assumptions
        )
        for j, candidates in original_candidates.items():
            for i, lag in candidates:
                self._prior_absent[i, j, -lag] = False
        result = self._pcalg_skeleton(
            graph,
            lagged_parents,
            "contemp_conds",
            pc_alpha,
            tau_min,
            tau_max,
            self._set_max_condition_dim(max_conds_dim, 0, 0),
            np.inf if max_combinations is None else max_combinations,
            max_conds_py,
            max_conds_px,
            max_conds_px_lagged,
        )
        for lag in range(1, tau_max + 1):
            untouched = initial[:, :, lag] == ""
            result["p_matrix"][:, :, lag][untouched] = p_matrix[:, :, lag][untouched]
            result["val_matrix"][:, :, lag][untouched] = val_matrix[:, :, lag][
                untouched
            ]
        for i in range(self.N):
            for j in range(i + 1, self.N):
                a, b = result["p_matrix"][i, j, 0], result["p_matrix"][j, i, 0]
                if initial[i, j, 0] in {"-->", "-?>"}:
                    source, target = i, j
                elif initial[j, i, 0] in {"-->", "-?>"}:
                    source, target = j, i
                else:
                    source, target = (i, j) if a >= b else (j, i)
                p = result["p_matrix"][source, target, 0]
                value = result["val_matrix"][source, target, 0]
                result["p_matrix"][i, j, 0] = result["p_matrix"][j, i, 0] = p
                result["val_matrix"][i, j, 0] = result["val_matrix"][j, i, 0] = value
        result["graph"] = np.char.replace(result["graph"], "?", "-")
        # Each stage refines the same running evidence matrices. Untested
        # links retain evidence from screening or the preceding joint stage.
        self.p_matrix, self.val_matrix = result["p_matrix"], result["val_matrix"]
        return result

    def _direction_permitted(self, i, j):
        knowledge = getattr(self, "_background_knowledge", None)
        return (j, i) not in self._orientation_locks and (
            knowledge is None
            or not hasattr(knowledge, "permits")
            or knowledge.permits(i, j, 0)
        )

    def _orient_batch(self, graph, requests, conflict_resolution):
        """Resolve all requests together, then apply hard direction constraints."""
        changed = False
        requests = {edge for edge in requests if self._direction_permitted(*edge)}
        for i, j in sorted(requests):
            if graph[i, j, 0] == "" or graph[i, j, 0] == "x-x":
                continue
            opposite = (j, i) in requests or graph[i, j, 0] == "<--"
            if opposite and conflict_resolution:
                if self._direction_permitted(j, i):
                    changed |= graph[i, j, 0] != "x-x"
                    graph[i, j, 0] = graph[j, i, 0] = "x-x"
            elif not opposite or not conflict_resolution:
                if opposite and graph[i, j, 0] == "<--":
                    continue
                if graph[i, j, 0] != "-->":
                    graph[i, j, 0], graph[j, i, 0] = "-->", "<--"
                    changed = True
        return changed

    @staticmethod
    def _triple_key(a, middle, b):
        """Canonical key for an unshielded triple at the middle node's time."""
        a, b = tuple(a), tuple(b)
        return (min(a, b), int(middle), max(a, b))

    def _unshielded_triples(self, graph):
        triples = []
        # Ordered endpoints retain both views of a zero-lag triple. Besides
        # making the CI order deterministic, this preserves endpoint-specific
        # diagnostics for asymmetric MCI conditioning.
        for j in range(self.N):
            for middle in range(self.N):
                if graph[j, middle, 0] != "o-o":
                    continue
                for i in range(self.N):
                    for lag in range(graph.shape[2]):
                        mark = graph[i, middle, lag]
                        if (lag and mark != "-->") or (
                            not lag and mark not in {"o-o", "-->"}
                        ):
                            continue
                        if (lag == 0 and i == j) or graph[i, j, lag] != "":
                            continue
                        triples.append(((i, -lag), middle, j))
        return triples

    def _pcmciplus_collider_phase(
        self,
        graph,
        sepsets,
        lagged_parents,
        pc_alpha,
        tau_min,
        tau_max,
        max_conds_py,
        max_conds_px,
        max_conds_px_lagged,
        conflict_resolution,
        contemp_collider_rule,
    ):
        if contemp_collider_rule not in {"none", "majority", "conservative"}:
            msg = "Unknown contemporaneous collider rule"
            raise ValueError(msg)
        graph = np.array(graph, copy=True)
        knowledge = getattr(self, "_background_knowledge", None)
        if knowledge is not None and hasattr(knowledge, "before_orientation"):
            graph = knowledge.before_orientation(graph)
        ambiguous, requests = [], set()
        adjacency = self._get_adj_time_series_contemp(graph)
        # All triples see the skeleton: collider discoveries cannot affect later tests.
        for source, middle, j in self._unshielded_triples(graph):
            i, negative_lag = source
            lag = -negative_lag
            if contemp_collider_rule == "none":
                knowledge = getattr(self, "_background_knowledge", None)
                structural_absence = (
                    knowledge is not None
                    and hasattr(knowledge, "adjacency_allowed")
                    and not knowledge.adjacency_allowed(i, j, lag)
                )
                prior = getattr(self, "_prior_absent", None)
                if structural_absence or (prior is not None and prior[i, j, lag]):
                    ambiguous.append((source, middle, j))
                    continue
                separators = [list(sepsets.get((source, j), []))]
            else:
                sets = []
                neighborhoods = [[node for node in adjacency[j] if node != source]]
                if lag == 0:
                    neighborhoods.append(
                        [node for node in adjacency[i] if node != (j, 0)]
                    )
                seen = set()
                for neighbors in neighborhoods:
                    for depth in range(len(neighbors) + 1):
                        for S in combinations(neighbors, depth):
                            key = frozenset(S)
                            if key not in seen:
                                seen.add(key)
                                sets.append(S)
                separators = []
                test_graph = graph.copy()
                # The endpoint link is absent. CI must still be performed instead
                # of treating its absent mark as a required edge.
                test_graph[i, j, lag] = "-?>" if lag else "o?o"
                for S in sets:
                    _, _, _, dependent = self._run_pcalg_test(
                        test_graph,
                        i,
                        lag,
                        j,
                        S,
                        lagged_parents,
                        max_conds_py,
                        max_conds_px,
                        max_conds_px_lagged,
                        tau_max,
                        pc_alpha,
                    )
                    if not dependent:
                        separators.append(list(S))
            membership = [(middle, 0) in S for S in separators]
            if not membership:
                ambiguous.append((source, middle, j))
                continue
            fraction = sum(membership) / len(membership)
            collider = (
                fraction < 0.5
                if contemp_collider_rule == "majority"
                else fraction == 0.0
            )
            noncollider = (
                fraction > 0.5
                if contemp_collider_rule == "majority"
                else fraction == 1.0
            )
            if collider:
                requests.add((j, middle))
                if lag == 0:
                    requests.add((i, middle))
                current = list(sepsets.get((source, j), []))
                sepsets[(source, j)] = [node for node in current if node != (middle, 0)]
                if lag == 0:
                    sepsets[((j, 0), i)] = list(sepsets[(source, j)])
            elif noncollider:
                sepsets[(source, j)] = _unique(
                    [*sepsets.get((source, j), []), (middle, 0)]
                )
                if lag == 0:
                    sepsets[((j, 0), i)] = list(sepsets[(source, j)])
            else:
                ambiguous.append((source, middle, j))
        self._orient_batch(graph, requests, conflict_resolution)
        return {"graph": graph, "sepsets": sepsets, "ambiguous_triples": ambiguous}

    def _pcmciplus_rule_orientation_phase(
        self, graph, ambiguous_triples, conflict_resolution
    ):
        """Close the CPDAG under time-series Meek R1–R3 and knowledge R4."""
        graph = np.array(graph, copy=True)
        ambiguous = {
            self._triple_key(a, middle, (b, 0)) for a, middle, b in ambiguous_triples
        }

        def unambiguous(a, middle, b):
            return self._triple_key(a, middle, b) not in ambiguous

        def adjacent(i, j):
            return graph[i, j, 0] != ""

        def directed(i, j):
            return graph[i, j, 0] == "-->"

        def undirected(i, j):
            return graph[i, j, 0] == "o-o"

        while True:
            requests = set()
            for i in range(self.N):
                for j in range(self.N):
                    if not undirected(i, j):
                        continue
                    # R1: a -> i o-o j, a and j nonadjacent.
                    for a in range(self.N):
                        for lag in range(graph.shape[2]):
                            if (
                                graph[a, i, lag] == "-->"
                                and graph[a, j, lag] == ""
                                and unambiguous((a, -lag), i, (j, 0))
                            ):
                                requests.add((i, j))
                    # R2: i -> k -> j.
                    if any(directed(i, k) and directed(k, j) for k in range(self.N)):
                        requests.add((i, j))
                    # R3: two nonadjacent parents of j, each undirected with i.
                    candidates = [
                        k for k in range(self.N) if undirected(i, k) and directed(k, j)
                    ]
                    for k, other in combinations(candidates, 2):
                        if not adjacent(k, other) and unambiguous(
                            (k, 0), i, (other, 0)
                        ):
                            requests.add((i, j))
                    # R4 is necessary after external direction knowledge shrinks
                    # the equivalence class: i-c -> d -> j, c !~ j, i ~ d.
                    if (
                        getattr(self, "_background_knowledge", None) is not None
                        or self._orientation_locks
                    ):
                        for c in range(self.N):
                            if not undirected(i, c) or adjacent(c, j):
                                continue
                            if not unambiguous((c, 0), i, (j, 0)):
                                continue
                            if any(
                                directed(c, d) and directed(d, j) and adjacent(i, d)
                                for d in range(self.N)
                            ):
                                requests.add((i, j))
            if not self._orient_batch(graph, requests, conflict_resolution):
                break
        return graph

    def _print_pc_params(self, *args):
        print("PC1 lagged parent selection")

    def _print_pc_sel_results(self, alphas, results, j, scores, optimal):
        print(f"Target {j}: selected pc_alpha={optimal}")

    def _print_parents(self, parents, *args):
        for j, nodes in parents.items():
            print(f"{self.var_names[j]}: {nodes}")
