"""Independent PC1 followed by momentary conditional independence (PCMCI).

The two-stage procedure follows Runge et al. (Science Advances, 2019),
supplement Algorithms S1--S2: estimate lagged parent supersets, then test
each source against its target given target and time-shifted source parents.
The search, lag alignment and CI implementations are local, independent code.
"""

from numbers import Real
from typing import cast

import numpy as np

from ._joint_search import JointSearch, _unique
from ._mci import adjust_pvalues, symmetrize_evidence


class PCMCIPipeline(JointSearch):
    """Native PCMCI with separate parent-selection and MCI thresholds."""

    implementation = "Native PCMCI (Runge et al. 2019)"

    @staticmethod
    def _check_alpha(alpha_level):
        if (
            isinstance(alpha_level, (bool, np.bool_))
            or not isinstance(alpha_level, Real)
            or not 0 <= cast("float", alpha_level) <= 1
        ):
            msg = "alpha_level must be a scalar in [0, 1]"
            raise ValueError(msg)

    def _matrix_shape(self, matrix, tau_max):
        array = np.asarray(matrix, dtype=float)
        if array.shape != (self.N, self.N, tau_max + 1):
            msg = "Matrix shape must be (N, N, tau_max + 1)"
            raise ValueError(msg)
        return array

    def symmetrize_p_and_val_matrix(
        self, p_matrix, val_matrix, link_assumptions, conf_matrix=None
    ):
        return symmetrize_evidence(p_matrix, val_matrix, link_assumptions, conf_matrix)

    def get_corrected_pvalues(
        self,
        p_matrix,
        fdr_method="fdr_bh",
        exclude_contemporaneous=True,
        tau_min=0,
        tau_max=1,
        link_assumptions=None,
    ):
        self._check_tau_limits(tau_min, tau_max)
        matrix = self._matrix_shape(p_matrix, tau_max)
        assumptions = (
            None
            if link_assumptions is None
            else self._set_link_assumptions(link_assumptions, tau_min, tau_max)
        )
        return adjust_pvalues(
            matrix,
            fdr_method=fdr_method,
            exclude_contemporaneous=exclude_contemporaneous,
            link_assumptions=assumptions,
        )

    def get_graph_from_pmatrix(
        self, p_matrix, alpha_level, tau_min, tau_max, link_assumptions=None
    ):
        """Threshold final MCI evidence while retaining required candidates."""
        self._check_tau_limits(tau_min, tau_max)
        self._check_alpha(alpha_level)
        matrix = self._matrix_shape(p_matrix, tau_max)
        assumptions = self._set_link_assumptions(link_assumptions, tau_min, tau_max)
        graph = np.full(matrix.shape, "", dtype="<U3")
        for j, candidates in assumptions.items():
            for (i, lag), mark in candidates.items():
                if "?" not in mark or matrix[i, j, -lag] <= alpha_level:
                    graph[i, j, -lag] = mark.replace("?", "-")
        return graph

    def _mci_conditions(self, source, target, parents, max_conds_py, max_conds_px):
        """Cap ordered parent lists, shift the source's list, and deduplicate."""
        i, lag = source
        target_parents = list(parents.get(target, []))
        source_parents = list(parents.get(i, []))
        if max_conds_py is not None:
            target_parents = target_parents[:max_conds_py]
        if max_conds_px is not None:
            source_parents = source_parents[:max_conds_px]
        conditions = [node for node in target_parents if node != source]
        conditions += [(node, parent_lag + lag) for node, parent_lag in source_parents]
        return [
            node for node in _unique(conditions) if node not in {source, (target, 0)}
        ]

    def _run_mci_target(
        self,
        j,
        candidates,
        parents,
        tau_max,
        max_conds_py,
        max_conds_px,
        alpha_level,
        confidence_enabled,
    ):
        """Collect raw evidence for one target before global MCI finalization."""
        evidence = []
        for source, mark in candidates.items():
            if mark in {"<?-", "<--"}:
                continue
            i, lag = source
            conditions = self._mci_conditions(
                source, j, parents, max_conds_py, max_conds_px
            )
            fixed = "?" not in mark
            if fixed and not self._measure_required(i, j, -lag):
                value, pvalue = 1.0, 0.0
            else:
                outcome = self.cond_ind_test.run_test(
                    X=[source],
                    Y=[(j, 0)],
                    Z=conditions,
                    tau_max=tau_max,
                    alpha_or_thres=alpha_level,
                )
                value, pvalue = outcome[:2]
                if fixed and pvalue > alpha_level:
                    self._record_preserved(i, j, -lag, pvalue)
            if not np.isfinite(value) or not np.isfinite(pvalue) or pvalue < 0:
                msg = "Undefined MCI result: test values must be finite"
                raise ValueError(msg)
            # A custom CI may reuse its interval buffer on the next call.
            interval = (
                np.array(
                    self.cond_ind_test.get_confidence(
                        X=[source], Y=[(j, 0)], Z=conditions, tau_max=tau_max
                    ),
                    copy=True,
                )
                if confidence_enabled
                else None
            )
            evidence.append((i, lag, value, pvalue, interval))
        return evidence

    def _iter_mci_results(self, assumptions, parents, **parameters):
        for j, candidates in assumptions.items():
            yield j, self._run_mci_target(j, candidates, parents, **parameters)

    def run_mci(
        self,
        selected_links=None,
        link_assumptions=None,
        tau_min=0,
        tau_max=1,
        parents=None,
        max_conds_py=None,
        max_conds_px=None,
        val_only=False,
        alpha_level=0.05,
        fdr_method="none",
    ):
        """Test all allowed links, including candidates removed by PC1.

        PC1 selects conditions; it does not restrict the MCI hypothesis family.
        For lag tau, source parents at lag s become conditions at lag s-tau.
        Required BK links retain their measured evidence and are preserved by
        graph thresholding. Hard assumptions without BK use their fixed mark.
        """
        if selected_links is not None:
            msg = "Use link_assumptions instead of selected_links"
            raise ValueError(msg)
        self._check_tau_limits(tau_min, tau_max)
        self._check_alpha(alpha_level)
        if fdr_method not in {None, "none", "fdr_bh"}:
            msg = "Supported FDR methods are none and fdr_bh"
            raise ValueError(msg)
        for limit in (max_conds_py, max_conds_px):
            if limit is not None:
                self._set_max_condition_dim(limit, 0, 0)
        assumptions = self._set_link_assumptions(link_assumptions, tau_min, tau_max)
        parents = {} if parents is None else parents
        probabilities = np.ones((self.N, self.N, tau_max + 1))
        values = np.zeros_like(probabilities)
        confidence_enabled = bool(getattr(self.cond_ind_test, "confidence", None))
        confidence = np.zeros((*probabilities.shape, 2)) if confidence_enabled else None
        if confidence_enabled and not hasattr(self.cond_ind_test, "get_confidence"):
            msg = "The CI test must implement get_confidence to request confidence intervals"
            raise ValueError(msg)
        self.mode = "mci"
        for j, evidence in self._iter_mci_results(
            assumptions,
            parents,
            tau_max=tau_max,
            max_conds_py=max_conds_py,
            max_conds_px=max_conds_px,
            alpha_level=alpha_level,
            confidence_enabled=confidence_enabled,
        ):
            for i, lag, value, pvalue, interval in evidence:
                values[i, j, -lag] = value
                probabilities[i, j, -lag] = 1.0 if val_only else pvalue
                if confidence_enabled:
                    cast("np.ndarray", confidence)[i, j, -lag] = interval
        result = self.symmetrize_p_and_val_matrix(
            probabilities, values, assumptions, confidence
        )
        if val_only:
            return {
                "val_matrix": result["val_matrix"],
                "conf_matrix": result["conf_matrix"],
            }
        result["p_matrix"] = self.get_corrected_pvalues(
            result["p_matrix"],
            fdr_method=fdr_method,
            tau_min=tau_min,
            tau_max=tau_max,
            link_assumptions=assumptions,
        )
        result["graph"] = self.get_graph_from_pmatrix(
            result["p_matrix"], alpha_level, tau_min, tau_max, assumptions
        )
        self.results = result
        return result

    def run_pcmci(
        self,
        selected_links=None,
        link_assumptions=None,
        tau_min=0,
        tau_max=1,
        save_iterations=False,
        pc_alpha=0.2,
        max_conds_dim=None,
        max_combinations=1,
        max_conds_py=None,
        max_conds_px=None,
        alpha_level=0.05,
        fdr_method="none",
    ):
        """Estimate PC1 parent supersets and apply final MCI significance."""
        self._check_alpha(alpha_level)
        if fdr_method not in {None, "none", "fdr_bh"}:
            msg = "Supported FDR methods are none and fdr_bh"
            raise ValueError(msg)
        self.diagnostics = []
        self.mode = "pc1"
        parents = self.run_pc_stable(
            selected_links=selected_links,
            link_assumptions=link_assumptions,
            tau_min=tau_min,
            tau_max=tau_max,
            save_iterations=save_iterations,
            pc_alpha=pc_alpha,
            max_conds_dim=max_conds_dim,
            max_combinations=max_combinations,
        )
        return self.run_mci(
            link_assumptions=link_assumptions,
            tau_min=tau_min,
            tau_max=tau_max,
            parents=parents,
            max_conds_py=max_conds_py,
            max_conds_px=max_conds_px,
            alpha_level=alpha_level,
            fdr_method=fdr_method,
        )
