"""RAI learner with lazy CUDA imports and CPU fallback for CI tests."""

import time
from typing import TYPE_CHECKING

import numpy as np

from ..standalone.rai_algo import _PDAG, _CIOracle
from ..standalone.rai_algo import RAIStableLearner as _CPUReferenceLearner
from ._timings import records_timings
from .cuda_contracts import validate_cuda_ci_test
from .independence_tests import CIT

# Instances acquire the upstream CUDA learner in _integrated_class.
if TYPE_CHECKING:
    from .cuda_rai import RAIStableCudaLearner as _BridgeBase
else:
    _BridgeBase = _CPUReferenceLearner

_SPEC_MAX_TESTS = 8_000_000
_P_RECHECK_EPS = 1e-7
_INTEGRATED_CLASS = None


def _upstream():
    from . import cuda_rai

    return cuda_rai


def _validate_cuda_ci_test(cit):
    return validate_cuda_ci_test(cit, "RAI-stable-cuda")


def _build_gpu_batcher(cit, dtype=None):
    method = _validate_cuda_ci_test(cit)
    if method == "fisherz":
        from .cpu_cuda_fisher_bridge import CudaFisherZ

        if cit._corr_matrix is not None:
            batcher = CudaFisherZ(cit.corr_matrix, cit.n_samples, dtype=dtype)
        else:
            batcher = CudaFisherZ.from_data(
                cit.data, n_samples=cit.n_samples, dtype=dtype
            )
        batcher.cpu_test = cit
        batcher.decision_alpha = getattr(cit, "decision_alpha", None)
        return batcher
    from .cuda_chisq import CudaChisqHost

    return CudaChisqHost(cit)


def _bounded_combinations(n: int, k: int, limit: int) -> int:
    """Return ``limit + 1`` once the count exceeds the budget."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    count = 1
    for i in range(1, k + 1):
        count = count * (n - k + i) // i
        if count > limit:
            return limit + 1
    return count


def _speculation_within_budget(graph, pairs, order: int, limit: int) -> bool:
    """Count candidate tests before native enumeration allocates buffers.

    Overlapping endpoint pools can trigger CPU replay early. Native replay can
    leave ``_FastPDAG._adj`` stale, so use the live neighbor sets.
    """
    total = 0
    pools = {}

    def pool_size(node, other):
        if node not in pools:
            parents = graph.parents(node)
            undirected = graph.undirected_neighbors(node)
            pools[node] = (parents, undirected, len(parents) + len(undirected))
        parents, undirected, size = pools[node]
        return size - (other in parents) - (other in undirected)

    for a, b in pairs:
        if not _PDAG.is_connected(graph, a, b):
            continue
        if order == 0:
            total += 1
        else:
            for node, other in ((a, b), (b, a)):
                total += _bounded_combinations(
                    pool_size(node, other), order, limit - total
                )
                if total > limit:
                    return False
        if total > limit:
            return False
    return True


def _integrated_class():
    global _INTEGRATED_CLASS
    if _INTEGRATED_CLASS is None:
        original = _upstream()

        _INTEGRATED_CLASS = type(
            "IntegratedRAIStableCudaLearner",
            (RAIStableCudaLearner, original.RAIStableCudaLearner),
            {"__module__": __name__},
        )
    return _INTEGRATED_CLASS


class RAIStableCudaLearner(_BridgeBase):
    """Use upstream CUDA search with CI fallback through the configured tester."""

    _bridge_prepared_ci_test: CIT

    def __new__(
        cls,
        data,
        alpha,
        indep_test,
        node_names,
        verbose=False,
        dtype=None,
        ci_test=None,
    ):
        """Allocate the learner and retain the validated CI tester for initialization."""
        tester = CIT(data, method=indep_test) if ci_test is None else ci_test
        _validate_cuda_ci_test(tester)
        concrete_class = _integrated_class() if cls is RAIStableCudaLearner else cls
        instance = object.__new__(concrete_class)
        instance._bridge_prepared_ci_test = tester
        return instance

    def __init__(
        self,
        data,
        alpha,
        indep_test,
        node_names,
        verbose=False,
        dtype=None,
        ci_test=None,
    ):
        # The upstream constructor would build its own GPU batcher.
        _CPUReferenceLearner.__init__(
            self,
            data=data,
            alpha=alpha,
            indep_test=indep_test,
            node_names=node_names,
            verbose=verbose,
        )
        self.ci_test = self.__dict__.pop("_bridge_prepared_ci_test")
        self.oracle = _CIOracle(self.ci_test, alpha)
        self._gpu = _build_gpu_batcher(self.ci_test, dtype=dtype)
        self.graph = _upstream()._FastPDAG(self.nodes_set)
        self.graph.create_complete_graph(self.nodes_set)
        self._last_orient_key = None
        self._parallel = None
        self.max_cond_set_size = None
        self.stats.update(gpu_phases=0, gpu_tests=0, gpu_borderline=0)
        self.stats["gpu_seconds"] = 0.0

    def _cpu_pvalue(self, a, b, key):
        return self.ci_test(a, b, tuple(key))

    def _ci_memo(self, a, b, cset, memo):
        key = tuple(sorted(cset))
        self.n_ci_tests += 1
        self.stats["ci_total"] += 1
        p = memo.get(a, b, key) if memo is not None else None
        if p is not None and np.isnan(p):
            p = None
        elif p is not None and abs(p - self.oracle.alpha) <= _P_RECHECK_EPS:
            p = None
            self.stats["gpu_borderline"] += 1
        if p is None:
            p = self._cpu_pvalue(a, b, key)
            if memo is not None:
                self.stats["ci_sync"] += 1
        independent = p > self.oracle.alpha
        if self.trace is not None:
            self.trace.append((a, b, key, independent))
        return independent

    def _speculate(self, pairs, order):
        if not _speculation_within_budget(self.graph, pairs, order, _SPEC_MAX_TESTS):
            return None
        return super()._speculate(pairs, order)

    def _speculate_fast(self, pairs, order):
        if not pairs or not _speculation_within_budget(
            self.graph, pairs, order, _SPEC_MAX_TESTS
        ):
            return None
        return super()._speculate_fast(pairs, order)


def _write_back_causal_graph(cg, learner):
    return _upstream()._write_back_causal_graph(cg, learner)


@records_timings
def run_rai_cuda(cg, alpha=0.05, *, dtype=None, max_cond_set_size=None):
    """Write the learned edges and separating sets into ``cg``."""
    test = cg.test
    data = getattr(test, "data", None)
    if data is None:
        msg = "run_rai_cuda needs cg.test to hold the raw data matrix (CIT)"
        raise ValueError(msg)
    learner = RAIStableCudaLearner(
        data=data,
        alpha=float(alpha),
        indep_test=test.method,
        node_names=[f"X{i + 1}" for i in range(data.shape[1])],
        verbose=False,
        dtype=dtype,
        ci_test=test,
    )
    limit = max_cond_set_size
    if limit is None:
        limit = getattr(cg, "max_cond_set_size", None)
    learner.max_cond_set_size = limit
    learner.ci_test.max_cond_set_size = limit
    start = time.perf_counter()
    final = learner.learn_structure()
    elapsed = time.perf_counter() - start
    np.copyto(cg.G.graph, np.asarray(final.graph))
    cg.sepset_S, cg.sepset_ranges = _upstream()._sepset_csr(learner, int(cg.G.num_vars))
    run_rai_cuda.last_times = {
        "total": elapsed,
        "n_ci_tests": learner.n_ci_tests,
        **{
            key: value for key, value in learner.stats.items() if key.startswith("gpu_")
        },
    }


__all__ = ["RAIStableCudaLearner", "run_rai_cuda"]
