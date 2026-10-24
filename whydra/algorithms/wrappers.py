import sys
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

try:
    from .standalone.orig_tigramite import data_processing as pp_orig
    from .standalone.orig_tigramite.jpcmciplus import JPCMCIplus as JPCMCIplusOrig
    from .standalone.orig_tigramite.independence_tests import ParCorrMult as ParCorrMultOrig
except ModuleNotFoundError as exc:
    pp_orig = None
    JPCMCIplusOrig = None
    ParCorrMultOrig = None
    ORIG_TIGRAMITE_IMPORT_ERROR = exc
else:
    ORIG_TIGRAMITE_IMPORT_ERROR = None

from .base import CausalAlgorithm
from .graph_core import CausalGraph, CIT, GeneralGraph, Node as GraphNode, Edge, Endpoint

# Хак для импорта standalone файлов
current_dir = os.path.dirname(os.path.abspath(__file__))
standalone_dir = os.path.join(current_dir, 'standalone')
if standalone_dir not in sys.path:
    sys.path.append(standalone_dir)

from .standalone import pc_algo
from .standalone import parallel_pc_stable
from .standalone import fci_algo_v2
from .standalone import fci_algo_cash
from .standalone import fci_algo_paralel_cash
from .standalone import cfci_algo
from .standalone import cfci_algo_paralel_cash
from .standalone import fci_plus_algo
from .standalone import fci_plus_algo_paralel_cash
from .standalone import rai_algo
from .standalone import rfci_algo
from .standalone import gfci_algo
from .standalone import standalone_raioptimized
# NOTEARS/NOBEARS тянут TensorFlow. Импорт на уровне модуля грузил его в
# каждый `import whydra` — ради двух алгоритмов из семнадцати. Грузим лениво,
# в __init__ соответствующих обёрток, где проверка ошибки уже стояла.
standalone_notears_stable = None
NOTEARS_IMPORT_ERROR = None
standalone_nobears_stable = None
NOBEARS_IMPORT_ERROR = None


def _load_notears():
    global standalone_notears_stable, NOTEARS_IMPORT_ERROR
    if standalone_notears_stable is None and NOTEARS_IMPORT_ERROR is None:
        try:
            from .standalone import standalone_notears_stable as _m
        except ImportError as exc:
            NOTEARS_IMPORT_ERROR = exc
        else:
            standalone_notears_stable = _m
    return standalone_notears_stable


def _load_nobears():
    global standalone_nobears_stable, NOBEARS_IMPORT_ERROR
    if standalone_nobears_stable is None and NOBEARS_IMPORT_ERROR is None:
        try:
            from .standalone import standalone_nobears_stable as _m
        except ImportError as exc:
            NOBEARS_IMPORT_ERROR = exc
        else:
            standalone_nobears_stable = _m
    return standalone_nobears_stable
from ..background_knowledge import MatrixEncoding
from ..evaluation.graph_utils import default_node_names, pag_matrix_to_general_graph


class _PagMatrixResult:
    """Turns an algorithm's endpoint matrix into a local graph.

    Which of the project's two endpoint encodings the matrix carries is
    declared per class rather than inherited by accident. Previously the map
    lived inline in two look-alike `_matrix_to_graph` methods, so a wrapper
    that subclassed the wrong family silently swapped arrowheads and circles.
    """

    #: Endpoint encoding this algorithm's matrix comes back in.
    OUTPUT_ENCODING: MatrixEncoding = None

    def _matrix_to_graph(self, matrix, n_nodes):
        if self.OUTPUT_ENCODING is None:
            raise NotImplementedError(
                f"{type(self).__name__} must declare OUTPUT_ENCODING"
            )
        return pag_matrix_to_general_graph(
            matrix, default_node_names(n_nodes), encoding=self.OUTPUT_ENCODING
        )



class StandalonePCStable(CausalAlgorithm):
    """
    PC-stable: скелет по CI-тестам, затем коллайдеры и правила Мика.
    Результат — CPDAG. Последовательная реализация (``pc_algo``).
    """
    def __init__(self, alpha=0.05, indep_test='fisherz', n_jobs=4, **kwargs):
        self.alpha = alpha
        self.indep_test = indep_test
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        node_names = default_node_names(data.shape[1])
        cg = CausalGraph(data.shape[1], node_names)
        cg.set_ind_test(CIT(data, method=self.indep_test))

        params = dict(self.kwargs)
        params.update(kwargs)
        result = pc_algo.pc_stable(
            cg,
            alpha=self.alpha,
            n_jobs=self.n_jobs,
            **params
        )
        return result.G if hasattr(result, "G") else result


class ParallelPCStable(StandalonePCStable):
    """
    PC-stable с параллельным поиском скелета (``parallel_pc_stable``).
    На тех же данных и тех же фоновых знаниях обязан давать тот же CPDAG,
    что и :class:`StandalonePCStable`.
    """
    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        node_names = default_node_names(data.shape[1])
        cg = CausalGraph(data.shape[1], node_names)
        cg.set_ind_test(CIT(data, method=self.indep_test))

        params = dict(self.kwargs)
        params.update(kwargs)
        result = parallel_pc_stable.pc_stable(
            cg,
            alpha=self.alpha,
            n_jobs=self.n_jobs,
            **params
        )
        return result.G if hasattr(result, "G") else result


class StandaloneFCIStable(_PagMatrixResult, CausalAlgorithm):
    """
    FCI: скелет, Possible-D-SEP, затем правила ориентации R0–R10.
    Допускает скрытые переменные, результат — PAG. Реализация
    ``fci_algo_v2`` с исправленными правилами R1/R2/R5/R10.
    """
    OUTPUT_ENCODING = MatrixEncoding.STANDARD

    def __init__(self, alpha=0.05, indep_test='fisherz', n_jobs=4, verbose=False, **kwargs):
        self.alpha = alpha
        self.indep_test = indep_test
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.kwargs = kwargs

    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        node_names = default_node_names(data.shape[1])
        cg = CausalGraph(data.shape[1], node_names)
        cg.set_ind_test(CIT(data, method=self.indep_test))
        params = dict(self.kwargs)
        params.update(kwargs)

        # indep_test обязателен: fci_stable_v2 создаёт собственный CIT для
        # шагов pdsep и udag2pag, и без явной передачи взял бы дефолтный
        # fisherz — то есть на дискретных данных считал бы не тем тестом.
        pag_matrix = fci_algo_v2.fci_stable_v2(
            cg,
            data,
            alpha=self.alpha,
            indep_test=self.indep_test,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **params
        )

        return self._matrix_to_graph(pag_matrix, data.shape[1])



class StandaloneFCIStableCached(StandaloneFCIStable):
    """
    FCI с мемоизацией CI-тестов (``fci_algo_cash``). Результат совпадает
    с :class:`StandaloneFCIStable`, отличается только скоростью.
    """
    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        n_nodes = data.shape[1]

        node_names = default_node_names(data.shape[1])
        cg = CausalGraph(data.shape[1], node_names)
        cg.set_ind_test(CIT(data, method=self.indep_test))
        params = dict(self.kwargs)
        params.update(kwargs)

        pag_matrix = fci_algo_cash.fci_stable(
            cg,
            data,
            alpha=self.alpha,
            indep_test=self.indep_test,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **params
        )

        return self._matrix_to_graph(pag_matrix, n_nodes)


class StandaloneFCIStableParallelCached(StandaloneFCIStable):
    """
    FCI с мемоизацией и параллельным исполнением
    (``fci_algo_paralel_cash``). ``n_jobs`` задаёт число потоков или
    процессов; результат тот же, что у :class:`StandaloneFCIStable`.
    """
    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        n_nodes = data.shape[1]

        node_names = default_node_names(data.shape[1])
        cg = CausalGraph(data.shape[1], node_names)
        cg.set_ind_test(CIT(data, method=self.indep_test))
        params = dict(self.kwargs)
        params.update(kwargs)

        pag_matrix = fci_algo_paralel_cash.fci_stable(
            cg,
            data,
            alpha=self.alpha,
            indep_test=self.indep_test,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **params
        )

        return self._matrix_to_graph(pag_matrix, n_nodes)


class StandaloneCFCIStable(StandaloneFCIStable):
    """
    Консервативный FCI: неоднозначные тройки не ориентируются как
    коллайдеры, а помечаются. Меньше ложных стрелок ценой большего
    числа кружков в PAG.
    """
    def __init__(self, alpha=0.05, indep_test='fisherz', n_jobs=4, verbose=False, maj_rule=False, **kwargs):
        super().__init__(alpha=alpha, indep_test=indep_test, n_jobs=n_jobs, verbose=verbose, **kwargs)
        self.maj_rule = maj_rule

    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        n_nodes = data.shape[1]

        node_names = default_node_names(data.shape[1])
        cg = CausalGraph(data.shape[1], node_names)
        cg.set_ind_test(CIT(data, method=self.indep_test))
        params = dict(self.kwargs)
        params.update(kwargs)

        pag_matrix = cfci_algo.cfci_stable(
            cg,
            data,
            alpha=self.alpha,
            indep_test=self.indep_test,
            maj_rule=self.maj_rule,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **params
        )

        return self._matrix_to_graph(pag_matrix, n_nodes)


class StandaloneCFCIStableParallelCached(StandaloneCFCIStable):
    """
    Консервативный FCI с мемоизацией и параллельным исполнением.
    Результат тот же, что у :class:`StandaloneCFCIStable`.
    """
    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        n_nodes = data.shape[1]

        node_names = default_node_names(data.shape[1])
        cg = CausalGraph(data.shape[1], node_names)
        cg.set_ind_test(CIT(data, method=self.indep_test))
        params = dict(self.kwargs)
        params.update(kwargs)

        pag_matrix = cfci_algo_paralel_cash.cfci_stable(
            cg,
            data,
            alpha=self.alpha,
            indep_test=self.indep_test,
            maj_rule=self.maj_rule,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **params
        )

        return self._matrix_to_graph(pag_matrix, n_nodes)


class StandaloneRFCIStable(_PagMatrixResult, CausalAlgorithm):
    """
    RFCI: вместо полного Possible-D-SEP — проверки на дискриминирующих
    путях. Заметно дешевле FCI на больших графах, PAG может быть слабее.
    """
    OUTPUT_ENCODING = MatrixEncoding.STANDARD

    def __init__(self, alpha=0.05, indep_test='fisherz', n_jobs=4, verbose=False, **kwargs):
        self.alpha = alpha
        self.indep_test = indep_test
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.kwargs = kwargs

    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        n_nodes = data.shape[1]

        node_names = default_node_names(data.shape[1])
        cg = CausalGraph(data.shape[1], node_names)
        cg.set_ind_test(CIT(data, method=self.indep_test))
        params = dict(self.kwargs)
        params.update(kwargs)

        pag_matrix = rfci_algo.rfci_stable(
            cg,
            data,
            alpha=self.alpha,
            indep_test=self.indep_test,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **params
        )

        return self._matrix_to_graph(pag_matrix, n_nodes)



class StandaloneGFCIStable(StandaloneRFCIStable):
    """
    GFCI: стартовый CPDAG от GES (score-based), дальше правила FCI.
    ``start_from_complete_graph=True`` заменяет GES полным o-o графом.
    """
    def __init__(
        self,
        alpha=0.05,
        indep_test='fisherz',
        n_jobs=4,
        verbose=False,
        score_func='auto',
        use_max_p=True,
        depth=-1,
        start_from_complete_graph=False,
        use_ci_cache=True,
        **kwargs,
    ):
        super().__init__(
            alpha=alpha,
            indep_test=indep_test,
            n_jobs=n_jobs,
            verbose=verbose,
            **kwargs,
        )
        self.score_func = score_func
        self.use_max_p = use_max_p
        self.depth = depth
        self.start_from_complete_graph = start_from_complete_graph
        self.use_ci_cache = use_ci_cache

    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        n_nodes = data.shape[1]
        node_names = default_node_names(n_nodes)
        cg = CausalGraph(n_nodes, node_names)
        params = dict(self.kwargs)
        params.update(kwargs)

        pag_matrix = gfci_algo.gfci_stable(
            cg,
            data,
            alpha=self.alpha,
            indep_test=self.indep_test,
            score_func=self.score_func,
            use_max_p=self.use_max_p,
            depth=self.depth,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            start_from_complete_graph=self.start_from_complete_graph,
            node_names=node_names,
            use_ci_cache=self.use_ci_cache,
            **params,
        )
        return self._matrix_to_graph(pag_matrix, n_nodes)


class StandaloneFCIPlusStable(StandaloneRFCIStable):
    """
    FCI+: скелет строится по возрастанию порядка с аугментацией графа,
    что сокращает число CI-тестов на разреженных графах.
    """
    def __init__(self, alpha=0.05, indep_test='fisherz', n_jobs=4, verbose=False, k=None, **kwargs):
        super().__init__(alpha=alpha, indep_test=indep_test, n_jobs=n_jobs, verbose=verbose, **kwargs)
        self.k = k

    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        n_nodes = data.shape[1]

        node_names = default_node_names(data.shape[1])
        cg = CausalGraph(data.shape[1], node_names)
        cg.set_ind_test(CIT(data, method=self.indep_test))
        params = dict(self.kwargs)
        params.update(kwargs)

        pag_matrix = fci_plus_algo.fci_plus_stable(
            cg,
            data,
            alpha=self.alpha,
            indep_test=self.indep_test,
            k=self.k,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **params
        )

        return self._matrix_to_graph(pag_matrix, n_nodes)


class StandaloneFCIPlusStableParallelCached(StandaloneFCIPlusStable):
    """
    FCI+ с мемоизацией и параллельным исполнением. Результат тот же,
    что у :class:`StandaloneFCIPlusStable`.
    """
    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        n_nodes = data.shape[1]

        node_names = default_node_names(data.shape[1])
        cg = CausalGraph(data.shape[1], node_names)
        cg.set_ind_test(CIT(data, method=self.indep_test))
        params = dict(self.kwargs)
        params.update(kwargs)

        pag_matrix = fci_plus_algo_paralel_cash.fci_plus_stable(
            cg,
            data,
            alpha=self.alpha,
            indep_test=self.indep_test,
            k=self.k,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **params
        )

        return self._matrix_to_graph(pag_matrix, n_nodes)

class StandaloneRAIOptimized(CausalAlgorithm):
    """
    RAI (recursive autonomy identification), оптимизированная реализация.
    Фоновые знания не поддерживаются.
    """
    def __init__(self, alpha=0.05, indep_test='fisherz', **kwargs):
        self.alpha = alpha
        self.indep_test = indep_test
        self.kwargs = kwargs

    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        node_names = default_node_names(data.shape[1])
        params = dict(self.kwargs)
        params.update(kwargs)
        cg = standalone_raioptimized.rai_optimized(
            data,
            alpha=self.alpha,
            indep_test=self.indep_test,
            node_names=node_names,
            **params
        )
        if hasattr(cg, 'G') and isinstance(cg.G, GeneralGraph):
            return cg.G
        return cg

class StandaloneRAIStable(CausalAlgorithm):
    """
    RAI (recursive autonomy identification), опорная реализация.
    Фоновые знания не поддерживаются.
    """
    def __init__(self, alpha=0.05, indep_test='fisherz', n_jobs=4, verbose=False, **kwargs):
        self.alpha = alpha
        self.indep_test = indep_test
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.verbose = verbose

    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        node_names = default_node_names(data.shape[1])
        params = dict(self.kwargs)
        params.update(kwargs)
        cg = rai_algo.rai_stable(
            data,
            alpha=self.alpha,
            indep_test=self.indep_test,
            node_names=node_names,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **params
        )
        if hasattr(cg, 'G') and isinstance(cg.G, GeneralGraph):
            return cg.G
        return cg

class StandaloneNoTearsStable(CausalAlgorithm):
    def __init__(
            self,
            alpha=0.05,
            indep_test: str | None = None,
            lambda1: float | None = None,
            alpha_init: float = 0.0,
            rho_init: float = 1.0,
            rho_multiplier: float = 1.25,
            outer_iter: int = 50,
            inner_iter: int = 100,
            step_size: float = 1e-2,
            h_tol: float = 1e-8,
            weight_threshold: float = 1e-3,
            verbose: bool = False,
            **kwargs
    ):
        """
        Standalone NOTEARS implementation.
        alpha is retained for compatibility with other configs; it serves as
        the default value for lambda1 (L1 penalty) unless explicitly provided.
        """
        _load_notears()
        if NOTEARS_IMPORT_ERROR is not None:
            raise ImportError(
                "StandaloneNoTearsStable requires TensorFlow. "
                "Install the dependencies listed in requirements.txt or use another algorithm."
            ) from NOTEARS_IMPORT_ERROR
        self.lambda1 = lambda1 if lambda1 is not None else alpha
        self.alpha_init = alpha_init
        self.rho_init = rho_init
        self.rho_multiplier = rho_multiplier
        self.outer_iter = outer_iter
        self.inner_iter = inner_iter
        self.step_size = step_size
        self.h_tol = h_tol
        self.weight_threshold = weight_threshold
        self.verbose = verbose
        self.kwargs = kwargs

    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        node_names = default_node_names(data.shape[1])
        weights = standalone_notears_stable.estimate_notears_weights(
            data,
            lambda1=self.lambda1,
            alpha_init=self.alpha_init,
            rho_init=self.rho_init,
            rho_multiplier=self.rho_multiplier,
            outer_iter=self.outer_iter,
            inner_iter=self.inner_iter,
            step_size=self.step_size,
            h_tol=self.h_tol,
            verbose=self.verbose,
        )
        return self._matrix_to_graph(weights, node_names)

    def _matrix_to_graph(self, weights: np.ndarray, node_names):
        nodes = [GraphNode(name) for name in node_names]
        graph = GeneralGraph(nodes)

        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                if i == j:
                    continue
                if abs(weights[i, j]) >= self.weight_threshold:
                    if not graph.is_adjacent_to(nodes[i], nodes[j]):
                        edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW)
                        graph.add_edge(edge)
        return graph

class StandaloneNoBearsStable(CausalAlgorithm):
    def __init__(
        self,
        alpha: float = 0.05,
        indep_test: str | None = None,
        beta1: float | None = None,
        beta2: float = 0.001,
        alpha_init: float = 0.01,
        rho_init: float = 1.0,
        poly_degree: int = 3,
        l1_ratio: float = 0.5,
        init_iter: int = 200,
        outer_iter: int = 200,
        inner_iter: int = 100,
        step_size: float = 1e-2,
        noise_level: float = 0.1,
        weight_threshold: float = 1e-3,
        use_reg_init: bool = True,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Standalone NOBEARS implementation.

        `alpha` is kept for API compatibility and used as the default value
        for the sparsity penalty `beta1` unless explicitly provided.
        """
        _load_nobears()
        if NOBEARS_IMPORT_ERROR is not None:
            raise ImportError(
                "StandaloneNoBearsStable requires TensorFlow. "
                "Install the dependencies listed in requirements.txt or use another algorithm."
            ) from NOBEARS_IMPORT_ERROR
        self.beta1 = beta1 if beta1 is not None else alpha
        self.beta2 = beta2
        self.alpha_init = alpha_init
        self.rho_init = rho_init
        self.poly_degree = poly_degree
        self.l1_ratio = l1_ratio
        self.init_iter = init_iter
        self.outer_iter = outer_iter
        self.inner_iter = inner_iter
        self.step_size = step_size
        self.noise_level = noise_level
        self.weight_threshold = weight_threshold
        self.use_reg_init = use_reg_init
        self.verbose = verbose
        self.kwargs = kwargs

    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        node_names = default_node_names(data.shape[1])
        weights = standalone_nobears_stable.estimate_nobears_weights(
            data,
            beta1=self.beta1,
            beta2=self.beta2,
            alpha_init=self.alpha_init,
            rho_init=self.rho_init,
            poly_degree=self.poly_degree,
            l1_ratio=self.l1_ratio,
            init_iter=self.init_iter,
            outer_iter=self.outer_iter,
            inner_iter=self.inner_iter,
            step_size=self.step_size,
            noise_level=self.noise_level,
            use_reg_init=self.use_reg_init,
        )
        return self._matrix_to_graph(weights, node_names)

    def _matrix_to_graph(self, weights: np.ndarray, node_names):
        nodes = [GraphNode(name) for name in node_names]
        graph = GeneralGraph(nodes)

        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                if i == j:
                    continue
                if abs(weights[i, j]) >= self.weight_threshold:
                    if not graph.is_adjacent_to(nodes[i], nodes[j]):
                        edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW)
                        graph.add_edge(edge)
        return graph


class OrigJPCMCIPlus(CausalAlgorithm):
    def __init__(self, alpha=0.05, tau_max=2, **kwargs):
        if ORIG_TIGRAMITE_IMPORT_ERROR is not None:
            raise ImportError(
                "OrigJPCMCIPlus requires tigramite dependencies. "
                "Install tigramite or use a non-time-series algorithm."
            ) from ORIG_TIGRAMITE_IMPORT_ERROR
        self.alpha = alpha
        self.tau_max = tau_max
        self.kwargs = kwargs

    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        """
        Запуск оригинальной реализации J-PCMCI+ из Tigramite.
        data: (n_samples, n_time_steps, n_nodes)
        """
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim != 3:
            raise ValueError("OrigJPCMCIPlus expects a 2D or 3D numpy array")

        n_samples, n_time, n_nodes = data.shape

        # 1. Подготовка данных для Tigramite (DataFrame)
        # J-PCMCI+ ожидает словарь {id_реализации: данные(Time, Nodes)}
        data_dict = {i: data[i] for i in range(n_samples)}

        var_names = default_node_names(n_nodes)

        # Создаем DataFrame из нашей локальной копии data_processing
        dataframe = pp_orig.DataFrame(
            data=data_dict,
            var_names=var_names,
            analysis_mode='multiple'
        )

        # 2. Классификация узлов (Node Classification)
        # Для CausalTime (traffic) у нас только системные переменные.
        # J-PCMCI+ требует словарь node_classification.
        # Если контекстов нет, все узлы помечаем как "system".
        node_classification = {i: "system" for i in range(n_nodes)}

        # 3. Инициализация теста независимости
        # Используем ParCorrMult из нашей локальной копии
        cond_ind_test = ParCorrMultOrig(significance='analytic', correlation_type='PCCA_WilksLambda')

        # 4. Инициализация алгоритма
        jpcmci = JPCMCIplusOrig(
            dataframe=dataframe,
            cond_ind_test=cond_ind_test,
            node_classification=node_classification,
            verbosity=2 # 0 чтобы ничего не принтовалось
        )

        # 5. Запуск
        # Параметры J-PCMCI+
        results = jpcmci.run_jpcmciplus(
            tau_min=0,
            tau_max=self.tau_max,
            pc_alpha=self.alpha,

            max_conds_dim=3  # ограничение сложности
            # Остальные параметры по умолчанию
        )

        # 6. Конвертация результатов в GeneralGraph
        # Результат в results['graph'] - это матрица (N, N, tau_max+1) со строками '-->', 'o-o' и т.д.
        res_graph = results['graph']

        nodes = [GraphNode(name) for name in var_names]
        g = GeneralGraph(nodes)

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j: continue

                # Ищем связь на любом лаге (Summary Graph)
                has_edge = False
                link_type = ""

                for tau in range(self.tau_max + 1):
                    link = res_graph[i, j, tau]
                    if link != "":
                        has_edge = True
                        link_type = link
                        break

                if has_edge:
                    # Tigramite: graph[i, j] - связь МЕЖДУ i и j.
                    # Направление кодируется строкой.
                    # '-->' значит i -> j
                    # '<--' значит i <- j
                    # 'o-o' значит i o-o j

                    end1 = Endpoint.TAIL
                    end2 = Endpoint.TAIL

                    if link_type == "-->":
                        end1 = Endpoint.TAIL
                        end2 = Endpoint.ARROW
                    elif link_type == "<--":
                        end1 = Endpoint.ARROW
                        end2 = Endpoint.TAIL
                    elif link_type == "o-o":
                        end1 = Endpoint.CIRCLE
                        end2 = Endpoint.CIRCLE
                    elif link_type == "x-x":  # Tigramite иногда использует это для "нет связи", но мы проверяем != ""
                        continue

                    # Проверяем, не добавлено ли уже ребро (GeneralGraph хранит ненаправленные ребра один раз)
                    if not g.get_edge(nodes[i], nodes[j]):
                        edge = Edge(nodes[i], nodes[j], end1, end2)
                        g.add_edge(edge)

        return g

try:
    from .standalone.dag_gnn import runner as dagnn_runner
except ModuleNotFoundError as exc:
    dagnn_runner = None
    DAG_GNN_IMPORT_ERROR = exc
else:
    DAG_GNN_IMPORT_ERROR = None


class DagGNN(CausalAlgorithm):
    _RUNNER_PARAMETERS = frozenset({
        "lambda_A",
        "c_A",
        "h_tol",
        "tau_A",
        "k_max_iter",
        "l2_reg_diag",
        "device",
    })

    def __init__(self,
                 epochs=300, batch_size=100, lr=3e-3,
                 graph_threshold=0.3, encoder_hidden=64, decoder_hidden=64,
                 indep_test='fisherz',
                 alpha=0.05, n_jobs=1, verbose=False, show_progress=False,
                 **kwargs):
        if DAG_GNN_IMPORT_ERROR is not None:
            raise ImportError(
                "DagGNN requires PyTorch. Install torch or use another algorithm."
            ) from DAG_GNN_IMPORT_ERROR

        unknown = sorted(set(kwargs) - self._RUNNER_PARAMETERS)
        if unknown:
            raise TypeError(
                "Unsupported DagGNN parameters: " + ", ".join(unknown)
            )

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.graph_threshold = graph_threshold
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.indep_test = indep_test
        # Эти параметры входят в общий контракт публичных обёрток. DagGNN
        # непосредственно их не использует и не должен передавать в runner.
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.show_progress = show_progress
        self.runner_kwargs = dict(kwargs)

    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        # Определяем тип данных по тесту независимости
        if self.indep_test in ('chisq', 'gsq'):
            data_type = 'discrete'
            # Данные должны быть целочисленными
            data = data.astype(int)
        else:  # fisherz, parcorr, etc.
            data_type = 'continuous'
            # Нормализация для непрерывных данных
            data_mean = np.mean(data, axis=0)
            data_std = np.std(data, axis=0)
            data_std[data_std == 0] = 1.0
            data = (data - data_mean) / data_std

        actual_batch_size = min(self.batch_size, data.shape[0])

        adj_matrix = dagnn_runner.train_dag_gnn(
            data,
            data_type=data_type,
            epochs=self.epochs,
            batch_size=actual_batch_size,
            lr=self.lr,
            graph_threshold=self.graph_threshold,
            encoder_hidden=self.encoder_hidden,
            decoder_hidden=self.decoder_hidden,
            **self.runner_kwargs
        )

        return self._matrix_to_graph(adj_matrix, data.shape[1])

    def _matrix_to_graph(self, matrix, n_nodes):
        # ... (этот метод не меняется)
        node_names = default_node_names(n_nodes)
        nodes = [GraphNode(name) for name in node_names]
        g = GeneralGraph(nodes)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j: continue
                if matrix[i, j] != 0:
                    edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW)
                    g.add_edge(edge)
        return g

#: Публичные обёртки standalone-алгоритмов: класс -> короткие алиасы.
#: Каноническое имя (``ClassName.lower()``) регистрируется автоматически.
_STANDALONE_ALGORITHM_ALIASES: List[Tuple[Type[CausalAlgorithm], Tuple[str, ...]]] = [
    (StandalonePCStable, ("pc_stable", "pcstable", "pc_stable_seq")),
    (ParallelPCStable, ("pc_stable_parallel", "parallel_pc_stable", "pc_stable_par")),
    (StandaloneFCIStable, ("fci_stable", "fcistable")),
    (StandaloneFCIStableCached, ("fci_stable_cached", "fci_stable_cash")),
    (StandaloneFCIStableParallelCached, ("fci_stable_parallel_cached", "fci_stable_paralel_cash")),
    (StandaloneCFCIStable, ("cfci_stable",)),
    (StandaloneCFCIStableParallelCached, ("cfci_stable_parallel_cached", "cfci_stable_paralel_cash")),
    (StandaloneRFCIStable, ("rfci_stable",)),
    (StandaloneGFCIStable, ("gfci_stable",)),
    (StandaloneFCIPlusStable, ("fci_plus_stable",)),
    (StandaloneFCIPlusStableParallelCached, ("fci_plus_stable_parallel_cached", "fci_plus_stable_paralel_cash")),
    (StandaloneRAIOptimized, ("rai_optimized",)),
    (StandaloneRAIStable, ("rai_stable",)),
    (StandaloneNoTearsStable, ("notears_stable",)),
    (StandaloneNoBearsStable, ("nobears_stable",)),
    (OrigJPCMCIPlus, ("jpcmci_plus", "orig_j_pcmci_plus")),
    (DagGNN, ("dag_gnn",)),
]


def _build_standalone_registry(
    pairs: List[Tuple[Type[CausalAlgorithm], Tuple[str, ...]]],
) -> Dict[str, Type[CausalAlgorithm]]:
    registry: Dict[str, Type[CausalAlgorithm]] = {}
    for algo_cls, aliases in pairs:
        registry[algo_cls.__name__.lower()] = algo_cls
        for alias in aliases:
            registry[alias] = algo_cls
    return registry


_STANDALONE_ALGORITHM_REGISTRY: Dict[str, Type[CausalAlgorithm]] = _build_standalone_registry(
    _STANDALONE_ALGORITHM_ALIASES
)


def _normalize_algorithm_name(name: str) -> str:
    """Normalize algorithm name to make lookups more user-friendly."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def available_standalone_algorithms() -> List[str]:
    """
    Return canonical standalone algorithm names supported by the unified API.
    """
    return [algo_cls.__name__ for algo_cls, _ in _STANDALONE_ALGORITHM_ALIASES]


def create_standalone_algorithm(name: str, **kwargs) -> CausalAlgorithm:
    """
    Create a standalone algorithm wrapper by name.

    Examples:
        create_standalone_algorithm("pc_stable", alpha=0.01)
        create_standalone_algorithm("StandaloneRAIStable", n_jobs=8)
    """
    normalized_name = _normalize_algorithm_name(name)
    algo_cls = _STANDALONE_ALGORITHM_REGISTRY.get(normalized_name)
    if algo_cls is None:
        available = ", ".join(available_standalone_algorithms())
        raise ValueError(
            f"Unknown standalone algorithm '{name}'. "
            f"Available algorithms: {available}"
        )
    return algo_cls(**kwargs)


def _text_table_has_header(path: Path) -> bool:
    """True when the first non-empty line of a text table is labels, not data.

    Assuming a header unconditionally silently drops the first observation of
    every headerless file, which is invisible downstream: the array is simply
    one row shorter. Deciding from the file itself keeps both layouts correct.
    """
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                first_line = line
                break
        else:
            return False

    tokens = first_line.replace(",", " ").replace(";", " ").replace("	", " ").split()
    if not tokens:
        return False
    for token in tokens:
        try:
            float(token)
        except ValueError:
            return True
    return False


def _load_data_like_main(
    data_or_path: Union[np.ndarray, str, Path],
    header: Optional[bool] = None,
) -> np.ndarray:
    """
    Load data from numpy array or file path.

    Path behavior mirrors common loading patterns used in the project:
    - .pickle/.pkl: pickle.load (see the security note in the Notes section of run_standalone_algorithm)
    - .npy: np.load
    - text files: header detected from the file; pass ``header`` to override.
    """
    if isinstance(data_or_path, np.ndarray):
        return data_or_path

    if isinstance(data_or_path, (str, Path)):
        path = Path(data_or_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        suffix = path.suffix.lower()
        if suffix in {".pickle", ".pkl"}:
            with path.open("rb") as f:
                loaded = pickle.load(f)
            if not isinstance(loaded, np.ndarray):
                raise ValueError(
                    f"Pickle file '{path}' does not contain np.ndarray "
                    f"(got {type(loaded).__name__})."
                )
            return loaded

        if suffix == ".npy":
            return np.load(path)

        has_header = _text_table_has_header(path) if header is None else bool(header)
        try:
            return np.loadtxt(path, skiprows=1 if has_header else 0)
        except ValueError:
            df = pd.read_csv(path, sep=r"\s+", header=0 if has_header else None)
            return df.to_numpy()

    raise TypeError(
        "data_or_path must be np.ndarray, str, or pathlib.Path. "
        f"Got {type(data_or_path).__name__}."
    )


def run_standalone_algorithm(
    name: str,
    data_or_path: Union[np.ndarray, str, Path],
    header: Optional[bool] = None,
    **kwargs
) -> GeneralGraph:
    """
    Запускает standalone-алгоритм поиска causal graph.

    Функция объединяет три шага:

    1. Нормализацию имени и создание обёртки алгоритма через
       :func:`create_standalone_algorithm`.
    2. Загрузку/преобразование входных данных через
       :func:`_load_data_like_main`.
    3. Выполнение метода ``run`` у созданного экземпляра алгоритма.

    Подходит как для интерактивного использования, так и для пайплайнов,
    где нужно единообразно запускать разные standalone-алгоритмы без
    явного ручного создания классов-обёрток.

    Parameters
    ----------
    name : str
        Имя алгоритма (каноническое имя класса или алиас). Допускаются
        различные формы записи, например: ``"StandalonePCStable"``,
        ``"pc_stable"``, ``"pcstable"``, ``"fci_stable"``,
        ``"rai_optimized"``, ``"rai_stable"``.
    data_or_path : numpy.ndarray | str | pathlib.Path
        Данные для запуска алгоритма:

        - ``numpy.ndarray``: массив наблюдений уже в памяти;
        - ``str`` / ``Path``: путь к файлу с данными.

        Поддерживаемые форматы файлов:

        - ``.pickle`` / ``.pkl`` — объект должен быть ``numpy.ndarray``;
        - ``.npy`` — загрузка через ``numpy.load``;
        - текстовые файлы — заголовок определяется автоматически
          (``_text_table_has_header``), затем ``numpy.loadtxt`` с нужным
          ``skiprows``; при неуспехе fallback на ``pandas.read_csv(...)``.

    header : bool, optional
        Переопределяет автоопределение заголовка для текстовых файлов:
        ``True`` — первая строка заголовок, ``False`` — данные с первой
        строки, ``None`` (по умолчанию) — определить по содержимому.
        Для ``.npy``/``.pickle`` игнорируется.
    **kwargs
        Параметры конструктора выбранной обёртки алгоритма.
        Передаются в :func:`create_standalone_algorithm` (например,
        ``alpha``, ``indep_test``, ``n_jobs``, ``verbose``).

    Returns
    -------
    GeneralGraph
        Граф во внутреннем формате Whydra. В зависимости от выбранного алгоритма
        это может быть: DAG, CPDAG/PDAG или PAG (например, для FCI — PAG).

    Raises
    ------
    ValueError
        Если передано неизвестное имя алгоритма.
    FileNotFoundError
        Если путь к файлу данных не существует.
    TypeError
        Если ``data_or_path`` имеет неподдерживаемый тип.
    ValueError
        Если ``.pkl/.pickle`` файл не содержит ``numpy.ndarray``.
    Exception
        Любые исключения, проброшенные конструктором выбранной обёртки
        или её методом ``run``.

    Notes
    -----
    - Функция не нормализует результат к «строго DAG»: тип структуры определяется
      выбранным алгоритмом (например, FCI возвращает PAG).
    - ``**kwargs`` применяются только на этапе создания алгоритма.
      В текущей реализации они не прокидываются отдельным шагом в
      ``algorithm.run(...)`` из этой функции.
    - Безопасность: ``.pkl``/``.pickle`` читаются через ``pickle.load``,
      который исполняет произвольный код при десериализации. Загружайте
      только файлы из доверенного источника; для чужих данных используйте
      ``.npy`` или текстовый формат.
    - Функция является удобным фасадом над ``create_standalone_algorithm``
      и ``_load_data_like_main`` для унифицированного API.

    Examples
    --------
    Запуск на массиве NumPy:

    >>> graph = run_standalone_algorithm(
    ...     "pc_stable",
    ...     data,
    ...     alpha=0.01,
    ...     indep_test="fisherz",
    ...     n_jobs=8
    ... )

    Запуск на данных из файла:

    >>> graph = run_standalone_algorithm(
    ...     "StandaloneFCIStable",
    ...     "data/observations.npy",
    ...     alpha=0.05,
    ...     verbose=True
    ... )
    """
    algorithm = create_standalone_algorithm(name, **kwargs)
    data = _load_data_like_main(data_or_path, header=header)
    return algorithm.run(data)


__all__ = [
    "StandalonePCStable",
    "ParallelPCStable",
    "StandaloneFCIStable",
    "StandaloneFCIStableCached",
    "StandaloneFCIStableParallelCached",
    "StandaloneCFCIStable",
    "StandaloneCFCIStableParallelCached",
    "StandaloneRFCIStable",
    "StandaloneGFCIStable",
    "StandaloneFCIPlusStable",
    "StandaloneFCIPlusStableParallelCached",
    "StandaloneRAIOptimized",
    "StandaloneRAIStable",
    "StandaloneNoTearsStable",
    "StandaloneNoBearsStable",
    "OrigJPCMCIPlus",
    "DagGNN",
    "available_standalone_algorithms",
    "create_standalone_algorithm",
    "run_standalone_algorithm",
]
