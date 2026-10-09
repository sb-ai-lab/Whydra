import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from .base import CausalAlgorithm
from .graph_core import CausalGraph, CIT, GeneralGraph, Node as GraphNode, Edge, Endpoint

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
    "available_standalone_algorithms",
    "create_standalone_algorithm",
    "run_standalone_algorithm",
]
