from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# from causal_graphs.data.loaders import BnLearnLoader, FeedbacksLoader, LucasLoader, TetradLoader
from .loader_registry import LoaderRegistry


class BenchmarkCaseRepository:
    """
    Репозиторий для поиска бенчмарк-кейсов и загрузки их датасетов/эталонных графов.

    Parameters
    ----------
    cases_root : str, optional
        Корневая директория с бенчмарками. Ожидаемая структура:

        - ``{cases_root}/{benchmark}/{case}/input``
        - ``{cases_root}/{benchmark}/{case}/ground.truth``

    Notes
    -----
    Директория считается валидным кейсом, если содержит обе подпапки:
    ``input`` и ``ground.truth``.
    """
    
    def __init__(self, cases_root: str = "whydra_benchmarks/cases"):
        """
        Инициализировать репозиторий с корневым путем кейсов.

        Parameters
        ----------
        cases_root : str, optional
            Путь к корневой директории с кейсами бенчмарков.
        """
        self.cases_root = Path(cases_root).resolve()

    def list_available_cases(self, benchmark_name: Optional[str] = None) -> List[str]:
        """
        Получить список доступных кейсов.

        Parameters
        ----------
        benchmark_name : str, optional
            Имя бенчмарка. Если задано, возвращаются только кейсы внутри
            этого бенчмарка как пути относительно его директории.
            Если не задано, возвращаются кейсы всех бенчмарков как пути
            относительно ``cases_root``.

        Returns
        -------
        List[str]
            Отсортированный список идентификаторов кейсов в POSIX-формате.

        Raises
        ------
        FileNotFoundError
            Если ``benchmark_name`` задан, но соответствующая директория не найдена.
        """
        if benchmark_name is not None:
            root = self.cases_root / benchmark_name
            if not root.is_dir():
                raise FileNotFoundError(f"Benchmark directory not found: {root}")
            return sorted([str(p.relative_to(root)).replace("\\", "/") for p in self._iter_case_dirs(root)])

        if not self.cases_root.is_dir():
            return []

        result: List[str] = []
        for benchmark_dir in sorted([p for p in self.cases_root.iterdir() if p.is_dir()]):
            for case_dir in self._iter_case_dirs(benchmark_dir):
                result.append(str(case_dir.relative_to(self.cases_root)).replace("\\", "/"))
        return sorted(result)


    def get_case_datasets(
        self,
        case_name: Optional[str] = None,
        benchmark_name: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Загрузить все датасеты для указанного кейса.

        Parameters
        ----------
        case_name : str, optional
            Идентификатор кейса. Может быть:
            - коротким именем кейса (например, ``"child"``), или
            - относительным путем внутри ``cases_root`` (например, ``"bnlearn/child"``).
        benchmark_name : str, optional
            Имя бенчмарка для прямого разрешения пути вида
            ``{cases_root}/{benchmark_name}/{case_name}``.

        Returns
        -------
        Dict[str, np.ndarray]
            Словарь вида ``ключ_датасета -> загруженный массив``,
            возвращаемый выбранным загрузчиком.

        Raises
        ------
        ValueError
            Если ``case_name`` не передан.
        FileNotFoundError
            Если кейс не найден или не обнаружены входные файлы датасета.
        """
        if not case_name:
            raise ValueError(
                "case_name is required, e.g. get_case_datasets(case_name='child', benchmark_name='bnlearn')")
        case_dir = self._resolve_case_dir(case_name, benchmark_name)
        loader, load_specs = self._build_loader_and_specs(case_dir)

        datasets: Dict[str, np.ndarray] = {}
        for dataset_key, load_name in load_specs:
            datasets[dataset_key] = loader.load_data(load_name)

        if not datasets:
            raise FileNotFoundError(f"No dataset files found in: {case_dir / 'input'}")
        return datasets

    def get_case_data(
        self,
        case_name: str,
        benchmark_name: Optional[str] = None,
        dataset_key: Optional[str] = None,
    ) -> np.ndarray:
        """
        Загрузить и вернуть один датасет для указанного кейса.

        Parameters
        ----------
        case_name : str
            Идентификатор кейса (короткое имя или относительный путь внутри ``cases_root``).
        benchmark_name : str, optional
            Имя бенчмарка для прямого разрешения пути
            ``{cases_root}/{benchmark_name}/{case_name}``.
        dataset_key : str, optional
            Ключ конкретного датасета из словаря, который возвращает
            ``get_case_datasets``. Если не задан, возвращается первый найденный датасет.

        Returns
        -------
        np.ndarray
            Загруженные данные выбранного датасета.

        Raises
        ------
        FileNotFoundError
            Если кейс не найден, датасеты не найдены или указан несуществующий ``dataset_key``.
        """
        try:
            datasets = self.get_case_datasets(case_name=case_name, benchmark_name=benchmark_name)
        except (FileNotFoundError, ValueError) as e:
            raise FileNotFoundError("не найден кейс") from e

        if dataset_key is not None:
            if dataset_key not in datasets:
                raise FileNotFoundError("не найден кейс")
            return datasets[dataset_key]

        return next(iter(datasets.values()))

    def get_case_ground_truth(self, case_name: str, benchmark_name: Optional[str] = None) -> str:
        """
        Определить путь к файлу ground truth для кейса.

        Parameters
        ----------
        case_name : str
            Идентификатор кейса.
        benchmark_name : str, optional
            Имя бенчмарка для прямого разрешения пути.

        Returns
        -------
        str
            Путь к выбранному файлу ground truth. Если найдено несколько
            поддерживаемых файлов, приоритет отдается файлу, в имени которого
            есть ``"graph"``; иначе возвращается первый в сортировке.

        Raises
        ------
        FileNotFoundError
            Если кейс, директория ground truth или нужные файлы не найдены.
        """
        case_dir = self._resolve_case_dir(case_name, benchmark_name)
        gt_dir = case_dir / "ground.truth"
        if not gt_dir.is_dir():
            raise FileNotFoundError(f"Ground truth directory not found for case '{case_name}': {gt_dir}")

        files = self._supported_files(gt_dir)
        if not files:
            raise FileNotFoundError(f"No supported ground truth files found in: {gt_dir}")

        graph_candidates = sorted([p for p in files if "graph" in p.name.lower()])
        if graph_candidates:
            return str(graph_candidates[0])

        return str(files[0])

    def _supported_files(self, directory: Path) -> List[Path]:
        """
        Вернуть отсортированный список поддерживаемых текстовых/табличных файлов.

        Parameters
        ----------
        directory : Path
            Директория для сканирования.

        Returns
        -------
        List[Path]
            Список файлов с расширениями
            ``{.txt, .data, .csv, .dat, .tsv}``.
        """
        allowed_suffixes = (".txt", ".data", ".csv", ".dat", ".tsv")
        return sorted(
            [
                p for p in directory.iterdir()
                if p.is_file()
                   and p.name.lower().endswith(allowed_suffixes)
                   and p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg"}
            ]
        )

    def _iter_case_dirs(self, root: Path) -> List[Path]:
        """
        Рекурсивно найти валидные директории кейсов внутри ``root``.

        Parameters
        ----------
        root : Path
            Корневая директория поиска.

        Returns
        -------
        List[Path]
            Отсортированный список директорий, содержащих ``input`` и ``ground.truth``.
        """
        case_dirs = []
        for p in root.rglob("*"):
            if p.is_dir() and (p / "input").is_dir() and (p / "ground.truth").is_dir():
                case_dirs.append(p)
        return sorted(case_dirs)

    # replace this function entirely
    def _resolve_case_dir(self, case_name: Optional[str], benchmark_name: Optional[str]) -> Path:
        """
        Разрешить путь к директории кейса по переданным идентификаторам.

        Parameters
        ----------
        case_name : str, optional
            Идентификатор кейса (короткое имя или относительный путь).
        benchmark_name : str, optional
            Имя бенчмарка для прямого разрешения.

        Returns
        -------
        Path
            Абсолютный путь к найденной директории кейса.

        Raises
        ------
        ValueError
            Если ``case_name`` не задан или неоднозначен.
        FileNotFoundError
            Если корневая директория или сам кейс не найдены.
        """
        if not case_name:
            raise ValueError("case_name must not be None")

        if benchmark_name is not None:
            case_dir = (self.cases_root / benchmark_name / case_name).resolve()
            if not case_dir.is_dir():
                raise FileNotFoundError(f"Case not found: {case_dir}")
            if not (case_dir / "input").is_dir():
                raise FileNotFoundError(f"Input directory not found for case '{case_name}': {case_dir / 'input'}")
            return case_dir

        direct = (self.cases_root / case_name).resolve()
        if direct.is_dir() and (direct / "input").is_dir() and (direct / "ground.truth").is_dir():
            return direct

        if not self.cases_root.is_dir():
            raise FileNotFoundError(f"Cases root not found: {self.cases_root}")

        all_cases = self._iter_case_dirs(self.cases_root)
        norm_case_name = case_name.replace("\\", "/")

        exact_rel = [p for p in all_cases if str(p.relative_to(self.cases_root)).replace("\\", "/") == norm_case_name]
        if len(exact_rel) == 1:
            return exact_rel[0]
        if len(exact_rel) > 1:
            options = ", ".join(sorted([str(p.relative_to(self.cases_root)).replace("\\", "/") for p in exact_rel]))
            raise ValueError(f"Case name '{case_name}' is ambiguous. Available matches: {options}")

        by_leaf = [p for p in all_cases if p.name == case_name]
        if not by_leaf:
            raise FileNotFoundError(f"Case '{case_name}' not found under {self.cases_root}")
        if len(by_leaf) > 1:
            options = ", ".join(sorted([str(p.relative_to(self.cases_root)).replace('\\', '/') for p in by_leaf]))
            raise ValueError(
                f"Case name '{case_name}' is ambiguous. Specify benchmark_name. Available matches: {options}")
        return by_leaf[0]

    def _tabular_files(self, input_dir: Path) -> List[Path]:
        """
        Вернуть отсортированный список табличных файлов из ``input`` директории.

        Parameters
        ----------
        input_dir : Path
            Директория с входными данными.

        Returns
        -------
        List[Path]
            Файлы с расширениями ``{.txt, .dat, .data, .csv, .tsv, .cov}``.
        """
        allowed = {".txt", ".dat", ".data", ".csv", ".tsv", ".cov"}
        return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed])

    def _build_loader_and_specs(self, case_dir: Path) -> Tuple[object, List[Tuple[str, str]]]:
        """
        Создать загрузчик и спецификацию загрузки для директории кейса.

        Parameters
        ----------
        case_dir : Path
            Путь к директории кейса (абсолютный или относительный).

        Returns
        -------
        Tuple[object, List[Tuple[str, str]]]
            Пара ``(loader, load_specs)``, где:
            - ``loader`` — экземпляр загрузчика из ``LoaderRegistry.create(...)``;
            - ``load_specs`` — список кортежей ``(dataset_key, load_name)``.
              ``dataset_key`` используется как ключ в результирующем словаре,
              ``load_name`` передается в ``loader.load_data(load_name)``.

        Raises
        ------
        FileNotFoundError
            Если для общего случая бенчмарка не найдены поддерживаемые входные файлы.
        """
        case_dir = case_dir.resolve()
        benchmark = case_dir.relative_to(self.cases_root).parts[0].lower()
        input_dir = case_dir / "input"
        gt_dir = case_dir / "ground.truth"
        case_leaf = case_dir.name

        if benchmark == "lucas":
            loader = LoaderRegistry.create(
                "lucas",
                data_dir=str(input_dir / "{name}_train.data"),
                targets_dir=str(input_dir / "{name}_train.targets"),
                ground_truth_dir=str(gt_dir / "{name}.ground.truth.graph.txt"),
            )
            bases = sorted(
                {
                    p.name[:-11]
                    for p in input_dir.glob("*_train.data")
                    if (input_dir / f"{p.name[:-11]}_train.targets").is_file()
                }
            )
            if not bases:
                bases = [case_leaf]
            return loader, [(base, base) for base in bases]

        if benchmark == "tetrad":
            loader = LoaderRegistry.create(
                "tetrad",
                data_dir=str(input_dir),
                ground_truth_dir=str(gt_dir),
            )
            raw_names = sorted(
                [
                    p.name[5:].rsplit(".", 1)[0]
                    for p in input_dir.iterdir()
                    if p.is_file() and p.name.startswith("data_")
                ]
            )
            if not raw_names:
                short = case_leaf.replace("tetrad_", "", 1)
                raw_names = [short]
            return loader, [(f"data_{name}", name) for name in raw_names]

        input_files = self._supported_files(input_dir)
        if not input_files:
            raise FileNotFoundError(f"No supported input files found in: {input_dir}")

        loader = LoaderRegistry.create(
            benchmark,
            data_dir=str(input_dir / "{name}"),
            ground_truth_dir=str(gt_dir / "{name}"),
        )
        return loader, [(p.name, p.name) for p in input_files]