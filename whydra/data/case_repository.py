from pathlib import Path

import numpy as np
import pandas as pd

from .base import DatasetLoader
from .loader_registry import LoaderRegistry


class BenchmarkCaseRepository:
    """
    Find benchmark cases and load their datasets and ground-truth graphs.

    Parameters
    ----------
    cases_root : str, optional
        Root directory for benchmarks. Cases use the following layout:

        - ``{cases_root}/{benchmark}/{case}/input``
        - ``{cases_root}/{benchmark}/{case}/ground.truth``

    Notes
    -----
    A valid case directory contains both an ``input`` subdirectory and a
    ``ground.truth`` subdirectory.
    """

    def __init__(self, cases_root: str | None = None):
        """
        Set the root directory for benchmark cases.

        Parameters
        ----------
        cases_root : str, optional
            Path to the data directory. Defaults to ``whydra_benchmarks/cases``
            relative to the current working directory.
        """
        self.cases_root = (
            Path(cases_root)
            if cases_root is not None
            else Path("whydra_benchmarks/cases")
        ).resolve()

    def get_case_metadata(self, case_name, benchmark_name=None):
        """Return the declared family model and input-column order, including case.json overrides.

        Headerless/unknown families must supply metadata; correspondence to
        ground-truth node order is never guessed by the metric layer.
        """
        import json

        case = self._resolve_case_dir(case_name, benchmark_name)
        family = case.relative_to(self.cases_root).parts[0]
        result: dict[str, object] = {"family": family, "truth_kind": "dag"}
        if family in {"bnlearn", "lucas"} or (
            family == "tetrad" and "discrete" in case.name
        ):
            result["data_type"] = "categorical"
        elif family == "feedback" or (family == "tetrad" and "linear" in case.name):
            result["data_type"] = "continuous"
        if family == "lucas":
            # LucasLoader explicitly concatenates target first, then features.
            result["node_names"] = [f"X{i + 1}" for i in range(12)]
        else:
            files = self._tabular_files(case / "input")
            if files:
                frame = pd.read_csv(files[0], sep=r"\s+", nrows=0)
                result["node_names"] = list(frame.columns)
        metadata = case / "case.json"
        if metadata.exists():
            result.update(json.loads(metadata.read_text(encoding="utf-8")))
        if result.get("data_type") not in {"continuous", "categorical"}:
            msg = f"Case {case_name} needs data_type metadata in case.json"
            raise ValueError(msg)
        return result

    def list_available_cases(self, benchmark_name: str | None = None) -> list[str]:
        """
        List available benchmark cases.

        Parameters
        ----------
        benchmark_name : str, optional
            If supplied, list only cases in this benchmark, using paths
            relative to its directory. Otherwise, list cases from every
            benchmark using paths relative to ``cases_root``.

        Returns
        -------
        List[str]
            Case identifiers in sorted order, using POSIX path separators.

        Raises
        ------
        FileNotFoundError
            If ``benchmark_name`` is supplied and its directory does not exist.
        """
        if benchmark_name is not None:
            root = self.cases_root / benchmark_name
            if not root.is_dir():
                msg = f"Benchmark directory not found: {root}"
                raise FileNotFoundError(msg)
            return sorted(
                [
                    str(p.relative_to(root)).replace("\\", "/")
                    for p in self._iter_case_dirs(root)
                ]
            )

        if not self.cases_root.is_dir():
            return []

        result: list[str] = []
        for benchmark_dir in sorted(
            [p for p in self.cases_root.iterdir() if p.is_dir()]
        ):
            for case_dir in self._iter_case_dirs(benchmark_dir):
                result.append(
                    str(case_dir.relative_to(self.cases_root)).replace("\\", "/")
                )
        return sorted(result)

    def get_case_datasets(
        self, case_name: str | None = None, benchmark_name: str | None = None
    ) -> dict[str, np.ndarray]:
        """
        Load every dataset for a case.

        Parameters
        ----------
        case_name : str, optional
            Either of the following case identifiers:
            - A short case name, such as ``"child"``.
            - A path relative to ``cases_root``, such as ``"bnlearn/child"``.
        benchmark_name : str, optional
            Benchmark name used to resolve the path directly as
            ``{cases_root}/{benchmark_name}/{case_name}``.

        Returns
        -------
        Dict[str, np.ndarray]
            Maps each dataset key to the array returned by the selected loader.

        Raises
        ------
        ValueError
            If ``case_name`` is missing.
        FileNotFoundError
            If the case or its dataset input files cannot be found.
        """
        if not case_name:
            msg = "case_name is required, e.g. get_case_datasets(case_name='child', benchmark_name='bnlearn')"
            raise ValueError(msg)
        case_dir = self._resolve_case_dir(case_name, benchmark_name)
        loader, load_specs = self._build_loader_and_specs(case_dir)

        datasets: dict[str, np.ndarray] = {}
        for dataset_key, load_name in load_specs:
            datasets[dataset_key] = loader.load_data(load_name)

        if not datasets:
            msg = f"No dataset files found in: {case_dir / 'input'}"
            raise FileNotFoundError(msg)
        return datasets

    def get_case_data(
        self,
        case_name: str,
        benchmark_name: str | None = None,
        dataset_key: str | None = None,
    ) -> np.ndarray:
        """
        Load and return a single dataset for a case.

        Parameters
        ----------
        case_name : str
            Short case name or path relative to ``cases_root``.
        benchmark_name : str, optional
            Benchmark name used to resolve the path directly as
            ``{cases_root}/{benchmark_name}/{case_name}``.
        dataset_key : str, optional
            Key in the dictionary returned by ``get_case_datasets``. If omitted,
            return the first dataset found.

        Returns
        -------
        np.ndarray
            Data from the selected dataset.

        Raises
        ------
        FileNotFoundError
            If the case, its datasets, or the requested ``dataset_key`` cannot be found.
        """
        try:
            datasets = self.get_case_datasets(
                case_name=case_name, benchmark_name=benchmark_name
            )
        except (FileNotFoundError, ValueError) as e:
            msg = "Case not found"
            raise FileNotFoundError(msg) from e

        if dataset_key is not None:
            if dataset_key not in datasets:
                msg = "Case not found"
                raise FileNotFoundError(msg)
            return datasets[dataset_key]

        return next(iter(datasets.values()))

    def get_case_ground_truth(
        self, case_name: str, benchmark_name: str | None = None
    ) -> str:
        """
        Find the ground-truth file for a case.

        Parameters
        ----------
        case_name : str
            Case identifier.
        benchmark_name : str, optional
            Benchmark name used to resolve the path directly.

        Returns
        -------
        str
            Path to the selected ground-truth file. Prefer filenames containing
            ``"graph"``, ignoring case. Within the matching files, or among all
            supported files if none match, take the first in sorted order.

        Raises
        ------
        FileNotFoundError
            If the case, its ground-truth directory, or a supported file is missing.
        """
        case_dir = self._resolve_case_dir(case_name, benchmark_name)
        gt_dir = case_dir / "ground.truth"
        if not gt_dir.is_dir():
            msg = f"Ground truth directory not found for case '{case_name}': {gt_dir}"
            raise FileNotFoundError(msg)

        files = self._supported_files(gt_dir)
        if not files:
            msg = f"No supported ground truth files found in: {gt_dir}"
            raise FileNotFoundError(msg)

        graph_candidates = sorted([p for p in files if "graph" in p.name.lower()])
        if graph_candidates:
            return str(graph_candidates[0])

        return str(files[0])

    def _supported_files(self, directory: Path) -> list[Path]:
        """
        List supported text and tabular files in sorted order.

        Parameters
        ----------
        directory : Path
            Directory to scan.

        Returns
        -------
        List[Path]
            Files with one of the following extensions:
            ``{.txt, .data, .csv, .dat, .tsv}``.
        """
        allowed_suffixes = (".txt", ".data", ".csv", ".dat", ".tsv")
        return sorted(
            [
                p
                for p in directory.iterdir()
                if p.is_file()
                and p.name.lower().endswith(allowed_suffixes)
                and p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg"}
            ]
        )

    def _iter_case_dirs(self, root: Path) -> list[Path]:
        """
        Find valid case directories recursively under ``root``.

        Parameters
        ----------
        root : Path
            Root directory to search.

        Returns
        -------
        List[Path]
            Directories containing both ``input`` and ``ground.truth``, in sorted order.
        """
        case_dirs = []
        for p in root.rglob("*"):
            if p.is_dir() and (p / "input").is_dir() and (p / "ground.truth").is_dir():
                case_dirs.append(p)
        return sorted(case_dirs)

    def _resolve_case_dir(
        self, case_name: str | None, benchmark_name: str | None
    ) -> Path:
        """
        Find the case directory from the supplied identifiers.

        Parameters
        ----------
        case_name : str, optional
            Short case name or relative path.
        benchmark_name : str, optional
            Benchmark name used to resolve the path directly.

        Returns
        -------
        Path
            Absolute path to the case directory.

        Raises
        ------
        ValueError
            If ``case_name`` is missing or ambiguous.
        FileNotFoundError
            If the root directory or the case cannot be found.
        """
        if not case_name:
            msg = "case_name must not be None"
            raise ValueError(msg)

        if benchmark_name is not None:
            case_dir = (self.cases_root / benchmark_name / case_name).resolve()
            if not case_dir.is_dir():
                msg = f"Case not found: {case_dir}"
                raise FileNotFoundError(msg)
            if not (case_dir / "input").is_dir():
                msg = f"Input directory not found for case '{case_name}': {case_dir / 'input'}"
                raise FileNotFoundError(msg)
            return case_dir

        direct = (self.cases_root / case_name).resolve()
        if (
            direct.is_dir()
            and (direct / "input").is_dir()
            and (direct / "ground.truth").is_dir()
        ):
            return direct

        if not self.cases_root.is_dir():
            msg = f"Cases root not found: {self.cases_root}"
            raise FileNotFoundError(msg)

        all_cases = self._iter_case_dirs(self.cases_root)
        norm_case_name = case_name.replace("\\", "/")

        exact_rel = [
            p
            for p in all_cases
            if str(p.relative_to(self.cases_root)).replace("\\", "/") == norm_case_name
        ]
        if len(exact_rel) == 1:
            return exact_rel[0]
        if len(exact_rel) > 1:
            options = ", ".join(
                sorted(
                    [
                        str(p.relative_to(self.cases_root)).replace("\\", "/")
                        for p in exact_rel
                    ]
                )
            )
            msg = f"Case name '{case_name}' is ambiguous. Available matches: {options}"
            raise ValueError(msg)

        by_leaf = [p for p in all_cases if p.name == case_name]
        if not by_leaf:
            msg = f"Case '{case_name}' not found under {self.cases_root}"
            raise FileNotFoundError(msg)
        if len(by_leaf) > 1:
            options = ", ".join(
                sorted(
                    [
                        str(p.relative_to(self.cases_root)).replace("\\", "/")
                        for p in by_leaf
                    ]
                )
            )
            msg = f"Case name '{case_name}' is ambiguous. Specify benchmark_name. Available matches: {options}"
            raise ValueError(msg)
        return by_leaf[0]

    def _tabular_files(self, input_dir: Path) -> list[Path]:
        """
        List tabular files from the ``input`` directory in sorted order.

        Parameters
        ----------
        input_dir : Path
            Directory containing the input files.

        Returns
        -------
        List[Path]
            Files with extensions ``{.txt, .dat, .data, .csv, .tsv, .cov}``.
        """
        allowed = {".txt", ".dat", ".data", ".csv", ".tsv", ".cov"}
        return sorted(
            [
                p
                for p in input_dir.iterdir()
                if p.is_file() and p.suffix.lower() in allowed
            ]
        )

    def _build_loader_and_specs(
        self, case_dir: Path
    ) -> tuple[DatasetLoader, list[tuple[str, str]]]:
        """
        Create a loader and its loading instructions for a case directory.

        Parameters
        ----------
        case_dir : Path
            Absolute or relative path to the case directory.

        Returns
        -------
        Tuple[object, List[Tuple[str, str]]]
            A pair ``(loader, load_specs)``:
            - ``loader`` is the instance returned by ``LoaderRegistry.create(...)``.
            - ``load_specs`` is a list of ``(dataset_key, load_name)`` pairs.
              ``dataset_key`` identifies the array in the result dictionary.
              ``load_name`` is passed to ``loader.load_data(load_name)``.

        Raises
        ------
        FileNotFoundError
            If a case handled by the generic benchmark loader has no supported input files.
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
            msg = f"No supported input files found in: {input_dir}"
            raise FileNotFoundError(msg)

        loader = LoaderRegistry.create(
            benchmark,
            data_dir=str(input_dir / "{name}"),
            ground_truth_dir=str(gt_dir / "{name}"),
        )
        return loader, [(p.name, p.name) for p in input_files]
