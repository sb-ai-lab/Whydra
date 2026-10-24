"""Тесты публичного API: реестр алгоритмов, загрузка данных, контракт обёрток.

Это поверхность, которую видит пользователь библиотеки. Аудит показал, что
именно здесь расхождения дороже всего: молча потерянное наблюдение при
загрузке, параметр, роняющий половину алгоритмов, и кодировка результата,
выбираемая базовым классом.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import whydra
import whydra.algorithms.wrappers as wrappers_module
from whydra.algorithms.base import CausalAlgorithm
from whydra.algorithms.wrappers import (
    _load_data_like_main,
    _text_table_has_header,
    available_standalone_algorithms,
    create_standalone_algorithm,
    run_standalone_algorithm,
)
from whydra.background_knowledge import MatrixEncoding
from whydra.evaluation.graph_utils import get_adj_matrix

NULL, TAIL, ARROW, CIRCLE = 0, -1, 1, 2

#: Алгоритмы, доступные без тяжёлых опциональных зависимостей.
def _available():
    keys = []
    for name in available_standalone_algorithms():
        try:
            create_standalone_algorithm(name, alpha=0.05, n_jobs=1)
        except ImportError:
            continue
        keys.append(name)
    return keys


AVAILABLE = _available()


def _common_algorithm_kwargs(algo_key, **overrides):
    params = {
        "alpha": 0.05,
        "n_jobs": 1,
        "verbose": False,
        "show_progress": False,
    }
    # При наличии torch DagGNN входит в AVAILABLE. Ограничиваем обучение,
    # чтобы контрактные тесты не запускали полный 300-epoch benchmark.
    if str(algo_key).lower() == "daggnn":
        params.update(epochs=1, batch_size=400, k_max_iter=1)
    params.update(overrides)
    return params


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((400, 5))
    for a, b in [(0, 2), (1, 2), (2, 3), (3, 4)]:
        X[:, b] += 1.2 * X[:, a]
    return X


# ============================================================
#  Реестр алгоритмов
# ============================================================

def test_registry_is_not_empty_and_names_are_class_names():
    names = available_standalone_algorithms()
    assert len(names) >= 13
    assert all(n[0].isupper() for n in names)


@pytest.mark.parametrize("alias,expected", [
    ("pc_stable", "StandalonePCStable"),
    ("PC_STABLE", "StandalonePCStable"),
    ("pc-stable", "StandalonePCStable"),
    (" pc stable ", "StandalonePCStable"),
    ("fci_stable", "StandaloneFCIStable"),
    ("rfci_stable", "StandaloneRFCIStable"),
    ("gfci_stable", "StandaloneGFCIStable"),
    ("rai_optimized", "StandaloneRAIOptimized"),
    ("StandalonePCStable", "StandalonePCStable"),
])
def test_aliases_resolve_case_and_separator_insensitively(alias, expected):
    assert type(create_standalone_algorithm(alias)).__name__ == expected


def test_unknown_algorithm_lists_the_available_ones():
    with pytest.raises(ValueError, match="Unknown standalone algorithm"):
        create_standalone_algorithm("не существует")


def test_optional_dependencies_fail_with_a_readable_message():
    """Отсутствие тяжёлой зависимости не должно ломать импорт пакета."""
    optional = set(available_standalone_algorithms()) - set(AVAILABLE)
    for name in optional:
        with pytest.raises(ImportError, match="requires"):
            create_standalone_algorithm(name)


def test_every_wrapper_subclasses_the_base_contract():
    for name in AVAILABLE:
        assert isinstance(create_standalone_algorithm(name), CausalAlgorithm)


# ============================================================
#  Контракт параметров
# ============================================================

@pytest.mark.parametrize("algo_key", AVAILABLE)
def test_common_parameters_are_accepted_by_every_algorithm(algo_key, data):
    """alpha / n_jobs / verbose / show_progress обязаны приниматься всеми."""
    algo = create_standalone_algorithm(algo_key, **_common_algorithm_kwargs(algo_key))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = algo.run(data)
    assert result is not None


@pytest.mark.parametrize("algo_key", AVAILABLE)
def test_result_is_convertible_to_an_endpoint_matrix(algo_key, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = create_standalone_algorithm(
            algo_key, **_common_algorithm_kwargs(algo_key)
        ).run(data)
    graph = result if hasattr(result, "get_graph_edges") else result.G
    m = get_adj_matrix(graph)
    assert m.shape == (data.shape[1], data.shape[1])
    assert np.all(np.diag(m) == NULL)


@pytest.mark.parametrize("algo_key", AVAILABLE)
def test_every_result_uses_the_library_encoding(algo_key, data):
    """Одна кодировка на всю библиотеку — коды вне {0,-1,1,2} недопустимы."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = create_standalone_algorithm(
            algo_key, **_common_algorithm_kwargs(algo_key)
        ).run(data)
    graph = result if hasattr(result, "get_graph_edges") else result.G
    values = set(int(v) for v in get_adj_matrix(graph).ravel())
    assert values <= {NULL, TAIL, ARROW, CIRCLE}, f"{algo_key}: чужие коды {values}"


@pytest.mark.parametrize("indep_test,expected_type", [
    ("fisherz", "continuous"),
    ("chisq", "discrete"),
])
def test_daggnn_common_parameters_are_not_forwarded_to_runner(
    monkeypatch, data, indep_test, expected_type,
):
    """Общий контракт обёртки не должен становиться kwargs контракта runner."""
    received = {}

    def fake_train_dag_gnn(
        values,
        *,
        data_type,
        epochs,
        batch_size,
        lr,
        graph_threshold,
        encoder_hidden,
        decoder_hidden,
        k_max_iter=100,
    ):
        received.update({
            "data_type": data_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "graph_threshold": graph_threshold,
            "encoder_hidden": encoder_hidden,
            "decoder_hidden": decoder_hidden,
            "k_max_iter": k_max_iter,
        })
        return np.zeros((values.shape[1], values.shape[1]))

    monkeypatch.setattr(wrappers_module, "DAG_GNN_IMPORT_ERROR", None)
    monkeypatch.setattr(
        wrappers_module,
        "dagnn_runner",
        SimpleNamespace(train_dag_gnn=fake_train_dag_gnn),
    )

    algo = wrappers_module.DagGNN(
        indep_test=indep_test,
        alpha=0.05,
        n_jobs=2,
        verbose=True,
        show_progress=True,
        k_max_iter=1,
    )
    graph = algo.run(data)

    assert received["k_max_iter"] == 1
    assert received["data_type"] == expected_type
    assert graph.get_graph_edges() == []


def test_daggnn_rejects_unknown_runner_parameter(monkeypatch):
    monkeypatch.setattr(wrappers_module, "DAG_GNN_IMPORT_ERROR", None)
    with pytest.raises(TypeError, match="Unsupported DagGNN parameters: typo"):
        wrappers_module.DagGNN(typo=True)


def test_pag_wrappers_declare_their_output_encoding():
    """Кодировка объявляется классом, а не наследуется по случайности."""
    from whydra.algorithms.wrappers import _PagMatrixResult

    for name in AVAILABLE:
        algo = create_standalone_algorithm(name)
        if isinstance(algo, _PagMatrixResult):
            assert algo.OUTPUT_ENCODING is MatrixEncoding.STANDARD, name


def test_matrix_result_wrapper_rejects_an_undeclared_encoding():
    from whydra.algorithms.wrappers import _PagMatrixResult

    class Broken(_PagMatrixResult):
        OUTPUT_ENCODING = None

    with pytest.raises(NotImplementedError, match="OUTPUT_ENCODING"):
        Broken()._matrix_to_graph(np.zeros((2, 2), dtype=int), 2)


# ============================================================
#  Загрузка данных
# ============================================================

@pytest.mark.parametrize("first_line,expected", [
    ("A B C", True),
    ("1.0 2.0 3.0", False),
    ("1,2,3", False),
    ("x1;x2;x3", True),
    ("-1 0.5 1e3", False),
])
def test_header_detection(tmp_path, first_line, expected):
    f = tmp_path / "t.txt"
    f.write_text(f"{first_line}\n1 2 3\n4 5 6\n", encoding="utf-8")
    assert _text_table_has_header(f) is expected


def test_headerless_file_keeps_every_observation(tmp_path):
    """Безусловный skiprows=1 молча терял первое наблюдение."""
    f = tmp_path / "nh.txt"
    f.write_text("1.0 2.0\n3.0 4.0\n5.0 6.0\n", encoding="utf-8")
    a = _load_data_like_main(f)
    assert a.shape == (3, 2)
    assert a[0].tolist() == [1.0, 2.0]


def test_file_with_header_drops_only_the_header(tmp_path):
    f = tmp_path / "h.txt"
    f.write_text("A B\n1.0 2.0\n3.0 4.0\n5.0 6.0\n", encoding="utf-8")
    a = _load_data_like_main(f)
    assert a.shape == (3, 2)
    assert a[0].tolist() == [1.0, 2.0]


def test_explicit_header_flag_overrides_detection(tmp_path):
    f = tmp_path / "nh.txt"
    f.write_text("1.0 2.0\n3.0 4.0\n5.0 6.0\n", encoding="utf-8")
    assert _load_data_like_main(f, header=True).shape == (2, 2)
    assert _load_data_like_main(f, header=False).shape == (3, 2)


def test_npy_roundtrip(tmp_path):
    f = tmp_path / "d.npy"
    arr = np.arange(6, dtype=float).reshape(3, 2)
    np.save(f, arr)
    assert np.array_equal(_load_data_like_main(f), arr)


def test_array_is_passed_through_unchanged():
    arr = np.zeros((3, 2))
    assert _load_data_like_main(arr) is arr


def test_missing_file_and_wrong_type_are_reported():
    with pytest.raises(FileNotFoundError):
        _load_data_like_main("нет-такого-файла.txt")
    with pytest.raises(TypeError, match="must be np.ndarray"):
        _load_data_like_main(42)


def test_pickle_must_contain_an_array(tmp_path):
    import pickle

    f = tmp_path / "bad.pkl"
    f.write_bytes(pickle.dumps({"не": "массив"}))
    with pytest.raises(ValueError, match="does not contain np.ndarray"):
        _load_data_like_main(f)


def test_run_standalone_algorithm_from_a_file(tmp_path, data):
    f = tmp_path / "d.npy"
    np.save(f, data)
    graph = run_standalone_algorithm("pc_stable", f, alpha=0.05, n_jobs=1,
                                     show_progress=False)
    assert graph.get_nodes()


# ============================================================
#  Пакет
# ============================================================

def test_public_names_are_exported():
    for name in ["BackgroundKnowledge", "BKPhase", "MatrixEncoding",
                 "BKApplicationReport", "BKEvent", "BKConflictWarning",
                 "run_standalone_algorithm", "BenchmarkCaseRepository"]:
        assert hasattr(whydra, name), f"{name} не экспортирован"


def test_all_is_consistent_with_the_module():
    for name in whydra.__all__:
        assert hasattr(whydra, name), f"__all__ обещает {name}, которого нет"


def test_loader_registry_is_populated_on_import():
    from whydra.data import LoaderRegistry

    assert set(LoaderRegistry.available()) >= {"bnlearn", "lucas", "tetrad"}
    with pytest.raises(ValueError, match="Unknown loader"):
        LoaderRegistry.create("не-существует")
