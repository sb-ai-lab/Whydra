"""
Единый модуль метрик и конвертаций для каузальных графов.

Это единственное место, где определены:
  - Конвертация результата алгоритма → endpoint-матрица
  - Конвертация DAG → CPDAG (ground truth)
  - Конвертация PAG → CPDAG proxy
  - Вычисление всех метрик (SHD, adj, arrow, common edges)

Бенчмарки (benchmark_parallelism.py, benchmark_metrics.py) импортируют отсюда.
"""

import numpy as np
from whydra.algorithms.graph_core import GeneralGraph, Node as GraphNode, Edge, Endpoint

from whydra.background_knowledge import MatrixEncoding, endpoint_codes
from whydra.evaluation.graph_utils import default_node_names, get_adj_matrix


# ============================================================
#  КОНВЕРТАЦИЯ РЕЗУЛЬТАТОВ АЛГОРИТМОВ → ENDPOINT-МАТРИЦА
# ============================================================

def result_to_endpoint_matrix(
    result,
    node_names: list,
    encoding: MatrixEncoding = MatrixEncoding.STANDARD,
) -> np.ndarray:
    """
    Универсальная конвертация результата любого алгоритма в endpoint-матрицу.

    Возвращает матрицу n×n, где mat[i,j] — endpoint у узла j на ребре от i к j.
    Кодировка результата: TAIL=-1, NULL=0, ARROW=1, CIRCLE=2.

    Поддерживает:
      - np.ndarray: PAG-матрица. Все алгоритмы библиотеки отдают кодировку
        causal-learn, она же по умолчанию. ``encoding`` нужен только для
        матриц из внешних источников (например, ``MatrixEncoding.PCALG`` из R).
      - GeneralGraph: CPDAG из локального PC
      - ResultWrapper (объект с атрибутом .G → GeneralGraph): RAI
    """
    n = len(node_names)

    if isinstance(result, np.ndarray) and result.ndim == 2:
        null, tail, arrow, circle = endpoint_codes(encoding)
        # Коды исходной кодировки -> внутренние стандартные коды.
        to_endpoint = {null: 0, tail: -1, arrow: 1, circle: 2}
        present = {int(v) for v in np.asarray(result).ravel().tolist()}
        unexpected = present - set(to_endpoint)
        if unexpected:
            raise ValueError(
                f"Endpoint codes {sorted(unexpected)} are not valid for encoding "
                f"{MatrixEncoding(encoding).value!r}; the matrix is probably in the "
                "other encoding — pass `encoding=` explicitly"
            )
        mat = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                mat[i, j] = to_endpoint[int(result[i, j])]
        return mat

    # PC / RAI: GeneralGraph (или ResultWrapper с .G)
    graph_obj = result
    if hasattr(result, "G"):
        graph_obj = result.G

    if hasattr(graph_obj, "get_nodes") and hasattr(graph_obj, "get_graph_edges"):
        return get_adj_matrix(graph_obj)

    raise TypeError(
        f"Не удалось конвертировать результат типа {type(result)} в endpoint-матрицу. "
        f"Ожидается np.ndarray (PAG), GeneralGraph или объект с .G"
    )


def is_pag_result(result) -> bool:
    """Определяет, является ли результат PAG-матрицей (FCI/FCI_v2)."""
    return isinstance(result, np.ndarray) and result.ndim == 2


# ============================================================
#  КОНВЕРТАЦИЯ GROUND TRUTH: DAG → CPDAG
# ============================================================

def dag_adj_to_cpdag_endpoint_matrix(dag_adj: np.ndarray) -> np.ndarray:
    """
    Конвертирует бинарную DAG-матрицу (adj[i,j]=1 → i→j) в endpoint-матрицу CPDAG.

    Строит CPDAG локально: сначала сохраняется скелет DAG, затем фиксируются
    незащёкленные коллайдеры и применяются правила Мика к обратимым рёбрам.
    """
    n = dag_adj.shape[0]

    dag = np.asarray(dag_adj)
    if dag.ndim != 2 or dag.shape[0] != dag.shape[1]:
        raise ValueError("dag_adj must be a square matrix")
    unexpected = set(np.unique(dag).tolist()) - {0, 1}
    if unexpected:
        raise ValueError(
            f"dag_adj must be a binary DAG matrix (0/1, adj[i,j]=1 means i->j), "
            f"got unexpected values {sorted(unexpected)} — looks like an endpoint "
            "matrix (TAIL=-1/ARROW=1/CIRCLE=2) was passed instead"
        )
    skeleton = (dag != 0) | (dag.T != 0)
    mat = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if skeleton[i, j]:
                mat[i, j] = mat[j, i] = Endpoint.TAIL.value

    def adjacent(i, j):
        return bool(skeleton[i, j])

    def undirected(i, j):
        return (
            mat[i, j] == Endpoint.TAIL.value
            and mat[j, i] == Endpoint.TAIL.value
        )

    def directed(i, j):
        return (
            mat[i, j] == Endpoint.ARROW.value
            and mat[j, i] == Endpoint.TAIL.value
        )

    def orient(i, j):
        if undirected(i, j):
            mat[i, j] = Endpoint.ARROW.value
            mat[j, i] = Endpoint.TAIL.value
            return True
        return False

    # Unshielded colliders are invariant across a Markov-equivalence class.
    for y in range(n):
        parents = [x for x in range(n) if dag[x, y] != 0]
        for x in range(len(parents)):
            for z in parents[x + 1:]:
                if not adjacent(parents[x], z):
                    orient(parents[x], y)
                    orient(z, y)

    # Meek R1-R3. Repeating the pass reaches the fixed point for the rules
    # needed by DAG -> CPDAG conversion without an external graph package.
    changed = True
    while changed:
        changed = False
        for a in range(n):
            for b in range(n):
                if not directed(a, b):
                    continue
                for c in range(n):
                    if c in (a, b) or not undirected(b, c):
                        continue
                    if not adjacent(a, c):
                        changed |= orient(b, c)       # R1

        for a in range(n):
            for b in range(n):
                if not directed(a, b):
                    continue
                for c in range(n):
                    if c in (a, b) or not directed(b, c):
                        continue
                    if undirected(a, c):
                        changed |= orient(a, c)       # R2

        for a in range(n):
            for b in range(n):
                if not undirected(a, b):
                    continue
                for c in range(n):
                    if c in (a, b) or not undirected(a, c) or adjacent(b, c):
                        continue
                    for d in range(n):
                        if d in (a, b, c):
                            continue
                        if directed(b, d) and directed(c, d):
                            changed |= orient(a, d)  # R3
    return mat


def dag_adj_to_general_graph(dag_adj: np.ndarray, node_names: list = None) -> GeneralGraph:
    """Конвертирует бинарную DAG-матрицу в объект GeneralGraph."""
    n = dag_adj.shape[0]
    if node_names is None:
        node_names = default_node_names(n)

    nodes = [GraphNode(name) for name in node_names]
    g = GeneralGraph(nodes)
    for i in range(n):
        for j in range(n):
            if dag_adj[i, j] != 0:
                g.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW))
    return g


# ============================================================
#  КОНВЕРТАЦИЯ PAG → CPDAG PROXY
# ============================================================

def convert_pag_to_cpdag_proxy(adj_mat: np.ndarray) -> np.ndarray:
    """
    Преобразует endpoint-матрицу PAG (с кружочками) в прокси-матрицу CPDAG.

    Логика:
      CIRCLE (2) → TAIL (-1): трактуем неопределенность 'o' как хвост.
      o→ превращается в →,  o-o превращается в —
      ARROW (1) остаётся ARROW, TAIL (-1) остаётся TAIL.
    """
    mat = adj_mat.copy()
    mat[mat == Endpoint.CIRCLE.value] = Endpoint.TAIL.value
    return mat


# ============================================================
#  МЕТРИКИ КАЧЕСТВА
# ============================================================

def f1_score(prec, rec):
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def compute_metrics_from_endpoint_matrices(
    true_ep: np.ndarray,
    est_ep: np.ndarray,
) -> dict:
    """
    Вычисляет полный набор метрик из двух endpoint-матриц (обе в кодировке TAIL/NULL/ARROW/CIRCLE).

    Обе матрицы должны быть в одном пространстве (обе CPDAG или обе после PAG→proxy).
    Вызывающий код отвечает за предварительные конвертации (DAG→CPDAG, PAG→proxy).
    """
    n = true_ep.shape[0]

    # --- Adjacency (Скелет) ---
    true_skel = (true_ep != 0)
    est_skel = (est_ep != 0)

    # Симметризуем: ребро есть, если хотя бы один endpoint ненулевой
    true_skel_sym = true_skel | true_skel.T
    est_skel_sym = est_skel | est_skel.T
    np.fill_diagonal(true_skel_sym, False)
    np.fill_diagonal(est_skel_sym, False)

    # Считаем по верхнему треугольнику, чтобы каждое ребро — один раз
    triu = np.triu_indices(n, k=1)
    true_edges = true_skel_sym[triu]
    est_edges = est_skel_sym[triu]

    tp_adj = int(np.sum(true_edges & est_edges))
    fp_adj = int(np.sum(~true_edges & est_edges))
    fn_adj = int(np.sum(true_edges & ~est_edges))

    adj_prec = tp_adj / (tp_adj + fp_adj) if (tp_adj + fp_adj) > 0 else 0.0
    adj_rec = tp_adj / (tp_adj + fn_adj) if (tp_adj + fn_adj) > 0 else 0.0
    adj_f1 = f1_score(adj_prec, adj_rec)

    # --- Arrow (ориентация) ---
    true_arrows = (true_ep == Endpoint.ARROW.value)
    est_arrows = (est_ep == Endpoint.ARROW.value)

    tp_arr = int(np.sum(true_arrows & est_arrows))
    fp_arr = int(np.sum(~true_arrows & est_arrows))
    fn_arr = int(np.sum(true_arrows & ~est_arrows))

    arrow_prec = tp_arr / (tp_arr + fp_arr) if (tp_arr + fp_arr) > 0 else 0.0
    arrow_rec = tp_arr / (tp_arr + fn_arr) if (tp_arr + fn_arr) > 0 else 0.0
    arrow_f1 = f1_score(arrow_prec, arrow_rec)

    # --- Arrow на общих рёбрах (Common Edges) ---
    common_edges_mask = true_skel & est_skel
    if np.sum(common_edges_mask) > 0:
        true_arrows_ce = true_arrows & common_edges_mask
        est_arrows_ce = est_arrows & common_edges_mask
        tp_arr_ce = int(np.sum(true_arrows_ce & est_arrows_ce))
        fp_arr_ce = int(np.sum(~true_arrows_ce & est_arrows_ce))
        fn_arr_ce = int(np.sum(true_arrows_ce & ~est_arrows_ce))
        arrow_prec_ce = tp_arr_ce / (tp_arr_ce + fp_arr_ce) if (tp_arr_ce + fp_arr_ce) > 0 else 0.0
        arrow_rec_ce = tp_arr_ce / (tp_arr_ce + fn_arr_ce) if (tp_arr_ce + fn_arr_ce) > 0 else 0.0
        arrow_f1_ce = f1_score(arrow_prec_ce, arrow_rec_ce)
    else:
        arrow_prec_ce = arrow_rec_ce = arrow_f1_ce = 0.0

    # --- SHD ---
    # Считается напрямую, а не через внешнюю реализацию SHD: метрика ниже —
    # число пар узлов, у которых расходится хотя бы один конец ребра.
    # Это не то же самое, что SHD из causal-learn, и сравнивать значения
    # между библиотеками нельзя.
    shd = 0
    for i in range(n):
        for j in range(i + 1, n):
            if true_ep[i, j] != est_ep[i, j] or true_ep[j, i] != est_ep[j, i]:
                shd += 1

    # --- Доп. статистики ---
    n_edges = int(np.sum(est_edges))
    degrees = np.sum(est_skel_sym, axis=1)
    max_degree = int(np.max(degrees)) if len(degrees) > 0 else 0

    return {
        "SHD": shd,
        "adj_f1": adj_f1,
        "arrow_f1": arrow_f1,
        "arrow_f1_common_edges": arrow_f1_ce,
        "adj_precision": adj_prec,
        "adj_recall": adj_rec,
        "arrow_precision": arrow_prec,
        "arrow_recall": arrow_rec,
        "arrow_precision_common_edges": arrow_prec_ce,
        "arrow_recall_common_edges": arrow_rec_ce,
        # При пустом знаменателе доля не определена, а не равна единице:
        # идеальный ответ на графе без рёбер — это 0% потерь, не 100%.
        "missing_edges_share": (1.0 - adj_rec) if (tp_adj + fn_adj) > 0 else 0.0,
        "false_edges_share": (1.0 - adj_prec) if (tp_adj + fp_adj) > 0 else 0.0,
        "n_edges": n_edges,
        "max_degree": max_degree,
    }


def compute_metrics_from_dag_and_result(
    true_dag_adj: np.ndarray,
    est_ep_mat: np.ndarray,
    is_pag: bool = False,
) -> dict:
    """
    Удобная обёртка: принимает DAG ground truth и endpoint-матрицу результата,
    сама делает DAG→CPDAG и PAG→proxy конвертации, возвращает метрики.

    Parameters
    ----------
    true_dag_adj : бинарная DAG-матрица (adj[i,j]=1 → i→j)
    est_ep_mat : endpoint-матрица (TAIL=-1, NULL=0, ARROW=1, CIRCLE=2)
    is_pag : если True, est_ep_mat — это PAG, нужно конвертировать в CPDAG proxy
    """
    true_ep = dag_adj_to_cpdag_endpoint_matrix(true_dag_adj)
    est_ep = convert_pag_to_cpdag_proxy(est_ep_mat) if is_pag else est_ep_mat
    return compute_metrics_from_endpoint_matrices(true_ep, est_ep)


# ============================================================
#  ОБРАТНАЯ СОВМЕСТИМОСТЬ: calculate_metrics(true_G, est_G)
# ============================================================

def calculate_metrics(true_G: GeneralGraph, est_G: GeneralGraph) -> dict:
    """
    Оригинальный API для основного пайплайна (main.py, wrappers.py).
    Принимает два объекта GeneralGraph, возвращает метрики.
    """
    raw_est_mat = get_adj_matrix(est_G)
    is_pag = np.any(raw_est_mat == Endpoint.CIRCLE.value)
    est_mat = convert_pag_to_cpdag_proxy(raw_est_mat) if is_pag else raw_est_mat

    true_dag_adj = (get_adj_matrix(true_G) == Endpoint.ARROW.value).astype(int)
    true_mat = dag_adj_to_cpdag_endpoint_matrix(true_dag_adj)

    return compute_metrics_from_endpoint_matrices(true_mat, est_mat)
