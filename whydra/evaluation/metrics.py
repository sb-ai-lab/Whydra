import numpy as np
from causallearn.graph.SHD import SHD
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.DAG2CPDAG import dag2cpdag

from .graph_utils import get_adj_matrix


def convert_pag_to_cpdag_proxy(adj_mat):
    """
    Преобразует матрицу PAG (с кружочками) в прокси-матрицу CPDAG для расчета метрик.
    Логика:
    1. CIRCLE (2) -> TAIL (-1).
       Мы трактуем неопределенность 'o' как отсутствие стрелки (хвост).
       o-> превращается в -->
       o-o превращается в ---
    2. ARROW (1) остается ARROW.
    3. TAIL (-1) остается TAIL.
    """
    # Копируем, чтобы не менять оригинал
    mat = adj_mat.copy()

    # Endpoint.CIRCLE.value обычно равен 2 в causal-learn
    # Endpoint.TAIL.value равен -1

    # Заменяем все кружочки (2) на хвосты (-1)
    mat[mat == Endpoint.CIRCLE.value] = Endpoint.TAIL.value

    return mat


def f1_score(prec, rec):
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def calculate_metrics(true_G: GeneralGraph, est_G: GeneralGraph):
    """
    Считает полный набор метрик качества.
    """

    # 1. Получаем матрицу предсказанного графа
    raw_est_mat = get_adj_matrix(est_G)

    # Проверяем, является ли граф PAG-ом (есть ли кружочки)
    is_pag = np.any(raw_est_mat == Endpoint.CIRCLE.value)

    # Конвертируем PAG в прокси-CPDAG для корректного сравнения ориентаций
    if is_pag:
        est_mat = convert_pag_to_cpdag_proxy(raw_est_mat)
    else:
        est_mat = raw_est_mat

    # 2. Подготовка Истинного Графа (Ground Truth)
    # Всегда конвертируем истинный DAG в CPDAG (класс марковской эквивалентности),
    # чтобы сравнивать ориентации с учётом марковской эквивалентности.
    # Мы НЕ используем dag2pag, так как он вызывает зависание.
    true_class_G = dag2cpdag(true_G)
    true_mat = get_adj_matrix(true_class_G)

    # 3. SHD (Structural Hamming Distance)
    # SHD умеет работать с разными типами графов, передаем оригинальные объекты
    # Но если est_G это PAG, а true_class_G это CPDAG, SHD может насчитать лишнего за кружочки.
    # Однако это лучше, чем зависание.
    try:
        shd = SHD(true_class_G, est_G).get_shd()
    except Exception:
        # Фолбек на простой подсчет разницы матриц, если библиотека упадет
        diff = np.abs(true_mat - est_mat)
        diff[diff > 0] = 1
        shd = np.sum(diff) / 2

    # 4. Adjacency (Скелет)
    # Ребро есть, если эндпоинт не NULL (0)
    true_skel = (true_mat != 0)
    est_skel = (est_mat != 0)

    tp_adj = np.sum(true_skel & est_skel) / 2
    fp_adj = np.sum((~true_skel) & est_skel) / 2
    fn_adj = np.sum(true_skel & (~est_skel)) / 2

    adj_prec = tp_adj / (tp_adj + fp_adj) if (tp_adj + fp_adj) > 0 else 0
    adj_rec = tp_adj / (tp_adj + fn_adj) if (tp_adj + fn_adj) > 0 else 0
    adj_f1 = f1_score(adj_prec, adj_rec)

    # 5. Orientation (Arrows)
    # Стрелка есть, если endpoint == ARROW (1)
    # Благодаря convert_pag_to_cpdag_proxy, o-> (2, 1) стало --> (-1, 1),
    # поэтому стрелка на конце (1) совпадет с истиной.
    true_arrows = (true_mat == Endpoint.ARROW.value)
    est_arrows = (est_mat == Endpoint.ARROW.value)

    tp_arr = np.sum(true_arrows & est_arrows)
    fp_arr = np.sum((~true_arrows) & est_arrows)
    fn_arr = np.sum(true_arrows & (~est_arrows))

    arrow_prec = tp_arr / (tp_arr + fp_arr) if (tp_arr + fp_arr) > 0 else 0
    arrow_rec = tp_arr / (tp_arr + fn_arr) if (tp_arr + fn_arr) > 0 else 0
    arrow_f1 = f1_score(arrow_prec, arrow_rec)

    # 6. Orientation (Arrows) - Только на правильно найденных ребрах (Common Edges)
    common_edges_mask = true_skel & est_skel

    if np.sum(common_edges_mask) > 0:
        true_arrows_common = true_arrows & common_edges_mask
        est_arrows_common = est_arrows & common_edges_mask

        tp_arr_ce = np.sum(true_arrows_common & est_arrows_common)
        fp_arr_ce = np.sum((~true_arrows_common) & est_arrows_common)
        fn_arr_ce = np.sum(true_arrows_common & (~est_arrows_common))

        arrow_prec_ce = tp_arr_ce / (tp_arr_ce + fp_arr_ce) if (tp_arr_ce + fp_arr_ce) > 0 else 0
        arrow_rec_ce = tp_arr_ce / (tp_arr_ce + fn_arr_ce) if (tp_arr_ce + fn_arr_ce) > 0 else 0
        arrow_f1_ce = f1_score(arrow_prec_ce, arrow_rec_ce)
    else:
        arrow_prec_ce = 0.0
        arrow_rec_ce = 0.0
        arrow_f1_ce = 0.0

    # 7. Дополнительные метрики
    missing_edges_share = 1.0 - adj_rec
    false_edges_share = 1.0 - adj_prec

    # 8. Статистика графа
    n_edges = np.sum(est_skel) / 2
    degrees = np.sum(est_skel, axis=1)
    max_degree = np.max(degrees) if len(degrees) > 0 else 0

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
        "missing_edges_share": missing_edges_share,
        "false_edges_share": false_edges_share,
        "n_edges": n_edges,
        "max_degree": max_degree
    }
