import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx import kamada_kawai_layout

# ==========================================
# 1. НАСТРОЙКА ОКРУЖЕНИЯ И ИМПОРТЫ
# ==========================================

# Добавляем путь к вашей библиотеке SCL-sber
# Предполагаем, что папка SCL-sber лежит рядом со скриптом
sys.path.insert(0, '../')
# Если библиотека установлена через pip install -e ., эта строка не нужна

try:
    from causal_graphs.algorithms.standalone.parallel_pc_stable import pc_stable
    from causal_graphs.algorithms.graph_core import CausalGraph, CIT
    from causal_graphs.algorithms.graph_core.general_graph import GeneralGraph
    from causal_graphs.algorithms.graph_core.endpoints import Endpoint

    print("[OK] Библиотека causal_graphs успешно импортирована.")
except ImportError as e:
    print(f"[ERROR] Не удалось импортировать causal_graphs. Проверьте путь sys.path. Ошибка: {e}")
    sys.exit(1)

# ==========================================
# 2. КОНФИГУРАЦИЯ
# ==========================================

CACHE_DIR = "./cache_cg_flg_sale_flg_2024-05-31"
INPUT_FILE = f"{CACHE_DIR}/graph_ready_dataset_pd_full_parallel.parquet"

target_column = 'sdo_pass_5m'
treatment_column = 'comm_fl'
date_column = 'report_dt'
mediator_column = 'sale_fl'

technical_columns = {
    'epk_id', 'actual_epk_id', 'row_upd_dttm', 'ctl_loading', 'score_uplift'
}

# Параметры алгоритма
ALPHA = 0.01
INDEP_TEST = "fisherz"
N_JOBS = -1  # Использовать все ядра процессора


# ==========================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

def convert_general_graph_to_nx(graph: GeneralGraph):
    """Конвертация из GeneralGraph (causal-learn/SCL) в NetworkX"""
    nx_graph = nx.DiGraph()

    for node in graph.get_nodes():
        nx_graph.add_node(node.get_name())

    for edge in graph.get_graph_edges():
        u = edge.get_node1().get_name()
        v = edge.get_node2().get_name()

        ep1 = edge.get_endpoint1()
        ep2 = edge.get_endpoint2()

        # Направленное ребро u -> v
        if ep1 == Endpoint.TAIL and ep2 == Endpoint.ARROW:
            nx_graph.add_edge(u, v, type='directed')

        # Направленное ребро v -> u
        elif ep1 == Endpoint.ARROW and ep2 == Endpoint.TAIL:
            nx_graph.add_edge(v, u, type='directed')

        # Двунаправленное (конфаундер) u <-> v
        elif ep1 == Endpoint.ARROW and ep2 == Endpoint.ARROW:
            nx_graph.add_edge(u, v, type='bidirected')
            nx_graph.add_edge(v, u, type='bidirected')

        # Неориентированное u --- v
        else:
            nx_graph.add_edge(u, v, type='undirected')
            nx_graph.add_edge(v, u, type='undirected')

    return nx_graph


def draw_graph_matplotlib(nx_g, title):
    """Простая визуализация через matplotlib (замена pyvis для локального теста)"""
    plt.figure(figsize=(15, 10))
    pos = kamada_kawai_layout(nx_g)

    node_colors = []
    for node in nx_g.nodes():
        if node == target_column:
            node_colors.append('red')
        elif node == treatment_column:
            node_colors.append('green')
        elif node == mediator_column:
            node_colors.append('yellow')
        else:
            node_colors.append('lightblue')

    nx.draw(nx_g, pos, with_labels=True, node_color=node_colors,
            node_size=800, font_size=8, edge_color='gray', arrows=True)

    plt.title(title)
    plt.savefig(f"{CACHE_DIR}/graph_viz.png")
    print(f"Граф сохранен в {CACHE_DIR}/graph_viz.png")
    # plt.show() # Раскомментируйте, если запускаете в Jupyter


# ==========================================
# 4. ОСНОВНОЙ ПАЙПЛАЙН
# ==========================================

def main():
    # 1. Загрузка данных
    print(f"Загрузка данных из {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print("Файл не найден! Сначала запустите data_generator.py")
        return

    df = pd.read_parquet(INPUT_FILE)
    print(f"Размер датасета: {df.shape}")

    # 2. Отбор колонок для графа
    all_node_names = [x for x in df.columns if x not in [date_column, *technical_columns]]

    # Удаление лишних колонок (как в вашем примере)
    cols_to_remove = ["pl_12m_cltv", "cltv", "pnl_oi_total_amt"]
    for col in cols_to_remove:
        if col in all_node_names:
            all_node_names.remove(col)

    print(f"Количество узлов в графе: {len(all_node_names)}")

    graph_dataset = df[all_node_names]
    graph_dataset_np = graph_dataset.to_numpy()

    # 3. Инициализация CausalGraph
    print("Инициализация CausalGraph...")
    cg = CausalGraph(no_of_var=len(all_node_names), node_names=all_node_names)

    # Установка теста независимости
    # Важно: FisherZ предполагает гауссовское распределение.
    # Если данные сильно отличаются, тест может работать некорректно.
    cg.set_ind_test(CIT(graph_dataset_np, method=INDEP_TEST))

    # 4. Запуск PC Stable (Parallel)
    print(f"Запуск Parallel PC Stable (alpha={ALPHA}, jobs={N_JOBS})...")
    start_time = time.perf_counter()

    result_wrapper = pc_stable(
        cg,
        alpha=ALPHA,
        n_jobs=N_JOBS,
        verbose=True,
        show_progress=True
    )

    end_time = time.perf_counter()
    print(f"Время выполнения: {end_time - start_time:.2f} сек")

    g = result_wrapper.G

    # 5. Анализ ребер (проверка на ориентированность)
    print("\n--- Статистика ребер ---")
    tail_tail = 0  # Неориентированные (---)
    arrow_tail = 0  # Ориентированные (-->)
    arrow_arrow = 0  # Двунаправленные (<->)

    for edge in g.get_graph_edges():
        end1 = edge.get_endpoint1()
        end2 = edge.get_endpoint2()

        if end1 == Endpoint.TAIL and end2 == Endpoint.TAIL:
            tail_tail += 1
        elif (end1 == Endpoint.TAIL and end2 == Endpoint.ARROW) or \
                (end1 == Endpoint.ARROW and end2 == Endpoint.TAIL):
            arrow_tail += 1
        elif end1 == Endpoint.ARROW and end2 == Endpoint.ARROW:
            arrow_arrow += 1

    print(f"Всего ребер: {tail_tail + arrow_tail + arrow_arrow}")
    print(f"Неориентированных (---): {tail_tail}")
    print(f"Направленных (-->):      {arrow_tail}")
    print(f"Двунаправленных (<->):   {arrow_arrow}")

    # 6. Сохранение и конвертация
    tag = f"pc_parallel_{ALPHA}_{INDEP_TEST}_{len(all_node_names)}_parallel"
    joblib.dump(g, f"./{CACHE_DIR}/{tag}.pkl")

    nx_g = convert_general_graph_to_nx(g)
    joblib.dump(nx_g, f"./{CACHE_DIR}/{tag}_nx_parallel.pkl")

    # 7. Визуализация
    draw_graph_matplotlib(nx_g, f"Causal Graph (PC Parallel) - {len(all_node_names)} nodes")


if __name__ == "__main__":
    main()