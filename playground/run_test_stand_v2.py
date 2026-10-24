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
# 1. НАСТРОЙКА ОКРУЖЕНИЯ
# ==========================================
sys.path.insert(0, './SCL-sber/')

try:
    from whydra.algorithms.graph_core import (
        GeneralGraph,
        Node as GraphNode,
        Edge,
        Endpoint,
    )
    # from causal_graphs.algorithms.standalone.parallel_pc_stable import pc_stable
    # from causal_graphs.algorithms.standalone import pc_algo, fci_algo
    from whydra.algorithms.standalone import pc_algo, fci_algo, rai_algo
    from whydra.algorithms.graph_core import CausalGraph, CIT

    print("[OK] Библиотека causal_graphs успешно импортирована.")
except ImportError as e:
    print(f"[ERROR] Ошибка импорта: {e}")
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

# Технические колонки, которые нужно исключить из графа
technical_columns = {'epk_id', 'actual_epk_id', 'row_upd_dttm', 'ctl_loading', 'score_uplift'}

ALPHA = 0.05  # Чуть повысим alpha, так как данные зашумлены бинаризацией
INDEP_TEST = "fisherz"
N_JOBS = -1

# ВЫБОР АЛГОРИТМА: "PC", "FCI" или "RAI"
ALGORITHM = "RAI"

# ==========================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

def convert_general_graph_to_nx(graph):
    nx_graph = nx.DiGraph()
    for node in graph.get_nodes():
        nx_graph.add_node(node.get_name())

    for edge in graph.get_graph_edges():
        u = edge.get_node1().get_name()
        v = edge.get_node2().get_name()
        ep1 = edge.get_endpoint1().value
        ep2 = edge.get_endpoint2().value

        # TAIL=-1, ARROW=1, CIRCLE=2
        if ep1 == -1 and ep2 == 1:
            nx_graph.add_edge(u, v, type='directed')
        elif ep1 == 1 and ep2 == -1:
            nx_graph.add_edge(v, u, type='directed')
        elif ep1 == 1 and ep2 == 1:
            nx_graph.add_edge(u, v, type='bidirected')
            nx_graph.add_edge(v, u, type='bidirected')
        elif ep1 == 2 and ep2 == 1: # o->
            nx_graph.add_edge(u, v, type='directed', style='circle')
        elif ep1 == 1 and ep2 == 2: # <-o
            nx_graph.add_edge(v, u, type='directed', style='circle')
        elif ep1 == 2 and ep2 == 2: # o-o
            nx_graph.add_edge(u, v, type='undirected', style='circle')
        elif ep1 == -1 and ep2 == -1:
            nx_graph.add_edge(u, v, type='undirected')

    return nx_graph


def draw_graph_matplotlib(nx_g, title):
    plt.figure(figsize=(20, 15))
    try:
        pos = kamada_kawai_layout(nx_g)
    except:
        pos = nx.spring_layout(nx_g)

    node_colors = []
    node_sizes = []
    for node in nx_g.nodes():
        if node == target_column:
            node_colors.append('red')
            node_sizes.append(2000)
        elif node == treatment_column:
            node_colors.append('green')
            node_sizes.append(2000)
        elif node == mediator_column:
            node_colors.append('yellow')
            node_sizes.append(2000)
        else:
            node_colors.append('lightblue')
            node_sizes.append(800)

    nx.draw(nx_g, pos, with_labels=True, node_color=node_colors,
            node_size=node_sizes, font_size=10, edge_color='gray', arrows=True)

    plt.title(title)
    output_path = f"{CACHE_DIR}/graph_viz_{ALGORITHM}.png"
    plt.savefig(output_path)
    print(f"Граф сохранен в {output_path}")


def pag_matrix_to_graph(matrix, node_names):
    """
    Преобразует матрицу PAG (результат FCI) в объект GeneralGraph.
    В реализации fci_algo.py используются коды:
    1: Circle (o), 2: Arrow (>), 3: Tail (-)
    """
    nodes = [GraphNode(name) for name in node_names]
    g = GeneralGraph(nodes)
    n = len(nodes)

    # Маппинг кодов из fci_algo в типы Endpoint causal-learn
    # В causal-learn: TAIL=-1, ARROW=1, CIRCLE=2
    endpoint_map = {
        0: Endpoint.NULL,
        1: Endpoint.CIRCLE, # o
        2: Endpoint.ARROW,  # >
        3: Endpoint.TAIL    # -
    }

    for i in range(n):
        for j in range(i + 1, n):
            # matrix[i, j] - это конец ребра у узла j (стрелка/хвост/круг в j)
            val_j = int(matrix[i, j])
            val_i = int(matrix[j, i])

            if val_j != 0 or val_i != 0:
                # Создаем ребро
                # endpoint_map[val_i] - это то, что стоит у узла i
                edge = Edge(nodes[i], nodes[j], endpoint_map[val_i], endpoint_map[val_j])
                g.add_edge(edge)
    return g

# ==========================================
# 4. ОСНОВНОЙ ПАЙПЛАЙН
# ==========================================

def main():
    # os.makedirs("results", exist_ok=True)
    # 1. Загрузка
    print(f"Загрузка данных из {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print("Файл не найден! Запустите data_generator.py")
        return

    df = pd.read_parquet(INPUT_FILE)
    print(f"Размер датасета: {df.shape}")
    print("Типы данных до обработки:")
    print(df[['comm_fl', 'sale_fl']].dtypes)

    # 2. Отбор колонок для графа
    all_node_names = [x for x in df.columns if x not in [date_column, *technical_columns]]
    print(f"Количество узлов в графе: {len(all_node_names)}")

    graph_dataset = df[all_node_names]

    # ВАЖНО: Приведение к float для корректной работы FisherZ
    # FisherZ работает с корреляциями Пирсона. Если подать int, он может отработать,
    # но лучше явно привести к float, чтобы избежать warning'ов и ошибок типов внутри C++ частей (если они есть).
    graph_dataset_np = graph_dataset.astype(float).to_numpy()

    # 3. Инициализация CausalGraph
    print("Инициализация CausalGraph...")
    cg = CausalGraph(no_of_var=len(all_node_names), node_names=all_node_names)

    # Тест FisherZ
    cg.set_ind_test(CIT(graph_dataset_np, method=INDEP_TEST))

    # 4. Запуск Алгоритма
    print(f"Запуск алгоритма {ALGORITHM} (alpha={ALPHA}, jobs={N_JOBS})...")
    start_time = time.perf_counter()

    if ALGORITHM == "PC":
        # PC возвращает объект графа напрямую (или wrapper, который ведет себя как граф)
        g = pc_algo.pc_stable(
            cg,
            alpha=ALPHA,
            n_jobs=N_JOBS,
            verbose=True,
            show_progress=True
        )

    elif ALGORITHM == "FCI":
        # FCI возвращает матрицу (numpy array), требует передачи данных отдельно
        pag_matrix = fci_algo.fci_stable(
            cg,
            graph_dataset_np,  # <--- Важно: передаем сырые данные
            alpha=ALPHA,
            indep_test=INDEP_TEST,
            n_jobs=N_JOBS,
            verbose=True
        )
        # Конвертируем матрицу в граф, чтобы дальше код работал одинаково
        g = pag_matrix_to_graph(pag_matrix, all_node_names)

    elif ALGORITHM == "RAI":
        # RAI: Принимает данные, возвращает ResultWrapper
        # ВАЖНО: rai_stable требует явной передачи node_names
        result_wrapper = rai_algo.rai_stable(
            graph_dataset_np,
            alpha=ALPHA,
            indep_test=INDEP_TEST,
            node_names=all_node_names,
            n_jobs=N_JOBS,
            verbose=True
        )
        g = result_wrapper.G  # Извлекаем граф из обертки

    else:
        raise ValueError(f"Неизвестный алгоритм: {ALGORITHM}")

    end_time = time.perf_counter()
    print(f"Время выполнения: {end_time - start_time:.2f} сек")


    # 5. Анализ ребер
    print(f"\n--- Статистика ребер ({ALGORITHM}) ---")

    # Счетчики типов ребер
    directed = 0  # -->
    undirected = 0  # ---
    bidirected = 0  # <->
    part_directed = 0  # o->
    part_undirected = 0  # o-o
    nondeter = 0  # o-o (иногда так обозначают) или другие типы

    for edge in g.get_graph_edges():
        # Получаем значения endpoint'ов (-1: Tail, 1: Arrow, 2: Circle)
        ep1 = edge.get_endpoint1().value
        ep2 = edge.get_endpoint2().value

        # Сортируем, чтобы не проверять (1, -1) и (-1, 1) отдельно
        pair = sorted([ep1, ep2])

        if pair == [-1, 1]:  # Tail, Arrow
            directed += 1
        elif pair == [-1, -1]:  # Tail, Tail
            undirected += 1
        elif pair == [1, 1]:  # Arrow, Arrow
            bidirected += 1
        elif pair == [1, 2]:  # Arrow, Circle
            part_directed += 1
        elif pair == [2, 2]:  # Circle, Circle
            part_undirected += 1
        else:
            nondeter += 1

    print(f"Всего ребер: {len(g.get_graph_edges())}")
    print(f"Направленных (-->):           {directed}")
    print(f"Неориентированных (---):      {undirected}")
    print(f"Двунаправленных (<->):        {bidirected}")
    if ALGORITHM == "FCI":
        print(f"Частично направленных (o->):  {part_directed}")
        print(f"Неопределенных (o-o):         {part_undirected}")

    # Проверка ключевых связей
    print("\n--- Проверка ключевых связей ---")
    try:
        # Ищем ребро comm_fl -> sale_fl
        node_comm = [n for n in g.get_nodes() if n.get_name() == treatment_column][0]
        node_sale = [n for n in g.get_nodes() if n.get_name() == mediator_column][0]
        edge = g.get_edge(node_comm, node_sale)
        if edge:
            print(f"Ребро {treatment_column} - {mediator_column}: {edge}")
        else:
            print(f"Ребро {treatment_column} - {mediator_column} НЕ найдено")
    except Exception as e:
        print(f"Ошибка при проверке конкретного ребра: {e}")

    # 6. Сохранение и визуализация
    nx_g = convert_general_graph_to_nx(g)
    draw_graph_matplotlib(nx_g, f"Causal Graph ({ALGORITHM} Parallel)")


if __name__ == "__main__":
    main()
