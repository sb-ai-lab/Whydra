import sys
import time
import numpy as np
import networkx as nx
from typing import List, Optional, Union

from whydra.algorithms.graph_core import (
    GeneralGraph,
    Node as GraphNode,
    Edge,
    Endpoint,
    CausalGraph,
    CIT,
)
from whydra.algorithms.standalone import pc_algo, fci_algo_v2, rai_algo


class CausalRunner:
    def __init__(self, library_path: str = './SCL-sber/', alpha: float = 0.05, indep_test: str = 'fisherz',
                 n_jobs: int = -1):
        if library_path not in sys.path:
            sys.path.insert(0, library_path)

        # Значения по умолчанию
        self.default_alpha = alpha
        self.default_indep_test = indep_test
        self.n_jobs = n_jobs

    def _pag_matrix_to_graph(self, matrix, node_names):
        nodes = [GraphNode(name) for name in node_names]
        g = GeneralGraph(nodes)
        n = len(nodes)
        endpoint_map = {0: Endpoint.NULL, 1: Endpoint.CIRCLE, 2: Endpoint.ARROW, 3: Endpoint.TAIL}

        for i in range(n):
            for j in range(i + 1, n):
                val_j = int(matrix[i, j])
                val_i = int(matrix[j, i])
                if val_j != 0 or val_i != 0:
                    try:
                        edge = Edge(nodes[i], nodes[j], endpoint_map[val_i], endpoint_map[val_j])
                        g.add_edge(edge)
                    except KeyError:
                        pass
        return g

    def _convert_to_nx(self, graph: GeneralGraph) -> nx.DiGraph:
        nx_graph = nx.DiGraph()
        for node in graph.get_nodes():
            nx_graph.add_node(node.get_name())

        for edge in graph.get_graph_edges():
            u = edge.get_node1().get_name()
            v = edge.get_node2().get_name()
            ep1 = edge.get_endpoint1().value
            ep2 = edge.get_endpoint2().value

            if ep1 == -1 and ep2 == 1:
                nx_graph.add_edge(u, v, type='directed')
            elif ep1 == 1 and ep2 == -1:
                nx_graph.add_edge(v, u, type='directed')
            elif ep1 == 1 and ep2 == 1:
                nx_graph.add_edge(u, v, type='bidirected')
                nx_graph.add_edge(v, u, type='bidirected')
            elif ep1 == 2 and ep2 == 1:
                nx_graph.add_edge(u, v, type='directed', style='circle')
            elif ep1 == 1 and ep2 == 2:
                nx_graph.add_edge(v, u, type='directed', style='circle')
            elif ep1 == 2 and ep2 == 2:
                nx_graph.add_edge(u, v, type='undirected', style='circle')
            elif ep1 == -1 and ep2 == -1:
                nx_graph.add_edge(u, v, type='undirected')

        return nx_graph

    def run(self, df, method: str = 'PC', ignore_cols: Optional[List[str]] = None,
            alpha: Optional[float] = None, indep_test: Optional[str] = None):
        """
        Запускает алгоритм с возможностью переопределения гиперпараметров.
        """
        # 1. Определяем актуальные параметры (аргумент > дефолт)
        current_alpha = alpha if alpha is not None else self.default_alpha
        current_test = indep_test if indep_test is not None else self.default_indep_test

        # 2. Подготовка данных
        if ignore_cols:
            cols = [c for c in df.columns if c not in ignore_cols]
        else:
            cols = df.columns.tolist()

        data_np = df[cols].astype(float).to_numpy()
        node_names = cols

        # 3. Инициализация CausalGraph с нужным тестом
        cg = CausalGraph(no_of_var=len(node_names), node_names=node_names)
        cg.set_ind_test(CIT(data_np, method=current_test))

        print(f"Running {method} (alpha={current_alpha}, test={current_test}, jobs={self.n_jobs})...")
        start_time = time.perf_counter()

        if method == "PC":
            g = pc_algo.pc_stable(
                cg, alpha=current_alpha, n_jobs=self.n_jobs, verbose=True, show_progress=True
            )
        elif method == "FCI":
            pag_matrix = fci_algo_v2.fci_stable_v2(
                cg, data_np, alpha=current_alpha, indep_test=current_test, n_jobs=self.n_jobs, verbose=True
            )
            g = self._pag_matrix_to_graph(pag_matrix, node_names)
        elif method == "RAI":
            res = rai_algo.rai_stable(
                data_np, alpha=current_alpha, indep_test=current_test, node_names=node_names, n_jobs=self.n_jobs,
                verbose=True
            )
            g = res.G
        else:
            raise ValueError(f"Unknown method: {method}")

        duration = time.perf_counter() - start_time
        print(f"Done in {duration:.2f}s")

        # Анализ ребер
        print(f"\n--- Статистика ребер ({method}) ---")

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
        if method == "FCI":
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

        return {
            'general_graph': g,
            'nx_graph': self._convert_to_nx(g),
            'duration': duration,
            'params': {'alpha': current_alpha, 'test': current_test}  # Возвращаем использованные параметры
        }
