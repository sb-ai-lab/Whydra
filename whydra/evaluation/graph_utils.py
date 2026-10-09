"""
Утилиты для работы с графами: парсинг, конвертация, визуализация.

Содержит функции конвертации между форматами:
  GeneralGraph ↔ endpoint-матрица ↔ NetworkX DiGraph
"""

import os
import re
import numpy as np
import networkx as nx
from whydra.algorithms.graph_core import GeneralGraph, Node as GraphNode, Edge, Endpoint

from whydra.background_knowledge import MatrixEncoding, endpoint_codes


def default_node_names(n: int) -> list:
    """Canonical variable names for a graph of ``n`` nodes: ``X1 … Xn``.

    One convention for the whole package. Ground truth used to be loaded as
    ``X1..Xn`` while the DAG helpers generated ``X0..Xn-1``, so matching an
    estimate against its ground truth by name missed on every node.
    """
    return [f"X{i + 1}" for i in range(n)]


def endpoint_lookup(encoding) -> dict:
    """Map raw endpoint codes of ``encoding`` onto local endpoint values."""
    null, tail, arrow, circle = endpoint_codes(encoding)
    return {
        null: Endpoint.NULL,
        tail: Endpoint.TAIL,
        arrow: Endpoint.ARROW,
        circle: Endpoint.CIRCLE,
    }


# ============================================================
#  ENDPOINT-МАТРИЦА ↔ GENERAL GRAPH
# ============================================================

def get_adj_matrix(g: GeneralGraph) -> np.ndarray:
    """
    Конвертирует GeneralGraph в endpoint-матрицу.
    mat[i, j] — endpoint у узла j на ребре от i к j.
    Кодировка: TAIL=-1, NULL=0, ARROW=1, CIRCLE=2.
    """
    nodes = g.get_nodes()
    n = len(nodes)
    node_map = {node.get_name(): i for i, node in enumerate(nodes)}

    mat = np.zeros((n, n), dtype=int)

    for edge in g.get_graph_edges():
        i = node_map[edge.get_node1().get_name()]
        j = node_map[edge.get_node2().get_name()]

        mat[i, j] = edge.get_endpoint2().value  # endpoint at j
        mat[j, i] = edge.get_endpoint1().value  # endpoint at i

    return mat


def adj_matrix_to_graph(adj_mat: np.ndarray, node_names: list) -> GeneralGraph:
    """
    Конвертирует endpoint-матрицу обратно в GeneralGraph.
    Округляет float-значения (например, после усреднения при бэггинге).
    """
    nodes = [GraphNode(name) for name in node_names]
    g = GeneralGraph(nodes)
    n = len(nodes)

    for i in range(n):
        for j in range(i + 1, n):
            val_j = adj_mat[i, j]  # endpoint at j
            val_i = adj_mat[j, i]  # endpoint at i

            ep_j_val = int(round(val_j))
            ep_i_val = int(round(val_i))

            if ep_j_val == 0 and ep_i_val == 0:
                continue

            try:
                ep1 = Endpoint(ep_i_val)
                ep2 = Endpoint(ep_j_val)
            except ValueError:
                continue

            edge = Edge(nodes[i], nodes[j], ep1, ep2)
            g.add_edge(edge)

    return g


# ============================================================
#  GENERAL GRAPH → NETWORKX
# ============================================================

def convert_general_graph_to_nx(graph: GeneralGraph) -> nx.DiGraph:
    """
    Конвертирует GeneralGraph в NetworkX DiGraph с атрибутами type/style.

    Типы рёбер:
      u --> v:  add_edge(u, v, type='directed')
      u <-> v:  add_edge(u, v, type='bidirected') + add_edge(v, u, type='bidirected')
      u --- v:  add_edge(u, v, type='undirected') + add_edge(v, u, type='undirected')
      u o-> v:  add_edge(u, v, type='directed', style='circle')
      u o-o v:  add_edge(u, v, type='undirected', style='circle') + обратно
    """
    nx_graph = nx.DiGraph()
    for node in graph.get_nodes():
        nx_graph.add_node(node.get_name())

    for edge in graph.get_graph_edges():
        u = edge.get_node1().get_name()
        v = edge.get_node2().get_name()
        ep1 = edge.get_endpoint1().value
        ep2 = edge.get_endpoint2().value

        if ep1 == -1 and ep2 == 1:      # u --> v
            nx_graph.add_edge(u, v, type='directed')
        elif ep1 == 1 and ep2 == -1:    # u <-- v
            nx_graph.add_edge(v, u, type='directed')
        elif ep1 == 1 and ep2 == 1:     # u <-> v
            nx_graph.add_edge(u, v, type='bidirected')
            nx_graph.add_edge(v, u, type='bidirected')
        elif ep1 == 2 and ep2 == 1:     # o-> v
            nx_graph.add_edge(u, v, type='directed', style='circle')
        elif ep1 == 1 and ep2 == 2:     # u <-o
            nx_graph.add_edge(v, u, type='directed', style='circle')
        elif ep1 == 2 and ep2 == 2:     # o-o
            nx_graph.add_edge(u, v, type='undirected', style='circle')
            nx_graph.add_edge(v, u, type='undirected', style='circle')
        elif ep1 == -1 and ep2 == -1:   # u --- v
            nx_graph.add_edge(u, v, type='undirected')
            nx_graph.add_edge(v, u, type='undirected')

    return nx_graph


def pag_matrix_to_general_graph(
    matrix: np.ndarray,
    node_names: list,
    encoding: MatrixEncoding = MatrixEncoding.STANDARD,
) -> GeneralGraph:
    """
    Конвертирует PAG-матрицу в GeneralGraph.

    Все алгоритмы библиотеки работают в кодировке causal-learn
    (``0/-1=tail/1=arrow/2=circle``) — она и стоит по умолчанию. Параметр
    оставлен для матриц из внешних источников в кодировке pcalg
    (``0/1=circle/2=arrow/3=tail``), например из R.
    """
    endpoint_map = endpoint_lookup(encoding)

    nodes = [GraphNode(name) for name in node_names]
    g = GeneralGraph(nodes)
    n = len(nodes)

    for i in range(n):
        for j in range(i + 1, n):
            val_j = int(matrix[i, j])
            val_i = int(matrix[j, i])
            if val_j != 0 or val_i != 0:
                try:
                    end_i, end_j = endpoint_map[val_i], endpoint_map[val_j]
                except KeyError as exc:
                    raise ValueError(
                        f"Endpoint code {exc.args[0]!r} at ({i}, {j}) is not valid for "
                        f"encoding {MatrixEncoding(encoding).value!r}; "
                        "the matrix is probably in the other encoding"
                    ) from None
                g.add_edge(Edge(nodes[i], nodes[j], end_i, end_j))

    return g


def result_to_nx(
    result,
    node_names: list,
    encoding: MatrixEncoding = MatrixEncoding.STANDARD,
) -> nx.DiGraph:
    """
    Универсальная конвертация результата алгоритма в NetworkX DiGraph.

    Поддерживает:
      - np.ndarray (PAG из FCI) → pag_matrix_to_general_graph → convert_general_graph_to_nx
      - GeneralGraph (CPDAG из PC) → convert_general_graph_to_nx
      - ResultWrapper (.G → GeneralGraph, из RAI) → convert_general_graph_to_nx
    """
    if isinstance(result, np.ndarray) and result.ndim == 2:
        g = pag_matrix_to_general_graph(result, node_names, encoding)
        return convert_general_graph_to_nx(g)

    graph_obj = result
    if hasattr(result, "G"):
        graph_obj = result.G

    if hasattr(graph_obj, "get_nodes") and hasattr(graph_obj, "get_graph_edges"):
        return convert_general_graph_to_nx(graph_obj)

    raise TypeError(f"Не удалось конвертировать результат типа {type(result)} в NetworkX")


# ============================================================
#  ПАРСИНГ GROUND TRUTH
# ============================================================

def parse_ground_truth_graph(file_path: str) -> GeneralGraph:
    """Парсит файл с истинным графом и возвращает объект GeneralGraph."""
    with open(file_path, 'r') as f:
        content = f.read()

    nodes_match = re.search(r"Graph Nodes:\n(.*?)\n\n", content, re.DOTALL)
    if not nodes_match:
        nodes_match = re.search(r"Graph Nodes:\n(.*?)\nGraph Edges:", content, re.DOTALL)

    if not nodes_match:
        raise ValueError(f"Could not parse nodes from {file_path}")

    nodes_str = nodes_match.group(1).replace('\n', '').strip()
    splitter = ';' if ';' in nodes_str else ','
    node_names = [name.strip() for name in nodes_str.split(splitter) if name.strip()]

    nodes = [GraphNode(name) for name in node_names]
    g = GeneralGraph(nodes)
    node_map = {node.get_name(): node for node in nodes}

    edges_match = re.search(r"Graph Edges:\n(.*?)$", content, re.DOTALL)
    if edges_match:
        edge_lines = edges_match.group(1).strip().split('\n')
        for line in edge_lines:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue

            node1_name = parts[1]
            arrow_str = parts[2]
            node2_name = parts[3]

            if node1_name not in node_map or node2_name not in node_map:
                continue

            def char_to_endpoint(c):
                if c == '-': return Endpoint.TAIL
                if c == '>' or c == '<': return Endpoint.ARROW
                if c == 'o': return Endpoint.CIRCLE
                return Endpoint.TAIL

            end1 = char_to_endpoint(arrow_str[0])
            end2 = char_to_endpoint(arrow_str[-1])

            edge = Edge(node_map[node1_name], node_map[node2_name], end1, end2)
            g.add_edge(edge)

    return g


def load_ground_truth(path: str) -> GeneralGraph:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    if path.endswith('.npy'):
        adj_matrix = np.load(path)
        n_nodes = adj_matrix.shape[0]
        nodes = [GraphNode(name) for name in default_node_names(n_nodes)]
        g = GeneralGraph(nodes)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_matrix[i, j] != 0:
                    edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW)
                    g.add_edge(edge)
        return g

    return parse_ground_truth_graph(path)


def draw_graph(graph: GeneralGraph, output_dir: str, filename: str):
    """Сохраняет изображение графа в PNG."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, f"{filename}.png")
        nx_graph = convert_general_graph_to_nx(graph)
        nx.nx_pydot.to_pydot(nx_graph).write_png(full_path)
    except Exception as e:
        print(f"Warning: Could not save graph visualization. Error: {e}")
