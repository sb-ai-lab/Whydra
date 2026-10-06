import os
import re
import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.GraphUtils import GraphUtils


def parse_ground_truth_graph(file_path):
    """
    Парсит файл с истинным графом и возвращает объект GeneralGraph.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # 1. Парсим узлы
    nodes_match = re.search(r"Graph Nodes:\n(.*?)\n\n", content, re.DOTALL)
    if not nodes_match:
        nodes_match = re.search(r"Graph Nodes:\n(.*?)\nGraph Edges:", content, re.DOTALL)

    if not nodes_match:
        raise ValueError(f"Could not parse nodes from {file_path}")

    nodes_str = nodes_match.group(1).replace('\n', '').strip()

    if ';' in nodes_str:
        splitter = ';'
    else:
        splitter = ','

    node_names = [name.strip() for name in nodes_str.split(splitter) if name.strip()]

    nodes = [GraphNode(name) for name in node_names]
    g = GeneralGraph(nodes)
    node_map = {node.get_name(): node for node in nodes}

    # 2. Парсим ребра
    edges_match = re.search(r"Graph Edges:\n(.*?)$", content, re.DOTALL)
    if edges_match:
        edge_lines = edges_match.group(1).strip().split('\n')
        for line in edge_lines:
            if not line.strip(): continue

            parts = line.split()
            if len(parts) < 4: continue

            node1_name = parts[1]
            arrow_str = parts[2]
            node2_name = parts[3]

            if node1_name not in node_map or node2_name not in node_map:
                continue

            node1 = node_map[node1_name]
            node2 = node_map[node2_name]

            end1_char = arrow_str[0]
            end2_char = arrow_str[-1]

            def char_to_endpoint(c):
                if c == '-': return Endpoint.TAIL
                if c == '>': return Endpoint.ARROW
                if c == '<': return Endpoint.ARROW
                if c == 'o': return Endpoint.CIRCLE
                return Endpoint.TAIL

            end1 = char_to_endpoint(end1_char)
            end2 = char_to_endpoint(end2_char)

            edge = Edge(node1, node2, end1, end2)
            g.add_edge(edge)

    return g


def load_ground_truth(path: str) -> GeneralGraph:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    # Если это .npy (CausalTime)
    if path.endswith('.npy'):
        adj_matrix = np.load(path, allow_pickle=False)
        # CausalTime graph shape: (N, N). 1 = edge, 0 = no edge.
        # Обычно это DAG.
        n_nodes = adj_matrix.shape[0]
        nodes = [GraphNode(f"X{i + 1}") for i in range(n_nodes)]
        g = GeneralGraph(nodes)

        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_matrix[i, j] != 0:  # i -> j
                    # В CausalTime матрица обычно [source, target]
                    edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW)
                    g.add_edge(edge)
        return g

    # Иначе используем старый парсер текстовых файлов
    return parse_ground_truth_graph(path)


def draw_graph(graph: GeneralGraph, output_dir: str, filename: str):
    """
    Сохраняет изображение графа в PNG.

    Args:
        graph: Объект GeneralGraph
        output_dir: Папка для сохранения
        filename: Имя файла (без расширения)
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Конвертируем в pydot объект
        pyd = GraphUtils.to_pydot(graph)

        # Полный путь
        full_path = os.path.join(output_dir, f"{filename}.png")

        # Сохраняем
        pyd.write_png(full_path)

    except Exception as e:
        print(f"Warning: Could not save graph visualization. Error: {e}")


def get_adj_matrix(g: GeneralGraph):
    """Конвертирует граф в матрицу смежности, где элементы - Endpoints."""
    nodes = g.get_nodes()
    n = len(nodes)
    node_map = {node.get_name(): i for i, node in enumerate(nodes)}

    mat = np.zeros((n, n), dtype=int)

    for edge in g.get_graph_edges():
        i = node_map[edge.get_node1().get_name()]
        j = node_map[edge.get_node2().get_name()]

        # mat[i, j] - это эндпоинт у узла j (куда приходит ребро от i)
        mat[i, j] = edge.get_endpoint2().value
        mat[j, i] = edge.get_endpoint1().value

    return mat


def adj_matrix_to_graph(adj_mat, node_names):
    """
    Converts an adjacency matrix back to a GeneralGraph.
    Assumes the matrix contains Endpoint values (or their averages).
    Rounds values to the nearest integer to map back to Endpoint types.
    """
    nodes = [GraphNode(name) for name in node_names]
    g = GeneralGraph(nodes)
    # node_map is implicitly index-based since we created nodes from node_names list
    # which corresponds to matrix indices

    n = len(nodes)

    for i in range(n):
        for j in range(i + 1, n):
            # i < j ensures we check each pair once.
            # We look at both (i,j) and (j,i) to determine the edge endpoints.

            val_j = adj_mat[i, j]  # endpoint at j
            val_i = adj_mat[j, i]  # endpoint at i

            # Map float back to integer Endpoint value
            ep_j_val = int(round(val_j))
            ep_i_val = int(round(val_i))

            # If both ends are NULL (0), there is no edge
            if ep_j_val == 0 and ep_i_val == 0:
                continue

            # Create edge
            node1 = nodes[i]
            node2 = nodes[j]

            # Map integer values to Endpoint objects
            try:
                ep1 = Endpoint(ep_i_val)  # endpoint at i
                ep2 = Endpoint(ep_j_val)  # endpoint at j
            except ValueError:
                # Should not happen with standard Endpoint values (-1, 0, 1, 2)
                # But if averaging produced something weird (e.g. 1.5 -> 2), it's handled.
                # If something out of range, skip.
                continue

            edge = Edge(node1, node2, ep1, ep2)
            g.add_edge(edge)

    return g
