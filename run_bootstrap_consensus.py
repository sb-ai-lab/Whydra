import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from whydra.evaluation.graph_utils import load_ground_truth
from whydra.evaluation.metrics import calculate_metrics, get_adj_matrix

# Импорты causal-learn
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint


# ===============================================================
#                  У Т И Л И Т Ы
# ===============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def validate_adj_matrix(adj_mat):
    """
    Проверка адекватности матрицы смежности.
    Ожидаем значения: -1 (Tail), 1 (Arrow), 2 (Circle), 0 (Null).
    """
    if adj_mat.ndim != 2 or adj_mat.shape[0] != adj_mat.shape[1]:
        raise ValueError("Adjacency matrix must be square")

    uniq = np.unique(adj_mat)
    allowed = {-1, 0, 1, 2}
    if not set(uniq).issubset(allowed):
        print("WARNING: unexpected values in adjacency matrix:", uniq)


def matrix_to_graph(matrix, node_names):
    """
    Превращает матрицу endpoint-кодов в граф.
    """
    nodes = [GraphNode(name) for name in node_names]
    g = GeneralGraph(nodes)
    n = len(nodes)

    mat = np.array(matrix, dtype=int)

    for i in range(n):
        for j in range(i + 1, n):

            if mat[i, j] == 0 and mat[j, i] == 0:
                continue

            end_j = Endpoint(mat[i, j])  # endpoint у j
            end_i = Endpoint(mat[j, i])  # endpoint у i

            edge = Edge(nodes[i], nodes[j], end_i, end_j)
            g.add_edge(edge)

    return g


# ===============================================================
#                  ОСНОВНОЙ АЛГОРИТМ
# ===============================================================

@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_consensus(cfg: DictConfig):

    from hydra.utils import get_original_cwd

    # ---------------------------------------------------------------
    # 1. Настройки
    # ---------------------------------------------------------------
    bname = cfg.benchmark.names[0]
    print(f"Running Bootstrap Consensus (Sequential Loop) for: {bname}")

    loader = hydra.utils.instantiate(cfg.benchmark.loader)

    # алгоритм, который сам использует параллелизм внутри
    algo = hydra.utils.instantiate(cfg.algorithm)

    # ---------------------------------------------------------------
    # 2. Загрузка данных
    # ---------------------------------------------------------------
    data = loader.load_data(bname)
    gt_path = loader.load_ground_truth_path(bname)
    true_G = load_ground_truth(gt_path)

    # Приводим данные к numpy для стабильности
    if hasattr(data, "values"):
        data_arr = data.values
        col_names = data.columns.tolist()
    else:
        data_arr = np.asarray(data)
        col_names = [f"X{i}" for i in range(data_arr.shape[1])]

    n_samples, n_nodes = data_arr.shape

    # ---------------------------------------------------------------
    # 3. Параметры бутстрэпа
    # ---------------------------------------------------------------
    B = cfg.n_bootstraps
    base_seed = cfg.get("seed", 42)

    print(f"\nBootstraps = {B}")
    print("Running sequentially because the algorithm itself is parallelized.\n")

    edge_counts = np.zeros((n_nodes, n_nodes), dtype=float)

    # ---------------------------------------------------------------
    # 4. Последовательный бутстрэп
    # ---------------------------------------------------------------
    for i in tqdm(range(B), desc="Bootstrap Iterations"):

        rng = np.random.RandomState(base_seed + i)
        indices = rng.choice(n_samples, n_samples, replace=True)

        sample_data = data_arr[indices]

        # Если алгоритм ожидает DataFrame — раскомментируйте:
        # sample_data = pd.DataFrame(sample_data, columns=col_names)

        est_G = algo.run(sample_data)

        adj_mat = get_adj_matrix(est_G)

        # Проверка корректности и shape
        validate_adj_matrix(adj_mat)
        if adj_mat.shape != (n_nodes, n_nodes):
            raise ValueError(f"Adj matrix shape {adj_mat.shape} != expected ({n_nodes},{n_nodes})")

        # только наличие ребра
        edge_counts += (adj_mat != 0).astype(float)

    # ---------------------------------------------------------------
    # 5. Агрегация
    # ---------------------------------------------------------------
    confidence_matrix = edge_counts / B

    # ---------------------------------------------------------------
    # 6. Порог
    # ---------------------------------------------------------------
    tau = cfg.get("tau", 0.5)
    robust_mask = (confidence_matrix >= tau)
    print(f"Applying confidence threshold τ = {tau}")

    # ---------------------------------------------------------------
    # 7. Финальный прогон на всех данных
    # ---------------------------------------------------------------
    print("Running final algorithm pass on full data to determine orientations...")

    full_data = data_arr  # или DataFrame, если алгоритм требует
    full_G = algo.run(full_data)

    full_mat = get_adj_matrix(full_G)
    validate_adj_matrix(full_mat)

    if full_mat.shape != (n_nodes, n_nodes):
        raise ValueError("Shape mismatch in final adjacency matrix")

    node_names = [n.get_name() for n in full_G.get_nodes()]

    # итоговая матрица
    final_mat = (full_mat * robust_mask).astype(int)

    # ---------------------------------------------------------------
    # 8. Сборка итогового графа
    # ---------------------------------------------------------------
    robust_G = matrix_to_graph(final_mat, node_names)

    # ---------------------------------------------------------------
    # 9. Метрики
    # ---------------------------------------------------------------
    metrics = calculate_metrics(true_G, robust_G)

    print("\n=== Results of Bootstrap Consensus (Robust Graph) ===")
    print(f"Benchmark:     {bname}")
    print(f"Algorithm:     {cfg.algorithm._target_}")
    print(f"Bootstraps:    {B}")
    print(f"Threshold (τ): {tau}")
    print("-" * 40)
    print(f"Adj Precision: {metrics['adj_precision']:.4f}")
    print(f"Adj Recall:    {metrics['adj_recall']:.4f}")
    print(f"Adj F1:        {metrics['adj_f1']:.4f}")
    print(f"SHD:           {metrics['SHD']}")
    print("-" * 40)

    # ---------------------------------------------------------------
    # 10. Гистограмма
    # ---------------------------------------------------------------
    out_dir = os.path.join(get_original_cwd(), "results")
    ensure_dir(out_dir)

    flat_conf = confidence_matrix[confidence_matrix > 0].flatten()

    if flat_conf.size == 0:
        print("No edges found across bootstraps — histogram skipped.")
        return

    plt.figure(figsize=(8, 6))
    plt.hist(flat_conf, bins=10, edgecolor='black', alpha=0.7)
    plt.title(f"Distribution of Edge Confidence ({bname})")
    plt.xlabel("Confidence Probability")
    plt.ylabel("Count of Edges")
    plt.grid(axis='y', alpha=0.5)

    plot_path = os.path.join(out_dir, f"bootstrap_confidence_{bname}.png")
    plt.savefig(plot_path)
    print(f"Confidence histogram saved to {plot_path}")


if __name__ == "__main__":
    run_consensus()
