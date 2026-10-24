"""
Скрипт оценки метрик алгоритмов Causal Discovery из SCL.

Все конвертации и метрики импортируются из causal_graphs.evaluation.metrics
и causal_graphs.evaluation.graph_utils — единственных источников правды.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append("../../")

try:
    from causal_graphs.algorithms.graph_core import CausalGraph, CIT
    from causal_graphs.algorithms.standalone import pc_algo, fci_algo, rai_algo
    from causal_graphs.algorithms.standalone import fci_algo_v2

    from causal_graphs.evaluation.metrics import (
        result_to_endpoint_matrix,
        is_pag_result,
        compute_metrics_from_dag_and_result,
    )
    from causal_graphs.evaluation.graph_utils import result_to_nx

    print("[OK] causal_graphs imported successfully.")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    sys.exit(1)


# ============================================================
#  ЗАГРУЗКА GROUND TRUTH
# ============================================================

def load_dag_from_file(dag_path: str) -> Dict[str, List[str]]:
    dag = {}
    with open(dag_path, "r") as f:
        for line in f:
            line = line.strip()
            if " -> " in line:
                parent, child = line.split(" -> ")
                parent, child = parent.strip(), child.strip()
                dag.setdefault(child, []).append(parent)
    return dag


def dag_to_adjacency_matrix(dag: Dict[str, List[str]], node_names: List[str]) -> np.ndarray:
    n = len(node_names)
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    adj = np.zeros((n, n), dtype=int)
    for child, parents in dag.items():
        if child not in name_to_idx:
            continue
        j = name_to_idx[child]
        for parent in parents:
            if parent in name_to_idx:
                adj[name_to_idx[parent], j] = 1
    return adj


def nx_to_adjacency_matrix(nx_g: nx.DiGraph, node_names: List[str]) -> np.ndarray:
    """Бинарная матрица для skeleton aggregation в бэггинге."""
    n = len(node_names)
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    adj = np.zeros((n, n), dtype=int)
    for u, v in nx_g.edges():
        if u in name_to_idx and v in name_to_idx:
            adj[name_to_idx[u], name_to_idx[v]] = 1
    return adj


# ============================================================
#  ЗАПУСК АЛГОРИТМОВ
# ============================================================

def _run_single_algo(data_np, node_names, algo_name, alpha, indep_test, n_jobs):
    if algo_name == "PC":
        cg = CausalGraph(no_of_var=len(node_names), node_names=node_names)
        cg.set_ind_test(CIT(data_np, method=indep_test))
        return pc_algo.pc_stable(cg, alpha=alpha, n_jobs=n_jobs, verbose=False, show_progress=False)
    elif algo_name == "FCI":
        cg = CausalGraph(no_of_var=len(node_names), node_names=node_names)
        cg.set_ind_test(CIT(data_np, method=indep_test))
        return fci_algo.fci_stable(cg, data_np, alpha=alpha, indep_test=indep_test, n_jobs=n_jobs, verbose=False)
    elif algo_name == "FCI_v2":
        cg = CausalGraph(no_of_var=len(node_names), node_names=node_names)
        cg.set_ind_test(CIT(data_np, method=indep_test))
        return fci_algo_v2.fci_stable_v2(cg, data_np, alpha=alpha, indep_test=indep_test, n_jobs=n_jobs)
    elif algo_name == "RAI":
        return rai_algo.rai_stable(data_np, alpha=alpha, indep_test=indep_test,
                                   node_names=node_names, n_jobs=n_jobs, verbose=False)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


# ============================================================
#  BOOTSTRAP CONSENSUS
# ============================================================

def run_with_bagging(data_np, node_names, algo_name, alpha, indep_test,
                     n_jobs, n_bootstraps=10, tau=0.5, seed=42, subsample_ratio=0.8):
    n_vars = len(node_names)
    rng = np.random.RandomState(seed)
    n_samples = data_np.shape[0]
    sample_size = int(n_samples * subsample_ratio)

    edge_counts = np.zeros((n_vars, n_vars), dtype=float)

    print(f"    Bagging (Subsampling {subsample_ratio * 100}%): {n_bootstraps} iterations...")
    for b in range(n_bootstraps):
        indices = rng.choice(n_samples, sample_size, replace=False)
        sample = data_np[indices]

        try:
            result = _run_single_algo(sample, node_names, algo_name, alpha, indep_test, n_jobs)
            nx_g = result_to_nx(result, node_names)
            adj = nx_to_adjacency_matrix(nx_g, node_names)

            skeleton = ((adj != 0) | (adj.T != 0)).astype(float)
            edge_counts += skeleton
        except Exception as e:
            print(f"      Bootstrap {b + 1} failed: {e}")

        if (b + 1) % max(1, n_bootstraps // 5) == 0:
            print(f"      Completed {b + 1}/{n_bootstraps}")

    confidence = edge_counts / n_bootstraps
    robust_mask = (confidence >= tau)

    print("    Running final pass on full data...")
    full_result = _run_single_algo(data_np, node_names, algo_name, alpha, indep_test, n_jobs)

    # Получаем endpoint-матрицу и маскируем нестабильные рёбра
    full_ep = result_to_endpoint_matrix(full_result, node_names)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if not robust_mask[i, j] and not robust_mask[j, i]:
                full_ep[i, j] = 0
                full_ep[j, i] = 0

    return full_ep, confidence


# ============================================================
#  ВИЗУАЛИЗАЦИЯ
# ============================================================

def _plot_edge_confidence_histogram(confidence_matrix, algo_name, output_dir):
    triu_conf = np.triu(confidence_matrix, k=1)
    flat_conf = triu_conf[triu_conf > 0].flatten()
    if flat_conf.size == 0:
        print(f"    No edges found across bootstraps for {algo_name} — histogram skipped.")
        return

    plt.figure(figsize=(8, 6))
    plt.hist(flat_conf, bins=np.linspace(0, 1, 11), edgecolor='black', alpha=0.7, color='#4C72B0')
    plt.title(f"Distribution of Edge Confidence ({algo_name})", fontsize=14)
    plt.xlabel("Confidence Probability", fontsize=12)
    plt.ylabel("Count of Edges", fontsize=12)
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(os.path.join(output_dir, f"bootstrap_confidence_{algo_name}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_metrics_comparison(results_df, output_dir):
    metrics_to_plot = ["SHD", "adj_f1", "arrow_f1"]
    available = [m for m in metrics_to_plot if m in results_df.columns]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        algos = results_df["algorithm"].unique()
        x = np.arange(len(algos))
        width = 0.35

        no_bag = results_df[results_df["bagging"] == False]
        with_bag = results_df[results_df["bagging"] == True]

        vals_no = [no_bag[no_bag["algorithm"] == a][metric].values[0]
                   if len(no_bag[no_bag["algorithm"] == a]) > 0 else 0 for a in algos]
        vals_bag = [with_bag[with_bag["algorithm"] == a][metric].values[0]
                    if len(with_bag[with_bag["algorithm"] == a]) > 0 else 0 for a in algos]

        ax.bar(x - width / 2, vals_no, width, label="Without Bagging", color="#4C72B0")
        ax.bar(x + width / 2, vals_bag, width, label="With Bagging", color="#DD8452")
        ax.set_ylabel(metric); ax.set_title(metric); ax.set_xticks(x)
        ax.set_xticklabels(algos); ax.legend(); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
#  ОСНОВНОЙ БЕНЧМАРК МЕТРИК
# ============================================================

def run_metrics_benchmark(data_path, dag_path, algorithms, n_jobs=8, alpha=0.01,
                          indep_test="fisherz", n_bootstraps=10, tau=0.5, seed=42,
                          output_dir="./metrics_results", exclude_cols=None):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data: {data_path}")
    df = pd.read_parquet(data_path)
    if exclude_cols:
        df = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        df = df.drop(columns=non_numeric)

    node_names = df.columns.tolist()
    data_np = df.astype(float).to_numpy()
    print(f"Dataset: {data_np.shape[0]} rows x {len(node_names)} columns")

    print(f"Loading DAG: {dag_path}")
    dag = load_dag_from_file(dag_path)
    true_adj = dag_to_adjacency_matrix(dag, node_names)
    print(f"Ground truth: {np.sum(true_adj != 0)} directed edges")

    all_results = []

    for algo_name in algorithms:
        print(f"\n{'=' * 60}\nAlgorithm: {algo_name}\n{'=' * 60}")

        # --- Без bagging ---
        print(f"\n  [Without bagging]")
        start = time.perf_counter()
        try:
            result = _run_single_algo(data_np, node_names, algo_name, alpha, indep_test, n_jobs)
            elapsed = time.perf_counter() - start

            est_ep = result_to_endpoint_matrix(result, node_names)
            is_pag = is_pag_result(result)
            metrics = compute_metrics_from_dag_and_result(true_adj, est_ep, is_pag=is_pag)
            metrics.update({"algorithm": algo_name, "bagging": False, "time_seconds": round(elapsed, 2), "n_bootstraps": 0})
            all_results.append(metrics)
            print(f"    Time: {elapsed:.2f}s, SHD={metrics['SHD']}, adj_f1={metrics['adj_f1']}, arrow_f1={metrics['arrow_f1']}")
        except Exception as e:
            print(f"    FAILED: {e}")
            all_results.append({"algorithm": algo_name, "bagging": False, "time_seconds": -1,
                                "SHD": -1, "adj_f1": -1, "arrow_f1": -1, "n_bootstraps": 0})

        # --- С bagging ---
        print(f"\n  [With bagging, B={n_bootstraps}, tau={tau}]")
        start = time.perf_counter()
        try:
            bagged_ep, confidence = run_with_bagging(
                data_np, node_names, algo_name, alpha, indep_test, n_jobs, n_bootstraps, tau, seed)
            elapsed = time.perf_counter() - start

            is_pag = algo_name in ("FCI", "FCI_v2")
            metrics = compute_metrics_from_dag_and_result(true_adj, bagged_ep, is_pag=is_pag)
            metrics.update({"algorithm": algo_name, "bagging": True, "time_seconds": round(elapsed, 2), "n_bootstraps": n_bootstraps})
            all_results.append(metrics)
            print(f"    Time: {elapsed:.2f}s, SHD={metrics['SHD']}, adj_f1={metrics['adj_f1']}, arrow_f1={metrics['arrow_f1']}")
            _plot_edge_confidence_histogram(confidence, algo_name, output_dir)
        except Exception as e:
            print(f"    FAILED: {e}")
            all_results.append({"algorithm": algo_name, "bagging": True, "time_seconds": -1,
                                "SHD": -1, "adj_f1": -1, "arrow_f1": -1, "n_bootstraps": n_bootstraps})

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, "metrics_benchmark.csv"), index=False)
    _plot_metrics_comparison(results_df, output_dir)

    print(f"\n{'=' * 80}\nSUMMARY TABLE\n{'=' * 80}")
    cols = ["algorithm", "bagging", "SHD", "adj_f1", "arrow_f1", "adj_precision", "adj_recall", "time_seconds"]
    print(results_df[[c for c in cols if c in results_df.columns]].to_string(index=False))

    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dag_path", type=str, required=True)
    parser.add_argument("--algorithms", nargs="+", default=["PC", "FCI", "RAI"],
                        choices=["PC", "FCI", "FCI_v2", "RAI"])
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--indep_test", type=str, default="fisherz")
    parser.add_argument("--n_bootstraps", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./metrics_results")
    parser.add_argument("--exclude_cols", nargs="*", default=["epk_id", "report_dt"])
    args = parser.parse_args()

    run_metrics_benchmark(args.data_path, args.dag_path, args.algorithms, args.n_jobs,
                          args.alpha, args.indep_test, args.n_bootstraps, args.tau,
                          args.seed, args.output_dir, args.exclude_cols)


if __name__ == "__main__":
    main()