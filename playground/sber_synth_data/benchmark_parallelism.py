"""
Бенчмарк распараллеливания алгоритмов Causal Discovery из библиотеки SCL.

Все конвертации и метрики импортируются из causal_graphs.evaluation.metrics
и causal_graphs.evaluation.graph_utils — единственных источников правды.
"""

import argparse
import os
import sys
import time
from itertools import combinations
from typing import Any, Dict, List, Optional

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
    print("Убедитесь, что SCL добавлен в sys.path или установлен.")
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


# ============================================================
#  ПОПАРНОЕ СРАВНЕНИЕ ГРАФОВ (NX-based, для проверки детерминизма)
# ============================================================

def _get_edge_type(G: nx.DiGraph, u, v) -> str:
    has_uv = G.has_edge(u, v)
    has_vu = G.has_edge(v, u)

    if not has_uv and not has_vu:
        return "null"

    attrs_uv = G.get_edge_data(u, v) if has_uv else {}
    attrs_vu = G.get_edge_data(v, u) if has_vu else {}
    is_circle_uv = attrs_uv.get('style') == 'circle'
    is_circle_vu = attrs_vu.get('style') == 'circle'

    if has_uv and has_vu:
        t_uv = attrs_uv.get('type')
        if t_uv == 'undirected':
            return "circle_undirected" if is_circle_uv else "undirected"
        return "bidirected"
    elif has_uv:
        return "circle_directed" if is_circle_uv else "directed_uv"
    else:
        return "circle_directed_rev" if is_circle_vu else "directed_vu"


def _calculate_nx_metrics(g1: nx.DiGraph, g2: nx.DiGraph) -> Dict[str, float]:
    all_nodes = sorted(set(g1.nodes()) | set(g2.nodes()))
    tp_skel = fp_skel = fn_skel = tp_arrow = shd = 0

    for u, v in combinations(all_nodes, 2):
        t1 = _get_edge_type(g1, u, v)
        t2 = _get_edge_type(g2, u, v)
        e1 = t1 != "null"
        e2 = t2 != "null"

        if e1 and e2:       tp_skel += 1
        elif not e1 and e2:  fp_skel += 1
        elif e1 and not e2:  fn_skel += 1

        if t1 != t2:        shd += 1
        elif t1 != "null":  tp_arrow += 1

    adj_prec = tp_skel / (tp_skel + fp_skel) if (tp_skel + fp_skel) > 0 else 0.0
    adj_rec = tp_skel / (tp_skel + fn_skel) if (tp_skel + fn_skel) > 0 else 0.0
    adj_f1 = 2 * adj_prec * adj_rec / (adj_prec + adj_rec) if (adj_prec + adj_rec) > 0 else 0.0
    tot_est = tp_skel + fp_skel
    tot_true = tp_skel + fn_skel
    arrow_prec = tp_arrow / tot_est if tot_est > 0 else 0.0
    arrow_rec = tp_arrow / tot_true if tot_true > 0 else 0.0
    arrow_f1 = 2 * arrow_prec * arrow_rec / (arrow_prec + arrow_rec) if (arrow_prec + arrow_rec) > 0 else 0.0

    return {"SHD": shd, "Adj_F1": round(adj_f1, 4), "Arrow_F1": round(arrow_f1, 4)}


# ============================================================
#  ЗАПУСК АЛГОРИТМОВ
# ============================================================

def run_pc(data_np, node_names, alpha, indep_test, n_jobs):
    cg = CausalGraph(no_of_var=len(node_names), node_names=node_names)
    cg.set_ind_test(CIT(data_np, method=indep_test))
    return pc_algo.pc_stable(cg, alpha=alpha, n_jobs=n_jobs, verbose=False, show_progress=False)

def run_fci(data_np, node_names, alpha, indep_test, n_jobs):
    cg = CausalGraph(no_of_var=len(node_names), node_names=node_names)
    cg.set_ind_test(CIT(data_np, method=indep_test))
    return fci_algo.fci_stable(cg, data_np, alpha=alpha, indep_test=indep_test, n_jobs=n_jobs, verbose=False)

def run_fci_v2(data_np, node_names, alpha, indep_test, n_jobs):
    cg = CausalGraph(no_of_var=len(node_names), node_names=node_names)
    cg.set_ind_test(CIT(data_np, method=indep_test))
    return fci_algo_v2.fci_stable_v2(cg, data_np, alpha=alpha, indep_test=indep_test, n_jobs=n_jobs)

def run_rai(data_np, node_names, alpha, indep_test, n_jobs):
    return rai_algo.rai_stable(data_np, alpha=alpha, indep_test=indep_test,
                               node_names=node_names, n_jobs=n_jobs, verbose=False)

ALGO_RUNNERS = {"PC": run_pc, "FCI": run_fci, "FCI_v2": run_fci_v2, "RAI": run_rai}


# ============================================================
#  ОСНОВНОЙ БЕНЧМАРК
# ============================================================

def run_benchmark(data_path, algorithms, n_jobs_list, alpha=0.05, indep_test="fisherz",
                  output_dir="./benchmark_results", exclude_cols=None, dag_path=None):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    if exclude_cols:
        df = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"Dropping non-numeric columns: {non_numeric}")
        df = df.drop(columns=non_numeric)

    node_names = df.columns.tolist()
    data_np = df.astype(float).to_numpy()
    print(f"Dataset: {data_np.shape[0]} rows x {data_np.shape[1]} columns")

    true_adj = None
    if dag_path and os.path.exists(dag_path):
        print(f"Loading ground truth DAG from: {dag_path}")
        dag = load_dag_from_file(dag_path)
        true_adj = dag_to_adjacency_matrix(dag, node_names)
        print(f"Ground truth edges: {np.sum(true_adj)}")

    timing_results, comparison_results, gt_metrics_results = [], [], []

    for algo_name in algorithms:
        if algo_name not in ALGO_RUNNERS:
            print(f"[WARN] Unknown algorithm: {algo_name}, skipping.")
            continue

        runner = ALGO_RUNNERS[algo_name]
        print(f"\n{'=' * 60}\nAlgorithm: {algo_name}\n{'=' * 60}")

        nx_graphs, raw_results, n_edges_map = {}, {}, {}

        for n_jobs in n_jobs_list:
            print(f"\n  n_jobs={n_jobs}...", end=" ", flush=True)
            start = time.perf_counter()
            try:
                result = runner(data_np, node_names, alpha, indep_test, n_jobs)
                elapsed = time.perf_counter() - start
                success = True
            except Exception as e:
                elapsed, success, result = -1, False, None
                print(f"FAILED: {e}")

            if success:
                nx_g = result_to_nx(result, node_names)
                nx_graphs[n_jobs] = nx_g
                raw_results[n_jobs] = result
                n_edges = nx_g.number_of_edges()
                n_edges_map[n_jobs] = n_edges
                print(f"Done in {elapsed:.2f}s, edges={n_edges}")
            else:
                n_edges_map[n_jobs] = -1

            timing_results.append({
                "algorithm": algo_name, "n_jobs": n_jobs,
                "time_seconds": round(elapsed, 4) if success else -1,
                "n_edges": n_edges_map.get(n_jobs, -1),
                "n_rows": data_np.shape[0], "n_cols": data_np.shape[1],
                "alpha": alpha, "indep_test": indep_test, "success": success,
            })

        # Попарное сравнение (NX-based, для детерминизма)
        successful_jobs = sorted(nx_graphs.keys())
        if len(successful_jobs) >= 2:
            base_jobs = successful_jobs[0]
            print(f"\n  --- Попарное сравнение (base: n_jobs={base_jobs}) ---")
            all_identical = True
            for n_jobs in successful_jobs:
                if n_jobs == base_jobs:
                    comparison_results.append({
                        "algorithm": algo_name, "base_n_jobs": base_jobs, "compare_n_jobs": n_jobs,
                        "SHD": 0, "Adj_F1": 1.0, "Arrow_F1": 1.0, "identical": True})
                    continue
                m = _calculate_nx_metrics(nx_graphs[base_jobs], nx_graphs[n_jobs])
                ident = m["SHD"] == 0
                if not ident: all_identical = False
                comparison_results.append({
                    "algorithm": algo_name, "base_n_jobs": base_jobs, "compare_n_jobs": n_jobs,
                    "SHD": m["SHD"], "Adj_F1": m["Adj_F1"], "Arrow_F1": m["Arrow_F1"], "identical": ident})
                print(f"  {'[PASS]' if ident else '[DIFF]'} n_jobs={base_jobs} vs {n_jobs}: "
                      f"SHD={m['SHD']}, Adj_F1={m['Adj_F1']:.4f}")
            print(f"\n  {'[SUCCESS]' if all_identical else '[WARNING]'} {algo_name}: "
                  f"{'Все графы ИДЕНТИЧНЫ' if all_identical else 'Обнаружены различия!'}")

        # Ground Truth метрики (endpoint-aware, через metrics.py)
        if true_adj is not None and len(successful_jobs) > 0:
            base_result = raw_results[successful_jobs[0]]
            est_ep = result_to_endpoint_matrix(base_result, node_names)
            is_pag = is_pag_result(base_result)
            gt_m = compute_metrics_from_dag_and_result(true_adj, est_ep, is_pag=is_pag)
            gt_m["algorithm"] = algo_name
            gt_metrics_results.append(gt_m)

    # Сохранение
    timing_df = pd.DataFrame(timing_results)
    timing_df.to_csv(os.path.join(output_dir, "parallelism_timing.csv"), index=False)
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(os.path.join(output_dir, "parallelism_graph_comparison.csv"), index=False)
    _plot_speedup(timing_df, output_dir)

    print(f"\n{'=' * 80}\nTIMING SUMMARY\n{'=' * 80}")
    print(timing_df[["algorithm", "n_jobs", "time_seconds", "n_edges", "success"]].to_string(index=False))

    if len(comparison_df) > 0:
        print(f"\n{'=' * 80}\nGRAPH COMPARISON\n{'=' * 80}")
        base_comp = comparison_df[
            comparison_df["base_n_jobs"] == comparison_df.groupby("algorithm")["base_n_jobs"].transform("min")]
        print(base_comp[["algorithm", "base_n_jobs", "compare_n_jobs", "SHD", "Adj_F1", "Arrow_F1", "identical"]]
              .to_string(index=False))

    if gt_metrics_results:
        gt_df = pd.DataFrame(gt_metrics_results)
        gt_df.to_csv(os.path.join(output_dir, "parallelism_gt_metrics.csv"), index=False)
        print(f"\n{'=' * 80}\nGROUND TRUTH METRICS\n{'=' * 80}")
        print(gt_df[["algorithm", "SHD", "adj_f1", "arrow_f1", "adj_precision", "adj_recall"]].to_string(index=False))

    return timing_df, comparison_df


def _plot_speedup(results_df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for algo in results_df["algorithm"].unique():
        sub = results_df[(results_df["algorithm"] == algo) & (results_df["success"])]
        if len(sub) == 0: continue
        axes[0].plot(sub["n_jobs"], sub["time_seconds"], "o-", label=algo, linewidth=2, markersize=8)
    axes[0].set(xlabel="CPU cores", ylabel="Time (s)", title="Execution Time vs Cores")
    axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_xticks(results_df["n_jobs"].unique())

    for algo in results_df["algorithm"].unique():
        sub = results_df[(results_df["algorithm"] == algo) & (results_df["success"])].sort_values("n_jobs")
        if len(sub) == 0: continue
        t1 = sub[sub["n_jobs"] == sub["n_jobs"].min()]["time_seconds"].values
        if len(t1) == 0: continue
        axes[1].plot(sub["n_jobs"], t1[0] / sub["time_seconds"], "o-", label=algo, linewidth=2, markersize=8)
    max_j = results_df["n_jobs"].max()
    axes[1].plot(range(1, max_j + 1), range(1, max_j + 1), "k--", alpha=0.3, label="Ideal")
    axes[1].set(xlabel="CPU cores", ylabel="Speedup", title="Parallelization Speedup")
    axes[1].legend(); axes[1].grid(True, alpha=0.3); axes[1].set_xticks(results_df["n_jobs"].unique())

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parallelism_speedup.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dag_path", type=str, default=None)
    parser.add_argument("--algorithms", nargs="+", default=["PC", "FCI", "RAI"],
                        choices=["PC", "FCI", "RAI", "FCI_v2"])
    parser.add_argument("--n_jobs_list", nargs="+", type=int, default=[1, 2, 4, 8, 12, 24, 32])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--indep_test", type=str, default="fisherz")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results")
    parser.add_argument("--exclude_cols", nargs="*", default=["epk_id", "report_dt"])
    args = parser.parse_args()

    run_benchmark(args.data_path, args.algorithms, args.n_jobs_list, args.alpha,
                  args.indep_test, args.output_dir, args.exclude_cols, args.dag_path)


if __name__ == "__main__":
    main()