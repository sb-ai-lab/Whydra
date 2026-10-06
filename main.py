import shutil
import hydra
from omegaconf import DictConfig
import pandas as pd
import os
import numpy as np
import time
from tqdm import tqdm
import sys
from pathlib import Path

np.set_printoptions(threshold=sys.maxsize, linewidth=200)

from causal_graphs.evaluation.metrics import calculate_metrics
from causal_graphs.evaluation.graph_utils import load_ground_truth, draw_graph, adj_matrix_to_graph, get_adj_matrix
from causal_graphs.algorithms.standalone.profiler import profiler


def cleanup_intermediate_files(directories):
    """
    Deletes the specified directories and their contents.
    """
    for d in directories:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
            except Exception as e:
                print(f"Error cleaning up {d}: {e}")


def generate_bootstrap_samples(data, n_bootstraps, samples_dir, loader=None):
    """
    Generates bootstrap samples and saves them to the specified directory.
    Returns a list of corresponding file paths.
    """
    os.makedirs(samples_dir, exist_ok=True)
    sample_paths = []

    print(f"Generating {n_bootstraps} bootstrap samples in {samples_dir}...")
    for i in range(n_bootstraps):
        sample_path = os.path.join(samples_dir, f"sample_{i}.npy")
        # Generate if it doesn't exist
        if not os.path.exists(sample_path):
            if i == 0:
                sample_data = data
            else:
                sample_size = int(data.shape[0] * 0.8)
                indices = np.random.choice(data.shape[0], sample_size, replace=False)
                sample_data = data[indices]

            # Save as a plain NumPy array (no pickle) for safe reloading
            np.save(sample_path, sample_data, allow_pickle=False)

            # Optional: Keep the text format if loader is provided, for compatibility with legacy logging
            if loader:
                try:
                    loader.save_data(sample_data, samples_dir, f"sample_data_{i}.txt")
                except Exception:
                    pass  # Ignore if save_data is not compatible or fails

        sample_paths.append(sample_path)

    return sample_paths


def process_single_benchmark(i, sample_path, algo, bname, n_bootstrap, n_jobs, method_name, result_dir):
    """
    Runs the algorithm on a single sample.
    Returns the estimated graph and execution time.
    """
    # Construct result filename
    filename = f"{method_name}_{bname}_{n_bootstrap}_{i}_{n_jobs}.npz"
    file_algo_path = os.path.join(result_dir, filename)
    os.makedirs(result_dir, exist_ok=True)

    execution_time = 0.0

    # If result already exists, load it (data-only cache: adjacency matrix + node names)
    if os.path.exists(file_algo_path):
        with np.load(file_algo_path, allow_pickle=False) as cached:
            est_G = adj_matrix_to_graph(cached["adj_matrix"],
                                        [str(n) for n in cached["node_names"]])
    else:
        # Otherwise load sample and run algo
        sample_data = np.load(sample_path, allow_pickle=False)

        start_time = time.perf_counter()
        est_G = algo.run(sample_data)
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Save result as plain arrays (no pickle): adjacency matrix + node names
        node_names = [n.get_name() for n in est_G.get_nodes()]
        np.savez(file_algo_path,
                 adj_matrix=get_adj_matrix(est_G),
                 node_names=np.array(node_names))

    return est_G, execution_time


def print_debug_matrices(true_G, est_G):
    print("DEBUG: Estimated Graph Matrix (first):")
    est_mat = get_adj_matrix(est_G)
    print(est_mat[:50, :50])

    print("DEBUG: True Graph Matrix (first):")
    true_mat = get_adj_matrix(true_G)
    print(true_mat[:50, :50])


def evaluate_result(true_G, est_G, i, plot_dir, save_plots):
    print("DEBUG: Calculating metrics...")
    m = calculate_metrics(true_G, est_G)
    print("DEBUG: Metrics calculated.")

    if save_plots and plot_dir:
        print("DEBUG: Drawing graph...")
        draw_graph(est_G, plot_dir, f"iter_{i}")
        print("DEBUG: Graph drawn.")

    return m


def calculate_consensus_matrix(graphs, tau, save_path=None):
    """
    Extracts adjacency matrices from all estimated graphs,
    sums them up element-wise, and divides by the number of graphs.
    """
    if not graphs:
        return None

    print(f"Calculating consensus matrix from {len(graphs)} graphs...")
    matrices = [np.abs(get_adj_matrix(g)) for g in graphs]

    # Sum all matrices and divide by count
    consensus_matrix_raw = np.sum(matrices, axis=0) / len(matrices)

    consensus_matrix = np.zeros_like(consensus_matrix_raw)
    consensus_matrix[consensus_matrix_raw > tau] = 1

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                for i, mat in enumerate(matrices):
                    f.write(f"--- Matrix {i} ---\n")
                    np.savetxt(f, mat, fmt='%.4f')
                    f.write("\n")

                f.write(f"--- Consensus Matrix (tau={tau}) ---\n")
                np.savetxt(f, consensus_matrix, fmt='%.4f')
        except Exception as e:
            print(f"Error saving matrices: {e}")

    return consensus_matrix

def save_metrics_for_by_bootstrap(estimated_graphs, true_G, output_dir, output_filename):
    print("Calculating metrics by bootstraps...")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    results = []
    for i, est_G in enumerate(estimated_graphs):
        res_row = calculate_metrics(true_G, est_G)
        res_row['num_iter_bootstrap']=i
        results.append(res_row)
    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


def run_metrics_evaluation(n_bootstraps,
                           estimated_graphs,
                           true_G,
                           current_plot_dir,
                           save_plots,
                           consensus_matrix=None,
                           node_names=None):
    metrics_buffer = []
    print("Starting evaluation phase...")
    # for i in tqdm(range(n_bootstraps), desc="Calculating metrics"):
    est_G = estimated_graphs[0]

    if consensus_matrix is not None and node_names is not None:
        est_mat = get_adj_matrix(est_G)
        est_mat = est_mat * consensus_matrix
        est_G = adj_matrix_to_graph(est_mat, node_names)

    m = calculate_metrics(true_G, est_G)
    metrics_buffer.append(m)

    if save_plots and current_plot_dir:
        print("DEBUG: Drawing graph...")
        draw_graph(est_G, current_plot_dir, f"iter_{0}")
        print("DEBUG: Graph drawn.")

    return metrics_buffer


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # ---------------------------------------------------------------
    # 0. Чтение конфигурационных переменных
    # ---------------------------------------------------------------
    if cfg.get("seed"):
        np.random.seed(cfg.seed)
    tau = cfg.get("tau", 0.5)
    save_plots = cfg.get("save_plots", False)
    n_runs = cfg.get("n_bootstraps", 1)
    n_jobs =cfg.algorithm.get("n_jobs", 1)
    # Получаем тест независимости
    target_indep_test = cfg.benchmark.get("indep_test", "fisherz")

    method_name = cfg.algorithm._target_.split('.')[-1]
    library_name = "causal_graphs"

    print(f"Running algorithm: {method_name}")
    print(f"Using independence test: {target_indep_test}")

    loader = hydra.utils.instantiate(cfg.benchmark.loader)
    algo = hydra.utils.instantiate(cfg.algorithm, indep_test=target_indep_test)

    results = []

    for bname in cfg.benchmark.names:
        print(f"Processing {bname}...")

        try:
            # 1. Загрузка данных и Ground Truth
            data = loader.load_data(bname)
            gt_path = loader.load_ground_truth_path(bname)
            true_G = load_ground_truth(gt_path)
            n_nodes = true_G.get_num_nodes()

            # --- ВИЗУАЛИЗАЦИЯ: Папка для текущего эксперимента ---
            # Структура: plots / ИмяАлгоритма / ИмяБенчмарка
            current_plot_dir = None
            if save_plots:
                current_plot_dir = os.path.join(cfg.plots_dir, method_name, bname)

                # Сохраняем истинный граф (Ground Truth) для сравнения
                draw_graph(true_G, current_plot_dir, "ground_truth")
            # -----------------------------------------------------

            execution_times = []

            # ---------------------------------------------------------------
            # --- STAGE 1: Data Preparation ---
            # ---------------------------------------------------------------
            # Create a dedicated directory for bootstrap samples for this benchmark
            samples_dir = os.path.join("input_samples", bname)
            algo_results_dir_base = os.path.join("algo_results", bname)

            if not cfg.get("resume", True):
                print(f"Resume disabled. Cleaning up intermediate files for {bname}...")
                cleanup_intermediate_files([samples_dir, algo_results_dir_base])

            if cfg.get("bootstrap", True):
                print(f"Preparing data in {samples_dir}...")
                sample_paths = generate_bootstrap_samples(data, n_runs, samples_dir, loader)

            else:
                os.makedirs(samples_dir, exist_ok=True)
                print(f"Bootstrap disabled. Using original data.")
                sample_path = os.path.join(samples_dir, "sample_0.npy")
                sample_paths = [sample_path]
                n_runs = 1
                np.save(sample_path, data, allow_pickle=False)

                # Optional: Keep the text format if loader is provided, for compatibility with legacy logging
                try:
                    loader.save_data(data, samples_dir, f"sample_data_0.txt")
                except Exception:
                    pass  # Ignore if save_data is not compatible or fails

            # ---------------------------------------------------------------
            # --- STAGE 2: Algorithm Execution ---
            # ---------------------------------------------------------------
            # Now iterate over the prepared samples and run the algorithm
            iterator = tqdm(range(n_runs), desc=f"Benchmarking {bname}")
            estimated_graphs = []

            for i in iterator:

                algo_results_dir = os.path.join("algo_results", bname)
                est_G, duration = process_single_benchmark(
                    i, sample_paths[i], algo, bname, n_runs, n_jobs, method_name, algo_results_dir
                )

                # Log execution time (only if we actually ran the algorithm)
                if duration > 0:
                    execution_times.append(duration)

                estimated_graphs.append(est_G)

                # print(type(est_G))
                # print("DEBUG: Algo run finished.")
                #
                # print_debug_matrices(true_G, est_G)

            consensus_metrics = {}
            consensus_matrix = None
            node_names = [n.get_name() for n in true_G.get_nodes()]

            if cfg.get("bootstrap", True):
                # Consensus Matrix from Bootstrap
                consensus_matrix = calculate_consensus_matrix(estimated_graphs, tau,
                                                              save_path=os.path.join("algo_results", bname,
                                                                                     "matrix.txt"))
                # print("DEBUG: Consensus Matrix (mean of adjacency matrices):")
                # print(consensus_matrix)

            # ---------------------------------------------------------------
            # --- STAGE 3: Metrics Evaluation ---
            # ---------------------------------------------------------------
            filename_for_boot = Path(cfg.output_filename_for_bootstrap).stem
            filename_suffix = Path(cfg.output_filename_for_bootstrap).suffix
            output_filename_for_bootstrap=f"{filename_for_boot}_{method_name}_{bname}{filename_suffix}"
            save_metrics_for_by_bootstrap(estimated_graphs, true_G, cfg.output_dir, output_filename_for_bootstrap)

            metrics_buffer = run_metrics_evaluation(
                n_runs, estimated_graphs, true_G, current_plot_dir, save_plots, consensus_matrix, node_names
            )

            df_metrics = pd.DataFrame(metrics_buffer)

            res_row = {
                "bname": bname,
                "n_bootstraps": n_runs,
                "parallel_n_jobs": n_jobs,
                "method": method_name,
                "independence_test": target_indep_test,
                "library": library_name,
                "n_nodes": n_nodes,
                "execution_time.mean": np.mean(execution_times),
                "execution_time.std": np.std(execution_times),
                "execution_time.total": np.sum(execution_times)
            }

            for col in df_metrics.columns:
                res_row[f"{col}.mean"] = df_metrics[col].mean()
                res_row[f"{col}.std"] = df_metrics[col].std()

            # Add consensus metrics
            for k, v in consensus_metrics.items():
                res_row[f"{k}.consensus"] = v

            results.append(res_row)

            # Cleanup intermediate files if keep_intermediate_files is False
            if not cfg.get("keep_intermediate_files", True):
                cleanup_intermediate_files([samples_dir, algo_results_dir_base])

        except Exception as e:
            print(f"Error on {bname}: {e}")
            import traceback
            traceback.print_exc()


    if results:
        df_res = pd.DataFrame(results)
        output_path = os.path.join(cfg.output_dir, cfg.output_filename)
        os.makedirs(cfg.output_dir, exist_ok=True)
        df_res.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

        stats = profiler.get_stats()
        if not stats.empty:
            base_name = os.path.splitext(cfg.output_filename)[0]
            profile_filename = f"profile_{base_name}.csv"
            profile_path = os.path.join(cfg.output_dir, profile_filename)
            stats.to_csv(profile_path, index=False)

        profiler.clear()

if __name__ == "__main__":
    main()
