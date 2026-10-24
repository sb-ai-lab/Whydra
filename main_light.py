import hydra
from omegaconf import DictConfig
import pandas as pd
import os
import numpy as np
import time
from tqdm import tqdm
import sys
np.set_printoptions(threshold=sys.maxsize, linewidth=200)

from whydra.evaluation.graph_utils import draw_graph
from whydra.algorithms.standalone.profiler import profiler


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if cfg.get("seed"):
        np.random.seed(cfg.seed)

    method_name = cfg.algorithm._target_.split('.')[-1]
    library_name = "causal_graphs"

    # Получаем тест независимости
    target_indep_test = cfg.benchmark.get("indep_test", cfg.algorithm.get("indep_test", "fisherz"))

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

            execution_times = []

            iterator = tqdm(range(cfg.n_bootstraps), desc=f"Benchmarking {bname}")

            for i in iterator:
                # indices = np.random.choice(data.shape[0], data.shape[0], replace=True)
                # sample_data = data[indices]
                sample_data = data

                start_time = time.perf_counter()
                est_G = algo.run(sample_data)

                print("DEBUG: Algo run finished.")
                end_time = time.perf_counter()

                execution_times.append(end_time - start_time)

                # --- ОТЛАДКА МАТРИЦ ---
                from causal_graphs.evaluation.metrics import get_adj_matrix
                print("DEBUG: Estimated Graph Matrix (first):")
                est_mat = get_adj_matrix(est_G)
                print(est_mat[:50, :50])
                # ----------------------

                if cfg.get("save_plots", False):
                    print("DEBUG: Drawing graph...")
                    current_plot_dir = os.path.join(cfg.plots_dir, method_name, bname)
                    draw_graph(est_G, current_plot_dir, f"iter_{i}")
                    print("DEBUG: Graph drawn.")



            res_row = {
                "bname": bname,
                "n_bootstraps": cfg.n_bootstraps,
                "method": method_name,
                "independence_test": target_indep_test,
                "library": library_name
            }

            res_row["execution_time.mean"] = np.mean(execution_times)
            res_row["execution_time.std"] = np.std(execution_times)
            res_row["execution_time.total"] = np.sum(execution_times)

            results.append(res_row)

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