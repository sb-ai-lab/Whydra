import os
import glob
import subprocess
import pandas as pd
from pathlib import Path


def get_config_names(config_dir):
    """Получает список имен конфигов (без .yaml) из указанной папки."""
    files = glob.glob(os.path.join(config_dir, "*.yaml"))
    return [Path(f).stem for f in files]


def run_experiment(algo_name, bench_name):
    # Формируем уникальное имя файла для результатов
    # Например: results_pc_stable_feedback.csv
    output_filename = f"results_{algo_name}_{bench_name}.csv"

    print(f"\nRunning: Algorithm=[{algo_name}] on Benchmark=[{bench_name}]")
    print(f"   Output will be saved to: results/{output_filename}")

    cmd = [
        "python", "main.py",
        f"algorithm={algo_name}",
        f"benchmark={bench_name}",
        f"output_filename={output_filename}",  # Переопределяем имя файла
        "hydra.run.dir=."  # Чтобы логи hydra не плодили папки с датами
    ]

    try:
        subprocess.run(cmd, check=True)
        return os.path.join("results", output_filename)
    except subprocess.CalledProcessError:
        print(f"Failed: {algo_name} on {bench_name}")
        return None


def aggregate_results():
    print("\nAggregating all results...")
    all_files = glob.glob(os.path.join("results", "results_*.csv"))

    dfs = []
    for f in all_files:
        # Пропускаем файл summary, если он уже есть
        if "summary_all.csv" in f:
            continue
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Skipping bad file {f}: {e}")

    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)

        # Сортировка для красоты
        cols = ['bname', 'method', 'library', 'n_nodes', 'SHD.mean', 'adj_f1.mean', 'arrow_f1.mean',
                'execution_time.mean']
        # Оставляем только те колонки, которые есть
        cols = [c for c in cols if c in full_df.columns]

        # Сохраняем сводный отчет
        full_df.to_csv("results/summary_all.csv", index=False)

        print("\n=== FINAL SUMMARY ===")
        print(full_df[cols].sort_values(by=['bname', 'method']).to_string(index=False))
        print(f"\nAggregated results saved to results/summary_all.csv")
    else:
        print("No results found to aggregate.")


if __name__ == "__main__":
    # 1. Ищем доступные конфиги
    algorithms = get_config_names("configs/algorithm")
    benchmarks = get_config_names("configs/benchmark")

    print(f"Found algorithms: {algorithms}")
    print(f"Found benchmarks: {benchmarks}")

    # 2. Запускаем все комбинации
    # Вы можете исключить какие-то алгоритмы вручную, если нужно
    # algorithms = [a for a in algorithms if "stable" in a]

    for algo in algorithms:
        for bench in benchmarks:
            run_experiment(algo, bench)

    # 3. Собираем итоговую таблицу
    aggregate_results()