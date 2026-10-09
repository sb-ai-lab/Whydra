import os
import glob
import subprocess
import importlib
import sys
import yaml
import pandas as pd
from pathlib import Path


# Бенчмарки, исключённые из прогона. Чтобы вернуть — убрать имя из списка.
#
# munin: кейса с таким именем нет в BenchmarkCaseRepository (проверено:
# 58 доступных кейсов, ни одного munin), поэтому оба конфига запускают
# ровно тот же перебор, что и остальные, только впустую тратят часы.
SKIPPED_BENCHMARKS = [
    "bnlearn_munin",
    "bnlearn_munin_with_truth",
]


def get_config_names(config_dir):
    """Получает список имен конфигов (без .yaml) из указанной папки."""
    files = glob.glob(os.path.join(config_dir, "*.yaml"))
    return [Path(f).stem for f in files]


def get_unavailable_reason(algo_name, config_dir="configs/algorithm"):
    """Проверяет доступность алгоритма по конфигу, возвращает причину недоступности или None."""
    try:
        config_path = os.path.join(config_dir, f"{algo_name}.yaml")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            return f"не удалось прочитать конфиг: {type(e).__name__}"

        if not config or "_target_" not in config:
            return "в конфиге нет _target_"

        target = config["_target_"]
        module_name, _, class_name = target.rpartition(".")
        if not module_name:
            return f"в _target_ нет пути к модулю: {target}"

        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            return f"модуль {module_name} не импортируется: {type(e).__name__}"

        if not hasattr(module, class_name):
            return f"класс {class_name} отсутствует в {module_name}"

        return None
    except Exception as e:
        return f"неожиданная ошибка: {type(e).__name__}"


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
    all_benchmarks = get_config_names("configs/benchmark")

    benchmarks = [b for b in all_benchmarks if b not in SKIPPED_BENCHMARKS]
    skipped_benchmarks = [b for b in all_benchmarks if b in SKIPPED_BENCHMARKS]

    print(f"Found algorithms: {algorithms}")
    print(f"Found benchmarks: {benchmarks}")
    if skipped_benchmarks:
        print(f"Skipped benchmarks: {skipped_benchmarks}")

    # 2. Фильтруем алгоритмы по доступности классов
    available_algos = []
    unavailable_algos = []

    for algo in algorithms:
        reason = get_unavailable_reason(algo)
        if reason is None:
            available_algos.append(algo)
        else:
            unavailable_algos.append((algo, reason))

    print(f"Available algorithms: {available_algos}")
    if unavailable_algos:
        print("Unavailable algorithms:")
        for algo, reason in unavailable_algos:
            print(f"  - {algo}: {reason}")

    if not available_algos:
        print("No available algorithms to run. Exiting.")
        sys.exit(0)

    # 3. Запускаем все комбинации
    for algo in available_algos:
        for bench in benchmarks:
            run_experiment(algo, bench)

    # 4. Собираем итоговую таблицу
    aggregate_results()