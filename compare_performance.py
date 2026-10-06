import subprocess
import pandas as pd
import glob
import os
import sys
import numpy as np


def run_benchmark(algo_config, benchmark="feedback"):
    print(f"\n--- Running benchmark with config: {algo_config} ---")
    cmd = [
        "python", "main.py",
        f"benchmark={benchmark}",
        f"algorithm={algo_config}",
        "hydra.run.dir=."
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!!! Error running {algo_config}. Exit code: {e.returncode}")
        return None

    # Ищем results.csv
    results_path = os.path.join("results", "results.csv")
    if os.path.exists(results_path):
        return pd.read_csv(results_path)
    elif os.path.exists("results.csv"):
        return pd.read_csv("results.csv")
    else:
        print(f"Warning: results.csv was not created.")
        return None


def analyze_results():
    print("\n==============================")
    print("      ANALYZING RESULTS       ")
    print("==============================")

    # Очистка
    results_path = os.path.join("results", "results.csv")
    if os.path.exists(results_path): os.remove(results_path)
    for f in glob.glob(os.path.join("results", "profile_*.csv")): os.remove(f)

    # 1. Запуск Sequential
    df_seq = run_benchmark("pc_stable_seq")

    # 2. Запуск Parallel
    df_par = run_benchmark("pc_stable_par")

    if df_seq is None or df_par is None:
        print("One of the runs failed. Cannot compare.")
        return

    # 3. Создание таблицы сравнения (Validation Table)
    # Объединяем два датафрейма по имени бенчмарка
    merged = pd.merge(
        df_seq,
        df_par,
        on='bname',
        suffixes=('_seq', '_par')
    )

    # Выбираем метрики для сравнения
    metrics_to_compare = ['SHD.mean', 'adj_f1.mean', 'arrow_f1.mean']

    validation_data = []

    print("\n=== VALIDATION REPORT (Sequential vs Parallel) ===")

    all_passed = True

    for index, row in merged.iterrows():
        bname = row['bname']
        n_nodes = row['n_nodes_seq']  # или _par, они одинаковые

        # Считаем ускорение
        t_seq = row['execution_time.mean_seq']
        t_par = row['execution_time.mean_par']
        speedup = t_seq / t_par if t_par > 0 else 0

        row_res = {
            'Benchmark': bname,
            'Nodes': n_nodes,
            'Time (Seq)': f"{t_seq:.4f}",
            'Time (Par)': f"{t_par:.4f}",
            'Speedup': f"{speedup:.2f}x"
        }

        # Проверяем метрики
        mismatch_found = False
        for m in metrics_to_compare:
            val_seq = row[f"{m}_seq"]
            val_par = row[f"{m}_par"]
            diff = abs(val_seq - val_par)

            row_res[f"{m} (Seq)"] = f"{val_seq:.4f}"
            row_res[f"{m} (Par)"] = f"{val_par:.4f}"

            # Допускаем небольшую погрешность float
            if diff > 1e-5:
                mismatch_found = True
                row_res[f"{m} Diff"] = f"!!! {diff:.4f}"
            else:
                row_res[f"{m} Diff"] = "OK"

        if mismatch_found:
            row_res['Status'] = 'FAIL'
            all_passed = False
        else:
            row_res['Status'] = 'PASS'

        validation_data.append(row_res)

    # Создаем DataFrame для красивого вывода
    val_df = pd.DataFrame(validation_data)

    # Настраиваем порядок колонок для вывода
    cols_order = ['Benchmark', 'Nodes', 'Status', 'Speedup',
                  'SHD.mean (Seq)', 'SHD.mean (Par)', 'SHD.mean Diff',
                  'arrow_f1.mean (Seq)', 'arrow_f1.mean (Par)', 'arrow_f1.mean Diff']

    # Выводим таблицу
    print(val_df[cols_order].to_string(index=False))

    if all_passed:
        print("\n[SUCCESS] All metrics match between Sequential and Parallel versions.")
    else:
        print("\n[WARNING] Some metrics do not match! Check the table above.")

    # Сохраняем полную таблицу
    val_df.to_csv("comparison_validation.csv", index=False)
    print("\nFull validation report saved to comparison_validation.csv")

    # 4. Профилирование (как и раньше)
    print("\n--- Profiling Summary ---")
    files = glob.glob(os.path.join("results", "profile_*.csv"))
    dfs_prof = []
    for f in files:
        try:
            data = pd.read_csv(f)
            data['type'] = 'Parallel' if "Parallel" in f else 'Sequential'
            dfs_prof.append(data)
        except:
            pass

    if dfs_prof:
        full_prof = pd.concat(dfs_prof)
        summary = full_prof.groupby(['type', 'name'])['duration'].sum().unstack(0)
        if 'Sequential' in summary.columns and 'Parallel' in summary.columns:
            summary['Speedup'] = summary['Sequential'] / summary['Parallel']
        print(summary)


if __name__ == "__main__":
    analyze_results()