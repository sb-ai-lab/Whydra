
# Как использовать этот код в 2_graph

Заменить код в секции **"# Строим графы"** на следующий код. Он позволяет задать список алгоритмов и выполнит их последовательно, сохраняя результаты так же, как это делалось раньше.

```python
# ============================================================
# ИМПОРТ ВРАППЕРА И НАСТРОЙКА
# ============================================================
sys.path.insert(0, './SCL-sber/')
from playground.causal_wrapper import CausalRunner
import joblib
import copy
import os

# Автоматическое определение ядер
N_JOBS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
print(f"Detected available cores: {N_JOBS}")


# Можно задавать разные параметры для разных методов
RUN_CONFIGS = [
    {"method": "RAI", "alpha": 0.05, "indep_test": "fisherz"},
    # {"method": "PC",  "alpha": 0.01, "indep_test": "fisherz"}, # Более строгий PC
    # {"method": "FCI", "alpha": 0.05, "indep_test": "fisherz"},
]

# Инициализация раннера (дефолтные значения, если в конфиге не указаны)
runner = CausalRunner(
    library_path='./SCL-sber/', 
    n_jobs=N_JOBS
)

# Колонки для исключения
ignore_columns = [date_column, *technical_columns]

# ============================================================
# ЗАПУСК ЦИКЛА
# ============================================================
for config in RUN_CONFIGS:
    method = config["method"]
    # Извлекаем параметры, если они есть, иначе None (будут использованы дефолтные из runner)
    alpha = config.get("alpha")
    test = config.get("indep_test")
    
    print(f"\n=== ЗАПУСК {method} ===")
    
    # Запуск с передачей конкретных параметров
    result = runner.run(
        graph_ready_dataset_pd, 
        method=method, 
        ignore_cols=ignore_columns,
        alpha=alpha,
        indep_test=test
    )
    
    g = result['general_graph']
    nx_g = result['nx_graph']
    used_params = result['params']
    
    # Формирование уникального имени файла
    n_nodes = len(nx_g.nodes)
    # tag включает метод, альфу и тест, чтобы файлы не перезаписывались
    tag = f"{method.lower()}_{used_params['alpha']}_{used_params['test']}_{n_nodes}"
    
    print(f"Сохранение результатов: {tag}")
    joblib.dump(g, f"./{CACHE_DIR}/{tag}.pkl")
    joblib.dump(nx_g, f"./{CACHE_DIR}/{tag}_nx.pkl")
    
    last_tag = tag

print("\nВсе расчеты завершены.")
```

### Визуализация (PyVis / Matplotlib)

В секции **"# Визуализация pyvis"** (или где у вас `draw`) ничего глобально менять не нужно, если вы используете переменную `tag` (или `last_tag` из кода выше).

Если вы хотите визуализировать **все** результаты сразу после расчета, можно добавить этот блок сразу после цикла:

```python
from src.utils.draw import draw
from networkx import kamada_kawai_layout

# Визуализация последнего запущенного алгоритма (или можно сделать цикл по всем сохраненным тегам)
# Используем last_tag из предыдущей ячейки
current_tag = last_tag 
print(f"Визуализация для: {current_tag}")

nx_g_loaded = joblib.load(f"./{CACHE_DIR}/{current_tag}_nx.pkl")

draw(
    nx_g_loaded,
    current_tag,
    layout=kamada_kawai_layout,
    font_size_labels=15,
    node_color={
        node: "red" if node==target_column else \
        "green" if node==treatment_column else \
        "yellow" if node==mediator_column else \
        "blue" \
        for node in nx_g_loaded.nodes()
    },
    figsize=(20, 20)
)
```
