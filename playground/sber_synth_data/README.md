# Генерация и бенчмаркинг Causal Discovery для SCL (Cayman)


Для конкретного примера, я работал с реальными данными такого порядка:  
Количество строк: 621193 <br>
Количество признаков: 	47 <br>
Но на 47 признаках FCI скорее всего подавиться. 
Для параллельного FCI может быть выгодно что-то ~ 20-30 фичей.

## Обзор

Три скрипта для полного цикла тестирования библиотеки:

| Скрипт | Назначение |
|--------|-----------|
| `data_generator_advanced.py` | Генерация синтетических банковских данных с каузальной структурой |
| `benchmark_parallelism.py` | Замер ускорения при распараллеливании (1→32 ядра) |
| `benchmark_metrics.py` | Сравнение метрик с bagging и без |

---

## 1. Генерация данных

### Быстрый старт (один датасет, приоритет 1)

```bash
python data_generator_advanced.py \
    --n_rows 500000 \
    --n_features 30 \
    --nonlinearity linear \
    --seed 42 \
    --output_dir ./synthetic_data
```

### Матрица 3×3 датасетов (все варианты)

```bash
# Линейные зависимости (100%)
for nf in 20 40 60; do
    python data_generator_advanced.py --n_rows 500000 --n_features $nf --nonlinearity linear --output_dir ./synthetic_data
done

# Квази-линейные (80-90% линейные)
for nf in 20 40 60; do
    python data_generator_advanced.py --n_rows 500000 --n_features $nf --nonlinearity quasi --output_dir ./synthetic_data
done

# Нелинейные (<50% линейные)
for nf in 20 40 60; do
    python data_generator_advanced.py --n_rows 500000 --n_features $nf --nonlinearity nonlinear --output_dir ./synthetic_data
done
```

### Что генерируется

Для каждого запуска создаются два файла:
- `synth_banking_{N}f_{type}.parquet` — данные для алгоритмов
- `synth_banking_{N}f_{type}_dag.txt` — ground truth DAG (формат: `parent -> child`)

### Каузальная структура

- **Treatment**: `comm_fl` (бинарный — отправка SMS)
- **Mediator**: `sale_fl` (бинарный — открытие кредита)
- **Target**: `sdo_pass_5m` (float — сумма в рублях)
- Цепочка: `comm_fl → sale_fl → sdo_pass_5m`
- Скрытый конфаундер "активность клиента" влияет на все три
- Фоновые признаки с именами из реального банковского домена

---


### 🧪 Как теперь запускать генерацию

Вы получили полный контроль над математической природой данных:

1. **Все возможные нелинейности (самый жесткий вариант):**
```bash
python data_generator_v4.py --nonlinearity nonlinear --density sparse --nl_types mixed
```

2. **Только монотонные нелинейности (логарифмы, кубы, сигмоиды):**
*(Fisher-Z часто может "схватить" их через аппроксимацию, поэтому PC может показать неплохие результаты)*
```bash
python data_generator_v4.py --nonlinearity nonlinear --density sparse --nl_types cube log sig tanh
```

3. **Только периодические и симметричные нелинейности (sin, cos, модуль):**
*(Линейная корреляция Пирсона для $Y = \sin(X)$ или $Y = |X|$ строго равна 0. На этих данных Fisher-Z полностью провалится, что докажет необходимость использования ядерных тестов)*
```bash
python data_generator_v4.py --nonlinearity nonlinear --density sparse --nl_types sin cos abs
```

4. **Квази-линейный сценарий с мягкими искажениями:**
```bash
python data_generator_v4.py --nonlinearity quasi --density sparse --nl_types log sq
```


## 2. Бенчмарк распараллеливания

### Запуск

```bash
python benchmark_parallelism.py \
    --data_path ./synthetic_data/synth_banking_30f_linear.parquet \
    --algorithms PC FCI RAI \
    --n_jobs_list 1 2 4 8 12 24 32 \
    --alpha 0.05 \
    --output_dir ./benchmark_results
```

### Что делает

1. Загружает синтетические данные
2. Для каждого алгоритма (PC, FCI, RAI) и каждого значения n_jobs:
   - Запускает алгоритм
   - Замеряет время
3. **Проверяет детерминизм**: метрики графов должны быть близки к идеальным при разном n_jobs
4. Строит два графика:
   - Время выполнения vs количество ядер
   - Ускорение (speedup = T₁/Tₙ) vs количество ядер

### Выходные файлы

- `benchmark_results/parallelism_benchmark.csv` — таблица результатов
- `benchmark_results/parallelism_speedup.png` — графики ускорения

### Пример CSV

| algorithm | n_jobs | time_seconds | n_edges | graph_hash | success |
|-----------|--------|-------------|---------|------------|---------|
| PC        | 1      | 120.34      | 45      | a1b2c3d4   | True    |
| PC        | 4      | 38.12       | 45      | a1b2c3d4   | True    |
| PC        | 8      | 22.56       | 45      | a1b2c3d4   | True    |

---

## 3. Бенчмарк метрик (с bagging и без)

### Запуск

```bash
python benchmark_metrics.py \
    --data_path ./synthetic_data/synth_banking_30f_linear.parquet \
    --dag_path ./synthetic_data/synth_banking_30f_linear_dag.txt \
    --algorithms PC FCI RAI \
    --n_jobs 8 \
    --n_bootstraps 10 \
    --tau 0.5 \
    --output_dir ./metrics_results
```

### Что делает

Для каждого алгоритма:

1. **Без bagging**: один запуск → метрики по ground truth
2. **С bagging** (bootstrap consensus):
   - N бутстреп-выборок → N графов
   - Матрица уверенности (частота появления ребра)
   - Порог τ → робастный скелет
   - Финальный прогон на полных данных → ориентация
   - Метрики по ground truth

### Метрики

| Метрика | Описание |
|---------|----------|
| SHD | Structural Hamming Distance (чем меньше — тем лучше) |
| adj_precision | Точность обнаружения ребер скелета |
| adj_recall | Полнота обнаружения ребер скелета |
| adj_f1 | F1-мера для скелета |
| arrow_precision | Точность направленных ребер |
| arrow_recall | Полнота направленных ребер |
| arrow_f1 | F1-мера для направленных ребер |

### Выходные файлы

- `metrics_results/metrics_benchmark.csv` — таблица метрик
- `metrics_results/metrics_comparison.png` — bar chart сравнения

---

## Полный пайплайн (copy-paste)

```bash
# 0. Переход в директорию SCL
cd /path/to/SCL

# 1. Генерация приоритетного датасета (30 фич, линейный, 500К строк)
python playground/data_generator_advanced.py \
    --n_rows 500000 --n_features 30 --nonlinearity linear \
    --output_dir ./synthetic_data

# 2. Бенчмарк распараллеливания
python playground/benchmark_parallelism.py \
    --data_path ./synthetic_data/synth_banking_30f_linear.parquet \
    --algorithms PC \
    --n_jobs_list 1 2 4 8 12 24 32 \
    --output_dir ./benchmark_results

# 3. Метрики с bagging и без
python playground/benchmark_metrics.py \
    --data_path ./synthetic_data/synth_banking_30f_linear.parquet \
    --dag_path ./synthetic_data/synth_banking_30f_linear_dag.txt \
    --algorithms PC FCI RAI \
    --n_jobs 8 --n_bootstraps 10 \
    --output_dir ./metrics_results
```

---

## Зависимости

Все скрипты используют только стандартные зависимости SCL:
- numpy, pandas, scipy, matplotlib, joblib
- causal-learn (для типов графов)
- pyarrow (для Parquet)

Synthcity **не требуется** для запуска — генератор самодостаточен и реализует DAG-based generation с контролируемой нелинейностью напрямую.

---

## Таблица приоритетов датасетов

| Количество признаков | Линейный (100%) | Квази-линейный (80-90%) | Нелинейный (<50%) |
|---------------------|-----------------|------------------------|--------------------|
| 10-20               | 2 приоритет     | 2 приоритет            | 3 приоритет        |
| **20-40**           | **1 приоритет** | 2 приоритет            | 3 приоритет        |
| 40-60               | 1 приоритет     | 1 приоритет            | 3 приоритет        |

Количество строк в датасете: 400 000 — 600 000
