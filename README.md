# Whydra — Causal graph analysis

Библиотека для построения причинных графов и оценки uplift-эффекта в прикладных
задачах (в т.ч. banking).

## Содержание

- [Минимальные требования](#минимальные-требования)
- [Установка](#установка)
- [Опциональные зависимости](#опциональные-зависимости)
- [Поддерживаемые алгоритмы](#поддерживаемые-алгоритмы)
- [Быстрый старт: унифицированный API](#быстрый-старт-унифицированный-api)
- [Запуск пайплайна через Hydra](#запуск-пайплайна-через-hydra)
- [Данные к бенчмаркам](#данные-к-бенчмаркам)
- [Встроенные benchmark-кейсы](#встроенные-benchmark-кейсы)
- [Тесты](#тесты)

## Минимальные требования

- Python >= 3.10 (основной набор пинов); в `requirements.txt` справочно сохранены
  наборы для python 3.12.13 и клиентский набор для python 3.8
- pip >= 23
- Рекомендуется виртуальное окружение (`venv` или `conda`)

## Установка

```bash
conda create -n sber_env python=3.10 pip
conda activate sber_env
pip install -r requirements.txt
```

Пакет не публикуется в PyPI, но `pyproject.toml` в корне есть, поэтому
установка работает обычным образом:

```bash
pip install .                                   # из клона
pip install git+https://github.com/Causal-Fin-AI/Whydra.git
```

Можно и без установки — из корня репозитория, `whydra/` импортируется
как локальный пакет.

## Опциональные зависимости

### Graphviz (визуализация графов)

Ставится в систему, не через pip:
* **Linux:** `sudo apt-get install graphviz`
* **Windows:** инсталлятор с [сайта Graphviz](https://graphviz.org/download/),
  установить и **добавить путь к bin в PATH**.

### PyTorch (нужен алгоритму `dag_gnn`)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Если GPU не требуется — CPU-вариант по официальной инструкции PyTorch.

### Остальные опциональные зависимости

- **TensorFlow** — нужен `notears_stable` и `nobears_stable`
- **tigramite** — нужен `orig_j_pcmci_plus` (временные ряды)

Все они опциональны: без них соответствующий класс поднимает понятный
`ImportError`, а остальная библиотека работает.

## Поддерживаемые алгоритмы

Значения для поля `algorithm` (файлы в `configs/algorithm/`):

| Конфиг | Класс | Комментарий |
|---|---|---|
| `pc_stable` | `StandalonePCStable` | последовательный и параллельный PC-stable |
| `pc_stable_par` | `ParallelPCStable` | явно параллельная версия |
| `pc_stable_seq`, `pc_stable_4/8/16` | `StandalonePCStable` | преднастроенные `n_jobs` |
| `fci_stable` | `StandaloneFCIStable` | FCI-stable v2 (исправлены правила R1/R2/R5/R10) |
| `fci_stable_cash` | `StandaloneFCIStableCached` | с кэшем CI-тестов |
| `fci_stable_paralel_cash` | `StandaloneFCIStableParallelCached` | параллельный + кэш |
| `cfci_stable`, `cfci_stable_paralel_cash` | `StandaloneCFCIStable*` | conservative FCI |
| `rfci_stable` | `StandaloneRFCIStable` | RFCI |
| `gfci_stable` | `StandaloneGFCIStable` | GFCI |
| `fci_plus_stable`, `fci_plus_stable_paralel_cash` | `StandaloneFCIPlusStable*` | FCI+ |
| `rai_stable` | `StandaloneRAIStable` | RAI |
| `rai_optimized` | `StandaloneRAIOptimized` | оптимизированный RAI |
| `notears_stable` | `StandaloneNoTearsStable` | нужен TensorFlow |
| `nobears_stable` | `StandaloneNoBearsStable` | нужен TensorFlow |
| `dag_gnn` | `DagGNN` | нужен PyTorch |
| `orig_j_pcmci_plus` | `OrigJPCMCIPlus` | только временные ряды, нужен tigramite |

Единственная реализация FCI — `fci_algo_v2.py`: в ней исправлены правила
ориентации R1/R2/R5/R10. Прежняя v1 (`fci_algo.py`) удалена, сверка идёт
с causal-learn, pcalg и Tetrad в отдельных локальных верификационных
прогонах; временные сценарии и их результаты в репозиторий не входят.

### Фоновые знания (background knowledge)

`background_knowledge` поддерживают все алгоритмы семейств PC и FCI:
`pc_stable`, `pc_stable_par`, `fci_stable`, `fci_stable_cash`,
`fci_stable_paralel_cash`, `cfci_stable`, `cfci_stable_paralel_cash`,
`rfci_stable`, `gfci_stable`, `fci_plus_stable`,
`fci_plus_stable_paralel_cash`. Ограничения применяются на трёх фазах:

* `PRE_SEARCH` — до поиска скелета; обязательные рёбра запрещено удалять
  на шагах skeleton и pdsep;
* `BETWEEN_ORIENTATION` — после ориентации коллайдеров и после каждого
  прохода правил ориентации;
* `POST_ORIENTATION` — после сходимости.

```python
import numpy as np

from whydra import create_standalone_algorithm
from whydra.background_knowledge import BackgroundKnowledge, BKPhase

data = np.random.default_rng(0).standard_normal((500, 8))

bk = BackgroundKnowledge(
    enabled_phases=(BKPhase.PRE_SEARCH, BKPhase.BETWEEN_ORIENTATION, BKPhase.POST_ORIENTATION),
)
bk.require("X1", "X3")   # ребро X1 -> X3 обязательно
bk.forbid("X8", "X1")    # направление X8 -> X1 запрещено

algo = create_standalone_algorithm("fci_stable", n_jobs=4)
graph = algo.run(data, background_knowledge=bk)

# что именно применилось — в bk.history
for report in bk.history:
    print(report.checkpoint, report.phase, len(report.events))
```

Семейство RAI фоновые знания пока не поддерживает.

Для `pc_stable`, `fci_stable`, `rai_stable` параллелизм задаётся через `n_jobs`.
При `n_jobs=1` используется последовательный режим.

## Тесты условной независимости

Реализованы два теста (`whydra.algorithms.graph_core.CIT`):

| `indep_test` | Тип данных | Статистика |
|---|---|---|
| `fisherz` | непрерывные | Fisher-Z по частной корреляции |
| `chisq`   | дискретные  | хи-квадрат Пирсона |

Актуальный список всегда доступен как
`whydra.algorithms.graph_core.IMPLEMENTED_TESTS`.

Имена `parcorr` и `kci` встречаются в константе `CONTINUOUS_TESTS`, но **не
реализованы**: эта константа лишь классифицирует тесты по типу данных (её
использует выбор score-функции для GES/xGES), а вызов такого теста поднимает
`NotImplementedError`.

Замечание по дискретным данным: наш `chisq` — это хи-квадрат Пирсона, он
совпадает с causal-learn; `pcalg::disCItest` считает G², поэтому расхождения
с R на дискретных кейсах — это разница статистик, а не дефект алгоритма.

## Быстрый старт: унифицированный API

### Кейс 1: синтетические данные

```python
import numpy as np
from whydra import run_standalone_algorithm

rng = np.random.default_rng(42)
data = rng.standard_normal((1000, 10))

graph = run_standalone_algorithm(
    "pc_stable",
    data,
    alpha=0.05,
    indep_test="fisherz",
    n_jobs=4,
)
print(graph)
```

`run_standalone_algorithm` принимает как `np.ndarray`, так и путь к файлу
(`.pickle`/`.pkl`, `.npy`, текстовые таблицы). Список доступных имён —
`whydra.algorithms.wrappers.available_standalone_algorithms()`.

### Кейс 2: встроенные benchmark-данные

```python
from whydra import BenchmarkCaseRepository, run_standalone_algorithm

repo = BenchmarkCaseRepository()          # whydra_benchmarks/cases
print(list(repo.list_available_cases()))

data = repo.get_case_data(case_name="alarm", benchmark_name="bnlearn")
graph = run_standalone_algorithm("pc_stable", data, alpha=0.05, n_jobs=4)
print(graph)
```

## Запуск пайплайна через Hydra

1. Прописываем в `configs/config.yaml`:

```yaml
defaults:
  - _self_
  # выбираем нужный алгоритм
  - algorithm: rai_stable
  # выбираем нужный датасет
  - benchmark: feedback

algorithm:
  n_jobs: 2

# Общие настройки
output_dir: "results"
output_filename: "results.csv"
output_filename_for_bootstrap: "results_by_bootstraps.csv"

# Сохранить промежуточные файлы и графы
keep_intermediate_files: True
# Продолжить расчет с уже посчитанными графами в algo_results/{benchmark_name}
resume: False

# Настройки bootstrap
bootstrap: true
n_bootstraps: 2
seed: 42

# Настройки визуализации
save_plots: true
plots_dir: "plots"
```

2. Запускаем:

```bash
python main.py
```

Метрики появятся в `output_dir/output_filename`, изображения графов — в `plots_dir`.
Алгоритм и бенчмарк можно переопределить из командной строки:

```bash
python main.py algorithm=pc_stable benchmark=bnlearn
python main.py benchmark=tetrad_discrete algorithm=pc_stable
python main.py benchmark=tetrad_linear algorithm=pc_stable
python main.py algorithm=orig_j_pcmci_plus benchmark=causaltime_traffic
```

`run_all.py` прогоняет все комбинации доступных конфигов, автоматически
пропуская алгоритмы, чей `_target_` не импортируется (например, при
отсутствующем TensorFlow), и бенчмарки из списка `SKIPPED_BENCHMARKS`.

## Данные к бенчмаркам

В поле `benchmark` доступны:

```
tetrad_discrete, tetrad_linear   # по 10 бенчмарков в каждом конфиге
bnlearn                          # alarm, child, insurance, water
bnlearn_child, bnlearn_munin, bnlearn_munin_with_truth
causaltime_traffic               # временные ряды, только orig_j_pcmci_plus
                                 # (traffic, medical, pm25 — может работать очень долго)
feedback, lucas                  # вспомогательные, меньшего размера
```

Датасеты не лежат в репозитории — их нужно скачать и прописать пути в конфигах.

**feedback и lucas:**
https://drive.google.com/file/d/1CrjmgghvCyx0l5RsABnn53Kiyl-LfxuP/view?usp=sharing

Распаковать и указать в `configs/benchmark/feedback.yaml` путь к папке
`example-causal-datasets/simulated/feedbacks/`, а в `configs/benchmark/lucas.yaml`
— путь к `lucas/data/`.

**bnlearn:**
https://drive.google.com/file/d/1t3DUjOpOn7Zztt252mGGHowxC3j2n8x7/view?usp=sharing

Распаковать и прописать путь в `configs/benchmark/bnlearn.yaml`.

**Tetrad:**
https://drive.google.com/file/d/1PEUl6F9y2urkgHGR-YdPDbPe7UsFHmwm/view?usp=sharing

Распаковать и прописать пути в `configs/benchmark/tetrad_discrete.yaml`
и `configs/benchmark/tetrad_linear.yaml`.

**Временные ряды для JPCMCI+ (causaltime_traffic):**
https://drive.google.com/file/d/11ZOd4n_zgtIv8oaG-vM_FoaHLWfi5mvT/view?usp=sharing

Путь прописывается в `configs/benchmark/causaltime_traffic.yaml`.

## Встроенные benchmark-кейсы

В отличие от датасетов выше, эти данные лежат прямо в репозитории
(`whydra_benchmarks/cases`) и доступны через `BenchmarkCaseRepository`:

- **bnlearn:** alarm, andes, asia, barley, cancer, child, earthquake, hailfinder,
  hepar2, insurance, sachs, survey, water, win95pts
- **feedback:** Network1_amp … Network9_cont_amp (18 кейсов)
- **lucas:** lucas0, lucas1, lucas2
- **real:** htru2, pairs, sachs
- **tetrad:** tetrad_discrete_1 … tetrad_discrete_10, tetrad_linear_1 … tetrad_linear_10

## Тесты

```bash
pytest tests/ -q
```

Отдельно — сверка с эталонными реализациями causal-learn на реальных данных
bnlearn (долгая, расхождения задокументированы и намеренно не замаскированы):

```bash
pytest tests/test_vs_causallearn_on_bnlearn.py -q --run-bnlearn
```

Известные расхождения перечислены в `KNOWN_DEVIATIONS` внутри
`tests/test_vs_causallearn.py` и помечены `xfail`. Если расхождение устранено,
тест переходит в XPASS — таблицу нужно обновить.
