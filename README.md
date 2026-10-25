# Whydra

Быстрая библиотека для поиска причинно-следственных связей (causal discovery): реализации PC-stable, семейства FCI (FCI-stable, CFCI, RFCI, GFCI, FCI+) и RAI с поддержкой параллельных вычислений и фоновых знаний (background knowledge).

## Установка

```bash
pip install whydra
```

Требуется Python 3.10+.

Для сохранения изображений графов (`whydra.evaluation.graph_utils.draw_graph`) в системе должен быть установлен Graphviz:
* **Linux:** `sudo apt-get install graphviz`
* **Windows:** Скачайте инсталлятор с [сайта Graphviz](https://graphviz.org/download/), установите и **добавьте путь к bin в PATH**.

## Быстрый старт

```python
import numpy as np
from whydra import StandalonePCStable

rng = np.random.default_rng(0)
x1, x2 = rng.normal(size=(2, 1000))
x3 = x1 + x2 + rng.normal(scale=0.5, size=1000)
data = np.column_stack([x1, x2, x3])  # (n_samples, n_features)

graph = StandalonePCStable(alpha=0.05, indep_test="fisherz", n_jobs=4).run(data)
print([str(edge) for edge in graph.get_graph_edges()])  # ['X1 --> X3', 'X2 --> X3']
```

Доступные алгоритмы: `StandalonePCStable`, `ParallelPCStable`, `StandaloneFCIStable` (и варианты с кэшем CI-тестов `StandaloneFCIStableCached`, `StandaloneFCIStableParallelCached`), `StandaloneCFCIStable`, `StandaloneRFCIStable`, `StandaloneGFCIStable`, `StandaloneFCIPlusStable`, `StandaloneRAIStable`, `StandaloneRAIOptimized`. Алгоритм можно создать и по имени: `whydra.create_standalone_algorithm("rfci_stable", alpha=0.01)`. Тесты независимости: `fisherz` (непрерывные данные) и `chisq` (дискретные).

## Лицензия

Whydra распространяется по лицензии [Apache License 2.0](https://github.com/sb-ai-lab/Whydra/blob/main/LICENSE).

# Бенчмарки

Для запуска бенчмарков склонируйте репозиторий и установите пакет вместе с зависимостями для бенчмарков (Hydra и др.):
```bash
git clone https://github.com/sb-ai-lab/Whydra.git
cd Whydra
pip install -e ".[bench]"
```
Точные версии библиотек, на которых проводились замеры, зафиксированы в `requirements.txt` (`pip install -r requirements.txt`).

##  Как запускать
1. Прописываем в configs/config.yaml
```yaml
defaults:
  - _self_
  # выбираем нужный алгоритм 
  - algorithm: rai_stable
  # выбираем нужные датасет
  - benchmark: feedback

# Общие настройки
output_dir: "results"
output_filename: "results.csv"
output_filename_for_bootstrap: "results_by_bootstraps.csv"

# Сохранить промежуточные файлы и графы
keep_intermediate_files: False
# Продолжить расчет с уже посчитанными графами в algo_results/{benchmark_name}
resume: False

# включаем бутстрап 
bootstrap: True
# количество итераций бутстрапа
n_bootstraps: 1
# сид для бутстрапинга
seed: 42

# Настройки визуализации
save_plots: true
plots_dir: "plots"
```

2. После этого запускаем 

```bash
python main.py
```
После выполнения скрипта в папке output_filename появятся метрики в файле output_filename.
А сами изображения графов появятся в plots_dir

В поле **algorithm** можно выбрать алгоритм:
```
pc_stable, pc_stable_par, pc_stable_seq, pc_stable_4/8/16
fci_stable, fci_stable_16, fci_stable_cash, fci_stable_paralel_cash
cfci_stable, cfci_stable_paralel_cash
rfci_stable
gfci_stable
fci_plus_stable, fci_plus_stable_paralel_cash
rai_stable, rai_optimized
```

Для запуска параллельной версии алгоритмов необходимо заполнить количество параллельных потоков в соответствующем файле алгоритма. При отсутствии переменной или при значении 1 расчет запускается в последовательном режиме.
```
n_jobs: 6
```

В поле benchmark можно выбрать следующие бенчмарки: 

```
tetrad_discrete
tetrad_linear
# оба содержат в себе по 10 бенчмарков
# в configs/benchmark/tetrad_discrete.yaml
# и в configs/benchmark/tetrad_linear.yaml

bnlearn
# дискретные сети bnlearn: alarm, andes, asia, barley, cancer, child, earthquake,
# hailfinder, hepar2, insurance, sachs, survey, water, win95pts
```
также есть два дополнительных бенчмарка меньшего размера:
```  
feedback  
lucas
```  
они использовались как вспомогательные во время разработки. 


# Данные к бенчмаркам

Датасеты в репозиторий не входят. `main.py` прогоняет алгоритм на всех кейсах,
которые найдёт в папке `whydra_benchmarks/cases` (относительно папки запуска).
Каждый кейс раскладывается так:
```
whydra_benchmarks/cases/<benchmark>/<case>/input/<case>.txt               # данные
whydra_benchmarks/cases/<benchmark>/<case>/ground.truth/<case>.graph.txt  # эталонный граф
```
Например, `whydra_benchmarks/cases/bnlearn/child/input/child.txt`.

Архивы с данными:
* feedback и lucas: https://drive.google.com/file/d/1CrjmgghvCyx0l5RsABnn53Kiyl-LfxuP/view?usp=sharing
* bnlearn: https://drive.google.com/file/d/1t3DUjOpOn7Zztt252mGGHowxC3j2n8x7/view?usp=sharing
* Tetrad: https://drive.google.com/file/d/1PEUl6F9y2urkgHGR-YdPDbPe7UsFHmwm/view?usp=sharing

Запуск:
```commandline
python main.py algorithm=pc_stable benchmark=bnlearn
python main.py algorithm=rfci_stable benchmark=bnlearn
```
Из конфига бенчмарка `main.py` берёт тест независимости (`indep_test`: `chisq` для
дискретных данных, `fisherz` для непрерывных).

`run_bootstrap_consensus.py` строит устойчивый граф по бутстрэп-выборкам одного кейса;
он читает данные через `loader` из конфига бенчмарка (пути в `configs/benchmark/*.yaml`).
