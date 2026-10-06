# Whydra

Быстрая библиотека для поиска причинно-следственных связей (causal discovery): реализации PC-stable, FCI-stable и RAI с поддержкой параллельных вычислений.

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
print(graph)  # GeneralGraph из causal-learn: X1 --> X3, X2 --> X3
```

Доступные алгоритмы: `StandalonePCStable`, `StandaloneFCIStable`, `StandaloneRAIStable`, `StandaloneRAIOptimized`. Тесты независимости: `fisherz` (непрерывные данные) и `chisq` (дискретные).

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
pc_stable
fci_stable
rai_stable
```

Для запуска параллельной версии алгоритмов только для pc_stable, fci_stable, rai_stable необходимо заполнить количество параллельных потоков в соответствующем файле алгоритма. При отсутствии переменной или при значении 1 расчет запускается в последовательном режиме.
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
# содержит в себе в configs/benchmark/bnlearn.yaml   
# следующие бенчмарки 
  - alarm
  - child
  - insurance
  - water
```
также есть два дополнительных бенчмарка меньшего размера:
```  
feedback  
lucas
```  
они использовались как вспомогательные во время разработки. 


# Данные к бенчмаркам feedback и lucas:
https://drive.google.com/file/d/1CrjmgghvCyx0l5RsABnn53Kiyl-LfxuP/view?usp=sharing

Их нужно распаковать и прописать в конфиге пути
```
configs/benchmark/feedback.yaml
```
там нужен путь к папке 
```
example-causal-datasets/simulated/feedbacks/  (увидите как в конфиге)
``` 

```
configs/benchmark/lucas.yaml
```
нужен путь к папке 
```
lucas/data/   (увидите как в конфиге)
```

# Данные к бенчмаркам bnlearn:
https://drive.google.com/file/d/1t3DUjOpOn7Zztt252mGGHowxC3j2n8x7/view?usp=sharing

Их нужно распаковать и прописать в конфиге пути
```
configs/benchmark/bnlearn.yaml
```
Запускать алгоритм можно командой:
```commandline
python main.py algorithm=pc_stable benchmark=bnlearn 
```

# Данные к бенчмаркам Tetrad


https://drive.google.com/file/d/1PEUl6F9y2urkgHGR-YdPDbPe7UsFHmwm/view?usp=sharing

Их нужно распаковать и прописать пути в конфигах:
```
configs/benchmark/tetrad_discrete.yaml
configs/benchmark/tetrad_linear.yaml
```
Запускаются командами: 

```commandline
python main.py benchmark=tetrad_discrete algorithm=pc_stable
python main.py benchmark=tetrad_linear algorithm=pc_stable
```
