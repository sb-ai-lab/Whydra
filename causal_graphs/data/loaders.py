import numpy as np
import os
from .base import DatasetLoader
import pandas as pd


class FeedbacksLoader(DatasetLoader):
    def __init__(self, data_dir, ground_truth_dir):
        # Hydra передает аргументы по именам из конфига.
        # В конфиге у нас шаблоны (строки с {name}), сохраняем их.
        self.data_path_template = data_dir
        self.gt_path_template = ground_truth_dir

    def load_data(self, name: str) -> np.ndarray:
        # Используем .format() для подстановки имени в шаблон
        path = self.data_path_template.format(name=name)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found for '{name}'. Expected at: {path}")

        return np.loadtxt(path, skiprows=1)

    def save_data(self, data: np.ndarray, name: str):
        path = self.data_path_template.format(name=name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # load_data skips row 1, so we write a header
        header = " ".join([f"X{i}" for i in range(data.shape[1])])
        np.savetxt(path, data, header=header, comments='')

    def load_ground_truth_path(self, name: str) -> str:
        # Используем .format() для подстановки имени в шаблон
        path = self.gt_path_template.format(name=name)

        # Проверка существования здесь не обязательна (она есть в graph_utils),
        # но полезна для отладки
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ground truth file not found for '{name}'. Expected at: {path}")

        return path


class LucasLoader(DatasetLoader):
    def __init__(self, data_dir, targets_dir, ground_truth_dir):
        self.data_path_template = data_dir
        self.targets_path_template = targets_dir
        self.gt_path_template = ground_truth_dir

    def load_data(self, name: str) -> np.ndarray:
        features_path = self.data_path_template.format(name=name)
        targets_path = self.targets_path_template.format(name=name)

        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Data file not found: {features_path}")
        if not os.path.exists(targets_path):
            raise FileNotFoundError(f"Targets file not found: {targets_path}")

        # Загружаем признаки и целевую переменную (разделитель - пробелы)
        features = pd.read_csv(features_path, header=None, sep=r'\s+')
        targets = pd.read_csv(targets_path, header=None, sep=r'\s+')

        # Объединяем: целевая переменная (X1) + остальные признаки (X2-X12)
        # В LUCAS таргет обычно идет первым или последним,
        # но для causal-learn важен порядок колонок, соответствующий ground truth.
        # В старой библиотеке было concat([targets, features]), значит таргет - это X0 (или X1 в нотации графа)
        full_data = pd.concat([targets, features], axis=1)

        return full_data.to_numpy()

    def save_data(self, data: np.ndarray, name: str):
        features_path = self.data_path_template.format(name=name)
        targets_path = self.targets_path_template.format(name=name)

        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        os.makedirs(os.path.dirname(targets_path), exist_ok=True)

        # Assuming target is the first column based on load_data concat order
        targets = data[:, [0]]
        features = data[:, 1:]

        # Save with space separator, no header as per load_data
        pd.DataFrame(targets).to_csv(targets_path, header=False, index=False, sep=' ')
        pd.DataFrame(features).to_csv(features_path, header=False, index=False, sep=' ')

    def load_ground_truth_path(self, name: str) -> str:
        path = self.gt_path_template.format(name=name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ground truth file not found: {path}")
        return path


class BnLearnLoader(DatasetLoader):
    def __init__(self, data_dir, ground_truth_dir):
        self.data_path_template = data_dir
        self.gt_path_template = ground_truth_dir

    def load_data(self, name: str) -> np.ndarray:
        # Формируем путь
        path = self.data_path_template.format(name=name)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        # Загружаем данные. Обычно в bnlearn файлах есть заголовок (X1, X2...),
        # поэтому используем pandas для надежности, затем конвертируем в numpy.
        # sep=r'\s+' обрабатывает любое количество пробелов/табов.
        df = pd.read_csv(path, sep=r'\s+')

        return df.to_numpy()

    def save_data(self, data: np.ndarray, path: str, name: str):
        # path = self.data_path_template.format(name=name)
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        filename = os.path.join(path, name)
        columns = [f"X{i + 1}" for i in range(data.shape[1])]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(filename, sep=' ', index=False)

    def load_ground_truth_path(self, name: str) -> str:
        path = self.gt_path_template.format(name=name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ground truth file not found: {path}")
        return path


class TetradLoader(DatasetLoader):
    def __init__(self, data_dir, ground_truth_dir):
        self.data_dir = data_dir
        self.ground_truth_dir = ground_truth_dir

    def load_data(self, name: str) -> np.ndarray:
        # name будет, например, "linear_10" или "discrete_5"
        # Файл данных называется "data_linear_10.txt"
        filename = f"data_{name}.txt"
        path = os.path.join(self.data_dir, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        # В файлах Tetrad обычно есть заголовок, пропускаем его
        try:
            return np.loadtxt(path, skiprows=1)
        except ValueError:
            # Если разделитель не пробел, а таб, или есть проблемы с форматом
            import pandas as pd
            df = pd.read_csv(path, sep=r'\s+')
            return df.to_numpy()

    def save_data(self, data: np.ndarray, name: str):
        filename = f"data_{name}.txt"
        path = os.path.join(self.data_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # load_data skips row 1, so we write a header
        header = "\t".join([f"X{i + 1}" for i in range(data.shape[1])])
        np.savetxt(path, data, header=header, comments='', delimiter='\t')

    def load_ground_truth_path(self, name: str) -> str:
        # name: "linear_10" -> id: "10"
        # name: "discrete_5" -> id: "5"
        dataset_id = name.split('_')[-1]

        # Файл графа называется "graph.10.txt"
        filename = f"graph.{dataset_id}.txt"
        path = os.path.join(self.ground_truth_dir, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Ground truth file not found: {path}")

        return path


class CausalTimeLoader(DatasetLoader):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_data(self, name: str) -> np.ndarray:
        # name: например "traffic", "medical", "pm25"
        # Файл: data_dir/name/gen_data.npy
        path = os.path.join(self.data_dir, name, "gen_data.npy")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        # Загружаем (Sample, Time, Node)
        data = np.load(path, allow_pickle=False)

        # Методы анализа временных рядов обычно ожидают (Time, Node)
        # или список таких массивов.
        # Наш интерфейс pipeline ожидает np.ndarray.
        # Мы вернем 3D массив как есть, а Wrapper алгоритма сам разберется, как его "съесть".
        return data

    def save_data(self, data: np.ndarray, name: str):
        path = os.path.join(self.data_dir, name, "gen_data.npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, data)

    def load_ground_truth_path(self, name: str) -> str:
        # CausalTime хранит граф в .npy, а не в .txt, как наши предыдущие лоадеры.
        # Нам придется немного схитрить: вернуть путь к .npy,
        # а функцию загрузки графа (load_ground_truth в graph_utils) научить читать .npy.
        path = os.path.join(self.data_dir, name, "graph.npy")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Ground truth file not found: {path}")

        return path
