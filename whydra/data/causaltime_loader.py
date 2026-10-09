import os
import numpy as np
from typing import Optional
from .base import DatasetLoader
from .loader_registry import LoaderRegistry


@LoaderRegistry.register("causaltime")
class CausalTimeLoader(DatasetLoader):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_data(self, name: str) -> np.ndarray:
        # name: например "traffic", "medical", "pm25"
        # Файл: data_dir/name/gen_data.npy
        path = os.path.join(self.data_dir, name, "gen_data.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        # Методы для временных рядов обычно ожидают (Time, Node) или список таких массивов.
        # Наш интерфейс pipeline ожидает np.ndarray.
        # Мы вернем 3D массив как есть, а Wrapper алгоритма сам разберется, как его "съесть".
        
        # Загружаем (Sample, Time, Node)
        return np.load(path)

    def save_data(self,
                  data: np.ndarray,
                  name: str,
                  output_dir: Optional[str] = None,
                  filename: Optional[str] = None
                  ) -> str:
        if output_dir is None:
            path = os.path.join(self.data_dir, name, "gen_data.npy")
        else:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, filename or f"{name}.npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, data)
        return path

    def load_ground_truth_path(self, name: str) -> str:
        # CausalTime хранит граф в .npy, а не в .txt, как наши предыдущие лоадеры.
        # Нам придется немного схитрить: вернуть путь к .npy,
        # а функцию загрузки графа (load_ground_truth в graph_utils) научить читать .npy.
        path = os.path.join(self.data_dir, name, "graph.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ground truth file not found: {path}")
        return path

