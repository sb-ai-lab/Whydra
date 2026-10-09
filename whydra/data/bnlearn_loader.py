import os
import numpy as np
import pandas as pd
from typing import Optional
from .base import DatasetLoader
from .loader_registry import LoaderRegistry


@LoaderRegistry.register("bnlearn")
class BnLearnLoader(DatasetLoader):
    def __init__(self, data_dir, ground_truth_dir):
        self.data_path_template = data_dir
        self.gt_path_template = ground_truth_dir

    def load_data(self, name: str) -> np.ndarray:
        path = self.data_path_template.format(name=name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        # Загружаем данные. Обычно в bnlearn файлах есть заголовок (X1, X2...),
        # поэтому используем pandas для надежности, затем конвертируем в numpy.
        # sep=r'\s+' обрабатывает любое количество пробелов/табов.
        return pd.read_csv(path, sep=r"\s+").to_numpy()

    def save_data(self,
                  data: np.ndarray,
                  name: str,
                  output_dir: Optional[str] = None,
                  filename: Optional[str] = None
                  ) -> str:
        if output_dir is None:
            path = self.data_path_template.format(name=name)
        else:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, filename or f"{name}.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        columns = [f"X{i + 1}" for i in range(data.shape[1])]
        pd.DataFrame(data, columns=columns).to_csv(path, sep=" ", index=False)
        return path

    def load_ground_truth_path(self, name: str) -> str:
        path = self.gt_path_template.format(name=name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ground truth file not found: {path}")
        return path
