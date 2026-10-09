import os
import numpy as np
from typing import Optional
from .base import DatasetLoader
from .loader_registry import LoaderRegistry


@LoaderRegistry.register("feedback", "real", "car-evaluation")
class FeedbacksLoader(DatasetLoader):
    def __init__(self, data_dir: str, ground_truth_dir: str):
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
        # load_data skips row 1, so we write a header
        header = " ".join([f"X{i}" for i in range(data.shape[1])])
        np.savetxt(path, data, header=header, comments="")
        return path

    def load_ground_truth_path(self, name: str) -> str:
        # Используем .format() для подстановки имени в шаблон
        path = self.gt_path_template.format(name=name)
        # Проверка существования здесь не обязательна (она есть в graph_utils),
        # но полезна для отладки
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ground truth file not found for '{name}'. Expected at: {path}")
        return path
    