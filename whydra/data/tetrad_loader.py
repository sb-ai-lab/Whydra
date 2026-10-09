import os
import numpy as np
import pandas as pd
from typing import Optional
from .base import DatasetLoader
from .loader_registry import LoaderRegistry


@LoaderRegistry.register("tetrad")
class TetradLoader(DatasetLoader):
    def __init__(self, data_dir: str, ground_truth_dir: str):
        self.data_dir = data_dir
        self.ground_truth_dir = ground_truth_dir

    def load_data(self, name: str) -> np.ndarray:
        # name будет, например, "linear_10" или "discrete_5"
        # Файл данных называется "data_linear_10.txt"
        path = os.path.join(self.data_dir, f"data_{name}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        
        # В файлах Tetrad обычно есть заголовок, пропускаем его
        try:
            return np.loadtxt(path, skiprows=1)
        except ValueError:
            # Если разделитель не пробел, а таб, или есть проблемы с форматом
            return pd.read_csv(path, sep=r"\s+").to_numpy()

    def save_data(self,
                  data: np.ndarray,
                  name: str,
                  output_dir: Optional[str] = None,
                  filename: Optional[str] = None
                  ) -> str:
        if output_dir is None:
            path = os.path.join(self.data_dir, f"data_{name}.txt")
        else:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, filename or f"data_{name}.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # load_data skips row 1, so we write a header
        header = "\t".join([f"X{i + 1}" for i in range(data.shape[1])])
        np.savetxt(path, data, header=header, comments="", delimiter="\t")
        return path

    def load_ground_truth_path(self, name: str) -> str:
        # name: "linear_10" -> id: "10"
        # name: "discrete_5" -> id: "5"
        dataset_id = name.split("_")[-1]
        
        # Файл графа называется "graph.10.txt"
        path = os.path.join(self.ground_truth_dir, f"graph.{dataset_id}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ground truth file not found: {path}")
        return path
