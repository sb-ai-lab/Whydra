import os
import numpy as np
import pandas as pd
from typing import Optional
from .base import DatasetLoader
from .loader_registry import LoaderRegistry


@LoaderRegistry.register("lucas")
class LucasLoader(DatasetLoader):
    def __init__(self, data_dir: str, targets_dir: str, ground_truth_dir: str):
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
        features = pd.read_csv(features_path, header=None, sep=r"\s+")
        targets = pd.read_csv(targets_path, header=None, sep=r"\s+")
        
        # Объединяем: целевая переменная (X1) + остальные признаки (X2-X12)
        # В LUCAS таргет обычно идет первым или последним,
        # но для causal-learn важен порядок колонок, соответствующий ground truth.
        # В старой библиотеке было concat([targets, features]), значит таргет - это X0 (или X1 в нотации графа)
        return pd.concat([targets, features], axis=1).to_numpy()

    def save_data(self,
                  data: np.ndarray,
                  name: str,
                  output_dir: Optional[str] = None,
                  filename: Optional[str] = None
                  ) -> str:
        if output_dir is None:
            features_path = self.data_path_template.format(name=name)
            targets_path = self.targets_path_template.format(name=name)
        else:
            os.makedirs(output_dir, exist_ok=True)
            base = filename or name
            features_path = os.path.join(output_dir, f"{base}_features.txt")
            targets_path = os.path.join(output_dir, f"{base}_targets.txt")
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        os.makedirs(os.path.dirname(targets_path), exist_ok=True)
        
        # Assuming target is the first column based on load_data concat order
        targets = data[:, [0]]
        features = data[:, 1:]

        # Save with space separator, no header as per load_data
        pd.DataFrame(targets).to_csv(targets_path, header=False, index=False, sep=" ")
        pd.DataFrame(features).to_csv(features_path, header=False, index=False, sep=" ")
        return features_path

    def load_ground_truth_path(self, name: str) -> str:
        path = self.gt_path_template.format(name=name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ground truth file not found: {path}")
        return path

