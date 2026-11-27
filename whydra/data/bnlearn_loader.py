import os

import numpy as np
import pandas as pd

from .base import DatasetLoader
from .loader_registry import LoaderRegistry


@LoaderRegistry.register("bnlearn")
class BnLearnLoader(DatasetLoader):
    """Load and save whitespace-delimited bnlearn tables."""

    def __init__(self, data_dir, ground_truth_dir):
        self.data_path_template = data_dir
        self.gt_path_template = ground_truth_dir

    def load_data(self, name: str) -> np.ndarray:
        """Read the whitespace-delimited table, using its first row as column names."""
        path = self.data_path_template.format(name=name)
        if not os.path.exists(path):
            msg = f"Data file not found: {path}"
            raise FileNotFoundError(msg)

        # bnlearn files usually include a header (X1, X2, ...). Read it
        # with pandas before converting to NumPy. The whitespace separator
        # accepts any run of spaces or tabs.
        return pd.read_csv(path, sep=r"\s+").to_numpy()

    def save_data(
        self,
        data: np.ndarray,
        name: str,
        output_dir: str | None = None,
        filename: str | None = None,
    ) -> str:
        """Write a space-delimited table with numbered column names and return its path."""
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
        """Return the path to the named dataset’s ground-truth graph."""
        path = self.gt_path_template.format(name=name)
        if not os.path.exists(path):
            msg = f"Ground truth file not found: {path}"
            raise FileNotFoundError(msg)
        return path
