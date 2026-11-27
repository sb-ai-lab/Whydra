import os

import numpy as np
import pandas as pd

from .base import DatasetLoader
from .loader_registry import LoaderRegistry


@LoaderRegistry.register("tetrad")
class TetradLoader(DatasetLoader):
    """Read and write Tetrad data tables and locate their matching graphs."""

    def __init__(self, data_dir: str, ground_truth_dir: str):
        self.data_dir = data_dir
        self.ground_truth_dir = ground_truth_dir

    def load_data(self, name: str) -> np.ndarray:
        """Read the named Tetrad table, skipping the header row."""
        # Names such as "linear_10" or "discrete_5" identify the dataset.
        # For example, the data file for "linear_10" is "data_linear_10.txt".
        path = os.path.join(self.data_dir, f"data_{name}.txt")
        if not os.path.exists(path):
            msg = f"Data file not found: {path}"
            raise FileNotFoundError(msg)

        # Tetrad files usually have a header. Skip it when reading the data.
        try:
            return np.loadtxt(path, skiprows=1)
        except ValueError:
            # If NumPy cannot parse the table, let pandas read whitespace or tab separators.
            return pd.read_csv(path, sep=r"\s+").to_numpy()

    def save_data(
        self,
        data: np.ndarray,
        name: str,
        output_dir: str | None = None,
        filename: str | None = None,
    ) -> str:
        """Write a tab-delimited table with numbered column names and return its path."""
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
        """Return the path to the named dataset’s ground-truth graph."""
        # name: "linear_10" -> id: "10"
        # name: "discrete_5" -> id: "5"
        dataset_id = name.rsplit("_", maxsplit=1)[-1]

        # For example, the graph for size 10 is stored in "graph.10.txt".
        path = os.path.join(self.ground_truth_dir, f"graph.{dataset_id}.txt")
        if not os.path.exists(path):
            msg = f"Ground truth file not found: {path}"
            raise FileNotFoundError(msg)
        return path
