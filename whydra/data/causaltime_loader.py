import os

import numpy as np

from .base import DatasetLoader
from .loader_registry import LoaderRegistry


@LoaderRegistry.register("causaltime")
class CausalTimeLoader(DatasetLoader):
    """Read and write CausalTime arrays with sample, time and node dimensions."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_data(self, name: str) -> np.ndarray:
        """Read the named dataset’s gen_data.npy array."""
        # Dataset names include "traffic", "medical", and "pm25". Each uses
        # data_dir/name/gen_data.npy.
        path = os.path.join(self.data_dir, name, "gen_data.npy")
        if not os.path.exists(path):
            msg = f"Data file not found: {path}"
            raise FileNotFoundError(msg)
        # The pipeline expects a NumPy array. Keep all three dimensions so the
        # algorithm wrapper can interpret them.

        # Read the axes in (sample, time, node) order.
        return np.load(path)

    def save_data(
        self,
        data: np.ndarray,
        name: str,
        output_dir: str | None = None,
        filename: str | None = None,
    ) -> str:
        """Save the NumPy data array and return the configured output path."""
        if output_dir is None:
            path = os.path.join(self.data_dir, name, "gen_data.npy")
        else:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, filename or f"{name}.npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, data)
        return path

    def load_ground_truth_path(self, name: str) -> str:
        """Return the path to the named dataset’s ground-truth graph."""
        # CausalTime stores the graph in a NumPy file. graph_utils.load_ground_truth
        # selects its NumPy or text reader from the path extension.
        path = os.path.join(self.data_dir, name, "graph.npy")
        if not os.path.exists(path):
            msg = f"Ground truth file not found: {path}"
            raise FileNotFoundError(msg)
        return path
