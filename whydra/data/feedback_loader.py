import os

import numpy as np

from .base import DatasetLoader
from .loader_registry import LoaderRegistry


@LoaderRegistry.register("feedback", "real", "car-evaluation")
class FeedbacksLoader(DatasetLoader):
    """Read and write tabular datasets using configured data and graph paths."""

    def __init__(self, data_dir: str, ground_truth_dir: str):
        # Hydra passes named arguments from the configuration. Keep the path
        # templates with their {name} placeholders for use when loading a case.
        self.data_path_template = data_dir
        self.gt_path_template = ground_truth_dir

    def load_data(self, name: str) -> np.ndarray:
        """Read observations from the configured text file, skipping its header."""
        # Substitute the case name into the configured path template.
        path = self.data_path_template.format(name=name)
        if not os.path.exists(path):
            msg = f"Data file not found for '{name}'. Expected at: {path}"
            raise FileNotFoundError(msg)
        return np.loadtxt(path, skiprows=1)

    def save_data(
        self,
        data: np.ndarray,
        name: str,
        output_dir: str | None = None,
        filename: str | None = None,
    ) -> str:
        """Write observations to a text file and return its path."""
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
        """Return the path to the named dataset’s ground-truth graph."""
        # Substitute the case name into the graph path template.
        path = self.gt_path_template.format(name=name)
        # graph_utils also checks the path. Checking here makes a missing
        # file easier to trace back to the loader.
        if not os.path.exists(path):
            msg = f"Ground truth file not found for '{name}'. Expected at: {path}"
            raise FileNotFoundError(msg)
        return path
