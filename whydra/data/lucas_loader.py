import os

import numpy as np
import pandas as pd

from .base import DatasetLoader
from .loader_registry import LoaderRegistry


@LoaderRegistry.register("lucas")
class LucasLoader(DatasetLoader):
    """Load and save the separate target and feature tables of a LUCAS dataset."""

    def __init__(self, data_dir: str, targets_dir: str, ground_truth_dir: str):
        self.data_path_template = data_dir
        self.targets_path_template = targets_dir
        self.gt_path_template = ground_truth_dir

    def load_data(self, name: str) -> np.ndarray:
        """Read the target and feature tables, placing the target columns first."""
        features_path = self.data_path_template.format(name=name)
        targets_path = self.targets_path_template.format(name=name)
        if not os.path.exists(features_path):
            msg = f"Data file not found: {features_path}"
            raise FileNotFoundError(msg)
        if not os.path.exists(targets_path):
            msg = f"Targets file not found: {targets_path}"
            raise FileNotFoundError(msg)

        # Read the whitespace-delimited features and target.
        features = pd.read_csv(features_path, header=None, sep=r"\s+")
        targets = pd.read_csv(targets_path, header=None, sep=r"\s+")

        # LUCAS datasets may list the target first or last. Put it in column 0
        # (X1), followed by the features X2–X12, to match the ground-truth
        # graph used by causal-learn.
        return pd.concat([targets, features], axis=1).to_numpy()

    def save_data(
        self,
        data: np.ndarray,
        name: str,
        output_dir: str | None = None,
        filename: str | None = None,
    ) -> str:
        """Write the first column as targets and the rest as features; return the feature path."""
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
        """Return the path to the named dataset’s ground-truth graph."""
        path = self.gt_path_template.format(name=name)
        if not os.path.exists(path):
            msg = f"Ground truth file not found: {path}"
            raise FileNotFoundError(msg)
        return path
