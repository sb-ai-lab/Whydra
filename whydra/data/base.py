from abc import ABC, abstractmethod

import numpy as np


class DatasetLoader(ABC):
    """Common interface for loading observations and locating their ground-truth graph."""

    @abstractmethod
    def load_data(self, name: str) -> np.ndarray:
        """Load observations for the named dataset into an array."""
        raise NotImplementedError

    @abstractmethod
    def load_ground_truth_path(self, name: str) -> str:
        """Return the path to the named dataset’s ground-truth graph."""
        raise NotImplementedError

    def save_data(
        self,
        data: np.ndarray,
        name: str,
        output_dir: str | None = None,
        filename: str | None = None,
    ) -> str:
        """Write observations and return their path when saving is supported."""
        msg = f"{self.__class__.__name__} does not implement save_data"
        raise NotImplementedError(msg)
