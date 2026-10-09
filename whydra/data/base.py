from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class DatasetLoader(ABC):
    @abstractmethod
    def load_data(self, name: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def load_ground_truth_path(self, name: str) -> str:
        raise NotImplementedError

    def save_data(
        self,
        data: np.ndarray,
        name: str,
        output_dir: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> str:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement save_data")