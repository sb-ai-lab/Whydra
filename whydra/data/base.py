from abc import ABC, abstractmethod
import numpy as np

class DatasetLoader(ABC):
    @abstractmethod
    def load_data(self, name: str) -> np.ndarray:
        """
        Загружает данные для конкретного бенчмарка.
        Returns:
            np.ndarray: Массив данных (n_samples, n_features)
        """
        pass

    @abstractmethod
    def load_ground_truth_path(self, name: str) -> str:
        """
        Возвращает путь к файлу с истинным графом.
        """
        pass