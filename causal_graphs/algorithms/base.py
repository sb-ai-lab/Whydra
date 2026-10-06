from abc import ABC, abstractmethod
import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph

class CausalAlgorithm(ABC):
    @abstractmethod
    def run(self, data: np.ndarray, **kwargs) -> GeneralGraph:
        """
        Принимает данные (n_samples, n_features) и возвращает GeneralGraph.
        """
        pass