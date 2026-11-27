from abc import ABC, abstractmethod

import numpy as np

from .graph_core import GeneralGraph, _optional_imports


class CausalAlgorithm(ABC):
    """
    Abstract base class for causal structure discovery algorithms.

    Each implementation must provide a ``run`` method that accepts an
    observation matrix and returns the estimated causal graph as the
    library's own ``GeneralGraph`` type.

    The PC, RFCI, RAI, and FCI wrappers share several options. Check the
    individual wrapper for supported options and additional parameters.

    ``alpha``
        Significance level for the conditional independence test.
    ``indep_test``
        Name of the CI test. See ``graph_core.IMPLEMENTED_TESTS``.
    ``n_jobs``
        Number of parallel jobs. Use ``1`` for sequential execution.
    ``verbose``
        Enable diagnostic output from the algorithm.
    ``show_progress``
        Show a progress bar during skeleton search. Off by default so the
        library writes to standard error only when requested.
    ``background_knowledge``
        Optional graph constraints for algorithms that support them. RAI
        does not support background knowledge.
    """

    @abstractmethod
    def run(
        self, data: "np.ndarray | _optional_imports.DataFrame", **kwargs
    ) -> GeneralGraph:
        """
        Estimate a causal graph from rows of observations.

        Parameters
        ----------
        data : numpy.ndarray or pandas.DataFrame
            Observation matrix of shape ``(n_samples, n_features)``, where
            ``n_samples`` counts observations and ``n_features`` counts variables.
        **kwargs
            Additional parameters for the chosen algorithm.

        Returns
        -------
        GeneralGraph
            Estimated causal graph in the ``whydra.algorithms.graph_core`` format.

        Notes
        -----
        Concrete subclasses must implement this method.
        """
