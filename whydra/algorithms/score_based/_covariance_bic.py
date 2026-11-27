"""Opt-in Gaussian BIC from centered sufficient statistics.

Scores can differ slightly from the reference least-squares calculation.
This can change the order of nearly tied moves. For a node and its candidate
parents, the scorer falls back to least squares when large offsets, poor
conditioning, or cancellation make the covariance calculation unreliable.
"""

import math
from collections import OrderedDict

import numpy as np


class CovarianceBIC:
    def __init__(self, data, *, max_factors=128, max_bytes=8 << 20):
        self.samples = len(data)
        self.scatter = None
        self.factors = OrderedDict()
        self.factor_bytes = 0
        self.max_factors, self.max_bytes = max_factors, max_bytes
        self.fast_scores = self.fallbacks = self.factor_hits = self.factor_misses = 0
        self.log_samples = math.log(self.samples)
        mean = np.mean(data, axis=0)
        centered = data - mean
        scale = np.max(np.abs(centered), axis=0)
        # The intercept in lstsq can change rank under extreme translation.
        # Keep that case on its original numerical path.
        if np.all(np.isfinite(centered)) and np.all(
            np.abs(mean) <= 1e4 * np.maximum(scale, np.finfo(float).tiny)
        ):
            with np.errstate(over="ignore", invalid="ignore"):
                scatter = centered.T @ centered
            if np.isfinite(scatter).all():
                self.scatter = scatter

    def _factor(self, parents):
        if parents in self.factors:
            self.factor_hits += 1
            self.factors.move_to_end(parents)
            return self.factors[parents]
        self.factor_misses += 1
        assert self.scatter is not None
        matrix = self.scatter[np.ix_(parents, parents)]
        factor = None
        try:
            spectrum = np.linalg.eigvalsh(matrix)
            if spectrum[0] > max(spectrum[-1] * 1e-8, 0.0):
                factor = np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            pass
        size = 0 if factor is None else factor.nbytes
        if self.max_factors and size <= self.max_bytes:
            while self.factors and (
                len(self.factors) >= self.max_factors
                or self.factor_bytes + size > self.max_bytes
            ):
                _, old = self.factors.popitem(last=False)
                self.factor_bytes -= 0 if old is None else old.nbytes
            self.factors[parents] = factor
            self.factor_bytes += size
        return factor

    def score(self, child, parents):
        if child in parents:
            msg = "A node cannot be its own parent"
            raise ValueError(msg)
        if self.scatter is None or len(parents) + 1 >= self.samples:
            self.fallbacks += 1
            return None
        total = float(self.scatter[child, child])
        explained = 0.0
        if parents:
            factor = self._factor(parents)
            if factor is None:
                self.fallbacks += 1
                return None
            coefficients = np.linalg.solve(factor, self.scatter[list(parents), child])
            explained = float(coefficients @ coefficients)
        residual = total - explained
        # Guard the subtraction and covariance conditioning. The penalty and
        # absolute RSS floor match the reference; the guard does not rescale it.
        tolerance = (
            256
            * np.finfo(float).eps
            * (self.samples + len(parents) + 1)
            * max(total, explained, 1.0)
        )
        if not math.isfinite(residual) or residual <= tolerance:
            self.fallbacks += 1
            return None
        self.fast_scores += 1
        return float(
            -self.samples * np.log(max(residual, np.finfo(float).eps) / self.samples)
            - (len(parents) + 1) * self.log_samples
        )
