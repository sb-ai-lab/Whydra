import numpy as np
try:
    from line_profiler import profile
except ImportError:  # Profiling is optional at runtime.
    def profile(func):
        return func
from scipy.stats import norm, chi2
from math import log, sqrt


class CIT:
    def __init__(self, data, method='fisherz'):
        self.data = data
        self.method = method
        self.n_samples = data.shape[0]
        if method == 'fisherz':
            self.corr_matrix = np.corrcoef(data.T)
        elif method == 'chisq':
            self.data_int = np.zeros_like(data, dtype=int)
            self.cardinalities = []
            for c in range(data.shape[1]):
                u, inv = np.unique(data[:, c], return_inverse=True)
                self.data_int[:, c] = inv
                self.cardinalities.append(len(u))
            self.cardinalities = np.array(self.cardinalities)

    def __call__(self, X, Y, condition_set=None):
        if condition_set is None:
            condition_set = []
        if self.method == 'fisherz':
            return self.fisher_z(X, Y, condition_set)
        if self.method == 'chisq':
            return self.chi_square(X, Y, condition_set)
        raise NotImplementedError(f"Test {self.method} not implemented locally")

    def fisher_z(self, X, Y, condition_set):
        var = [X, Y] + list(condition_set)
        sub_corr = self.corr_matrix[np.ix_(var, var)]
        try:
            inv = np.linalg.inv(sub_corr)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(sub_corr)
        r = -inv[0, 1] / sqrt(abs(inv[0, 0] * inv[1, 1]))
        cut_at = 0.9999999
        r = min(cut_at, max(-1 * cut_at, r))
        Z = 0.5 * log((1 + r) / (1 - r))
        stat = sqrt(self.n_samples - len(condition_set) - 3) * abs(Z)
        return 2 * (1 - norm.cdf(stat))

    @profile
    def chi_square(self, X, Y, condition_set):
        X_data = self.data_int[:, X]
        Y_data = self.data_int[:, Y]
        if len(condition_set) == 0:
            obs = np.bincount(
                X_data * self.cardinalities[Y] + Y_data,
                minlength=self.cardinalities[X] * self.cardinalities[Y]
            ).reshape(self.cardinalities[X], self.cardinalities[Y])
            row_sum = obs.sum(axis=1)
            col_sum = obs.sum(axis=0)
            expected = np.outer(row_sum, col_sum) / self.n_samples
            valid = expected > 0
            stat = np.sum((obs[valid] - expected[valid]) ** 2 / expected[valid])
            df = (self.cardinalities[X] - 1) * (self.cardinalities[Y] - 1)
            return chi2.sf(stat, df)
        if len(condition_set) == 1:
            Z_data = self.data_int[:, condition_set[0]]
        else:
            Z_cols = self.data_int[:, condition_set]
            dims = self.cardinalities[list(condition_set)]
            Z_data = np.ravel_multi_index(Z_cols.T, dims)
        unique_z = np.unique(Z_data)
        total_stat = 0
        total_df = 0
        for z_val in unique_z:
            mask = Z_data == z_val
            x_sub = X_data[mask]
            y_sub = Y_data[mask]
            n_sub = len(x_sub)
            if n_sub < 2:
                continue
            obs = np.bincount(
                x_sub * self.cardinalities[Y] + y_sub,
                minlength=self.cardinalities[X] * self.cardinalities[Y]
            ).reshape(self.cardinalities[X], self.cardinalities[Y])
            obs = obs[obs.sum(axis=1) > 0, :]
            obs = obs[:, obs.sum(axis=0) > 0]
            if obs.shape[0] < 2 or obs.shape[1] < 2:
                continue
            row_sum = obs.sum(axis=1)
            col_sum = obs.sum(axis=0)
            expected = np.outer(row_sum, col_sum) / n_sub
            valid = expected > 0
            if valid.sum() > 0:
                stat = np.sum((obs[valid] - expected[valid]) ** 2 / expected[valid])
                df = (obs.shape[0] - 1) * (obs.shape[1] - 1)
                total_stat += stat
                total_df += df
        if total_df == 0:
            return 1.0
        return chi2.sf(total_stat, total_df)


__all__ = ["CIT"]
