"""Reference learner for generator acceptance tests (T-3, T-4) and smoke-test baselines.

This deterministic ridge T-learner uses only NumPy. The generator's internal acceptance
tests use it to check that changing pathology settings really changes the problem's
difficulty. It is separate from the methods being benchmarked.
"""

import numpy as np


def _ridge_fit(
    x: np.ndarray, y: np.ndarray, alpha: float, sample_weight: np.ndarray | None = None
) -> np.ndarray:
    xb = np.column_stack([np.ones(len(x)), x])
    if sample_weight is not None:
        sw = np.sqrt(sample_weight)[:, None]
        xb, y = xb * sw, y * sw[:, 0]
    reg = alpha * np.eye(xb.shape[1])
    reg[0, 0] = 0.0  # Leave the intercept unpenalized.
    return np.linalg.solve(xb.T @ xb + reg, xb.T @ y)


def _ridge_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x]) @ w


def t_learner_ridge(
    x_train: np.ndarray,
    t_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float = 1.0,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate treatment effects with one ridge regression per treatment arm.

    If supplied, ``sample_weight`` gives one weight per training row (contract
    P4). For each test row, return the treated prediction minus the control
    prediction.
    """
    m1, m0 = t_train == 1, t_train == 0
    sw1 = sample_weight[m1] if sample_weight is not None else None
    sw0 = sample_weight[m0] if sample_weight is not None else None
    w1 = _ridge_fit(x_train[m1], y_train[m1], alpha, sw1)
    w0 = _ridge_fit(x_train[m0], y_train[m0], alpha, sw0)
    return _ridge_predict(x_test, w1) - _ridge_predict(x_test, w0)


def segment_dim(t: np.ndarray, y: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Estimate each row's effect by its segment's difference in means (RCT).

    Within each segment, subtract the mean control outcome from the mean
    treated outcome. Assign zero if either arm is absent. Each estimate uses
    only its segment's rows, so an error in one segment cannot affect another.
    """
    tau_hat = np.empty(y.size)
    for s in np.unique(segments):
        m = segments == s
        t1, t0 = m & (t == 1), m & (t == 0)
        dim = y[t1].mean() - y[t0].mean() if t1.any() and t0.any() else 0.0
        tau_hat[m] = dim
    return tau_hat


def pehe(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """Compute sqrt(PEHE), the root mean squared error of the estimated ITE."""
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))
