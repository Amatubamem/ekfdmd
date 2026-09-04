from __future__ import annotations

import numpy as np

from ._dmd import _snapshots

try:
    from numba import njit
except ImportError:
    # ponytail: no-Numba installations execute the same small kernel in Python;
    # add a separate backend only if future profiling shows this fallback is inadequate.
    def njit(*args, **kwargs):
        del args, kwargs

        def decorate(function):
            return function

        return decorate


@njit(cache=True)
def _online_core(X: np.ndarray, rho: float, gamma: float, identity: bool) -> tuple[np.ndarray, np.ndarray]:
    samples = np.ascontiguousarray(X.T)
    steps, n = samples.shape
    A = np.eye(n, dtype=X.dtype) if identity else np.zeros((n, n), dtype=X.dtype)
    P = np.eye(n, dtype=X.dtype) * gamma
    operators = np.empty((steps, n, n), dtype=X.dtype)
    predicted = np.empty((steps, n), dtype=X.dtype)
    operators[0], predicted[0] = A, samples[0]
    for k in range(1, steps):
        x, y = samples[k - 1], samples[k]
        predicted[k] = A @ x
        px = P @ x
        gain = px / (rho + np.vdot(x, px))
        A = A + np.outer(y - A @ x, gain.conj())
        P = (P - np.outer(gain, px.conj())) / rho
        P = 0.5 * (P + P.T.conj())
        operators[k] = A
    return operators, predicted.T


class OnlineDMD:
    """Recursive least-squares online DMD with a forgetting factor."""

    def __init__(
        self,
        *,
        rho: float = 1.0,
        initial_covariance: float = 1.0e4,
        initialize_with_identity: bool = True,
    ) -> None:
        if not 0 < rho <= 1:
            raise ValueError("rho must satisfy 0 < rho <= 1.")
        if initial_covariance <= 0:
            raise ValueError("initial_covariance must be positive.")
        self.rho = rho
        self.initial_covariance = initial_covariance
        self.initialize_with_identity = initialize_with_identity

    def fit(self, X: np.ndarray) -> OnlineDMD:
        X = _snapshots(X)
        self.operators, self.predicted_data = _online_core(
            X, self.rho, self.initial_covariance, self.initialize_with_identity
        )
        self.eigs, self.modes = np.linalg.eig(self.operators)
        self.operator = self.operators[-1]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "operator"):
            raise RuntimeError("fit() must be called before predict().")
        return self.operator @ np.asarray(x)
