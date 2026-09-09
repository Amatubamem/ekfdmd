from __future__ import annotations

import numpy as np

from ._dmd import _basis, _snapshots

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


@njit(cache=True)
def _recursive_dmdc_core(
    X: np.ndarray,
    U: np.ndarray,
    rho: float,
    gamma: float,
    identity: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = np.ascontiguousarray(X.T)
    controls = np.ascontiguousarray(U.T)
    steps, n = samples.shape
    m = controls.shape[1]
    gain_matrix = np.zeros((n, n + m), dtype=X.dtype)
    if identity:
        gain_matrix[:, :n] = np.eye(n, dtype=X.dtype)
    P = np.eye(n + m, dtype=X.dtype) * gamma
    operators = np.empty((steps, n, n), dtype=X.dtype)
    control_operators = np.empty((steps, n, m), dtype=X.dtype)
    predicted = np.empty((steps, n), dtype=X.dtype)
    operators[0] = gain_matrix[:, :n]
    control_operators[0] = gain_matrix[:, n:]
    predicted[0] = samples[0]
    for k in range(1, steps):
        z = np.concatenate((samples[k - 1], controls[k - 1]))
        y = samples[k]
        prediction = gain_matrix @ z
        predicted[k] = prediction
        pz = P @ z
        gain = pz / (rho + np.vdot(z, pz))
        gain_matrix = gain_matrix + np.outer(y - prediction, gain.conj())
        P = (P - np.outer(gain, pz.conj())) / rho
        P = 0.5 * (P + P.T.conj())
        operators[k] = gain_matrix[:, :n]
        control_operators[k] = gain_matrix[:, n:]
    return operators, control_operators, predicted.T


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


class RecursiveDMDc:
    """Recursive least-squares DMD with a known control input."""

    def __init__(
        self,
        *,
        rho: float = 1.0,
        initial_covariance: float = 1.0e4,
        rank: int | None = None,
        initialize_with_identity: bool = True,
    ) -> None:
        if not 0 < rho <= 1:
            raise ValueError("rho must satisfy 0 < rho <= 1.")
        if initial_covariance <= 0:
            raise ValueError("initial_covariance must be positive.")
        self.rho = rho
        self.initial_covariance = initial_covariance
        self.rank = rank
        self.initialize_with_identity = initialize_with_identity

    def fit(self, X: np.ndarray, U: np.ndarray) -> RecursiveDMDc:
        X = _snapshots(X)
        U = _snapshots(U, "U")
        if X.shape[1] != U.shape[1]:
            raise ValueError("X and U must contain the same number of snapshots.")
        projection = _basis(X, self.rank)
        reduced = projection.conj().T @ X
        operators, controls, predicted = _recursive_dmdc_core(
            reduced,
            U.astype(reduced.dtype, copy=False),
            self.rho,
            self.initial_covariance,
            self.initialize_with_identity,
        )
        self.operators = np.einsum("ij,tjk,lk->til", projection, operators, projection.conj())
        self.control_operators = np.einsum("ij,tjk->tik", projection, controls)
        self.predicted_data = projection @ predicted
        self.eigs, reduced_modes = np.linalg.eig(operators)
        self.modes = np.einsum("ij,tjk->tik", projection, reduced_modes)
        self.operator = self.operators[-1]
        self.control_operator = self.control_operators[-1]
        return self

    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        if not hasattr(self, "operator"):
            raise RuntimeError("fit() must be called before predict().")
        x, u = np.asarray(x), np.asarray(u)
        if (
            x.ndim not in (1, 2)
            or u.ndim not in (1, 2)
            or x.shape[0] != self.operator.shape[1]
            or u.shape[0] != self.control_operator.shape[1]
        ):
            raise ValueError("x or u has an incompatible dimension.")
        return self.operator @ x + self.control_operator @ u

    @property
    def effective_window(self) -> float:
        return float("inf") if self.rho == 1.0 else 1.0 / (1.0 - self.rho)
