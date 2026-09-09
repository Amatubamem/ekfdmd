from __future__ import annotations

import numpy as np

from ._dmd import _basis, _snapshots

try:
    from numba import njit
except ImportError:
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
    gain_matrix: np.ndarray,
    covariance: np.ndarray,
    start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = np.ascontiguousarray(X.T)
    controls = np.ascontiguousarray(U.T)
    steps, n = samples.shape
    m = controls.shape[1]
    P = covariance.copy()
    operators = np.empty((steps, n, n), dtype=X.dtype)
    control_operators = np.empty((steps, n, m), dtype=X.dtype)
    predicted = samples.copy()
    for k in range(start):
        operators[k] = gain_matrix[:, :n]
        control_operators[k] = gain_matrix[:, n:]
    for k in range(start, steps):
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


def _recursive_dmdc_initialization(
    X: np.ndarray,
    U: np.ndarray,
    rho: float,
    initial_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exponentially weighted initial DMDc fit and inverse regressor covariance."""
    omega = np.vstack((X[:, : initial_window - 1], U[:, : initial_window - 1]))
    target = X[:, 1:initial_window]
    ages = np.arange(initial_window - 2, -1, -1, dtype=float)
    weights = rho ** (0.5 * ages)
    weighted_omega = omega * weights
    weighted_target = target * weights
    covariance = weighted_omega @ weighted_omega.T.conj()
    inverse_covariance = np.linalg.pinv(covariance)
    gain_matrix = weighted_target @ weighted_omega.T.conj() @ inverse_covariance
    return gain_matrix, inverse_covariance


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
    """Recursive DMDc following Bou Hamdan et al. (2023).

    The recursive least-squares update is initialized from an exponentially
    weighted DMDc fit on an initial batch instead of an arbitrary identity
    operator. ``initial_window=None`` uses the smallest batch with at least as
    many transitions as regressors in ``[x; u]``.
    """

    def __init__(
        self,
        *,
        rho: float = 1.0,
        rank: int | None = None,
        initial_window: int | None = None,
    ) -> None:
        if not 0 < rho <= 1:
            raise ValueError("rho must satisfy 0 < rho <= 1.")
        if initial_window is not None and initial_window < 2:
            raise ValueError("initial_window must be at least 2.")
        self.rho = rho
        self.rank = rank
        self.initial_window = initial_window

    def fit(self, X: np.ndarray, U: np.ndarray) -> RecursiveDMDc:
        X = _snapshots(X)
        U = _snapshots(U, "U")
        if X.shape[1] != U.shape[1]:
            raise ValueError("X and U must contain the same number of snapshots.")
        projection = _basis(X, self.rank)
        reduced = projection.conj().T @ X
        minimum = reduced.shape[0] + U.shape[0] + 1
        initial_window = minimum if self.initial_window is None else self.initial_window
        if initial_window < minimum:
            raise ValueError(
                f"initial_window must be at least {minimum} for the reduced state/control dimensions."
            )
        if initial_window >= X.shape[1]:
            raise ValueError("initial_window must leave at least one snapshot for recursive updating.")
        controls = U.astype(reduced.dtype, copy=False)
        gain_matrix, covariance = _recursive_dmdc_initialization(
            reduced, controls, self.rho, initial_window
        )
        operators, control_operators, predicted = _recursive_dmdc_core(
            reduced,
            controls,
            self.rho,
            gain_matrix.astype(reduced.dtype, copy=False),
            covariance.astype(reduced.dtype, copy=False),
            initial_window,
        )
        self.operators = np.einsum("ij,tjk,lk->til", projection, operators, projection.conj())
        self.control_operators = np.einsum("ij,tjk->tik", projection, control_operators)
        self.predicted_data = projection @ predicted
        self.eigs, reduced_modes = np.linalg.eig(operators)
        self.modes = np.einsum("ij,tjk->tik", projection, reduced_modes)
        self.operator = self.operators[-1]
        self.control_operator = self.control_operators[-1]
        self.initial_window_ = initial_window
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
