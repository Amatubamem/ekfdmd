from __future__ import annotations

import numpy as np
from scipy.sparse import block_array, csr_array, eye

from ._dmd import _basis, _dmd, _dmdc, _snapshots


def _jacobian(vector: np.ndarray, rows: int) -> csr_array:
    cols = vector.size
    row = np.repeat(np.arange(rows), cols)
    col = np.arange(rows * cols)
    return csr_array((np.tile(vector, rows), (row, col)), shape=(rows, rows * cols))


class EKFDMD:
    """Extended Kalman filter Dynamic Mode Decomposition.

    The augmented-state Jacobian is stored sparsely while the covariance stays
    dense. This is the package's default EKF-DMD implementation.
    """

    def __init__(
        self,
        *,
        state_noise: float = 0.01,
        operator_noise: float = 0.0,
        measurement_noise: float = 0.01,
        initial_covariance: float = 100.0,
        rank: int | None = None,
        initialize_with_dmd: bool = True,
    ) -> None:
        for name, value in (
            ("state_noise", state_noise),
            ("operator_noise", operator_noise),
            ("measurement_noise", measurement_noise),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        if initial_covariance <= 0:
            raise ValueError("initial_covariance must be positive.")
        self.state_noise = state_noise
        self.operator_noise = operator_noise
        self.measurement_noise = measurement_noise
        self.initial_covariance = initial_covariance
        self.rank = rank
        self.initialize_with_dmd = initialize_with_dmd

    def fit(self, X: np.ndarray) -> EKFDMD:
        X = _snapshots(X)
        projection = _basis(X, self.rank)
        reduced = projection.conj().T @ X
        n, steps = reduced.shape
        A = _dmd(reduced)[0] if self.initialize_with_dmd else np.eye(n, dtype=reduced.dtype)
        x = reduced[:, 0].copy()
        state_size = n + n * n
        P = np.eye(state_size, dtype=reduced.dtype) * self.initial_covariance
        q = np.concatenate((np.full(n, self.state_noise), np.full(n * n, self.operator_noise)))
        Q = csr_array((q, (np.arange(state_size), np.arange(state_size))), shape=(state_size, state_size))
        R = np.eye(n, dtype=reduced.dtype) * self.measurement_noise

        operators = np.empty((steps, n, n), dtype=reduced.dtype)
        filtered = np.empty((n, steps), dtype=reduced.dtype)
        predictions = np.empty((n, steps), dtype=reduced.dtype)
        operators[0], filtered[:, 0], predictions[:, 0] = A, x, x

        for k in range(1, steps):
            x_pred = A @ x
            F = block_array(
                [[csr_array(A), _jacobian(x, n)], [None, eye(n * n, format="csr", dtype=reduced.dtype)]],
                format="csr",
            )
            P_pred = np.asarray(F @ P @ F.T.conj() + Q)
            cross = P_pred[:, :n]
            innovation_cov = P_pred[:n, :n] + R
            K = np.linalg.solve(innovation_cov.T.conj(), cross.T.conj()).T.conj()
            delta = K @ (reduced[:, k] - x_pred)
            x = x_pred + delta[:n]
            A = A + delta[n:].reshape(n, n)
            P = P_pred - K @ cross.T.conj()
            P = 0.5 * (P + P.T.conj())
            operators[k], filtered[:, k], predictions[:, k] = A, x, x_pred

        self.operators = np.einsum("ij,tjk,lk->til", projection, operators, projection.conj())
        self.filtered_data = projection @ filtered
        self.predictions = projection @ predictions
        self.eigs, reduced_modes = np.linalg.eig(operators)
        self.modes = np.einsum("ij,tjk->tik", projection, reduced_modes)
        self.operator = self.operators[-1]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "operator"):
            raise RuntimeError("fit() must be called before predict().")
        x = np.asarray(x)
        if x.ndim not in (1, 2) or x.shape[0] != self.operator.shape[1]:
            raise ValueError("x has an incompatible state dimension.")
        return self.operator @ x


class EKFDMDc(EKFDMD):
    """EKF-DMD with known control input and estimated state/control operators."""

    def fit(self, X: np.ndarray, U: np.ndarray) -> EKFDMDc:
        X = _snapshots(X)
        U = _snapshots(U, "U")
        if X.shape[1] != U.shape[1]:
            raise ValueError("X and U must contain the same number of snapshots.")
        projection = _basis(X, self.rank)
        reduced = projection.conj().T @ X
        n, steps = reduced.shape
        m = U.shape[0]
        if self.initialize_with_dmd:
            A, B = _dmdc(reduced, U)
        else:
            A = np.eye(n, dtype=reduced.dtype)
            B = np.zeros((n, m), dtype=reduced.dtype)
        x = reduced[:, 0].copy()
        state_size = n + n * n + n * m
        P = np.eye(state_size, dtype=reduced.dtype) * self.initial_covariance
        q = np.concatenate((np.full(n, self.state_noise), np.full(n * n + n * m, self.operator_noise)))
        Q = csr_array((q, (np.arange(state_size), np.arange(state_size))), shape=(state_size, state_size))
        R = np.eye(n, dtype=reduced.dtype) * self.measurement_noise

        operators = np.empty((steps, n, n), dtype=reduced.dtype)
        controls = np.empty((steps, n, m), dtype=reduced.dtype)
        filtered = np.empty((n, steps), dtype=reduced.dtype)
        predictions = np.empty((n, steps), dtype=reduced.dtype)
        operators[0], controls[0], filtered[:, 0], predictions[:, 0] = A, B, x, x

        for k in range(1, steps):
            u = U[:, k - 1]
            x_pred = A @ x + B @ u
            F = block_array(
                [
                    [csr_array(A), _jacobian(x, n), _jacobian(u, n)],
                    [None, eye(n * n, format="csr", dtype=reduced.dtype), None],
                    [None, None, eye(n * m, format="csr", dtype=reduced.dtype)],
                ],
                format="csr",
            )
            P_pred = np.asarray(F @ P @ F.T.conj() + Q)
            cross = P_pred[:, :n]
            innovation_cov = P_pred[:n, :n] + R
            K = np.linalg.solve(innovation_cov.T.conj(), cross.T.conj()).T.conj()
            delta = K @ (reduced[:, k] - x_pred)
            x = x_pred + delta[:n]
            A = A + delta[n : n + n * n].reshape(n, n)
            B = B + delta[n + n * n :].reshape(n, m)
            P = P_pred - K @ cross.T.conj()
            P = 0.5 * (P + P.T.conj())
            operators[k], controls[k], filtered[:, k], predictions[:, k] = A, B, x, x_pred

        self.operators = np.einsum("ij,tjk,lk->til", projection, operators, projection.conj())
        self.control_operators = np.einsum("ij,tjk->tik", projection, controls)
        self.filtered_data = projection @ filtered
        self.predictions = projection @ predictions
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
