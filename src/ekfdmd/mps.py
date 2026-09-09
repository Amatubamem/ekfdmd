from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only without the optional dependency
    raise ImportError("PyTorch is required for ekfdmd.mps. Install ekfdmd[mps].") from exc

from ._dmd import _basis, _dmd, _snapshots
from ._ekf import History, _history_indices

__all__ = ["MPSEKFDMD"]


def _h_left(vector: torch.Tensor, matrix: torch.Tensor, rows: int) -> torch.Tensor:
    """Multiply (I_rows kron vector.T) by matrix without forming the Jacobian."""
    return torch.einsum("j,ijk->ik", vector, matrix.reshape(rows, vector.numel(), -1))


class MPSEKFDMD:
    """Single-precision EKF-DMD backend for Apple Metal Performance Shaders.

    The EKF update runs in float32 on MPS. Results can therefore differ slightly
    from the default float64 NumPy/SciPy implementation after many sequential
    updates.
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
        history: History = "final",
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
        self.history = history

    def fit(self, X: np.ndarray) -> MPSEKFDMD:
        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS is not available.")
        return self._fit_device(X, torch.device("mps"))

    def _fit_device(self, X: np.ndarray, device: torch.device) -> MPSEKFDMD:
        """Run the float32 implementation on one Torch device for validation."""
        X = _snapshots(X)
        projection = _basis(X, self.rank)
        reduced_cpu = projection.conj().T @ X
        n, steps = reduced_cpu.shape
        history_indices = _history_indices(steps, self.history)
        A_cpu = _dmd(reduced_cpu)[0] if self.initialize_with_dmd else np.eye(n, dtype=reduced_cpu.dtype)

        dtype = torch.float32
        reduced = torch.as_tensor(reduced_cpu, dtype=dtype, device=device)
        A = torch.as_tensor(A_cpu, dtype=dtype, device=device)
        x = reduced[:, 0].clone()

        operator_size = n * n
        state_size = n + operator_size
        P = torch.eye(state_size, dtype=dtype, device=device) * self.initial_covariance
        q = torch.cat(
            (
                torch.full((n,), self.state_noise, dtype=dtype, device=device),
                torch.full((operator_size,), self.operator_noise, dtype=dtype, device=device),
            )
        )
        R = torch.eye(n, dtype=dtype, device=device) * self.measurement_noise

        operators = torch.empty((history_indices.size, n, n), dtype=dtype, device=device)
        filtered = torch.empty((n, steps), dtype=dtype, device=device)
        predicted = torch.empty((n, steps), dtype=dtype, device=device)
        filtered[:, 0], predicted[:, 0] = x, x
        history_position = 0
        if history_indices[0] == 0:
            operators[0] = A
            history_position = 1

        for k in range(1, steps):
            x_pred = A @ x

            Pxx = P[:n, :n]
            PxA = P[:n, n:]
            PAx = P[n:, :n]
            PAA = P[n:, n:]

            left_state = A @ Pxx + _h_left(x, PAx, n)
            upper_right = A @ PxA + _h_left(x, PAA, n)
            upper_left = left_state @ A.T + _h_left(x, upper_right.T, n).T

            P_pred = torch.empty_like(P)
            P_pred[:n, :n] = upper_left
            P_pred[:n, n:] = upper_right
            P_pred[n:, :n] = upper_right.T
            P_pred[n:, n:] = PAA
            P_pred.diagonal().add_(q)

            cross = P_pred[:, :n]
            innovation_cov = P_pred[:n, :n] + R
            K = torch.linalg.solve(innovation_cov.T, cross.T).T
            delta = K @ (reduced[:, k] - x_pred)
            x = x_pred + delta[:n]
            A = A + delta[n:].reshape(n, n)
            P = P_pred - K @ cross.T
            P = 0.5 * (P + P.T)
            filtered[:, k], predicted[:, k] = x, x_pred
            if history_position < history_indices.size and k == history_indices[history_position]:
                operators[history_position] = A
                history_position += 1

        operators_reduced = operators.cpu().numpy()
        filtered_reduced = filtered.cpu().numpy()
        predicted_reduced = predicted.cpu().numpy()
        eigs, reduced_modes = np.linalg.eig(operators_reduced)

        self.history_indices = history_indices
        self.operators = np.einsum("ij,tjk,lk->til", projection, operators_reduced, projection.conj())
        self.filtered_data = projection @ filtered_reduced
        self.predicted_data = projection @ predicted_reduced
        self.eigs = eigs
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
