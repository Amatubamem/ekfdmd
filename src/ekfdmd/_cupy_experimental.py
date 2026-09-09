from __future__ import annotations

import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.sparse import block_array, csr_array, eye
except ImportError as exc:  # pragma: no cover - exercised only without the optional dependency
    raise ImportError(
        "CuPy is required for ekfdmd._cupy_experimental. Install the wheel matching your CUDA version."
    ) from exc


def _snapshots(array: np.ndarray, name: str = "X") -> cp.ndarray:
    array = cp.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_features, n_snapshots).")
    if array.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two snapshots.")
    if not np.issubdtype(array.dtype, np.number) or not bool(cp.all(cp.isfinite(array)).item()):
        raise ValueError(f"{name} must contain only finite numeric values.")
    return array.astype(cp.result_type(array.dtype, cp.float64), copy=False)


def _rank(singular_values: cp.ndarray, requested: int | None) -> int:
    tolerance = cp.finfo(singular_values.dtype).eps * singular_values[0] * singular_values.size
    numerical = int(cp.count_nonzero(singular_values > tolerance).item())
    if numerical == 0:
        raise ValueError("Snapshot matrix has numerical rank zero.")
    if requested is None:
        return numerical
    if not 1 <= requested <= numerical:
        raise ValueError(f"svd_rank must be between 1 and the numerical rank ({numerical}).")
    return requested


def _dmd(window: cp.ndarray) -> cp.ndarray:
    x, y = window[:, :-1], window[:, 1:]
    u, s, vh = cp.linalg.svd(x, full_matrices=False)
    rank = _rank(s, None)
    u, s, v = u[:, :rank], s[:rank], vh.conj().T[:, :rank]
    return ((y @ v) / s) @ u.conj().T


def _basis(data: cp.ndarray, rank: int | None) -> cp.ndarray:
    n = data.shape[0]
    if rank is None or rank == n:
        return cp.eye(n, dtype=data.dtype)
    if not 1 <= rank < n:
        raise ValueError(f"rank must be between 1 and {n}.")
    u, s, _ = cp.linalg.svd(data[:, :-1], full_matrices=False)
    available = _rank(s, None)
    if rank > available:
        raise ValueError(f"rank={rank} exceeds the numerical snapshot rank ({available}).")
    return u[:, :rank]


def _jacobian(vector: cp.ndarray, rows: int) -> csr_array:
    cols = vector.size
    row = cp.repeat(cp.arange(rows), cols)
    col = cp.arange(rows * cols)
    return csr_array((cp.tile(vector, rows), (row, col)), shape=(rows, rows * cols))


class CuPyEKFDMD:
    """Experimental CUDA implementation of EKF-DMD using CuPy."""

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

    def fit(self, X: np.ndarray) -> CuPyEKFDMD:
        X_gpu = _snapshots(X)
        projection = _basis(X_gpu, self.rank)
        reduced = projection.conj().T @ X_gpu
        n, steps = reduced.shape
        A = _dmd(reduced) if self.initialize_with_dmd else cp.eye(n, dtype=reduced.dtype)
        x = reduced[:, 0].copy()
        state_size = n + n * n
        P = cp.eye(state_size, dtype=reduced.dtype) * self.initial_covariance
        q = cp.concatenate((cp.full(n, self.state_noise), cp.full(n * n, self.operator_noise)))
        diagonal = cp.arange(state_size)
        R = cp.eye(n, dtype=reduced.dtype) * self.measurement_noise

        operators = cp.empty((steps, n, n), dtype=reduced.dtype)
        filtered = cp.empty((n, steps), dtype=reduced.dtype)
        predicted = cp.empty((n, steps), dtype=reduced.dtype)
        operators[0], filtered[:, 0], predicted[:, 0] = A, x, x

        for k in range(1, steps):
            x_pred = A @ x
            F = block_array(
                [[csr_array(A), _jacobian(x, n)], [None, eye(n * n, format="csr", dtype=reduced.dtype)]],
                format="csr",
            )
            FP = F @ P
            P_pred = (F @ FP.T.conj()).T.conj()
            P_pred[diagonal, diagonal] += q
            cross = P_pred[:, :n]
            innovation_cov = P_pred[:n, :n] + R
            K = cp.linalg.solve(innovation_cov.T.conj(), cross.T.conj()).T.conj()
            delta = K @ (reduced[:, k] - x_pred)
            x = x_pred + delta[:n]
            A = A + delta[n:].reshape(n, n)
            P = P_pred - K @ cross.T.conj()
            P = 0.5 * (P + P.T.conj())
            operators[k], filtered[:, k], predicted[:, k] = A, x, x_pred

        operators_reduced = cp.asnumpy(operators)
        operators_full = cp.asnumpy(cp.einsum("ij,tjk,lk->til", projection, operators, projection.conj()))
        filtered_full = cp.asnumpy(projection @ filtered)
        predicted_full = cp.asnumpy(projection @ predicted)

        # ponytail: CuPy's general eigendecomposition requires cuSOLVER geev,
        # which is unavailable on some CUDA installations. Keep the EKF on the
        # GPU and use NumPy only for this post-processing step.
        eigs, reduced_modes = np.linalg.eig(operators_reduced)
        modes = np.einsum("ij,tjk->tik", cp.asnumpy(projection), reduced_modes)

        self.operators = operators_full
        self.filtered_data = filtered_full
        self.predicted_data = predicted_full
        self.eigs = eigs
        self.modes = modes
        self.operator = self.operators[-1]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "operator"):
            raise RuntimeError("fit() must be called before predict().")
        x = np.asarray(x)
        if x.ndim not in (1, 2) or x.shape[0] != self.operator.shape[1]:
            raise ValueError("x has an incompatible state dimension.")
        return self.operator @ x
