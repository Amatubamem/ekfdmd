from __future__ import annotations

import numpy as np


def _snapshots(array: np.ndarray, name: str = "X") -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_features, n_snapshots).")
    if array.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two snapshots.")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numeric values.")
    return array.astype(np.result_type(array.dtype, np.float64), copy=False)


def _rank(singular_values: np.ndarray, requested: int | None) -> int:
    tolerance = np.finfo(singular_values.dtype).eps * singular_values[0] * singular_values.size
    numerical = int(np.count_nonzero(singular_values > tolerance))
    if numerical == 0:
        raise ValueError("Snapshot matrix has numerical rank zero.")
    if requested is None:
        return numerical
    if not 1 <= requested <= numerical:
        raise ValueError(f"svd_rank must be between 1 and the numerical rank ({numerical}).")
    return requested


def _dmd(window: np.ndarray, svd_rank: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = window[:, :-1], window[:, 1:]
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    rank = _rank(s, svd_rank)
    u, s, v = u[:, :rank], s[:rank], vh.conj().T[:, :rank]
    yvsi = (y @ v) / s
    operator = yvsi @ u.conj().T
    eigs, vectors = np.linalg.eig(u.conj().T @ yvsi)
    return operator, eigs, yvsi @ vectors


def _dmdc(window: np.ndarray, controls: np.ndarray, svd_rank: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    omega = np.vstack((window[:, :-1], controls[:, :-1]))
    target = window[:, 1:]
    u, s, vh = np.linalg.svd(omega, full_matrices=False)
    rank = _rank(s, svd_rank)
    gain = ((target @ vh.conj().T[:, :rank]) / s[:rank]) @ u[:, :rank].conj().T
    n = window.shape[0]
    return gain[:, :n], gain[:, n:]


def _basis(data: np.ndarray, rank: int | None) -> np.ndarray:
    n = data.shape[0]
    if rank is None or rank == n:
        return np.eye(n, dtype=data.dtype)
    if not 1 <= rank < n:
        raise ValueError(f"rank must be between 1 and {n}.")
    u, s, _ = np.linalg.svd(data[:, :-1], full_matrices=False)
    available = _rank(s, None)
    if rank > available:
        raise ValueError(f"rank={rank} exceeds the numerical snapshot rank ({available}).")
    return u[:, :rank]


class SlidingWindowDMD:
    """DMD fitted independently on overlapping windows."""

    def __init__(self, window_size: int, *, svd_rank: int | None = None) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2.")
        self.window_size = window_size
        self.svd_rank = svd_rank

    def fit(self, X: np.ndarray) -> SlidingWindowDMD:
        X = _snapshots(X)
        if self.window_size > X.shape[1]:
            raise ValueError("window_size cannot exceed the number of snapshots.")
        fits = [
            _dmd(X[:, i : i + self.window_size], self.svd_rank)
            for i in range(X.shape[1] - self.window_size + 1)
        ]
        self.operators = np.stack([fit[0] for fit in fits])
        eigs = tuple(fit[1] for fit in fits)
        modes = tuple(fit[2] for fit in fits)
        if self.svd_rank is None:
            # ponytail: numerical rank may change by window; preserve each result
            # rather than padding. Specify svd_rank when a fixed-rank history is needed.
            self.eigs, self.modes = eigs, modes
        else:
            self.eigs, self.modes = np.stack(eigs), np.stack(modes)
        self.operator = self.operators[-1]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "operator"):
            raise RuntimeError("fit() must be called before predict().")
        return self.operator @ np.asarray(x)


class SlidingWindowDMDc:
    """Sliding-window DMD with known control input."""

    def __init__(self, window_size: int, *, svd_rank: int | None = None) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2.")
        self.window_size = window_size
        self.svd_rank = svd_rank

    def fit(self, X: np.ndarray, U: np.ndarray) -> SlidingWindowDMDc:
        X = _snapshots(X)
        U = _snapshots(U, "U")
        if X.shape[1] != U.shape[1]:
            raise ValueError("X and U must contain the same number of snapshots.")
        if self.window_size > X.shape[1]:
            raise ValueError("window_size cannot exceed the number of snapshots.")
        fits = [
            _dmdc(X[:, i : i + self.window_size], U[:, i : i + self.window_size], self.svd_rank)
            for i in range(X.shape[1] - self.window_size + 1)
        ]
        self.operators = np.stack([fit[0] for fit in fits])
        self.control_operators = np.stack([fit[1] for fit in fits])
        self.eigs, self.modes = np.linalg.eig(self.operators)
        self.operator = self.operators[-1]
        self.control_operator = self.control_operators[-1]
        return self

    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        if not hasattr(self, "operator"):
            raise RuntimeError("fit() must be called before predict().")
        return self.operator @ np.asarray(x) + self.control_operator @ np.asarray(u)
