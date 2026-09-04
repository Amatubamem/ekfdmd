"""Post-processing utilities for time-varying DMD results."""

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["track_modes"]


def track_modes(eigs: np.ndarray, modes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Track time-varying modes by maximizing the modal assurance criterion (MAC).

    The final snapshot defines the reference ordering. Earlier snapshots are
    matched backward in time with a Hungarian assignment between adjacent mode
    matrices.

    Args:
        eigs: Eigenvalue history with shape ``(n_steps, n_modes)``.
        modes: Mode-matrix history with shape ``(n_steps, n_features, n_modes)``.

    Returns:
        Eigenvalues and mode matrices reordered consistently over time.
    """
    eigs = np.asarray(eigs)
    modes = np.asarray(modes)
    if eigs.ndim != 2 or modes.ndim != 3:
        raise ValueError("eigs and modes must have shapes (steps, modes) and (steps, features, modes).")
    if modes.shape[0] != eigs.shape[0] or modes.shape[2] != eigs.shape[1] or eigs.size == 0:
        raise ValueError("eigs and modes must contain the same non-zero number of steps and modes.")
    if not np.all(np.isfinite(eigs)) or not np.all(np.isfinite(modes)):
        raise ValueError("eigs and modes must contain only finite values.")

    tracked_eigs = eigs.copy()
    tracked_modes = modes.copy()
    n_modes = eigs.shape[1]

    for step in range(eigs.shape[0] - 2, -1, -1):
        current = modes[step]
        reference = tracked_modes[step + 1]
        numerator = np.abs(current.conj().T @ reference) ** 2
        denominator = (
            np.sum(np.abs(current) ** 2, axis=0)[:, None]
            * np.sum(np.abs(reference) ** 2, axis=0)[None, :]
        )
        mac = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)
        current_index, reference_index = linear_sum_assignment(mac, maximize=True)
        order = np.empty(n_modes, dtype=int)
        order[reference_index] = current_index
        tracked_eigs[step] = eigs[step, order]
        tracked_modes[step] = modes[step, :, order]

    return tracked_eigs, tracked_modes
