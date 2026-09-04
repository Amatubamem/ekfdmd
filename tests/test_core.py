import unittest

import numpy as np

from ekfdmd import EKFDMD, EKFDMDc, OnlineDMD, SlidingWindowDMD, SlidingWindowDMDc
from ekfdmd.postprocess import track_modes


def trajectory(A, x0, steps, B=None, U=None):
    X = np.empty((A.shape[0], steps))
    X[:, 0] = x0
    for k in range(1, steps):
        X[:, k] = A @ X[:, k - 1]
        if B is not None:
            X[:, k] += B @ U[:, k - 1]
    return X


class CoreTests(unittest.TestCase):
    def test_ekfdmd_tracks_static_operator(self):
        A = np.array([[0.98, -0.08], [0.08, 0.98]])
        X = trajectory(A, np.array([1.0, 0.2]), 80)
        model = EKFDMD(state_noise=1e-8, measurement_noise=1e-8, initialize_with_dmd=False).fit(X)
        np.testing.assert_allclose(model.operator, A, atol=2e-3)
        np.testing.assert_allclose(model.predict(X[:, -1]), A @ X[:, -1], atol=2e-3)

    def test_ekfdmdc_tracks_controlled_operator(self):
        A = np.array([[0.95]])
        B = np.array([[0.2]])
        U = np.sin(np.arange(80, dtype=float))[None, :]
        X = trajectory(A, np.array([0.3]), 80, B, U)
        model = EKFDMDc(state_noise=1e-8, measurement_noise=1e-8, initialize_with_dmd=False).fit(X, U)
        np.testing.assert_allclose(model.operator, A, atol=2e-3)
        np.testing.assert_allclose(model.control_operator, B, atol=2e-3)

    def test_sliding_window_variants(self):
        A = np.array([[0.9]])
        B = np.array([[0.1]])
        U = np.cos(np.arange(20, dtype=float))[None, :]
        X = trajectory(A, np.array([1.0]), 20, B, U)
        swc = SlidingWindowDMDc(8).fit(X, U)
        np.testing.assert_allclose(swc.operator, A, atol=1e-12)
        np.testing.assert_allclose(swc.control_operator, B, atol=1e-12)

        autonomous = trajectory(A, np.array([1.0]), 20)
        sw = SlidingWindowDMD(8).fit(autonomous)
        np.testing.assert_allclose(sw.operator, A, atol=1e-12)

    def test_online_dmd(self):
        A = np.array([[0.97, -0.04], [0.04, 0.97]])
        X = trajectory(A, np.array([1.0, 0.4]), 200)
        model = OnlineDMD(initial_covariance=1e8).fit(X)
        np.testing.assert_allclose(model.operator, A, atol=2e-3)

    def test_track_modes_uses_mac_across_permutations_and_phase(self):
        first, second = 0.9 + 0.1j, 0.8 + 0.2j
        eigs = np.array([[first, second], [second, first], [first, second]])
        modes = np.array(
            [
                [[1, 0], [0, 1]],
                [[0, -1], [1, 0]],
                [[1j, 0], [0, 1]],
            ],
            dtype=complex,
        )

        tracked_eigs, tracked_modes = track_modes(eigs, modes)

        np.testing.assert_allclose(tracked_eigs, np.tile([first, second], (3, 1)))
        np.testing.assert_allclose(np.abs(tracked_modes[:, :, 0]), np.tile([[1, 0]], (3, 1)))
        np.testing.assert_allclose(np.abs(tracked_modes[:, :, 1]), np.tile([[0, 1]], (3, 1)))


if __name__ == "__main__":
    unittest.main()
