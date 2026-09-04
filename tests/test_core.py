import unittest

import numpy as np

from ekfdmd import EKFDMD, EKFDMDc, OnlineDMD, SlidingWindowDMD, SlidingWindowDMDc
from ekfdmd._online import _online_core
from ekfdmd.postprocess import track_modes


def trajectory(A, x0, steps, B=None, U=None):
    X = np.empty((A.shape[0], steps))
    X[:, 0] = x0
    for k in range(1, steps):
        X[:, k] = A @ X[:, k - 1]
        if B is not None:
            X[:, k] += B @ U[:, k - 1]
    return X


def rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


class CoreTests(unittest.TestCase):
    def test_ekfdmd_tracks_static_operator(self):
        A = np.array([[0.98, -0.08], [0.08, 0.98]])
        X = trajectory(A, np.array([1.0, 0.2]), 80)
        model = EKFDMD(state_noise=1e-8, measurement_noise=1e-8, initialize_with_dmd=False).fit(X)
        np.testing.assert_allclose(model.operator, A, atol=2e-3)
        np.testing.assert_allclose(model.predict(X[:, -1]), A @ X[:, -1], atol=2e-3)

    def test_ekfdmd_matches_legacy_sparse_reference(self):
        A = np.array([[0.97, -0.07], [0.05, 0.96]])
        X = trajectory(A, np.array([1.0, 0.3]), 25)
        X += 0.01 * np.random.default_rng(1234).standard_normal(X.shape)
        model = EKFDMD(
            state_noise=1e-3,
            measurement_noise=2e-3,
            operator_noise=1e-5,
            initial_covariance=50.0,
            rank=2,
        ).fit(X)

        # Regression value from dmdnn.dmd.EKFDMD (sparse implementation).
        expected = np.array(
            [[0.969007280011149, -0.070036791555886], [0.046526711347818, 0.963242857370069]]
        )
        np.testing.assert_allclose(model.operator, expected, atol=1e-12)

    def test_ekfdmdc_tracks_controlled_operator(self):
        A = np.array([[0.95]])
        B = np.array([[0.2]])
        U = np.sin(np.arange(80, dtype=float))[None, :]
        X = trajectory(A, np.array([0.3]), 80, B, U)
        model = EKFDMDc(state_noise=1e-8, measurement_noise=1e-8, initialize_with_dmd=False).fit(X, U)
        np.testing.assert_allclose(model.operator, A, atol=2e-3)
        np.testing.assert_allclose(model.control_operator, B, atol=2e-3)

    def test_ekfdmdc_matches_legacy_sparse_reference(self):
        A = np.array([[0.95, -0.04], [0.03, 0.94]])
        B = np.array([[0.1], [0.05]])
        U = np.sin(np.arange(30) * 0.4)[None, :]
        X = trajectory(A, np.array([0.4, -0.2]), 30, B, U)
        X += 0.005 * np.random.default_rng(4321).standard_normal(X.shape)
        model = EKFDMDc(
            state_noise=1e-3,
            measurement_noise=2e-3,
            operator_noise=1e-5,
            initial_covariance=50.0,
            rank=2,
        ).fit(X, U)

        # Regression values from dmdnn.dmd.EKFDMDc (sparse implementation).
        expected_A = np.array(
            [[0.949093646566917, -0.033816941228443], [0.029472884226107, 0.942182019154442]]
        )
        expected_B = np.array([[0.099319773800250], [0.049761468323242]])
        np.testing.assert_allclose(model.operator, expected_A, atol=1e-12)
        np.testing.assert_allclose(model.control_operator, expected_B, atol=1e-12)

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

    def test_online_dmd_forgetting_tracks_changed_operator(self):
        A1, A2 = rotation(0.03), rotation(0.12)
        X = np.empty((2, 161))
        X[:, 0] = [1.0, 0.2]
        for k in range(160):
            X[:, k + 1] = (A1 if k < 80 else A2) @ X[:, k]

        static = OnlineDMD(rho=1.0, initial_covariance=1e8).fit(X)
        adaptive = OnlineDMD(rho=0.95, initial_covariance=1e8).fit(X)
        self.assertLess(np.linalg.norm(adaptive.operator - A2), np.linalg.norm(static.operator - A2))

    def test_online_numba_matches_python_kernel(self):
        if not hasattr(_online_core, "py_func"):
            self.skipTest("Numba is not installed")
        A = rotation(0.05)
        X = trajectory(A, np.array([1.0, 0.3]), 40)
        compiled = _online_core(X, 0.97, 1e4, True)
        python = _online_core.py_func(X, 0.97, 1e4, True)
        np.testing.assert_allclose(compiled[0], python[0])
        np.testing.assert_allclose(compiled[1], python[1])

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
