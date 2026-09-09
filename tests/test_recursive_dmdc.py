import unittest

import numpy as np

from ekfdmd._online import _recursive_dmdc_core, _recursive_dmdc_initialization


class RecursiveDMDcTests(unittest.TestCase):
    def test_numba_matches_python_kernel(self):
        if not hasattr(_recursive_dmdc_core, "py_func"):
            self.skipTest("Numba is not installed")
        X = np.array(
            [[0.2, 0.3, 0.5, 0.7, 0.9], [0.1, 0.0, -0.1, -0.2, -0.3]],
            dtype=float,
        )
        U = np.array([[1.0, 0.5, -0.5, -1.0, -0.2]], dtype=float)
        gain, covariance = _recursive_dmdc_initialization(X, U, 0.97, 4)
        compiled = _recursive_dmdc_core(X, U, 0.97, gain, covariance, 4)
        interpreted = _recursive_dmdc_core.py_func(X, U, 0.97, gain, covariance, 4)
        for actual, expected in zip(compiled, interpreted, strict=True):
            np.testing.assert_allclose(actual, expected)

    def test_initialization_recovers_static_controlled_system(self):
        A = np.array([[0.95, 0.05], [-0.02, 0.90]])
        B = np.array([[0.10], [0.25]])
        U = np.array([[1.0, -0.5, 0.8, 0.2, -1.0, 0.4, 0.7, -0.3]])
        X = np.empty((2, U.shape[1]))
        X[:, 0] = (0.2, -0.1)
        for k in range(U.shape[1] - 1):
            X[:, k + 1] = A @ X[:, k] + B @ U[:, k]

        gain, _ = _recursive_dmdc_initialization(X, U, 1.0, X.shape[1])
        np.testing.assert_allclose(gain[:, :2], A, atol=1e-10)
        np.testing.assert_allclose(gain[:, 2:], B, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
