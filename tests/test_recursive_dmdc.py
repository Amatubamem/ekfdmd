import unittest

import numpy as np

from ekfdmd import RecursiveDMDc
from ekfdmd._online import _recursive_dmdc_core


def trajectory(A, x0, steps, B, U):
    X = np.empty((A.shape[0], steps))
    X[:, 0] = x0
    for k in range(1, steps):
        X[:, k] = A @ X[:, k - 1] + B @ U[:, k - 1]
    return X


class RecursiveDMDcTests(unittest.TestCase):
    def test_tracks_controlled_operator(self):
        A = np.array([[0.93]])
        B = np.array([[0.17]])
        U = np.sin(np.arange(200, dtype=float) * 0.37)[None, :]
        X = trajectory(A, np.array([0.4]), 200, B, U)

        solver = RecursiveDMDc(initial_covariance=1e8)
        self.assertIs(solver.fit(X, U), solver)
        np.testing.assert_allclose(solver.operator, A, atol=2e-3)
        np.testing.assert_allclose(solver.control_operator, B, atol=2e-3)
        self.assertEqual(solver.operators.shape, (200, 1, 1))
        self.assertEqual(solver.control_operators.shape, (200, 1, 1))
        self.assertEqual(solver.predicted_data.shape, X.shape)
        np.testing.assert_allclose(solver.predict(X[:, -1], U[:, -1]), A @ X[:, -1] + B @ U[:, -1], atol=2e-3)

    def test_numba_matches_python_kernel(self):
        if not hasattr(_recursive_dmdc_core, "py_func"):
            self.skipTest("Numba is not installed")
        X = np.array([[0.2, 0.3, 0.5, 0.7], [0.1, 0.0, -0.1, -0.2]])
        U = np.array([[1.0, 0.5, -0.5, -1.0]])
        compiled = _recursive_dmdc_core(X, U, 0.97, 1.0e4, True)
        interpreted = _recursive_dmdc_core.py_func(X, U, 0.97, 1.0e4, True)
        for actual, expected in zip(compiled, interpreted, strict=True):
            np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
