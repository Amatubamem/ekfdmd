import unittest

import numpy as np

from ekfdmd._online import _recursive_dmdc_core


class RecursiveDMDcTests(unittest.TestCase):
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
