import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("cupy"), "CuPy is not installed")
class CuPyTests(unittest.TestCase):
    def test_matches_default_ekfdmd(self):
        import cupy as cp

        try:
            if cp.cuda.runtime.getDeviceCount() == 0:
                self.skipTest("CUDA device is not available")
        except cp.cuda.runtime.CUDARuntimeError:
            self.skipTest("CUDA runtime is not available")

        from ekfdmd import EKFDMD
        from ekfdmd.cupy import CuPyEKFDMD

        X = np.random.default_rng(123).standard_normal((3, 12))
        kwargs = dict(state_noise=1e-3, operator_noise=1e-5, measurement_noise=2e-3)
        expected = EKFDMD(**kwargs).fit(X)
        actual = CuPyEKFDMD(**kwargs).fit(X)

        np.testing.assert_allclose(actual.operator, expected.operator, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(actual.filtered_data, expected.filtered_data, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
