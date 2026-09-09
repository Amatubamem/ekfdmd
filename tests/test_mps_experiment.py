import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class MPSExperimentTests(unittest.TestCase):
    def test_matches_default_ekfdmd(self):
        import torch

        if not torch.backends.mps.is_available():
            self.skipTest("Apple MPS is not available")

        from ekfdmd import EKFDMD
        from ekfdmd._mps_experimental import MPSEKFDMD

        X = np.random.default_rng(123).standard_normal((3, 12))
        kwargs = dict(state_noise=1e-3, operator_noise=1e-5, measurement_noise=2e-3)
        expected = EKFDMD(**kwargs).fit(X)
        actual = MPSEKFDMD(**kwargs).fit(X)

        np.testing.assert_allclose(actual.operator, expected.operator, rtol=2e-3, atol=2e-4)
        np.testing.assert_allclose(actual.filtered_data, expected.filtered_data, rtol=2e-3, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
