import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class MPSTests(unittest.TestCase):
    def test_matches_float32_cpu_reference(self):
        import torch

        if not torch.backends.mps.is_available():
            self.skipTest("Apple MPS is not available")

        from ekfdmd.mps import MPSEKFDMD

        X = np.random.default_rng(123).standard_normal((3, 12))
        kwargs = dict(state_noise=1e-3, operator_noise=1e-5, measurement_noise=2e-3, history=4)
        expected = MPSEKFDMD(**kwargs)._fit_device(X, torch.device("cpu"))
        actual = MPSEKFDMD(**kwargs).fit(X)

        np.testing.assert_array_equal(actual.history_indices, [0, 4, 8, 11])
        for actual_array, expected_array in (
            (actual.operator, expected.operator),
            (actual.operators, expected.operators),
            (actual.filtered_data, expected.filtered_data),
        ):
            difference = actual_array - expected_array
            relative_frobenius_error = np.linalg.norm(difference) / np.linalg.norm(expected_array)
            max_absolute_error = np.max(np.abs(difference))
            self.assertLess(relative_frobenius_error, 2e-3)
            self.assertLess(max_absolute_error, 2e-3)

    def test_repeated_mps_fit_is_deterministic(self):
        import torch

        if not torch.backends.mps.is_available():
            self.skipTest("Apple MPS is not available")

        from ekfdmd.mps import MPSEKFDMD

        X = np.random.default_rng(456).standard_normal((3, 12))
        first = MPSEKFDMD().fit(X)
        second = MPSEKFDMD().fit(X)
        np.testing.assert_array_equal(first.operator, second.operator)


if __name__ == "__main__":
    unittest.main()
