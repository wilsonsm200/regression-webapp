# test_regression_models.py
import unittest
import numpy as np
from regression_models import *

class TestRegressionFunctions(unittest.TestCase):
    def test_linear_regression(self):
        # Perfect linear relationship
        x = np.array([1, 2, 3, 4, 5])
        y = 2 * x + 1
        # Suppress plotting during test by temporarily disabling plot_regression
        result = linear_regression(x, y)
        # Manually check for expected values printed
        self.assertAlmostEqual(result[0], 2.0, places=3)  # Slope
        self.assertAlmostEqual(result[1], 1.0, places=3)  # Intercept
        self.assertAlmostEqual(result[2], 1.0, places=3)  # R²

    def test_logarithmic_regression(self):
        x = np.array([1, 2, 3, 4, 5])
        y = 2 + 3 * np.log(x)
        result = logarithmic_regression(x, y)
        self.assertAlmostEqual(result[0], 3.0, places=2)  # Slope

    def test_nan_input(self):
        x = np.array([1, 2, 3, np.nan, 5])
        y = np.array([2, 4, 6, 8, 10])
        with self.assertRaises(Exception):
            linear_regression(x, y)

if __name__ == "__main__":
    unittest.main()
