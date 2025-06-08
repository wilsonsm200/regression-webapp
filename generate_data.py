import os
import numpy as np
import pandas as pd

# 1. Create 'data' folder if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# 2. Generate synthetic data (n > 300)
n = 350
np.random.seed(42)
x = np.linspace(1, 100, n) + np.random.normal(0, 2, n)  # Add some noise for realism

# Create various y relationships (all noisy)
y_linear = 2.5 * x + 7 + np.random.normal(0, 10, n)
y_quadratic = 0.2 * x**2 - 3 * x + 20 + np.random.normal(0, 60, n)
y_log = 6 + 4 * np.log(x) + np.random.normal(0, 1.2, n)
y_exp = 1.2 * np.exp(0.045 * x) + np.random.normal(0, 200, n)
y_power = 0.7 * x ** 1.6 + np.random.normal(0, 30, n)
y_rational = 10 + 25 / (x + 1) + np.random.normal(0, 0.5, n)

# 3. Combine into DataFrame
df = pd.DataFrame({
    'x': x,
    'y_linear': y_linear,
    'y_quadratic': y_quadratic,
    'y_logarithmic': y_log,
    'y_exponential': y_exp,
    'y_power': y_power,
    'y_rational': y_rational
})

# 4. Save to CSV
csv_path = os.path.join('data', 'synthetic_regression_data.csv')
df.to_csv(csv_path, index=False)

print(f"Synthetic data with {n} rows saved to {csv_path}")
