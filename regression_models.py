import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import statsmodels.api as sm

# 1. Load data
df = pd.read_csv('data/synthetic_regression_data.csv')
x = df['x'].values

# 2. Create output folder if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')

# 3. Define regression functions

def linear_regression(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    y_pred = slope * x + intercept
    print(f"Linear Regression:\na = {slope:.4f}, b = {intercept:.4f}")
    print(f"R² = {r_value**2:.4f}, p-value = {p_value:.4g}")
    plot_regression(x, y, y_pred, "Linear Regression", "linear_regression.png")
    return slope, intercept, r_value**2, p_value

def quadratic_regression(x, y):
    coeffs = np.polyfit(x, y, 2)
    y_pred = coeffs[0]*x**2 + coeffs[1]*x + coeffs[2]
    r2 = 1 - np.sum((y-y_pred)**2) / np.sum((y-np.mean(y))**2)
    # p-value via OLS for quadratic
    X_quad = np.column_stack((x**2, x, np.ones(len(x))))
    model = sm.OLS(y, X_quad).fit()
    print(f"Quadratic Regression:\na = {coeffs[0]:.4f}, b = {coeffs[1]:.4f}, c = {coeffs[2]:.4f}")
    print(f"R² = {r2:.4f}, p-value = {model.f_pvalue:.4g}")
    plot_regression(x, y, y_pred, "Quadratic Regression", "quadratic_regression.png")
    return coeffs, r2, model.f_pvalue

def logarithmic_regression(x, y):
    X = np.log(x)
    slope, intercept, r_value, p_value, std_err = linregress(X, y)
    y_pred = slope * X + intercept
    print(f"Logarithmic Regression:\na = {intercept:.4f}, b = {slope:.4f}")
    print(f"R² = {r_value**2:.4f}, p-value = {p_value:.4g}")
    plot_regression(x, y, y_pred, "Logarithmic Regression", "logarithmic_regression.png")
    return slope, intercept, r_value**2, p_value

def exponential_regression(x, y):
    def exp_func(x, a, b):
        return a * np.exp(b * x)
    params, cov = curve_fit(exp_func, x, y, maxfev=10000)
    y_pred = exp_func(x, *params)
    # R² calculation
    r2 = 1 - np.sum((y-y_pred)**2) / np.sum((y-np.mean(y))**2)
    print(f"Exponential Regression:\na = {params[0]:.4f}, b = {params[1]:.4f}")
    print(f"R² = {r2:.4f}")
    plot_regression(x, y, y_pred, "Exponential Regression", "exponential_regression.png")
    return params, r2

def power_regression(x, y):
    X = np.log(x)
    Y = np.log(y)
    slope, intercept, r_value, p_value, std_err = linregress(X, Y)
    a = np.exp(intercept)
    b = slope
    y_pred = a * x ** b
    print(f"Power Regression (y = a*x^b):\na = {a:.4f}, b = {b:.4f}")
    print(f"R² = {r_value**2:.4f}, p-value = {p_value:.4g}")
    plot_regression(x, y, y_pred, "Power Regression", "power_regression.png")
    return a, b, r_value**2, p_value

def general_power_regression(x, y):
    Y = np.log(y)
    X = x
    slope, intercept, r_value, p_value, std_err = linregress(X, Y)
    a = np.exp(intercept)
    b = np.exp(slope)
    y_pred = a * b ** x
    print(f"General Power Regression (y = a*b^x):\na = {a:.4f}, b = {b:.4f}")
    print(f"R² = {r_value**2:.4f}, p-value = {p_value:.4g}")
    plot_regression(x, y, y_pred, "General Power Regression", "general_power_regression.png")
    return a, b, r_value**2, p_value

def rational_regression(x, y):
    X = 1 / x
    slope, intercept, r_value, p_value, std_err = linregress(X, y)
    y_pred = slope * X + intercept
    print(f"Rational Regression (y = a + b/x):\na = {intercept:.4f}, b = {slope:.4f}")
    print(f"R² = {r_value**2:.4f}, p-value = {p_value:.4g}")
    plot_regression(x, y, y_pred, "Rational Regression", "rational_regression.png")
    return intercept, slope, r_value**2, p_value

# 4. Visualization Function
def plot_regression(x, y, y_pred, title, filename):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='blue', s=10, label='Data')
    plt.plot(x, y_pred, color='red', linewidth=2, label='Fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('output', filename))
    plt.close()
    print(f"Plot saved as output/{filename}\n")

# 5. Menu for selection
def main():
    print("Choose regression type:")
    print("1. Linear (y = ax + b)")
    print("2. Quadratic (y = ax^2 + bx + c)")
    print("3. Logarithmic (y = a + b*ln(x))")
    print("4. Exponential (y = a * exp(bx))")
    print("5. Power (y = a * x^b)")
    print("6. General Power (y = a * b^x)")
    print("7. Rational (y = a + b/x)")

    choice = input("Enter choice (1-7): ")

    if choice == "1":
        y = df['y_linear'].values
        linear_regression(x, y)
    elif choice == "2":
        y = df['y_quadratic'].values
        quadratic_regression(x, y)
    elif choice == "3":
        y = df['y_logarithmic'].values
        logarithmic_regression(x, y)
    elif choice == "4":
        y = df['y_exponential'].values
        exponential_regression(x, y)
    elif choice == "5":
        y = df['y_power'].values
        power_regression(x, y)
    elif choice == "6":
        y = df['y_power'].values  
        general_power_regression(x, y)
    elif choice == "7":
        y = df['y_rational'].values
        rational_regression(x, y)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
