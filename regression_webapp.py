import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import statsmodels.api as sm

st.set_page_config(page_title="Regression Analysis Web App", layout="centered")
st.title("📈 Regression Analysis Web App (Minimal Debug Version)")

# 1. Data Loading
try:
    uploaded_file = st.sidebar.file_uploader("Upload CSV data", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Custom data loaded!")
    else:
        df = pd.read_csv("data/synthetic_regression_data.csv")
        st.info("Using default synthetic data.")
    st.dataframe(df.head(10))
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

x = df['x'].values

# 2. Regression Selection
reg_types = [
    "Linear (y = ax + b)",
    "Quadratic (y = ax^2 + bx + c)",
    "Logarithmic (y = a + b*ln(x))",
    "Exponential (y = a * exp(bx))",
    "Power (y = a * x^b)",
    "General Power (y = a * b^x)",
    "Rational (y = a + b/x)"
]

reg_name = st.sidebar.selectbox("Regression model", reg_types)
col_map = {
    "Linear (y = ax + b)": "y_linear",
    "Quadratic (y = ax^2 + bx + c)": "y_quadratic",
    "Logarithmic (y = a + b*ln(x))": "y_logarithmic",
    "Exponential (y = a * exp(bx))": "y_exponential",
    "Power (y = a * x^b)": "y_power",
    "General Power (y = a * b^x)": "y_power",
    "Rational (y = a + b/x)": "y_rational",
}
y = df[col_map[reg_name]].values

# 3. Run regression and plot
fig, ax = plt.subplots(figsize=(8, 5))
info = ""
try:
    if reg_name == "Linear (y = ax + b)":
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        y_pred = slope * x + intercept
        info = f"**a (slope):** {slope:.4f}  \n**b (intercept):** {intercept:.4f}  \n**R²:** {r_value**2:.4f}  \n**p-value:** {p_value:.4g}"
    elif reg_name == "Quadratic (y = ax^2 + bx + c)":
        coeffs = np.polyfit(x, y, 2)
        y_pred = coeffs[0]*x**2 + coeffs[1]*x + coeffs[2]
        r2 = 1 - np.sum((y-y_pred)**2) / np.sum((y-np.mean(y))**2)
        X_quad = np.column_stack((x**2, x, np.ones_like(x)))
        model = sm.OLS(y, X_quad).fit()
        info = f"**a:** {coeffs[0]:.4f}  \n**b:** {coeffs[1]:.4f}  \n**c:** {coeffs[2]:.4f}  \n**R²:** {r2:.4f}  \n**p-value:** {model.f_pvalue:.4g}"
    elif reg_name == "Logarithmic (y = a + b*ln(x))":
        mask = x > 0
        X = np.log(x[mask])
        y_masked = y[mask]
        slope, intercept, r_value, p_value, std_err = linregress(X, y_masked)
        y_pred = slope * X + intercept
        x = x[mask]
        y = y[mask]
        info = f"**a:** {intercept:.4f}  \n**b:** {slope:.4f}  \n**R²:** {r_value**2:.4f}  \n**p-value:** {p_value:.4g}"
    elif reg_name == "Exponential (y = a * exp(bx))":
        mask = y > 0
        def exp_func(x, a, b):
            return a * np.exp(b * x)
        params, _ = curve_fit(exp_func, x[mask], y[mask], maxfev=10000)
        y_pred = exp_func(x[mask], *params)
        r2 = 1 - np.sum((y[mask]-y_pred)**2) / np.sum((y[mask]-np.mean(y[mask]))**2)
        x = x[mask]
        y = y[mask]
        info = f"**a:** {params[0]:.4f}  \n**b:** {params[1]:.4f}  \n**R²:** {r2:.4f}"
    elif reg_name == "Power (y = a * x^b)":
        mask = (x > 0) & (y > 0)
        X = np.log(x[mask])
        Y = np.log(y[mask])
        slope, intercept, r_value, p_value, std_err = linregress(X, Y)
        a = np.exp(intercept)
        b = slope
        y_pred = a * x[mask] ** b
        x = x[mask]
        y = y[mask]
        info = f"**a:** {a:.4f}  \n**b:** {b:.4f}  \n**R²:** {r_value**2:.4f}  \n**p-value:** {p_value:.4g}"
    elif reg_name == "General Power (y = a * b^x)":
        mask = y > 0
        X = x[mask]
        Y = np.log(y[mask])
        slope, intercept, r_value, p_value, std_err = linregress(X, Y)
        a = np.exp(intercept)
        b = np.exp(slope)
        y_pred = a * (b ** X)
        x = X
        y = y[mask]
        info = f"**a:** {a:.4f}  \n**b:** {b:.4f}  \n**R²:** {r_value**2:.4f}  \n**p-value:** {p_value:.4g}"
    elif reg_name == "Rational (y = a + b/x)":
        mask = x != 0
        X = 1 / x[mask]
        y_masked = y[mask]
        slope, intercept, r_value, p_value, std_err = linregress(X, y_masked)
        y_pred = slope * X + intercept
        x = x[mask]
        y = y[mask]
        info = f"**a:** {intercept:.4f}  \n**b:** {slope:.4f}  \n**R²:** {r_value**2:.4f}  \n**p-value:** {p_value:.4g}"
    else:
        y_pred = np.zeros_like(y)
        info = "Invalid regression type."
    # Plot
    sort_idx = np.argsort(x)
    ax.scatter(x, y, color='blue', s=10, label='Data')
    ax.plot(x[sort_idx], y_pred[sort_idx], color='red', label='Regression Fit')
    ax.set_xlabel('x')
    ax.set_ylabel(col_map[reg_name])
    ax.set_title(f"{reg_name}")
    ax.legend()
    st.pyplot(fig)
except Exception as e:
    st.error(f"Regression error: {e}")

# Show results
st.subheader("Regression Results")
st.markdown(info)
