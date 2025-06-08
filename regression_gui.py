# regression_gui.py
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from regression_models import *

def run_selected_regression(selected, df, x):
    if selected == "Linear":
        linear_regression(x, df['y_linear'].values)
    elif selected == "Quadratic":
        quadratic_regression(x, df['y_quadratic'].values)
    elif selected == "Logarithmic":
        logarithmic_regression(x, df['y_logarithmic'].values)
    elif selected == "Exponential":
        exponential_regression(x, df['y_exponential'].values)
    elif selected == "Power":
        power_regression(x, df['y_power'].values)
    elif selected == "General Power":
        general_power_regression(x, df['y_power'].values)
    elif selected == "Rational":
        rational_regression(x, df['y_rational'].values)

def launch_gui():
    df = pd.read_csv("data/synthetic_regression_data.csv")
    x = df['x'].values

    root = tk.Tk()
    root.title("Regression Model Selector")

    ttk.Label(root, text="Select regression type:").pack(padx=10, pady=5)
    reg_types = ["Linear", "Quadratic", "Logarithmic", "Exponential", "Power", "General Power", "Rational"]
    combo = ttk.Combobox(root, values=reg_types, state="readonly")
    combo.set(reg_types[0])
    combo.pack(padx=10, pady=5)

    def on_run():
        sel = combo.get()
        run_selected_regression(sel, df, x)
        messagebox.showinfo("Done", f"{sel} regression completed!\nSee terminal and output folder.")

    ttk.Button(root, text="Run Regression", command=on_run).pack(padx=10, pady=10)
    root.mainloop()

if __name__ == "__main__":
    launch_gui()
