# regression_cli.py
import pandas as pd
from regression_models import *

def load_data():
    path = input("Enter CSV file path (default: data/synthetic_regression_data.csv): ").strip()
    if path == "":
        path = "data/synthetic_regression_data.csv"
    try:
        df = pd.read_csv(path)
        x = df['x'].values
        return df, x
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None, None

def cli_menu():
    df, x = load_data()
    if df is None:
        return
    while True:
        print("\nChoose regression type:")
        print("1. Linear (y = ax + b)           [column: y_linear]")
        print("2. Quadratic (y = ax^2 + bx + c)  [column: y_quadratic]")
        print("3. Logarithmic (y = a + b*ln(x))  [column: y_logarithmic]")
        print("4. Exponential (y = a*exp(bx))    [column: y_exponential]")
        print("5. Power (y = a*x^b)              [column: y_power]")
        print("6. General Power (y = a*b^x)      [column: y_power]")
        print("7. Rational (y = a + b/x)         [column: y_rational]")
        print("8. Load different CSV")
        print("9. Exit")
        choice = input("Enter choice (1-9): ").strip()
        if choice == "1":
            linear_regression(x, df['y_linear'].values)
        elif choice == "2":
            quadratic_regression(x, df['y_quadratic'].values)
        elif choice == "3":
            logarithmic_regression(x, df['y_logarithmic'].values)
        elif choice == "4":
            exponential_regression(x, df['y_exponential'].values)
        elif choice == "5":
            power_regression(x, df['y_power'].values)
        elif choice == "6":
            general_power_regression(x, df['y_power'].values)
        elif choice == "7":
            rational_regression(x, df['y_rational'].values)
        elif choice == "8":
            df, x = load_data()
            if df is None:
                return
        elif choice == "9":
            print("Exiting program.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    cli_menu()
