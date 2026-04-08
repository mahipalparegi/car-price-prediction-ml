# fuel_efficiency_user_predict.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np


DATA_FILE = "fuel_efficiency_dataset.csv"


def load_and_train_model():
    # Load dataset
    df = pd.read_csv(DATA_FILE)

    # Features and target
    target_col = "fuel_efficiency_kmpl"
    feature_cols = [
        "brand",
        "model",
        "engine_cc",
        "max_power_bhp",
        "kerb_weight_kg",
        "fuel_type",
        "transmission",
        "kilometers_driven",
    ]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # One-hot encoding for categorical variables
    X_encoded = pd.get_dummies(
        X, columns=["brand", "model", "fuel_type", "transmission"], drop_first=True
    )

    # Train-test split (we mainly care about having a trained model)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Keep the column structure for future encoding
    feature_columns = X_encoded.columns

    return model, feature_columns, df


def choose_from_list(options, prompt):
    """
    Show a numbered menu for the user to choose from a list.
    Returns the chosen element.
    """
    while True:
        print("\n" + prompt)
        for i, opt in enumerate(options, start=1):
            print(f"{i}. {opt}")
        choice = input("Enter number: ").strip()
        if not choice.isdigit():
            print("Invalid input. Please enter a number.")
            continue
        idx = int(choice)
        if 1 <= idx <= len(options):
            return options[idx - 1]
        else:
            print("Choice out of range. Try again.")


def get_float_input(prompt):
    while True:
        value = input(prompt).strip()
        try:
            return float(value)
        except ValueError:
            print("Invalid number. Please try again.")


def get_int_input(prompt):
    while True:
        value = input(prompt).strip()
        try:
            return int(value)
        except ValueError:
            print("Invalid integer. Please try again.")


def main():
    print("Loading data and training Random Forest model...")
    model, feature_columns, df = load_and_train_model()
    print("Model ready!\n")

    # Precompute available options from dataset
    brands = sorted(df["brand"].unique())
    fuel_types = sorted(df["fuel_type"].unique())
    transmissions = sorted(df["transmission"].unique())

    while True:
        print("\n=== Fuel Efficiency Prediction ===")

        # 1. Choose brand
        chosen_brand = choose_from_list(brands, "Select a car brand:")

        # 2. Show models for that brand
        models_for_brand = sorted(
            df[df["brand"] == chosen_brand]["model"].unique()
        )
        chosen_model = choose_from_list(models_for_brand, f"Select a model for {chosen_brand}:")

        # 3. Choose fuel type from existing dataset
        chosen_fuel = choose_from_list(fuel_types, "Select fuel type:")

        # 4. Choose transmission from existing dataset
        chosen_transmission = choose_from_list(transmissions, "Select transmission type:")

        # 5. Numeric inputs
        print("\nEnter numeric details (you can approximate):")
        engine_cc = get_int_input("Engine CC (e.g., 1200, 1500, 2000): ")
        max_power_bhp = get_float_input("Max Power (bhp) (e.g., 80, 100, 150): ")
        kerb_weight_kg = get_int_input("Kerb Weight (kg) (e.g., 900, 1200, 1500): ")
        kilometers_driven = get_int_input("Kilometers Driven (e.g., 10000, 45000, 120000): ")

        # Build a single-row DataFrame for prediction
        user_row = pd.DataFrame(
            {
                "brand": [chosen_brand],
                "model": [chosen_model],
                "engine_cc": [engine_cc],
                "max_power_bhp": [max_power_bhp],
                "kerb_weight_kg": [kerb_weight_kg],
                "fuel_type": [chosen_fuel],
                "transmission": [chosen_transmission],
                "kilometers_driven": [kilometers_driven],
            }
        )

        # One-hot encode with the same logic as training
        user_encoded = pd.get_dummies(
            user_row,
            columns=["brand", "model", "fuel_type", "transmission"],
            drop_first=True,
        )

        # Align columns with training data, fill missing with 0
        user_encoded = user_encoded.reindex(columns=feature_columns, fill_value=0)

        # Predict
        predicted_eff = model.predict(user_encoded)[0]

        print("\n=== Prediction Result ===")
        print(f"Car: {chosen_brand} {chosen_model}")
        print(f"Fuel Type: {chosen_fuel}, Transmission: {chosen_transmission}")
        print(f"Predicted Fuel Efficiency: {predicted_eff:.2f} km/l")

        # Ask to continue or exit
        cont = input("\nDo you want to predict for another car? (y/n): ").strip().lower()
        if cont != "y":
            print("Exiting. Goodbye!")
            break


if __name__ == "__main__":
    main()
