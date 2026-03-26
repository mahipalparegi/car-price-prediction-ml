import sys
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "price_model.pkl"
DATA_PATH = BASE_DIR / "car_price_dataset_usd.csv"

COMPANY_COL = "company_name"
MODEL_COL = "model"
YEAR_COL = "year"
CONDITION_COL = "condition"
KMS_COL = "km_driven"
FUEL_COL = "fuel_type"
TRANS_COL = "transmission"
ENGINE_CC_COL = "engine_cc"
POWER_COL = "max_power_bhp"
MILEAGE_COL = "mileage_kmpl"


def ask_choice(prompt, options):
    while True:
        print(f"\n{prompt}")
        for i, val in enumerate(options, start=1):
            print(f"{i}. {val}")
        choice = input("Enter option number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice. Try again.")


def ask_float(prompt, min_val=None, max_val=None):
    while True:
        val = input(prompt).strip()
        try:
            f = float(val)
            if min_val is not None and f < min_val:
                print(f"Value must be >= {min_val}.")
                continue
            if max_val is not None and f > max_val:
                print(f"Value must be <= {max_val}.")
                continue
            return f
        except ValueError:
            print("Enter a valid number.")


def main():
    if not MODEL_PATH.is_file():
        print("Model not found.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)

    if not DATA_PATH.is_file():
        print("Dataset not found.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    companies = sorted(df[COMPANY_COL].dropna().unique().tolist())
    fuel_types = sorted(df[FUEL_COL].dropna().unique().tolist())
    transmissions = sorted(df[TRANS_COL].dropna().unique().tolist())
    conditions = sorted(df[CONDITION_COL].dropna().unique().tolist())

    year_min = int(df[YEAR_COL].min())
    year_max = 2025
    kms_min, kms_max = df[KMS_COL].min(), df[KMS_COL].max()
    engine_min, engine_max = df[ENGINE_CC_COL].min(), df[ENGINE_CC_COL].max()
    power_min, power_max = df[POWER_COL].min(), df[POWER_COL].max()
    mileage_min, mileage_max = df[MILEAGE_COL].min(), df[MILEAGE_COL].max()

    print("\n=== Car Price Prediction ===")

    condition = ask_choice("Select car condition:", conditions)
    company = ask_choice("Select company:", companies)

    models = sorted(
        df.loc[df[COMPANY_COL] == company, MODEL_COL].dropna().unique().tolist()
    )
    if not models:
        models = sorted(df[MODEL_COL].dropna().unique().tolist())
    model_name = ask_choice(f"Select model for {company}:", models)

    year = ask_float(
        f"Enter manufacture year ({year_min} to {year_max}): ",
        min_val=year_min,
        max_val=year_max,
    )

    if "second" in condition.lower():
        kms = ask_float(
            f"Enter kilometers driven ({int(kms_min)} to {int(kms_max)}): ",
            min_val=kms_min,
            max_val=kms_max,
        )
    else:
        kms = 0.0

    fuel = ask_choice("Select fuel type:", fuel_types)
    transmission = ask_choice("Select transmission:", transmissions)

    engine_cc = ask_float(
        f"Enter engine CC ({int(engine_min)} to {int(engine_max)}): ",
        min_val=engine_min,
        max_val=engine_max,
    )

    power_bhp = ask_float(
        f"Enter max power (bhp) ({power_min} to {power_max}): ",
        min_val=power_min,
        max_val=power_max,
    )

    mileage = ask_float(
        f"Enter mileage kmpl ({mileage_min} to {mileage_max}): ",
        min_val=mileage_min,
        max_val=mileage_max,
    )

    row = {
        COMPANY_COL: company,
        MODEL_COL: model_name,
        YEAR_COL: year,
        CONDITION_COL: condition,
        KMS_COL: kms,
        FUEL_COL: fuel,
        TRANS_COL: transmission,
        ENGINE_CC_COL: engine_cc,
        POWER_COL: power_bhp,
        MILEAGE_COL: mileage,
    }

    input_df = pd.DataFrame([row])

    try:
        pred_usd = float(model.predict(input_df)[0])
    except Exception as e:
        print("Prediction error:", e)
        sys.exit(1)

    print("\n==============================")
    print(f"Estimated Price: {pred_usd:,.2f} USD")
    print("==============================\n")


if __name__ == "__main__":
    main()
