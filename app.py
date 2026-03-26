import math
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "car_price_dataset_usd.csv"
MODEL_PATH = BASE_DIR / "price_model.pkl"

NUMERIC_FEATURES = [
    "year",
    "km_driven",
    "engine_cc",
    "max_power_bhp",
    "mileage_kmpl",
]
CATEGORICAL_FEATURES = [
    "company_name",
    "model",
    "fuel_type",
    "transmission",
    "condition",
]

price_model = joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_dataset_metadata():
    df = pd.read_csv(DATA_PATH)
    companies = {
        company: sorted(group["model"].dropna().unique().tolist())
        for company, group in df.groupby("company_name")
    }
    ranges = {
        feature: {
            "min": float(df[feature].min()),
            "max": float(df[feature].max()),
        }
        for feature in NUMERIC_FEATURES
    }

    return {
        "companies": dict(sorted(companies.items())),
        "fuel_types": sorted(df["fuel_type"].dropna().unique().tolist()),
        "transmissions": sorted(df["transmission"].dropna().unique().tolist()),
        "conditions": sorted(df["condition"].dropna().unique().tolist()),
        "ranges": ranges,
        "training_rows": int(len(df)),
    }


def validate_car_payload(data: dict):
    errors = []

    for field in CATEGORICAL_FEATURES:
        if not str(data.get(field, "")).strip():
            errors.append(f"{field.replace('_', ' ').title()} is required")

    def check_num(name, *, min_val=None, max_val=None, allow_zero=True, required=True):
        label = name.replace("_", " ").title()
        raw = data.get(name, None)

        if raw in (None, ""):
            if required:
                errors.append(f"{label} is required")
            return

        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number")
            return

        if not math.isfinite(value):
            errors.append(f"{label} must be a finite number")
            return

        if not allow_zero and value == 0:
            errors.append(f"{label} cannot be zero")

        if min_val is not None and value < min_val:
            errors.append(f"{label} cannot be less than {min_val}")
        if max_val is not None and value > max_val:
            errors.append(f"{label} cannot be greater than {max_val}")

    check_num("year", min_val=1980, max_val=2025, allow_zero=False)

    condition = str(data.get("condition", "")).strip()

    if condition.lower() == "second hand":
        check_num("km_driven", min_val=0)

    check_num("engine_cc", min_val=600, allow_zero=False)
    check_num("max_power_bhp", min_val=30, allow_zero=False)
    check_num("mileage_kmpl", min_val=3, allow_zero=False)

    return errors


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/graph_visualizations/<path:filename>")
def chart(filename):
    return send_from_directory(BASE_DIR / "graph_visualizations", filename)


@app.get("/api/health")
def health():
    ready = DATA_PATH.is_file() and MODEL_PATH.is_file()
    return jsonify({"status": "ok" if ready else "not_ready", "model_ready": ready}), (
        200 if ready else 503
    )


@app.get("/api/metadata")
def metadata():
    return jsonify({"success": True, **get_dataset_metadata()})


@app.post("/api/predict")
@app.post("/predict")
def predict_price():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"success": False, "errors": ["A JSON object is required"]}), 400

        errors = validate_car_payload(data)
        if errors:
            return jsonify({"success": False, "errors": errors}), 400

        condition = str(data.get("condition", "")).strip()
        km_val = data.get("km_driven", 0)
        if not km_val or condition.lower() != "second hand":
            km_val = 0

        row = {
            "company_name": data.get("company_name"),
            "model": data.get("model"),
            "year": int(float(data.get("year"))),
            "km_driven": int(float(km_val)),
            "fuel_type": data.get("fuel_type"),
            "transmission": data.get("transmission"),
            "engine_cc": float(data.get("engine_cc")),
            "max_power_bhp": float(data.get("max_power_bhp")),
            "condition": condition,
            "mileage_kmpl": float(data.get("mileage_kmpl")),
        }

        df = pd.DataFrame([row])
        price = float(price_model.predict(df)[0])

        return jsonify(
            {
                "success": True,
                "predicted_price_usd": round(price, 2),
            }
        )

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"success": False, "error": "Prediction service unavailable"}), 500


if __name__ == "__main__":
    app.run(debug=False)
