import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "car_price_dataset_usd.csv"
MODEL_PATH = BASE_DIR / "price_model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"
TARGET_COL = "price_usd"


def build_model(df: pd.DataFrame, *, n_iter: int = 25, cv: int = 3):
    numeric_features = [
        "year",
        "km_driven",
        "engine_cc",
        "max_power_bhp",
        "mileage_kmpl",
    ]
    categorical_features = [
        "company_name",
        "model",
        "fuel_type",
        "transmission",
        "condition",
    ]

    required_cols = numeric_features + categorical_features + [TARGET_COL]
    df = df.dropna(subset=required_cols)

    X = df[numeric_features + categorical_features]
    y = df[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("rf", rf),
        ]
    )

    param_dist = {
        "rf__n_estimators": [100, 200, 300, 400, 500],
        "rf__max_depth": [None, 10, 15, 20, 30],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
        "rf__max_features": ["sqrt", "log2"],
        "rf__bootstrap": [True, False],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
        verbose=0,
        random_state=42,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)

    train_score = r2_score(y_train, y_train_pred)
    val_score = r2_score(y_val, y_val_pred)
    metrics = {
        "model": "RandomForestRegressor",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "rows_total": int(len(df)),
        "rows_train": int(len(X_train)),
        "rows_validation": int(len(X_val)),
        "train_r2": round(float(train_score), 5),
        "validation_r2": round(float(val_score), 5),
        "validation_mae_usd": round(float(mean_absolute_error(y_val, y_val_pred)), 2),
        "validation_rmse_usd": round(
            math.sqrt(float(mean_squared_error(y_val, y_val_pred))), 2
        ),
        "best_cv_r2": round(float(search.best_score_), 5),
        "best_params": search.best_params_,
        "features": numeric_features + categorical_features,
        "target": TARGET_COL,
    }

    print(f"Best Train R^2: {metrics['train_r2']:.5f}")
    print(f"Best Val   R^2: {metrics['validation_r2']:.5f}")
    print(f"Validation MAE: ${metrics['validation_mae_usd']:,.2f}")
    print(f"Validation RMSE: ${metrics['validation_rmse_usd']:,.2f}")

    return best_model, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train the car price prediction model.")
    parser.add_argument(
        "--n-iter",
        type=int,
        default=25,
        help="Number of randomized hyperparameter combinations (default: 25).",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Number of cross-validation folds (default: 3).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_iter < 1 or args.cv < 2:
        raise ValueError("--n-iter must be at least 1 and --cv must be at least 2")

    df = pd.read_csv(DATA_PATH)
    model, metrics = build_model(df, n_iter=args.n_iter, cv=args.cv)
    joblib.dump(model, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved tuned model to {MODEL_PATH.name}")
    print(f"Saved evaluation metrics to {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
