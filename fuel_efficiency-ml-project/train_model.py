# fuel_efficiency_rf_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Load dataset
# Make sure the CSV is in the same folder as this script
data = pd.read_csv("fuel_efficiency_dataset.csv")

# 2. Define features (X) and target (y)
target_col = "fuel_efficiency_kmpl"

# All other columns except target are features
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

X = data[feature_cols].copy()
y = data[target_col].copy()

# 3. One-hot encode categorical features
# RandomForest can't use string categories directly
X_encoded = pd.get_dummies(X, columns=["brand", "model", "fuel_type", "transmission"], drop_first=True)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# 5. Define Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=200,      # number of trees
    max_depth=None,       # let trees grow fully (can tune later)
    random_state=42,
    n_jobs=-1             # use all CPU cores
)

# 6. Train the model
rf_model.fit(X_train, y_train)

# 7. Evaluate on test set
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Random Forest Regression Results")
print("--------------------------------")
print(f"MAE  (Mean Absolute Error): {mae:.3f} km/l")
print(f"R² score                 : {r2:.3f}")

# 8. Save the trained model to a file
joblib.dump(rf_model, "efficiency_model.pkl")
print("\nModel saved to efficiency_model.pkl")

# 9. Example: predict for a single new car
example_car = {
    "brand": ["Toyota"],
    "model": ["Corolla"],
    "engine_cc": [1500],
    "max_power_bhp": [120.0],
    "kerb_weight_kg": [1300],
    "fuel_type": ["Petrol"],
    "transmission": ["Manual"],
    "kilometers_driven": [45000],
}

example_df = pd.DataFrame(example_car)

# One-hot encode with same columns as training
example_encoded = pd.get_dummies(
    example_df, columns=["brand", "model", "fuel_type", "transmission"], drop_first=True
)

# Align columns with training data (add missing columns = 0)
example_encoded = example_encoded.reindex(columns=X_encoded.columns, fill_value=0)

predicted_eff = rf_model.predict(example_encoded)[0]
print(f"\nPredicted fuel efficiency for example car: {predicted_eff:.2f} km/l")
