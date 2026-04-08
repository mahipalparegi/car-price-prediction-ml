# fuel_efficiency_visualization_save.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

DATA_FILE = "fuel_efficiency_dataset.csv"
SAVE_DIR = "graph_visualizations"

# Create save directory if not exists
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"Created directory: {SAVE_DIR}")


# 1. Load dataset
df = pd.read_csv(DATA_FILE)

# 2. Define features and target
target_col = "fuel_efficiency_kmpl"
feature_cols = [
    "brand", "model", "engine_cc", "max_power_bhp", "kerb_weight_kg",
    "fuel_type", "transmission", "kilometers_driven"
]

X = df[feature_cols].copy()
y = df[target_col].copy()

# 3. Encode categorical variables
X_encoded = pd.get_dummies(X, columns=["brand", "model", "fuel_type", "transmission"], drop_first=True)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# 5. Train Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# 6. Evaluation
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nRandom Forest Results")
print("------------------------")
print(f"MAE : {mae:.2f}")
print(f"R²  : {r2:.2f}")

# -----------------------------
# SAVE PLOTS IN graph_visualizations/
# -----------------------------


# === (A) Actual vs Predicted ===
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Fuel Efficiency (km/l)")
plt.ylabel("Predicted Fuel Efficiency (km/l)")
plt.title("Actual vs Predicted Fuel Efficiency")

# Diagonal reference line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linewidth=2)

# Save
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/actual_vs_predicted.png")
plt.show()
print("Saved: actual_vs_predicted.png")


# === (B) Feature Importance ===
importances = rf_model.feature_importances_
indices = np.argsort(importances)
feature_names = np.array(X_encoded.columns)

plt.figure(figsize=(8, 12))
plt.barh(range(len(importances)), importances[indices])
plt.yticks(range(len(importances)), feature_names[indices])
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/feature_importance.png")
plt.show()
print("Saved: feature_importance.png")


# === (C) Engine CC vs Fuel Efficiency ===
plt.figure()
plt.scatter(df["engine_cc"], df[target_col], alpha=0.5)
plt.xlabel("Engine CC")
plt.ylabel("Fuel Efficiency (km/l)")
plt.title("Engine CC vs Fuel Efficiency")

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/enginecc_vs_efficiency.png")
plt.show()
print("Saved: enginecc_vs_efficiency.png")

print("\nAll graphs saved successfully inside the 'graph_visualizations' folder!")
