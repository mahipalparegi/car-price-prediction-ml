from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "car_price_dataset_usd.csv"
MODEL_PATH = BASE_DIR / "price_model.pkl"
OUTPUT_FOLDER = BASE_DIR / "graph_visualizations"


def save_figure(filename: str):
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / filename, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def main():
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    missing_files = [path.name for path in (DATA_PATH, MODEL_PATH) if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")

    df = pd.read_csv(DATA_PATH)
    model_pipeline = joblib.load(MODEL_PATH)
    print("Data and model loaded successfully.")

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
    target_col = "price_usd"

    required_cols = numeric_features + categorical_features + [target_col]
    df_clean = df.dropna(subset=required_cols).copy()

    X = df_clean[numeric_features + categorical_features]
    y = df_clean[target_col]

    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_pred = model_pipeline.predict(X_val)

    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 7)

    filename = "1_actual_vs_predicted.png"
    plt.figure()
    sns.scatterplot(x=y_val, y=y_pred, alpha=0.6, color="#4f46e5")
    min_val, max_val = min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="red",
        linestyle="--",
        linewidth=2,
    )
    plt.title(
        f"Actual vs Predicted Price (R2: {r2_score(y_val, y_pred):.2f})",
        fontsize=14,
    )
    plt.xlabel("Actual Price (USD)")
    plt.ylabel("Predicted Price (USD)")
    save_figure(filename)

    filename = "2_residual_distribution.png"
    plt.figure()
    sns.histplot(y_val - y_pred, kde=True, color="#0ea5e9", bins=30)
    plt.axvline(x=0, color="red", linestyle="--", linewidth=2)
    plt.title("Distribution of Prediction Errors (Residuals)", fontsize=14)
    plt.xlabel("Error (USD)")
    save_figure(filename)

    filename = "3_feature_importance.png"
    rf_model = model_pipeline.named_steps["rf"]
    preprocessor = model_pipeline.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    importances = rf_model.feature_importances_
    feat_imp_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    )
    feat_imp_df["Feature"] = (
        feat_imp_df["Feature"]
        .str.replace("cat__", "", regex=False)
        .str.replace("num__", "", regex=False)
    )
    top_10 = feat_imp_df.sort_values(by="Importance", ascending=False).head(10)

    plt.figure()
    sns.barplot(
        data=top_10,
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="viridis",
        legend=False,
    )
    plt.title("Top 10 Features Influencing Car Price", fontsize=14)
    save_figure(filename)

    filename = "4_correlation_heatmap.png"
    plt.figure(figsize=(10, 8))
    corr_matrix = df_clean[numeric_features + [target_col]].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix", fontsize=14)
    save_figure(filename)

    filename = "5_company_avg_price.png"
    plt.figure(figsize=(14, 7))
    avg_price = (
        df_clean.groupby("company_name")[target_col].mean().sort_values(ascending=False)
    )
    sns.barplot(
        x=avg_price.index,
        y=avg_price.values,
        hue=avg_price.index,
        palette="magma",
        legend=False,
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("Average Car Price by Company", fontsize=16)
    plt.ylabel("Average Price (USD)")
    save_figure(filename)

    filename = "6_year_vs_price.png"
    plt.figure()
    sns.scatterplot(
        data=df_clean,
        x="year",
        y=target_col,
        hue="condition",
        alpha=0.6,
        palette="deep",
    )
    plt.title("Car Price vs. Year of Manufacture", fontsize=16)
    save_figure(filename)

    filename = "7_price_vs_mileage_brandnew.png"
    brand_new_df = df_clean[df_clean["condition"] == "Brand New"]
    if not brand_new_df.empty:
        plt.figure()
        sns.scatterplot(
            data=brand_new_df,
            x="mileage_kmpl",
            y=target_col,
            color="green",
            alpha=0.7,
        )
        plt.title("Price vs. Mileage (Brand New Cars)", fontsize=16)
        save_figure(filename)

    filename = "8_price_vs_km_secondhand.png"
    used_df = df_clean[df_clean["condition"] == "Second Hand"]
    if not used_df.empty:
        plt.figure()
        sns.scatterplot(
            data=used_df,
            x="km_driven",
            y=target_col,
            color="orange",
            alpha=0.6,
        )
        plt.title("Price vs. Kilometers Driven (Second Hand Cars)", fontsize=16)
        save_figure(filename)

    filename = "9_price_distribution.png"
    plt.figure()
    sns.histplot(df_clean[target_col], kde=True, color="purple", bins=40)
    plt.title("Overall Price Distribution", fontsize=16)
    save_figure(filename)

    filename = "10_price_by_fuel_type.png"
    plt.figure()
    sns.boxplot(
        data=df_clean,
        x="fuel_type",
        y=target_col,
        hue="fuel_type",
        palette="Set2",
        legend=False,
    )
    plt.title("Price Distribution by Fuel Type", fontsize=16)
    save_figure(filename)

    print("\nAll visualizations generated successfully!")


if __name__ == "__main__":
    main()