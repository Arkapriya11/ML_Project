import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix

from model_pipeline_pm10 import add_pm10_features, split_by_location_time, get_xy


BASE_DIR = r"c:\Users\annam\OneDrive\Desktop\ML_Project"
PM25_PRED_PATH = os.path.join(BASE_DIR, "MODEL_RESULTS", "test_predictions.csv")
PM10_DATA_PATH = os.path.join(BASE_DIR, "FEATURE_ENGINEERED", "Master_Hourly_All_FE.csv")
PM10_MODEL_PATH = os.path.join(BASE_DIR, "MODEL_RESULTS_PM10", "xgboost_pm10.joblib")
OUT_DIR = os.path.join(BASE_DIR, "REQUESTED_GRAPHS", "CONFUSION_MATRICES")


def categorize_pm25(values: np.ndarray) -> np.ndarray:
    bins = [0, 30, 60, 90, 120, 250, np.inf]
    labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    clipped = np.clip(values, 0, None)
    return np.asarray(pd.cut(clipped, bins=bins, labels=labels, include_lowest=True).astype(str))


def categorize_pm10(values: np.ndarray) -> np.ndarray:
    bins = [0, 50, 100, 250, 350, 430, np.inf]
    labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    clipped = np.clip(values, 0, None)
    return np.asarray(pd.cut(clipped, bins=bins, labels=labels, include_lowest=True).astype(str))


def save_region_confusion(df: pd.DataFrame, pollutant: str, y_true_col: str, y_pred_col: str):
    categories = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    pollutant_dir = os.path.join(OUT_DIR, pollutant)
    os.makedirs(pollutant_dir, exist_ok=True)

    for region, part in df.groupby("Location"):
        y_true = part[y_true_col].to_numpy()
        y_pred = part[y_pred_col].to_numpy()

        if pollutant == "PM25":
            true_cat = categorize_pm25(y_true)
            pred_cat = categorize_pm25(y_pred)
        else:
            true_cat = categorize_pm10(y_true)
            pred_cat = categorize_pm10(y_pred)

        cm = confusion_matrix(true_cat, pred_cat, labels=categories)
        cm_df = pd.DataFrame(cm, index=categories, columns=categories)

        csv_path = os.path.join(pollutant_dir, f"{region}_confusion_matrix.csv")
        cm_df.to_csv(csv_path)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{pollutant} Confusion Matrix - {region}")
        plt.xlabel("Predicted Category")
        plt.ylabel("Actual Category")
        plt.tight_layout()
        png_path = os.path.join(pollutant_dir, f"{region}_confusion_matrix.png")
        plt.savefig(png_path, dpi=150)
        plt.close()


def build_pm10_test_predictions() -> pd.DataFrame:
    df = pd.read_csv(PM10_DATA_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime", "Location", "PM10"]).copy()

    df = add_pm10_features(df)
    req = ["PM10_lag1", "PM10_lag2", "PM10_lag3", "PM10_roll_mean_3", "PM10_roll_mean_6", "PM10_ema_6"]
    df = df.dropna(subset=req + ["PM10"]).copy()

    _, _, test_df = split_by_location_time(df)
    X_test, y_test = get_xy(test_df)

    model_bundle = joblib.load(PM10_MODEL_PATH)
    X_test_t = model_bundle["preprocessor"].transform(X_test)
    y_pred = model_bundle["model"].predict(X_test_t)

    out = test_df[["DateTime", "Location"]].copy()
    out["Actual_PM10"] = y_test
    out["Predicted_PM10"] = y_pred
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # PM2.5 region-wise matrix from saved PM2.5 predictions.
    pm25_df = pd.read_csv(PM25_PRED_PATH)
    pm25_use = pm25_df[["Location", "Actual", "XGBoost"]].dropna().copy()
    pm25_use = pm25_use.rename(columns={"Actual": "Actual_PM25", "XGBoost": "Predicted_PM25"})
    save_region_confusion(pm25_use, pollutant="PM25", y_true_col="Actual_PM25", y_pred_col="Predicted_PM25")

    # PM10 region-wise matrix from rebuilt PM10 test predictions.
    pm10_pred_df = build_pm10_test_predictions()
    save_region_confusion(pm10_pred_df, pollutant="PM10", y_true_col="Actual_PM10", y_pred_col="Predicted_PM10")

    print(f"Saved region-wise confusion matrices in: {OUT_DIR}")


if __name__ == "__main__":
    main()
