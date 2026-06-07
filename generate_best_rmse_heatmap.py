import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error

from model_pipeline_pm10 import add_pm10_features, split_by_location_time, get_xy


BASE_DIR = r"c:\Users\annam\OneDrive\Desktop\ML_Project"
OUT_DIR = os.path.join(BASE_DIR, "REQUESTED_GRAPHS")

PM25_PRED_PATH = os.path.join(BASE_DIR, "MODEL_RESULTS", "test_predictions.csv")
PM10_DATA_PATH = os.path.join(BASE_DIR, "FEATURE_ENGINEERED", "Master_Hourly_All_FE.csv")
PM10_LR_PATH = os.path.join(BASE_DIR, "MODEL_RESULTS_PM10", "linear_regression_pm10.joblib")
PM10_RF_PATH = os.path.join(BASE_DIR, "MODEL_RESULTS_PM10", "random_forest_pm10.joblib")
PM10_XGB_PATH = os.path.join(BASE_DIR, "MODEL_RESULTS_PM10", "xgboost_pm10.joblib")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_pm25_best_by_region():
    df = pd.read_csv(PM25_PRED_PATH)
    model_cols = [c for c in df.columns if c not in ["DateTime", "Location", "Actual"]]
    out = []
    for region, part in df.groupby("Location"):
        y_true = part["Actual"].to_numpy()
        best_model = None
        best_rmse = float("inf")
        for m in model_cols:
            pred = pd.to_numeric(part[m], errors="coerce")
            valid = pred.notna().to_numpy()
            if valid.sum() == 0:
                continue
            score = rmse(y_true[valid], pred.to_numpy()[valid])
            if score < best_rmse:
                best_rmse = score
                best_model = m
        out.append({"Region": region, "Target": "PM2.5", "BestModel": best_model, "BestRMSE": best_rmse})
    return pd.DataFrame(out)


def get_pm10_best_by_region():
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

    lr = joblib.load(PM10_LR_PATH)
    rf = joblib.load(PM10_RF_PATH)
    xgb_bundle = joblib.load(PM10_XGB_PATH)

    pred_lr = lr.predict(X_test)
    pred_rf = rf.predict(X_test)
    pred_xgb = xgb_bundle["model"].predict(xgb_bundle["preprocessor"].transform(X_test))

    pred_df = pd.DataFrame({
        "Location": test_df["Location"].values,
        "Actual": y_test,
        "Linear Regression": pred_lr,
        "Random Forest": pred_rf,
        "XGBoost": pred_xgb,
    })

    out = []
    for region, part in pred_df.groupby("Location"):
        y_true = part["Actual"].to_numpy()
        model_scores = {
            "Linear Regression": rmse(y_true, part["Linear Regression"].to_numpy()),
            "Random Forest": rmse(y_true, part["Random Forest"].to_numpy()),
            "XGBoost": rmse(y_true, part["XGBoost"].to_numpy()),
        }
        best_model = min(model_scores, key=model_scores.get)
        out.append({
            "Region": region,
            "Target": "PM10",
            "BestModel": best_model,
            "BestRMSE": model_scores[best_model],
        })
    return pd.DataFrame(out)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pm25_best = get_pm25_best_by_region()
    pm10_best = get_pm10_best_by_region()
    best_all = pd.concat([pm25_best, pm10_best], ignore_index=True)

    # Save detailed table.
    table_path = os.path.join(OUT_DIR, "best_rmse_by_region_and_target.csv")
    best_all.to_csv(table_path, index=False)

    regions = sorted(best_all["Region"].unique().tolist())
    targets = ["PM2.5", "PM10"]

    value_mat = pd.DataFrame(index=regions, columns=targets, dtype=float)
    label_mat = pd.DataFrame(index=regions, columns=targets, dtype=object)

    for _, row in best_all.iterrows():
        r = row["Region"]
        t = row["Target"]
        value_mat.loc[r, t] = float(row["BestRMSE"])
        label_mat.loc[r, t] = f'{row["BestRMSE"]:.2f}\n({row["BestModel"]})'

    plt.figure(figsize=(8, 6))
    sns.heatmap(value_mat, annot=label_mat.values, fmt="", cmap="YlOrRd", cbar_kws={"label": "Best RMSE"})
    plt.title("Best Test RMSE by Region and Target")
    plt.xlabel("Target")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "best_rmse_heatmap_region_target.png"), dpi=150)
    plt.close()

    print(f"Saved: {table_path}")
    print(f"Saved: {os.path.join(OUT_DIR, 'best_rmse_heatmap_region_target.png')}")


if __name__ == "__main__":
    main()
