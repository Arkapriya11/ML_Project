import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib


def main():
    out_dir = r"c:\Users\annam\OneDrive\Desktop\ML_Project\REQUESTED_GRAPHS"
    os.makedirs(out_dir, exist_ok=True)

    models = ["Linear", "RF", "XGB", "LSTM"]
    rmse = [17.04, 4.36, 4.11, 9.74]
    r2 = [-0.24, 0.918, 0.927, 0.58]

    plt.figure()
    plt.bar(models, rmse)
    plt.title("RMSE Comparison (PM2.5)")
    plt.xlabel("Models")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rmse_comparison_pm25.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.bar(models, r2)
    plt.title("R² Score Comparison (PM2.5)")
    plt.xlabel("Models")
    plt.ylabel("R²")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "r2_comparison_pm25.png"), dpi=150)
    plt.close()

    pm25_rmse = [4.11, 4.36]
    pm10_rmse = [16.81, 18.83]
    labels = ["XGB", "RF"]
    x = range(len(labels))

    plt.figure()
    plt.bar(x, pm25_rmse, width=0.4, label="PM2.5")
    plt.bar([i + 0.4 for i in x], pm10_rmse, width=0.4, label="PM10")
    plt.xticks([i + 0.2 for i in x], labels)
    plt.title("PM2.5 vs PM10 RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pm25_vs_pm10_rmse.png"), dpi=150)
    plt.close()

    # For your final plot, y_test/y_pred were not defined in the snippet.
    # We use saved test predictions and plot Actual vs XGBoost for first 100 points.
    pred_csv = r"c:\Users\annam\OneDrive\Desktop\ML_Project\MODEL_RESULTS\test_predictions.csv"
    if os.path.exists(pred_csv):
        df = pd.read_csv(pred_csv)
        y_test = df["Actual"]
        y_pred = df["XGBoost"]

        plt.figure()
        plt.plot(y_test[:100], label="Actual")
        plt.plot(y_pred[:100], label="Predicted")
        plt.legend()
        plt.title("Actual vs Predicted PM2.5")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "actual_vs_predicted_pm25.png"), dpi=150)
        plt.close()

    print(f"Saved graphs to: {out_dir}")

    # PM10: build y_test_pm10 and y_pred_pm10 from saved PM10 model artifacts.
    try:
        from model_pipeline_pm10 import add_pm10_features, split_by_location_time, get_xy

        pm10_data_path = r"c:\Users\annam\OneDrive\Desktop\ML_Project\FEATURE_ENGINEERED\Master_Hourly_All_FE.csv"
        pm10_model_path = r"c:\Users\annam\OneDrive\Desktop\ML_Project\MODEL_RESULTS_PM10\xgboost_pm10.joblib"

        pm10_df = pd.read_csv(pm10_data_path)
        if "Unnamed: 0" in pm10_df.columns:
            pm10_df = pm10_df.drop(columns=["Unnamed: 0"])
        pm10_df["DateTime"] = pd.to_datetime(pm10_df["DateTime"], errors="coerce")
        pm10_df = pm10_df.dropna(subset=["DateTime", "Location", "PM10"]).copy()
        pm10_df = add_pm10_features(pm10_df)
        req = ["PM10_lag1", "PM10_lag2", "PM10_lag3", "PM10_roll_mean_3", "PM10_roll_mean_6", "PM10_ema_6"]
        pm10_df = pm10_df.dropna(subset=req + ["PM10"]).copy()

        _, _, test_pm10_df = split_by_location_time(pm10_df)
        X_test_pm10, y_test_pm10 = get_xy(test_pm10_df)

        xgb_bundle = joblib.load(pm10_model_path)
        X_test_transformed = xgb_bundle["preprocessor"].transform(X_test_pm10)
        y_pred_pm10 = xgb_bundle["model"].predict(X_test_transformed)

        plt.figure()
        plt.plot(y_test_pm10[:100], label="Actual PM10")
        plt.plot(y_pred_pm10[:100], label="Predicted PM10")
        plt.title("Actual vs Predicted PM10")
        plt.xlabel("Time")
        plt.ylabel("PM10")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "actual_vs_predicted_pm10.png"), dpi=150)
        plt.close()
    except Exception as exc:
        print(f"PM10 plot generation skipped: {exc}")

    # MAE charts for each output variable from saved metrics.
    metrics_pm25_path = r"c:\Users\annam\OneDrive\Desktop\ML_Project\MODEL_RESULTS\metrics_final.csv"
    metrics_pm10_path = r"c:\Users\annam\OneDrive\Desktop\ML_Project\MODEL_RESULTS_PM10\metrics_pm10.csv"

    if os.path.exists(metrics_pm25_path):
        m25 = pd.read_csv(metrics_pm25_path)
        if {"Model", "MAE"}.issubset(m25.columns):
            plt.figure(figsize=(10, 5))
            plt.bar(m25["Model"], m25["MAE"])
            plt.title("MAE Comparison (PM2.5)")
            plt.xlabel("Models")
            plt.ylabel("MAE")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "mae_comparison_pm25.png"), dpi=150)
            plt.close()

    if os.path.exists(metrics_pm10_path):
        m10 = pd.read_csv(metrics_pm10_path)
        if {"Model", "MAE"}.issubset(m10.columns):
            plt.figure(figsize=(8, 5))
            plt.bar(m10["Model"], m10["MAE"])
            plt.title("MAE Comparison (PM10)")
            plt.xlabel("Models")
            plt.ylabel("MAE")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "mae_comparison_pm10.png"), dpi=150)
            plt.close()

            # Combined MAE chart for shared models only.
            shared_models = [m for m in ["Linear Regression", "Random Forest", "XGBoost", "LSTM"] if m in set(m25["Model"]) and m in set(m10["Model"])] if os.path.exists(metrics_pm25_path) else []
            if shared_models:
                pm25_mae = [float(m25.loc[m25["Model"] == m, "MAE"].iloc[0]) for m in shared_models]
                pm10_mae = [float(m10.loc[m10["Model"] == m, "MAE"].iloc[0]) for m in shared_models]
                x = range(len(shared_models))

                plt.figure(figsize=(9, 5))
                plt.bar(x, pm25_mae, width=0.4, label="PM2.5")
                plt.bar([i + 0.4 for i in x], pm10_mae, width=0.4, label="PM10")
                plt.xticks([i + 0.2 for i in x], shared_models, rotation=20, ha="right")
                plt.title("MAE: PM2.5 vs PM10 (Shared Models)")
                plt.ylabel("MAE")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "mae_pm25_vs_pm10_shared_models.png"), dpi=150)
                plt.close()

    # R2 comparison chart for PM2.5 vs PM10 for the 4 common models.
    if os.path.exists(metrics_pm25_path) and os.path.exists(metrics_pm10_path):
        m25 = pd.read_csv(metrics_pm25_path)
        m10 = pd.read_csv(metrics_pm10_path)
        required_cols = {"Model", "R2"}
        if required_cols.issubset(m25.columns) and required_cols.issubset(m10.columns):
            models_4 = ["Linear Regression", "Random Forest", "XGBoost", "LSTM"]
            available = [m for m in models_4 if m in set(m25["Model"]) and m in set(m10["Model"])]
            if available:
                pm25_r2 = [float(m25.loc[m25["Model"] == m, "R2"].iloc[0]) for m in available]
                pm10_r2 = [float(m10.loc[m10["Model"] == m, "R2"].iloc[0]) for m in available]
                x = range(len(available))

                plt.figure(figsize=(9, 5))
                plt.bar(x, pm25_r2, width=0.4, label="PM2.5")
                plt.bar([i + 0.4 for i in x], pm10_r2, width=0.4, label="PM10")
                plt.xticks([i + 0.2 for i in x], available, rotation=20, ha="right")
                plt.title("R² Comparison: PM2.5 vs PM10 (4 Models)")
                plt.ylabel("R²")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "r2_pm25_vs_pm10_4models.png"), dpi=150)
                plt.close()


if __name__ == "__main__":
    main()
