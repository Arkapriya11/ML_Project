import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


DATA_PATH = r'c:\Users\annam\OneDrive\Desktop\ML_Project\FEATURE_ENGINEERED\Master_Hourly_All_FE.csv'
RESULTS_DIR = r'c:\Users\annam\OneDrive\Desktop\ML_Project\MODEL_RESULTS_PM10'
TARGET_COL = 'PM10'
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_model(name, y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    y_safe = np.where(np.abs(y_true) < 1e-8, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / y_safe)) * 100)
    print(f"[{name}] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")
    return {"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def make_sequences(X, y, lookback=24):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i])
    if not X_seq:
        return np.empty((0, lookback, X.shape[1])), np.array([])
    return np.array(X_seq), np.array(y_seq)


def split_by_location_time(df, train_ratio=0.7, val_ratio=0.15):
    tr, va, te = [], [], []
    for loc, g in df.groupby("Location"):
        g = g.sort_values("DateTime")
        n = len(g)
        if n < 80:
            continue
        a = int(n * train_ratio)
        b = int(n * (train_ratio + val_ratio))
        tr.append(g.iloc[:a])
        va.append(g.iloc[a:b])
        te.append(g.iloc[b:])
    if not tr or not va or not te:
        raise ValueError("Insufficient rows for PM10 split.")
    return (
        pd.concat(tr).sort_values("DateTime"),
        pd.concat(va).sort_values("DateTime"),
        pd.concat(te).sort_values("DateTime"),
    )


def add_pm10_features(df):
    df = df.copy().sort_values(["Location", "DateTime"])
    gp = df.groupby("Location")[TARGET_COL]

    df["PM10_lag1"] = gp.shift(1)
    df["PM10_lag2"] = gp.shift(2)
    df["PM10_lag3"] = gp.shift(3)
    df["PM10_lag6"] = gp.shift(6)
    df["PM10_lag12"] = gp.shift(12)
    df["PM10_lag24"] = gp.shift(24)
    df["PM10_lag48"] = gp.shift(48)
    df["PM10_lag72"] = gp.shift(72)
    df["PM10_lag168"] = gp.shift(168)
    df["PM10_roll_mean_3"] = gp.shift(1).rolling(3, min_periods=3).mean().reset_index(level=0, drop=True)
    df["PM10_roll_mean_6"] = gp.shift(1).rolling(6, min_periods=6).mean().reset_index(level=0, drop=True)
    df["PM10_roll_mean_12"] = gp.shift(1).rolling(12, min_periods=6).mean().reset_index(level=0, drop=True)
    df["PM10_roll_mean_24"] = gp.shift(1).rolling(24, min_periods=6).mean().reset_index(level=0, drop=True)
    df["PM10_roll_std_3"] = gp.shift(1).rolling(3, min_periods=3).std().reset_index(level=0, drop=True)
    df["PM10_roll_std_6"] = gp.shift(1).rolling(6, min_periods=6).std().reset_index(level=0, drop=True)
    df["PM10_roll_std_12"] = gp.shift(1).rolling(12, min_periods=6).std().reset_index(level=0, drop=True)
    df["PM10_roll_std_24"] = gp.shift(1).rolling(24, min_periods=6).std().reset_index(level=0, drop=True)
    df["PM10_ema_6"] = gp.shift(1).ewm(span=6, adjust=False).mean().reset_index(level=0, drop=True)
    df["PM10_ema_12"] = gp.shift(1).ewm(span=12, adjust=False).mean().reset_index(level=0, drop=True)

    # Safe trend feature
    df["PM10_diff"] = df["PM10_lag1"] - df["PM10_lag2"]
    lag2_safe = df["PM10_lag2"].replace(0, np.nan)
    df["PM10_pct_change"] = (df["PM10_lag1"] - df["PM10_lag2"]) / lag2_safe

    if {"NOx", "PM10_lag1"}.issubset(df.columns):
        df["NOx_x_PM10lag1"] = df["NOx"] * df["PM10_lag1"]
    return df


def get_xy(df):
    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL, "DateTime"], errors="ignore").copy()
    return X, y


def build_preprocessor(X_sample):
    num_cols = X_sample.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in ["Location", "Frequency"] if c in X_sample.columns]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="drop"
    )


def run():
    df = pd.read_csv(DATA_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime", "Location", TARGET_COL]).copy()
    df = add_pm10_features(df)

    # Keep rows where PM10-target autoregressive features exist
    need = ["PM10_lag1", "PM10_lag2", "PM10_lag3", "PM10_roll_mean_3", "PM10_roll_mean_6", "PM10_ema_6"]
    df = df.dropna(subset=need + [TARGET_COL]).copy()

    train_df, val_df, test_df = split_by_location_time(df)
    y_test = test_df[TARGET_COL].values
    if np.var(y_test) <= 1e-12:
        raise ValueError("PM10 test target is constant; cannot evaluate robustly.")

    X_train, y_train = get_xy(train_df)
    X_val, y_val = get_xy(val_df)
    X_test, y_test = get_xy(test_df)

    X_train_full = pd.concat([X_train, X_val], ignore_index=True)
    y_train_full = np.concatenate([y_train, y_val])
    pre = build_preprocessor(X_train_full)

    results = []

    # Linear Regression
    lr = Pipeline([("prep", pre), ("model", LinearRegression())])
    lr.fit(X_train_full, y_train_full)
    p_lr = lr.predict(X_test)
    results.append(evaluate_model("Linear Regression", y_test, p_lr))
    joblib.dump(lr, os.path.join(RESULTS_DIR, "linear_regression_pm10.joblib"))

    # Random Forest tuning on validation
    rf_candidates = []
    for n_estimators in [600, 800]:
        for max_depth in [16, 20, 24]:
            for min_samples_leaf in [1, 2]:
                rf_try = Pipeline([("prep", pre), ("model", RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=max_depth, min_samples_split=2,
                    min_samples_leaf=min_samples_leaf, n_jobs=-1, random_state=42
                ))])
                rf_try.fit(X_train, y_train)
                pred_val = rf_try.predict(X_val)
                rmse_val = np.sqrt(mean_squared_error(y_val, pred_val))
                rf_candidates.append((rmse_val, n_estimators, max_depth, min_samples_leaf))
    rf_candidates.sort(key=lambda x: x[0])
    _, best_n, best_d, best_leaf = rf_candidates[0]
    print(f"[RF] Best params -> n_estimators={best_n}, max_depth={best_d}, min_samples_leaf={best_leaf}")
    rf = Pipeline([("prep", pre), ("model", RandomForestRegressor(
        n_estimators=best_n, max_depth=best_d, min_samples_split=2,
        min_samples_leaf=best_leaf, n_jobs=-1, random_state=42
    ))])
    rf.fit(X_train_full, y_train_full)
    p_rf = rf.predict(X_test)
    results.append(evaluate_model("Random Forest", y_test, p_rf))
    joblib.dump(rf, os.path.join(RESULTS_DIR, "random_forest_pm10.joblib"))

    # XGBoost tuning with early stopping (manual transformed matrix)
    num_cols = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in ["Location", "Frequency"] if c in X_train_full.columns]
    xgb_pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )
    Xtr_xgb = xgb_pre.fit_transform(X_train)
    Xva_xgb = xgb_pre.transform(X_val)
    Xtrf_xgb = xgb_pre.fit_transform(X_train_full)
    Xte_xgb = xgb_pre.transform(X_test)

    xgb_candidates = []
    for n_estimators in [700, 900, 1100]:
        for lr in [0.01, 0.02, 0.03]:
            for md in [4, 5, 6]:
                x_try = XGBRegressor(
                    n_estimators=n_estimators, learning_rate=lr, max_depth=md,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                    objective="reg:squarederror", random_state=42
                )
                x_try.fit(Xtr_xgb, y_train, eval_set=[(Xva_xgb, y_val)], verbose=False)
                pred_val = x_try.predict(Xva_xgb)
                rmse_val = np.sqrt(mean_squared_error(y_val, pred_val))
                xgb_candidates.append((rmse_val, n_estimators, lr, md))
    xgb_candidates.sort(key=lambda x: x[0])
    _, best_ne, best_lr, best_md = xgb_candidates[0]
    print(f"[XGB] Best params -> n_estimators={best_ne}, learning_rate={best_lr}, max_depth={best_md}")
    xgb = XGBRegressor(
        n_estimators=best_ne, learning_rate=best_lr, max_depth=best_md,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
        objective="reg:squarederror", random_state=42
    )
    xgb.fit(Xtrf_xgb, y_train_full, verbose=False)
    p_xgb = xgb.predict(Xte_xgb)
    results.append(evaluate_model("XGBoost", y_test, p_xgb))
    joblib.dump({"preprocessor": xgb_pre, "model": xgb}, os.path.join(RESULTS_DIR, "xgboost_pm10.joblib"))

    # LSTM (numeric features only)
    np.random.seed(42)
    tf.random.set_seed(42)
    num_cols = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
    imp = SimpleImputer(strategy="median")
    sx = StandardScaler()
    sy = StandardScaler()

    Xtr = sx.fit_transform(imp.fit_transform(X_train[num_cols]))
    Xva = sx.transform(imp.transform(X_val[num_cols]))
    Xte = sx.transform(imp.transform(X_test[num_cols]))
    ytr = sy.fit_transform(y_train.reshape(-1, 1)).flatten()
    yva = sy.transform(y_val.reshape(-1, 1)).flatten()

    lookback = 24
    Xtr_seq, ytr_seq = make_sequences(Xtr, ytr, lookback)
    Xva_seq, yva_seq = make_sequences(Xva, yva, lookback)
    Xte_seq, _ = make_sequences(Xte, y_test, lookback)
    yte_aligned = y_test[lookback:]

    if len(Xtr_seq) > 0 and len(Xva_seq) > 0 and len(Xte_seq) > 0:
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(Xtr_seq.shape[1], Xtr_seq.shape[2])),
            Dropout(0.2),
            LSTM(32),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        model.fit(
            Xtr_seq, ytr_seq,
            validation_data=(Xva_seq, yva_seq),
            epochs=80,
            batch_size=32,
            shuffle=False,
            verbose=0,
            callbacks=[es]
        )
        pred_sc = model.predict(Xte_seq, verbose=0).flatten()
        p_lstm = sy.inverse_transform(pred_sc.reshape(-1, 1)).flatten()
        results.append(evaluate_model("LSTM", yte_aligned, p_lstm))
        model.save(os.path.join(RESULTS_DIR, "lstm_pm10.keras"))
    else:
        print("[WARN] LSTM skipped: insufficient sequence rows.")

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(RESULTS_DIR, "metrics_pm10.csv"), index=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=res, x="Model", y="RMSE")
    plt.xticks(rotation=30, ha="right")
    plt.title("PM10 Model RMSE Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "rmse_comparison_pm10.png"))
    plt.close()

    print(f"\nSaved PM10 results in: {RESULTS_DIR}")


if __name__ == "__main__":
    run()
