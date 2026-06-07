import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# CONFIGURATION
DATA_PATH = r'c:\Users\annam\OneDrive\Desktop\ML_Project\FEATURE_ENGINEERED\Master_Hourly_All_FE.csv'
RESULTS_DIR = r'c:\Users\annam\OneDrive\Desktop\ML_Project\MODEL_RESULTS'
os.makedirs(RESULTS_DIR, exist_ok=True)

def split_by_location_time(df, train_ratio=0.7, val_ratio=0.15):
    train_parts, val_parts, test_parts = [], [], []
    for loc, part in df.groupby('Location'):
        part = part.sort_values('DateTime').copy()
        n = len(part)
        if n < 50:
            print(f"[WARN] Skipping location {loc} (too few rows: {n})")
            continue
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        train_parts.append(part.iloc[:train_end])
        val_parts.append(part.iloc[train_end:val_end])
        test_parts.append(part.iloc[val_end:])

    if not train_parts or not val_parts or not test_parts:
        raise ValueError("Split failed. Check location counts and data completeness.")

    train_df = pd.concat(train_parts).sort_values('DateTime')
    val_df = pd.concat(val_parts).sort_values('DateTime')
    test_df = pd.concat(test_parts).sort_values('DateTime')
    return train_df, val_df, test_df


def load_and_split():
    df = pd.read_csv(DATA_PATH)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    required = {'DateTime', 'Location', 'PM2.5'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df = df.dropna(subset=['DateTime', 'PM2.5', 'Location']).sort_values(['Location', 'DateTime'])
    return split_by_location_time(df)


def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    y_true_safe = np.where(np.abs(y_true) < 1e-8, np.nan, y_true)
    mape = np.nanmean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    print(f"[{name}] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")
    return {'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}


def build_features(df):
    feature_drop = ['PM2.5', 'DateTime']
    X = df.drop(columns=feature_drop, errors='ignore').copy()
    y = df['PM2.5'].values
    return X, y


def build_preprocessor(X_sample):
    numeric_features = X_sample.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in ['Location', 'Frequency'] if c in X_sample.columns]

    return ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )


def clean_training_rows(df):
    df = df.copy()
    # Drop rows with excessive missingness (>40%)
    max_missing = int(np.floor(0.4 * df.shape[1]))
    df = df[df.isna().sum(axis=1) <= max_missing].copy()

    # IQR outlier filtering on target
    q1 = df['PM2.5'].quantile(0.25)
    q3 = df['PM2.5'].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    df = df[(df['PM2.5'] >= low) & (df['PM2.5'] <= high)].copy()
    return df


def select_features_by_importance(X_train, y_train):
    """Use RF importances on numeric columns to keep informative features."""
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return X_train.columns.tolist()

    imp_model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    imp_model.fit(X_train[numeric_cols], y_train)
    importances = imp_model.named_steps['model'].feature_importances_
    imp_df = pd.DataFrame({'feature': numeric_cols, 'importance': importances}).sort_values('importance', ascending=False)

    # Keep strongly useful + mandatory time-series features
    cutoff = max(imp_df['importance'].median() * 0.25, 0.0025)
    keep_numeric = set(imp_df[imp_df['importance'] >= cutoff]['feature'].tolist())
    mandatory = {
        'PM2.5_lag1', 'PM2.5_lag2', 'PM2.5_lag3', 'PM2.5_lag6', 'PM2.5_lag12', 'PM2.5_lag24', 'PM2.5_lag48',
        'PM2.5_roll_mean_3', 'PM2.5_roll_mean_6', 'PM2.5_roll_mean_12', 'PM2.5_roll_mean_24',
        'PM2.5_roll_std_3', 'PM2.5_roll_std_6', 'PM2.5_roll_std_12', 'PM2.5_roll_std_24',
        'PM2.5_ema_6', 'PM2.5_ema_12',
        'PM2.5_diff', 'PM2.5_pct_change',
        'NO_x_NO2', 'PM10_div_PM25', 'CO_x_NOx',
        'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos'
    }
    keep_numeric.update(set(numeric_cols).intersection(mandatory))

    # Remove highly correlated numeric redundancy (>0.9), keep higher-importance feature
    keep_numeric_list = [c for c in numeric_cols if c in keep_numeric]
    corr_df = X_train[keep_numeric_list].copy()
    corr_df = corr_df.fillna(corr_df.median(numeric_only=True))
    corr = corr_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    importance_map = dict(zip(imp_df['feature'], imp_df['importance']))
    to_drop = set()
    for col in upper.columns:
        for row in upper.index:
            val = upper.loc[row, col]
            if pd.notna(val) and val > 0.9:
                if importance_map.get(row, 0.0) >= importance_map.get(col, 0.0):
                    to_drop.add(col)
                else:
                    to_drop.add(row)

    keep_numeric = [c for c in keep_numeric_list if c not in to_drop]
    top_n = min(30, max(20, len(keep_numeric)))
    keep_numeric = sorted(keep_numeric, key=lambda c: importance_map.get(c, 0.0), reverse=True)[:top_n]

    keep_cols = [c for c in X_train.columns if c in keep_numeric or c in ['Location', 'Frequency']]
    print(f"Selected {len(keep_cols)} / {X_train.shape[1]} features after importance+correlation filter.")
    imp_df.to_csv(os.path.join(RESULTS_DIR, 'feature_importance_rf.csv'), index=False)
    plt.figure(figsize=(10, 6))
    top_imp = imp_df.head(25).sort_values('importance')
    sns.barplot(data=top_imp, x='importance', y='feature')
    plt.title('Top Feature Importances (RF)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance_rf.png'))
    plt.close()
    return keep_cols


def tscv_rmse(model, X, y, n_splits=4):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses = []
    for tr_idx, va_idx in tscv.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        rmses.append(np.sqrt(mean_squared_error(y_va, pred)))
    return float(np.mean(rmses))


def make_lstm_sequences(X_num, y, lookback=24):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X_num)):
        X_seq.append(X_num[i - lookback:i])
        y_seq.append(y[i])
    if not X_seq:
        return np.empty((0, lookback, X_num.shape[1])), np.array([])
    return np.array(X_seq), np.array(y_seq)


def build_baselines(test_df):
    test_df = test_df.sort_values(['Location', 'DateTime']).copy()
    baseline_last = []
    baseline_t24 = []

    for _, g in test_df.groupby('Location'):
        baseline_last.append(g['PM2.5_lag1'].values if 'PM2.5_lag1' in g.columns else np.full(len(g), np.nan))
        baseline_t24.append(g['PM2.5_roll_mean_24'].values if 'PM2.5_roll_mean_24' in g.columns else np.full(len(g), np.nan))

    return np.concatenate(baseline_last), np.concatenate(baseline_t24)


def save_diagnostics(test_df, pred_df):
    # Actual vs predicted
    plt.figure(figsize=(14, 7))
    plt.plot(test_df['DateTime'].values, test_df['PM2.5'].values, label='Actual', color='black', linewidth=2, alpha=0.7)
    for model_name in pred_df.columns:
        if model_name not in ['DateTime', 'Location', 'Actual']:
            plt.plot(test_df['DateTime'].values, pred_df[model_name].values, label=model_name, alpha=0.8)
    plt.title('Actual vs Predicted PM2.5')
    plt.xlabel('Time')
    plt.ylabel('PM2.5')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'actual_vs_predicted.png'))
    plt.close()

    # Residual plot for best model by RMSE
    metrics = pd.read_csv(os.path.join(RESULTS_DIR, 'metrics_final.csv'))
    best_model = metrics.sort_values('RMSE').iloc[0]['Model']
    if best_model in pred_df.columns:
        residual = test_df['PM2.5'].values - pred_df[best_model].values
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=pred_df[best_model].values, y=residual, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f'Residual Plot ({best_model})')
        plt.xlabel('Predicted PM2.5')
        plt.ylabel('Residual')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'residual_plot_best_model.png'))
        plt.close()

    # Error vs time by best model
    if best_model in pred_df.columns:
        abs_err = np.abs(test_df['PM2.5'].values - pred_df[best_model].values)
        err_df = pd.DataFrame({
            'DateTime': test_df['DateTime'].values,
            'AbsoluteError': abs_err
        })
        plt.figure(figsize=(14, 5))
        plt.plot(err_df['DateTime'], err_df['AbsoluteError'])
        plt.title(f'Absolute Error over Time ({best_model})')
        plt.xlabel('Time')
        plt.ylabel('Absolute Error')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'error_vs_time_best_model.png'))
        plt.close()

    # Error vs location for best model
    if best_model in pred_df.columns:
        loc_err = pd.DataFrame({
            'Location': test_df['Location'].values,
            'AbsoluteError': np.abs(test_df['PM2.5'].values - pred_df[best_model].values)
        })
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=loc_err, x='Location', y='AbsoluteError')
        plt.title(f'Absolute Error by Location ({best_model})')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'error_by_location_best_model.png'))
        plt.close()


def save_missingness_chart(data_path):
    df = pd.read_csv(data_path)
    missing_pct = df.isna().mean().sort_values(ascending=False) * 100
    missing_pct = missing_pct[missing_pct > 0]
    if missing_pct.empty:
        return
    plt.figure(figsize=(12, 5))
    sns.barplot(x=missing_pct.index, y=missing_pct.values)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Missing %')
    plt.title('Missing Data Percentage by Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'missing_data_overview.png'))
    plt.close()

def run_pipeline():
    print("--- STARTING ML PIPELINE ---")
    train_df, val_df, test_df = load_and_split()
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    train_df = clean_training_rows(train_df)
    val_df = clean_training_rows(val_df)
    test_df = clean_training_rows(test_df)
    print(f"After quality filtering -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    y_test = test_df['PM2.5'].values
    if np.var(y_test) <= 1e-12:
        raise ValueError("Test target variance is zero. Stop and fix data generation before model evaluation.")

    X_train, y_train = build_features(train_df)
    X_val, y_val = build_features(val_df)
    X_test, _ = build_features(test_df)

    selected_cols = select_features_by_importance(X_train, y_train)
    X_train = X_train[selected_cols].copy()
    X_val = X_val[selected_cols].copy()
    X_test = X_test[selected_cols].copy()

    # Train on train+val after split validation is established.
    X_train_full = pd.concat([X_train, X_val], ignore_index=True)
    y_train_full = np.concatenate([y_train, y_val])

    preprocessor = build_preprocessor(X_train_full)
    results = []

    # Baselines
    baseline_last, baseline_t24 = build_baselines(test_df)
    valid_last = ~np.isnan(baseline_last)
    valid_t24 = ~np.isnan(baseline_t24)
    if valid_last.any():
        results.append(evaluate_model('Baseline_LastValue', y_test[valid_last], baseline_last[valid_last]))
    if valid_t24.any():
        results.append(evaluate_model('Baseline_Seasonal24', y_test[valid_t24], baseline_t24[valid_t24]))

    # Linear-style baseline family: Linear, Ridge, Lasso (TSCV on train)
    print("Training linear baselines (Linear/Ridge/Lasso)...")
    linear_candidates = []
    linear_model = Pipeline([('prep', preprocessor), ('model', LinearRegression())])
    linear_candidates.append(('Linear Regression', linear_model))
    for alpha in [0.1, 1.0, 5.0]:
        ridge = Pipeline([('prep', preprocessor), ('model', Ridge(alpha=alpha))])
        linear_candidates.append((f'Ridge(alpha={alpha})', ridge))
    for alpha in [0.001, 0.01, 0.1]:
        lasso = Pipeline([('prep', preprocessor), ('model', Lasso(alpha=alpha, max_iter=5000))])
        linear_candidates.append((f'Lasso(alpha={alpha})', lasso))

    linear_scores = []
    for name, m in linear_candidates:
        score = tscv_rmse(m, X_train, y_train, n_splits=4)
        linear_scores.append((score, name, m))
    linear_scores.sort(key=lambda x: x[0])
    _, best_linear_name, best_linear_model = linear_scores[0]
    print(f"Best linear variant: {best_linear_name}")

    # Always report plain Linear Regression explicitly.
    linear_model.fit(X_train_full, y_train_full)
    p_linear = linear_model.predict(X_test)
    results.append(evaluate_model('Linear Regression', y_test, p_linear))
    joblib.dump(linear_model, os.path.join(RESULTS_DIR, 'linear_regression.joblib'))

    best_linear_model.fit(X_train_full, y_train_full)
    p_lr = best_linear_model.predict(X_test)
    if best_linear_name != 'Linear Regression':
        results.append(evaluate_model(best_linear_name, y_test, p_lr))
    joblib.dump(best_linear_model, os.path.join(RESULTS_DIR, 'best_linear_model.joblib'))

    # Random Forest tuning on validation
    print("Tuning Random Forest...")
    rf_candidates = []
    for n_estimators in [500, 650, 800]:
        for max_depth in [15, 20, 25]:
            for min_samples_split in [2, 5]:
                for min_samples_leaf in [1, 2]:
                    model = Pipeline([
                        ('prep', preprocessor),
                        ('model', RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            random_state=42,
                            n_jobs=-1
                        ))
                    ])
                    rmse_val = tscv_rmse(model, X_train, y_train, n_splits=4)
                    rf_candidates.append((rmse_val, model, {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf
                    }))
    rf_candidates.sort(key=lambda x: x[0])
    _, _, best_rf_params = rf_candidates[0]
    print(f"Best RF params: {best_rf_params}")
    rf = Pipeline([
        ('prep', preprocessor),
        ('model', RandomForestRegressor(random_state=42, n_jobs=-1, **best_rf_params))
    ])
    rf.fit(X_train_full, y_train_full)
    p_rf = rf.predict(X_test)
    results.append(evaluate_model('Random Forest', y_test, p_rf))
    joblib.dump(rf, os.path.join(RESULTS_DIR, 'random_forest.joblib'))

    # XGBoost tuning on validation
    print("Tuning XGBoost...")
    xgb_candidates = []
    for n_estimators in [600, 800, 1000]:
        for learning_rate in [0.01, 0.02, 0.03]:
            for max_depth in [5, 6, 8]:
                for subsample in [0.8]:
                    for colsample_bytree in [0.8]:
                        model = Pipeline([
                            ('prep', preprocessor),
                            ('model', XGBRegressor(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                reg_alpha=0.0,
                                reg_lambda=1.0,
                                objective='reg:squarederror',
                                random_state=42
                            ))
                        ])
                        rmse_val = tscv_rmse(model, X_train, y_train, n_splits=4)
                        xgb_candidates.append((rmse_val, model, {
                            'n_estimators': n_estimators,
                            'learning_rate': learning_rate,
                            'max_depth': max_depth,
                            'subsample': subsample,
                            'colsample_bytree': colsample_bytree,
                            'reg_alpha': 0.0,
                            'reg_lambda': 1.0,
                            'objective': 'reg:squarederror',
                            'random_state': 42
                        }))
    xgb_candidates.sort(key=lambda x: x[0])
    _, _, best_xgb_params = xgb_candidates[0]
    print(f"Best XGB params: {best_xgb_params}")
    xgb = Pipeline([
        ('prep', preprocessor),
        ('model', XGBRegressor(**best_xgb_params))
    ])
    xgb.fit(X_train_full, y_train_full)
    p_xgb = xgb.predict(X_test)
    results.append(evaluate_model('XGBoost', y_test, p_xgb))
    joblib.dump(xgb, os.path.join(RESULTS_DIR, 'xgboost.joblib'))

    # Weighted ensemble RF + XGB tuned on validation
    pred_val_rf = rf.predict(X_val)
    pred_val_xgb = xgb.predict(X_val)
    best_w = 0.5
    best_rmse = float('inf')
    for w in np.arange(0.1, 1.0, 0.1):
        blend = w * pred_val_rf + (1 - w) * pred_val_xgb
        rmse = np.sqrt(mean_squared_error(y_val, blend))
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = float(w)
    p_ens = best_w * p_rf + (1 - best_w) * p_xgb
    results.append(evaluate_model(f'Ensemble_RF_XGB(w={best_w:.1f})', y_test, p_ens))

    # LSTM: numeric-only sequence model with strict temporal validation.
    print("Training LSTM...")
    rng_seed = 42
    np.random.seed(rng_seed)
    tf.random.set_seed(rng_seed)
    lookback = 24

    lstm_numeric_cols = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
    if len(lstm_numeric_cols) == 0:
        print("[WARN] Skipping LSTM: no numeric features available.")
        p_lstm = None
    else:
        imputer = SimpleImputer(strategy='median')
        scaler_X = StandardScaler()
        y_scaler = StandardScaler()

        Xtr_num = imputer.fit_transform(X_train[lstm_numeric_cols])
        Xva_num = imputer.transform(X_val[lstm_numeric_cols])
        Xte_num = imputer.transform(X_test[lstm_numeric_cols])

        Xtr_num = scaler_X.fit_transform(Xtr_num)
        Xva_num = scaler_X.transform(Xva_num)
        Xte_num = scaler_X.transform(Xte_num)

        ytr_sc = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        yva_sc = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

        Xtr_seq, ytr_seq = make_lstm_sequences(Xtr_num, ytr_sc, lookback=lookback)
        Xva_seq, yva_seq = make_lstm_sequences(Xva_num, yva_sc, lookback=lookback)
        Xte_seq, _ = make_lstm_sequences(Xte_num, y_test, lookback=lookback)

        if len(Xtr_seq) == 0 or len(Xva_seq) == 0 or len(Xte_seq) == 0:
            print("[WARN] Skipping LSTM: not enough sequence rows after lookback.")
            p_lstm = None
        else:
            lstm = Sequential([
                LSTM(64, return_sequences=True, input_shape=(Xtr_seq.shape[1], Xtr_seq.shape[2])),
                Dropout(0.2),
                LSTM(32),
                Dense(1)
            ])
            lstm.compile(optimizer='adam', loss='mse')
            es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
            lstm.fit(
                Xtr_seq, ytr_seq,
                validation_data=(Xva_seq, yva_seq),
                epochs=80,
                batch_size=32,
                shuffle=False,
                verbose=0,
                callbacks=[es]
            )
            pred_lstm_sc = lstm.predict(Xte_seq, verbose=0).flatten()
            p_lstm = y_scaler.inverse_transform(pred_lstm_sc.reshape(-1, 1)).flatten()
            y_test_lstm = y_test[lookback:]
            results.append(evaluate_model('LSTM', y_test_lstm, p_lstm))
            lstm.save(os.path.join(RESULTS_DIR, 'lstm_model.keras'))

    # --- EVALUATION SUMMARY & VISUALIZATION ---
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(RESULTS_DIR, 'metrics_final.csv'), index=False)

    sns.set_palette("muted")

    # RMSE comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=res_df, x='Model', y='RMSE')
    plt.title('Error Comparison (Lower is Better)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'rmse_bar_chart.png'))

    pred_df = pd.DataFrame({
        'DateTime': test_df['DateTime'].values,
        'Location': test_df['Location'].values,
        'Actual': y_test,
        'Linear Regression': p_linear,
        best_linear_name: p_lr,
        'Random Forest': p_rf,
        'XGBoost': p_xgb,
        f'Ensemble_RF_XGB(w={best_w:.1f})': p_ens
    })
    if p_lstm is not None:
        # Align to the last portion due to lookback window.
        pred_df.loc[pred_df.index[lookback:], 'LSTM'] = p_lstm
    pred_df.to_csv(os.path.join(RESULTS_DIR, 'test_predictions.csv'), index=False)

    save_diagnostics(test_df, pred_df)
    save_missingness_chart(DATA_PATH)

    print(f"\nPipeline Complete! Models and Plots saved in {RESULTS_DIR}")

if __name__ == "__main__":
    run_pipeline()
