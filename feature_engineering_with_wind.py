import pandas as pd
import numpy as np
import os
import glob
import warnings

warnings.filterwarnings('ignore')

# CONFIGURATION
INPUT_DIR_INDIVIDUAL = r'c:\Users\annam\OneDrive\Desktop\ML_Project\DATASETS_CLEANED_WIND\Individual'
INPUT_DIR_COMBINED = r'c:\Users\annam\OneDrive\Desktop\ML_Project\DATASETS_CLEANED_WIND\Combined'
OUTPUT_DIR = r'c:\Users\annam\OneDrive\Desktop\ML_Project\FEATURE_ENGINEERED_WIND'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Full set of pollutants we want to TRY and lag
POLLUTANTS_OPZ = ['NO', 'NO2', 'NOx', 'SO2', 'PM2.5', 'PM10', 'CO', 'NH3']

def perform_feature_engineering_with_wind(df, is_combined=False):
    """Feature engineering with leakage-safe lag/rolling features and minimal row loss, retaining wind data."""
    df = df.copy()

    # --- PHASE 1: INITIAL VALIDATION & SORTING ---
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df = df.dropna(subset=['DateTime'])

    if is_combined and 'Location' in df.columns:
        df = df.sort_values(['Location', 'DateTime'])
        group_keys = ['Location']
    else:
        df = df.sort_values('DateTime')
        group_keys = None

    initial_shape = df.shape

    # --- PHASE 2: TEMPORAL FEATURES ---
    df['Hour'] = df['DateTime'].dt.hour
    df['Day'] = df['DateTime'].dt.day
    df['Month'] = df['DateTime'].dt.month
    df['Weekday'] = df['DateTime'].dt.weekday
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)

    # --- PHASE 3: PM2.5 LAGS + ROLLING FEATURES ---
    if 'PM2.5' not in df.columns:
        return df.iloc[0:0].copy(), initial_shape, (0, df.shape[1]), len(df)

    if group_keys:
        grouped_pm = df.groupby(group_keys)['PM2.5']
        df['PM2.5_lag1'] = grouped_pm.shift(1)
        df['PM2.5_lag2'] = grouped_pm.shift(2)
        df['PM2.5_lag3'] = grouped_pm.shift(3)
        df['PM2.5_lag6'] = grouped_pm.shift(6)
        df['PM2.5_lag12'] = grouped_pm.shift(12)
        df['PM2.5_lag24'] = grouped_pm.shift(24)
        df['PM2.5_lag48'] = grouped_pm.shift(48)
        df['PM2.5_roll_mean_3'] = grouped_pm.shift(1).rolling(3, min_periods=3).mean().reset_index(level=0, drop=True)
        df['PM2.5_roll_mean_6'] = grouped_pm.shift(1).rolling(6, min_periods=6).mean().reset_index(level=0, drop=True)
        df['PM2.5_roll_mean_12'] = grouped_pm.shift(1).rolling(12, min_periods=6).mean().reset_index(level=0, drop=True)
        df['PM2.5_roll_mean_24'] = grouped_pm.shift(1).rolling(24, min_periods=6).mean().reset_index(level=0, drop=True)
        df['PM2.5_roll_std_3'] = grouped_pm.shift(1).rolling(3, min_periods=3).std().reset_index(level=0, drop=True)
        df['PM2.5_roll_std_6'] = grouped_pm.shift(1).rolling(6, min_periods=6).std().reset_index(level=0, drop=True)
        df['PM2.5_roll_std_12'] = grouped_pm.shift(1).rolling(12, min_periods=6).std().reset_index(level=0, drop=True)
        df['PM2.5_roll_std_24'] = grouped_pm.shift(1).rolling(24, min_periods=6).std().reset_index(level=0, drop=True)
        df['PM2.5_ema_6'] = grouped_pm.shift(1).ewm(span=6, adjust=False).mean().reset_index(level=0, drop=True)
        df['PM2.5_ema_12'] = grouped_pm.shift(1).ewm(span=12, adjust=False).mean().reset_index(level=0, drop=True)
    else:
        pm_shift = df['PM2.5'].shift(1)
        df['PM2.5_lag1'] = pm_shift
        df['PM2.5_lag2'] = df['PM2.5'].shift(2)
        df['PM2.5_lag3'] = df['PM2.5'].shift(3)
        df['PM2.5_lag6'] = df['PM2.5'].shift(6)
        df['PM2.5_lag12'] = df['PM2.5'].shift(12)
        df['PM2.5_lag24'] = df['PM2.5'].shift(24)
        df['PM2.5_lag48'] = df['PM2.5'].shift(48)
        df['PM2.5_roll_mean_3'] = pm_shift.rolling(3, min_periods=3).mean()
        df['PM2.5_roll_mean_6'] = pm_shift.rolling(6, min_periods=6).mean()
        df['PM2.5_roll_mean_12'] = pm_shift.rolling(12, min_periods=6).mean()
        df['PM2.5_roll_mean_24'] = pm_shift.rolling(24, min_periods=6).mean()
        df['PM2.5_roll_std_3'] = pm_shift.rolling(3, min_periods=3).std()
        df['PM2.5_roll_std_6'] = pm_shift.rolling(6, min_periods=6).std()
        df['PM2.5_roll_std_12'] = pm_shift.rolling(12, min_periods=6).std()
        df['PM2.5_roll_std_24'] = pm_shift.rolling(24, min_periods=6).std()
        df['PM2.5_ema_6'] = pm_shift.ewm(span=6, adjust=False).mean()
        df['PM2.5_ema_12'] = pm_shift.ewm(span=12, adjust=False).mean()

    # Leakage-safe trend proxy using only lagged target values
    df['PM2.5_diff'] = df['PM2.5_lag1'] - df['PM2.5_lag2']
    lag2_safe = df['PM2.5_lag2'].replace(0, np.nan)
    df['PM2.5_pct_change'] = (df['PM2.5_lag1'] - df['PM2.5_lag2']) / lag2_safe

    # Interaction features
    if {'NO', 'NO2'}.issubset(df.columns):
        df['NO_x_NO2'] = df['NO'] * df['NO2']
    if {'PM10_lag1', 'PM2.5_lag1'}.issubset(df.columns):
        pm25_lag_safe = df['PM2.5_lag1'].replace(0, np.nan)
        df['PM10_div_PM25'] = df['PM10_lag1'] / pm25_lag_safe
    if {'CO', 'NOx'}.issubset(df.columns):
        df['CO_x_NOx'] = df['CO'] * df['NOx']

    # Optional lag1 features for available pollutants (informative but not mandatory)
    for poll in POLLUTANTS_OPZ:
        if poll == 'PM2.5' or poll not in df.columns:
            continue
        if group_keys:
            df[f'{poll}_lag1'] = df.groupby(group_keys)[poll].shift(1)
        else:
            df[f'{poll}_lag1'] = df[poll].shift(1)

    # --- PHASE 4: SUPERVISED FILTER (minimal required columns only) ---
    # ADDED WS & WD
    required_cols = [
        'PM2.5', 'PM2.5_lag1', 'PM2.5_lag2', 'PM2.5_lag3',
        'PM2.5_lag6', 'PM2.5_lag12', 'PM2.5_lag24', 'PM2.5_lag48',
        'PM2.5_roll_mean_3', 'PM2.5_roll_mean_6', 'PM2.5_roll_mean_12', 'PM2.5_roll_mean_24',
        'PM2.5_roll_std_3', 'PM2.5_roll_std_6', 'PM2.5_roll_std_12', 'PM2.5_roll_std_24',
        'PM2.5_ema_6', 'PM2.5_ema_12', 'PM2.5_diff', 'PM2.5_pct_change'
    ]
    
    # Identify which metadata and static cols to keep
    static_cols = ['DateTime', 'Location', 'Frequency', 'Year', 'Hour', 'Day', 'Month', 'Weekday', 
                   'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 'is_imputed', 'gap_length']
                   
    # Any extra columns like other pollutants and interaction features that exist
    extra_cols = [c for c in df.columns if c.endswith('_lag1') or c in ('NO_x_NO2', 'PM10_div_PM25', 'CO_x_NOx') 
                  or c in POLLUTANTS_OPZ or c in ('WS', 'WD')]
                  
    cols_to_keep = list(set(static_cols + required_cols + extra_cols))
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]

    final_df = df.dropna(subset=[c for c in required_cols if c in df.columns]).copy()
    final_df = final_df[cols_to_keep] # Subset to only clean kept columns
    
    rows_removed = len(df) - len(final_df)

    return final_df, initial_shape, final_df.shape, rows_removed

def process_all_datasets():
    individual_files = glob.glob(os.path.join(INPUT_DIR_INDIVIDUAL, '*.csv'))
    combined_files = glob.glob(os.path.join(INPUT_DIR_COMBINED, '*.csv'))
    
    report_data = []
    
    for f in individual_files + combined_files:
        filename = os.path.basename(f)
        is_comb = 'Combined' in f
        try:
            df = pd.read_csv(f)
            fe_df, init_s, final_s, removed = perform_feature_engineering_with_wind(df, is_combined=is_comb)
            
            output_name = filename.replace('.csv', '_FE.csv')
            fe_df.to_csv(os.path.join(OUTPUT_DIR, output_name), index=False)
            loc_count = fe_df['Location'].nunique() if 'Location' in fe_df.columns else 1
            report_data.append({
                'File': output_name,
                'Init': init_s[0],
                'Final': final_s[0],
                'Locations': loc_count,
                'Status': 'Success'
            })
        except Exception as e:
            report_data.append({
                'File': filename,
                'Init': 0,
                'Final': 0,
                'Locations': 0,
                'Status': f'Error: {e}'
            })
        
    # Print Report
    print("\n" + "="*70)
    print("FINAL FEATURE ENGINEERING WITH WIND SUMMARY")
    print("="*70)
    print(f"{'Filename':<30} | {'Rows Init':<10} | {'Rows Final':<10} | {'Locations':<10} | {'Status'}")
    print("-" * 70)
    for r in report_data:
        print(f"{r['File']:<30} | {r['Init']:<10} | {r['Final']:<10} | {r['Locations']:<10} | {r['Status']}")

if __name__ == "__main__":
    process_all_datasets()
