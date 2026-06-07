import pandas as pd
import numpy as np
import os
import glob
import warnings

warnings.filterwarnings('ignore')

# CONFIGURATION
INPUT_DIR_INDIVIDUAL = r'c:\Users\annam\OneDrive\Desktop\ML_Project\DATASETS_CLEANED\Individual'
INPUT_DIR_COMBINED = r'c:\Users\annam\OneDrive\Desktop\ML_Project\DATASETS_CLEANED\Combined'
OUTPUT_DIR = r'c:\Users\annam\OneDrive\Desktop\ML_Project\FEATURE_ENGINEERED'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Full set of pollutants we expect
POLLUTANTS = ['NO', 'NO2', 'NOx', 'SO2', 'PM2.5', 'PM10', 'CO', 'NH3']

def perform_feature_engineering(df, is_combined=False):
    """Applies temporal and lag feature engineering to a dataframe with robust column handling."""
    
    # --- PHASE 0: COLUMN PADDING ---
    # Ensure all expected columns exist, even if NaN
    for col in POLLUTANTS:
        if col not in df.columns:
            df[col] = np.nan
            
    # --- PHASE 1: INITIAL VALIDATION & SORTING ---
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    if is_combined:
        df = df.sort_values(['Location', 'DateTime'])
    else:
        df = df.sort_values('DateTime')
        
    initial_shape = df.shape
    
    # --- PHASE 2: TEMPORAL FEATURES ---
    df['Hour'] = df['DateTime'].dt.hour
    df['Day'] = df['DateTime'].dt.day
    df['Month'] = df['DateTime'].dt.month
    df['Weekday'] = df['DateTime'].dt.weekday
    
    # --- PHASE 3: LAG FEATURES ---
    group_col = 'Location' if is_combined else None
    
    # PM2.5 Lags (1, 2, 3)
    for i in [1, 2, 3]:
        col_name = f'PM2.5_lag{i}'
        if group_col:
            df[col_name] = df.groupby(group_col)['PM2.5'].shift(i)
        else:
            df[col_name] = df['PM2.5'].shift(i)
            
    # Other Pollutants Lag1
    for poll in POLLUTANTS:
        if poll == 'PM2.5': continue 
        col_name = f'{poll}_lag1'
        if group_col:
            df[col_name] = df.groupby(group_col)[poll].shift(1)
        else:
            df[col_name] = df[poll].shift(1)
                
    # --- PHASE 4: HANDLE NEW MISSING VALUES ---
    # We drop any rows that have NaNs in the newly created features or target
    # NOTE: If a location was completely missing a pollutant, this will drop all rows for that location.
    cols_to_check = ['Hour', 'Day', 'Month', 'Weekday'] + \
                    [f'PM2.5_lag{i}' for i in [1, 2, 3]] + \
                    [f'{p}_lag1' for p in POLLUTANTS if p != 'PM2.5'] + \
                    ['PM2.5'] # Target must exist
                    
    final_df = df.dropna(subset=cols_to_check).copy()
    rows_removed = len(df) - len(final_df)
    
    return final_df, initial_shape, final_df.shape, rows_removed

def process_all_datasets():
    # Identify all cleaned files
    individual_files = glob.glob(os.path.join(INPUT_DIR_INDIVIDUAL, '*.csv'))
    combined_files = glob.glob(os.path.join(INPUT_DIR_COMBINED, '*.csv'))
    
    report_data = []
    
    # Process Individual Files
    for f in individual_files:
        filename = os.path.basename(f)
        try:
            df = pd.read_csv(f)
            fe_df, init_s, final_s, removed = perform_feature_engineering(df, is_combined=False)
            
            output_name = filename.replace('.csv', '_FE.csv')
            fe_df.to_csv(os.path.join(OUTPUT_DIR, output_name), index=False)
            report_data.append({'File': output_name, 'Original': init_s[0], 'Final': final_s[0], 'Removed': removed, 'Status': 'Success'})
        except Exception as e:
            report_data.append({'File': filename, 'Original': 0, 'Final': 0, 'Removed': 0, 'Status': f'Error: {e}'})
        
    # Process Combined Files
    for f in combined_files:
        filename = os.path.basename(f)
        try:
            df = pd.read_csv(f)
            fe_df, init_s, final_s, removed = perform_feature_engineering(df, is_combined=True)
            
            output_name = filename.replace('.csv', '_FE.csv')
            fe_df.to_csv(os.path.join(OUTPUT_DIR, output_name), index=False)
            report_data.append({'File': output_name, 'Original': init_s[0], 'Final': final_s[0], 'Removed': removed, 'Status': 'Success'})
        except Exception as e:
            report_data.append({'File': filename, 'Original': 0, 'Final': 0, 'Removed': 0, 'Status': f'Error: {e}'})
        
    # Print Report
    print("\n" + "="*80)
    print("FEATURE ENGINEERING REPORT (REVISED)")
    print("="*80)
    print(f"{'Dataset':<30} | {'Rows Init':<10} | {'Rows Final':<10} | {'Status'}")
    print("-" * 80)
    for r in report_data:
        print(f"{r['File']:<30} | {r['Original']:<10} | {r['Final']:<10} | {r['Status']}")

if __name__ == "__main__":
    process_all_datasets()
