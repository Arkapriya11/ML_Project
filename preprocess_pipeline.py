import os
import pandas as pd
import numpy as np
import glob
import warnings
import re

warnings.filterwarnings('ignore')

# CONFIGURATION
BASE_DIR = r'c:\Users\annam\OneDrive\Desktop\ML_Project\Datasets'
OUTPUT_DIR = r'c:\Users\annam\OneDrive\Desktop\ML_Project\DATASETS_CLEANED'
INDIVIDUAL_DIR = os.path.join(OUTPUT_DIR, 'Individual')
COMBINED_DIR = os.path.join(OUTPUT_DIR, 'Combined')

# Target columns
TARGET_COLS = ['DateTime', 'NO', 'NO2', 'NOx', 'SO2', 'PM2.5', 'PM10', 'CO', 'NH3']
POLLUTANT_COLS = ['NO', 'NO2', 'NOx', 'SO2', 'PM2.5', 'PM10', 'CO', 'NH3']

# Ensure output directories exist
os.makedirs(INDIVIDUAL_DIR, exist_ok=True)
os.makedirs(COMBINED_DIR, exist_ok=True)

def find_header_row(df):
    """Search for a row that looks like it contains the pollutants."""
    for i in range(min(30, len(df))):
        row = [str(x).strip() for x in df.iloc[i]]
        pollutant_hits = 0
        has_datetime = False
        for val in row:
            mapped = infer_pollutant(val)
            if mapped == 'DateTime':
                has_datetime = True
            elif mapped in POLLUTANT_COLS:
                pollutant_hits += 1

        # Avoid metadata rows like "Date:" / "Time Base:" by requiring enough pollutant columns.
        if has_datetime and pollutant_hits >= 4:
            return i
    return None


def normalize_column_name(col_name):
    """Lowercase and remove separators/special chars for robust matching."""
    col = str(col_name).strip().lower()
    col = col.replace('_', '').replace(' ', '')
    col = re.sub(r'[^a-z0-9\.]', '', col)
    return col


def infer_pollutant(col_name):
    """Map a noisy source column name to a canonical pollutant name."""
    norm = normalize_column_name(col_name)
    if 'date' in norm or 'time' in norm:
        return 'DateTime'

    # Normalize common PM2.5 variants into the same token
    norm = norm.replace('2.5', '25').replace('_', '')

    # Ordered matching prevents NO from swallowing NO2/NOx
    if norm.startswith('pm25'):
        return 'PM2.5'
    if norm.startswith('pm10'):
        return 'PM10'
    if norm.startswith('no2'):
        return 'NO2'
    if norm.startswith('nox'):
        return 'NOx'
    if norm.startswith('so2'):
        return 'SO2'
    if norm.startswith('nh3'):
        return 'NH3'
    if norm.startswith('co'):
        return 'CO'
    if norm.startswith('no'):
        return 'NO'
    return None


def infer_expected_frequency(freq_label):
    return '15min' if freq_label.lower().startswith('quarter') else '1H'

def clean_file(file_path, location, frequency):
    """Processes a single excel file and returns a standardized DataFrame."""
    try:
        if file_path.endswith('.xlsb'): engine = 'pyxlsb'
        elif file_path.endswith('.xls'): engine = 'xlrd'
        else: engine = 'openpyxl'
        
        df_raw = pd.read_excel(file_path, header=None, engine=engine)
        
        header_idx = find_header_row(df_raw)
        if header_idx is None:
            print(f"  [WARN] Could not find header in {file_path}")
            return None
            
        header_row = [str(x).strip() for x in df_raw.iloc[header_idx]]
        df = df_raw.iloc[header_idx+1:].copy()
        df.columns = header_row
        
        # Map columns with robust normalization
        mapping = {}
        for col in df.columns:
            mapped = infer_pollutant(col)
            if mapped:
                mapping[col] = mapped
        
        df = df.rename(columns=mapping)
        df = df.loc[:, ~pd.Index(df.columns).duplicated(keep='first')]
        # Keep only mapped targets and drop the rest
        cols_to_keep = [c for c in df.columns if c in TARGET_COLS]
        df = df[cols_to_keep]
        
        # Ensure 'DateTime' exists
        if 'DateTime' not in df.columns:
            print(f"  [WARN] Missing DateTime column in {file_path}")
            return None
            
        # Clean DateTime
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
        df = df.dropna(subset=['DateTime'])
        # Guard against corrupted timestamps to prevent massive fake reindex ranges.
        df = df[(df['DateTime'].dt.year >= 2020) & (df['DateTime'].dt.year <= 2030)]
        if df.empty:
            print(f"  [WARN] No valid DateTime rows in {file_path}")
            return None
        
        # Convert pollutants to numeric
        pollutant_cols = [c for c in df.columns if c in POLLUTANT_COLS]
        for col in pollutant_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Add metadata
        year_match = re.search(r'202\d', file_path)
        df['Year'] = year_match.group(0) if year_match else 'Unknown'
        df['Location'] = location
        df['Frequency'] = frequency
        
        return df
        
    except Exception as e:
        print(f"  [ERROR] Processing {file_path}: {e}")
        return None

def process_all_files():
    locations = {
        'AIIMS': 'DCR AIIMS',
        'Siltara': 'SILTARA DCR',
        'IGKV': 'IGKV DCR',
        'Bhatagaon': 'Bhatagaon DCR'
    }
    
    # Store aggregated dataframes: key = (location, frequency)
    aggregates = {}
    
    for loc_name, folder in locations.items():
        loc_path = os.path.join(BASE_DIR, folder)
        print(f"Processing Location: {loc_name}")
        
        all_files = []
        for ext in ['*.xls', '*.xlsx', '*.xlsb']:
            all_files.extend(glob.glob(os.path.join(loc_path, '**', ext), recursive=True))
        all_files = [f for f in all_files if not os.path.basename(f).startswith('~$')]
            
        for f in all_files:
            # Determine frequency from filename or path
            fname = os.path.basename(f).upper()
            if 'QUAT' in fname or 'QUART' in fname:
                freq = 'Quarterly'
            else:
                freq = 'Hourly'
                
            df_clean = clean_file(f, loc_name, freq)
            if df_clean is not None:
                key = (loc_name, freq)
                if key not in aggregates:
                    aggregates[key] = []
                aggregates[key].append(df_clean)
                
    quality_rows = []

    # Final Combine and Cleanup per site-freq
    final_8 = {}
    
    for (loc, freq), dfs in aggregates.items():
        print(f"Finalizing {loc} {freq} dataset...")
        master_site = pd.concat(dfs, ignore_index=True)
        
        # Deduplicate and Sort
        master_site = master_site.drop_duplicates(subset=['DateTime'])
        master_site = master_site.sort_values('DateTime')
        
        # Apply Clipping [0, 500]
        pollutant_cols = [c for c in master_site.columns if c in POLLUTANT_COLS]
        master_site[pollutant_cols] = master_site[pollutant_cols].clip(lower=0, upper=500)

        # Reindex to a proper frequency grid and only impute short gaps.
        expected_freq = infer_expected_frequency(freq)
        master_site = master_site.set_index('DateTime')
        full_idx = pd.date_range(master_site.index.min(), master_site.index.max(), freq=expected_freq)
        # Prevent exploding row counts on very sparse long spans.
        if len(full_idx) <= max(len(master_site) * 3, 10000):
            master_site = master_site.reindex(full_idx)
            master_site.index.name = 'DateTime'
        else:
            print(f"  [WARN] Skipping full reindex for {loc}-{freq} due sparse long range.")
            master_site.index.name = 'DateTime'

        # Restore static metadata after reindex
        master_site['Location'] = loc
        master_site['Frequency'] = freq
        if 'Year' in master_site.columns:
            master_site['Year'] = master_site['Year'].ffill()

        # Flag gaps before imputation
        missing_before = master_site[pollutant_cols].isna().any(axis=1)
        master_site['is_imputed'] = missing_before.astype(int)

        # Contiguous gap length marker (0 for non-missing rows)
        grp = (missing_before != missing_before.shift(fill_value=False)).cumsum()
        gap_len = missing_before.groupby(grp).transform('sum')
        master_site['gap_length'] = np.where(missing_before, gap_len, 0).astype(int)

        # Interpolate short gaps only; never backfill globally (prevents leakage)
        master_site[pollutant_cols] = master_site[pollutant_cols].interpolate(
            method='time',
            limit=3,
            limit_direction='forward'
        )

        master_site = master_site.reset_index()

        missing_after = master_site[pollutant_cols].isna().mean().mean() * 100
        pm25_present = 'PM2.5' in master_site.columns and master_site['PM2.5'].notna().any()
        quality_rows.append({
            'Location': loc,
            'Frequency': freq,
            'Rows': len(master_site),
            'MissingRateAfterPct': round(missing_after, 3),
            'PM25Present': bool(pm25_present)
        })
        
        # Save Individual
        out_name = f"{loc}_{freq}.csv"
        master_site.to_csv(os.path.join(INDIVIDUAL_DIR, out_name), index=False)
        final_8[(loc, freq)] = master_site
        
    # Create 2 Master Combined Datasets
    print("Creating Master Combined datasets...")
    hourly_masters = [df for (loc, freq), df in final_8.items() if freq == 'Hourly']
    quarterly_masters = [df for (loc, freq), df in final_8.items() if freq == 'Quarterly']
    
    if hourly_masters:
        master_hourly_all = pd.concat(hourly_masters, ignore_index=True)
        master_hourly_all.to_csv(os.path.join(COMBINED_DIR, 'Master_Hourly_All.csv'), index=False)
        print("Saved Master_Hourly_All.csv")
        
    if quarterly_masters:
        master_quarterly_all = pd.concat(quarterly_masters, ignore_index=True)
        master_quarterly_all.to_csv(os.path.join(COMBINED_DIR, 'Master_Quarter_All.csv'), index=False)
        print("Saved Master_Quarter_All.csv")

    if quality_rows:
        quality_df = pd.DataFrame(quality_rows)
        quality_df.to_csv(os.path.join(OUTPUT_DIR, 'data_quality_summary.csv'), index=False)
        print("Saved data_quality_summary.csv")

if __name__ == "__main__":
    process_all_files()
    print("\nPipeline Complete! Check DATASETS_CLEANED folder.")
