import pandas as pd
import numpy as np
import os
import itertools
from contextlib import redirect_stdout
import networkx as nx
import matplotlib.pyplot as plt

# OUTPUT DIRECTORY
OUT_DIR = r'c:\Users\annam\OneDrive\Desktop\ML_Project\TRANSPORT_ANALYSIS'
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH = r'c:\Users\annam\OneDrive\Desktop\ML_Project\FEATURE_ENGINEERED_WIND\Master_Hourly_All_FE.csv'

def expected_direction(A, B):
    dirs = {
        ('Siltara', 'AIIMS'): 270,
        ('Siltara', 'IGKV'): 315,
        ('Siltara', 'Bhatagaon'): 225,
        ('AIIMS', 'Siltara'): 90,
        ('IGKV', 'Siltara'): 135,
        ('Bhatagaon', 'Siltara'): 45,
        ('AIIMS', 'IGKV'): 45,
        ('IGKV', 'AIIMS'): 225,
        ('AIIMS', 'Bhatagaon'): 135,
        ('Bhatagaon', 'AIIMS'): 315,
        ('IGKV', 'Bhatagaon'): 225,
        ('Bhatagaon', 'IGKV'): 45,
    }
    return dirs.get((A, B), 0)

def is_aligned(wd, expected, tol=45):
    diff = np.abs((wd - expected + 180) % 360 - 180)
    return diff <= tol

def compute_lag_correlation(x, y, max_lag=12):
    corrs = []
    for k in range(1, max_lag + 1):
        x_shifted = x.shift(k)
        valid = ~x_shifted.isna() & ~y.isna()
        if valid.sum() > 30:
            corrs.append(np.corrcoef(x_shifted[valid], y[valid])[0, 1])
        else:
            corrs.append(0)
    return corrs

def check_distance_category(A, B):
    pair = set([A, B])
    if pair == {'AIIMS', 'Bhatagaon'}:
        return 'near'
    return 'moderate'

def analyze_extended_transport():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    stations = ["Siltara", "AIIMS", "IGKV", "Bhatagaon"]
    results = []
    
    print("Evaluating pairs with randomized baselines...")
    np.random.seed(42) # For reproducibility
    
    for A, B in itertools.permutations(stations, 2):
        df_A = df[df['Location'] == A].set_index('DateTime')
        df_B = df[df['Location'] == B].set_index('DateTime')
        
        df_pair = df_A.join(df_B, lsuffix='_A', rsuffix='_B', how='inner').reset_index()
        if df_pair.empty: continue
        
        x_actual = df_pair['PM2.5_A']
        y_actual = df_pair['PM2.5_B']
        
        # 0. Unfiltered initial baseline
        corrs_unfiltered = compute_lag_correlation(x_actual, y_actual)
        if max(corrs_unfiltered) <= 0: continue
        
        # 1. Randomized Baseline Validation
        x_shuffled = x_actual.copy()
        x_shuffled = pd.Series(np.random.permutation(x_shuffled.values), index=x_shuffled.index)
        
        corrs_random = compute_lag_correlation(x_shuffled, y_actual)
        
        max_random_corr = max(corrs_random) if corrs_random else 0
        random_lag_idx = np.argmax(corrs_random) + 1 if max_random_corr > 0 else 0
        
        # Wind Validation
        expected = expected_direction(A, B)
        wd_mask = is_aligned(df_pair['WD_A'], expected)
        ws_mask = df_pair['WS_A'] > 1.5
        
        # 2. Wind Alignment Percentage
        total_valid_wind_samples = (~df_pair['WD_A'].isna()).sum()
        if total_valid_wind_samples > 0:
            wind_alignment_percent = (wd_mask & ws_mask).sum() / total_valid_wind_samples
        else:
            wind_alignment_percent = 0
            
        df_filtered = df_pair[wd_mask & ws_mask]
        wind_alignment_score = 0
        pollutant_consistency_score = 0
        
        if len(df_filtered) > 50:
            corrs_filtered = compute_lag_correlation(df_filtered['PM2.5_A'], df_filtered['PM2.5_B'])
            if max(corrs_filtered) > 0:
                best_lag = np.argmax(corrs_filtered) + 1
                actual_corr = np.max(corrs_filtered)
                wind_alignment_score = 1
            else:
                best_lag = np.argmax(corrs_unfiltered) + 1
                actual_corr = np.max(corrs_unfiltered)
        else:
            best_lag = np.argmax(corrs_unfiltered) + 1
            actual_corr = np.max(corrs_unfiltered)
            df_filtered = df_pair # fallback
            
        if wind_alignment_score == 1:
            no2_corrs = compute_lag_correlation(df_filtered['NO2_A'], df_filtered['NO2_B'])
            so2_corrs = compute_lag_correlation(df_filtered['SO2_A'], df_filtered['SO2_B'])
            valid_p = sum([1 for corrs, lag in [(no2_corrs, best_lag), (so2_corrs, best_lag)] 
                         if corrs and max(corrs) > 0 and abs((np.argmax(corrs)+1) - lag) <= 1])
            if valid_p >= 1:
                pollutant_consistency_score = 1
                
        # 3. Distance-Lag Consistency Check
        dist_cat = check_distance_category(A, B)
        lag_consistent = 0
        if dist_cat == 'near' and best_lag <= 2:
            lag_consistent = 1
        elif dist_cat == 'moderate' and 2 <= best_lag <= 8:
            lag_consistent = 1
            
        corr_diff = actual_corr - max_random_corr
        passes_baseline = corr_diff > 0.15
        
        # 4. Update Confidence Score
        corr_norm = min(max(actual_corr / 0.8, 0), 1)
        score = (0.4 * corr_norm + 
                 0.2 * pollutant_consistency_score + 
                 0.2 * wind_alignment_percent + 
                 0.2 * lag_consistent)
                 
        results.append({
            'Source': A,
            'Target': B,
            'Lag': best_lag,
            'Correlation': actual_corr,
            'Random Corr': max_random_corr,
            'Corr Diff': corr_diff,
            'Wind Alignment %': wind_alignment_percent * 100, # Display as percentage
            'Distance Category': dist_cat,
            'Lag Consistent': lag_consistent,
            'Confidence Score': score,
            'Passes Baseline': passes_baseline
        })

    # 5. Output
    results_df = pd.DataFrame(results)
    
    # Filter strictly for final accepted pathways
    accepted_df = results_df[results_df['Passes Baseline'] & (results_df['Confidence Score'] > 0.6)]
    
    # Display columns ordered properly
    display_cols = ['Source', 'Target', 'Lag', 'Correlation', 'Random Corr', 'Corr Diff', 
                    'Wind Alignment %', 'Distance Category', 'Lag Consistent', 'Confidence Score']
    
    results_df.to_csv(os.path.join(OUT_DIR, 'extended_transport_matrix.csv'), index=False)
    
    print("\n--- EXTENDED TRANSPORT PATHWAY ANALYSIS ---")
    print(f"Total Pairs Analyzed: {len(results_df)}")
    print(f"Pairs Passing Randomized Baseline Test (>0.15 diff): {results_df['Passes Baseline'].sum()}")
    print(f"Final Accepted Valid Pathways (Passes Baseline & Score > 0.6): {len(accepted_df)}\n")
    
    print("ALL PAIRS:")
    print(results_df[display_cols].to_string())
    
    print("\nACCEPTED PATHWAYS:")
    print(accepted_df[display_cols].to_string() if not accepted_df.empty else "None")
    
    # 6. Interpretation Block
    print("\n--- INTERPRETATION ---")
    print("Randomized Baselines:")
    passed_baseline = results_df['Passes Baseline'].sum()
    print(f"- {passed_baseline} out of {len(results_df)} links exhibited actual correlations significantly above noise (shuffle-baseline + 0.15).")
    
    print("Wind Alignment Strength:")
    avg_wind = results_df['Wind Alignment %'].mean()
    max_wind_pair = results_df.loc[results_df['Wind Alignment %'].idxmax()]
    print(f"- The average wind alignment supporting any pair was {avg_wind:.1f}%.")
    print(f"- {max_wind_pair['Source']} -> {max_wind_pair['Target']} showed the strongest meteorological potential with {max_wind_pair['Wind Alignment %']:.1f}% suitable wind.")
    
    print("Geospatial Lag Consistency:")
    lag_c = results_df['Lag Consistent'].sum()
    print(f"- {lag_c} out of {len(results_df)} links correctly mapped their empirical delays to physical distance heuristics (near<=2h, moderate 2-8h).")
    
if __name__ == "__main__":
    analyze_extended_transport()
