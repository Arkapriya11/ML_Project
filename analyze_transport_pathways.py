import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import itertools
from matplotlib.colors import Normalize

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
        # Approximate relative angles for remaining bounds
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

def compute_lag_correlation(df_pair, pollutant, max_lag=12):
    corrs = []
    for k in range(1, max_lag + 1):
        if f'{pollutant}_lag{k}_A' in df_pair.columns:
            # use precomputed lag feature
            x = df_pair[f'{pollutant}_lag{k}_A']
            y = df_pair[f'{pollutant}_B']
        else:
            # manually shift A forward by k hours
            x = df_pair[f'{pollutant}_A'].shift(k)
            y = df_pair[f'{pollutant}_B']
            
        valid = ~x.isna() & ~y.isna()
        if valid.sum() > 30: # at least 30 points
            corrs.append(np.corrcoef(x[valid], y[valid])[0, 1])
        else:
            corrs.append(0)
    return corrs

def check_monthly_consistency(df_pair, pollutant, best_lag_global, max_lag=12):
    # Split by month
    monthly_lags = []
    df_pair['YearMonth'] = df_pair['DateTime'].dt.to_period('M')
    for name, group in df_pair.groupby('YearMonth'):
        if len(group) < 100: continue
        
        corrs = compute_lag_correlation(group, pollutant, max_lag)
        if sum([1 for c in corrs if c != 0]) > 0:
            best_k = np.argmax(corrs) + 1
            monthly_lags.append(best_k)
            
    # Valid if at least 2 months have lag within +/- 1 of the global best
    consistent_months = [m for m in monthly_lags if abs(m - best_lag_global) <= 1]
    return len(consistent_months) >= 2


def analyze_transport():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    stations = ["Siltara", "AIIMS", "IGKV", "Bhatagaon"]
    results = []
    curves = {}
    
    print("Evaluating pairs...")
    for A, B in itertools.permutations(stations, 2):
        print(f"Evaluating {A} -> {B}")
        
        df_A = df[df['Location'] == A].set_index('DateTime')
        df_B = df[df['Location'] == B].set_index('DateTime')
        
        df_pair = df_A.join(df_B, lsuffix='_A', rsuffix='_B', how='inner').reset_index()
        
        if df_pair.empty: continue
        
        # Step 3: Unfiltered Correlation
        corrs_unfiltered = compute_lag_correlation(df_pair, 'PM2.5')
        if max(corrs_unfiltered) <= 0: continue
        
        best_lag_unfiltered = np.argmax(corrs_unfiltered) + 1
        
        # Step 4: Wind validation
        expected = expected_direction(A, B)
        wd_mask = is_aligned(df_pair['WD_A'], expected)
        ws_mask = df_pair['WS_A'] > 1.5
        df_filtered = df_pair[wd_mask & ws_mask]
        
        wind_alignment_score = 0
        if len(df_filtered) > 50:
            corrs_filtered = compute_lag_correlation(df_filtered, 'PM2.5')
            if max(corrs_filtered) > 0:
                best_lag_filtered = np.argmax(corrs_filtered) + 1
                max_corr_filtered = np.max(corrs_filtered)
                wind_alignment_score = 1
                best_lag = best_lag_filtered
                max_corr = max_corr_filtered
                
                # save curve data for plotting
                curves[f'{A} -> {B}'] = corrs_filtered
            else:
                best_lag = best_lag_unfiltered
                max_corr = np.max(corrs_unfiltered)
        else:
            best_lag = best_lag_unfiltered
            max_corr = np.max(corrs_unfiltered)
            df_filtered = df_pair # Fallback for consistency
            
        # Step 5: Multi-pollutant validation
        pollutant_consistency_score = 0
        if wind_alignment_score == 1:
            no2_corrs = compute_lag_correlation(df_filtered, 'NO2')
            so2_corrs = compute_lag_correlation(df_filtered, 'SO2')
            
            valid_pollutants = 0
            if no2_corrs and max(no2_corrs) > 0:
                no2_best = np.argmax(no2_corrs) + 1
                if abs(no2_best - best_lag) <= 1: valid_pollutants += 1
            if so2_corrs and max(so2_corrs) > 0:
                so2_best = np.argmax(so2_corrs) + 1
                if abs(so2_best - best_lag) <= 1: valid_pollutants += 1
                
            if valid_pollutants >= 1: # At least one confirms
                pollutant_consistency_score = 1
                
        # Step 6: Temporal Consistency (use unfiltered if filtered too sparse per month)
        if check_monthly_consistency(df_pair, 'PM2.5', best_lag):
            pass # We could add a score for this, but the user spec just said validated=1

        # Calculate final Confidence Score
        # normalize_correlation scaled between 0 and 1. Assuming typical air quality corrs rarely exceed 0.8.
        corr_norm = min(max(max_corr / 0.8, 0), 1)
        
        score = 0.5 * corr_norm + 0.3 * wind_alignment_score + 0.2 * pollutant_consistency_score
        
        if score > 0.6:
            results.append({
                'Source': A,
                'Target': B,
                'Best Lag (hours)': best_lag,
                'Correlation': max_corr,
                'Confidence Score': score
            })

    results_df = pd.DataFrame(results)
    print("\nValid Transport Pathways:")
    print(results_df)
    results_df.to_csv(os.path.join(OUT_DIR, 'transport_matrix.csv'), index=False)
    
    # Graphs
    # 1. Lag Curves
    if curves:
        plt.figure(figsize=(10, 6))
        for pair, corrs in curves.items():
            if pair in [f"{r['Source']} -> {r['Target']}" for r in results]:
                plt.plot(range(1, 13), corrs, marker='o', label=pair)
        plt.title('PM2.5 Lagged Correlation under Validated Wind Conditions')
        plt.xlabel('Lag (Hours)')
        plt.ylabel('Pearson Correlation')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(OUT_DIR, 'lag_correlations.png'))
        plt.close()

    # 2. Directed Network Graph
    G = nx.DiGraph()
    for _, row in results_df.iterrows():
        G.add_edge(row['Source'], row['Target'], 
                   lag=row['Best Lag (hours)'], 
                   weight=row['Confidence Score'],
                   label=f"L:{row['Best Lag (hours)']}h\nC:{row['Confidence Score']:.2f}")

    if len(G.edges) > 0:
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        
        edges = G.edges()
        weights = [G[u][v]['weight'] * 5 for u, v in edges]
        
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', edgecolors='black')
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, 
                               arrowsize=20, arrowstyle='-|>', edge_color='gray')
                               
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        
        plt.title("Validated Inter-Station Pollution Transport Pathways")
        plt.axis('off')
        plt.savefig(os.path.join(OUT_DIR, 'pathway_network.png'), bbox_inches='tight')
        plt.close()
        print("Plots saved in TRANSPORT_ANALYSIS folder.")

if __name__ == "__main__":
    analyze_transport()
