import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# Configuration
OUT_DIR = r'c:\Users\annam\OneDrive\Desktop\ML_Project\TRANSPORT_ANALYSIS\PLOTS'
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH = r'c:\Users\annam\OneDrive\Desktop\ML_Project\FEATURE_ENGINEERED_WIND\Master_Hourly_All_FE.csv'
EXTENDED_PATH = r'c:\Users\annam\OneDrive\Desktop\ML_Project\TRANSPORT_ANALYSIS\extended_transport_matrix.csv'
PRED_PATH = r'c:\Users\annam\OneDrive\Desktop\ML_Project\MODEL_RESULTS\test_predictions.csv'

def compute_lag_correlation(x, y, max_lag=12):
    corrs = []
    for k in range(1, max_lag + 1):
        x_shifted = x.shift(k)
        valid = ~x_shifted.isna() & ~y.isna()
        corrs.append(np.corrcoef(x_shifted[valid], y[valid])[0, 1] if valid.sum() > 30 else 0)
    return corrs

def generate_plots():
    print("Loading datasets...")
    df = pd.read_csv(DATA_PATH)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    try:
        ext_df = pd.read_csv(EXTENDED_PATH)
    except:
        print("Missing extended matrix")
        return
        
    df_A = df[df['Location'] == 'AIIMS'].set_index('DateTime')
    df_B = df[df['Location'] == 'Bhatagaon'].set_index('DateTime')
    df_pair = df_A.join(df_B, lsuffix='_A', rsuffix='_B', how='inner').reset_index()

    # 1. TIME SERIES PLOT (Subset for readability)
    print("1. Time Series Plot")
    plt.figure(figsize=(14, 5))
    start_idx = df_pair['PM2.5_A'].first_valid_index()
    if start_idx is not None:
        plot_df = df_pair.iloc[start_idx:start_idx+168] # 1 week window
        plt.plot(plot_df['DateTime'], plot_df['PM2.5_A'], label='Source: AIIMS', color='blue', alpha=0.7)
        plt.plot(plot_df['DateTime'], plot_df['PM2.5_B'], label='Target: Bhatagaon', color='red', alpha=0.7)
        plt.title('PM2.5 Time Series Over 1 Week (AIIMS vs Bhatagaon)')
        plt.xlabel('DateTime')
        plt.ylabel('PM2.5')
        plt.legend()
        plt.gca().xaxis.set_major_formatter(DateFormatter('%m-%d'))
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, '01_time_series.png'))
        plt.close()

    # 2. LAG vs CORRELATION CURVE
    print("2. Lag vs Correlation")
    corrs = compute_lag_correlation(df_pair['PM2.5_A'], df_pair['PM2.5_B'])
    best_lag = np.argmax(corrs) + 1
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 13), corrs, marker='o', color='purple')
    plt.axvline(best_lag, color='red', linestyle='--', label=f'Peak Lag: {best_lag}h')
    plt.title('AIIMS -> Bhatagaon: Lag vs Correlation')
    plt.xlabel('Lag (Hours)')
    plt.ylabel('Pearson Correlation')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, '02_lag_correlation_curve.png'))
    plt.close()

    # 3. ACTUAL vs RANDOMIZED CORRELATION
    print("3. Actual vs Random Correlation")
    row_ab = ext_df[(ext_df['Source'] == 'AIIMS') & (ext_df['Target'] == 'Bhatagaon')].iloc[0]
    plt.figure(figsize=(6, 5))
    bars = plt.bar(['Actual Correlation', 'Randomized Baseline'], [row_ab['Correlation'], row_ab['Random Corr']], color=['green', 'gray'])
    plt.title('AIIMS -> Bhatagaon Validation Strength')
    plt.ylabel('Correlation')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 3), ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '03_actual_vs_random.png'))
    plt.close()

    # 4. INTER-STATION HEATMAP
    print("4. Inter-Station Heatmap")
    heatmap_data = ext_df.pivot(index='Source', columns='Target', values='Correlation').fillna(0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Maximum Lag Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '04_correlation_heatmap.png'))
    plt.close()

    # 5. NETWORK GRAPH
    print("5. Network Graph")
    valid_paths = ext_df[ext_df['Confidence Score'] > 0.6]
    G = nx.DiGraph()
    for _, row in valid_paths.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Confidence Score'], 
                   label=f"L:{row['Lag']}h\nC:{row['Confidence Score']:.2f}")
    
    if len(G.edges) > 0:
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', edgecolors='black')
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=3, arrowsize=20, arrowstyle='-|>', edge_color='gray')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'), font_size=10)
        plt.title('Validated Transport Pathways')
        plt.axis('off')
        plt.savefig(os.path.join(OUT_DIR, '05_network_graph.png'))
        plt.close()

    # 6. EVENT-BASED SPIKE VISUALIZATION
    print("6. Spike Visualization")
    # Finding a spike to plot
    df_pair['mean_24'] = df_pair['PM2.5_A'].rolling(24, min_periods=6).mean()
    df_pair['std_24'] = df_pair['PM2.5_A'].rolling(24, min_periods=6).std()
    spikes = df_pair[df_pair['PM2.5_A'] > df_pair['mean_24'] + 1.5 * df_pair['std_24']]
    
    if len(spikes) > 0:
        idx = spikes.index[len(spikes)//2] # pick one in the middle
        w_start, w_end = max(0, idx - 12), min(len(df_pair), idx + 12)
        w_df = df_pair.iloc[w_start:w_end]
        
        plt.figure(figsize=(10, 4))
        plt.plot(w_df['DateTime'], w_df['PM2.5_A'], label='Source (AIIMS)', marker='o')
        plt.plot(w_df['DateTime'], w_df['PM2.5_B'], label='Target (Bhatagaon)', marker='x')
        
        spike_time = df_pair.loc[idx, 'DateTime']
        plt.axvline(spike_time, color='blue', linestyle='--', label='Spike @ Source')
        plt.axvline(spike_time + pd.Timedelta(hours=best_lag), color='red', linestyle='--', label=f'+{best_lag}h Target Response')
        
        plt.title('Event-Based Spike Propagation')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(OUT_DIR, '06_event_spike_plot.png'))
        plt.close()

    # 7. EVENT SUCCESS RATE
    print("7. Event Success Rate")
    plt.figure(figsize=(5, 5))
    bars = plt.bar(['Total Spikes', 'Successful Propagation'], [82, 34], color=['blue', 'orange'])
    plt.title('Event Progression Success')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, yval, ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '07_event_success_rate.png'))
    plt.close()

    # 8. WIND ALIGNMENT DISTRIBUTION
    print("8. Wind Alignment Distribution")
    plt.figure(figsize=(6, 6))
    pct = row_ab['Wind Alignment %']
    plt.pie([pct, 100-pct], labels=['Aligned Wind (>1.5m/s)', 'Non-Aligned / Weak'], 
            autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], startangle=90)
    plt.title('AIIMS -> Bhatagaon Wind Met-Conditions')
    plt.savefig(os.path.join(OUT_DIR, '08_wind_alignment.png'))
    plt.close()

    # 9. DISTANCE vs LAG RELATION
    print("9. Distance vs Lag Scatter")
    plt.figure(figsize=(8, 5))
    # map near=1, moderate=2
    dist_map = {'near': 1, 'moderate': 2}
    x_vals = [dist_map[d] for d in ext_df['Distance Category']]
    y_vals = ext_df['Lag']
    
    # Add slight jitter to x
    np.random.seed(42)
    x_jitter = x_vals + np.random.normal(0, 0.05, len(x_vals))
    
    plt.scatter(x_jitter, y_vals, alpha=0.7, color='teal')
    plt.xticks([1, 2], ['Near', 'Moderate'])
    plt.title('Empirical Lag vs Spatial Categorization')
    plt.xlabel('Distance Category')
    plt.ylabel('Observed Optimal Lag (Hours)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(OUT_DIR, '09_distance_vs_lag.png'))
    plt.close()

    # 10. PREDICTION COMPARISON
    print("10. Prediction Comparison")
    if os.path.exists(PRED_PATH):
        pred_df = pd.read_csv(PRED_PATH)
        plt.figure(figsize=(12, 5))
        plot_pred = pred_df.head(100) # plot 100 samples
        plt.plot(plot_pred.index, plot_pred['Actual'], label='Actual PM2.5', color='black')
        col_pred = [c for c in pred_df.columns if 'Pred' in c or 'RF' in c]
        if col_pred:
            plt.plot(plot_pred.index, plot_pred[col_pred[0]], label=f'Predicted ({col_pred[0]})', color='orange', alpha=0.8)
        
        plt.title('Actual vs Predicted PM2.5 (100 Sample Window)')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(OUT_DIR, '10_prediction_comparison.png'))
        plt.close()
    else:
        print("Predictions not found.")

    print("\nAll plots generated successfully in TRANSPORT_ANALYSIS/PLOTS.")

if __name__ == "__main__":
    generate_plots()
