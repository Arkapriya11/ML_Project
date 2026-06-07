import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = r"c:\Users\annam\OneDrive\Desktop\ML_Project"
DATA_PATH = os.path.join(BASE_DIR, "FEATURE_ENGINEERED", "Master_Hourly_All_FE.csv")
OUT_DIR = os.path.join(BASE_DIR, "REQUESTED_GRAPHS", "LOCATION_WISE_MET_ANALYSIS")


def get_feature_list(df: pd.DataFrame):
    # Prefer meteorological-style names if present; otherwise use environmental + temporal proxies.
    preferred_patterns = [
        "temp",
        "temperature",
        "humid",
        "rh",
        "wind",
        "ws",
        "wd",
        "rain",
        "pressure",
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    met_cols = []
    for cl, original in cols_lower.items():
        if any(p in cl for p in preferred_patterns):
            met_cols.append(original)

    if met_cols:
        return sorted(set(met_cols))

    fallback = [
        "NO",
        "NO2",
        "NOx",
        "SO2",
        "CO",
        "NH3",
        "Hour",
        "Month",
        "Weekday",
        "Hour_sin",
        "Hour_cos",
        "Month_sin",
        "Month_cos",
        "is_imputed",
        "gap_length",
    ]
    return [c for c in fallback if c in df.columns]


def save_location_correlation_heatmaps(df: pd.DataFrame, features: list[str], target: str):
    recs = []
    for loc, part in df.groupby("Location"):
        use_cols = features + [target]
        sub = part[use_cols].copy()
        for c in use_cols:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub = sub.dropna()
        if len(sub) < 30:
            continue

        corr = sub.corr(numeric_only=True)
        vec = corr[target].drop(labels=[target], errors="ignore").sort_values(ascending=False)
        vec_df = vec.to_frame(name=f"Corr_with_{target}")

        plt.figure(figsize=(6, max(4, 0.35 * len(vec_df))))
        sns.heatmap(vec_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={"label": "Correlation"})
        plt.title(f"{target} vs Features Correlation - {loc}")
        plt.tight_layout()
        png_path = os.path.join(OUT_DIR, f"{loc}_{target}_feature_corr_heatmap.png")
        plt.savefig(png_path, dpi=150)
        plt.close()

        # Keep top positive and negative links for summary analysis.
        top_pos = vec.sort_values(ascending=False).head(3)
        top_neg = vec.sort_values(ascending=True).head(3)
        for feat, val in top_pos.items():
            recs.append({"Location": loc, "Target": target, "Feature": feat, "Correlation": float(val), "Type": "Top_Positive"})
        for feat, val in top_neg.items():
            recs.append({"Location": loc, "Target": target, "Feature": feat, "Correlation": float(val), "Type": "Top_Negative"})

    return recs


def save_overall_pollution_by_location_heatmap(df: pd.DataFrame):
    pivot = (
        df.groupby("Location")[["PM2.5", "PM10"]]
        .mean()
        .sort_index()
    )
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={"label": "Average Concentration"})
    plt.title("Average Pollution by Location")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "location_vs_pollution_average_heatmap.png"), dpi=150)
    plt.close()

    pivot.to_csv(os.path.join(OUT_DIR, "location_vs_pollution_average_table.csv"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    needed = ["Location", "PM2.5", "PM10"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    features = get_feature_list(df)
    if len(features) == 0:
        raise ValueError("No meteorological/environmental features found for analysis.")

    analysis_rows = []
    analysis_rows.extend(save_location_correlation_heatmaps(df, features, target="PM2.5"))
    analysis_rows.extend(save_location_correlation_heatmaps(df, features, target="PM10"))

    save_overall_pollution_by_location_heatmap(df)

    pd.DataFrame(analysis_rows).to_csv(
        os.path.join(OUT_DIR, "location_wise_pollution_feature_analysis.csv"),
        index=False,
    )
    pd.DataFrame({"Features_Used": features}).to_csv(
        os.path.join(OUT_DIR, "features_used_for_location_heatmaps.csv"),
        index=False,
    )

    print(f"Saved location-wise heatmaps and analysis in: {OUT_DIR}")


if __name__ == "__main__":
    main()
