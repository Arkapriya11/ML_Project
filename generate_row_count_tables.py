import os
import pandas as pd


BASE_DIR = r"c:\Users\annam\OneDrive\Desktop\ML_Project"
AUDIT_PATH = os.path.join(BASE_DIR, "preprocessing_file_audit.csv")
CLEANED_DIR = os.path.join(BASE_DIR, "DATASETS_CLEANED", "Individual")
OUT_DIR = os.path.join(BASE_DIR, "PREPROCESSING_TABLES")


def build_before_table(audit_df: pd.DataFrame) -> pd.DataFrame:
    use = audit_df[audit_df["status"] == "ok"].copy()
    use["rows_after_datetime_parse"] = pd.to_numeric(
        use["rows_after_datetime_parse"], errors="coerce"
    ).fillna(0)
    before = (
        use.groupby(["location", "frequency"], as_index=False)["rows_after_datetime_parse"]
        .sum()
        .rename(columns={"location": "Region", "frequency": "Frequency", "rows_after_datetime_parse": "Rows_Before_Preprocessing"})
    )
    before["Rows_Before_Preprocessing"] = before["Rows_Before_Preprocessing"].astype(int)
    return before


def build_after_table() -> pd.DataFrame:
    rows = []
    for name in os.listdir(CLEANED_DIR):
        if not name.lower().endswith(".csv"):
            continue
        full = os.path.join(CLEANED_DIR, name)
        stem = os.path.splitext(name)[0]
        region, freq = stem.rsplit("_", 1)
        df = pd.read_csv(full)
        rows.append(
            {
                "Region": region,
                "Frequency": freq,
                "Rows_After_Preprocessing": int(len(df)),
            }
        )
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    audit_df = pd.read_csv(AUDIT_PATH)
    before = build_before_table(audit_df)
    after = build_after_table()

    merged = before.merge(after, on=["Region", "Frequency"], how="outer").fillna(0)
    merged["Rows_Before_Preprocessing"] = merged["Rows_Before_Preprocessing"].astype(int)
    merged["Rows_After_Preprocessing"] = merged["Rows_After_Preprocessing"].astype(int)
    merged["Rows_Change"] = merged["Rows_After_Preprocessing"] - merged["Rows_Before_Preprocessing"]
    merged = merged.sort_values(["Frequency", "Region"]).reset_index(drop=True)

    hourly = merged[merged["Frequency"].str.lower() == "hourly"].copy()
    quarterly = merged[merged["Frequency"].str.lower() == "quarterly"].copy()

    merged.to_csv(os.path.join(OUT_DIR, "rows_before_after_all.csv"), index=False)
    hourly.to_csv(os.path.join(OUT_DIR, "rows_before_after_hourly.csv"), index=False)
    quarterly.to_csv(os.path.join(OUT_DIR, "rows_before_after_quarterly.csv"), index=False)

    # Also export simple text tables for easy viewing without extra dependencies.
    with open(os.path.join(OUT_DIR, "rows_before_after_hourly.txt"), "w", encoding="utf-8") as f:
        f.write(hourly.to_string(index=False))
    with open(os.path.join(OUT_DIR, "rows_before_after_quarterly.txt"), "w", encoding="utf-8") as f:
        f.write(quarterly.to_string(index=False))

    print(f"Saved tables in: {OUT_DIR}")


if __name__ == "__main__":
    main()
