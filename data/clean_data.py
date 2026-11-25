"""Deterministic row-level cleaning and data-quality report.

Learned feature imputation/scaling/encoding is intentionally handled later
inside src.data after the train/validation/test split.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data/raw/online_shoppers.csv"
OUT_DIR = ROOT / "data/processed"
CLEAN_FILE = OUT_DIR / "online_shoppers_clean.csv"


def load_data():
    if not RAW_FILE.exists():
        raise FileNotFoundError("Run python data/download_data.py first.")
    return pd.read_csv(RAW_FILE)


def clean_data(df):
    if "Revenue" not in df.columns:
        raise ValueError("Expected Revenue target column.")
    result = df.copy()
    before = len(result)
    missing = result.isna().sum().astype(int).to_dict()
    duplicate_count = int(result.duplicated().sum())
    result = result.drop_duplicates().dropna(subset=["Revenue"]).copy()

    labels = result["Revenue"]
    if labels.dtype == object or str(labels.dtype) == "bool":
        normalized = labels.astype(str).str.strip().str.lower()
        mapped = normalized.map({"true": 1, "false": 0, "1": 1, "0": 0})
        if mapped.isna().any():
            raise ValueError("Revenue contains labels other than True/False or 1/0.")
        result["Revenue"] = mapped.astype(int)
    else:
        numeric = pd.to_numeric(labels, errors="coerce")
        if numeric.isna().any() or not set(numeric.unique()).issubset({0, 1}):
            raise ValueError("Revenue must contain only binary labels.")
        result["Revenue"] = numeric.astype(int)

    print(f"Rows before cleaning: {before}")
    print(f"Exact duplicate rows removed: {duplicate_count}")
    print(f"Missing values before cleaning: {sum(missing.values())}")
    print(f"Rows after row-level cleaning: {len(result)}")
    print(f"Missing feature values to impute after train split: {result.drop(columns=['Revenue']).isna().sum().sum()}")
    return result.reset_index(drop=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned = clean_data(load_data())
    cleaned.to_csv(CLEAN_FILE, index=False)
    print(f"Saved row-cleaned data to {CLEAN_FILE}")
    print("Feature imputers/scalers/encoders are fitted on training data by src.data.")


if __name__ == "__main__":
    main()
