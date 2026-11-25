"""Inspect the downloaded UCI dataset from the repository root."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data/raw/online_shoppers.csv"


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError("Run python data/download_data.py first.")
    df = pd.read_csv(RAW_FILE)
    print("\n1. Dataset shape:")
    print(df.shape)
    print("\n2. Data types:")
    print(df.dtypes)
    print("\n3. Missing values by column:")
    print(df.isna().sum())
    print("\n4. Total missing cells:")
    print(int(df.isna().sum().sum()))
    print("\n5. Exact duplicate rows:")
    print(int(df.duplicated().sum()))
    if "Revenue" not in df.columns:
        raise ValueError("Expected Revenue target column.")
    print("\n6. Revenue counts:")
    print(df["Revenue"].value_counts(dropna=False))
    print("\n7. Revenue percentages:")
    print((df["Revenue"].value_counts(normalize=True, dropna=False) * 100).round(2))


if __name__ == "__main__":
    main()
