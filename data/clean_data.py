from pathlib import Path
import pandas as pd


# Define paths
DATA_DIR = Path(__file__).resolve().parent
RAW_FILE = DATA_DIR / "raw" / "online_shoppers.csv"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_FILE = PROCESSED_DIR / "online_shoppers_clean.csv"


def load_data():
    """Load the raw dataset."""
    return pd.read_csv(RAW_FILE)


def clean_data(df):
    """Check data quality without removing valid sessions."""
    df = df.copy()

    missing_values = df.isna().sum().sum()
    duplicate_rows = df.duplicated().sum()

    print(f"Missing values: {missing_values}")
    print(f"Exact duplicate rows: {duplicate_rows}")

    return df


def main():
    df = load_data()

    print("Original dataset shape:", df.shape)

    cleaned_df = clean_data(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(CLEAN_FILE, index=False)

    print("Processed dataset shape:", cleaned_df.shape)
    print("Clean dataset saved successfully!")


if __name__ == "__main__":
    main()