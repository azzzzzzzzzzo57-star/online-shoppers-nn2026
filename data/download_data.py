
"""Download the UCI Online Shoppers Purchasing Intention dataset."""

from pathlib import Path
from ucimlrepo import fetch_ucirepo

RAW_DIR = Path(__file__).resolve().parent / "raw"


def main() -> None:
    RAW_DIR.mkdir(exist_ok=True)

    ds = fetch_ucirepo(id=468)

    df = ds.data.features.join(ds.data.targets)

    df.to_csv(RAW_DIR / "online_shoppers.csv", index=False)

    print(f"Saved {len(df)} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    main()
