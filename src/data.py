
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT_DIR / "data" / "processed" / "online_shoppers_clean.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"

TARGET = "Revenue"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data():
    """Load the processed dataset."""
    return pd.read_csv(DATA_FILE)


def clean_data(df):
    """Remove exact duplicate rows."""
    return df.drop_duplicates().copy()


def split_data(df):
    """Split features and target."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def build_preprocessor(X_train):
    """Create preprocessing transformations."""

    categorical_columns = [
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend"
    ]

    numerical_columns = [
        column for column in X_train.columns
        if column not in categorical_columns
    ]

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numerical_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            )
        ]
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = clean_data(load_data())

    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor(X_train)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    joblib.dump(
        {"X": X_train_transformed, "y": y_train.to_numpy()},
        OUTPUT_DIR / "train_data.joblib"
    )

    joblib.dump(
        {"X": X_test_transformed, "y": y_test.to_numpy()},
        OUTPUT_DIR / "test_data.joblib"
    )

    joblib.dump(
        preprocessor,
        OUTPUT_DIR / "preprocessor.joblib"
    )

    print("Data pipeline completed successfully!")
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Training features:", X_train_transformed.shape[1])
    print("Testing features:", X_test_transformed.shape[1])
    print("Training purchase rate:", round(y_train.mean() * 100, 2), "%")
    print("Testing purchase rate:", round(y_test.mean() * 100, 2), "%")


if __name__ == "__main__":
    main()
