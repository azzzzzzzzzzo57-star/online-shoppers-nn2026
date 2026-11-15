"""Leakage-safe data pipeline for Online Shoppers Purchasing Intention."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT_DIR / "data" / "raw" / "online_shoppers.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
TARGET, RANDOM_STATE = "Revenue", 42
CATEGORICAL_COLUMNS = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]

def load_data(path=RAW_FILE):
    """Load the raw UCI CSV and validate the target column."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run: python data/download_data.py")
    frame = pd.read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError(f"Expected target column {TARGET!r}.")
    return frame

def clean_data(frame):
    """Remove exact duplicates and normalize the binary target."""
    frame = frame.drop_duplicates().dropna(subset=[TARGET]).copy()
    if frame[TARGET].dtype == object:
        frame[TARGET] = frame[TARGET].astype(str).str.strip().str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
    frame[TARGET] = frame[TARGET].astype(int)
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Revenue must contain only binary values.")
    return frame

def split_data(frame, random_state=RANDOM_STATE):
    """Return stratified 70/15/15 train, validation and test splits."""
    X, y = frame.drop(columns=[TARGET]), frame[TARGET].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=random_state, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_preprocessor(X_train):
    """Create numeric scaling and categorical one-hot transformations."""
    categorical = [c for c in CATEGORICAL_COLUMNS if c in X_train.columns]
    numeric = [c for c in X_train.columns if c not in categorical]
    return ColumnTransformer([("numeric", StandardScaler(), numeric),
                              ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)])

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = clean_data(load_data())
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(frame)
    preprocessor = build_preprocessor(X_train)
    transformed = {
        "train": (preprocessor.fit_transform(X_train), y_train.to_numpy()),
        "validation": (None, y_val.to_numpy()),
        "test": (None, y_test.to_numpy()),
    }
    transformed["validation"] = (preprocessor.transform(X_val), transformed["validation"][1])
    transformed["test"] = (preprocessor.transform(X_test), transformed["test"][1])
    for name, (features, labels) in transformed.items():
        joblib.dump({"X": features, "y": labels}, OUTPUT_DIR / f"{name}_data.joblib")
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    joblib.dump({int(k): float(v) for k, v in zip(np.unique(y_train), weights)}, OUTPUT_DIR / "class_weights.joblib")
    joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.joblib")
    joblib.dump(list(X_train.columns), OUTPUT_DIR / "raw_feature_columns.joblib")
    print(f"Rows after duplicate removal: {len(frame)}")
    print(f"Train/validation/test: {len(y_train)}/{len(y_val)}/{len(y_test)}")
    print(f"Encoded features: {transformed['train'][0].shape[1]}")
    print(f"Train purchase rate: {y_train.mean():.2%}; validation: {y_val.mean():.2%}; test: {y_test.mean():.2%}")
    print("Class weights:", joblib.load(OUTPUT_DIR / "class_weights.joblib"))

if __name__ == "__main__":
    main()
