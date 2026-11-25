"""Leakage-safe preprocessing for the UCI Online Shoppers dataset."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data/raw/online_shoppers.csv"
OUT = ROOT / "data/processed"
TARGET = "Revenue"
SEED = 42
CATEGORICAL = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]


def load_data(path=RAW_FILE):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run python data/download_data.py first.")
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError("Required target column Revenue is missing.")
    return df


def clean_data(df):
    df = df.copy()
    before = len(df)
    missing_before = df.isna().sum().astype(int).to_dict()
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().dropna(subset=[TARGET]).copy()
    values = df[TARGET]
    if values.dtype == object or str(values.dtype) == "bool":
        normalized = values.astype(str).str.strip().str.lower()
        mapped = normalized.map({"true": 1, "false": 0, "1": 1, "0": 0})
        if mapped.isna().any():
            raise ValueError(f"Unexpected Revenue labels: {normalized[mapped.isna()].unique().tolist()}")
        df[TARGET] = mapped.astype(int)
    else:
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any() or not set(numeric.unique()).issubset({0, 1}):
            raise ValueError("Revenue must contain only 0/1 or False/True.")
        df[TARGET] = numeric.astype(int)
    if df.empty or df[TARGET].nunique() != 2:
        raise ValueError("Both Revenue classes must remain after cleaning.")
    report = {
        "rows_before": int(before),
        "exact_duplicates_removed": duplicates,
        "missing_target_rows_removed": int(missing_before.get(TARGET, 0)),
        "rows_after": int(len(df)),
        "missing_values_before_by_column": missing_before,
        "missing_feature_values_after": int(df.drop(columns=[TARGET]).isna().sum().sum()),
    }
    return df.reset_index(drop=True), report


def split_data(df):
    X, y = df.drop(columns=[TARGET]), df[TARGET].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X_train):
    cat = [c for c in CATEGORICAL if c in X_train.columns]
    num = [c for c in X_train.columns if c not in cat]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, num),
        ("categorical", categorical_pipe, cat),
    ])


def counts(y):
    return {str(int(k)): int(v) for k, v in y.value_counts().sort_index().items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, cleaning = clean_data(load_data())
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Fit all learned preprocessing on training data only.
    prep = build_preprocessor(X_train)
    X_train_t = prep.fit_transform(X_train)
    X_val_t = prep.transform(X_val)
    X_test_t = prep.transform(X_test)

    for name, X, y in [
        ("train", X_train_t, y_train),
        ("validation", X_val_t, y_val),
        ("test", X_test_t, y_test),
    ]:
        joblib.dump({"X": X, "y": y.to_numpy()}, OUT / f"{name}_data.joblib")

    classes = np.unique(y_train)
    weight_values = compute_class_weight("balanced", classes=classes, y=y_train.to_numpy())
    weights = {int(k): float(v) for k, v in zip(classes, weight_values)}
    joblib.dump(weights, OUT / "class_weights.joblib")
    joblib.dump(prep, OUT / "preprocessor.joblib")
    joblib.dump(list(X_train.columns), OUT / "raw_feature_columns.joblib")

    summary = {
        "seed": SEED,
        "split_ratio": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "split_rows": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "class_counts": {"train": counts(y_train), "validation": counts(y_val), "test": counts(y_test)},
        "purchase_rate": {"train": float(y_train.mean()), "validation": float(y_val.mean()), "test": float(y_test.mean())},
        "class_weights_from_train_only": weights,
        "encoded_feature_count": int(X_train_t.shape[1]),
        "cleaning": cleaning,
        "leakage_checks": {
            "preprocessor_fit": "X_train only",
            "validation_and_test": "transform only; never fit",
            "split_method": "stratified random split; dataset has no entity ID or time-series order key",
        },
    }
    (OUT / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved processed splits and artifacts in {OUT}")


if __name__ == "__main__":
    main()
