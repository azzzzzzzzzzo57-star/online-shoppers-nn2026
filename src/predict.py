"""Raw-row inference using the exact training-time preprocessing object."""
from pathlib import Path
import json, time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR, DATA_DIR = ROOT / "models", ROOT / "data" / "processed"

def load_predictor():
    selection = json.loads((MODEL_DIR / "selected_model.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(MODEL_DIR / selection["model_path"])
    preprocessor = joblib.load(DATA_DIR / "preprocessor.joblib")
    columns = joblib.load(DATA_DIR / "raw_feature_columns.joblib")
    return model, preprocessor, columns

def predict_raw(raw_input):
    """Accept a DataFrame or dict representing one raw session."""
    frame = raw_input.copy() if isinstance(raw_input, pd.DataFrame) else pd.DataFrame([raw_input])
    model, preprocessor, columns = load_predictor()
    missing = [column for column in columns if column not in frame.columns]
    if missing: raise ValueError(f"Missing required raw features: {missing}")
    transformed = preprocessor.transform(frame[columns])
    if hasattr(transformed, "toarray"): transformed = transformed.toarray()
    started = time.perf_counter()
    probability = float(model.predict(np.asarray(transformed), verbose=0).ravel()[0])
    elapsed_ms = (time.perf_counter()-started)*1000
    return {"prediction": "purchase" if probability >= 0.5 else "no_purchase",
        "confidence": probability if probability >= 0.5 else 1-probability,
        "purchase_probability": probability, "inference_time_ms": elapsed_ms}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict purchase intention from CSV rows.")
    parser.add_argument("csv_file", help="CSV containing original raw feature columns")
    args = parser.parse_args()
    for _, row in pd.read_csv(args.csv_file).iterrows():
        print(json.dumps(predict_raw(row.to_dict()), indent=2))
