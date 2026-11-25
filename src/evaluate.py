"""Final test evaluation. Run once, only after all model choices are fixed."""
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def main():
    model_path = ROOT / "models/week10_logistic_balanced.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Run python -m src.baselines first.")
    test = joblib.load(ROOT / "data/processed/test_data.joblib")
    X_test, y_test = test["X"], np.asarray(test["y"], dtype=int)
    model = joblib.load(model_path)
    pred = model.predict(X_test)
    score = model.predict_proba(X_test)[:, 1]
    result = {
        "warning": "Final held-out test evaluation. Do not use this result for further tuning.",
        "split": "test",
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, score)),
        "confusion_matrix_labels_0_then_1": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, pred, labels=[0, 1], target_names=["no_purchase", "purchase"],
            output_dict=True, zero_division=0
        ),
    }
    out = ROOT / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "final_test_evaluation.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved final test report to {path}")


if __name__ == "__main__":
    main()
