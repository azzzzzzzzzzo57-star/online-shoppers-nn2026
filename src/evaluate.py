"""One final held-out test evaluation after model selection on validation."""
from pathlib import Path
import json
import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def main():
    # Candidate selection uses validation macro-F1 only; test labels are not read here.
    baseline_file = EXPERIMENTS / "week10_baseline_validation.json"
    model_a_file = EXPERIMENTS / "week10_model_a_validation.json"
    if not baseline_file.exists() or not model_a_file.exists():
        raise FileNotFoundError("Run python -m src.run_week10 before final evaluation.")

    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    model_a = json.loads(model_a_file.read_text(encoding="utf-8"))
    candidates = {
        "logistic_regression_unweighted": baseline["models"]["logistic_regression_unweighted"]["macro_f1"],
        "logistic_regression_class_weight_balanced": baseline["models"]["logistic_regression_class_weight_balanced"]["macro_f1"],
        "model_a_neural_network": model_a["metrics"]["macro_f1"],
    }
    selected_name = max(candidates, key=candidates.get)

    test = joblib.load(ROOT / "data/processed/test_data.joblib")
    X_test = test["X"]
    y_test = np.asarray(test["y"], dtype=int)

    if selected_name == "model_a_neural_network":
        model = tf.keras.models.load_model(ROOT / "models/week10_model_a.keras")
        X_test = X_test.toarray() if hasattr(X_test, "toarray") else np.asarray(X_test)
        probabilities = model.predict(X_test, verbose=0).ravel()
        model_path = "models/week10_model_a.keras"
    else:
        model_file = "week10_logistic_unweighted.joblib" if selected_name == "logistic_regression_unweighted" else "week10_logistic_balanced.joblib"
        model = joblib.load(ROOT / "models" / model_file)
        probabilities = model.predict_proba(X_test)[:, 1]
        model_path = f"models/{model_file}"

    predictions = (probabilities >= 0.5).astype(int)
    result = {
        "warning": "One-time final held-out test evaluation. Do not use these results for further tuning.",
        "selection_rule": "highest validation macro-F1",
        "validation_candidate_macro_f1": candidates,
        "selected_model": selected_name,
        "selected_model_path": model_path,
        "test_macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_confusion_matrix_labels_0_then_1": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "test_classification_report": classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=["no_purchase", "purchase"],
            output_dict=True, zero_division=0,
        ),
    }
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENTS / "final_test_evaluation.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved final test report to {path}")


if __name__ == "__main__":
    main()
