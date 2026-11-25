"""Week 10 majority and logistic-regression baselines; validation only."""
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
MODELS = ROOT / "models"
EXPERIMENTS = ROOT / "experiments"


def load_split(name):
    item = joblib.load(DATA / f"{name}_data.joblib")
    return item["X"], np.asarray(item["y"], dtype=int)


def metrics(y, pred, score=None):
    recalls = recall_score(y, pred, labels=[0, 1], average=None, zero_division=0)
    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "purchase_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "purchase_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "per_class_recall": {"no_purchase_0": float(recalls[0]), "purchase_1": float(recalls[1])},
        "confusion_matrix_labels_0_then_1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y, pred, labels=[0, 1], target_names=["no_purchase", "purchase"],
            output_dict=True, zero_division=0
        ),
    }
    if score is not None and len(np.unique(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, score))
    return result


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")

    majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    majority_metrics = metrics(y_val, majority.predict(X_val))

    plain = LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)
    plain.fit(X_train, y_train)
    plain_metrics = metrics(y_val, plain.predict(X_val), plain.predict_proba(X_val)[:, 1])

    balanced = LogisticRegression(
        max_iter=2000, solver="liblinear", class_weight="balanced", random_state=42
    )
    balanced.fit(X_train, y_train)
    balanced_metrics = metrics(y_val, balanced.predict(X_val), balanced.predict_proba(X_val)[:, 1])

    joblib.dump(plain, MODELS / "week10_logistic_unweighted.joblib")
    joblib.dump(balanced, MODELS / "week10_logistic_balanced.joblib")
    result = {
        "evaluation_split": "validation",
        "primary_metric": "macro_f1",
        "test_set_used": False,
        "models": {
            "majority_class": majority_metrics,
            "logistic_regression_unweighted": plain_metrics,
            "logistic_regression_class_weight_balanced": balanced_metrics,
        },
        "imbalance_ablation": {
            "technique": "class_weight='balanced'",
            "purchase_recall_change": balanced_metrics["purchase_recall"] - plain_metrics["purchase_recall"],
            "comparison": "same logistic model and same validation split, with vs without training class weights",
        },
    }
    output = EXPERIMENTS / "week10_baseline_validation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {output}. Test split was not loaded.")


if __name__ == "__main__":
    main()
