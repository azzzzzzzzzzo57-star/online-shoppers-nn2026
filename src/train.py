"""Train Model A after the Week 10 baselines; evaluate on validation only."""
from pathlib import Path
import json
import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from src.models import build_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
MODELS = ROOT / "models"
EXPERIMENTS = ROOT / "experiments"


def dense(features):
    return features.toarray() if hasattr(features, "toarray") else np.asarray(features)


def evaluate_metrics(y_true, probabilities):
    y_pred = (probabilities >= 0.5).astype(int)
    recalls = recall_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "purchase_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "purchase_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "per_class_recall": {
            "no_purchase_0": float(recalls[0]),
            "purchase_1": float(recalls[1]),
        },
        "confusion_matrix_labels_0_then_1": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["no_purchase", "purchase"],
            output_dict=True, zero_division=0,
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def main():
    tf.keras.utils.set_random_seed(42)
    MODELS.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)

    train = joblib.load(DATA / "train_data.joblib")
    validation = joblib.load(DATA / "validation_data.joblib")
    X_train, y_train = dense(train["X"]), np.asarray(train["y"], dtype=int)
    X_val, y_val = dense(validation["X"]), np.asarray(validation["y"], dtype=int)
    weights = joblib.load(DATA / "class_weights.joblib")

    model = build_model(X_train.shape[1])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=128,
        class_weight=weights,
        callbacks=callbacks,
        verbose=1,
    )
    probabilities = model.predict(X_val, verbose=0).ravel()
    result = {
        "model": "Model A — feed-forward neural network",
        "evaluation_split": "validation",
        "test_set_used": False,
        "epochs_trained": len(history.history["loss"]),
        "early_stopping": "monitor val_loss, patience 5, restore best weights",
        "class_weight": weights,
        "metrics": evaluate_metrics(y_val, probabilities),
    }
    model.save(MODELS / "week10_model_a.keras")
    (EXPERIMENTS / "week10_model_a_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    print("Saved Model A and validation metrics. Test set was not loaded.")


if __name__ == "__main__":
    main()
