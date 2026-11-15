"""Final held-out test evaluation and inspection of at least 20 errors."""
from pathlib import Path
import json, time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS, RESULTS = ROOT / "data" / "processed", ROOT / "models", ROOT / "experiments"

def dense(features):
    return features.toarray() if hasattr(features, "toarray") else np.asarray(features)

def main():
    RESULTS.mkdir(exist_ok=True)
    test = joblib.load(DATA / "test_data.joblib")
    x_test, y_test = dense(test["X"]), np.asarray(test["y"]).astype(int)
    selection = json.loads((MODELS / "selected_model.json").read_text(encoding="utf-8"))
    models = {"final_neural_network": tf.keras.models.load_model(MODELS / selection["model_path"])}
    for name in ("majority", "logistic_regression"):
        models[name] = joblib.load(MODELS / f"{name}.joblib")
    summary, final_probabilities = [], None
    for name, model in models.items():
        started = time.perf_counter()
        probabilities = model.predict(x_test, verbose=0).ravel() if name == "final_neural_network" else model.predict_proba(x_test)[:, 1]
        elapsed = time.perf_counter()-started
        predictions = (probabilities >= 0.5).astype(int)
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        summary.append({"model": name, "accuracy": accuracy_score(y_test, predictions),
            "macro_f1": f1_score(y_test, predictions, average="macro"),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "inference_ms_per_sample": elapsed*1000/len(y_test)})
        if name == "final_neural_network":
            final_probabilities = probabilities
            (RESULTS / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            ConfusionMatrixDisplay(confusion_matrix(y_test, predictions), display_labels=["No purchase", "Purchase"]).plot()
            plt.tight_layout(); plt.savefig(RESULTS / "confusion_matrix.png", dpi=160); plt.close()
            fpr, tpr, _ = roc_curve(y_test, probabilities)
            plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, probabilities):.3f}")
            plt.plot([0,1], [0,1], linestyle="--"); plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate"); plt.title("Final model ROC curve"); plt.legend()
            plt.tight_layout(); plt.savefig(RESULTS / "roc_curve.png", dpi=160); plt.close()
    pd.DataFrame(summary).to_csv(RESULTS / "test_metrics.csv", index=False)
    predictions = (final_probabilities >= 0.5).astype(int)
    errors = np.flatnonzero(predictions != y_test)[:max(20, min(100, len(y_test)))]
    pd.DataFrame({"test_row_index": errors, "actual": y_test[errors],
        "predicted": predictions[errors], "purchase_probability": final_probabilities[errors],
        "error_type": np.where(predictions[errors] == 1, "false_positive", "false_negative")
    }).to_csv(RESULTS / "error_analysis.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"Saved {len(errors)} misclassified examples for error analysis.")

if __name__ == "__main__": main()
