"""Train baselines and three MLP variants; choose by validation macro-F1."""
from pathlib import Path
import csv, json, random, time
import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from src.models import build_baselines, build_mlp

ROOT = Path(__file__).resolve().parents[1]
DATA, MODEL_DIR, EXPERIMENT_DIR = ROOT / "data" / "processed", ROOT / "models", ROOT / "experiments"

def dense(features):
    return features.toarray() if hasattr(features, "toarray") else np.asarray(features)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)

def main():
    MODEL_DIR.mkdir(exist_ok=True); EXPERIMENT_DIR.mkdir(exist_ok=True)
    train = joblib.load(DATA / "train_data.joblib")
    val = joblib.load(DATA / "validation_data.joblib")
    x_train, y_train = dense(train["X"]), np.asarray(train["y"]).astype(int)
    x_val, y_val = dense(val["X"]), np.asarray(val["y"]).astype(int)
    class_weights = joblib.load(DATA / "class_weights.joblib")
    rows = []
    for name, estimator in build_baselines().items():
        started = time.perf_counter(); estimator.fit(x_train, y_train)
        probabilities = estimator.predict_proba(x_val)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        rows.append({"run_id": name, "model": name, "seed": 42,
            "training_seconds": round(time.perf_counter()-started, 3), "parameters": "",
            "val_accuracy": accuracy_score(y_val, predictions),
            "val_macro_f1": f1_score(y_val, predictions, average="macro"),
            "val_auc": roc_auc_score(y_val, probabilities)})
        joblib.dump(estimator, MODEL_DIR / f"{name}.joblib")
    best_score, best_path, best_name = -1.0, None, None
    for variant in ("A", "B", "C"):
        set_seed(42); model = build_mlp(x_train.shape[1], variant=variant)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ]
        started = time.perf_counter()
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
            epochs=100, batch_size=128, class_weight=class_weights, callbacks=callbacks, verbose=1)
        elapsed = time.perf_counter()-started
        probs = model.predict(x_val, verbose=0).ravel(); preds = (probs >= 0.5).astype(int)
        macro_f1 = f1_score(y_val, preds, average="macro")
        path = MODEL_DIR / f"mlp_{variant}.keras"; model.save(path)
        np.savez(EXPERIMENT_DIR / f"mlp_{variant}_history.npz", **history.history)
        rows.append({"run_id": f"mlp_{variant}_seed42", "model": f"MLP {variant}",
            "seed": 42, "training_seconds": round(elapsed, 3), "parameters": model.count_params(),
            "val_accuracy": accuracy_score(y_val, preds), "val_macro_f1": macro_f1,
            "val_auc": roc_auc_score(y_val, probs)})
        if macro_f1 > best_score: best_score, best_path, best_name = macro_f1, path, f"MLP_{variant}"
    with (EXPERIMENT_DIR / "experiment_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    (MODEL_DIR / "selected_model.json").write_text(json.dumps({
        "model_name": best_name, "model_path": best_path.name,
        "selection_metric": "validation_macro_f1", "validation_macro_f1": best_score}, indent=2), encoding="utf-8")
    print(f"Selected {best_name} using validation macro-F1={best_score:.4f}")
    print("The test set was not loaded or used during model selection.")

if __name__ == "__main__": main()
