"""Model A: a compact feed-forward neural network for tabular session data."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=(input_dim,), name="session_features"),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.15),
        layers.Dense(1, activation="sigmoid", name="purchase_probability"),
    ], name="online_shoppers_model_a")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model
