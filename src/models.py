"""Baseline and Keras MLP architectures for binary purchase-intention prediction."""
import tensorflow as tf
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

def build_baselines(random_state=42):
    """Return trivial majority-class and learned logistic-regression baselines."""
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
    }

def build_mlp(input_dim, variant="A", learning_rate=1e-3):
    """Build one of three increasingly sophisticated tabular MLPs."""
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    if variant == "A":
        x = tf.keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
        x = tf.keras.layers.Dropout(0.20, name="dropout_1")(x)
    elif variant == "B":
        x = tf.keras.layers.Dense(128, activation="relu", name="dense_1")(inputs)
        x = tf.keras.layers.BatchNormalization(name="batch_norm_1")(x)
        x = tf.keras.layers.Dropout(0.35, name="dropout_1")(x)
        x = tf.keras.layers.Dense(64, activation="relu", name="dense_2")(x)
        x = tf.keras.layers.Dropout(0.25, name="dropout_2")(x)
    elif variant == "C":
        x = tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.40)(x)
        x = tf.keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.30)(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.20)(x)
    else:
        raise ValueError("variant must be A, B, or C")
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="purchase_probability")(x)
    model = tf.keras.Model(inputs, outputs, name=f"MLP_{variant}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    return model
