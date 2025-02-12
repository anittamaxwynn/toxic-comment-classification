from typing import Optional

import keras
import tensorflow as tf

from . import config

logger = config.setup_logger(__name__)


# ----------------------------
# TYPE ALIASES
# ----------------------------


Sequential = keras.models.Sequential
Model = keras.Model
TextVectorizer = keras.layers.TextVectorization
Layer = keras.layers.Layer
Metric = keras.metrics.Metric
Loss = keras.losses.Loss


# ----------------------------
# METRICS
# ----------------------------


def get_metrics() -> list[Metric]:
    return [
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
        keras.metrics.AUC(name="prc", curve="PR"),
    ]


# ----------------------------
# MODEL BUILDERS
# ----------------------------


def build_model(
    vocab_size: int,
    max_length: int,
    embedding_dim: int,
    lstm_units: Optional[int] = None,
    hidden_units: Optional[int] = None,
    dropout_rate: Optional[float] = None,
) -> Model:
    """
    Builds a keras model for classifying toxic comments.
    Inputs are tokenized text, and outputs are the probability of the comment being toxic.
    """
    logger.info(
        f"Building model with vocab_size={vocab_size}, max_length={max_length}, embedding_dim={embedding_dim}, lstm_units={lstm_units}, hidden_units={hidden_units}, dropout_rate={dropout_rate}"
    )

    input_tokens = keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="tokens"
    )
    embedding_layer = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="embedding",
    )

    lstm_layer = (
        keras.layers.LSTM(lstm_units, return_sequences=True, name="lstm")
        if lstm_units
        else keras.layers.Identity()
    )
    hidden_layer = (
        keras.layers.Dense(hidden_units, activation="relu", name="hidden")
        if hidden_units
        else keras.layers.Identity()
    )

    x = embedding_layer(input_tokens)
    x = lstm_layer(x)
    x = keras.layers.GlobalMaxPool1D()(x)
    x = (
        keras.layers.Dropout(dropout_rate) if dropout_rate else keras.layers.Identity()
    )(x)
    x = hidden_layer(x)
    x = (
        keras.layers.Dropout(dropout_rate) if dropout_rate else keras.layers.Identity()
    )(x)

    outputs = keras.layers.Dense(6, activation="sigmoid", name="toxic_labels")(x)

    model = keras.models.Model(
        inputs=input_tokens, outputs=outputs, name="training_model"
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=get_metrics())

    return model


def build_export_model(
    vectorize_layer: TextVectorizer,
    training_model: Model,
) -> Sequential:
    # Check if the vectorize layer and models have been built
    if not vectorize_layer.built:
        raise ValueError(
            "The vectorize layer must be built before building the export model."
        )
    if not training_model.built:
        raise ValueError("The model must be built before building the export model.")

    logger.info("Chaining vectorize layer and training model to build export model...")
    export_model = Sequential(
        [
            keras.layers.Input(shape=(1,), dtype="string", name="raw_text"),
            vectorize_layer,
            training_model,
        ]
    )

    export_model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=get_metrics()
    )
    return export_model


# ----------------------------
# CALLBACKS
# ----------------------------


def early_stopping() -> keras.callbacks.EarlyStopping:
    return keras.callbacks.EarlyStopping(
        monitor="val_prc",
        verbose=1,
        patience=10,
        mode="max",
        restore_best_weights=True,
    )


if __name__ == "__main__":
    test_model = build_model(
        vocab_size=10000,
        max_length=100,
        embedding_dim=32,
        lstm_units=32,
        hidden_units=50,
        dropout_rate=0.2,
    )

    test_model.summary()

    other_model = build_model(
        vocab_size=10000,
        max_length=100,
        embedding_dim=32,
    )

    other_model.summary()
