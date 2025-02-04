import tensorflow as tf
from keras import Sequential, layers, metrics


def build_vectorize_layer(
    max_tokens: int, sequence_length: int
) -> layers.TextVectorization:
    vectorizer = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    return vectorizer


def build_model(
    max_tokens: int,
    sequence_length: int,
    embedding_dim: int,
    dropout_rate: float,
    loss: str,
    metrics: list[metrics.Metric],
) -> Sequential:
    model = Sequential(
        [
            layers.Input(
                shape=(sequence_length,), dtype=tf.int32, name="vectorized_text"
            ),
            layers.Embedding(
                input_dim=max_tokens, output_dim=embedding_dim, name="embedding"
            ),
            layers.LSTM(embedding_dim, return_sequences=False, name="lstm"),
            # layers.GlobalAveragePooling1D(),
            # layers.Dropout(dropout_rate),
            # layers.Dense(32, activation="relu"),
            # layers.Dropout(dropout_rate),
            layers.Dense(6, activation="sigmoid", name="toxic_labels"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=loss,
        metrics=metrics,
    )

    return model


def build_export_model(
    vectorize_layer: layers.TextVectorization,
    base_model: Sequential,
    loss: str,
    metrics: list[metrics.Metric],
) -> Sequential:
    if not vectorize_layer.built:
        raise ValueError(
            "The vectorize layer must be built before building the export model."
        )
    if not base_model.built:
        raise ValueError(
            "The base model must be built before building the export model."
        )

    export_model = Sequential(
        [
            vectorize_layer,
            base_model,
        ]
    )
    export_model.compile(
        optimizer="adam",
        loss=loss,
        metrics=metrics,
    )

    return export_model
