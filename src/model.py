import tensorflow as tf
from keras import Sequential, layers


def build_vectorize_layer(
    max_tokens: int, sequence_length: int
) -> layers.TextVectorization:
    vectorizer = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    return vectorizer


def build_and_compile_model(
    max_tokens: int,
    sequence_length: int,
    embedding_dim: int,
    dropout_rate: float,
) -> Sequential:
    model = Sequential(
        [
            layers.Input(shape=(sequence_length,), dtype=tf.int32, name="vectorized_text"),
            layers.Embedding(max_tokens, embedding_dim),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(dropout_rate),
            layers.Dense(50, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(6, activation="sigmoid"),
        ]
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[tf.metrics.BinaryAccuracy(threshold=0.5)],
    )

    return model
