import keras
from keras import Sequential, layers


def build_vectorize_layer(
    max_tokens: int = 10000, output_sequence_length: int = 100
) -> keras.layers.TextVectorization:
    vectorizer = keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=output_sequence_length,
    )
    return vectorizer


def build_model(
    max_tokens: int = 10000,
    output_sequence_length: int = 100,
    embedding_dim: int = 16,
) -> Sequential:
    model = Sequential(
        [
            layers.Embedding(max_tokens, embedding_dim),
            # layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.GlobalAveragePooling1D(),
            # layers.Dropout(0.1),
            layers.Dense(50, activation="relu"),
            # layers.Dropout(0.1),
            layers.Dense(6, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def build_export_model(vectorize_layer, model) -> Sequential:
    export_model = Sequential([vectorize_layer, model, layers.Activation("sigmoid")])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return export_model

    # text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="text")
    # vectorized_text = vectorizer(text_input)
    # x = tf.keras.layers.Embedding(max_tokens, 128)(vectorized_text)
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # x = tf.keras.layers.Dense(128, activation="relu")(x)
    # output = tf.keras.layers.Dense(6, activation="sigmoid")(x)
    #
    #


if __name__ == "__main__":
    pass
