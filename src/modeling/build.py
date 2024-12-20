import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

from ..data.convert_to_tf import load_tf_datasets


def build_model(
    training_data: tf.data.Dataset, max_tokens=10000, output_sequence_length=100
):
    """Builds a text classification model with vectorization."""
    # Define the vectorization layer
    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=output_sequence_length,
    )

    # Adapt the vectorization layer to the training data
    vectorizer.adapt(training_data.map(lambda x, y: x))  # Extract only the text inputs

    # Build the model
    text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="text")
    vectorized_text = vectorizer(text_input)
    x = tf.keras.layers.Embedding(max_tokens, 128)(vectorized_text)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    output = tf.keras.layers.Dense(6, activation="sigmoid")(x)

    model = tf.keras.Model(text_input, output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    train_ds, _, _ = load_tf_datasets()
    model = build_model(train_ds)
    print(model.summary())
