import keras
import tensorflow as tf


def decode_vectorized_text(
    layer: keras.layers.TextVectorization, text: tf.Tensor | list
) -> str:
    """Decodes a vectorized text tensor or list of strings to a string."""
    # Check that the layer has been built
    if not layer.built:
        raise ValueError("Layer has not been built yet.")

    # Convert the text tensor to a list of strings
    if isinstance(text, tf.Tensor):
        text = text.numpy().tolist()

    vocab = layer.get_vocabulary()

    decoded_text = " ".join([vocab[idx] for idx in text if idx != 0])
    return decoded_text
