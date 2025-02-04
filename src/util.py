import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import metrics


def plot_history(
    history: keras.callbacks.History, metrics: list[metrics.Metric]
) -> None:
    history_dict = history.history

    for metric in metrics:
        train_values = history_dict[metric.name]
        val_values = history_dict[f"val_{metric.name}"]
        epochs = range(1, len(train_values) + 1)
        plt.plot(epochs, train_values, label=f"train_{metric.name}")
        plt.plot(epochs, val_values, label=f"val_{metric.name}")
        plt.title(f"Training and Validation {metric.name}")
        plt.ylabel(metric.name)
        plt.xlabel("Epoch")
        plt.legend(loc="upper left")
        plt.show()


def decode_text(
    vectorize_layer: keras.layers.TextVectorization, vectorized_text: tf.Tensor | list
) -> str:
    """Decodes a vectorized text tensor or list of strings to a string."""
    if not vectorize_layer.built:
        raise ValueError("Layer has not been built yet.")

    # Convert the text tensor to a list of strings
    if isinstance(vectorized_text, tf.Tensor):
        vectorized_text = vectorized_text.numpy().tolist()

    vocab = vectorize_layer.get_vocabulary()
    decoded_text = " ".join([vocab[idx] for idx in vectorized_text if idx != 0])
    return decoded_text
