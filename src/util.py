import keras
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_history(history: keras.callbacks.History) -> None:
    history_dict = history.history
    acc = history_dict["binary_accuracy"]
    val_acc = history_dict["val_binary_accuracy"]
    loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, "b", label="Training Accuracy")
    plt.plot(epochs, val_acc, "r", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.show()

    plt.plot(epochs, loss, "b", label="Training Loss")
    plt.plot(epochs, val_loss, "r", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.show()


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
