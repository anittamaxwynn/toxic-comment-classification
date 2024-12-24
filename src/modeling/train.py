import os

import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from keras.models import Model

from ..config import BASE_DIR
from ..data.convert_to_tf import load_tf_datasets
from .build import build_export_model, build_model, build_vectorize_layer

model_output_path: str = os.path.join(BASE_DIR, "models")


def main() -> None:
    train_ds, val_ds, test_ds = load_tf_datasets()

    vectorize_layer = build_vectorize_layer()
    vectorize_layer.adapt(train_ds.map(lambda x, y: x))

    # retrieve a batch (of 32 reviews and labels) from the dataset
    text_batch, label_batch = next(iter(train_ds))
    first_review, first_label = text_batch[0], label_batch[0]
    print("Review:", first_review)
    print("Label:", first_label)
    print("Vectorized review:", vectorize_layer(first_review))

    train_ds = train_ds.map(lambda x, y: (vectorize_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (vectorize_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (vectorize_layer(x), y))

    train_ds = optimize_dataset_for_performance(train_ds)
    val_ds = optimize_dataset_for_performance(val_ds)
    test_ds = optimize_dataset_for_performance(test_ds)

    model = build_model()
    print(model.summary())

    history = model.fit(train_ds, validation_data=val_ds, epochs=3)
    print(history.history)
    plot_history(history)

    loss, accuracy = model.evaluate(test_ds)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    export_model = build_export_model(vectorize_layer, model)
    print(export_model.summary())

    loss, accuracy = export_model.evaluate(test_ds)
    print("Export Loss: ", loss)
    print("Export Accuracy: ", accuracy)


def save_model(model: Sequential, model_path: str) -> None:
    model.save(model_path)
    print(f"Model saved to {model_path}")


def load_model(model_path: str) -> Model:
    return keras.models.load_model(model_path)


def plot_history(history: keras.callbacks.History) -> None:
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Training and Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


def optimize_dataset_for_performance(dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


if __name__ == "__main__":
    main()
