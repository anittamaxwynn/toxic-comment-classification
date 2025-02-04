import numpy as np
import tensorflow as tf
from keras import metrics

from src import download, load_data, model, util

# Preproccessing parameters:
VAL_SIZE: float = 0.2
BATCH_SIZE: int = 32
MAX_TOKENS: int = 1000
SEQUENCE_LENGTH: int = 50
SHUFFLE: bool = True

# Model parameters:
EMBEDDING_DIM: int = 16
DROPOUT_RATE: float = 0.1

# Training parameters:
EPOCHS: int = 2
LOSS: str = "binary_crossentropy"
METRICS: list[metrics.Metric] = [
    metrics.Precision(name="precision"),
    metrics.Recall(name="recall"),
]


def main() -> None:
    # Download the data (if it doesn't exist)
    download.download_data("train", force_download=False)
    download.download_data("test", force_download=False)

    # Load preprocessed TensorFlow datasets
    train_ds = load_data.load_dataset("train", BATCH_SIZE, SHUFFLE)
    test_ds = load_data.load_dataset("test", BATCH_SIZE, SHUFFLE)

    # Split the training dataset into train and validation sets
    train_ds, val_ds = load_data.split_dataset(train_ds, VAL_SIZE, SHUFFLE)

    # Create a vectorize layer and adapt it to the training text
    vectorize_layer = model.build_vectorize_layer(MAX_TOKENS, SEQUENCE_LENGTH)
    vectorize_layer.adapt(train_ds.map(lambda x, _: x))

    # Vectorize the text in the training, validation, and test datasets
    vectorized_train_ds = train_ds.map(lambda x, y: (vectorize_layer(x), y))
    vectorized_val_ds = val_ds.map(lambda x, y: (vectorize_layer(x), y))
    vectorized_test_ds = test_ds.map(lambda x, y: (vectorize_layer(x), y))

    # Optimize the datasets for performance
    vectorized_train_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    vectorized_val_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    vectorized_test_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    print("Train batches: ", len(vectorized_train_ds))
    print("Validation batches: ", len(vectorized_val_ds))
    print("Test batches: ", len(vectorized_test_ds))

    # Get the first sample from the training dataset
    # text_batch, label_batch = train_ds.take(1).as_numpy_iterator().get_next()
    # text_sample, label_sample = text_batch[0], label_batch[0]
    #
    # vectorized_text_batch, _ = vectorized_train_ds.take(1).as_numpy_iterator().next()
    # vectorized_text_sample = vectorized_text_batch[0]
    #
    # decded_text_sample = util.decode_text(vectorize_layer, vectorized_text_sample)
    #
    # print("Text: ", text_sample)
    # print("Label: ", label_sample)
    # print("Vectorized Text: ", vectorized_text_sample)
    # print("Decoded Text: ", decded_text_sample)

    # Build and compile the base model (without the vectorization layer)
    base_model = model.build_model(
        MAX_TOKENS,
        SEQUENCE_LENGTH,
        EMBEDDING_DIM,
        DROPOUT_RATE,
        LOSS,
        METRICS,
    )

    print(base_model.summary())

    # Train the base model on vectorized training data
    HISTORY = base_model.fit(
        vectorized_train_ds,
        validation_data=vectorized_val_ds,
        epochs=EPOCHS,
    )

    # Evaluate the base model on the vectorized test dataset
    test_metrics = base_model.evaluate(vectorized_test_ds, return_dict=True)
    print("Test metrics (vectorized): ", test_metrics)

    # Plot the training and validation history (binary accuracy and loss)
    util.plot_history(HISTORY, METRICS)

    # Combine the vectorization layer with the base model to make an export model
    export_model = model.build_export_model(vectorize_layer, base_model, LOSS, METRICS)
    print(export_model.summary())

    # Test the export model with `raw_test_ds`, which yields raw strings
    test_metrics = export_model.evaluate(test_ds, return_dict=True)
    print("Test metrics (not vectorirzed): ", test_metrics)

    # Get predictions from the export model
    toxic_examples = tf.constant(
        [
            "You are a motherfucker",
            "I hate you",
            "You are a fucking idiot",
            "I want to kill you",
            "I hate black people",
        ]
    )

    safe_examples = tf.constant(
        [
            "You are a good person",
            "I love you",
            "You are a good human",
            "I want to protect you",
        ]
    )

    toxic_predictions = export_model.predict(toxic_examples)
    safe_predictions = export_model.predict(safe_examples)

    toxic_predictions = [np.round(row, 2).tolist() for row in toxic_predictions]
    safe_predictions = [np.round(row, 2).tolist() for row in safe_predictions]

    for text, prediction in zip(toxic_examples, toxic_predictions):
        print(f"Text: {text}")
        print(f"Prediction: {prediction}")
        print("")

    for text, prediction in zip(safe_examples, safe_predictions):
        print(f"Text: {text}")
        print(f"Prediction: {prediction}")
        print("")


if __name__ == "__main__":
    main()
