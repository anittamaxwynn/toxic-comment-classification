import keras
import tensorflow as tf
from keras import losses

from src import dataset, load_data, model, util

# Preproccessing parameters:
VAL_SIZE: float = 0.2
BATCH_SIZE: int = 32
MAX_TOKENS: int = 10000
SEQUENCE_LENGTH: int = 100
SHUFFLE: bool = True

# Model parameters:
EMBEDDING_DIM: int = 16
DROPOUT_RATE: float = 0.1

# TRAINING PARAMETERS
EPOCHS: int = 5

FORCE_DOWNLOAD: bool = False


def main() -> None:
    TRAIN_SPLIT, TEST_SPLIT = load_data.Split.TRAIN, load_data.Split.TEST
    # Load clean data
    train_features, train_labels = dataset.load_clean_data(TRAIN_SPLIT)
    test_features, test_labels = dataset.load_clean_data(TEST_SPLIT)

    # Break the dataset into train and validation sets
    (train_features, train_labels), (val_features, val_labels) = dataset.split_data(
        train_features, train_labels, VAL_SIZE
    )

    print("Raw train samples: ", len(train_features))
    print("Raw validation samples: ", len(val_features))
    print("Raw test samples: ", len(test_features))

    # Create a vectorize layer and adapt it to the training dataset
    vectorize_layer = model.build_vectorize_layer(MAX_TOKENS, SEQUENCE_LENGTH)
    vectorize_layer.adapt(train_features)

    # Vectorize the text in the training, validation, and test datasets
    train_features = vectorize_layer(train_features)
    val_features = vectorize_layer(val_features)
    test_features = vectorize_layer(test_features)

    # Create a dataset from the vectorized features and labels
    train_ds = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_features, test_labels))

    # Shuffle the training dataset
    if SHUFFLE:
        train_ds = train_ds.shuffle(len(train_features))

    # Convert the datasets to batches
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=True)
    test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=True)

    # Optimize the datasets for performance
    train_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    print("Train batches: ", len(train_ds))
    print("Validation batches: ", len(val_ds))
    print("Test batches: ", len(test_ds))

    # Build and compile the base model (without the vectorization layer)
    base_model = model.build_and_compile_model(
        MAX_TOKENS,
        SEQUENCE_LENGTH,
        EMBEDDING_DIM,
        DROPOUT_RATE,
    )

    print(base_model.summary())

    # Train the base model
    history = base_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    # Evaluate the base model on the test dataset
    loss, accuracy = base_model.evaluate(test_ds)
    print("Test Loss: ", loss)
    print("Test Accuracy: ", accuracy)

    # Plot the training and validation history (binary accuracy and loss)
    util.plot_history(history)

    # Combine the vectorization layer with the base model to make an export model
    export_model = keras.models.Sequential(
        [
            vectorize_layer,
            base_model,
            keras.layers.Activation("sigmoid"),
        ]
    )

    # Compile the export model
    export_model.compile(
        loss=losses.BinaryCrossentropy(from_logits=False),
        optimizer="adam",
        metrics=["accuracy"],
    )

    # Test the export model with `raw_test_ds`, which yields raw strings
    # metrics = export_model.evaluate(raw_test_ds, return_dict=True)
    # print(metrics)

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
    print("Toxic predictions: ", toxic_predictions)
    print("Safe predictions: ", safe_predictions)


if __name__ == "__main__":
    main()
