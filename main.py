import tensorflow as tf

from src import dataset, load_data, model

# Preproccessing parameters:
VAL_SIZE: float = 0.2
BATCH_SIZE: int = 32
MAX_TOKENS: int = 1000
SEQUENCE_LENGTH: int = 100
SHUFFLE: bool = True

# Model parameters:
EMBEDDING_DIM: int = 16
DROPOUT_RATE: float = 0.001

# TRAINING PARAMETERS
EPOCHS: int = 5

FORCE_DOWNLOAD: bool = False


def main() -> None:
    TRAIN_SPLIT, TEST_SPLIT = load_data.Split.TRAIN, load_data.Split.TEST
    # Load the processed TensorFlow datasets
    train_ds = dataset.make_tf_dataset(TRAIN_SPLIT, BATCH_SIZE, SHUFFLE)
    test_ds = dataset.make_tf_dataset(TEST_SPLIT, BATCH_SIZE, SHUFFLE)

    # Break the dataset into train and validation sets
    train_ds, val_ds = dataset.split_tf_dataset(train_ds, VAL_SIZE, SHUFFLE)

    # Create a vectorize layer and adapt it to the training dataset
    vectorize_layer = model.build_vectorize_layer(MAX_TOKENS, SEQUENCE_LENGTH)
    vectorize_layer.adapt(train_ds.map(lambda x, _: x))

    # Vectorize the text in the training, validation, and test datasets
    train_ds = train_ds.map(lambda x, y: (vectorize_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (vectorize_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (vectorize_layer(x), y))

    # Optimize the datasets for performance
    train_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    print("Train batches: ", len(train_ds))
    print("Validation batches: ", len(val_ds))
    print("Test batches: ", len(test_ds))

    base_model = model.build_and_compile_model(
        MAX_TOKENS,
        SEQUENCE_LENGTH,
        EMBEDDING_DIM,
        DROPOUT_RATE,
    )

    print(base_model.summary())

    history = base_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    loss, accuracy = base_model.evaluate(test_ds)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)


if __name__ == "__main__":
    main()
