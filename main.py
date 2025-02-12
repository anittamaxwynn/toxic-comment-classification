from typing import Optional

from src import download, model, preprocess, utils

FORCE_DOWNLOAD: bool = False

VAL_SIZE: float = 0.2
BATCH_SIZE: int = 128
VOCAB_SIZE: int = 20000
MAX_LENGTH: int = 200
SHUFFLE: bool = True
FORCE_PREPROCESS: bool = False

EMBEDDING_DIM: int = 128
LSTM_UNITS: Optional[int] = 60
HIDDEN_UNITS: Optional[int] = 50
DROPOUT_RATE: Optional[float] = 0.1

EPOCHS: int = 5


def main() -> None:
    # Download raw data
    download.download_data(FORCE_DOWNLOAD)

    # Preprocess data and create TF datasets
    datasets, param_hash = preprocess.make_datasets(
        val_size=VAL_SIZE,
        vocab_size=VOCAB_SIZE,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        force_preprocess=FORCE_PREPROCESS,
    )
    train_ds, val_ds, test_ds = datasets.values()
    print("Training batches: ", len(train_ds))
    print("Validation batches: ", len(val_ds))
    print("Test batches: ", len(test_ds))

    # Extract features and labels arrays from dataset
    _, train_labels = utils.dataset_to_numpy(train_ds)
    _, val_labels = utils.dataset_to_numpy(val_ds)
    _, test_labels = utils.dataset_to_numpy(test_ds)

    # Plot label distributions
    utils.plot_label_counts(train_labels, normalize=True)
    utils.plot_label_counts(val_labels, normalize=True, is_val=True)
    utils.plot_label_counts(test_labels, normalize=True, is_test=True)

    vectorize_layer = preprocess.load_vectorize_layer(param_hash=param_hash)

    # Build training model
    training_model = model.build_model(
        VOCAB_SIZE,
        MAX_LENGTH,
        EMBEDDING_DIM,
        LSTM_UNITS,
        HIDDEN_UNITS,
        DROPOUT_RATE,
    )
    print(training_model.summary())

    # Train model
    early_stopping = model.early_stopping()
    history = training_model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[early_stopping],
    )

    # Plot training metrics
    utils.plot_metrics(history)

    # Evaluate model on test data
    test_evaluation = training_model.evaluate(test_ds, return_dict=True)
    for metric, value in test_evaluation.items():
        print(f"Test {metric}: {value:.3f}")

    # Chain the vectorize layer and the training model
    export_model = model.build_export_model(
        vectorize_layer=vectorize_layer,
        training_model=training_model,
    )
    print(export_model.summary())

    # save_model()


#     )
#
#     utils.plot_metrics(HISTORY)
#
#     test_metrics = base_model.evaluate(test_ds, return_dict=True)
#     print("Test metrics: ", {k: f"{v:.3f}" for k, v in test_metrics.items()})
#
# # Combine the vectorization layer with the base model to make an export model
# export_model = model.build_export_model(vectorize_layer, base_model, METRICS)
# print(export_model.summary())
#
# # Test the export model with string inputs (not vectorized)
# test_metrics = export_model.evaluate(test_ds, return_dict=True)
# print(
#     "Test metrics (not vectorized): ",
#     {k: f"{v:.3f}" for k, v in test_metrics.items()},
# )

# # Get predictions from the export model
# toxic_examples = tf.constant(
#     [
#         "You are a motherfucker",
#         "I hate you",
#         "You are a fucking idiot",
#         "I want to kill you",
#         "I hate black people",
#     ]
# )
#
# safe_examples = tf.constant(
#     [
#         "You are a good person",
#         "I love you",
#         "You are a good human",
#         "I want to protect you",
#     ]
# )
#
# toxic_predictions = export_model.predict(toxic_examples)
# safe_predictions = export_model.predict(safe_examples)
#
# toxic_predictions = [row.tolist() for row in toxic_predictions]
# safe_predictions = [row.tolist() for row in safe_predictions]
#
# for text, prediction in zip(toxic_examples.numpy().tolist(), toxic_predictions):
#     print(f"Text: {text}")
#     print(f"Prediction: {[f'{val:.3f}' for val in prediction]}")
#     print("")
#
# for text, prediction in zip(safe_examples.numpy().tolist(), safe_predictions):
#     print(f"Text: {text}")
#     print(f"Prediction: {[f'{val:.3f}' for val in prediction]}")
#     print("")


if __name__ == "__main__":
    main()
