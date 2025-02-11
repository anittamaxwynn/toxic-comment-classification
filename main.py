from src import download, preprocess

FORCE_DOWNLOAD: bool = False

VAL_SIZE: float = 0.2
BATCH_SIZE: int = 64
VOCAB_SIZE: int = 10000
MAX_LENGTH: int = 100
SHUFFLE: bool = True
FORCE_PREPROCESS: bool = False

# EMBEDDING_DIM: int = 32
# DROPOUT_RATE: float = 0.1

# EPOCHS: int = 5
# METRICS: list[metrics.Metric] = [
#     metrics.Precision(name="precision"),
#     metrics.Recall(name="recall"),
#     metrics.AUC(name="auc"),
#     metrics.AUC(name="prc", curve="PR"),
# ]
#


def main() -> None:
    download.download_data(FORCE_DOWNLOAD)

    datasets, param_hash = preprocess.make_datasets(
        val_size=VAL_SIZE,
        vocab_size=VOCAB_SIZE,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        force_preprocess=FORCE_PREPROCESS,
    )

    train_ds, val_ds, test_ds = datasets.values()

    print(len(train_ds), len(val_ds), len(test_ds))

    vectorize_layer = preprocess.load_vectorize_layer(param_hash=param_hash)
    print(vectorize_layer.get_config())

    # build_model()
    # train_model()
    # plot_metrics()
    # evaluate_model()
    # save_model()


# def main() -> None:
#     download.download_data("train", force_download=False)
#     download.download_data("test", force_download=False)
#
#     train_ds, val_ds, test_ds = load_data.load_datasets(
#         val_size=VAL_SIZE, shuffle=SHUFFLE
#     )
#     train_ds.batch(BATCH_SIZE)
#
#     print("Adpating vectorize layer...")
#     vectorize_layer = model.VectorizeLayer(MAX_TOKENS, SEQUENCE_LENGTH)
#     vectorize_layer.adapt(train_ds.map(lambda x, _: x))
#     print("Vectorize layer adapted.")
#
#     print("Vectorizing datasets...")
#     train_ds = vectorize_layer.vectorize(train_ds)
#     val_ds = vectorize_layer.vectorize(val_ds)
#     test_ds = vectorize_layer.vectorize(test_ds)
#     print("Datasets vectorized.")
#
#     print("Optimizing datasets...")
#     # Optimize ALL datasets for performance
#     train_ds = load_data.optimize_ds(train_ds)
#     val_ds = load_data.optimize_ds(val_ds)
#     test_ds = load_data.optimize_ds(test_ds)
#     print("Datasets optimized.")
#
#     print("Building model...")
#     base_model = model.build_model(
#         MAX_TOKENS,
#         SEQUENCE_LENGTH,
#         EMBEDDING_DIM,
#         DROPOUT_RATE,
#         METRICS,
#     )
#
#     print(base_model.summary())
#
#     print("Training model...")
#     HISTORY = base_model.fit(
#         train_ds,
#         validation_data=val_ds,
#         epochs=EPOCHS,
#         callbacks=[model.early_stopping()],
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
