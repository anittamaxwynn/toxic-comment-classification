import os
os.environ['TF_USE_LEGACY_KERAS']='1'

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


from src import bert, download, load_data, logging

DATA_PARAMS = {
    "val_size": 0.2,
    "batch_size": 128,
    "shuffle": True,
    "force_make": False,
}

MODEL_PARAMS = {
    "max_tokens": 20000,
    "sequence_length": 200,
    "embedding_dim": 128,
    "conv_filters": 128,
    "conv_k_size": 7,
    "hidden_neurons": 128,
    "dropout_rate": 0.1,
}

TRAIN_PARMS = {
    "epochs": 3,
    "verbose": "auto",
}


def main() -> None:
    _ = download.download_kaggle()
    datasets = load_data.Datasets(**DATA_PARAMS)
    train_ds = datasets.train
    val_ds = datasets.val
    test_ds = datasets.test

    print("Train size:", len(train_ds))
    print("Val size:", len(val_ds))
    print("Test size:", len(test_ds))

    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()

    bert_classifier = bert.build_compiled_bert_classifier(
        epochs=TRAIN_PARMS["epochs"],
        steps_per_epoch=steps_per_epoch,
    )

    callbacks = logging.setup_callbacks("bert_classifier")

    print("Training BERT Model:")
    history = bert_classifier.fit(
        train_ds,
        validation_data=val_ds,
        callbacks=callbacks,
        **TRAIN_PARMS,
    )

    test_results = bert_classifier.evaluate(test_ds, return_dict=True)
    print(test_results)


if __name__ == "__main__":
    main()
