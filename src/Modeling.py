from typing import Optional

import keras
import tensorflow as tf
from pydantic import BaseModel, Field, PositiveInt

from . import logging, types

logger = logging.setup_logger(__name__)


def get_metrics() -> list[types.Metric]:
    return [
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
        keras.metrics.AUC(name="prc", curve="PR"),
        keras.metrics.F1Score(threshold=0.5, average="macro", name="f1_score"),
    ]


class TextClassifier(BaseModel):
    max_tokens: PositiveInt
    sequence_length: PositiveInt
    embedding_dim: PositiveInt
    conv_filters: PositiveInt
    conv_k_size: PositiveInt
    hidden_neurons: PositiveInt
    dropout_rate: float = Field(gt=0, lt=1)

    _model: Optional[types.Model] = None
    _history: Optional[types.History] = None

    @property
    def config(self) -> str:
        return self.model_dump_json(
            indent=2,
            exclude={"_model", "_history"},
        )

    @property
    def model(self) -> Optional[types.Model]:
        return self._model

    @property
    def history(self) -> Optional[types.History]:
        return self._history

    def train(
        self,
        train_dataset: types.Dataset,
        epochs: int,
        metrics: list[types.Metric],
        val_dataset: Optional[types.Dataset] = None,
        callbacks: Optional[list[keras.callbacks.Callback]] = None,
        verbose: int = 1,
    ) -> None:
        # Build model
        model = self._build_model(train_dataset, metrics)
        print(model.summary())

        # Train the model
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=str(verbose),
        )

        # Update atrributes
        self._model = model
        self._history = history

    def evaluate(self, test_dataset: types.Dataset, verbose: int = 1):
        if self._model is None:
            raise ValueError(
                "Model does not exist. Run `train` first to build and train the model."
            )
        assert isinstance(self._model, types.Model)
        assert self._model.built, "Expected the model to be built (i.e. trained)"
        return self._model.evaluate(
            test_dataset, return_dict=True, verbose=str(verbose)
        )

    def _build_model(
        self, train_dataset: types.Dataset, metrics: list[types.Metric]
    ) -> types.Model:
        text_ds = train_dataset.map(lambda x, _: x)
        vectorize_layer = self._make_vectorize_layer()
        vectorize_layer.adapt(text_ds)

        # Build the model
        inputs = keras.Input(shape=(1,), dtype=tf.string, name="text")
        x = vectorize_layer(inputs)

        x = keras.layers.Embedding(
            input_dim=(self.max_tokens + 1),
            output_dim=self.embedding_dim,
        )(x)
        x = keras.layers.Dropout(self.dropout_rate)(x)

        x = keras.layers.Conv1D(
            self.conv_filters,
            self.conv_k_size,
            padding="valid",
            activation="relu",
            strides=3,
        )(x)
        x = keras.layers.Conv1D(
            self.conv_filters,
            self.conv_k_size,
            padding="valid",
            activation="relu",
            strides=3,
        )(x)
        x = keras.layers.GlobalMaxPooling1D()(x)

        x = keras.layers.Dense(128, activation="relu")(x)
        x = keras.layers.Dropout(self.dropout_rate)(x)

        outputs = keras.layers.Dense(6, activation="sigmoid", name="predictions")(x)

        model = keras.Model(inputs, outputs)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=metrics)

        return model

    def _make_vectorize_layer(self) -> types.Vectorizer:
        return keras.layers.TextVectorization(
            max_tokens=self.max_tokens,
            output_mode="int",
            output_sequence_length=self.sequence_length,
        )


if __name__ == "__main__":
    pass
