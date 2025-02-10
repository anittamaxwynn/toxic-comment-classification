from typing import Literal, Tuple

import keras
import tensorflow as tf
from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class VectorizeLayerConfig(BaseModel):
    max_tokens: PositiveInt = Field(
        description="Maximum number of tokens in the vocabulary"
    )
    sequence_length: PositiveInt = Field(description="Length of output sequences")


class VectorizeLayer:
    def __init__(self, max_tokens: int, sequence_length: int) -> None:
        # Validate configuration
        self.config = VectorizeLayerConfig(
            max_tokens=max_tokens, sequence_length=sequence_length
        )

        # Initialize the layer
        self.layer = keras.layers.TextVectorization(
            max_tokens=self.config.max_tokens,
            output_mode="int",
            output_sequence_length=self.config.sequence_length,
            sparse=False,
        )
        self.adapted = False

    def adapt(self, text: tf.Tensor | tf.data.Dataset) -> None:
        self.layer.adapt(text)
        self.adapted = True

    def vectorize(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.map(self._vectorize_text)

    def _vectorize_text(
        self, text: tf.Tensor, labels: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        if not self.adapted:
            raise ValueError(
                "The vectorize layer must be adapted before vectorizing text."
            )
        text = tf.expand_dims(text, axis=-1)
        return self.layer(text), labels


class ModelConfig(BaseModel):
    """Configuration for a machine learning model.

    Allows flexible specification of model parameters with type validation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_tokens: PositiveInt = Field(
        description="Maximum number of tokens in the vocabulary"
    )
    sequence_length: PositiveInt = Field(description="Length of output sequences")
    embedding_dim: PositiveInt = Field(description="Dimension of the embedding")
    dropout_rate: float = Field(
        ge=0.0, le=1.0, description="Dropout rate for the model (between 0 and 1)"
    )
    optimizer: Literal["adam", "sgd"] = Field(description="Optimizer for the model")
    loss: keras.losses.Loss = Field(description="Loss function for the model")
    metrics: list[keras.metrics.Metric] = Field(
        description="Training and validation metrics for the model"
    )


def build_model(
    max_tokens: int,
    sequence_length: int,
    embedding_dim: int,
    dropout_rate: float,
    metrics: list[keras.metrics.Metric],
    optimizer: Literal["adam", "sgd"] = "adam",
    loss: keras.losses.Loss = keras.losses.BinaryCrossentropy(),
) -> keras.models.Sequential:
    config = ModelConfig(
        max_tokens=max_tokens,
        sequence_length=sequence_length,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    model = keras.models.Sequential(
        [
            keras.layers.Input(
                shape=(config.sequence_length,), dtype=tf.int32, name="vectorized_text"
            ),
            keras.layers.Embedding(
                input_dim=config.max_tokens,
                output_dim=config.embedding_dim,
                name="embedding",
            ),
            keras.layers.LSTM(
                config.embedding_dim, return_sequences=False, name="lstm"
            ),
            keras.layers.Dropout(config.dropout_rate, name="dropout_1"),
            keras.layers.Dense(32, activation="relu", name="hidden"),
            keras.layers.Dropout(config.dropout_rate, name="dropout_2"),
            keras.layers.Dense(6, activation="sigmoid", name="toxic_labels"),
        ]
    )

    model.compile(optimizer=config.optimizer, loss=config.loss, metrics=config.metrics)

    return model


def build_export_model(
    vectorize_layer: VectorizeLayer,
    model: keras.models.Sequential,
    metrics: list[keras.metrics.Metric],
    optimizer: Literal["adam", "sgd"] = "adam",
    loss: keras.losses.Loss = keras.losses.BinaryCrossentropy(),
) -> keras.models.Sequential:
    # Check if the vectorize layer and models have been built
    if not vectorize_layer.layer.built:
        raise ValueError(
            "The vectorize layer must be built before building the export model."
        )
    if not model.built:
        raise ValueError("The model must be built before building the export model.")

    # Create the export model
    export_model = keras.models.Sequential(
        [
            vectorize_layer.layer,
            model,
        ]
    )

    export_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return export_model


def early_stopping() -> keras.callbacks.EarlyStopping:
    return keras.callbacks.EarlyStopping(
        monitor="val_prc",
        verbose=1,
        patience=10,
        mode="max",
        restore_best_weights=True,
    )
