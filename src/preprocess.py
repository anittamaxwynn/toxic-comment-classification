from dataclasses import dataclass
from enum import Enum

import keras
import pandas as pd
import tensorflow as tf

from . import config

logger = config.get_logger(__name__)


class DatasetSplit(Enum):
    """Enumeration of dataset split types."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass
class DatasetConfig:
    val_size: float
    batch_size: int
    max_tokens: int
    output_sequence_length: int
    vectorize: bool = True
    shuffle: bool = True
    optimize: bool = True

    @property
    def dir_name(self) -> str:
        return self.generate_config_str()

    def generate_config_str(self) -> str:
        string = ""
        for key, value in self.__dict__.items():
            string += f"{key}={value}__"
        return string.lower().strip("_")


class Preprocessor:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.vectorize_layer: keras.layers.TextVectorization | None = None

        self._is_adapted = False

    def vectorize_dataset(
        self,
        dataset: tf.data.Dataset,
        split: DatasetSplit,
        adapt: bool = False,
    ) -> tf.data.Dataset:
        """Vectorizes the text inputs of a given dataset."""
        if self.vectorize_layer is None:
            self.vectorize_layer = self._make_vectorize_layer()

        if adapt:
            self._adapt_vectorize_layer(dataset, split)

        logger.info(f"Vectorizing {split.name} text data...")

        return dataset.map(self._vectorize_text)

    def _make_vectorize_layer(self) -> keras.layers.TextVectorization:
        """Make a vectorize layer."""
        logger.info(
            f"Making vectorize layer with max_tokens={self.config.max_tokens} and output_sequence_length={self.config.output_sequence_length}..."
        )
        return keras.layers.TextVectorization(
            max_tokens=self.config.max_tokens,
            output_mode="int",
            output_sequence_length=self.config.output_sequence_length,
        )

    def _adapt_vectorize_layer(
        self, dataset: tf.data.Dataset, split: DatasetSplit
    ) -> None:
        """Trains the vectorizer on a given dataset."""
        if self.vectorize_layer is None:
            raise ValueError(
                "Vectorizer is not initialized. First call `_make_vectorize_layer`."
            )

        logger.info(f"Adapting vectorize layer to {split.name} text data...")

        # Make a text-only dataset (without labels)
        text_dataset = dataset.map(lambda x, _: x)
        self.vectorize_layer.adapt(text_dataset)
        self._is_adapted = True

    def _vectorize_text(
        self, text: tf.Tensor, labels: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Vectorizes text data."""
        text = tf.expand_dims(text, -1)
        return self.vectorize_layer(text), labels

    def convert_to_dataset(
        self,
        df: pd.DataFrame,
        split: DatasetSplit,
    ) -> tf.data.Dataset:
        """Convert a pandas DataFrame to a TensorFlow dataset."""
        ds = tf.data.Dataset.from_tensor_slices((df[config.INPUT], df[config.LABELS]))

        if self.config.shuffle:
            ds = ds.shuffle(len(df), seed=0)

        ds = ds.batch(self.config.batch_size, drop_remainder=True)

        logger.info(
            f"Converting {split.name} data to TensorFlow dataset with {len(ds)} batches of size {self.config.batch_size}..."
        )

        return ds

    def split_dataset(
        self,
        dataset: tf.data.Dataset,
        split: DatasetSplit,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """Split a TensorFlow dataset into training and test (or validation) sets."""
        if self.config.val_size < 0 or self.config.val_size > 1:
            raise ValueError("Test size must be a float value between 0 and 1.")

        n_samples = len(dataset)
        n_val_samples = int(n_samples * self.config.val_size)

        if self.config.shuffle:
            dataset = dataset.shuffle(n_samples, seed=0)

        val_ds = dataset.take(n_val_samples)
        train_ds = dataset.skip(n_val_samples)

        logger.info(
            f"Splitting {split.name} dataset into training ({len(train_ds)}) and test ({len(val_ds)}) sets..."
        )
        return train_ds, val_ds

    def optimize_dataset(
        self, dataset: tf.data.Dataset, split: DatasetSplit
    ) -> tf.data.Dataset:
        """Optimize a TensorFlow dataset for performance."""
        if not self.config.optimize:
            raise ValueError(
                "Dataset optimization is disabled. Set `optimize` to True."
            )

        logger.info(f"Optimizing {split.name} dataset...")
        AUTOTUNE = tf.data.AUTOTUNE

        return dataset.cache().prefetch(buffer_size=AUTOTUNE)

    def clean_dataframe(self, df: pd.DataFrame, split: DatasetSplit) -> pd.DataFrame:
        """Performs data cleaning operations on a given dataframe."""
        df = self._drop_missing_samples(df, split)
        df = self._drop_duplicate_samples(df, split)
        df = self._drop_non_binary_samples(df, split)

        return df

    def _drop_missing_samples(
        self, df: pd.DataFrame, split: DatasetSplit
    ) -> pd.DataFrame:
        """Remove samples with missing data."""
        df_filtered = df.dropna()
        if len(df_filtered) != len(df):
            logger.warning(
                f"Removed {len(df) - len(df_filtered)} samples with missing data from {split.name} set."
            )
        return df_filtered.reset_index(drop=True)

    def _drop_duplicate_samples(
        self, df: pd.DataFrame, split: DatasetSplit
    ) -> pd.DataFrame:
        """Remove duplicate samples."""
        df_filtered = df.drop_duplicates(subset=config.ID, keep="first")

        if len(df_filtered) != len(df):
            logger.warning(
                f"Removed {len(df) - len(df_filtered)} duplicate samples from {split.name} set."
            )

        return df_filtered.reset_index(drop=True)

    def _drop_non_binary_samples(
        self, df: pd.DataFrame, split: DatasetSplit
    ) -> pd.DataFrame:
        """Remove samples with non-binary labels."""
        df_filtered = df[df[config.LABELS].isin([0, 1]).all(axis=1)]

        if len(df_filtered) != len(df):
            logger.warning(
                f"Removed {len(df) - len(df_filtered)} samples with non-binary labels from {split.name} set."
            )

        return pd.DataFrame(df_filtered.reset_index(drop=True))
