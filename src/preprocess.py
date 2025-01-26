from dataclasses import dataclass, fields
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
    val_size: float = 0.2
    batch_size: int = 32
    max_tokens: int = 10000
    output_sequence_length: int = 100
    vectorize: bool = True
    shuffle: bool = True
    optimize: bool = True

    @property
    def dir_name(self) -> str:
        return self.generate_dir_string()

    def generate_dir_string(self) -> str:
        parts = []
        for field in fields(self):
            value = getattr(self, field.name)

            # Skip if vectorize is False and these specific fields
            if not self.vectorize and field.name in [
                "max_tokens",
                "output_sequence_length",
            ]:
                continue

            # Only add to string if boolean fields are False
            if isinstance(value, bool):
                if not value:
                    parts.append(f"{field.name}={value}")
            else:
                parts.append(f"{field.name}={value}")

        return "__".join(parts).lower()


class Preprocessor:
    def __init__(self, config: DatasetConfig):
        self.config = config

        if self.config.vectorize:
            self.vectorize_layer = keras.layers.TextVectorization(
                standardize="lower_and_strip_punctuation",
                max_tokens=self.config.max_tokens,
                output_mode="int",
                output_sequence_length=self.config.output_sequence_length,
            )
        else:
            self.vectorize_layer = None

        self._is_adapted = False

    def vectorize_dataset(
        self,
        dataset: tf.data.Dataset,
        split: DatasetSplit,
        adapt: bool = False,
    ) -> tf.data.Dataset:
        """Vectorizes the text inputs of a given dataset."""
        if self.vectorize_layer is None:
            raise ValueError(
                "TextVectorization layer not enabled. Set `vectorize` to True in the data config."
            )
        if adapt:
            self._adapt_vectorize_layer(dataset, split)

        logger.info(f"Vectorizing {split.name} text data...")

        return dataset.map(self._vectorize_text)

    def _adapt_vectorize_layer(
        self, dataset: tf.data.Dataset, split: DatasetSplit
    ) -> None:
        """Trains the vectorizer on a given dataset."""
        if self.vectorize_layer is None:
            raise ValueError(
                "TextVectorization layer is not enabled. Set `vectorize` to True in the data config."
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


if __name__ == "__main__":
    data_config = DatasetConfig(
        val_size=0.2,
        batch_size=32,
        max_tokens=10000,
        output_sequence_length=100,
        vectorize=False,
        shuffle=False,
        optimize=True,
    )

    data_dir_name = data_config.dir_name
    print(data_dir_name)
