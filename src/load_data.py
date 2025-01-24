"""
DataLoader module for loading and preprocessing machine learning datasets.

Provides functionality to load raw data, preprocess datasets, and create TensorFlow datasets
with configurable preprocessing and splitting options.
"""

from enum import Enum
from pathlib import Path

import pandas as pd
import tensorflow as tf

from . import config, download

logger = config.get_logger(__name__)


class Split(Enum):
    """Enumeration of dataset split types."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DataLoader:
    """
    Manages data loading, preprocessing, and conversion to TensorFlow datasets.

    Handles raw data loading, preprocessing, and creation of train, validation, and test datasets.
    """

    RAW_DATA_DIR: Path = config.DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = config.DATA_DIR / "processed"

    def __init__(self):
        """Initialize the DataLoader with an empty dataset configuration."""
        self._dataset_config = None

    def get_tensorflow_datasets(
        self,
        val_size: float,
        batch_size: int,
        shuffle: bool = True,
        cache: bool = True,
        prefetch: bool = True,
        force_preprocess: bool = False,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Load or generate TensorFlow datasets with specified preprocessing parameters.

        Returns train, validation, and test datasets based on configuration.
        """
        self._dataset_config = {
            "val_size": val_size,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "cache": cache,
            "prefetch": prefetch,
        }

        if not force_preprocess and self._tensorflow_datasets_exist():
            return self.load_tensorflow_datasets()
        else:
            return self.make_and_save_tensorflow_datasets(
                val_size=val_size,
                batch_size=batch_size,
                shuffle=shuffle,
                cache=cache,
                prefetch=prefetch,
            )

    def load_raw_data(self, dtype: download.RawDataType) -> pd.DataFrame:
        """Load raw data from CSV files into a pandas DataFrame."""
        schema = download.RawDataScheme.get_schema(dtype)
        filepath = self.RAW_DATA_DIR / schema.filename

        if not filepath.exists():
            raise FileNotFoundError(f"Raw data file not found: {filepath}")

        logger.info(f"Loaded raw {dtype.name} data.")
        return pd.read_csv(filepath, dtype={config.ID: str})

    def make_and_save_tensorflow_datasets(
        self,
        val_size: float,
        batch_size: int,
        shuffle: bool = True,
        cache: bool = True,
        prefetch: bool = True,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Create, preprocess, and save TensorFlow datasets to disk.

        Returns train, validation, and test datasets after processing.
        """
        train_ds, val_ds, test_ds = self._make_tensorflow_datasets(
            val_size, batch_size, shuffle, cache, prefetch
        )
        self.save_tensorflow_datasets(train_ds, val_ds, test_ds)
        return train_ds, val_ds, test_ds

    def save_tensorflow_datasets(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
    ) -> None:
        """Save the processed TensorFlow datasets to disk."""
        if self._dataset_config is None:
            raise ValueError(
                "Cannot save TensorFlow datasets without dataset config. Please call `get_tensorflow_datasets` first."
            )

        dataset_dir = self._generate_dataset_dir_name()
        train_ds.save(
            str(
                self.PROCESSED_DATA_DIR.joinpath(dataset_dir).joinpath(
                    Split.TRAIN.value
                )
            )
        )
        val_ds.save(
            str(
                self.PROCESSED_DATA_DIR.joinpath(dataset_dir).joinpath(
                    Split.VALIDATION.value
                )
            )
        )
        test_ds.save(
            str(
                self.PROCESSED_DATA_DIR.joinpath(dataset_dir).joinpath(Split.TEST.value)
            )
        )

        logger.info(
            f"Saved TensorFlow datasets to {self.PROCESSED_DATA_DIR.joinpath(dataset_dir)}"
        )

    def load_tensorflow_datasets(
        self,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load the processed TensorFlow datasets from disk."""
        if self._dataset_config is None:
            raise ValueError(
                "Cannot load TensorFlow datasets without dataset config. Please call `get_tensorflow_datasets` first."
            )

        dataset_dir = self._generate_dataset_dir_name()
        train_ds = tf.data.Dataset.load(
            str(
                self.PROCESSED_DATA_DIR.joinpath(dataset_dir).joinpath(
                    Split.TRAIN.value
                )
            )
        )
        val_ds = tf.data.Dataset.load(
            str(
                self.PROCESSED_DATA_DIR.joinpath(dataset_dir).joinpath(
                    Split.VALIDATION.value
                )
            )
        )
        test_ds = tf.data.Dataset.load(
            str(
                self.PROCESSED_DATA_DIR.joinpath(dataset_dir).joinpath(Split.TEST.value)
            )
        )

        logger.info("Loaded TensorFlow datasets from disk.")
        return train_ds, val_ds, test_ds

    def load_and_preprocess_train(self) -> pd.DataFrame:
        """Load and preprocess the raw training data into a pandas DataFrame."""
        df = self.load_raw_data(download.RawDataType.TRAIN)
        return self.preprocess(df, split=Split.TRAIN)

    def load_and_preprocess_test(self) -> pd.DataFrame:
        """Load and preprocess the raw test data into a pandas DataFrame."""
        inputs = self.load_raw_data(download.RawDataType.TEST_INPUTS)
        labels = self.load_raw_data(download.RawDataType.TEST_LABELS)

        assert len(inputs) == len(labels), "Inputs and labels must have the same length"

        # Merge inputs and labels on ID column, ensure 1:1 mapping
        df = inputs.merge(labels, on=config.ID, how="inner")
        logger.info(
            f"Merged {download.RawDataType.TEST_INPUTS.name} and {download.RawDataType.TEST_LABELS.name} data into a {Split.TEST.name} set."
        )

        assert len(df) == len(inputs), "Merged data must have the same length as inputs"

        return self.preprocess(df, split=Split.TEST)

    def preprocess(self, df: pd.DataFrame, split: Split) -> pd.DataFrame:
        """Preprocess a pandas DataFrame by removing non-binary, duplicate, and missing samples."""
        df = self._drop_non_binary_samples(df, split=split)
        df = self._drop_duplicate_samples(df, split=split)
        df = self._drop_missing_samples(df, split=split)

        logger.info(f"Preprocessed {split.name} data.")
        return df

    def _generate_dataset_dir_name(self) -> str:
        """Generate the dataset directory name from the dataset configuration."""
        if self._dataset_config is None:
            raise ValueError(
                "Cannot generate dataset directory name without dataset config. Please call `get_tensorflow_datasets` first."
            )
        dir_str = f"val_size_{self._dataset_config['val_size']}_batch_size_{self._dataset_config['batch_size']}_shuffle_{self._dataset_config['shuffle']}_cache_{self._dataset_config['cache']}_prefetch_{self._dataset_config['prefetch']}"
        return dir_str.lower()

    def _tensorflow_datasets_exist(self) -> bool:
        """Check if the processed TensorFlow datasets exist on disk."""
        if self._dataset_config is None:
            raise ValueError(
                "Cannot check for TensorFlow datasets without dataset config. Please call `get_tensorflow_datasets` first."
            )

        dataset_dir = self._generate_dataset_dir_name()

        return all(
            self.PROCESSED_DATA_DIR.joinpath(dataset_dir).joinpath(split.value).exists()
            for split in Split
        )

    def _make_tensorflow_datasets(
        self,
        val_size: float,
        batch_size: int,
        shuffle: bool = True,
        cache: bool = True,
        prefetch: bool = True,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Create TensorFlow datasets from preprocessed data."""
        logger.info("Generating TensorFlow datasets...")

        train_df = self.load_and_preprocess_train()
        test_df = self.load_and_preprocess_test()

        train_ds = self._convert_to_tensorflow_dataset(
            train_df, Split.TRAIN, batch_size, shuffle, cache, prefetch
        )

        test_ds = self._convert_to_tensorflow_dataset(
            test_df, Split.TEST, batch_size, shuffle, cache, prefetch
        )

        train_ds, val_ds = self._split_tensorflow_dataset(
            train_ds, Split.TRAIN, val_size
        )

        self._dataset_config = {
            "val_size": val_size,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "cache": cache,
            "prefetch": prefetch,
        }

        return train_ds, val_ds, test_ds

    def _convert_to_tensorflow_dataset(
        self,
        df: pd.DataFrame,
        split: Split,
        batch_size: int,
        shuffle: bool,
        cache: bool,
        prefetch: bool,
    ) -> tf.data.Dataset:
        """Convert a pandas DataFrame to a TensorFlow dataset."""
        ds = tf.data.Dataset.from_tensor_slices((df[config.INPUT], df[config.LABELS]))

        if cache:
            ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(len(df), seed=0)

        ds = ds.batch(batch_size)

        if prefetch:
            ds = ds.prefetch(tf.data.AUTOTUNE)

        logger.info(
            f"Converted {split.name} data to TensorFlow dataset with {len(ds)} batches of size {batch_size}."
        )

        return ds

    def _split_tensorflow_dataset(
        self, ds: tf.data.Dataset, split: Split, split_size: float
    ) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """Split a TensorFlow dataset into training and validation sets."""
        if split_size < 0 or split_size > 1:
            raise ValueError("Split size must be a float value between 0 and 1.")

        n_samples = len(ds)
        n_test_samples = int(n_samples * split_size)
        ds = ds.shuffle(n_samples, seed=0)
        test_ds = ds.take(n_test_samples)
        train_ds = ds.skip(n_test_samples)

        logger.info(
            f"Split {split.name} dataset into training ({len(train_ds)}) and testing ({len(test_ds)}) sets."
        )

        return train_ds, test_ds

    def _drop_missing_samples(self, df: pd.DataFrame, split: Split) -> pd.DataFrame:
        """Remove samples with missing data."""
        df_filtered = df.dropna()
        if len(df_filtered) != len(df):
            logger.info(
                f"Removed {len(df) - len(df_filtered)} samples with missing data from {split.name} set."
            )
        return df_filtered.reset_index(drop=True)

    def _drop_duplicate_samples(self, df: pd.DataFrame, split: Split) -> pd.DataFrame:
        """Remove duplicate samples."""
        df_filtered = df.drop_duplicates(subset=config.ID, keep="first")

        if len(df_filtered) != len(df):
            logger.info(
                f"Removed {len(df) - len(df_filtered)} duplicate samples from {split.name} set."
            )

        return df_filtered.reset_index(drop=True)

    def _drop_non_binary_samples(self, df: pd.DataFrame, split: Split) -> pd.DataFrame:
        """Remove samples with non-binary labels."""
        df_filtered = df[df[config.LABELS].isin([0, 1]).all(axis=1)]

        if len(df_filtered) != len(df):
            logger.info(
                f"Removed {len(df) - len(df_filtered)} samples with non-binary labels from {split.name} set."
            )

        return pd.DataFrame(df_filtered.reset_index(drop=True))


if __name__ == "__main__":
    loader = DataLoader()

    train_ds, val_ds, test_ds = loader.get_tensorflow_datasets(
        val_size=0.2,
        batch_size=32,
        shuffle=True,
        cache=True,
        prefetch=True,
        force_preprocess=True,
    )

    print("TensorFlow training samples: ", len(train_ds))
    print("TensorFlow validation samples: ", len(val_ds))
    print("TensorFlow testing samples: ", len(test_ds))
