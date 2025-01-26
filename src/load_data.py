"""
DataLoader module for loading and preprocessing machine learning datasets.

Provides functionality to load raw data, preprocess datasets, and create TensorFlow datasets
with configurable preprocessing and splitting options.
"""

from pathlib import Path

import pandas as pd
import tensorflow as tf

from . import config, download, preprocess

logger = config.get_logger(__name__)


class DataLoader:
    """
    Manages data loading, preprocessing, and conversion to TensorFlow datasets.

    Handles raw data loading, preprocessing, and creation of train, validation, and test datasets.
    """

    RAW_DATA_DIR: Path = config.DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = config.DATA_DIR / "processed"

    def __init__(self, dataset_config: preprocess.DatasetConfig):
        """Initialize the DataLoader with an empty dataset configuration."""
        self.dataset_config = dataset_config
        self.preprocessor = preprocess.Preprocessor(dataset_config)

    def get_datasets(
        self,
        force_preprocess: bool = True,
        save: bool = True,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Load or generate TensorFlow datasets with specified preprocessing parameters.

        Returns train, validation, and test datasets based on configuration.
        """

        if not force_preprocess and self._datasets_exist():
            return self.load_datasets()

        train_ds, val_ds, test_ds = self.make_datasets()

        if save:
            self.save_datasets(train_ds, val_ds, test_ds)

        return train_ds, val_ds, test_ds

    def make_datasets(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Create TensorFlow datasets from preprocessed data."""
        logger.info("Generating TensorFlow datasets...")

        train_split = preprocess.DatasetSplit.TRAIN
        val_split = preprocess.DatasetSplit.VALIDATION
        test_split = preprocess.DatasetSplit.TEST

        train_df = self._load_and_clean_train_df()
        test_df = self._load_and_clean_test_df()

        train_ds = self.preprocessor.convert_to_dataset(
            df=train_df,
            split=preprocess.DatasetSplit.TRAIN,
        )

        test_ds = self.preprocessor.convert_to_dataset(
            df=test_df,
            split=preprocess.DatasetSplit.TEST,
        )

        train_ds, val_ds = self.preprocessor.split_dataset(
            dataset=train_ds,
            split=preprocess.DatasetSplit.TRAIN,
        )

        if self.dataset_config.vectorize:
            train_ds = self.preprocessor.vectorize_dataset(
                train_ds, split=train_split, adapt=True
            )
            val_ds = self.preprocessor.vectorize_dataset(
                val_ds, split=val_split, adapt=False
            )
            test_ds = self.preprocessor.vectorize_dataset(
                test_ds, split=test_split, adapt=False
            )

        if self.dataset_config.optimize:
            train_ds = self.preprocessor.optimize_dataset(
                train_ds, split=preprocess.DatasetSplit.TRAIN
            )
            val_ds = self.preprocessor.optimize_dataset(
                val_ds, split=preprocess.DatasetSplit.VALIDATION
            )
            test_ds = self.preprocessor.optimize_dataset(
                test_ds, split=preprocess.DatasetSplit.TEST
            )

        return train_ds, val_ds, test_ds

    def save_datasets(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
    ) -> None:
        """Save the processed TensorFlow datasets to disk."""
        logger.info(f"Saving TensorFlow datasets to {self.PROCESSED_DATA_DIR}...")

        dataset_dir = self.dataset_config.dir_name
        train_name = preprocess.DatasetSplit.TRAIN.value
        val_name = preprocess.DatasetSplit.VALIDATION.value
        test_name = preprocess.DatasetSplit.TEST.value

        train_ds.save(
            str(self.PROCESSED_DATA_DIR.joinpath(dataset_dir).joinpath(train_name))
        )
        val_ds.save(
            str(self.PROCESSED_DATA_DIR.joinpath(dataset_dir).joinpath(val_name))
        )
        test_ds.save(
            str(self.PROCESSED_DATA_DIR.joinpath(dataset_dir).joinpath(test_name))
        )

    def load_datasets(
        self,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load the processed TensorFlow datasets from disk."""

        if not self._datasets_exist():
            raise FileNotFoundError(f"Datasets not found in {self.PROCESSED_DATA_DIR}")

        dataset_dir = self.dataset_config.dir_name

        train_name = preprocess.DatasetSplit.TRAIN.value
        val_name = preprocess.DatasetSplit.VALIDATION.value
        test_name = preprocess.DatasetSplit.TEST.value

        train_ds = tf.data.Dataset.load(
            str(self.PROCESSED_DATA_DIR.joinpath(dataset_dir).joinpath(train_name))
        )
        val_ds = tf.data.Dataset.load(
            str(self.PROCESSED_DATA_DIR.joinpath(dataset_dir).joinpath(val_name))
        )
        test_ds = tf.data.Dataset.load(
            str(self.PROCESSED_DATA_DIR.joinpath(dataset_dir).joinpath(test_name))
        )

        logger.info("Loaded TensorFlow datasets from disk.")
        return train_ds, val_ds, test_ds

    def _load_raw_dataframe(self, dtype: download.RawDataType) -> pd.DataFrame:
        """Load raw data from CSV files into a pandas DataFrame."""
        schema = download.RawDataScheme.get_schema(dtype)
        filepath = self.RAW_DATA_DIR / schema.filename

        if not filepath.exists():
            raise FileNotFoundError(f"Raw data file not found: {filepath}")

        logger.info(f"Loading raw {dtype.name} data from {filepath}.")

        return pd.read_csv(filepath, dtype={config.ID: str})

    def _load_and_clean_train_df(self) -> pd.DataFrame:
        """Load and clean the raw training data into a pandas DataFrame."""
        split = preprocess.DatasetSplit.TRAIN
        df = self._load_raw_dataframe(download.RawDataType.TRAIN)
        return self.preprocessor.clean_dataframe(df, split)

    def _load_and_clean_test_df(self) -> pd.DataFrame:
        """Load and clean the raw test data into a pandas DataFrame."""
        split = preprocess.DatasetSplit.TEST

        inputs = self._load_raw_dataframe(download.RawDataType.TEST_INPUTS)
        labels = self._load_raw_dataframe(download.RawDataType.TEST_LABELS)

        assert len(inputs) == len(labels), "Inputs and labels must have the same length"

        # Merge inputs and labels on ID column, ensure 1:1 mapping
        df = inputs.merge(labels, on=config.ID, how="inner")
        logger.info(
            f"Merging {download.RawDataType.TEST_INPUTS.name} and {download.RawDataType.TEST_LABELS.name} data into a {split.name} set."
        )

        assert len(df) == len(inputs), "Merged data must have the same length as inputs"

        return self.preprocessor.clean_dataframe(df, split)

    def _datasets_exist(self) -> bool:
        """Check if the processed TensorFlow datasets exist on disk."""
        dataset_dir = self.dataset_config.dir_name

        return all(
            self.PROCESSED_DATA_DIR.joinpath(dataset_dir).joinpath(split.value).exists()
            for split in preprocess.DatasetSplit
        )


if __name__ == "__main__":
    pass
