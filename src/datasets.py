"""
DataLoader module for loading and preprocessing machine learning datasets.

Handles loading raw data, preprocessing, and creating TensorFlow datasets with configurable options.
"""

import shutil
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union

import keras
import pandas as pd
import tensorflow as tf

from . import config

logger = config.get_logger(__name__)


class Split(Enum):
    """Dataset split types."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class BaseDataset:
    """Base class for datasets with preprocessing support."""

    def __init__(
        self,
        split: Split,
        data_dir: str | Path = config.DATA_DIR,
        process: bool = True,
    ) -> None:
        logger.info(f"Initializing {self.__class__.__name__} for {split.name} split")
        self.split: Split = split
        self.data_dir: Path = Path(data_dir)
        self.process: bool = process

        self.raw_dir: Path = self.data_dir / "raw"
        self.processed_dir: Path = self.data_dir / "processed"
        logger.debug(
            f"Using data directories - Raw: {self.raw_dir}, Processed: {self.processed_dir}"
        )

        if not self.process:
            if self._raw_exists():
                logger.info(f"Loading raw {split.name} data")
                self.data = self._load_raw_data()
                logger.info(
                    f"Successfully loaded raw {split.name} data with {len(self)} samples"
                )
            else:
                logger.error(f"Raw {split.name} data not found in '{self.raw_dir}'")
                raise FileNotFoundError(
                    f"Raw {self.split.name} data not found in '{self.raw_dir}'. Please preprocess the data first."
                )
        else:
            logger.info(f"Starting data processing for {split.name} split")
            self._process()
            self.data: Union[pd.DataFrame, tf.data.Dataset] = (
                self._load_processed_data()
            )
            logger.info(
                f"Successfully loaded processed {split.name} data with {len(self)} samples"
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if isinstance(self.data, pd.DataFrame):
            return len(self.data)
        return self.data.cardinality().numpy()  # type: ignore

    def __getitem__(self, idx: int) -> Union[Tuple[tf.Tensor, tf.Tensor], pd.Series]:
        """Get a sample from the dataset."""
        if idx < 0:
            idx += len(self)
        if idx >= len(self) or idx < 0:
            logger.error(f"Index {idx} out of range for dataset of length {len(self)}")
            raise IndexError("Dataset index out of range")

        if isinstance(self.data, pd.DataFrame):
            return self.data.iloc[idx]
        return tuple(next(iter(self.data.skip(idx).take(1))))  # type: ignore

    def _raw_exists(self) -> bool:
        """Check if raw dataset files exist."""
        files = (
            [self.raw_dir / "train.csv"]
            if self.split == Split.TRAIN
            else [
                self.raw_dir / "test.csv",
                self.raw_dir / "test_labels.csv",
            ]
        )
        exists = all(file.exists() for file in files)
        logger.debug(
            f"Raw data files {'exist' if exists else 'do not exist'} for {self.split.name} split"
        )
        return exists

    def _load_raw_data(self) -> pd.DataFrame:
        """Load raw dataset from disk."""
        logger.info(f"Loading raw {self.split.name} dataset from disk")
        if self.split == Split.TRAIN:
            df = pd.read_csv(self.raw_dir / "train.csv")
            logger.debug(f"Loaded training data with {len(df)} samples")
            return df

        logger.debug("Loading test data and labels")
        inputs = pd.read_csv(self.raw_dir / "test.csv")
        labels = pd.read_csv(self.raw_dir / "test_labels.csv")
        df = pd.merge(inputs, labels, on="id", how="inner")
        logger.debug(f"Loaded test data with {len(df)} samples")
        return df

    @property
    def processed_file(self) -> Path:
        """Return the path to the processed dataset file."""
        return self.processed_dir / self.split.value

    def _process(self) -> None:
        """Process raw data and save it to disk."""
        raise NotImplementedError

    def _processed_exists(self) -> bool:
        """Check if the processed dataset file exists."""
        exists = self.processed_file.exists()
        logger.debug(
            f"Processed data {'exists' if exists else 'does not exist'} at {self.processed_file}"
        )
        return exists

    def _load_processed_data(self) -> tf.data.Dataset:
        """Load processed dataset from disk."""
        logger.info(
            f"Loading processed {self.split.name} dataset from {self.processed_file}"
        )
        return tf.data.Dataset.load(str(self.processed_file))


class KaggleDataset(BaseDataset):
    """Dataset class for Kaggle competition data."""

    _vectorization_layer: Optional[keras.layers.TextVectorization] = None
    _is_adapted: bool = False

    def __init__(
        self,
        split: Split,
        max_tokens: int,
        sequence_length: int,
        data_dir: str | Path = config.DATA_DIR,
        process: bool = True,
    ) -> None:
        self.max_tokens = max_tokens
        self.sequence_length = sequence_length

        if process and KaggleDataset._vectorization_layer is None:
            logger.info("Initializing TextVectorization layer")
            logger.debug(
                f"TextVectorization parameters: max_tokens={max_tokens}, sequence_length={sequence_length}"
            )
            KaggleDataset._vectorization_layer = keras.layers.TextVectorization(
                max_tokens=max_tokens,
                output_mode="int",
                output_sequence_length=sequence_length,
            )

        super().__init__(split, data_dir, process)

    def _process(self) -> None:
        """Process raw Kaggle data."""
        logger.info(f"Starting data processing pipeline for {self.split.name} split")

        logger.debug("Loading and cleaning raw data")
        df = self._load_raw_data()
        df = self._clean_dataframe(df)
        logger.debug(f"Cleaned data shape: {df.shape}")

        logger.debug("Converting DataFrame to tensors")
        inputs = tf.convert_to_tensor(df[config.INPUT].values)
        labels = tf.convert_to_tensor(df[config.LABELS].values)

        if (
            self.split == Split.TRAIN
            and KaggleDataset._vectorization_layer is not None
            and not KaggleDataset._is_adapted
        ):
            logger.info("Adapting TextVectorization layer to training data")
            KaggleDataset._vectorization_layer.adapt(inputs)
            KaggleDataset._is_adapted = True
            logger.info("TextVectorization layer adaptation completed")

        logger.info(f"Vectorizing {self.split.name} text data")
        inputs = self._vectorize_text(inputs)
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))

        logger.debug("Saving processed dataset to disk")
        self._save_and_overwrite(dataset)
        logger.info(f"Completed processing pipeline for {self.split.name} split")

    def _vectorize_text(self, text: tf.Tensor) -> tf.Tensor:
        """Vectorize text samples."""
        if KaggleDataset._vectorization_layer is None:
            logger.error("Vectorization layer not initialized")
            raise ValueError(
                "Vectorization layer not initialized. Process training data first."
            )
        if not KaggleDataset._is_adapted and self.split != Split.TRAIN:
            logger.error(
                "Attempting to vectorize non-training data before adapting layer"
            )
            raise ValueError(
                "Vectorization layer not adapted. Process training data first."
            )
        return KaggleDataset._vectorization_layer(text)

    def _save_and_overwrite(self, data: tf.data.Dataset) -> None:
        """Save processed dataset and overwrite if it exists."""
        logger.info(f"Saving processed dataset to {self.processed_file}")
        if self.processed_file.exists():
            logger.debug(f"Removing existing processed data at {self.processed_file}")
            shutil.rmtree(self.processed_file)
        data.save(str(self.processed_file))
        logger.debug("Dataset save completed")

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw Kaggle data."""
        logger.debug("Starting DataFrame cleaning")
        initial_size = len(df)

        df = self._filter_columns(df)
        df = self._drop_non_binary_labels(df)

        logger.debug(f"Cleaning complete. Rows: {initial_size} -> {len(df)}")
        return df

    def _filter_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter relevant columns from dataframe."""
        logger.debug(f"Dropping ID column from {self.split.name} dataset")
        return df.drop(columns=["id"])

    def _drop_non_binary_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with non-binary labels."""
        initial_size = len(df)
        df_filtered = df[df[config.LABELS].isin([0, 1]).all(axis=1)]

        dropped_count = initial_size - len(df_filtered)
        if dropped_count > 0:
            logger.warning(f"Removed {dropped_count} samples with non-binary labels")
        return pd.DataFrame(df_filtered.reset_index(drop=True))
