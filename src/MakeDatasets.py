"""Process text data into TensorFlow datasets with parameter versioning."""

import hashlib
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import tensorflow as tf
from pydantic import BaseModel, Field, PositiveInt

from . import Config, DownloadData, Logging, Preprocessing
from .Types import Dataset

logger = Logging.setup_logger(__name__)


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Datasets(BaseModel):
    """Manages the creation, storage, and retrieval of TensorFlow datasets with version control."""

    val_size: float = Field(gt=0, lt=1)
    batch_size: PositiveInt
    shuffle: bool = True
    force_make: bool = False
    features: list[str] = Field(default=Config.FEATURES)
    labels: list[str] = Field(default=Config.LABELS)

    _datasets: Optional[Dict[Split, Dataset]] = None
    _hash: Optional[str] = None

    _save_path: Path = Config.TENSORFLOW_DIR

    @property
    def datasets(self) -> Dict[Split, Dataset]:
        if self._datasets is None:
            if self.force_make or not self._datasets_exist():
                logger.debug("Creating new datasets with hash: %s", self.hash)
                self._datasets = self._make_datasets()
                self._save_datasets()
                self._save_config()
            else:
                logger.debug("Found existing datasets with hash f: %s", self.hash)
                self._verify_config()
                logger.debug("Existing dataset config matches current config")
                self._datasets = self._load_datasets()
        return self._datasets

    @property
    def train(self) -> Dataset:
        """Returns the training dataset."""
        return self.datasets[Split.TRAIN]

    @property
    def val(self) -> Dataset:
        """Returns the validation dataset."""
        return self.datasets[Split.VAL]

    @property
    def test(self) -> Dataset:
        """Returns the test dataset."""
        return self.datasets[Split.TEST]

    @property
    def hash(self) -> str:
        """Generates a unique hash based on the dataset configuration."""
        if self._hash is None:
            self._hash = self._generate_hash()
            logger.debug("Generated new hash: %s", self._hash)
        return self._hash

    @property
    def config(self) -> str:
        """Returns the dataset configuration as a JSON string."""
        return self.model_dump_json(
            indent=2,
            exclude={"force_make", "_datasets", "_hash"},
        )

    def _make_datasets(self) -> Dict[Split, Dataset]:
        """Creates new TensorFlow datasets from raw data."""
        logger.debug("Starting dataset creation process")
        raw_data = self._load_raw_data()
        split_data = self._split_data(raw_data)
        clean_data = self._clean_data(split_data)
        datasets = self._convert_to_tensorflow(clean_data)
        logger.debug("Completed dataset creation")
        return datasets

    def _save_datasets(self) -> None:
        """Saves TensorFlow datasets to disk with version control."""
        dataset_dir = self._save_path / self.hash
        dataset_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Saving datasets to directory: %s", dataset_dir)

        for split, ds in self.datasets.items():
            split_dir = dataset_dir / split.value
            split_dir.mkdir(parents=True, exist_ok=True)
            ds.save(str(split_dir), compression="GZIP")
            logger.debug("Saved %s split to: %s", split.value, split_dir)

        assert self._datasets_exist(), f"Datasets not found in {dataset_dir}"
        logger.debug("Successfully verified all datasets were saved")
        return None

    def _save_config(self) -> None:
        dataset_dir = self._save_path / self.hash
        config = self.config
        config_path = dataset_dir / "config.json"
        with open(config_path, "w") as f:
            f.write(config)

        assert config_path.exists(), f"Config not saved to {config_path}"
        logger.debug("Successfully saved config json to %s", config_path)
        return None

    def _load_config(self) -> str:
        """Load the config json from the dataset directory."""
        dataset_dir = self._save_path / self.hash
        config_path = dataset_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")
        else:
            with open(config_path, "r") as f:
                config = f.read()
            return config

    def _verify_config(self) -> None:
        """Verify that a provided config json matches the current config."""
        loaded_config = self._load_config()
        if loaded_config != self.config:
            raise ValueError(
                f"Config mismatch: provided config {loaded_config} does not match current config {self.config}"
            )
        return None

    def _datasets_exist(self) -> bool:
        """Checks if versioned datasets exist on disk."""
        dataset_dir = self._save_path / self.hash
        if not dataset_dir.exists():
            logger.debug("Dataset directory does not exist: %s", dataset_dir)
            return False

        split_exists = all((dataset_dir / split.value).exists() for split in Split)

        if not split_exists:
            logger.debug("One or more split directories missing in: %s", dataset_dir)
        return split_exists

    def _load_datasets(self) -> dict[Split, Dataset]:
        """Loads versioned TensorFlow datasets from disk."""
        hash_dir = self._save_path / self.hash
        logger.debug("Loading datasets from directory: %s", hash_dir)

        datasets = {
            split: tf.data.Dataset.load(str(hash_dir / split.value), compression="GZIP")
            for split in Split
        }

        logger.debug("Successfully loaded all datasets")
        return datasets

    def _generate_hash(self) -> str:
        """Generates a unique hash for the current dataset configuration."""
        return hashlib.sha256(self.config.encode()).hexdigest()[:10]

    @staticmethod
    def _load_raw_data() -> Dict[DownloadData.Files, pd.DataFrame]:
        """Loads raw data from CSV files."""
        logger.debug("Loading raw data from CSV files")
        kaggle_dataset_path = Path(DownloadData.download_kaggle_dataset())

        raw_data = {
            file: pd.read_csv(kaggle_dataset_path.joinpath(file.value))
            for file in DownloadData.Files
        }

        logger.debug("Loaded %d CSV files", len(raw_data))
        return raw_data

    def _split_data(
        self, raw_data: Dict[DownloadData.Files, pd.DataFrame]
    ) -> Dict[Split, pd.DataFrame]:
        """Splits raw data into train, validation, and test sets."""
        logger.debug("Splitting data with validation size: %f", self.val_size)

        # Join test and test_labels on ID column
        test_inputs_df = raw_data[DownloadData.Files.TEST]
        test_labels_df = raw_data[DownloadData.Files.TEST_LABELS]
        test_df = test_inputs_df.merge(test_labels_df, on="id", validate="one_to_one")

        # Split train into train and val
        train_df = raw_data[DownloadData.Files.TRAIN]
        train_df, val_df = Preprocessing.iter_train_val_split(
            train_df,
            self.features,
            self.labels,
            self.val_size,
            self.shuffle,
        )

        splits = {
            Split.TRAIN: train_df,
            Split.VAL: val_df,
            Split.TEST: test_df,
        }

        logger.debug(
            "Split sizes - Train: %d, Val: %d, Test: %d",
            len(train_df),
            len(val_df),
            len(test_df),
        )
        return splits

    def _clean_data(self, data: dict[Split, pd.DataFrame]) -> dict[Split, pd.DataFrame]:
        """Cleans data by removing unnecessary columns and non-binary labels."""
        logger.debug("Cleaning data for all splits")

        cleaned_data = {
            split: Preprocessing.drop_non_binary_labels(
                df.drop(columns=["id"]), self.labels
            )
            for split, df in data.items()
        }

        logger.debug("Completed data cleaning")
        return cleaned_data

    def _convert_to_tensorflow(
        self, data: Dict[Split, pd.DataFrame]
    ) -> Dict[Split, Dataset]:
        """Converts pandas DataFrames to TensorFlow datasets with batching and prefetching."""
        logger.debug(
            "Converting to TensorFlow datasets with batch size: %d", self.batch_size
        )

        datasets = {}
        for split, df in data.items():
            features = df[self.features].values
            labels = df[self.labels].values
            ds = tf.data.Dataset.from_tensor_slices((features, labels))
            ds = ds.batch(self.batch_size, drop_remainder=True)
            datasets[split] = ds.cache().prefetch(tf.data.AUTOTUNE)
            logger.debug("Created %s dataset with %d batches", split.value, len(ds))

        return datasets


def main() -> None:
    DATA_PARAMS = {
        "val_size": 0.2,
        "batch_size": 128,
        "shuffle": True,
        "force_make": False,
    }

    datasets = Datasets(**DATA_PARAMS)

    train_ds = datasets.train
    val_ds = datasets.val
    test_ds = datasets.test

    print("Train batches:", len(train_ds))
    print("Validation batches:", len(val_ds))
    print("Test batches:", len(test_ds))


if __name__ == "__main__":
    main()
