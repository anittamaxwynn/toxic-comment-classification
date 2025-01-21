import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .download import RawDataScheme, RawDataType
from .preprocess import DataPreprocessor, DataSplit

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RawDataLoader:
    def __init__(self, raw_data_dir: Path | str):
        self.raw_data_dir = Path(raw_data_dir)

    def get_raw_train(self) -> tuple[np.ndarray, np.ndarray]:
        schema = RawDataScheme.get_schema(RawDataType.TRAIN)
        filepath = self.raw_data_dir / schema.filename

        df = pd.read_csv(filepath)

        input_cols = list(schema._INPUT_COLUMNS)
        label_cols = list(schema._LABEL_COLUMNS)

        inputs = df[input_cols].values
        labels = df[label_cols].values

        assert len(inputs) == len(labels), "Inputs and labels must have the same length"

        return inputs, labels

    def get_raw_test(self) -> tuple[np.ndarray, np.ndarray]:
        input_schema = RawDataScheme.get_schema(RawDataType.TEST_INPUTS)
        label_schema = RawDataScheme.get_schema(RawDataType.TEST_LABELS)
        input_filepath = self.raw_data_dir / input_schema.filename
        label_filepath = self.raw_data_dir / label_schema.filename

        input_df = pd.read_csv(input_filepath)
        label_df = pd.read_csv(label_filepath)

        input_cols = list(input_schema._INPUT_COLUMNS)
        label_cols = list(label_schema._LABEL_COLUMNS)

        inputs = input_df[input_cols].values
        labels = label_df[label_cols].values

        assert len(inputs) == len(labels), "Inputs and labels must have the same length"

        return inputs, labels


class DataLoader(RawDataLoader):
    def __init__(self, raw_data_dir: Path | str):
        super().__init__(raw_data_dir)
        self.preprocessor = DataPreprocessor()
        self._dataset_config = None

    def get_tensorflow_datasets(
        self,
        val_size: float,
        batch_size: int,
        shuffle: bool = True,
        cache: bool = True,
        prefetch: bool = True,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        self._dataset_config = {
            "val_size": val_size,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "cache": cache,
            "prefetch": prefetch,
        }

        train_inputs, train_labels = self.get_raw_train()
        test_inputs, test_labels = self.get_raw_test()

        train_inputs, train_label = self.preprocessor.preprocess(
            train_inputs, train_labels, split=DataSplit.TRAIN
        )

        test_inputs, test_labels = self.preprocessor.preprocess(
            test_inputs, test_labels, split=DataSplit.TEST
        )

        train_inputs, val_inputs, train_labels, val_labels = train_test_split(
            train_inputs,
            train_labels,
            test_size=val_size,
            shuffle=shuffle,
            random_state=0,
        )

        logger.info(
            f"Created validation set from training set with {len(val_inputs)} samples"
        )

        assert len(train_inputs) == len(train_labels), (
            "Train inputs and labels must have the same length"
        )
        assert len(val_inputs) == len(val_labels), (
            "Validation inputs and labels must have the same length"
        )
        assert len(test_inputs) == len(test_labels), (
            "Test inputs and labels must have the same length"
        )

        train_ds = self._create_tf_dataset(
            train_inputs, train_labels, split=DataSplit.TRAIN, batch_size=batch_size
        )
        val_ds = self._create_tf_dataset(
            val_inputs, val_labels, split=DataSplit.VAL, batch_size=batch_size
        )
        test_ds = self._create_tf_dataset(
            test_inputs, test_labels, split=DataSplit.TEST, batch_size=batch_size
        )

        return train_ds, val_ds, test_ds

    def save_tensorflow_datasets(
        self,
        datasets: tuple[tf.data.Dataset, ...],
        dir: Path | str,
        overwrite: bool = True,
    ) -> None:
        if self._dataset_config is None:
            msg = "No dataset configuration found. Please run `get_tensorflow_datasets` first."
            raise ValueError(msg)

        dir = Path(dir)
        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)
        elif overwrite:
            shutil.rmtree(dir)
            dir.mkdir(parents=True, exist_ok=True)

        # Save datasets
        for split, ds in zip(
            [DataSplit.TRAIN, DataSplit.VAL, DataSplit.TEST], datasets
        ):
            ds_path = dir / split.value
            ds.save(str(ds_path))

        logger.info(f"Saved TensorFlow datasets to {dir}")

    @staticmethod
    def load_tensorflow_datasets(
        dir: Path | str,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load previously saved TensorFlow datasets."""
        dir = Path(dir)
        if not dir.exists():
            raise ValueError(f"Directory {dir} does not exist")

        train_ds = tf.data.Dataset.load(str(dir / DataSplit.TRAIN.value))
        val_ds = tf.data.Dataset.load(str(dir / DataSplit.VAL.value))
        test_ds = tf.data.Dataset.load(str(dir / DataSplit.TEST.value))

        logger.info(f"Loaded TensorFlow datasets from {dir}")

        return train_ds, val_ds, test_ds

    @staticmethod
    def _create_tf_dataset(
        inputs: np.ndarray,
        labels: np.ndarray,
        split: DataSplit,
        batch_size: int,
        shuffle: bool = True,
        cache: bool = True,
        prefetch: bool = True,
    ) -> tf.data.Dataset:
        """Create a TensorFlow dataset from numpy arrays."""
        ds = tf.data.Dataset.from_tensor_slices((inputs, labels))

        if split.name == "train" and shuffle:
            ds = ds.shuffle(len(inputs), seed=0)

        ds = ds.batch(batch_size)

        if prefetch:
            ds = ds.prefetch(tf.data.AUTOTUNE)

        if cache:
            ds = ds.cache()

        logger.info(
            f"Converted {split.name} data to TensorFlow dataset with {len(ds)} batches of size {batch_size}"
        )

        return ds
