from enum import Enum
from pathlib import Path

import pandas as pd

from . import config, download

logger = config.get_logger(__name__)


class DataSplit(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DataLoader:
    RAW_DATA_DIR: Path = config.DATA_DIR / "raw"

    def _raw_data_exists(self) -> bool:
        return all(
            self.RAW_DATA_DIR.joinpath(
                download.RawDataScheme.get_schema(dtype).filename
            ).exists()
            for dtype in download.RawDataType
        )

    def _load_raw_data(self, dtype: download.RawDataType) -> pd.DataFrame:
        schema = download.RawDataScheme.get_schema(dtype)
        filepath = self.RAW_DATA_DIR / schema.filename

        if not filepath.exists():
            raise FileNotFoundError(f"Raw data file not found: {filepath}")
        return pd.read_csv(filepath, dtype={config.ID: str})

    def load_train(self) -> pd.DataFrame:
        df = self._load_raw_data(download.RawDataType.TRAIN)
        df = self.preprocess(df, DataSplit.TRAIN)

        return df

    def load_test(self) -> pd.DataFrame:
        inputs = self._load_raw_data(download.RawDataType.TEST_INPUTS)
        labels = self._load_raw_data(download.RawDataType.TEST_LABELS)

        assert len(inputs) == len(labels), "Inputs and labels must have the same length"

        # Merge inputs and labels on ID column, insude 1:1 mapping
        df = inputs.merge(labels, on=config.ID, how="inner")

        assert len(df) == len(inputs), "Merged data must have the same length as inputs"

        df = self.preprocess(df, DataSplit.TEST)

        return df

    def preprocess(self, df: pd.DataFrame, split: DataSplit) -> pd.DataFrame:
        df = self._remove_non_binary_labels(df, split)
        df = self._remove_duplicates(df, split)
        df = self._remove_missing_data(df, split)

        return df

    def _remove_missing_data(self, df: pd.DataFrame, split: DataSplit) -> pd.DataFrame:
        """Remove samples with missing data."""
        df_filtered = df.dropna()
        if len(df_filtered) != len(df):
            logger.info(
                f"Removed {len(df) - len(df_filtered)} samples with missing data from {split.name} set"
            )
        return df_filtered.reset_index(drop=True)

    def _remove_duplicates(self, df: pd.DataFrame, split: DataSplit) -> pd.DataFrame:
        """Remove duplicate samples."""
        df_filtered = df.drop_duplicates(subset=config.ID, keep="first")

        if len(df_filtered) != len(df):
            logger.info(
                f"Removed {len(df) - len(df_filtered)} duplicate samples from {split.name} set"
            )

        return df_filtered.reset_index(drop=True)

    def _remove_non_binary_labels(
        self, df: pd.DataFrame, split: DataSplit
    ) -> pd.DataFrame:
        """Remove samples with non-binary labels."""
        df_filtered = df[df[config.LABELS].isin([0, 1]).all(axis=1)]

        if len(df_filtered) != len(df):
            logger.info(
                f"Removed {len(df) - len(df_filtered)} samples with non-binary labels from {split.name} set"
            )

        return df_filtered.reset_index(drop=True)


# class DataLoader(RawDataLoader):
#     def __init__(self, raw_data_dir: Path | str):
#         super().__init__(raw_data_dir)
#         self.preprocessor = DataPreprocessor()
#         self._dataset_config = None
#
#     def get_tensorflow_datasets(
#         self,
#         val_size: float,
#         batch_size: int,
#         shuffle: bool = True,
#         cache: bool = True,
#         prefetch: bool = True,
#     ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
#         self._dataset_config = {
#             "val_size": val_size,
#             "batch_size": batch_size,
#             "shuffle": shuffle,
#             "cache": cache,
#             "prefetch": prefetch,
#         }
#
#         train_inputs, train_labels = self.get_raw_train()
#         test_inputs, test_labels = self.get_raw_test()
#
#         train_inputs, train_label = self.preprocessor.preprocess(
#             train_inputs, train_labels, split=DataSplit.TRAIN
#         )
#
#         test_inputs, test_labels = self.preprocessor.preprocess(
#             test_inputs, test_labels, split=DataSplit.TEST
#         )
#
#         train_inputs, val_inputs, train_labels, val_labels = train_test_split(
#             train_inputs,
#             train_labels,
#             test_size=val_size,
#             shuffle=shuffle,
#             random_state=0,
#         )
#
#         logger.info(
#             f"Created validation set from training set with {len(val_inputs)} samples"
#         )
#
#         assert len(train_inputs) == len(train_labels), (
#             "Train inputs and labels must have the same length"
#         )
#         assert len(val_inputs) == len(val_labels), (
#             "Validation inputs and labels must have the same length"
#         )
#         assert len(test_inputs) == len(test_labels), (
#             "Test inputs and labels must have the same length"
#         )
#
#         train_ds = self._create_tf_dataset(
#             train_inputs, train_labels, split=DataSplit.TRAIN, batch_size=batch_size
#         )
#         val_ds = self._create_tf_dataset(
#             val_inputs, val_labels, split=DataSplit.VAL, batch_size=batch_size
#         )
#         test_ds = self._create_tf_dataset(
#             test_inputs, test_labels, split=DataSplit.TEST, batch_size=batch_size
#         )
#
#         return train_ds, val_ds, test_ds
#
#     def save_tensorflow_datasets(
#         self,
#         datasets: tuple[tf.data.Dataset, ...],
#         dir: Path | str,
#         overwrite: bool = True,
#     ) -> None:
#         if self._dataset_config is None:
#             msg = "No dataset configuration found. Please run `get_tensorflow_datasets` first."
#             raise ValueError(msg)
#
#         dir = Path(dir)
#         if not dir.exists():
#             dir.mkdir(parents=True, exist_ok=True)
#         elif overwrite:
#             shutil.rmtree(dir)
#             dir.mkdir(parents=True, exist_ok=True)
#
#         # Save datasets
#         for split, ds in zip(
#             [DataSplit.TRAIN, DataSplit.VAL, DataSplit.TEST], datasets
#         ):
#             ds_path = dir / split.value
#             ds.save(str(ds_path))
#
#         logger.info(f"Saved TensorFlow datasets to {dir}")
#
#     @staticmethod
#     def load_tensorflow_datasets(
#         dir: Path | str,
#     ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
#         """Load previously saved TensorFlow datasets."""
#         dir = Path(dir)
#         if not dir.exists():
#             raise ValueError(f"Directory {dir} does not exist")
#
#         train_ds = tf.data.Dataset.load(str(dir / DataSplit.TRAIN.value))
#         val_ds = tf.data.Dataset.load(str(dir / DataSplit.VAL.value))
#         test_ds = tf.data.Dataset.load(str(dir / DataSplit.TEST.value))
#
#         logger.info(f"Loaded TensorFlow datasets from {dir}")
#
#         return train_ds, val_ds, test_ds
#
#     @staticmethod
#     def _create_tf_dataset(
#         inputs: np.ndarray,
#         labels: np.ndarray,
#         split: DataSplit,
#         batch_size: int,
#         shuffle: bool = True,
#         cache: bool = True,
#         prefetch: bool = True,
#     ) -> tf.data.Dataset:
#         """Create a TensorFlow dataset from numpy arrays."""
#         ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
#
#         if split.name == "train" and shuffle:
#             ds = ds.shuffle(len(inputs), seed=0)
#
#         ds = ds.batch(batch_size)
#
#         if prefetch:
#             ds = ds.prefetch(tf.data.AUTOTUNE)
#
#         if cache:
#             ds = ds.cache()
#
#         logger.info(
#             f"Converted {split.name} data to TensorFlow dataset with {len(ds)} batches of size {batch_size}"
#         )
#
#         return ds


if __name__ == "__main__":
    loader = DataLoader()
    train_df = loader.load_train()
    test_df = loader.load_test()
    print(train_df.head())
    print(test_df.head())
