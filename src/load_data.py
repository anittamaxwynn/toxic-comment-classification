import os
from typing import Literal

import pandas as pd
import tensorflow as tf

from . import config


def load_dataset(
    dataset: Literal["train", "test"],
    batch_size: int,
    shuffle: bool = True,
    preprocess: bool = True,
) -> tf.data.Dataset:
    """Load preprocessed data."""
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")

    df = load_raw_df(dataset)

    if preprocess:
        df = preprocess_df(df)

    features = df[config.INPUT].values
    labels = df[config.LABELS].values

    ds = tf.data.Dataset.from_tensor_slices((features, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=ds.cardinality(), seed=config.SEED)

    return ds.batch(batch_size, drop_remainder=True)


def load_raw_df(dataset: Literal["train", "test"]) -> pd.DataFrame:
    """Load raw data from CSV files."""
    raw_data_path = f"{config.DATA_DIR}/raw"
    filepath = f"{raw_data_path}/{dataset}.csv"

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw {dataset} data not found in '{raw_data_path}'.")

    else:
        return pd.read_csv(filepath)


def split_dataset(
    dataset: tf.data.Dataset, test_size: float = 0.2, shuffle: bool = True
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Splits a TensorFlow dataset into training and testing (or validation) sets."""
    if test_size < 0 or test_size > 1:
        raise ValueError("Test size must be a float value between 0 and 1.")

    n_samples = len(dataset)
    n_test_samples = int(n_samples * test_size)

    if shuffle:
        dataset = dataset.shuffle(n_samples, seed=config.SEED)

    test_ds = dataset.take(n_test_samples)
    train_ds = dataset.skip(n_test_samples)

    assert len(train_ds) + len(test_ds) == n_samples

    return train_ds, test_ds


def preprocess_df(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess raw data."""
    df = raw_data.copy()

    # Drop unnecessary columns
    cols_to_keep = [config.INPUT] + config.LABELS
    df = df[cols_to_keep]

    assert isinstance(df, pd.DataFrame)

    # Drop non-binary labels
    df = _drop_non_binary_labels(df)

    return df


def _drop_non_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-binary labels."""
    df = df.copy()

    binary_condition = df[config.LABELS].isin([0, 1]).all(axis=1)
    non_binary_indices = list(set(df.index) - set(df[binary_condition].index))
    df = df.drop(non_binary_indices)

    return df


def main() -> None:
    train_ds = load_dataset(dataset="train", batch_size=32)
    test_ds = load_dataset(dataset="test", batch_size=32)

    train_ds, val_ds = split_dataset(train_ds, test_size=0.2, shuffle=True)

    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print(f"Validation samples: {len(val_ds)}")


if __name__ == "__main__":
    main()
