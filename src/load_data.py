import os
from typing import Literal

import numpy as np
import pandas as pd
import tensorflow as tf

from . import config


def main() -> None:
    train_ds, val_ds, test_ds = load_datasets(val_size=0.2, shuffle=True)

    print("Train samples: ", len(train_ds))
    print("Validation samples: ", len(val_ds))
    print("Test samples: ", len(test_ds))


def load_datasets(val_size: float, shuffle: bool = True):
    """Load preprocessed data."""
    # Load preprocessed dataframes
    train_df, val_df, test_df = load_preprocessed_dfs(
        val_size=val_size, shuffle=shuffle
    )

    # Convert to TensorFlow datasets
    train_ds = make_ds(
        train_df[config.INPUT].to_numpy(), train_df[config.LABELS].to_numpy()
    )
    val_ds = make_ds(val_df[config.INPUT].to_numpy(), val_df[config.LABELS].to_numpy())
    test_ds = make_ds(
        test_df[config.INPUT].to_numpy(), test_df[config.LABELS].to_numpy()
    )

    return train_ds, val_ds, test_ds


def make_ds(features: np.ndarray, labels: np.ndarray) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.shuffle(buffer_size=ds.cardinality(), seed=config.SEED)
    return ds


def optimize_ds(ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


def load_preprocessed_dfs(
    val_size: float, shuffle: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed data."""
    # Load raw dataframes
    raw_train_df = load_raw_df(dataset="train")
    raw_test_df = load_raw_df(dataset="test")

    # Break off validation set from training set
    train_df, val_df = split_df(raw_train_df, test_size=val_size, shuffle=shuffle)

    # Preprocess dataframes
    train_df = preprocess_df(train_df)
    val_df = preprocess_df(val_df)
    test_df = preprocess_df(raw_test_df)

    return train_df, val_df, test_df


def load_raw_df(dataset: Literal["train", "test"]) -> pd.DataFrame:
    """Load raw data from CSV files."""
    raw_data_path = f"{config.DATA_DIR}/raw"
    filepath = f"{raw_data_path}/{dataset}.csv"

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw {dataset} data not found in '{raw_data_path}'.")

    else:
        return pd.read_csv(filepath)


def split_df(
    df: pd.DataFrame, test_size: float = 0.2, shuffle: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_size < 0 or test_size > 1:
        raise ValueError("Test size must be a float value between 0 and 1.")
    n_samples = len(df)
    n_test_samples = int(n_samples * test_size)
    if shuffle:
        df = df.sample(frac=1, random_state=config.SEED)
    test_df = df.iloc[:n_test_samples]
    train_df = df.iloc[n_test_samples:]
    return train_df, test_df


def preprocess_df(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess raw data."""
    df = raw_data.copy()

    cols_to_keep = [config.INPUT] + config.LABELS
    df = df[cols_to_keep]

    assert isinstance(df, pd.DataFrame)
    df = _drop_non_binary_labels(df)
    return df


def _drop_non_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-binary labels."""
    df = df.copy()
    binary_condition = df[config.LABELS].isin([0, 1]).all(axis=1)
    non_binary_indices = list(set(df.index) - set(df[binary_condition].index))
    df = df.drop(non_binary_indices)
    return df


if __name__ == "__main__":
    main()
