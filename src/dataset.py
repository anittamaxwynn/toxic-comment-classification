from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from . import config, load_data


def load_clean_data(
    split: load_data.Split,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the data from the raw data files.
    """
    df = load_data.load_and_validate_raw_df(split)
    df = process_df(df)

    assert isinstance(df, pd.DataFrame), "Expected df to be a DataFrame"

    features = df[config.INPUT].values.astype(str)
    labels = df[config.LABELS].values.astype(int)

    return features, labels


def make_tf_dataset(
    features: np.ndarray, labels: np.ndarray, batch_size: int
) -> tf.data.Dataset:
    """
    Convert the data to a TensorFlow dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    return dataset


def split_tf_dataset(
    dataset: tf.data.Dataset, test_size: float = 0.2, shuffle: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    if test_size < 0 or test_size > 1:
        raise ValueError("Test size must be a float value between 0 and 1.")

    n_samples = len(dataset)
    n_test_samples = int(n_samples * test_size)

    if shuffle:
        dataset = dataset.shuffle(n_samples, seed=0)

    test_ds = dataset.take(n_test_samples)
    train_ds = dataset.skip(n_test_samples)

    return train_ds, test_ds


def split_data(
    inputs: np.ndarray, labels: np.ndarray, test_size: float = 0.2
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split the data into training and testing sets.
    """
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(
        inputs, labels, test_size=test_size, random_state=0
    )

    train = (train_inputs, train_labels)
    test = (test_inputs, test_labels)

    assert isinstance(train, tuple), "Expected train to be a tuple"
    assert isinstance(test, tuple), "Expected test to be a tuple"

    return train, test


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the DataFrame
    """
    clean_df = df.copy()
    cols_to_keep = [config.INPUT] + config.LABELS

    clean_df = clean_df[[col for col in cols_to_keep if col in clean_df.columns]]
    clean_df = clean_df.drop_duplicates().reset_index(drop=True)

    assert isinstance(clean_df, pd.DataFrame), "Expected clean_df to be a DataFrame"

    clean_df = _drop_non_binary_labels(clean_df)

    return clean_df


def _drop_non_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to include only binary labels
    """
    clean_df = df.copy()
    clean_df = clean_df[clean_df[config.LABELS].isin([0, 1]).all(axis=1)]

    assert isinstance(clean_df, pd.DataFrame), "Expected clean_df to be a DataFrame"

    return clean_df
