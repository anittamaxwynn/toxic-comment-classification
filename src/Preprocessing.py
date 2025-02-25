from typing import List, Tuple

import keras
import pandas as pd
import tensorflow as tf
from skmultilearn.model_selection import iterative_train_test_split

# ----- Type aliases -----

Dataset = tf.data.Dataset
DatasetPair = Tuple[tf.data.Dataset, Tuple[tf.data.Dataset, ...]]
TextVectorizer = keras.layers.TextVectorization


# ----- Pandas DataFrame preprocessing -----


def iter_train_val_split(
    df: pd.DataFrame,
    features: list[str],
    labels: list[str],
    val_size: float,
    shuffle: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train and test sets."""
    if shuffle:
        df = df.sample(frac=1, random_state=0)

    X = df[features].to_numpy()
    y = df[labels].to_numpy()

    X_train, y_train, X_val, y_val = iterative_train_test_split(
        X.reshape(-1, 1),
        y,
        val_size,
    )

    train_df = pd.DataFrame(columns=df.columns)
    val_df = pd.DataFrame(columns=df.columns)

    train_df[features] = X_train
    val_df[features] = X_val

    train_df[labels] = y_train
    val_df[labels] = y_val

    assert len(train_df) + len(val_df) == len(df)
    return train_df, val_df


def drop_non_binary_labels(
    df: pd.DataFrame,
    label_columns: str | List[str],
) -> pd.DataFrame:
    """Remove rows with non-binary labels."""
    binary_condition = df[label_columns].isin([0, 1]).all(axis=1)
    non_binary_indices = list(set(df.index) - set(df[binary_condition].index))
    return df.drop(non_binary_indices).reset_index(drop=True)


# ----- TensorFlow Dataset preprocessing -----


def vectorize_dataset(dataset: Dataset, vectorize_layer: TextVectorizer) -> Dataset:
    """Apply vectorization to a dataset."""
    return dataset.map(
        lambda x, y: (vectorize_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )


def optimize_dataset(dataset: Dataset) -> Dataset:
    """Apply performance optimizations to datasets."""
    return dataset.cache().prefetch(tf.data.AUTOTUNE)


def optimize_dataset_pair(
    text_dataset: Dataset, label_datasets: Tuple[Dataset, ...]
) -> DatasetPair:
    """Apply performance optimizations to dataset pair."""
    text_dataset = optimize_dataset(text_dataset)
    label_datasets = tuple(optimize_dataset(ds) for ds in label_datasets)
    return text_dataset, label_datasets


def convert_to_dataset(
    df: pd.DataFrame,
    features: str | List[str],
    labels: str | List[str],
) -> Dataset:
    """Convert pandas DataFrame to TensorFlow Dataset."""
    return tf.data.Dataset.from_tensor_slices((df[features].values, df[labels].values))


def split_dataset(
    dataset: Dataset,
    test_size: float,
    shuffle: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Split dataset into train and test sets."""
    n_samples = len(dataset)
    n_test_samples = int(n_samples * test_size)
    if shuffle:
        dataset = dataset.shuffle(n_samples, seed=0)
    return dataset.skip(n_test_samples), dataset.take(n_test_samples)
