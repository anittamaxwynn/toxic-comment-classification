"""
Module for preprocessing text data into TensorFlow datasets with parameter versioning.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import keras
import pandas as pd
import tensorflow as tf
from keras.api.layers import TextVectorization

from . import config

# ----------------------------
# TYPE ALIASES
# ----------------------------


Dataset = tf.data.Dataset
TextVectorizer = keras.layers.TextVectorization
DatasetDict = Dict[str, Dataset]


# ----------------------------
# CORE DATASET CREATION AND LOADING FUNCTIONS
# ----------------------------


def make_datasets(
    val_size: float,
    vocab_size: int,
    max_length: int,
    batch_size: int,
    shuffle: bool = True,
    optimize: bool = True,
    force_preprocess: bool = False,
    data_dir: Path = config.DATA_DIR,
    model_dir: Path = config.MODEL_DIR,
) -> Tuple[DatasetDict, str]:
    """Create or load preprocessed datasets with given parameters."""
    params = {
        "val_size": val_size,
        "vocab_size": vocab_size,
        "max_length": max_length,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "optimize": optimize,
    }
    param_hash = _create_parameter_hash(params)

    if force_preprocess or not _all_datasets_exist(data_dir, param_hash):
        datasets = _create_new_datasets(params, param_hash, data_dir, model_dir)
    else:
        datasets = _load_existing_datasets(params, param_hash, data_dir)

    if optimize:
        datasets = _optimize_datasets(datasets)

    return datasets, param_hash


def load_dataset(
    dataset: Literal["train", "val", "test"],
    data_dir: Path = config.DATA_DIR,
    param_hash: Optional[str] = None,
) -> Dataset:
    """Load a specific dataset with given parameter hash."""
    if param_hash is None:
        raise ValueError("param_hash must be provided")

    filepath = str(data_dir.joinpath(f"processed/{dataset}_{param_hash}"))
    if not os.path.exists(filepath):
        raise ValueError(
            f"No dataset found for {dataset} with parameter hash {param_hash}. "
            "Try running with force_preprocess=True"
        )
    return Dataset.load(filepath)


# ----------------------------
# VECTORIZER HANDLING FUNCTIONS
# ----------------------------


def save_vectorize_layer(
    vectorize_layer: TextVectorizer,
    model_dir: Path = config.MODEL_DIR,
    param_hash: Optional[str] = None,
) -> None:
    """Save vectorization layer with parameter hash."""
    if param_hash is None:
        raise ValueError("param_hash must be provided")

    os.makedirs(model_dir, exist_ok=True)
    vectorize_layer_model = keras.models.Sequential(
        [keras.Input(shape=(1,), dtype=tf.string), vectorize_layer]
    )

    filepath = model_dir.joinpath(f"vectorize_layer_model_{param_hash}.keras")
    vectorize_layer_model.save(filepath)
    print(f"Vectorizer saved to {filepath}")


def load_vectorize_layer(
    model_dir: Path = config.MODEL_DIR,
    param_hash: Optional[str] = None,
) -> keras.layers.Layer:
    """Load vectorization layer with specific parameters."""
    if param_hash is None:
        raise ValueError("param_hash must be provided")

    filepath = model_dir.joinpath(f"vectorize_layer_model_{param_hash}.keras")
    if not os.path.exists(filepath):
        raise ValueError(
            f"No vectorize layer found with parameter hash {param_hash}. "
            "Try running with force_preprocess=True"
        )

    vectorize_layer_model = keras.models.load_model(filepath)
    vectorize_layer = vectorize_layer_model.layers[-1]  # type: ignore
    assert vectorize_layer.built, "Vectorizer is not built (or adapted)"
    return vectorize_layer


# ----------------------------
# PARAMETER HANDLING FUNCTIONS
# ----------------------------


def _create_parameter_hash(params: Dict[str, Any]) -> str:
    """Create deterministic hash of preprocessing parameters."""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()[:10]


def _save_preprocessing_params(
    params: Dict[str, Any],
    data_dir: Path,
    param_hash: str,
) -> None:
    """Save preprocessing parameters to JSON file."""
    params_dir = data_dir.joinpath("processed/params")
    os.makedirs(params_dir, exist_ok=True)

    params_file = params_dir.joinpath(f"{param_hash}.json")
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2)


def _load_preprocessing_params(data_dir: Path, param_hash: str) -> Dict[str, Any]:
    """Load preprocessing parameters from JSON file."""
    params_file = data_dir.joinpath(f"processed/params/{param_hash}.json")
    if not params_file.exists():
        raise ValueError(f"No parameters file found for hash {param_hash}")

    with open(params_file) as f:
        return json.load(f)


# ----------------------------
# DATASET PROCESSING HELPER FUNCTIONS
# ----------------------------


def _create_new_datasets(
    params: Dict[str, Any],
    param_hash: str,
    data_dir: Path,
    model_dir: Path,
) -> Dict[str, Dataset]:
    """Create new preprocessed datasets from raw data."""
    print(f"Preprocessing data with parameter hash {param_hash}...")

    # Load and clean data
    train = _clean_dataframe(_load_raw_train(data_dir))
    test = _clean_dataframe(_load_raw_test(data_dir))

    # Create datasets
    train_ds = _convert_dataframe_to_dataset(train, config.INPUT, config.LABELS)
    test_ds = _convert_dataframe_to_dataset(test, config.INPUT, config.LABELS)
    train_ds, val_ds = _split_dataset(train_ds, params["val_size"], params["shuffle"])

    # Batch datasets
    train_ds = train_ds.batch(params["batch_size"], drop_remainder=True)
    val_ds = val_ds.batch(params["batch_size"], drop_remainder=True)
    test_ds = test_ds.batch(params["batch_size"], drop_remainder=True)

    # Create and adapt vectorizer
    vectorize_layer = TextVectorization(
        max_tokens=params["vocab_size"],
        output_mode="int",
        output_sequence_length=params["max_length"],
    )
    vectorize_layer.adapt(train[config.INPUT])

    # Vectorize datasets
    train_ds = _vectorize_dataset(train_ds, vectorize_layer)
    val_ds = _vectorize_dataset(val_ds, vectorize_layer)
    test_ds = _vectorize_dataset(test_ds, vectorize_layer)

    # Save all artifacts
    _save_preprocessing_params(params, data_dir, param_hash)
    save_vectorize_layer(vectorize_layer, model_dir, param_hash)
    _save_datasets(
        {"train": train_ds, "val": val_ds, "test": test_ds}, data_dir, param_hash
    )

    return {"train": train_ds, "val": val_ds, "test": test_ds}


def _load_existing_datasets(
    params: Dict[str, Any],
    param_hash: str,
    data_dir: Path,
) -> Dict[str, Dataset]:
    """Load existing preprocessed datasets."""
    print(f"Loading datasets with parameter hash {param_hash}...")
    loaded_params = _load_preprocessing_params(data_dir, param_hash)
    if loaded_params != params:
        raise ValueError(
            f"Provided parameters do not match parameters used to create dataset with hash {param_hash}."
        )

    return {
        "train": load_dataset("train", data_dir, param_hash),
        "val": load_dataset("val", data_dir, param_hash),
        "test": load_dataset("test", data_dir, param_hash),
    }


def _vectorize_dataset(dataset: Dataset, vectorize_layer: TextVectorizer) -> Dataset:
    """Apply vectorization to a dataset."""
    return dataset.map(
        lambda x, y: (vectorize_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )


def _optimize_datasets(datasets: Dict[str, Dataset]) -> Dict[str, Dataset]:
    """Apply performance optimizations to datasets."""
    return {
        key: ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        for key, ds in datasets.items()
        if isinstance(ds, Dataset)
    }


def _save_datasets(
    datasets: Dict[str, Dataset],
    data_dir: Path,
    param_hash: str,
) -> None:
    """Save datasets to disk with parameter hash."""
    processed_dir = data_dir.joinpath("processed")
    os.makedirs(processed_dir, exist_ok=True)

    for name, ds in datasets.items():
        ds.save(str(processed_dir.joinpath(f"{name}_{param_hash}")))


def _convert_dataframe_to_dataset(
    df: pd.DataFrame,
    features: str = config.INPUT,
    labels: List[str] = config.LABELS,
) -> Dataset:
    """Convert pandas DataFrame to TensorFlow Dataset."""
    return Dataset.from_tensor_slices((df[features].values, df[labels].values))


def _split_dataset(
    dataset: Dataset,
    test_size: float,
    shuffle: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Split dataset into train and test sets."""
    n_samples = len(dataset)
    n_test_samples = int(n_samples * test_size)
    if shuffle:
        dataset = dataset.shuffle(n_samples, seed=config.SEED)
    return dataset.skip(n_test_samples), dataset.take(n_test_samples)


# ----------------------------
# DATA CLEANING FUNCTIONS
# ----------------------------


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame by removing invalid entries."""
    df = df.dropna().reset_index(drop=True)
    df = _drop_empty_text(df, text_column=config.INPUT)
    df = _drop_non_binary_labels(df, label_columns=config.LABELS)
    return df


def _drop_non_binary_labels(
    df: pd.DataFrame,
    label_columns: str | List[str],
) -> pd.DataFrame:
    """Remove rows with non-binary labels."""
    binary_condition = df[label_columns].isin([0, 1]).all(axis=1)
    non_binary_indices = list(set(df.index) - set(df[binary_condition].index))
    return df.drop(non_binary_indices).reset_index(drop=True)


def _drop_empty_text(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Remove rows with empty text."""
    mask = df[text_column] != ""
    df = df.loc[mask].reset_index(drop=True)
    return df


def _all_datasets_exist(data_dir: Path, param_hash: str) -> bool:
    """Check if all preprocessed datasets exist for parameter hash."""
    processed_dir = data_dir.joinpath("processed")
    return all(
        processed_dir.joinpath(f"{dataset}_{param_hash}").exists()
        for dataset in ["train", "val", "test"]
    )


# Data loading functions
def _load_raw_train(data_dir: Path) -> pd.DataFrame:
    """Load raw training data."""
    return pd.read_csv(data_dir.joinpath("raw/train.csv"))


def _load_raw_test(data_dir: Path) -> pd.DataFrame:
    """Load raw test data."""
    return pd.read_csv(data_dir.joinpath("raw/test.csv"))


if __name__ == "__main__":
    datasets, param_hash = make_datasets(
        val_size=0.2,
        vocab_size=20000,
        max_length=250,
        batch_size=1024,
    )

    train_ds, val_ds, test_ds = datasets.values()

    print(len(train_ds), len(val_ds), len(test_ds))

    vectorize_layer = load_vectorize_layer(param_hash=param_hash)
    print(vectorize_layer.get_config())
