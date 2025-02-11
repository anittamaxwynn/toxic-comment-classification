import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import keras
import pandas as pd
import tensorflow as tf

from . import config


def _create_parameter_hash(params: Dict[str, Any]) -> str:
    """Create a deterministic hash of preprocessing parameters."""
    # Sort parameters to ensure consistent ordering
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()[:10]


def _save_preprocessing_params(
    params: Dict[str, Any], data_dir: Path, param_hash: str
) -> None:
    """Save preprocessing parameters to a JSON file."""
    params_dir = data_dir.joinpath("processed/params")
    os.makedirs(params_dir, exist_ok=True)

    params_file = params_dir.joinpath(f"{param_hash}.json")
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2)


def _load_preprocessing_params(data_dir: Path, param_hash: str) -> Dict[str, Any]:
    """Load preprocessing parameters from a JSON file."""
    params_file = data_dir.joinpath(f"processed/params/{param_hash}.json")
    if not params_file.exists():
        raise ValueError(f"No parameters file found for hash {param_hash}")

    with open(params_file) as f:
        return json.load(f)


def load_dataset(
    dataset: Literal["train", "val", "test"],
    data_dir: Path = config.DATA_DIR,
    param_hash: Optional[str] = None,
) -> tf.data.Dataset:
    """Load a dataset with specific preprocessing parameters."""
    if param_hash is None:
        raise ValueError("param_hash must be provided")

    filepath = str(data_dir.joinpath(f"processed/{dataset}_{param_hash}"))
    if not os.path.exists(filepath):
        raise ValueError(
            f"No dataset found for {dataset} with parameter hash {param_hash}. "
            "Try running with force_preprocess=True"
        )
    return tf.data.Dataset.load(filepath)


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
) -> Dict[str, tf.data.Dataset | str]:
    # Create parameter dictionary
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
        print(f"Preprocessing data with parameter hash {param_hash}...")
        print("Loading raw data...")
        train = _load_raw_train(data_dir)
        test = _load_raw_test(data_dir)

        print("Cleaning data...")
        train = _clean_dataframe(train)
        test = _clean_dataframe(test)

        print("Converting dataframes to datasets...")
        train_ds = _convert_dataframe_to_dataset(train, config.INPUT, config.LABELS)
        test_ds = _convert_dataframe_to_dataset(test, config.INPUT, config.LABELS)

        print("Splitting datasets...")
        train_ds, val_ds = _split_dataset(train_ds, test_size=val_size, shuffle=shuffle)

        print("Batching datasets...")
        train_ds = train_ds.batch(batch_size, drop_remainder=True)
        val_ds = val_ds.batch(batch_size, drop_remainder=True)
        test_ds = test_ds.batch(batch_size, drop_remainder=True)

        print("Creating vectorize layer...")
        vectorize_layer = keras.layers.TextVectorization(
            max_tokens=vocab_size, output_mode="int", output_sequence_length=max_length
        )
        print("Adapting vectorize layer...")
        vectorize_layer.adapt(train[config.INPUT])

        print("Vectorizing datasets...")
        train_ds = train_ds.map(
            lambda x, y: (vectorize_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        val_ds = val_ds.map(
            lambda x, y: (vectorize_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        test_ds = test_ds.map(
            lambda x, y: (vectorize_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )

        # Save preprocessing parameters
        print("Saving preprocessing parameters...")
        _save_preprocessing_params(params, data_dir, param_hash)

        # Save vectorize layer with parameter hash
        print("Saving vectorize layer...")
        save_vectorize_layer(vectorize_layer, model_dir, param_hash)

        # Save datasets with parameter hash
        print("Saving datasets...")
        processed_dir = data_dir.joinpath("processed")
        os.makedirs(processed_dir, exist_ok=True)
        train_ds.save(str(processed_dir.joinpath(f"train_{param_hash}")))
        val_ds.save(str(processed_dir.joinpath(f"val_{param_hash}")))
        test_ds.save(str(processed_dir.joinpath(f"test_{param_hash}")))
    else:
        print(f"Loading datasets with parameter hash {param_hash}...")
        # Load parameters from param_hash and check they match provided params
        loaded_params = _load_preprocessing_params(data_dir, param_hash)
        if loaded_params != params:
            raise ValueError(
                f"Provided parameters do not match parameters used to create dataset with hash {param_hash}."
            )

        train_ds = load_dataset("train", data_dir, param_hash)
        val_ds = load_dataset("val", data_dir, param_hash)
        test_ds = load_dataset("test", data_dir, param_hash)

    print("Optimizing datasets...")
    if optimize:
        train_ds = train_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "param_hash": param_hash,
    }


def save_vectorize_layer(
    vectorize_layer: keras.layers.TextVectorization,
    model_dir: Path = config.MODEL_DIR,
    param_hash: Optional[str] = None,
) -> None:
    """Save vectorize layer with parameter hash."""
    if param_hash is None:
        raise ValueError("param_hash must be provided")

    os.makedirs(model_dir, exist_ok=True)
    vectorize_layer_model = keras.models.Sequential()
    vectorize_layer_model.add(keras.Input(shape=(1,), dtype=tf.string))
    vectorize_layer_model.add(vectorize_layer)

    filepath = model_dir.joinpath(f"vectorize_layer_model_{param_hash}.keras")
    vectorize_layer_model.save(filepath)
    print(f"Vectorizer saved to {filepath}")
    return None


def load_vectorize_layer(
    model_dir: Path = config.MODEL_DIR, param_hash: Optional[str] = None
) -> keras.layers.Layer:
    """Load vectorize layer with specific preprocessing parameters."""
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


def _convert_dataframe_to_dataset(
    df: pd.DataFrame,
    features: str = config.INPUT,
    labels: list[str] = config.LABELS,
) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices((df[features].values, df[labels].values))


def _split_dataset(
    dataset: tf.data.Dataset,
    test_size: float,
    shuffle: bool = True,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    n_samples = len(dataset)
    n_test_samples = int(n_samples * test_size)
    dataset = dataset.shuffle(n_samples, seed=config.SEED) if shuffle else dataset
    test_ds = dataset.take(n_test_samples)
    train_ds = dataset.skip(n_test_samples)
    return train_ds, test_ds


def _all_datasets_exist(data_dir: Path, param_hash: str) -> bool:
    """Check if all preprocessed datasets exist for a given parameter hash."""
    processed_dir = data_dir.joinpath("processed")
    return all(
        processed_dir.joinpath(f"{dataset}_{param_hash}").exists()
        for dataset in ["train", "val", "test"]
    )


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna().reset_index(drop=True)
    df = _drop_empty_text(df, text_column=config.INPUT)
    df = _drop_non_binary_labels(df, label_columns=config.LABELS)
    return df


def _drop_non_binary_labels(
    df: pd.DataFrame, label_columns: str | list[str]
) -> pd.DataFrame:
    binary_condition = df[label_columns].isin([0, 1]).all(axis=1)
    non_binary_indices = list(set(df.index) - set(df[binary_condition].index))
    return df.drop(non_binary_indices).reset_index(drop=True)


def _drop_empty_text(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    return df[df[text_column] != ""].reset_index(drop=True)  # type: ignore


def _load_raw_train(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir.joinpath("raw/train.csv"))


def _load_raw_test(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir.joinpath("raw/test.csv"))


if __name__ == "__main__":
    datasets = make_datasets(
        val_size=0.2,
        vocab_size=20000,
        max_length=250,
        batch_size=32,
    )

    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]
    param_hash = datasets["param_hash"]

    print(len(train_ds))
    print(len(val_ds))
    print(len(test_ds))

    vectorize_layer = load_vectorize_layer(param_hash=param_hash)
    print(vectorize_layer.get_config())
