import os
from pathlib import Path
from typing import Literal

import keras
import pandas as pd
import tensorflow as tf

from . import config


def make_datasets(
    val_size: float = 0.2,
    vocab_size: int = 10000,
    max_length: int = 100,
    shuffle: bool = True,
    force_preprocess: bool = False,
    data_dir: Path = config.DATA_DIR,
    model_dir: Path = config.MODEL_DIR,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    if force_preprocess or not _all_datasets_exist(data_dir):
        print("Preprocessing data...")
        train = _load_raw_train(data_dir)
        test = _load_raw_test(data_dir)

        train = _clean_dataframe(train)
        test = _clean_dataframe(test)

        train_ds = _convert_dataframe_to_dataset(train, config.INPUT, config.LABELS)
        test_ds = _convert_dataframe_to_dataset(test, config.INPUT, config.LABELS)

        train_ds, val_ds = _split_dataset(train_ds, test_size=val_size, shuffle=shuffle)

        vectorize_layer = keras.layers.TextVectorization(
            max_tokens=vocab_size, output_mode="int", output_sequence_length=max_length
        )
        print("Adapting vectorize layer...")
        vectorize_layer.adapt(train[config.INPUT])

        print("Vectorizing train data...")
        train_ds = train_ds.map(
            lambda x, y: (vectorize_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )

        print("Vectorizing val data...")
        val_ds = val_ds.map(
            lambda x, y: (vectorize_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )

        print("Vectorizing test data...")
        test_ds = test_ds.map(
            lambda x, y: (vectorize_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )

        print("Saving vectorize layer...")
        save_vectorize_layer(vectorize_layer, model_dir)

        print("Saving TensorFlow datasets...")
        processed_data_dir_str = str(data_dir.joinpath("processed"))
        os.makedirs(processed_data_dir_str, exist_ok=True)
        train_ds.save(processed_data_dir_str + "/train")
        val_ds.save(processed_data_dir_str + "/val")
        test_ds.save(processed_data_dir_str + "/test")

        return train_ds, val_ds, test_ds

    else:
        print("Loading datasets from disk...")
        train_ds = load_dataset("train", data_dir)
        val_ds = load_dataset("val", data_dir)
        test_ds = load_dataset("test", data_dir)

        return train_ds, val_ds, test_ds


def load_dataset(
    dataset: Literal["train", "val", "test"], data_dir: Path = config.DATA_DIR
) -> tf.data.Dataset:
    filepath = str(data_dir.joinpath(f"processed/{dataset}"))
    return tf.data.Dataset.load(filepath)


def save_vectorize_layer(
    vectorize_layer: keras.layers.TextVectorization, model_dir: Path = config.MODEL_DIR
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    vectorize_layer_model = keras.models.Sequential()
    vectorize_layer_model.add(keras.Input(shape=(1,), dtype=tf.string))
    vectorize_layer_model.add(vectorize_layer)
    filepath = model_dir.joinpath("vectorize_layer_model.keras")
    vectorize_layer_model.save(filepath)
    print(f"Vectorizer saved to {filepath}")
    return None


def load_vectorize_layer(model_dir: Path = config.MODEL_DIR) -> keras.layers.Layer:
    filepath = model_dir.joinpath("vectorize_layer_model.keras")
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


def _all_datasets_exist(data_dir: Path) -> bool:
    splits = ["train", "val", "test"]
    try:
        for split in splits:
            path = f"{data_dir}/processed/{split}"
            tf.data.Dataset.load(path)
        return True
    except (FileNotFoundError, tf.errors.NotFoundError):
        return False


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


# def _split_dataframe(
#     df: pd.DataFrame, test_size: float = 0.2, shuffle: bool = True
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     if test_size < 0 or test_size > 1:
#         raise ValueError("Test size must be a float value between 0 and 1.")
#     n_samples = len(df)
#     n_test_samples = int(n_samples * test_size)
#     if shuffle:
#         df = df.sample(frac=1, random_state=config.SEED)
#     test_df = df.iloc[:n_test_samples].reset_index(drop=True)
#     train_df = df.iloc[n_test_samples:].reset_index(drop=True)
#     return train_df, test_df


def _load_raw_train(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir.joinpath("raw/train.csv"))


def _load_raw_test(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir.joinpath("raw/test.csv"))


if __name__ == "__main__":
    train_ds, val_ds, test_ds = make_datasets(force_preprocess=True)
    print(len(train_ds))
    print(len(val_ds))
    print(len(test_ds))

    vectorize_layer = load_vectorize_layer()
    print(vectorize_layer.get_config())
