import os
from pathlib import Path

import pandas as pd

from . import config


def preprocess_data(
    data_dir: Path = config.DATA_DIR,
    val_size: float = 0.2,
    shuffle: bool = True,
    force_preprocess: bool = False,
) -> None:
    if force_preprocess or not _preprocess_data_exists(data_dir):
        print("Preprocessing data...")
        train_df = _load_raw_train_df(data_dir)
        test_df = _load_raw_test_df(data_dir)

        train_df = preprocess_df(train_df)
        test_df = preprocess_df(test_df)

        train_df, val_df = split_df(train_df, test_size=val_size, shuffle=shuffle)

        print("Saving preprocessed data...")
        processed_dir = data_dir.joinpath("processed")
        os.makedirs(processed_dir, exist_ok=True)
        train_df.to_csv(processed_dir.joinpath("train.csv"), index=False)
        val_df.to_csv(processed_dir.joinpath("val.csv"), index=False)
        test_df.to_csv(processed_dir.joinpath("test.csv"), index=False)
        print("Preprocessing complete.")
    else:
        print("Preprocessed data already exists. Skipping preprocessing.")


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna().reset_index(drop=True)
    df = _drop_empty_text(df, text_column=config.INPUT)
    df = _drop_non_binary_labels(df, label_columns=config.LABELS)
    return df


def split_df(
    df: pd.DataFrame, test_size: float = 0.2, shuffle: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_size < 0 or test_size > 1:
        raise ValueError("Test size must be a float value between 0 and 1.")
    n_samples = len(df)
    n_test_samples = int(n_samples * test_size)
    if shuffle:
        df = df.sample(frac=1, random_state=config.SEED)
    train_df = df.iloc[:n_test_samples].reset_index(drop=True)
    test_df = df.iloc[n_test_samples:].reset_index(drop=True)
    return train_df, test_df


def _load_raw_train_df(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir.joinpath("raw/train.csv"))


def _load_raw_test_df(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir.joinpath("raw/test.csv"))


def _drop_non_binary_labels(
    df: pd.DataFrame, label_columns: str | list[str]
) -> pd.DataFrame:
    binary_condition = df[label_columns].isin([0, 1]).all(axis=1)
    non_binary_indices = list(set(df.index) - set(df[binary_condition].index))
    return df.drop(non_binary_indices).reset_index(drop=True)


def _drop_empty_text(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    return df[df[text_column] != ""].reset_index(drop=True)  # type: ignore


def _preprocess_data_exists(data_dir: Path) -> bool:
    processed_dir = data_dir.joinpath("processed")
    return all(
        processed_dir.joinpath(dataset + ".csv").exists()
        for dataset in ["train", "val", "test"]
    )


if __name__ == "__main__":
    preprocess_data(force_preprocess=True)
