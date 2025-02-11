"""
Download and validate raw machine learning datasets from Kaggle.
"""

import os
from pathlib import Path
from typing import Literal

import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from pydantic import BaseModel, ValidationError

from . import config


def download_data(
    force_download: bool = False,
    download_path: Path = config.DATA_DIR.joinpath("raw"),
) -> None:
    if force_download or not _raw_data_exists(download_path):
        print("Downloading data from Kaggle...")
        os.makedirs(download_path, exist_ok=True)

        train_df = _download_train()
        test_df = _download_test()

        _validate_df(train_df)
        _validate_df(test_df)

        train_df.to_csv(download_path.joinpath("train.csv"), index=False)
        test_df.to_csv(download_path.joinpath("test.csv"), index=False)
        print(f"Data downloaded to '{download_path}'.")
    else:
        print("Data already exists. Skipping download.")


def _download_train() -> pd.DataFrame:
    return kagglehub.load_dataset(
        adapter=KaggleDatasetAdapter.PANDAS,
        handle=config.DATASET,
        path="train.csv",
    )


def _download_test() -> pd.DataFrame:
    inputs_df = kagglehub.load_dataset(
        adapter=KaggleDatasetAdapter.PANDAS,
        handle=config.DATASET,
        path="test.csv",
    )

    labels_df = kagglehub.load_dataset(
        adapter=KaggleDatasetAdapter.PANDAS,
        handle=config.DATASET,
        path="test_labels.csv",
    )

    return pd.merge(inputs_df, labels_df, how="inner", on="id", validate="one_to_one")


def _raw_data_exists(path: Path) -> bool:
    return all(
        path.joinpath(dataset + ".csv").exists() for dataset in ["train", "test"]
    )


class DataSchema(BaseModel):
    id: str
    comment_text: str
    toxic: Literal[0, -1, 1]
    severe_toxic: Literal[0, -1, 1]
    obscene: Literal[0, -1, 1]
    threat: Literal[0, -1, 1]
    insult: Literal[0, -1, 1]
    identity_hate: Literal[0, -1, 1]


def _validate_df(df: pd.DataFrame) -> None:
    print("Validating dataset...")

    samples = df.to_dict(orient="records")
    validation_errors = []

    # First pass: collect all validation errors
    for idx, sample in enumerate(samples):
        try:
            DataSchema(**sample)
        except ValidationError as e:
            validation_errors.append(f"Row {idx}: {str(e)}")

    # If any errors found, raise with all error details
    if validation_errors:
        error_msg = "\n".join(validation_errors)
        raise ValueError(
            f"Dataset validation failed with following errors:\n{error_msg}"
        )


if __name__ == "__main__":
    download_data(force_download=True)
