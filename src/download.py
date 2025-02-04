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


class DataSchema(BaseModel):
    """Define schema for raw data."""

    id: str
    comment_text: str
    toxic: Literal[0, -1, 1]
    severe_toxic: Literal[0, -1, 1]
    obscene: Literal[0, -1, 1]
    threat: Literal[0, -1, 1]
    insult: Literal[0, -1, 1]
    identity_hate: Literal[0, -1, 1]


def download_data(
    dataset: Literal["train", "test"],
    download_path: Path = config.DATA_DIR / "raw",
    force_download: bool = False,
) -> None:
    """Download data from Kaggle."""
    if not force_download:
        if _raw_data_exists(download_path):
            print(
                f"The {dataset} dataset already exists at {download_path}. Skipping download."
            )
            return
        else:
            raise FileNotFoundError(
                f"The {dataset} dataset was not found in '{download_path}'. Please download the dataset first by setting `force_download=True`."
            )
    else:
        print("Downloading data from Kaggle...")
        os.makedirs(download_path, exist_ok=True)
        if dataset == "train":
            df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                config.DATASET,
                "train.csv",
            )
        else:
            inputs_df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                config.DATASET,
                "test.csv",
            )
            labels_df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                config.DATASET,
                "test_labels.csv",
            )
            df = pd.merge(inputs_df, labels_df, on="id", validate="one_to_one")

            assert len(df) == len(inputs_df) == len(labels_df)

        # Validate the dataset
        valid_samples, validation_errors = _validate_dataset(df)
        if validation_errors:
            print("Validation errors:")
            for error in validation_errors:
                print(f"Sample {error['index']}: {error['error']}")
            raise ValueError("Validation errors occurred.")

        # Convert the valid samples back to a DataFrame
        valid_df = pd.DataFrame([sample.model_dump() for sample in valid_samples])

        # Save the valid DataFrame to CSV
        valid_df.to_csv(download_path.joinpath(dataset + ".csv"), index=False)

        print(f"The {dataset} dataset was downloaded to '{download_path}'.")


def _validate_dataset(df: pd.DataFrame) -> tuple[list, list]:
    """Validate a DataFrame against its dataset type requirements."""
    print("Validating dataset...")

    valid_samples = []
    validation_errors = []

    samples = df.to_dict(orient="records")
    for idx, sample in enumerate(samples):
        try:
            valid_sample = DataSchema(**sample)
            valid_samples.append(valid_sample)
        except ValidationError as e:
            validation_errors.append({"index": idx, "error": e.errors()})

    return valid_samples, validation_errors


def _raw_data_exists(download_path: Path = config.DATA_DIR / "raw") -> bool:
    """Check if all required raw data files exist."""
    return all(
        download_path.joinpath(dataset + ".csv").exists()
        for dataset in ["train", "test"]
    )


def main() -> None:
    download_data(dataset="train", force_download=True)
    download_data(dataset="test", force_download=True)

    download_data(dataset="train", force_download=False)
    download_data(dataset="test", force_download=False)


if __name__ == "__main__":
    main()
