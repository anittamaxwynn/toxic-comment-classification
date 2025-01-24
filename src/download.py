"""
Download and validate raw machine learning datasets from Kaggle.

Provides classes for dataset type management, validation, and downloading.
"""

import shutil
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Self

import kagglehub
import pandas as pd

from . import config

logger = config.get_logger(__name__)


class RawDataType(Enum):
    """Enumerate different raw dataset types."""

    TRAIN = auto()
    TEST_INPUTS = auto()
    TEST_LABELS = auto()


@dataclass
class RawDataScheme:
    """Define schema for raw data files with filename, valid columns, and data types."""

    filename: str
    valid_columns: set[str]
    valid_types: dict[str, str]

    _INPUT: str = field(default_factory=lambda: config.INPUT)
    _LABELS: list[str] = field(default_factory=lambda: config.LABELS)
    _ID: str = field(default_factory=lambda: config.ID)
    _TYPES: dict[str, str] = field(default_factory=lambda: config.TYPES)

    @classmethod
    def get_schema(cls, dataset_type: RawDataType) -> Self:
        """Generate schema based on the dataset type."""
        if dataset_type == RawDataType.TRAIN:
            filename = "train.csv"
            valid_columns = {
                cls._get_id_col(),
                cls._get_input_col(),
                *cls._get_label_cols(),
            }

        elif dataset_type == RawDataType.TEST_INPUTS:
            filename = "test.csv"
            valid_columns = {cls._get_id_col(), cls._get_input_col()}
        else:
            filename = "test_labels.csv"
            valid_columns = {cls._get_id_col(), *cls._get_label_cols()}

        valid_types = {col: cls._get_col_types()[col] for col in valid_columns}

        return cls(filename, valid_columns, valid_types)

    @classmethod
    def _get_id_col(cls):
        """Get the ID column name."""
        return config.ID

    @classmethod
    def _get_input_col(cls):
        """Get the input column name."""
        return config.INPUT

    @classmethod
    def _get_label_cols(cls):
        """Get the label column names."""
        return config.LABELS

    @classmethod
    def _get_col_types(cls):
        """Get column type configurations."""
        return config.TYPES


class RawDataValidator:
    """Validate raw dataset integrity and structure."""

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame, dataset_type: RawDataType) -> None:
        """Validate a DataFrame against its dataset type requirements."""
        logger.info("Validating %s dataset...", dataset_type.name)
        cls._validate_size(df, dataset_type)
        cls._validate_columns(df, dataset_type)
        cls._validate_types(df, dataset_type)

    @staticmethod
    def _validate_size(df: pd.DataFrame, dataset_type: RawDataType) -> None:
        """Ensure the dataset is not empty."""
        if df.empty:
            raise ValueError(f"The {dataset_type.name.lower()} dataset is empty.")

    @staticmethod
    def _validate_columns(df: pd.DataFrame, dataset_type: RawDataType) -> None:
        """Verify DataFrame columns match the expected schema."""
        schema = RawDataScheme.get_schema(dataset_type)

        missing_columns = schema.valid_columns - set(df.columns)
        unexpected_columns = set(df.columns) - schema.valid_columns

        if missing_columns:
            raise ValueError(
                f"The {dataset_type.name.lower()} dataset is missing columns: {missing_columns}"
            )

        if unexpected_columns:
            raise ValueError(
                f"The {dataset_type.name.lower()} dataset contains unexpected columns: {unexpected_columns}"
            )

    @staticmethod
    def _validate_types(df: pd.DataFrame, dataset_type: RawDataType) -> None:
        """Ensure column data types match the expected schema."""
        schema = RawDataScheme.get_schema(dataset_type)
        for column, dtype in df.dtypes.items():
            valid_type = schema.valid_types[column]
            if dtype != valid_type:
                raise ValueError(
                    f"The {dataset_type.name.lower()} dataset has invalid dtype for {column}: "
                    f"expected {valid_type}, got {dtype}"
                )


class DataDownloader:
    """Download and manage raw machine learning datasets from Kaggle."""

    DOWNLOAD_PATH: Path = config.DATA_DIR / "raw"

    def __init__(self) -> None:
        """Initialize the DataDownloader with a data validator."""
        self.validator = RawDataValidator()

    def download_data(self, force_download: bool = False) -> None:
        """
        Download data to specified path, with optional force redownload.

        Skips download if data exists unless force is True.
        """
        if not force_download and self._raw_data_exists():
            logger.info(
                f"Data already exists at {self.DOWNLOAD_PATH}. Skipping download."
            )
            return

        logger.info("Downloading data from Kaggle...")
        cache_dir = self._download_from_kaggle(force_download=force_download)

        self._prepare_download_dir()
        shutil.copytree(cache_dir, self.DOWNLOAD_PATH, dirs_exist_ok=True)

        self._validate_datasets()
        logger.info("Download complete. Data saved to %s", self.DOWNLOAD_PATH)

    def _raw_data_exists(self) -> bool:
        """Check if all required raw data files exist."""
        return all(
            self.DOWNLOAD_PATH.joinpath(
                RawDataScheme.get_schema(dtype).filename
            ).exists()
            for dtype in RawDataType
        )

    def _prepare_download_dir(self) -> None:
        """Prepare the download directory by clearing existing content."""
        if self.DOWNLOAD_PATH.exists():
            shutil.rmtree(self.DOWNLOAD_PATH)
        self.DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)

    def _validate_datasets(self) -> None:
        """Validate all downloaded datasets against their schemas."""
        test_input_size = None
        test_label_size = None

        for dataset_type in RawDataType:
            schema = RawDataScheme.get_schema(dataset_type)
            filepath = self.DOWNLOAD_PATH / schema.filename

            if not filepath.exists():
                raise FileNotFoundError(f"Expected dataset file missing: {filepath}")

            df = pd.read_csv(filepath, dtype={config.ID: str})
            self.validator.validate_dataframe(df, dataset_type)

            if dataset_type == RawDataType.TEST_INPUTS:
                test_input_size = len(df)
            elif dataset_type == RawDataType.TEST_LABELS:
                test_label_size = len(df)

        if test_input_size is not None and test_label_size is not None:
            if test_input_size != test_label_size:
                msg = f"Mismatch between test data ({test_input_size}) and test labels ({test_label_size}) rows."
                raise ValueError(msg)

    @staticmethod
    def _download_from_kaggle(force_download: bool) -> Path:
        """Download dataset from Kaggle, with optional force redownload."""
        try:
            return Path(
                kagglehub.dataset_download(
                    handle=config.DATASET, force_download=force_download
                )
            )
        except Exception as e:
            raise ValueError(f"Failed to download dataset: {e}")
