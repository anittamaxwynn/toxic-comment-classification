import shutil
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Self

import kagglehub
import pandas as pd

from . import config

logger = config.get_logger("my_logger")


class RawDataType(Enum):
    TRAIN = auto()
    TEST_INPUTS = auto()
    TEST_LABELS = auto()


@dataclass
class RawDataScheme:
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
        return config.ID

    @classmethod
    def _get_input_col(cls):
        return config.INPUT

    @classmethod
    def _get_label_cols(cls):
        return config.LABELS

    @classmethod
    def _get_col_types(cls):
        return config.TYPES


class RawDataValidator:
    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame, dataset_type: RawDataType) -> None:
        logger.info("Validating %s dataset...", dataset_type.name)
        cls._validate_size(df, dataset_type)
        cls._validate_columns(df, dataset_type)
        cls._validate_types(df, dataset_type)

    @staticmethod
    def _validate_size(df: pd.DataFrame, dataset_type: RawDataType) -> None:
        if df.empty:
            raise ValueError(f"The {dataset_type.name.lower()} dataset is empty.")

    @staticmethod
    def _validate_columns(df: pd.DataFrame, dataset_type: RawDataType) -> None:
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
        schema = RawDataScheme.get_schema(dataset_type)
        for column, dtype in df.dtypes.items():
            valid_type = schema.valid_types[column]
            if dtype != valid_type:
                raise ValueError(
                    f"The {dataset_type.name.lower()} dataset has invalid dtype for {column}: "
                    f"expected {valid_type}, got {dtype}"
                )


class DataDownloader:
    def __init__(self) -> None:
        self.validator = RawDataValidator()

    def download_data(self, download_path: Path | str, force: bool = False) -> None:
        download_path = Path(download_path)

        if not force and self._raw_data_exists(download_path):
            logger.info(
                "Raw data already exists at %s. Skipping download.", download_path
            )
            return

        logger.info("Downloading data from Kaggle...")
        cache_dir = self._download_from_kaggle(force)

        self._prepare_directory(download_path)
        shutil.copytree(cache_dir, download_path, dirs_exist_ok=True)

        self._validate_datasets(download_path)
        logger.info("Download complete. Data saved to %s", download_path)

    @staticmethod
    def _raw_data_exists(download_path: Path) -> bool:
        return all(
            download_path.joinpath(RawDataScheme.get_schema(dtype).filename).exists()
            for dtype in RawDataType
        )

    @staticmethod
    def _prepare_directory(download_path: Path) -> None:
        if download_path.exists():
            shutil.rmtree(download_path)
        download_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _download_from_kaggle(force: bool) -> Path:
        try:
            return Path(
                kagglehub.dataset_download(handle=config.DATASET, force_download=force)
            )
        except Exception as e:
            raise ValueError(f"Failed to download dataset: {e}")

    def _validate_datasets(self, download_path: Path) -> None:
        test_input_size = None
        test_label_size = None

        for dataset_type in RawDataType:
            schema = RawDataScheme.get_schema(dataset_type)
            filepath = download_path / schema.filename

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
