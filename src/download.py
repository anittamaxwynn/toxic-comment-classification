from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Set, Dict
import pandas as pd
import kagglehub
import shutil
from .config import DATASET, INPUT, LABELS

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetType(Enum):
    """Enum for the Kaggle dataset types."""

    TRAIN = auto()
    TEST = auto()
    TEST_LABELS = auto()


@dataclass
class DatasetScheme:
    filename: str
    valid_columns: Set[str]
    valid_dtypes: Dict[str, str]

    _BASE_DTYPES = {"id": "object"}
    _INPUT_DTYPES = {INPUT: "object"}
    _LABEL_DTYPES = {label: "int64" for label in LABELS}

    _BASE_COLUMNS = {"id"}
    _INPUT_COLUMNS = {INPUT}
    _LABEL_COLUMNS = set(LABELS)

    @classmethod
    def get_schema(cls, dataset_type: DatasetType) -> "DatasetScheme":
        schema_map = {
            DatasetType.TRAIN: cls(
                filename="train.csv",
                valid_columns=cls._BASE_COLUMNS
                | cls._INPUT_COLUMNS
                | cls._LABEL_COLUMNS,
                valid_dtypes=cls._merge_dicts(
                    cls._BASE_DTYPES, cls._INPUT_DTYPES, cls._LABEL_DTYPES
                ),
            ),
            DatasetType.TEST: cls(
                filename="test.csv",
                valid_columns=cls._BASE_COLUMNS | cls._INPUT_COLUMNS,
                valid_dtypes=cls._merge_dicts(cls._BASE_DTYPES, cls._INPUT_DTYPES),
            ),
            DatasetType.TEST_LABELS: cls(
                filename="test_labels.csv",
                valid_columns=cls._BASE_COLUMNS | cls._LABEL_COLUMNS,
                valid_dtypes=cls._merge_dicts(cls._BASE_DTYPES, cls._LABEL_DTYPES),
            ),
        }
        return schema_map[dataset_type]

    @staticmethod
    def _merge_dicts(*dicts: dict) -> dict:
        merged = {}
        for d in dicts:
            merged.update(d)
        return merged


class DatasetValidator:
    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame, dataset_type: DatasetType) -> None:
        logger.info("Validating %s dataset...", dataset_type.name)
        cls._validate_not_empty(df, dataset_type)
        cls._validate_columns(df, dataset_type)
        cls._validate_dtypes(df, dataset_type)

    @staticmethod
    def _validate_not_empty(df: pd.DataFrame, dataset_type: DatasetType) -> None:
        if df.empty:
            raise ValueError(f"The {dataset_type.name.lower()} dataset is empty.")

    @staticmethod
    def _validate_columns(df: pd.DataFrame, dataset_type: DatasetType) -> None:
        schema = DatasetScheme.get_schema(dataset_type)
        actual_columns = set(df.columns)

        missing_columns = schema.valid_columns - actual_columns
        if missing_columns:
            raise ValueError(
                f"The {dataset_type.name.lower()} dataset is missing columns: {missing_columns}"
            )

        unexpected_columns = actual_columns - schema.valid_columns
        if unexpected_columns:
            raise ValueError(
                f"The {dataset_type.name.lower()} dataset contains unexpected columns: {unexpected_columns}"
            )

    @staticmethod
    def _validate_dtypes(df: pd.DataFrame, dataset_type: DatasetType) -> None:
        schema = DatasetScheme.get_schema(dataset_type)
        for column, dtype in df.dtypes.items():
            expected_dtype = schema.valid_dtypes.get(column)
            if expected_dtype is None:
                raise ValueError(
                    f"The {dataset_type.name.lower()} dataset contains unexpected column: {column}"
                )
            if dtype != expected_dtype:
                raise ValueError(
                    f"The {dataset_type.name.lower()} dataset has invalid dtype for {column}: "
                    f"expected {expected_dtype}, got {dtype}"
                )


class DatasetDownloader:
    def __init__(self) -> None:
        self.validator = DatasetValidator()

    def download_data(self, download_path: Path | str, force: bool = False) -> None:
        download_path = Path(download_path)

        if self._raw_data_exists(download_path) and not force:
            logger.info(
                "Raw data already exists at %s. Skipping download. Use force=True to overwrite.",
                download_path,
            )
            return None

        logger.info("Downloading data from Kaggle...")
        cache_dir = Path(
            kagglehub.dataset_download(handle=DATASET, force_download=force)
        )
        self._copy_directory(cache_dir, download_path)

        self._validate_all_datasets(download_path)
        logger.info("Download complete. Data saved to %s", download_path)

    @staticmethod
    def _raw_data_exists(download_path: Path) -> bool:
        return all(
            download_path.joinpath(DatasetScheme.get_schema(dtype).filename).exists()
            for dtype in DatasetType
        )

    @staticmethod
    def _copy_directory(src: Path, dst: Path) -> None:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    def _validate_all_datasets(self, download_path: Path) -> None:
        test_df_size: int
        test_labels_df_size: int

        for dataset_type in DatasetType:
            schema = DatasetScheme.get_schema(dataset_type)
            df = pd.read_csv(download_path / schema.filename)
            self.validator.validate_dataframe(df, dataset_type)

            if dataset_type == DatasetType.TEST:
                test_df_size = len(df)
            elif dataset_type == DatasetType.TEST_LABELS:
                test_labels_df_size = len(df)

        if test_df_size != test_labels_df_size:
            raise ValueError(
                f"The test dataset has {test_df_size} rows, but the test labels dataset has {test_labels_df_size} rows."
            )
