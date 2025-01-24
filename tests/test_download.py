from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src import config
from src.download import DataDownloader, RawDataScheme, RawDataType, RawDataValidator

from .mock_data import get_mock_data


@pytest.fixture
def downloader(tmp_path):
    downloader = DataDownloader()
    downloader.DOWNLOAD_PATH = tmp_path / "test" / "download"
    return downloader


@pytest.fixture
def validator():
    return RawDataValidator()


class TestDatasetScheme:
    def test_get_schema_train(self) -> None:
        train_type = RawDataType.TRAIN
        train_schema = RawDataScheme.get_schema(train_type)

        assert train_schema.filename == "train.csv"
        assert train_schema.valid_columns == {config.ID, config.INPUT} | set(
            config.LABELS
        )
        assert train_schema.valid_types == {
            config.ID: "object",
            config.INPUT: "object",
            **{label: "int64" for label in config.LABELS},
        }

    def test_get_schema_test_inputs(self) -> None:
        test_inputs_type = RawDataType.TEST_INPUTS
        test_inputs_schema = RawDataScheme.get_schema(test_inputs_type)

        assert test_inputs_schema.filename == "test.csv"
        assert test_inputs_schema.valid_columns == {config.ID, config.INPUT}
        assert test_inputs_schema.valid_types == {
            config.ID: "object",
            config.INPUT: "object",
        }

    def test_get_schema_test_labels(self) -> None:
        test_labels_type = RawDataType.TEST_LABELS
        test_labels_schema = RawDataScheme.get_schema(test_labels_type)

        assert test_labels_schema.filename == "test_labels.csv"
        assert test_labels_schema.valid_columns == {config.ID} | set(config.LABELS)
        assert test_labels_schema.valid_types == {
            config.ID: "object",
            **{label: "int64" for label in config.LABELS},
        }


class TestDatasetValidator:
    def test_validate_size(self, validator) -> None:
        df = pd.DataFrame()

        with pytest.raises(ValueError, match="dataset is empty"):
            validator._validate_size(df, RawDataType.TRAIN)

    def test_missing_columns(self, validator) -> None:
        df = get_mock_data(RawDataType.TRAIN).drop(columns=["id"])

        with pytest.raises(ValueError, match="missing columns"):
            validator.validate_dataframe(df, RawDataType.TRAIN)

    def test_extra_columns(self, validator) -> None:
        df = get_mock_data(RawDataType.TRAIN).copy()
        df["extra_column"] = "extra_column"

        with pytest.raises(ValueError, match="unexpected columns"):
            validator.validate_dataframe(df, RawDataType.TRAIN)

    def test_validate_types_error(self, validator) -> None:
        df = get_mock_data(RawDataType.TRAIN).astype({"id": "int64"})

        with pytest.raises(ValueError, match="invalid dtype"):
            validator.validate_dataframe(df, RawDataType.TRAIN)


class TestDatasetDownloader:
    def test_init(self, downloader) -> None:
        assert hasattr(downloader, "validator")
        assert type(downloader.validator) is RawDataValidator

    def test_raw_data_exists(self, downloader) -> None:
        """Tests the _raw_data_exists method."""
        # Create download path
        download_path = downloader.DOWNLOAD_PATH
        download_path.mkdir(parents=True, exist_ok=True)

        # Create mock files
        for dtype in RawDataType:
            schema = RawDataScheme.get_schema(dtype)
            filepath = download_path / schema.filename
            pd.DataFrame().to_csv(filepath, index=False)

        assert downloader._raw_data_exists() is True

        # Remove one file to test negative case
        train_filename = RawDataScheme.get_schema(RawDataType.TRAIN).filename
        (download_path / train_filename).unlink()
        assert downloader._raw_data_exists() is False

    def test_prepare_directory(self, downloader, tmp_path):
        """Tests the _prepare_directory method."""
        # Create existing directory with a file
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        (existing_dir / "test.txt").write_text("test")

        # Set download path to existing directory
        downloader.DOWNLOAD_PATH = existing_dir

        downloader._prepare_download_dir()

        # Check directory is is now empty and exists
        assert existing_dir.exists()
        assert len(list(existing_dir.iterdir())) == 0

    def test_download_from_kaggle_success(self, downloader):
        """Tests the _download_from_kaggle method with mocking."""
        mock_cache_path = Path("/mock/cache/path")

        with patch(
            "kagglehub.dataset_download", return_value=str(mock_cache_path)
        ) as mock_download:
            result = downloader._download_from_kaggle(force_download=True)

            # Verify kagglehub.dataset_download was called correctly
            mock_download.assert_called_once_with(
                handle=config.DATASET,
                force_download=True,
            )

            assert result == mock_cache_path

    def test_download_from_kaggle_failure(self, downloader):
        """Tests the _download_from_kaggle method with download failure."""
        with patch("kagglehub.dataset_download", side_effect=Exception("API Error")):
            with pytest.raises(ValueError, match="Failed to download dataset:"):
                downloader._download_from_kaggle(force_download=True)

    def test_download_data_skip_existing(self, downloader):
        """Test download_data skips when data exisits and force=False."""
        download_path = downloader.DOWNLOAD_PATH
        download_path.mkdir(parents=True, exist_ok=True)

        # Create mock existing files
        for dtype in RawDataType:
            schema = RawDataScheme.get_schema(dtype)
            filepath = download_path / schema.filename
            pd.DataFrame().to_csv(filepath, index=False)

        with (
            patch.object(downloader, "_download_from_kaggle") as mock_download,
            patch("shutil.copytree") as mock_copy,
            patch.object(downloader, "_validate_datasets") as mock_validate,
        ):
            downloader.download_data(force_download=False)

            # Verify no download or copy occured
            mock_download.assert_not_called()
            mock_copy.assert_not_called()
            mock_validate.assert_not_called()

    def test_download_data_full_flow(self, downloader):
        """Test full download_data method flow."""
        mock_cache_path = Path("/mock/cache/path")

        with (
            patch.object(
                downloader, "_download_from_kaggle", return_value=mock_cache_path
            ) as mock_download,
            patch("shutil.copytree") as mock_copy,
            patch.object(downloader, "_validate_datasets") as mock_validate,
        ):
            downloader.download_data(force_download=True)

            # Verify download and copy occurred
            mock_download.assert_called_once_with(force_download=True)
            mock_copy.assert_called_once()
            mock_validate.assert_called_once()

    def test_validate_datasets_success(self, downloader):
        """Test the _validate_datasets method."""
        download_path = downloader.DOWNLOAD_PATH
        download_path.mkdir(parents=True, exist_ok=True)

        # Create mock valid CSV files
        for dtype in RawDataType:
            schema = RawDataScheme.get_schema(dtype)
            filepath = download_path / schema.filename
            get_mock_data(dtype).to_csv(filepath, index=False)

        # This should not raise any errors
        assert downloader._validate_datasets() is None

    def test_validate_datasets_missing(self, downloader):
        """Test _validate_datasets raises error for missing files."""
        download_path = downloader.DOWNLOAD_PATH
        download_path.mkdir(parents=True, exist_ok=True)

        # Create only the train file
        train_schema = RawDataScheme.get_schema(RawDataType.TRAIN)
        train_path = download_path / train_schema.filename
        get_mock_data(RawDataType.TRAIN).to_csv(train_path, index=False)

        # This should raise an error
        with pytest.raises(FileNotFoundError, match="Expected dataset file missing:"):
            downloader._validate_datasets()

    def test_validate_datasets_test_mismatch(self, downloader):
        """Test _validate_datasets raises error for mismatched test data."""
        # Create mock CSV files, but test_inputs with 10 rows and test_labels with 5 rows
        download_path = downloader.DOWNLOAD_PATH
        download_path.mkdir(parents=True, exist_ok=True)

        for dtype in RawDataType:
            schema = RawDataScheme.get_schema(dtype)
            filepath = download_path / schema.filename
            if dtype == RawDataType.TEST_LABELS:
                get_mock_data(dtype).head(5).to_csv(filepath, index=False)
            else:
                get_mock_data(dtype).head(10).to_csv(filepath, index=False)

        # This should raise an error
        with pytest.raises(ValueError, match="Mismatch between test data"):
            downloader._validate_datasets()
