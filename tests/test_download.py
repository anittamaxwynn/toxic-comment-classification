import pytest
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import pytest

from src.download import DatasetType, DatasetScheme, DatasetValidator, DatasetDownloader
from src.config import INPUT, LABELS

from .mock_data import mock_train_df, mock_test_df, mock_test_labels_df


@pytest.fixture
def temp_dir(tmp_path) -> Path:
    """Creates a temporary data directory for testing"""
    return tmp_path / "data"


@pytest.fixture
def mock_kaggle_download(tmp_path):
    """Mocks the kagglehub.dataset_download function with valid CSV files"""
    kaggle_dir = tmp_path / "kaggle_data"
    kaggle_dir.mkdir(exist_ok=True)

    # Create dummy CSV files
    for dtype in DatasetType:
        schema = DatasetScheme.get_schema(dtype)
        if dtype == DatasetType.TRAIN:
            df = mock_train_df
        elif dtype == DatasetType.TEST:
            df = mock_test_df
        else:
            df = mock_test_labels_df

        df.to_csv(kaggle_dir / schema.filename, index=False)

    return kaggle_dir


class TestDatasetScheme:
    def test_get_schema_train(self) -> None:
        schema = DatasetScheme.get_schema(DatasetType.TRAIN)
        assert schema.filename == "train.csv"
        assert schema.valid_columns == {"id", INPUT} | set(LABELS)
        assert schema.valid_dtypes == {
            "id": "object",
            INPUT: "object",
            LABELS[0]: "int64",
            LABELS[1]: "int64",
            LABELS[2]: "int64",
            LABELS[3]: "int64",
            LABELS[4]: "int64",
            LABELS[5]: "int64",
        }

    def test_get_schema_test(self) -> None:
        schema = DatasetScheme.get_schema(DatasetType.TEST)
        assert schema.filename == "test.csv"
        assert schema.valid_columns == {"id", INPUT}
        assert schema.valid_dtypes == {
            "id": "object",
            INPUT: "object",
        }

    def test_get_schema_test_labels(self) -> None:
        schema = DatasetScheme.get_schema(DatasetType.TEST_LABELS)
        assert schema.valid_columns == {"id"} | set(LABELS)
        assert schema.valid_dtypes == {
            "id": "object",
            LABELS[0]: "int64",
            LABELS[1]: "int64",
            LABELS[2]: "int64",
            LABELS[3]: "int64",
            LABELS[4]: "int64",
            LABELS[5]: "int64",
        }

    def test_merge_dicts(self) -> None:
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 3, "c": 4}
        expected_dict = {"a": 1, "b": 3, "c": 4}
        assert DatasetScheme._merge_dicts(dict1, dict2) == expected_dict


class TestDatasetValidator:
    def test_empty_dataframe(self) -> None:
        validator = DatasetValidator()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="dataset is empty"):
            validator.validate_dataframe(empty_df, DatasetType.TRAIN)

    def test_missing_columns(self, mock_train_df) -> None:
        validator = DatasetValidator()

        missing_columns_df = mock_train_df.drop(columns=["id"])
        with pytest.raises(ValueError, match="missing columns"):
            validator.validate_dataframe(missing_columns_df, DatasetType.TRAIN)

    def test_extra_columns(self, mock_train_df) -> None:
        validator = DatasetValidator()

        extra_columns_df = mock_train_df.copy()
        extra_columns_df["extra_column"] = "extra_column"

        with pytest.raises(ValueError, match="unexpected columns"):
            validator.validate_dataframe(extra_columns_df, DatasetType.TRAIN)

    def test_missing_dtypes(self, mock_train_df) -> None:
        validator = DatasetValidator()

        missing_dtypes_df = mock_train_df.astype({"id": "int64"})
        with pytest.raises(ValueError, match="invalid dtype"):
            validator.validate_dataframe(missing_dtypes_df, DatasetType.TRAIN)

class TestDatasetDownloader:
    def test_init(self) -> None:
        downloader = DatasetDownloader()
        assert type(downloader.validator) is DatasetValidator

    def test_raw_data_exists(self, mock_kaggle_download) -> None:
        downloader = DatasetDownloader()
        assert downloader._raw_data_exists(mock_kaggle_download) is True

        # Test with missing file
        (mock_kaggle_download / "train.csv").unlink()
        assert downloader._raw_data_exists(mock_kaggle_download) is False

    # def test_copy_directory(self, tmp_path) -> None:
    #     """Test _copy_directory method."""
    #     src = tmp_path / "src"
    #     dst = tmp_path / "dst"
    #
    #     # Create source directory with a file
    #     src.mkdir()
    #     (src / "test.txt").write_text("test")
    #
    #     # Test copying to new directory
    #     downloader = DatasetDownloader()
    #     downloader._copy_directory(src, dst)
    #     assert dst.exists()
    #     assert (dst / "test.txt").read_text() == "test"
    #
    #     # Test copying to existing directory (should overwrite)
    #     (dst / "old.txt").write_text("old")
    #     downloader._copy_directory(src, dst)
    #     assert not (dst / "old.txt").exists()
    #     assert (dst / "test.txt").exists()


# class testdatavalidator:
#     def test_empty_dataframe(self):
#         validator = datasetvalidator()
#         empty_df = pd.dataframe()
#
#         with pytest.raises(valueerror, match="empty"):
#             validator.validate_dataframe(empty_df, datasettype.train)
#
#     @pytest.mark.parametrize(
#         "dataset_type", [datasettype.train, datasettype.test, datasettype.test_labels]
#     )
#     def test_missing_columns(self, dataset_type):
#         validator = datasetvalidator()
#
#         def get_missing_columns_df(dataset_type) -> pd.dataframe:
#             if dataset_type == datasettype.train:
#                 return mock_train_df.copy().drop(columns=["id"])
#             elif dataset_type == datasettype.test:
#                 return mock_test_df.copy().drop(columns=[cfg.input], inplace=false)
#             elif dataset_type == datasettype.test_labels:
#                 return mock_test_labels_df.copy().drop(
#                     columns=[cfg.labels[0]], inplace=false
#                 )
#
#         missing_columns_df = get_missing_columns_df(dataset_type)
#
#         with pytest.raises(valueerror, match="missing"):
#             validator.validate_dataframe(missing_columns_df, dataset_type)
#
#     def test_extra_columns(self):
#         validator = datasetvalidator()
#         extra_columns_df = mock_train_df.copy()
#         extra_columns_df["extra_column"] = "extra_column"
#
#         with pytest.raises(valueerror, match="unexpected columns"):
#             validator.validate_dataframe(extra_columns_df, datasettype.train)
#
#
# class testdatadownloader:
#     def test_init(self) -> none:
#         downloader = datasetdownloader()
#         assert downloader.validator == datasetvalidator()
#
#     def test_raw_data_exists(self, mock_kaggle_download):
#         downloader = datasetdownloader()
#         assert downloader._raw_data_exists(mock_kaggle_download) is true
#
#         # test with missing file
#         (mock_kaggle_download / "train.csv").unlink()
#         assert downloader._raw_data_exists(mock_kaggle_download) is false
#
#     def test_copy_directory(self, tmp_path):
#         """test _copy_directory method."""
#         src = tmp_path / "src"
#         dst = tmp_path / "dst"
#
#         # create source directory with a file
#         src.mkdir()
#         (src / "test.txt").write_text("test")
#
#         # test copying to new directory
#         downloader = datasetdownloader()
#         downloader._copy_directory(src, dst)
#         assert dst.exists()
#         assert (dst / "test.txt").read_text() == "test"
#
#         # test copying to existing directory (should overwrite)
#         (dst / "old.txt").write_text("old")
#         downloader._copy_directory(src, dst)
#         assert not (dst / "old.txt").exists()
#         assert (dst / "test.txt").exists()
#
#     @patch("kagglehub.dataset_download")
#     def test_download_success(self, mock_download, tmp_path, mock_kaggle_download):
#         """test successful download and validation."""
#         # create a seperate target directory
#         target_dir = tmp_path / "target"
#
#         # mock kaggle download  to return our mock directory
#         mock_download.return_value = str(mock_kaggle_download)
#
#         # test downloading data
#         downloader = datasetdownloader()
#         downloader.download(target_dir)
#
#         for dtype in datasettype:
#             schema = datasetscheme.get_schema(dtype)
#             assert (target_dir / schema.filename).exists()
#
#     @patch("kagglehub.dataset_download")
#     def test_download_existing_data(self, mock_download, mock_kaggle_download):
#         """test download when data alreafy exists."""
#         with pytest.raises(fileexistserror, match="already exists"):
#             downloader = datasetdownloader()
#             downloader.download(mock_kaggle_download)
#         mock_download.assert_not_called()
#
#     @patch("kagglehub.dataset_download")
#     def test_download_force(self, mock_download, tmp_path, mock_kaggle_download):
#         """test download when force is true."""
#         # create a seperate target directory with existing data
#         target_dir = tmp_path / "target"
#         shutil.copytree(mock_kaggle_download, target_dir)
#
#         # mock kaggle download
#         mock_download.return_value = str(mock_kaggle_download)
#
#         # test forced download
#         downloader = datasetdownloader()
#         downloader.download(target_dir, force_download=true)
#         mock_download.assert_called_once_with(force_download=true)
