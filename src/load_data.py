from pathlib import Path
import logging
from typing import Tuple
import numpy as np
import pandas as pd
from .config import INPUT, LABELS
from .download import DatasetType, DatasetScheme

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Class for loading and processing Kaggle datasets."""

    def load_data(
        self,
        data_dir: Path | str,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Load both training and test datasets."""
        self._validate_data_dir(data_dir)

        train_data = self._load_train_data(data_dir)
        test_data = self._load_test_data(data_dir)

        return train_data, test_data

    @staticmethod
    def _load_train_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data and labels."""
        train_schema = DatasetScheme.get_schema(DatasetType.TRAIN)
        train_file_path = data_dir / train_schema.filename
        train_df = pd.read_csv(train_file_path)

        inputs = train_df[INPUT].values
        labels = train_df[LABELS].values

        logger.info(
            f"Loaded raw train data with shapes: inputs={inputs.shape}, labels={labels.shape}"
        )

        return (inputs, labels)

    @staticmethod
    def _load_test_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load testing data and labels."""
        test_schema = DatasetScheme.get_schema(DatasetType.TEST)
        test_labels_schema = DatasetScheme.get_schema(DatasetType.TEST_LABELS)

        test_file_path = data_dir / test_schema.filename
        test_labels_file_path = data_dir / test_labels_schema.filename

        test_df = pd.read_csv(test_file_path)
        test_labels_df = pd.read_csv(test_labels_file_path)

        inputs = test_df[INPUT].values
        labels = test_labels_df[LABELS].values

        logger.info(f"Loaded raw test data with shapes: inputs={inputs.shape}, labels={labels.shape}")

        return (inputs, labels)

    @staticmethod
    def _validate_data_dir(data_dir: Path) -> None:
        """Validates the data directory."""
        for dataset_type in DatasetType:
            schema = DatasetScheme.get_schema(dataset_type)
            file_path = data_dir / schema.filename
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Data file not found: {file_path}."
                    "Please download the data first using `download_data()`."
                )
