import logging
from enum import Enum

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DataPreprocessor:
    def preprocess(
        self,
        inputs=np.ndarray,
        labels=np.ndarray,
        split=DataSplit,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(inputs) != len(labels):
            raise ValueError(
                f"The number of inputs and labels must be equal: {len(inputs)} != {len(labels)}"
            )

        logger.info(f"Preprocessing {split.name} data...")

        inputs, labels = self._remove_duplicates(inputs, labels, split)
        inputs, labels = self._remove_missing_values(inputs, labels, split)
        inputs, labels = self._remove_untested_samples(inputs, labels, split)

        logger.info(f"Preprocessing {split.name} data complete.")

        return (inputs, labels)

    @staticmethod
    def _remove_duplicates(
        inputs=np.ndarray,
        labels=np.ndarray,
        split=DataSplit,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Removes duplicate rows."""
        unique_inputs, indices = np.unique(inputs, return_index=True)
        num_duplicates = inputs.shape[0] - unique_inputs.shape[0]

        if num_duplicates > 0:
            sorted_indices = np.sort(indices)
            inputs = inputs[sorted_indices]
            labels = labels[sorted_indices]

            logger.info(f"Removed {num_duplicates} duplicate rows from {split.name}.")

        return (inputs, labels)

    @staticmethod
    def _remove_missing_values(
        inputs=np.ndarray,
        labels=np.ndarray,
        split=DataSplit,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Removes rows with missing values from both X and y arrays.
        Missing inputs (X) are indicated by empty strings (length 0).
        Missing labels (y) are indicated by 'N/A'.

        Args:
            data: Tuple of (X, y) arrays where X contains string inputs and y contains labels

        Returns:
            Tuple of cleaned (X, y) arrays with missing values removed
        """
        # Find indices of missing values
        missing_input_indices = np.where([len(str(x)) == 0 for x in inputs])[0]
        missing_label_indices = np.where(labels == "N/A")[0]

        # Combine all indices to drop
        indices_to_drop = np.unique(
            np.concatenate([missing_input_indices, missing_label_indices])
        )

        if len(indices_to_drop) > 0:
            # Create mask of indices to keep
            keep_indices = np.ones(len(inputs), dtype=bool)
            keep_indices[indices_to_drop] = False

            inputs = inputs[keep_indices]
            labels = labels[keep_indices]

            logger.info(
                f"Removed {len(indices_to_drop)} samples with missing values from {split.name}."
            )

        return (inputs, labels)

    @staticmethod
    def _remove_untested_samples(
        inputs=np.ndarray,
        labels=np.ndarray,
        split=DataSplit,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Removes rows with invalid labels (e.g. not 0 or 1)."""
        num_untested = np.sum(np.any(labels == -1, axis=1))

        if num_untested > 0:
            mask = np.any(labels == -1, axis=1)
            inputs = inputs[~mask]
            labels = labels[~mask]

            logger.info(f"Removed {num_untested} untested samples from {split.name}.")

        return (inputs, labels)
