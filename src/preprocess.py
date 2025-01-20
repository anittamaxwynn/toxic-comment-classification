import logging
import numpy as np
from sklearn.model_selection import train_test_split


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocesor for preprocessing the raw Kaggle datasets optimized into TensorFlow datasets."""

    def preprocess_data(
        self, data: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Performs data cleaning and preprocessing."""
        (inputs, labels) = data

        if inputs.shape[0] != labels.shape[0]:
            raise ValueError(
                f"The number of inputs and labels ae not equal: {inputs.shape[0]} != {labels.shape[0]}"
            )

        data = self._remove_duplicates(data)
        data = self._remove_missing_values(data)
        data = self._remove_untested_samples(data)

        return data

    def split_data(
        self,
        data: tuple[np.ndarray, np.ndarray],
        split_size: float,
        shuffle: bool,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Splits the data into training and validation sets."""
        (inputs, labels) = data
        if inputs.shape[0] != labels.shape[0]:
            raise ValueError(
                f"The number of inputs and labels ae not equal: {inputs.shape[0]} != {labels.shape[0]}"
            )

        if split_size < 0 or split_size > 1:
            raise ValueError("Test size must be a float value between 0 and 1.")

        X_train, X_val, y_train, y_val = train_test_split(
            inputs, labels, test_size=split_size, shuffle=shuffle, random_state=42
        )

        logger.info(
            f"Split data into training and validation sets with sizes: train={X_train.shape[0]}, val={X_val.shape[0]}"
        )

        return (X_train, y_train), (X_val, y_val)

    @staticmethod
    def _remove_duplicates(
        data: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Removes duplicate rows."""
        (inputs, labels) = data
        unique_inputs, indices = np.unique(inputs, return_index=True)
        num_duplicates = inputs.shape[0] - unique_inputs.shape[0]

        if num_duplicates > 0:
            sorted_indices = np.sort(indices)
            inputs = inputs[sorted_indices]
            labels = labels[sorted_indices]

            logger.info(f"Removed {num_duplicates} duplicate rows.")

        return (inputs, labels)

    @staticmethod
    def _remove_missing_values(
        data: tuple[np.ndarray, np.ndarray],
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
        (inputs, labels) = data

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

            logger.info(f"Removed {len(indices_to_drop)} samples with missing values.")

        return (inputs, labels)

    @staticmethod
    def _remove_untested_samples(
        data: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Removes rows with invalid labels (e.g. not 0 or 1)."""
        (inputs, labels) = data
        num_untested = np.sum(np.any(labels == -1, axis=1))

        if num_untested > 0:
            mask = np.any(labels == -1, axis=1)
            inputs = inputs[~mask]
            labels = labels[~mask]

            logger.info(f"Removed {num_untested} untested samples.")

        return (inputs, labels)
