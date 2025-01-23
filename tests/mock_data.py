import numpy as np
import pandas as pd

from src import config, download

N_SAMPLES: int = 10
TOXIC_FRACTION: float = 0.2
UNTESTED_FRACTION: float = 0.2


def get_mock_data(data_type: download.RawDataType) -> pd.DataFrame:
    """Generate mock dataframes for testing."""
    mock_data = {
        download.RawDataType.TRAIN: make_mock_train(),
        download.RawDataType.TEST_INPUTS: make_mock_test_inputs(),
        download.RawDataType.TEST_LABELS: make_mock_test_labels(),
    }

    return mock_data[data_type]


def generate_toxic_sample_labels(n_labels: int) -> list[int]:
    """Generate labels for a toxic sample (at least one 1)."""
    # Ensure at least one 1, rest random
    labels = [0] * n_labels
    # Randomly place at least one 1
    n_ones = np.random.randint(1, n_labels + 1)
    positions = np.random.choice(n_labels, size=n_ones, replace=False)
    for pos in positions:
        labels[pos] = 1
    return labels


def make_mock_train() -> pd.DataFrame:
    """Create mock training dataset with sample-level toxicity."""
    n_labels = len(config.LABELS)
    n_toxic = int(N_SAMPLES * TOXIC_FRACTION)
    n_non_toxic = N_SAMPLES - n_toxic

    assert n_toxic + n_non_toxic == N_SAMPLES

    # Generate labels
    all_labels = []

    # Generate toxic samples (at least one 1)
    for _ in range(n_toxic):
        all_labels.append(generate_toxic_sample_labels(n_labels))

    # Generate non-toxic samples (all 0s)
    for _ in range(n_non_toxic):
        all_labels.append([0] * n_labels)

    # Shuffle samples
    np.random.shuffle(all_labels)

    # Convert to dataframe format
    labels_dict = {
        label: [sample[i] for sample in all_labels]
        for i, label in enumerate(config.LABELS)
    }

    return pd.DataFrame(
        {
            "id": [str(i) for i in range(N_SAMPLES)],
            config.INPUT: ["text"] * N_SAMPLES,
            **labels_dict,
        }
    )


def make_mock_test_inputs() -> pd.DataFrame:
    """Create mock test dataset without labels."""
    return pd.DataFrame(
        {
            "id": [str(i) for i in range(N_SAMPLES, 2 * N_SAMPLES)],
            config.INPUT: ["text"] * N_SAMPLES,
        }
    )


def make_mock_test_labels() -> pd.DataFrame:
    """
    Create mock test labels with sample-level toxicity and untested samples.

    - untested_fraction of samples will have all labels as -1
    - toxic_fraction of the remaining samples will have at least one label as 1
    - the rest will have all labels as 0
    """
    n_labels = len(config.LABELS)
    n_toxic = int(N_SAMPLES * TOXIC_FRACTION)
    n_untested = int(N_SAMPLES * UNTESTED_FRACTION)
    n_non_toxic = N_SAMPLES - n_toxic - n_untested

    assert n_toxic + n_untested + n_non_toxic == N_SAMPLES

    all_labels = []

    # Generate untested samples (all -1s)
    for _ in range(n_untested):
        all_labels.append([-1] * n_labels)

    # Generate toxic samples (at least one 1)
    for _ in range(n_toxic):
        all_labels.append(generate_toxic_sample_labels(n_labels))

    # Generate non-toxic samples (all 0s)
    for _ in range(n_non_toxic):
        all_labels.append([0] * n_labels)

    # Shuffle samples
    np.random.shuffle(all_labels)

    # Convert to dataframe format
    labels_dict = {
        label: [sample[i] for sample in all_labels]
        for i, label in enumerate(config.LABELS)
    }

    return pd.DataFrame(
        {
            "id": [str(i) for i in range(N_SAMPLES, 2 * N_SAMPLES)],
            **labels_dict,
        }
    )
