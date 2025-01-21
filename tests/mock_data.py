import numpy as np
import pandas as pd
import pytest

from src import config as cfg


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


@pytest.fixture
def mock_train_df(n_samples: int = 100, toxic_fraction: float = 0.2) -> pd.DataFrame:
    """Create mock training dataset with sample-level toxicity."""
    n_labels = len(cfg.LABELS)
    n_toxic = int(n_samples * toxic_fraction)
    n_non_toxic = n_samples - n_toxic

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
        for i, label in enumerate(cfg.LABELS)
    }

    return pd.DataFrame(
        {
            "id": range(n_samples),
            cfg.INPUT: ["text"] * n_samples,
            **labels_dict,
        }
    )


@pytest.fixture
def mock_test_df(n_samples: int = 100) -> pd.DataFrame:
    """Create mock test dataset without labels."""
    return pd.DataFrame(
        {
            "id": range(n_samples, 2 * n_samples),
            cfg.INPUT: ["text"] * n_samples,
        }
    )


@pytest.fixture
def mock_test_labels_df(
    n_samples: int = 100, toxic_fraction: float = 0.2, untested_fraction: float = 0.2
) -> pd.DataFrame:
    """
    Create mock test labels with sample-level toxicity and untested samples.

    - untested_fraction of samples will have all labels as -1
    - toxic_fraction of the remaining samples will have at least one label as 1
    - the rest will have all labels as 0
    """
    n_labels = len(cfg.LABELS)
    n_untested = int(n_samples * untested_fraction)
    n_tested = n_samples - n_untested
    n_toxic = int(n_tested * toxic_fraction)
    n_non_toxic = n_tested - n_toxic

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
        for i, label in enumerate(cfg.LABELS)
    }

    return pd.DataFrame(
        {
            "id": range(n_samples, 2 * n_samples),
            **labels_dict,
        }
    )
