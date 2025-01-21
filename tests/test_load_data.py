import numpy as np
import pandas as pd
import pytest

import src.config as cfg
from src.load_data import DataProcessor, DatasetConfig

from .mock_data import make_mock_test_labels, make_mock_train


@pytest.fixture
def default_config() -> DatasetConfig:
    return DatasetConfig()


@pytest.fixture
def processor(default_config) -> DataProcessor:
    return DataProcessor(default_config)


class TestDataProcessor:
    def test_init(self):
        config = DatasetConfig()
        processor = DataProcessor(config)

        assert processor.config == config

    def test_remove_untested_samples(self, processor):
        untested_fraction = 0.2
        df = make_mock_test_labels(
            n_samples=100, toxic_fraction=0.2, untested_fraction=untested_fraction
        )

        result = processor._remove_untested_samples(df)

        assert len(result) == len(df) * (1 - untested_fraction)
        assert all(set(result[label].unique()) == {0, 1} for label in cfg.LABELS)

    def test_remove_duplicates(self, processor):
        df = make_mock_train(n_samples=100, toxic_fraction=0.2)
        n_duplicates = 10
        df = pd.concat([df, df[:n_duplicates]], axis=0)

        result = processor._remove_duplicates(df)

        assert len(result) == len(df) - n_duplicates
        assert result.duplicated(subset=["id"]).sum() == 0

    def test_remove_missing_values(self, processor):
        df = make_mock_train(n_samples=100, toxic_fraction=0.2)
        n_missing_values = 10
        df[:n_missing_values] = None
        df = df.sample(frac=1)

        result = processor._remove_missing_values(df)

        assert result.isna().sum().sum() == 0

    def test_remove_empty_inputs(self, processor):
        df = make_mock_train(n_samples=100, toxic_fraction=0.2)
        n_empty = 10
        empty_indices = np.random.choice(df.index, size=n_empty, replace=False)
        df.loc[empty_indices, cfg.INPUT] = ""

        result = processor._remove_empty_inputs(df)

        assert all(len(text) > 0 for text in result[cfg.INPUT])

    def test_train_val_split(self, processor):
        df = make_mock_train(n_samples=100, toxic_fraction=0.2)
        train_df, val_df = processor._train_val_split(df)

        assert len(train_df) + len(val_df) == len(df)
        assert set(train_df["id"].unique()) & set(val_df["id"].unique()) == set()
