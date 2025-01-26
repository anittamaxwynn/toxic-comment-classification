import pytest
import keras
import tensorflow as tf
from src import config
from src.preprocess import DatasetConfig, Preprocessor, DatasetSplit

from . import mock_data

@pytest.fixture
def mock_train_dataset() -> tf.data.Dataset:
    """Mocks a TensorFlow dataset."""
    mock_train_df = mock_data.make_mock_train()
    inputs = mock_train_df[config.INPUT].values
    labels = mock_train_df[config.LABELS].values
    return tf.data.Dataset.from_tensor_slices((inputs, labels))


@pytest.fixture
def mock_config() -> DatasetConfig:
    """Mocks a DatasetConfig object."""
    return DatasetConfig(
        val_size=0.1,
        batch_size=32,
        max_tokens=10000,
        output_sequence_length=100,
    )


@pytest.fixture
def mock_preprocessor(mock_config) -> Preprocessor:
    """Mocks a Preprocessor object."""
    return Preprocessor(config=mock_config)


class TestDatasetConfig:
    def test_generate_dir_string_defaults(self):
        """Test the `generate_dir_string method with default values."""
        config = DatasetConfig()

        expected_dir_string = (
            "val_size=0.2__batch_size=32__max_tokens=10000__output_sequence_length=100"
        )
        assert config.generate_dir_string() == expected_dir_string == config.dir_name

    def test_generate_dir_string_with_custom_values(self):
        """Test the `generate_dir_string method with custom values."""
        config = DatasetConfig(
            val_size=0.1,
            batch_size=16,
            max_tokens=20000,
            output_sequence_length=300,
            vectorize=False,
            shuffle=False,
            optimize=False,
        )

        expected_dir_string = "val_size=0.1__batch_size=16__vectorize=false__shuffle=false__optimize=false"
        assert config.generate_dir_string() == expected_dir_string == config.dir_name


class TestPreprocessor:
    def test_init(self):
        """Test the initialization of the Preprocessor object."""
        # Test initialization with vectorize enabled
        config_with_vectorize = DatasetConfig(
            val_size=0.1,
            batch_size=32,
            max_tokens=10000,
            output_sequence_length=100,
            vectorize=True,
        )

        preproccessor = Preprocessor(config=config_with_vectorize)

        assert preproccessor.config == config_with_vectorize
        assert preproccessor.vectorize_layer is not None
        assert isinstance(preproccessor.vectorize_layer, keras.layers.TextVectorization)
        assert preproccessor.vectorize_layer._max_tokens == 10000
        assert preproccessor.vectorize_layer._output_mode == "int"
        assert preproccessor.vectorize_layer._output_sequence_length == 100
        assert not preproccessor._is_adapted

        # Test initialization with vectorize disabled
        config_without_vectorize = DatasetConfig(
            val_size=0.1,
            batch_size=32,
            max_tokens=10000,
            output_sequence_length=100,
            vectorize=False,
        )

        preproccessor = Preprocessor(config=config_without_vectorize)

        assert preproccessor.config == config_without_vectorize
        assert preproccessor.vectorize_layer is None
        assert not preproccessor._is_adapted

    def test_adapt_vectorize_layer(self, mock_preprocessor, mock_train_dataset):
        """Test the adaptation of the vectorize layer."""
        mock_preprocessor._adapt_vectorize_layer(mock_train_dataset, split=DatasetSplit.TRAIN)
        assert mock_preprocessor.vectorize_layer.built is True
        assert mock_preprocessor._is_adapted is True

        
