import tensorflow as tf

from . import config, datasets

logger = config.get_logger(__name__)


class DataLoader:
    """DataLoader class for batching and interating over datasets."""

    def __init__(
        self,
        split: datasets.Split,
        max_tokens: int,
        sequence_length: int,
        batch_size: int,
        shuffle: bool = True,
        cache: bool = True,
        prefetch: bool = True,
    ):
        logger.info(f"Initializing DataLoader for {split.name} split")
        logger.debug(
            f"Configuration - Batch size: {batch_size}, Max tokens: {max_tokens}, "
            f"Sequence length: {sequence_length}, Shuffle: {shuffle}, "
            f"Cache: {cache}, Prefetch: {prefetch}"
        )

        self.split = split
        self.max_tokens = max_tokens
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cache = cache
        self.prefetch = prefetch

        logger.info("Preparing TensorFlow dataset")
        self.dataset = self._prepare_tf_dataset()
        logger.info(
            f"DataLoader initialization complete. Dataset contains {len(self)} batches"
        )

    def _load_tf_dataset(self) -> tf.data.Dataset:
        """Load the pre-processed dataset."""
        logger.debug(f"Loading preprocessed {self.split.name} dataset")

        tf_dataset = datasets.KaggleDataset(
            split=self.split,
            max_tokens=self.max_tokens,
            sequence_length=self.sequence_length,
            process=True,
        ).data

        if isinstance(tf_dataset, tf.data.Dataset):
            dataset_size = tf_dataset.cardinality().numpy()
            logger.debug(f"Successfully loaded dataset with {dataset_size} samples")
            return tf_dataset
        else:
            logger.error(
                f"Invalid dataset type. Expected tf.data.Dataset but got {type(tf_dataset)}"
            )
            raise TypeError(
                f"Expected 'tf_dataset' to be a tf.data.Dataset, but got {type(tf_dataset)}"
            )

    def _prepare_tf_dataset(self) -> tf.data.Dataset:
        """Prepare the dataset with optimization for training."""
        logger.info("Starting dataset preparation pipeline")

        # Load the dataset
        logger.debug("Loading base dataset")
        tf_dataset = self._load_tf_dataset()
        initial_size = tf_dataset.cardinality().numpy()

        # Cache the dataset if requested
        if self.cache:
            logger.debug("Applying dataset caching")
            tf_dataset = tf_dataset.cache()

        # Shuffle if requested
        if self.shuffle:
            buffer_size = tf_dataset.cardinality()
            logger.debug(f"Shuffling dataset with buffer size {buffer_size}")
            tf_dataset = tf_dataset.shuffle(buffer_size=buffer_size, seed=0)

        # Batch the dataset
        logger.debug(f"Batching dataset with batch size {self.batch_size}")
        tf_dataset = tf_dataset.batch(
            self.batch_size,
            drop_remainder=True,
        )
        batched_size = tf_dataset.cardinality().numpy()

        # Prefetch next batch while current batch is being processed
        if self.prefetch:
            logger.debug("Enabling dataset prefetching with AUTOTUNE")
            tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

        logger.info(
            f"Dataset preparation complete. Samples: {initial_size}, "
            f"Batches: {batched_size}, Batch size: {self.batch_size}"
        )
        return tf_dataset

    def __len__(self) -> int:
        """Return the number of batches in the dataset."""
        return self.dataset.cardinality().numpy()

    def __iter__(self):
        """Create iterator over the dataset."""
        logger.debug("Creating new dataset iterator")
        return iter(self.dataset)

    def __getitem__(self, idx: int):
        """Get a batch from the dataset."""
        if idx < 0 or idx >= len(self):
            logger.error(f"Batch index {idx} out of range [0, {len(self)})")
            raise IndexError(f"Batch index {idx} out of range [0, {len(self)})")

        logger.debug(f"Retrieving batch {idx} from dataset")
        return next(iter(self.dataset.skip(idx).take(1)))


if __name__ == "__main__":
    # Example usage
    logger.info("Starting DataLoader demonstration")

    dataloader = DataLoader(
        split=datasets.Split.TRAIN,
        max_tokens=10000,
        sequence_length=200,
        batch_size=32,
    )

    logger.info("Iterating through first few batches")
    for i, (inputs, labels) in enumerate(dataloader):
        if i >= 3:  # Only show first 3 batches
            break
        logger.info(
            f"Batch {i}: Inputs shape {inputs.shape}, Labels shape {labels.shape}"
        )

    logger.info("DataLoader demonstration complete")
