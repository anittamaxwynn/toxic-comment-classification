import logging
from .preprocess import DataPreprocessor
from .load_data import DataLoader
import numpy as np
import tensorflow as tf
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DATASETS:

    def __init__(self, raw_data_dir: Path | str = None):
        self.raw_data_dir = Path(raw_data_dir)
        self.loader = DataLoader()
        self.preprocessor = DataPreprocessor()

        self.train = None
        self.val = None
        self.test = None

        self.is_tf_ready = False

    def load_raw_data(self) -> tuple[np.ndarray, np.ndarray]:
        if (self.train is not None and self.test is not None) and (self.is_tf_ready):
            return self.train, self.test
        else:
            self.train, self.test = self.loader.load_data(self.raw_data_dir)
            return self.train, self.test

    def make_tensorflow_datasets(
        self,
        val_size: float,
        batch_size: int,
        shuffle: bool = True,
        cache: bool = True,
        prefetch: bool = True,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        if self.is_tf_ready:
            return self.train, self.val, self.test
        else:
            train, test = self.loader.load_data(self.raw_data_dir)

            train = self.preprocessor.preprocess_data(train)
            test = self.preprocessor.preprocess_data(test)

            train, val = self.preprocessor.split_data(
                train, split_size=val_size, shuffle=shuffle
            )

            self.train = self._ceate_tf_dataset(
                data=train,
                is_training=True,
                batch_size=batch_size,
                shuffle=shuffle,
                cache=cache,
                prefetch=prefetch,
            )

            self.val = self._ceate_tf_dataset(
                data=val,
                is_training=False,
                batch_size=batch_size,
                shuffle=shuffle,
                cache=cache,
                prefetch=prefetch,
            )

            self.test = self._ceate_tf_dataset(
                data=test,
                is_training=False,
                batch_size=batch_size,
                shuffle=shuffle,
                cache=cache,
                prefetch=prefetch,
            )

            self.is_raw = False
            self.is_preprocessed = True
            self.is_tf_ready = True

            return self.train, self.val, self.test

    def save_tensorflow_datasets(
        self, save_dir: Path | str, overwrite: bool = True
    ) -> None:
        if self.is_tf_ready:
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            elif overwrite:
                shutil.rmtree(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)

            save_dir_str = str(save_dir)

            self.train.save(save_dir_str + "/train")
            self.val.save(save_dir_str + "/val")
            self.test.save(save_dir_str + "/test")

            logger.info(f"Saved TensorFlow datasets to {save_dir}")
        else:
            raise ValueError(
                "TensorFlow datasets are not ready yet. Please run `make_tensorflow_datasets` first."
            )

    def _ceate_tf_dataset(
        self,
        data: tuple[np.ndarray, np.ndarray],
        is_training: bool,
        batch_size: int,
        shuffle: bool = True,
        cache: bool = True,
        prefetch: bool = True,
    ) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(data)

        if is_training and shuffle:
            ds = ds.shuffle(len(data[0]), seed=0)

        ds = ds.batch(batch_size)

        if prefetch:
            ds = ds.prefetch(tf.data.AUTOTUNE)

        if cache:
            ds = ds.cache()

        logger.info(
            f"Created TensorFlow dataset with {len(ds)} batches of size {batch_size}"
        )

        return ds
