from src.download import DatasetDownloader
from pathlib import Path

from src.make_dataset import DATASETS

RAW_DATA_DIR: Path = Path("./data/raw")
PROCESSED_DATA_DIR: Path = Path("./data/processed")

VALIDATION_SIZE: float = 0.2
BATCH_SIZE: int = 32
EPOCHS: int = 2


def main() -> None:
    downloader = DatasetDownloader()
    downloader.download_data(RAW_DATA_DIR, force=False)

    datasets = DATASETS(raw_data_dir=RAW_DATA_DIR)

    train, val, test = datasets.make_tensorflow_datasets(
        val_size=VALIDATION_SIZE,
        batch_size=BATCH_SIZE,
    )

    datasets.save_tensorflow_datasets(PROCESSED_DATA_DIR)


if __name__ == "__main__":
    main()
