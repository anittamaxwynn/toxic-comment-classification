from src.download import DataDownloader
from pathlib import Path

from src.load_data import DataLoader

RAW_DATA_DIR: Path = Path("./data/raw")
PROCESSED_DATA_DIR: Path = Path("./data/processed")

VALIDATION_SIZE: float = 0.2
BATCH_SIZE: int = 32
EPOCHS: int = 2


def main() -> None:
    downloader = DataDownloader()
    downloader.download_data(download_path=RAW_DATA_DIR, force=False)

    loader = DataLoader(raw_data_dir=RAW_DATA_DIR)
    train, val, test = loader.get_tensorflow_datasets(
        val_size=VALIDATION_SIZE,
        batch_size=BATCH_SIZE,
    )
    loader.save_tensorflow_datasets((train, val, test), dir=PROCESSED_DATA_DIR)

    loaded_train, loaded_val, loaded_test = loader.load_tensorflow_datasets(
        dir=PROCESSED_DATA_DIR
    )

    assert train == loaded_train
    assert val == loaded_val
    assert test == loaded_test


if __name__ == "__main__":
    main()
