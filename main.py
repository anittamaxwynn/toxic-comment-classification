from pathlib import Path
import pandas as pd
from src.download import DataDownloader

RAW_DATA_DIR: Path = Path("./data/raw")
PROCESSED_DATA_DIR: Path = Path("./data/processed")

VALIDATION_SIZE: float = 0.2
BATCH_SIZE: int = 32
EPOCHS: int = 2


def main() -> None:
    downloader = DataDownloader()
    downloader.download_data(download_path=RAW_DATA_DIR, force=True)


if __name__ == "__main__":
    main()
