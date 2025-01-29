from src import datasets, download, load_data

# Preproccessing parameters:
# VALIDATION_SIZE: float = 0.2
BATCH_SIZE: int = 32
MAX_TOKENS: int = 1000
SEQUENCE_LENGTH: int = 100
SHUFFLE: bool = True
CACHE: bool = True
PREFETCH: bool = True

FORCE_DOWNLOAD: bool = False


def main() -> None:
    downloader = download.DataDownloader()
    downloader.download_data(force_download=FORCE_DOWNLOAD)

    TRAIN_SPLIT, TEST_SPLIT = datasets.Split.TRAIN, datasets.Split.TEST

    train_loader = load_data.DataLoader(
        TRAIN_SPLIT, MAX_TOKENS, SEQUENCE_LENGTH, BATCH_SIZE, SHUFFLE, CACHE, PREFETCH
    )
    test_loader = load_data.DataLoader(
        TEST_SPLIT, MAX_TOKENS, SEQUENCE_LENGTH, BATCH_SIZE, SHUFFLE, CACHE, PREFETCH
    )

    train_ds, test_ds = train_loader.dataset, test_loader.dataset
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Test dataset size: {len(test_ds)}")


if __name__ == "__main__":
    main()
