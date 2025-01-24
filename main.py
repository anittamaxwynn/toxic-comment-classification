from src import download, load_data

VALIDATION_SIZE: float = 0.2
BATCH_SIZE: int = 32
EPOCHS: int = 2

FORCE_DOWNLOAD: bool = False
FORCE_PREPROCESS: bool = False


def main() -> None:
    downloader = download.DataDownloader()
    downloader.download_data(force_download=FORCE_DOWNLOAD)

    loader = load_data.DataLoader()
    train_ds, val_ds, test_ds = loader.get_tensorflow_datasets(
        val_size=VALIDATION_SIZE,
        batch_size=BATCH_SIZE,
        force_preprocess=FORCE_PREPROCESS,
    )

    print("Training size:", len(train_ds))
    print("Validation size:", len(val_ds))
    print("Test size:", len(test_ds))


if __name__ == "__main__":
    main()
