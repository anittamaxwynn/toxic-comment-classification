from src import download, load_data, preprocess

# Preproccessing parameters:
VALIDATION_SIZE: float = 0.2
BATCH_SIZE: int = 32
MAX_TOKENS: int = 1000
OUTPUT_SEQUENCE_LENGTH: int = 100

VECTORIZE: bool = False
SHUFFLE: bool = False
OPTIMIZE: bool = True

FORCE_DOWNLOAD: bool = False
FORCE_PREPROCESS: bool = False


def main() -> None:
    downloader = download.DataDownloader()
    downloader.download_data(force_download=FORCE_DOWNLOAD)

    data_config = preprocess.DatasetConfig(
        val_size=VALIDATION_SIZE,
        batch_size=BATCH_SIZE,
        max_tokens=MAX_TOKENS,
        output_sequence_length=OUTPUT_SEQUENCE_LENGTH,
        vectorize=VECTORIZE,
        shuffle=SHUFFLE,
        optimize=OPTIMIZE,

    )

    loader = load_data.DataLoader(data_config)
    train_ds, val_ds, test_ds = loader.get_datasets(force_preprocess=FORCE_PREPROCESS)

    print("Train samples: ", len(train_ds))
    print("Validation samples: ", len(val_ds))
    print("Test samples: ", len(test_ds))


if __name__ == "__main__":
    main()
