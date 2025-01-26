from src import download, load_data, preprocess

# Preproccessing parameters:
VALIDATION_SIZE: float = 0.2
BATCH_SIZE: int = 32
MAX_TOKENS: int = 1000
OUTPUT_SEQUENCE_LENGTH: int = 100


FORCE_DOWNLOAD: bool = False
FORCE_PREPROCESS: bool = True


def main() -> None:
    downloader = download.DataDownloader()
    downloader.download_data(force_download=FORCE_DOWNLOAD)

    data_config = preprocess.DatasetConfig(
        val_size=VALIDATION_SIZE,
        batch_size=BATCH_SIZE,
        max_tokens=MAX_TOKENS,
        output_sequence_length=OUTPUT_SEQUENCE_LENGTH,
    )

    loader = load_data.DataLoader(data_config)

    raw_test_df = loader._load_and_clean_test_df()
    train_ds, val_ds, test_ds = loader.get_datasets(force_preprocess=FORCE_PREPROCESS)

    print(raw_test_df.head())

    # retrieve a batch (of 32 reviews and labels) from the dataset
    text_batch, label_batch = next(iter(test_ds))
    first_review, first_label = text_batch[0], label_batch[0]
    print("Review", first_review)
    print("Label", first_label)
    # print("Vectorized review", vectorize_text(first_review, first_label))


if __name__ == "__main__":
    main()
