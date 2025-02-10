from . import config, load_data, utils


def main() -> None:
    train_df, val_df, test_df = load_data.load_preprocessed_dfs(val_size=0.02, shuffle=True)

    utils.plot_label_counts(train_df, config.LABELS, normalize=True)
    utils.plot_label_counts(val_df, config.LABELS, normalize=True, is_val=True)
    utils.plot_label_counts(test_df, config.LABELS, normalize=True, is_test=True)

if __name__ == "__main__":
    main()
