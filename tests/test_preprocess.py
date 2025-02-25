import os

import pandas as pd

from src import preprocess


def test_drop_empty_text():
    df = pd.DataFrame({"id": [1, 2, 3], "text": ["a", "b", ""]})
    df = preprocess._drop_empty_text(df, "text")
    expected_df = pd.DataFrame({"id": [1, 2], "text": ["a", "b"]})

    pd.testing.assert_frame_equal(df, expected_df)


def test_drop_non_binary_labels():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "label1": [0, 1, 2],
            "label2": [0, -1, 1],
        }
    )
    df = preprocess._drop_non_binary_labels(df, ["label1", "label2"])
    expected_df = pd.DataFrame({"id": [1], "label1": [0], "label2": [0]})

    pd.testing.assert_frame_equal(df, expected_df)


def test_all_data_exists(tmp_path):
    data_dir = tmp_path / "data"
    param_hash = "12345"

    assert not preprocess._all_datasets_exist(data_dir, param_hash)

    # Create fake (empty) preprocessed data files
    processed_dir = data_dir.joinpath("processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame().to_csv(os.path.join(processed_dir, f"train.csv_{param_hash}"))
    pd.DataFrame().to_csv(os.path.join(processed_dir, f"val.csv_{param_hash}"))
    pd.DataFrame().to_csv(os.path.join(processed_dir, f"test.csv_{param_hash}"))

    assert not preprocess._all_datasets_exist(data_dir, param_hash)
