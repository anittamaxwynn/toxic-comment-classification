import os

import pandas as pd
import pytest

from src import preprocess


def test_split_df():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})
    train_df, test_df = preprocess.split_df(df, test_size=0.4)

    assert len(train_df) == 2
    assert len(test_df) == 3
    assert train_df.merge(test_df, how="inner").empty


def test_split_df_no_shuffle():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})
    train_df, test_df = preprocess.split_df(df, test_size=0.4, shuffle=False)
    expected_train_df = pd.DataFrame({"a": [1, 2], "b": [6, 7]})
    expected_test_df = pd.DataFrame({"a": [3, 4, 5], "b": [8, 9, 10]})

    pd.testing.assert_frame_equal(train_df, expected_train_df)
    pd.testing.assert_frame_equal(test_df, expected_test_df)


def test_split_invalid_test_size():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})
    with pytest.raises(
        ValueError, match="Test size must be a float value between 0 and 1."
    ):
        _ = preprocess.split_df(df, test_size=1.1)

    with pytest.raises(
        ValueError, match="Test size must be a float value between 0 and 1."
    ):
        _ = preprocess.split_df(df, test_size=-0.1)


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


def test_preprocess_data_exists(tmp_path):
    data_dir = tmp_path / "data"

    assert not preprocess._preprocess_data_exists(data_dir)

    # Create fake (empty) preprocessed data files
    processed_dir = data_dir.joinpath("processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame().to_csv(os.path.join(processed_dir, "train.csv"))
    pd.DataFrame().to_csv(os.path.join(processed_dir, "val.csv"))
    pd.DataFrame().to_csv(os.path.join(processed_dir, "test.csv"))

    assert preprocess._preprocess_data_exists(data_dir)
