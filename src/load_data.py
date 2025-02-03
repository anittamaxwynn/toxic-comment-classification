from enum import Enum
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ValidationError

from . import config


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class RawDataSchema(BaseModel):
    id: str
    comment_text: str
    toxic: Literal[0, -1, 1]
    severe_toxic: Literal[0, -1, 1]
    obscene: Literal[0, -1, 1]
    threat: Literal[0, -1, 1]
    insult: Literal[0, -1, 1]
    identity_hate: Literal[0, -1, 1]


def load_and_validate_raw_df(split: Split) -> pd.DataFrame:
    raw_df = _load_raw_df(split)

    valid_records, validation_errors = _validate_df(raw_df)
    if validation_errors:
        raise ValueError(
            f"Validation errors for {split.value} split: {validation_errors}"
        )

    valid_df = pd.DataFrame([record.model_dump() for record in valid_records])

    return valid_df


def _validate_df(df: pd.DataFrame) -> tuple[list, list]:
    valid_records = []
    validation_errors = []

    records = df.to_dict(orient="records")
    for idx, record in enumerate(records):
        try:
            valid_record = RawDataSchema(**record)
            valid_records.append(valid_record)
        except ValidationError as e:
            validation_errors.append({"row": idx, "error": e.errors()})

    return valid_records, validation_errors


def _load_raw_df(split: Split) -> pd.DataFrame:
    if split == Split.TRAIN:
        filepath = f"{config.DATA_DIR}/raw/{split.value}.csv"
        return pd.read_csv(filepath)
    elif split == Split.TEST:
        inputs_filepath = f"{config.DATA_DIR}/raw/{split.value}.csv"
        labels_filepath = f"{config.DATA_DIR}/raw/{split.value}_labels.csv"

        inputs_df = pd.read_csv(inputs_filepath)
        labels_df = pd.read_csv(labels_filepath)

        return pd.merge(inputs_df, labels_df, on="id", validate="1:1")
    else:
        raise ValueError(
            f"Raw {split.value} does not exist. Can only load raw train or test data."
        )
