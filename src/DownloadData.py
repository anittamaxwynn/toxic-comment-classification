import csv
import os
from enum import Enum

import kagglehub
from pydantic import BaseModel, Field

from . import Config


class Files(Enum):
    TRAIN = "train.csv"
    TEST = "test.csv"
    TEST_LABELS = "test_labels.csv"


class TrainModel(BaseModel):
    id: str
    comment_text: str
    toxic: int = Field(ge=0, le=1)
    severe_toxic: int = Field(ge=0, le=1)
    obscene: int = Field(ge=0, le=1)
    threat: int = Field(ge=0, le=1)
    insult: int = Field(ge=0, le=1)
    identity_hate: int = Field(ge=0, le=1)


class TestModel(BaseModel):
    id: str
    comment_text: str


class TestLabelsModel(BaseModel):
    id: str
    toxic: int = Field(ge=-1, le=1)
    severe_toxic: int = Field(ge=-1, le=1)
    obscene: int = Field(ge=-1, le=1)
    threat: int = Field(ge=-1, le=1)
    insult: int = Field(ge=-1, le=1)
    identity_hate: int = Field(ge=-1, le=1)


def download_kaggle_dataset(force_download: bool = False) -> str:
    destination = str(Config.DATA_DIR / "kaggle")
    os.makedirs(destination, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = destination

    download_path = kagglehub.dataset_download(
        handle=Config.DATASET_HANDLE,
        force_download=force_download,
    )

    for file in Files:
        file_path = download_path + f"/{file.value}"
        _ = _validate_file(file, file_path)

    return download_path


def _get_file_schema(file: Files):
    schemas = {
        Files.TRAIN: TrainModel,
        Files.TEST: TestModel,
        Files.TEST_LABELS: TestLabelsModel,
    }

    return schemas[file]


def _validate_file(file: Files, file_path: str):
    schema = _get_file_schema(file)
    with open(file_path) as f:
        reader = csv.DictReader(f)
        samples = [schema.model_validate(row) for row in reader]
    return samples


def main() -> None:
    download_path = download_kaggle_dataset()
    print("Data downloaded to:", download_path)

if __name__ == "__main__":
    main()
