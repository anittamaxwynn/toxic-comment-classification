import os
from typing import List

DATASET_HANDLER: str = "julian3833/jigsaw-toxic-comment-classification-challenge"

INPUT: str = "comment_text"
LABELS: List[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RAW_DATA_DIR: str = os.path.join(BASE_DIR, "data/raw")
PROCESSED_DATA_DIR: str = os.path.join(BASE_DIR, "data/processed")
INTERIM_DATA_DIR: str = os.path.join(BASE_DIR, "data/interim")

MODEL_DIR: str = os.path.join(BASE_DIR, "models")

BATCH_SIZE: int = 32
EPOCHS: int = 2
RANDOM_STATE: int = 0
