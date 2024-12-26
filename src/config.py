import os

COMPETITION: str = "julian3833/jigsaw-toxic-comment-classification-challenge"

INPUT: str = "comment_text"
LABELS: list[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data/raw/")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data/processed/")
INTERIM_DATA_DIR = os.path.join(BASE_DIR, "data/interim/")

MODEL_DIR = os.path.join(BASE_DIR, "models")
