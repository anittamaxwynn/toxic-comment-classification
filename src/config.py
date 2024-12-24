import os

COMPETITION: str = "jigsaw-toxic-comment-classification-challenge"

INPUT: list[str] = ["comment_text"]
LABELS: list[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

COMPETITION_DIR = os.path.join(BASE_DIR, "data", COMPETITION)
RAW_DATA_DIR = os.path.join(COMPETITION_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(COMPETITION_DIR, "processed")
INTERIM_DATA_DIR = os.path.join(COMPETITION_DIR, "interim")

MODEL_DIR = os.path.join(BASE_DIR, "models")
