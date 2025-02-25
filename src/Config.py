from pathlib import Path

DATASET_AUTHOR: str = "julian3833"
DATASET_NAME: str = "jigsaw-toxic-comment-classification-challenge"
DATASET_HANDLE: str = f"{DATASET_AUTHOR}/{DATASET_NAME}"

FEATURES: list[str] = ["comment_text"]
LABELS: list[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

DATA_DIR: Path = Path(__file__).parent.parent / "data"
TENSORFLOW_DIR: Path = DATA_DIR / "tensorflow"

MODEL_DIR: Path = Path(__file__).parent.parent / "models"
REPORTS_DIR: Path = Path(__file__).parent.parent / "reports"
LOG_DIR: Path = Path(__file__).parent.parent / "logs"


SEED: int = 0
