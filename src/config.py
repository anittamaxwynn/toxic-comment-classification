from pathlib import Path

DATASET: str = "julian3833/jigsaw-toxic-comment-classification-challenge"

INPUT: str = "comment_text"
LABELS: list[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

DATA_DIR: Path = Path(__file__).parent.parent / "data"
MODEL_DIR: Path = Path(__file__).parent.parent / "models"
REPORTS_DIR: Path = Path(__file__).parent.parent / "reports"

SEED: int = 0
