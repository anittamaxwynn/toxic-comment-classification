import logging
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
ID: str = "id"

TYPES: dict[str, str] = {
    ID: "object",
    INPUT: "object",
    **{label: "int64" for label in LABELS},
}

DATA_DIR: Path = Path(__file__).parent.parent / "data"
MODEL_DIR: Path = Path(__file__).parent.parent / "models"


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    return logger
