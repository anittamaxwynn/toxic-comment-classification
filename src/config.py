import logging

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


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def get_logger(name):
    return logging.getLogger(name)
