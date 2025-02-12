import logging
import sys
from pathlib import Path
from typing import Optional

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
LOG_DIR: Path = Path(__file__).parent.parent / "logs"

SEED: int = 0


def setup_logger(
    name: str,
    level: int = logging.DEBUG,
    log_file: Optional[Path] = LOG_DIR / "main.log",
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up and configure a logger instance.
    """
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding handlers multiple times
    if not logger.handlers:
        formatter = logging.Formatter(format_string)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (if log_file provided)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
