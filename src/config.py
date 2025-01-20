from dataclasses import dataclass

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

@dataclass
class DataConfig:
    """Configuration for data preprocessing."""
    raw_data_path: str
    processed_data_path: str
    interim_data_path: str
    input: str
    labels: list[str]

@dataclass
class ModelConfig:
    """Configuration for Tensorflow models."""

    max_tokens: int
    output_sequence_length: int
    embedding_dim: int
    epochs: int
