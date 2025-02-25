import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import keras

from . import Config
from .Types import Callback


def setup_logger(
    name: str,
    level: int = logging.DEBUG,
    log_dir: Path = Config.LOG_DIR.joinpath("data"),
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
        if log_dir:
            log_dir.mkdir(exist_ok=True)
            filename = datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"
            file_handler = logging.FileHandler(log_dir.joinpath(filename))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def setup_callbacks(
    model_name: str,
    model_dir: Path = Config.MODEL_DIR,
    include_timestamp: bool = True,
) -> list[Callback]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S") if include_timestamp else ""
    run_name = f"{model_name}_{timestamp}" if timestamp else model_name

    run_dir = model_dir / run_name
    checkpoint_dir = run_dir / "weights"
    tensorboard_dir = run_dir / "logs"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    best_weights_path = checkpoint_dir / "best.weights.h5"

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(best_weights_path),
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1,
    )

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=str(tensorboard_dir),
        histogram_freq=1,
        update_freq="epoch",
    )

    return [checkpoint_callback, tensorboard_callback]
