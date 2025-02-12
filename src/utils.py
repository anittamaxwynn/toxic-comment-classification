import os
from pathlib import Path
from typing import Tuple

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from . import config, preprocess

plt.rcParams.update(
    {
        # "figure.figsize": (12, 6),  # Default figure size
        "savefig.dpi": 300,  # High-resolution DPI for saved figures
        "axes.grid": True,  # Enable grid
        "grid.linestyle": "--",  # Dashed grid lines
        "grid.alpha": 0.7,  # Make grid slightly transparent
        "axes.labelsize": 12,  # Axis label font size
        "xtick.labelsize": 10,  # X-axis tick font size
        "ytick.labelsize": 10,  # Y-axis tick font size
    }
)

colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
]


# ----------------------------
# TYPE ALIASES
# ----------------------------


Dataset = tf.data.Dataset


# ----------------------------
# MODEL PLOTTING
# ----------------------------


def plot_metrics(
    history: keras.callbacks.History,
    save_dir: str | Path = config.REPORTS_DIR,
) -> None:
    # Get metrics from history
    metrics = _get_metrics(history)
    num_metrics = len(metrics)

    # Set number of rows and columns based on number of metrics
    num_rows = num_metrics // 2
    num_cols = 2

    # Create plot
    fig, axes = plt.subplots(num_rows, num_cols)

    # Ensure axes is iterable when num_metrics == 1
    # and flatten 2D array to 1D list if num_metrics > 1
    if num_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten().tolist()

    epochs = range(1, len(history.epoch) + 1)
    for ax, metric in zip(axes, metrics):
        name = metric.replace("_", " ").capitalize()

        # Plot training and validation metrics
        ax.plot(epochs, history.history[metric], color="C0", label="Train")
        ax.plot(
            epochs,
            history.history["val_" + metric],
            color="C1",
            linestyle="--",
            label="Validation",
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)

        ax.set_xticks(epochs)

        if metric == "loss":
            ax.set_ylim([0, ax.get_ylim()[1]])
        elif metric == "auc":
            ax.set_ylim([0.6, 1])
        else:
            ax.set_ylim([0, 1])

        ax.legend()

    fig.tight_layout()

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/metrics_" + "_".join(metrics) + ".png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    print(f"Plot saved as: {filename}")


def _get_metrics(history: keras.callbacks.History) -> list[str]:
    """
    Extract all the metric names from a History object.
    """
    all_metrics = history.history.keys()
    return [metric for metric in all_metrics if "val_" not in metric]


# ----------------------------
# DATASET PLOTS
# ----------------------------


def dataset_to_numpy(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    features_list = []
    labels_list = []
    for features_batch, labels_batch in dataset.as_numpy_iterator():  # type: ignore
        features_list.append(features_batch)
        labels_list.append(labels_batch)

    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)
    return features_array, labels_array


def get_category_counts(
    labels_array: np.ndarray, label_names: list[str] = config.LABELS, normalize=False
) -> dict[str, dict[str, int | float]]:
    """Get the number of times each category appears in the labels array."""
    num_samples, num_labels = labels_array.shape
    if len(label_names) != num_labels:
        raise ValueError(
            "The number of labels does not match the number of label names."
        )

    unique_counts = {}
    for col in range(labels_array.shape[1]):
        unique, counts = np.unique(labels_array[:, col], return_counts=True)

        if normalize:
            counts = counts / num_samples

        unique_counts[label_names[col]] = dict(zip(unique, counts))

    return unique_counts


def plot_label_counts(
    labels_array: np.ndarray,
    label_names: list[str] = config.LABELS,
    normalize: bool = False,
    is_val: bool = False,
    is_test: bool = False,
    save_dir: str | Path = config.REPORTS_DIR,
) -> None:
    """Generates and saves a grouped bar chart for label distributions."""
    # Set split name
    if is_val:
        split = "validation"
    elif is_test:
        split = "test"
    else:
        split = "train"

    # Get label counts for each unique label in numpy array
    label_counts = get_category_counts(labels_array, label_names, normalize)

    # Extract all unique categories (e.g., {0, 1, -1})
    categories = sorted(
        set(cat for counts in label_counts.values() for cat in counts.keys())
    )

    # Convert label_counts to a 2D list for plotting
    values = [
        [label_counts[label].get(cat, 0) for label in label_names] for cat in categories
    ]

    # Define bar width based on number of categories
    num_categories = len(categories)
    bar_width = 0.8 / num_categories  # Adjust width so bars fit within each group
    x = np.arange(len(label_names))  # X positions for groups

    # Create the plot
    fig, ax = plt.subplots()

    # Plot bars for each category
    for i, (cat, val) in enumerate(zip(categories, values)):
        bars = ax.bar(
            x + (i - (num_categories - 1) / 2) * bar_width,
            val,
            width=bar_width,
            label=f"Category {cat}",
        )

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Avoid labels on zero-height bars
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.3f}" if normalize else f"{height}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="black",
                )

    # Formatting the plot
    ax.set_xticks(x)  # Center x-axis labels
    ax.set_xticklabels(label_names, rotation=45)
    ax.set_xlabel("Labels")
    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.set_title(f"Distribution of Values in Each Label for {split} data")
    ax.legend(title="Category")

    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/{split}_label_distribution.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved as: {filename}")


# def decode_text(
#     vectorize_layer: keras.layers.TextVectorization, vectorized_text: tf.Tensor | list
# ) -> str:
#     """Decodes a vectorized text tensor or list of strings to a string."""
#     if not vectorize_layer.built:
#         raise ValueError("Layer has not been built yet.")
#
#     # Convert the text tensor to a list of strings
#     if isinstance(vectorized_text, tf.Tensor):
#         vectorized_text = vectorized_text.numpy().tolist()
#
#     vocab = vectorize_layer.get_vocabulary()
#     decoded_text = " ".join([vocab[idx] for idx in vectorized_text if idx != 0])
#     return decoded_text


if __name__ == "__main__":
    datasets, param_hash = preprocess.make_datasets(
        val_size=0.2,
        vocab_size=10000,
        max_length=100,
        batch_size=32,
    )
    train_ds, val_ds, test_ds = datasets.values()

    train_features, train_labels = dataset_to_numpy(train_ds)
    val_features, val_labels = dataset_to_numpy(val_ds)
    test_features, test_labels = dataset_to_numpy(test_ds)

    plot_label_counts(train_labels, config.LABELS, normalize=True)
    plot_label_counts(val_labels, config.LABELS, normalize=True, is_val=True)
    plot_label_counts(test_labels, config.LABELS, normalize=True, is_test=True)
