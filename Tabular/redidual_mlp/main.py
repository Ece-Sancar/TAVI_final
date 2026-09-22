#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone residual tabular MLP using the exact tabular feature extractor from the
combined CT + tabular model.

For fold model i:
    train      = the other four predefined folds
    validation = fold i
    test       = fixed independent test.xlsx

Tabular preprocessing is identical to the combined model:
    training-fold median imputation
    training-fold StandardScaler
    one binary missingness mask per original clinical feature

Feature importance uses paired permutation:
    a clinical feature's standardized value and its missingness mask are
    permuted together.

The importance plots intentionally match the TabPFN implementation:
    same metadata file
    same feature groups
    same grayscale colors
    same pretty names
    same percentage normalization
    same top-15 figures
    same repeat/fold variability figures
    same group-importance figures
"""

from __future__ import annotations

import gc
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

from dataset import (
    TabularDataset,
    compute_binary_class_weights,
    fit_preprocessor,
    get_fold_path,
    load_excel_rows,
    load_preprocessor,
    make_dataloader,
    save_preprocessor,
    transform_tabular,
)
from model import ResidualTabularMLP


# =============================================================================
# PLOT SETTINGS -- SAME AS TABPFN
# =============================================================================

BASE_FONTSIZE = 18
TITLE_FONTSIZE = 24
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 16

plt.rcParams.update(
    {
        "font.size": BASE_FONTSIZE,
        "axes.titlesize": TITLE_FONTSIZE,
        "axes.labelsize": LABEL_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "legend.fontsize": BASE_FONTSIZE,
    }
)

VALID_DATASETS = {"tum", "lmu", "merged"}
VALID_TRAIN_VALUES = VALID_DATASETS | {"pre"}
VALID_DATA_PERCENTAGES = {2, 5, 10, 20, 50, 100}

VALID_GROUPS = [
    "Radiomics",
    "Patient",
    "Procedural",
    "Electrocardiographic",
]

GROUP_COLORS = {
    "Radiomics": "#c9c9c9",
    "Patient": "#808080",
    "Procedural": "#555555",
    "Electrocardiographic": "#1a1a1a",
}


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(
        roc_auc_score(
            y_true,
            probabilities,
        )
    )


def safe_log_loss(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )
    probabilities = np.clip(
        probabilities,
        1e-7,
        1.0 - 1e-7,
    )

    return float(
        log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
        )
    )


def resolve_configuration(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    dataset_root = Path(
        config["dataset_root"]
    )

    train_choice = str(
        config["train_dataset"]
    ).strip().lower()

    test_choice = str(
        config["test_dataset"]
    ).strip().lower()

    data_percentage = int(
        config.get(
            "data_percentage",
            100,
        )
    )

    if train_choice not in VALID_TRAIN_VALUES:
        raise ValueError(
            f"train_dataset must be one of {sorted(VALID_TRAIN_VALUES)}"
        )

    if test_choice not in VALID_DATASETS:
        raise ValueError(
            f"test_dataset must be one of {sorted(VALID_DATASETS)}"
        )

    if data_percentage not in VALID_DATA_PERCENTAGES:
        raise ValueError(
            "data_percentage must be one of "
            f"{sorted(VALID_DATA_PERCENTAGES)}"
        )

    if train_choice == "pre":
        pretrained = config.get(
            "pretrained",
            {},
        )

        checkpoint_dir = Path(
            pretrained["checkpoint_dir"]
        )

        training_dataset = str(
            pretrained["train_dataset"]
        ).strip().lower()

        if training_dataset not in VALID_DATASETS:
            raise ValueError(
                "pretrained.train_dataset must be tum, lmu, or merged."
            )

        mode = "pre"

    else:
        checkpoint_dir = None
        training_dataset = train_choice
        mode = "train"

    if data_percentage == 100:
        training_folder = (
            dataset_root
            / training_dataset
        )
    else:
        training_folder = (
            dataset_root
            / "data_size"
            / f"{data_percentage}_percent"
            / training_dataset
        )

    # Test set NEVER changes with data percentage.
    test_file = (
        dataset_root
        / test_choice
        / "test.xlsx"
    )

    if not training_folder.is_dir():
        raise NotADirectoryError(
            f"Training folder does not exist: {training_folder}"
        )

    if not test_file.exists():
        raise FileNotFoundError(
            f"Independent test file does not exist: {test_file}"
        )

    return {
        "mode": mode,
        "training_dataset": training_dataset,
        "test_dataset": test_choice,
        "data_percentage": data_percentage,
        "training_folder": training_folder,
        "test_file": test_file,
        "checkpoint_dir": checkpoint_dir,
    }


# =============================================================================
# METRICS / PREDICTION
# =============================================================================

def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    predictions = (
        probabilities >= threshold
    ).astype(int)

    confusion = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = confusion.ravel()

    metrics = {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "auc_roc": safe_auc(
            y_true,
            probabilities,
        ),
        "log_loss": safe_log_loss(
            y_true,
            probabilities,
        ),
        "no_event_accuracy": (
            float(tn / (tn + fp))
            if (tn + fp) > 0
            else float("nan")
        ),
        "pacer_accuracy": (
            float(tp / (tp + fn))
            if (tp + fn) > 0
            else float("nan")
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    return metrics, predictions, confusion


@torch.no_grad()
def predict_probabilities(
    model: ResidualTabularMLP,
    X: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    actual_device = torch.device(
        "cuda"
        if device.startswith("cuda")
        and torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(actual_device)
    model.eval()

    X_tensor = torch.as_tensor(
        X,
        dtype=torch.float32,
    )

    probability_parts: List[np.ndarray] = []

    for start in range(
        0,
        len(X_tensor),
        batch_size,
    ):
        batch = X_tensor[
            start:start + batch_size
        ].to(actual_device)

        logits = model(batch)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        probability_parts.append(
            probabilities
            .detach()
            .cpu()
            .numpy()
        )

    return np.concatenate(
        probability_parts,
        axis=0,
    )


# =============================================================================
# STANDARD PLOTS
# =============================================================================

def save_confusion_matrix(
    confusion: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    axis.imshow(
        confusion,
        cmap="Greys",
    )

    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(confusion[row, column]),
                ha="center",
                va="center",
                fontsize=18,
            )

    axis.set_xticks(
        [0, 1],
        ["No event", "Pacer"],
    )
    axis.set_yticks(
        [0, 1],
        ["No event", "Pacer"],
    )

    axis.set_xlabel(
        "Predicted label"
    )
    axis.set_ylabel(
        "True label"
    )
    axis.set_title(
        title
    )

    figure.tight_layout()
    figure.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_calibration_plot(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    save_path: Path,
    title: str,
    n_bins: int,
) -> None:
    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        probabilities,
        n_bins=n_bins,
        strategy="uniform",
    )

    figure, axis = plt.subplots(
        figsize=(7, 7)
    )

    axis.plot(
        mean_predicted,
        fraction_positive,
        "o-",
        label="Model",
    )

    axis.plot(
        [0, 1],
        [0, 1],
        "--",
        color="gray",
        label="Perfect calibration",
    )

    axis.set_xlabel(
        "Mean predicted probability"
    )
    axis.set_ylabel(
        "Fraction of positives"
    )
    axis.set_title(
        title
    )
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_training_curves(
    trainer: pl.Trainer,
    fold_output: Path,
) -> None:
    """
    Create train/validation loss and accuracy plots after one fold finishes.
    """
    logger = trainer.logger

    if logger is None or not hasattr(
        logger,
        "log_dir",
    ):
        print(
            "WARNING: No compatible CSV logger; "
            "training curves not generated."
        )
        return

    metrics_path = (
        Path(logger.log_dir)
        / "metrics.csv"
    )

    if not metrics_path.exists():
        print(
            f"WARNING: {metrics_path} not found; "
            "training curves not generated."
        )
        return

    metrics = pd.read_csv(
        metrics_path
    )

    required_columns = {
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
    }

    missing_columns = (
        required_columns
        - set(metrics.columns)
    )

    if missing_columns:
        print(
            "WARNING: Missing logged metrics: "
            f"{sorted(missing_columns)}"
        )
        return

    def last_non_missing(
        series: pd.Series,
    ):
        values = series.dropna()

        if values.empty:
            return np.nan

        return values.iloc[-1]

    history = (
        metrics.loc[
            metrics["epoch"].notna()
        ]
        .groupby(
            "epoch",
            as_index=False,
        )
        .agg(
            {
                "train_loss": (
                    last_non_missing
                ),
                "train_accuracy": (
                    last_non_missing
                ),
                "val_loss": (
                    last_non_missing
                ),
                "val_accuracy": (
                    last_non_missing
                ),
            }
        )
        .sort_values(
            "epoch"
        )
        .reset_index(
            drop=True
        )
    )

    history = history.dropna(
        subset=[
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
        ]
    ).copy()

    if history.empty:
        print(
            "WARNING: No complete epochs found "
            "for training curves."
        )
        return

    history["epoch"] = (
        history["epoch"]
        .astype(int)
        + 1
    )

    history.to_csv(
        fold_output
        / "training_history.csv",
        index=False,
    )

    # Loss
    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    axis.plot(
        history["epoch"],
        history["train_loss"],
        linewidth=2.5,
        label="Training",
    )

    axis.plot(
        history["epoch"],
        history["val_loss"],
        linewidth=2.5,
        label="Validation",
    )

    axis.set_xlabel(
        "Epoch"
    )
    axis.set_ylabel(
        "Cross-entropy loss"
    )
    axis.set_title(
        "Training and Validation Loss",
        pad=18,
    )
    axis.legend(
        frameon=False
    )
    axis.spines[
        "top"
    ].set_visible(False)
    axis.spines[
        "right"
    ].set_visible(False)
    axis.grid(
        axis="y",
        linestyle=":",
        linewidth=1.0,
        alpha=0.5,
    )

    figure.tight_layout()
    figure.savefig(
        fold_output
        / "training_validation_loss.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    # Accuracy
    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    axis.plot(
        history["epoch"],
        history["train_accuracy"]
        * 100.0,
        linewidth=2.5,
        label="Training",
    )

    axis.plot(
        history["epoch"],
        history["val_accuracy"]
        * 100.0,
        linewidth=2.5,
        label="Validation",
    )

    axis.set_xlabel(
        "Epoch"
    )
    axis.set_ylabel(
        "Accuracy (%)"
    )
    axis.set_ylim(
        0,
        100,
    )
    axis.set_title(
        "Training and Validation Accuracy",
        pad=18,
    )
    axis.legend(
        frameon=False
    )
    axis.spines[
        "top"
    ].set_visible(False)
    axis.spines[
        "right"
    ].set_visible(False)
    axis.grid(
        axis="y",
        linestyle=":",
        linewidth=1.0,
        alpha=0.5,
    )

    figure.tight_layout()
    figure.savefig(
        fold_output
        / "training_validation_accuracy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


# =============================================================================
# FEATURE METADATA -- SAME AS TABPFN
# =============================================================================

def load_feature_metadata(
    metadata_path: Path,
    feature_columns: Sequence[str],
) -> Tuple[
    Dict[str, str],
    Dict[str, str],
]:
    metadata = pd.read_excel(
        metadata_path
    )

    required_columns = {
        "Feature",
        "Group",
        "Mapping Name",
    }

    missing_columns = (
        required_columns
        - set(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            "Feature metadata is missing columns: "
            f"{sorted(missing_columns)}"
        )

    metadata["Feature"] = (
        metadata["Feature"]
        .astype(str)
    )

    metadata["Group"] = (
        metadata["Group"]
        .astype(str)
    )

    metadata["Mapping Name"] = (
        metadata["Mapping Name"]
        .astype(str)
    )

    invalid_groups = (
        set(metadata["Group"])
        - set(VALID_GROUPS)
    )

    if invalid_groups:
        raise ValueError(
            "Invalid groups in feature metadata: "
            f"{sorted(invalid_groups)}"
        )

    metadata_features = set(
        metadata["Feature"]
    )

    missing_features = (
        set(feature_columns)
        - metadata_features
    )

    if missing_features:
        raise ValueError(
            "The following model features are missing "
            "from the feature metadata file:\n"
            f"{sorted(missing_features)}"
        )

    feature_to_group = dict(
        zip(
            metadata["Feature"],
            metadata["Group"],
        )
    )

    feature_to_pretty_name = dict(
        zip(
            metadata["Feature"],
            metadata["Mapping Name"],
        )
    )

    return (
        feature_to_group,
        feature_to_pretty_name,
    )


# =============================================================================
# PAIRED VALUE + MASK PERMUTATION IMPORTANCE
# =============================================================================

def calculate_paired_permutation_importance(
    model: ResidualTabularMLP,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_columns: Sequence[str],
    number_of_repeats: int,
    seed: int,
    batch_size: int,
    device: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    float,
]:
    """
    Importance = increase in test log loss after permutation.

    One ORIGINAL clinical feature corresponds to:
        standardized value
        missingness mask

    These two inputs are permuted together using the same row permutation.
    """
    feature_columns = list(
        feature_columns
    )

    number_of_features = len(
        feature_columns
    )

    if X_test.shape[1] != (
        2 * number_of_features
    ):
        raise ValueError(
            "Expected [values | masks] input width of "
            f"{2 * number_of_features}, "
            f"got {X_test.shape[1]}."
        )

    baseline_probabilities = (
        predict_probabilities(
            model=model,
            X=X_test,
            batch_size=batch_size,
            device=device,
        )
    )

    baseline_loss = safe_log_loss(
        y_test,
        baseline_probabilities,
    )

    importance_deltas = np.zeros(
        number_of_features,
        dtype=np.float64,
    )

    repeat_standard_deviations = (
        np.zeros(
            number_of_features,
            dtype=np.float64,
        )
    )

    for feature_index in range(
        number_of_features
    ):
        repeat_losses: List[
            float
        ] = []

        for repeat_number in range(
            number_of_repeats
        ):
            random_generator = (
                np.random.RandomState(
                    seed
                    + feature_index * 1009
                    + repeat_number
                )
            )

            permutation_indices = (
                random_generator.permutation(
                    len(X_test)
                )
            )

            permuted_X = X_test.copy()

            # Clinical value
            permuted_X[
                :,
                feature_index,
            ] = X_test[
                permutation_indices,
                feature_index,
            ]

            # Matching missingness mask
            mask_index = (
                feature_index
                + number_of_features
            )

            permuted_X[
                :,
                mask_index,
            ] = X_test[
                permutation_indices,
                mask_index,
            ]

            probabilities = (
                predict_probabilities(
                    model=model,
                    X=permuted_X,
                    batch_size=batch_size,
                    device=device,
                )
            )

            permuted_loss = safe_log_loss(
                y_test,
                probabilities,
            )

            repeat_losses.append(
                permuted_loss
                - baseline_loss
            )

        importance_deltas[
            feature_index
        ] = float(
            np.mean(
                repeat_losses
            )
        )

        if number_of_repeats > 1:
            repeat_standard_deviations[
                feature_index
            ] = float(
                np.std(
                    repeat_losses,
                    ddof=1,
                )
            )

    return (
        importance_deltas,
        repeat_standard_deviations,
        baseline_loss,
    )


# =============================================================================
# IMPORTANCE PLOTS -- SAME COSMETICS AS TABPFN
# =============================================================================

def plot_fold_feature_importance(
    importance_table: pd.DataFrame,
    save_path: Path,
    title: str,
) -> None:
    top_features = (
        importance_table
        .nlargest(
            15,
            "importance_percent",
        )
        .sort_values(
            "importance_percent",
            ascending=True,
        )
    )

    colors = [
        GROUP_COLORS[group]
        for group
        in top_features["group"]
    ]

    plt.figure(
        figsize=(14, 11)
    )

    plt.barh(
        top_features[
            "feature_pretty"
        ],
        top_features[
            "importance_percent"
        ],
        color=colors,
        edgecolor="black",
        linewidth=1.0,
    )

    plt.xlabel(
        "Importance (%)"
    )
    plt.title(
        title
    )
    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=300,
    )
    plt.close()


def plot_aggregate_feature_importance(
    feature_table: pd.DataFrame,
    error_column: str,
    save_path: Path,
    title: str,
) -> None:
    top_features = (
        feature_table
        .nlargest(
            15,
            "importance_mean",
        )
        .sort_values(
            "importance_mean",
            ascending=True,
        )
    )

    y_positions = np.arange(
        len(top_features)
    )

    colors = [
        GROUP_COLORS[group]
        for group
        in top_features["group"]
    ]

    plt.figure(
        figsize=(15, 12)
    )

    plt.barh(
        y_positions,
        top_features[
            "importance_mean"
        ],
        xerr=top_features[
            error_column
        ],
        color=colors,
        edgecolor="black",
        linewidth=1.0,
        capsize=6,
    )

    plt.yticks(
        y_positions,
        top_features[
            "feature_pretty"
        ],
    )

    plt.xlabel(
        "Importance (%)"
    )
    plt.title(
        title
    )
    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=300,
    )
    plt.close()


def plot_group_importance(
    group_table: pd.DataFrame,
    mean_column: str,
    standard_deviation_column: str,
    save_path: Path,
    title: str,
) -> None:
    sorted_table = (
        group_table.sort_values(
            mean_column,
            ascending=True,
        )
    )

    y_positions = np.arange(
        len(sorted_table)
    )

    colors = [
        GROUP_COLORS[group]
        for group
        in sorted_table["group"]
    ]

    plt.figure(
        figsize=(13, 9)
    )

    plt.barh(
        y_positions,
        sorted_table[
            mean_column
        ],
        xerr=sorted_table[
            standard_deviation_column
        ],
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        capsize=7,
    )

    plt.yticks(
        y_positions,
        sorted_table[
            "group"
        ],
    )

    plt.xlabel(
        "Importance (%)"
    )
    plt.title(
        title
    )
    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=300,
    )
    plt.close()


def aggregate_feature_importance(
    raw_importances: List[np.ndarray],
    repeat_standard_deviations: List[np.ndarray],
    feature_columns: Sequence[str],
    feature_to_group: Dict[str, str],
    feature_to_pretty_name: Dict[str, str],
    output_folder: Path,
) -> None:
    """
    Intentionally follows the TabPFN aggregation/normalization code.
    """
    raw_matrix = np.vstack(
        raw_importances
    )

    repeat_std_matrix = np.vstack(
        repeat_standard_deviations
    )

    mean_raw_importance = (
        raw_matrix.mean(
            axis=0
        )
    )

    if raw_matrix.shape[0] > 1:
        fold_raw_std = raw_matrix.std(
            axis=0,
            ddof=1,
        )
    else:
        fold_raw_std = np.zeros(
            raw_matrix.shape[1],
            dtype=np.float64,
        )

    normalization_denominator = float(
        np.sum(
            np.abs(
                mean_raw_importance
            )
        )
    ) or 1.0

    importance_mean = (
        100.0
        * np.abs(
            mean_raw_importance
        )
        / normalization_denominator
    )

    fold_importance_std = (
        100.0
        * np.abs(
            fold_raw_std
        )
        / normalization_denominator
    )

    repeat_importance_std = (
        100.0
        * np.abs(
            repeat_std_matrix.mean(
                axis=0
            )
        )
        / normalization_denominator
    )

    feature_table = pd.DataFrame(
        {
            "feature": (
                feature_columns
            ),
            "feature_pretty": [
                feature_to_pretty_name[
                    feature
                ]
                for feature
                in feature_columns
            ],
            "group": [
                feature_to_group[
                    feature
                ]
                for feature
                in feature_columns
            ],
            "importance_mean": (
                importance_mean
            ),
            "repeat_std": (
                repeat_importance_std
            ),
            "fold_std": (
                fold_importance_std
            ),
            "raw_mean": (
                mean_raw_importance
            ),
            "raw_fold_std": (
                fold_raw_std
            ),
        }
    ).sort_values(
        "importance_mean",
        ascending=False,
    )

    feature_table.to_csv(
        output_folder
        / "feature_importance_average.csv",
        index=False,
    )

    plot_aggregate_feature_importance(
        feature_table=(
            feature_table
        ),
        error_column=(
            "repeat_std"
        ),
        save_path=(
            output_folder
            / "features_top15_repeat_std.png"
        ),
        title=(
            "Feature Importances — "
            "Permutation Repeat Variability"
        ),
    )

    plot_aggregate_feature_importance(
        feature_table=(
            feature_table
        ),
        error_column=(
            "fold_std"
        ),
        save_path=(
            output_folder
            / "features_top15_fold_std.png"
        ),
        title=(
            "Feature Importances — "
            "Fold Model Variability"
        ),
    )

    # Same group summaries as TabPFN.
    group_total_rows: List[
        List[float]
    ] = []

    group_average_rows: List[
        List[float]
    ] = []

    for fold_importance in raw_matrix:
        absolute_importance = np.abs(
            fold_importance
        )

        fold_denominator = float(
            absolute_importance.sum()
        ) or 1.0

        fold_percentages = (
            100.0
            * absolute_importance
            / fold_denominator
        )

        group_totals: List[
            float
        ] = []

        group_averages: List[
            float
        ] = []

        for group in VALID_GROUPS:
            group_indices = [
                index
                for index, feature
                in enumerate(
                    feature_columns
                )
                if (
                    feature_to_group[
                        feature
                    ]
                    == group
                )
            ]

            if group_indices:
                group_totals.append(
                    float(
                        fold_percentages[
                            group_indices
                        ].sum()
                    )
                )

                group_averages.append(
                    float(
                        fold_percentages[
                            group_indices
                        ].mean()
                    )
                )
            else:
                group_totals.append(
                    0.0
                )
                group_averages.append(
                    0.0
                )

        group_total_rows.append(
            group_totals
        )

        group_average_rows.append(
            group_averages
        )

    group_total_matrix = np.asarray(
        group_total_rows,
        dtype=np.float64,
    )

    group_average_matrix = np.asarray(
        group_average_rows,
        dtype=np.float64,
    )

    standard_deviation_ddof = (
        1
        if raw_matrix.shape[0] > 1
        else 0
    )

    group_table = pd.DataFrame(
        {
            "group": VALID_GROUPS,
            "total_importance_mean": (
                group_total_matrix.mean(
                    axis=0
                )
            ),
            "total_importance_std": (
                group_total_matrix.std(
                    axis=0,
                    ddof=(
                        standard_deviation_ddof
                    ),
                )
            ),
            "average_per_feature_mean": (
                group_average_matrix.mean(
                    axis=0
                )
            ),
            "average_per_feature_std": (
                group_average_matrix.std(
                    axis=0,
                    ddof=(
                        standard_deviation_ddof
                    ),
                )
            ),
        }
    )

    group_table.to_csv(
        output_folder
        / "group_importance_average.csv",
        index=False,
    )

    plot_group_importance(
        group_table=(
            group_table
        ),
        mean_column=(
            "total_importance_mean"
        ),
        standard_deviation_column=(
            "total_importance_std"
        ),
        save_path=(
            output_folder
            / "groups_total_importance.png"
        ),
        title=(
            "Group Total Importance"
        ),
    )

    plot_group_importance(
        group_table=(
            group_table
        ),
        mean_column=(
            "average_per_feature_mean"
        ),
        standard_deviation_column=(
            "average_per_feature_std"
        ),
        save_path=(
            output_folder
            / "groups_average_per_feature.png"
        ),
        title=(
            "Group Average Importance per Feature"
        ),
    )


# =============================================================================
# TRAINER / SPLIT HELPERS
# =============================================================================

def validate_no_overlap(
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    test_ids: np.ndarray,
) -> None:
    train_set = set(
        train_ids.tolist()
    )

    validation_set = set(
        validation_ids.tolist()
    )

    test_set = set(
        test_ids.tolist()
    )

    if train_set & validation_set:
        raise RuntimeError(
            "Training and validation IDs overlap."
        )

    if train_set & test_set:
        raise RuntimeError(
            "Training and test IDs overlap."
        )

    if validation_set & test_set:
        raise RuntimeError(
            "Validation and test IDs overlap."
        )


def make_trainer(
    fold_output: Path,
    config: Dict[str, Any],
    device: str,
) -> Tuple[
    pl.Trainer,
    ModelCheckpoint,
]:
    training = config[
        "training"
    ]

    checkpoint_callback = (
        ModelCheckpoint(
            dirpath=(
                fold_output
            ),
            filename=(
                "best_model"
            ),
            monitor=(
                "val_loss"
            ),
            mode="min",
            save_top_k=1,
            save_last=False,
        )
    )

    early_stopping = (
        EarlyStopping(
            monitor=(
                "val_loss"
            ),
            mode="min",
            patience=int(
                training[
                    "patience"
                ]
            ),
            min_delta=0.0,
        )
    )

    accelerator = (
        "gpu"
        if (
            device.startswith(
                "cuda"
            )
            and torch.cuda.is_available()
        )
        else "cpu"
    )

    logger = (
        pl.loggers.CSVLogger(
            save_dir=str(
                fold_output
            ),
            name=(
                "training_log"
            ),
        )
    )

    trainer = pl.Trainer(
        max_epochs=int(
            training[
                "epochs"
            ]
        ),
        accelerator=(
            accelerator
        ),
        devices=1,
        deterministic=True,
        callbacks=[
            checkpoint_callback,
            early_stopping,
        ],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )

    return (
        trainer,
        checkpoint_callback,
    )


def checkpoint_from_folder(
    folder: Path,
) -> Path:
    candidates = sorted(
        folder.glob(
            "best_model*.ckpt"
        )
    )

    if not candidates:
        raise FileNotFoundError(
            f"No best_model*.ckpt found in {folder}"
        )

    return candidates[0]


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    with open(
        "config.json",
        "r",
        encoding="utf-8",
    ) as handle:
        config = json.load(
            handle
        )

    resolved = (
        resolve_configuration(
            config
        )
    )

    number_of_folds = int(
        config[
            "n_folds"
        ]
    )

    if number_of_folds != 5:
        raise ValueError(
            "n_folds must be 5."
        )

    seed = int(
        config[
            "random_seed"
        ]
    )

    device = str(
        config[
            "device"
        ]
    ).strip().lower()

    id_column = config[
        "id_col"
    ]

    label_column = config[
        "label_col"
    ]

    model_config = config[
        "model"
    ]

    training_config = config[
        "training"
    ]

    evaluation_config = (
        config.get(
            "evaluation",
            {},
        )
    )

    importance_config = (
        config.get(
            "feature_importance",
            {},
        )
    )

    importance_enabled = bool(
        importance_config.get(
            "enabled",
            True,
        )
    )

    threshold = float(
        evaluation_config.get(
            "threshold",
            0.5,
        )
    )

    calibration_bins = int(
        evaluation_config.get(
            "calibration_bins",
            10,
        )
    )

    output_root = (
        Path(
            config[
                "output_dir"
            ]
        )
        / (
            f"{resolved['data_percentage']}"
            "_percent"
        )
    )

    ensure_dir(
        output_root
    )

    feature_importance_folder = (
        output_root
        / "feature_importance"
    )

    if importance_enabled:
        ensure_dir(
            feature_importance_folder
        )

    set_global_seed(
        seed
    )

    fold_paths = [
        get_fold_path(
            resolved[
                "training_folder"
            ],
            fold_number,
        )
        for fold_number
        in range(
            1,
            number_of_folds + 1,
        )
    ]

    # Metadata is loaded after the first fold establishes feature order.
    feature_to_group: Dict[
        str,
        str,
    ] = {}

    feature_to_pretty_name: Dict[
        str,
        str,
    ] = {}

    fold_metrics: List[
        Dict[str, Any]
    ] = []

    probability_columns: Dict[
        str,
        np.ndarray
    ] = {}

    test_reference_ids: Optional[
        np.ndarray
    ] = None

    test_reference_y: Optional[
        np.ndarray
    ] = None

    raw_importances: List[
        np.ndarray
    ] = []

    repeat_importance_stds: List[
        np.ndarray
    ] = []

    final_feature_columns: Optional[
        List[str]
    ] = None

    for fold_number in range(
        1,
        number_of_folds + 1,
    ):
        print(
            "\n"
            + "=" * 80
        )

        print(
            f"FOLD MODEL "
            f"{fold_number}/"
            f"{number_of_folds}"
        )

        print(
            "=" * 80
        )

        fold_seed = (
            seed
            + fold_number
        )

        set_global_seed(
            fold_seed
        )

        fold_output = (
            output_root
            / f"fold_{fold_number}"
        )

        ensure_dir(
            fold_output
        )

        validation_path = (
            fold_paths[
                fold_number - 1
            ]
        )

        train_paths = [
            path
            for index, path
            in enumerate(
                fold_paths,
                start=1,
            )
            if index != fold_number
        ]

        # =============================================================
        # TRAIN
        # =============================================================
        if resolved["mode"] == "train":
            (
                raw_train,
                y_train,
                train_ids,
                feature_columns,
            ) = load_excel_rows(
                train_paths,
                id_column=(
                    id_column
                ),
                label_column=(
                    label_column
                ),
                expected_feature_columns=None,
            )

            (
                raw_validation,
                y_validation,
                validation_ids,
                _,
            ) = load_excel_rows(
                [
                    validation_path
                ],
                id_column=(
                    id_column
                ),
                label_column=(
                    label_column
                ),
                expected_feature_columns=(
                    feature_columns
                ),
            )

            (
                raw_test,
                y_test,
                test_ids,
                _,
            ) = load_excel_rows(
                [
                    resolved[
                        "test_file"
                    ]
                ],
                id_column=(
                    id_column
                ),
                label_column=(
                    label_column
                ),
                expected_feature_columns=(
                    feature_columns
                ),
            )

            validate_no_overlap(
                train_ids,
                validation_ids,
                test_ids,
            )

            (
                preprocessing_state,
                X_train,
            ) = fit_preprocessor(
                raw_train,
                feature_columns,
            )

            X_validation = (
                transform_tabular(
                    raw_validation,
                    preprocessing_state,
                )
            )

            X_test = (
                transform_tabular(
                    raw_test,
                    preprocessing_state,
                )
            )

            save_preprocessor(
                preprocessing_state,
                fold_output
                / "tabular_preprocessor.json",
            )

            (
                fold_output
                / "used_tabular_columns.txt"
            ).write_text(
                "\n".join(
                    feature_columns
                )
                + "\n",
                encoding="utf-8",
            )

            class_weights = (
                compute_binary_class_weights(
                    y_train
                )
            )

            train_dataset = (
                TabularDataset(
                    X_train,
                    y_train,
                    train_ids,
                )
            )

            validation_dataset = (
                TabularDataset(
                    X_validation,
                    y_validation,
                    validation_ids,
                )
            )

            train_loader = (
                make_dataloader(
                    train_dataset,
                    batch_size=int(
                        training_config[
                            "batch_size"
                        ]
                    ),
                    shuffle=True,
                    num_workers=int(
                        training_config[
                            "num_workers"
                        ]
                    ),
                )
            )

            validation_loader = (
                make_dataloader(
                    validation_dataset,
                    batch_size=int(
                        training_config[
                            "batch_size"
                        ]
                    ),
                    shuffle=False,
                    num_workers=int(
                        training_config[
                            "num_workers"
                        ]
                    ),
                )
            )

            model = ResidualTabularMLP(
                tabular_in=int(
                    preprocessing_state[
                        "model_input_dim"
                    ]
                ),
                hidden_dim=int(
                    model_config[
                        "hidden_dim"
                    ]
                ),
                bottleneck_dim=int(
                    model_config[
                        "bottleneck_dim"
                    ]
                ),
                embedding_dim=int(
                    model_config[
                        "embedding_dim"
                    ]
                ),
                n_residual_blocks=int(
                    model_config[
                        "n_residual_blocks"
                    ]
                ),
                dropout_rate_tabular=float(
                    model_config[
                        "dropout_rate_tabular"
                    ]
                ),
                learning_rate=float(
                    training_config[
                        "learning_rate"
                    ]
                ),
                weight_decay=float(
                    training_config.get(
                        "weight_decay",
                        0.0,
                    )
                ),
                optimizer_name=str(
                    training_config.get(
                        "optimizer",
                        "adam",
                    )
                ),
                class_weights=(
                    class_weights
                ),
            )

            (
                trainer,
                checkpoint_callback,
            ) = make_trainer(
                fold_output=(
                    fold_output
                ),
                config=config,
                device=device,
            )

            trainer.fit(
                model,
                train_dataloaders=(
                    train_loader
                ),
                val_dataloaders=(
                    validation_loader
                ),
            )

            save_training_curves(
                trainer=trainer,
                fold_output=(
                    fold_output
                ),
            )

            best_checkpoint = Path(
                checkpoint_callback.best_model_path
            )

            model = (
                ResidualTabularMLP.load_from_checkpoint(
                    best_checkpoint,
                    class_weights=(
                        class_weights
                    ),
                )
            )

        # =============================================================
        # LOAD PREVIOUSLY TRAINED STANDALONE MLP
        # =============================================================
        else:
            source_fold = (
                resolved[
                    "checkpoint_dir"
                ]
                / f"fold_{fold_number}"
            )

            preprocessing_state = (
                load_preprocessor(
                    source_fold
                    / "tabular_preprocessor.json"
                )
            )

            feature_columns = list(
                preprocessing_state[
                    "original_columns"
                ]
            )

            (
                raw_test,
                y_test,
                test_ids,
                _,
            ) = load_excel_rows(
                [
                    resolved[
                        "test_file"
                    ]
                ],
                id_column=(
                    id_column
                ),
                label_column=(
                    label_column
                ),
                expected_feature_columns=(
                    feature_columns
                ),
            )

            X_test = transform_tabular(
                raw_test,
                preprocessing_state,
            )

            best_checkpoint = (
                checkpoint_from_folder(
                    source_fold
                )
            )

            model = (
                ResidualTabularMLP.load_from_checkpoint(
                    best_checkpoint,
                    class_weights=np.ones(
                        2,
                        dtype=np.float32,
                    ),
                )
            )

        # Same feature order is mandatory for every fold model.
        if final_feature_columns is None:
            final_feature_columns = list(
                feature_columns
            )

            if importance_enabled:
                metadata_path = Path(
                    importance_config.get(
                        "metadata_path",
                        (
                            "/home/ubuntu/TAVI_new/"
                            "dataset/new_features_table.xlsx"
                        ),
                    )
                )

                (
                    feature_to_group,
                    feature_to_pretty_name,
                ) = load_feature_metadata(
                    metadata_path=(
                        metadata_path
                    ),
                    feature_columns=(
                        final_feature_columns
                    ),
                )

        elif list(
            feature_columns
        ) != final_feature_columns:
            raise AssertionError(
                "Feature order differs between fold models."
            )

        # =============================================================
        # TEST
        # =============================================================
        test_probabilities = (
            predict_probabilities(
                model=model,
                X=X_test,
                batch_size=int(
                    training_config[
                        "batch_size"
                    ]
                ),
                device=device,
            )
        )

        (
            metrics,
            predictions,
            confusion,
        ) = calculate_metrics(
            y_true=y_test,
            probabilities=(
                test_probabilities
            ),
            threshold=threshold,
        )

        metrics.update(
            {
                "fold_model": (
                    fold_number
                ),
                "mode": (
                    resolved["mode"]
                ),
                "training_dataset": (
                    resolved[
                        "training_dataset"
                    ]
                ),
                "test_dataset": (
                    resolved[
                        "test_dataset"
                    ]
                ),
                "data_percentage": (
                    resolved[
                        "data_percentage"
                    ]
                ),
                "test_samples": int(
                    len(y_test)
                ),
                "number_of_original_features": int(
                    len(feature_columns)
                ),
                "mlp_input_width": int(
                    X_test.shape[1]
                ),
            }
        )

        fold_metrics.append(
            metrics
        )

        pd.DataFrame(
            {
                "ID": test_ids,
                "True": y_test,
                "Pred": predictions,
                "PacemakerProbability": (
                    test_probabilities
                ),
                "Correct": (
                    predictions
                    == y_test
                ),
            }
        ).to_excel(
            fold_output
            / "test_predictions.xlsx",
            index=False,
        )

        with (
            fold_output
            / "test_metrics.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                metrics,
                handle,
                indent=2,
            )

        save_confusion_matrix(
            confusion=(
                confusion
            ),
            save_path=(
                fold_output
                / "confusion_matrix.png"
            ),
            title=(
                "Independent Test Confusion Matrix — "
                f"Fold Model {fold_number}"
            ),
        )

        save_calibration_plot(
            y_true=y_test,
            probabilities=(
                test_probabilities
            ),
            save_path=(
                fold_output
                / "calibration.png"
            ),
            title=(
                "Independent Test Calibration — "
                f"Fold Model {fold_number}"
            ),
            n_bins=(
                calibration_bins
            ),
        )

        # =============================================================
        # FEATURE IMPORTANCE
        # =============================================================
        if importance_enabled:
            (
                raw_importance,
                repeat_std,
                baseline_test_loss,
            ) = calculate_paired_permutation_importance(
                model=model,
                X_test=X_test,
                y_test=y_test,
                feature_columns=(
                    feature_columns
                ),
                number_of_repeats=int(
                    importance_config.get(
                        "n_repeats",
                        5,
                    )
                ),
                seed=fold_seed,
                batch_size=int(
                    training_config[
                        "batch_size"
                    ]
                ),
                device=device,
            )

            raw_importances.append(
                raw_importance
            )

            repeat_importance_stds.append(
                repeat_std
            )

            importance_denominator = float(
                np.sum(
                    np.abs(
                        raw_importance
                    )
                )
            ) or 1.0

            importance_percent = (
                100.0
                * np.abs(
                    raw_importance
                )
                / importance_denominator
            )

            fold_importance_table = (
                pd.DataFrame(
                    {
                        "feature": (
                            feature_columns
                        ),
                        "feature_pretty": [
                            feature_to_pretty_name[
                                feature
                            ]
                            for feature
                            in feature_columns
                        ],
                        "group": [
                            feature_to_group[
                                feature
                            ]
                            for feature
                            in feature_columns
                        ],
                        "raw_delta_log_loss": (
                            raw_importance
                        ),
                        "repeat_std_raw": (
                            repeat_std
                        ),
                        "importance_percent": (
                            importance_percent
                        ),
                        "baseline_test_log_loss": (
                            baseline_test_loss
                        ),
                    }
                )
                .sort_values(
                    "importance_percent",
                    ascending=False,
                )
            )

            fold_importance_table.to_csv(
                fold_output
                / "feature_importance.csv",
                index=False,
            )

            plot_fold_feature_importance(
                importance_table=(
                    fold_importance_table
                ),
                save_path=(
                    fold_output
                    / "feature_importance_top15.png"
                ),
                title=(
                    "Independent Test Feature Importance — "
                    f"Fold Model {fold_number}"
                ),
            )

        probability_columns[
            f"Prob_FoldModel_{fold_number}"
        ] = test_probabilities

        if test_reference_ids is None:
            test_reference_ids = (
                test_ids.copy()
            )

            test_reference_y = (
                y_test.copy()
            )

        else:
            if not np.array_equal(
                test_reference_ids,
                test_ids,
            ):
                raise AssertionError(
                    "Test patient/order changed "
                    "between fold evaluations."
                )

            if not np.array_equal(
                test_reference_y,
                y_test,
            ):
                raise AssertionError(
                    "Test labels changed "
                    "between fold evaluations."
                )

        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"F1:       "
            f"{metrics['f1_score']:.4f}"
        )

        print(
            f"AUC-ROC:  "
            f"{metrics['auc_roc']:.4f}"
        )

        del model
        clean_memory()

    # =============================================================================
    # FOLD SUMMARY
    # =============================================================================

    fold_metrics_table = (
        pd.DataFrame(
            fold_metrics
        )
    )

    fold_metrics_table.to_csv(
        output_root
        / "fold_metrics.csv",
        index=False,
    )

    metric_names = [
        "accuracy",
        "f1_score",
        "auc_roc",
        "log_loss",
        "pacer_accuracy",
        "no_event_accuracy",
        "false_negatives",
        "false_positives",
    ]

    mean_std: Dict[
        str,
        Dict[str, float],
    ] = {}

    for metric_name in metric_names:
        values = (
            fold_metrics_table[
                metric_name
            ]
            .astype(float)
            .to_numpy()
        )

        mean_std[
            metric_name
        ] = {
            "mean": float(
                np.nanmean(
                    values
                )
            ),
            "std": float(
                np.nanstd(
                    values
                )
            ),
        }

    # =============================================================================
    # ENSEMBLE
    # =============================================================================

    probability_matrix = np.column_stack(
        list(
            probability_columns.values()
        )
    )

    ensemble_probabilities = (
        probability_matrix.mean(
            axis=1
        )
    )

    (
        ensemble_metrics,
        ensemble_predictions,
        ensemble_confusion,
    ) = calculate_metrics(
        y_true=(
            test_reference_y
        ),
        probabilities=(
            ensemble_probabilities
        ),
        threshold=threshold,
    )

    pd.DataFrame(
        {
            "ID": (
                test_reference_ids
            ),
            "True": (
                test_reference_y
            ),
            **probability_columns,
            "EnsembleProbability": (
                ensemble_probabilities
            ),
            "EnsemblePred": (
                ensemble_predictions
            ),
        }
    ).to_excel(
        output_root
        / "ensemble_test_predictions.xlsx",
        index=False,
    )

    save_confusion_matrix(
        confusion=(
            ensemble_confusion
        ),
        save_path=(
            output_root
            / "confusion_matrix_ensemble.png"
        ),
        title=(
            "Independent Test Confusion Matrix — "
            "Five-Model Ensemble"
        ),
    )

    save_calibration_plot(
        y_true=(
            test_reference_y
        ),
        probabilities=(
            ensemble_probabilities
        ),
        save_path=(
            output_root
            / "calibration_ensemble.png"
        ),
        title=(
            "Independent Test Calibration — "
            "Five-Model Ensemble"
        ),
        n_bins=(
            calibration_bins
        ),
    )

    # =============================================================================
    # AGGREGATE FEATURE IMPORTANCE
    # =============================================================================

    if importance_enabled:
        aggregate_feature_importance(
            raw_importances=(
                raw_importances
            ),
            repeat_standard_deviations=(
                repeat_importance_stds
            ),
            feature_columns=(
                final_feature_columns
            ),
            feature_to_group=(
                feature_to_group
            ),
            feature_to_pretty_name=(
                feature_to_pretty_name
            ),
            output_folder=(
                feature_importance_folder
            ),
        )

    # =============================================================================
    # SUMMARY
    # =============================================================================

    summary = {
        "mode": (
            resolved["mode"]
        ),
        "training_dataset": (
            resolved[
                "training_dataset"
            ]
        ),
        "test_dataset": (
            resolved[
                "test_dataset"
            ]
        ),
        "data_percentage": (
            resolved[
                "data_percentage"
            ]
        ),
        "test_file": str(
            resolved[
                "test_file"
            ]
        ),
        "architecture": (
            "Input (F standardized values + F masks) "
            "-> 128 -> residual blocks -> 64 -> 32 embedding -> 2 logits"
        ),
        "preprocessing": (
            "Per fold model: training-only median imputation "
            "+ training-only StandardScaler "
            "+ one binary missingness mask per original feature."
        ),
        "feature_importance": (
            "Paired permutation of each standardized value "
            "and its corresponding missingness mask; "
            "plots/normalization/grouping match TabPFN."
        ),
        "mean_std_across_fold_models": (
            mean_std
        ),
        "five_model_ensemble_metrics": (
            ensemble_metrics
        ),
    }

    with (
        output_root
        / "cv_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    with (
        output_root
        / "cv_summary.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "STANDALONE TABULAR MLP SUMMARY\n"
        )

        handle.write(
            "=" * 72
            + "\n\n"
        )

        handle.write(
            f"Training dataset: "
            f"{resolved['training_dataset']}\n"
        )

        handle.write(
            f"Test dataset: "
            f"{resolved['test_dataset']}\n"
        )

        handle.write(
            f"Data percentage: "
            f"{resolved['data_percentage']}%\n"
        )

        handle.write(
            f"Test file: "
            f"{resolved['test_file']}\n\n"
        )

        handle.write(
            "PREPROCESSING\n"
            + "-" * 72
            + "\n"
        )

        handle.write(
            "Training-fold median imputation\n"
        )

        handle.write(
            "Training-fold StandardScaler\n"
        )

        handle.write(
            "One missingness mask per original clinical feature\n\n"
        )

        handle.write(
            "ARCHITECTURE\n"
            + "-" * 72
            + "\n"
        )

        handle.write(
            "Input -> 128 -> ResidualBlock x2 -> 64 -> 32 embedding -> 2 logits\n\n"
        )

        handle.write(
            "MEAN ± STD ACROSS FIVE FOLD MODELS\n"
            + "-" * 72
            + "\n"
        )

        for metric_name in metric_names:
            handle.write(
                f"{metric_name}: "
                f"{mean_std[metric_name]['mean']:.6f} "
                f"± "
                f"{mean_std[metric_name]['std']:.6f}\n"
            )

        handle.write(
            "\nFIVE-MODEL ENSEMBLE\n"
            + "-" * 72
            + "\n"
        )

        for metric_name, value in (
            ensemble_metrics.items()
        ):
            if isinstance(
                value,
                float,
            ):
                handle.write(
                    f"{metric_name}: "
                    f"{value:.6f}\n"
                )
            else:
                handle.write(
                    f"{metric_name}: "
                    f"{value}\n"
                )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        "Accuracy: "
        f"{mean_std['accuracy']['mean']:.4f} "
        f"± "
        f"{mean_std['accuracy']['std']:.4f}"
    )

    print(
        "F1 Score: "
        f"{mean_std['f1_score']['mean']:.4f} "
        f"± "
        f"{mean_std['f1_score']['std']:.4f}"
    )

    print(
        "Five-model ensemble accuracy: "
        f"{ensemble_metrics['accuracy']:.4f}"
    )

    print(
        f"\nAll results saved to: "
        f"{output_root}"
    )


if __name__ == "__main__":
    main()
