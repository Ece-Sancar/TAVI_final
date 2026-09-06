#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tuned TabPFN with predefined folds and a fixed independent test set.

Expected dataset structure
--------------------------

/home/ubuntu/dataset_splits/
├── tum/
│   ├── fold1.xlsx
│   ├── fold2.xlsx
│   ├── fold3.xlsx
│   ├── fold4.xlsx
│   ├── fold5.xlsx
│   └── test.xlsx
├── lmu/
│   ├── fold1.xlsx
│   ├── fold2.xlsx
│   ├── fold3.xlsx
│   ├── fold4.xlsx
│   ├── fold5.xlsx
│   └── test.xlsx
└── merged/
    ├── fold1.xlsx
    ├── fold2.xlsx
    ├── fold3.xlsx
    ├── fold4.xlsx
    ├── fold5.xlsx
    └── test.xlsx


Training behavior
-----------------

For model/fold i:

    Training data:
        The other four predefined fold files.

    Validation data:
        fold{i}.xlsx

    Final TabPFN inference context:
        All five non-test fold files.

    Final test data:
        Only test.xlsx from test_dataset.

The final test data always comes from test.xlsx, regardless of whether the
training and test dataset names are the same or different.


Pretrained behavior
-------------------

Set:

    "train_dataset": "pre"

The script then:

1. Reads pretrained.train_dataset to determine which five fold files provide
   the TabPFN inference context.
2. Loads:
       checkpoint_dir/fold_1/best_model.pt
       checkpoint_dir/fold_2/best_model.pt
       ...
3. Evaluates every checkpoint only on test_dataset/test.xlsx.
"""

import gc
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabpfn import TabPFNClassifier
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator


# =============================================================================
# Plot settings
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

VALID_DATA_PERCENTAGES = {
    2,
    5,
    10,
    20,
    50,
    100,
}

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
# General utilities
# =============================================================================

def set_global_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_gpu_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clone_state_dict_to_cpu(
    model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    """
    Create a true checkpoint snapshot.

    model.state_dict().copy() is insufficient because its tensors can continue
    changing while the model is trained.
    """
    return {
        key: tensor.detach().cpu().clone()
        for key, tensor in model.state_dict().items()
    }


def to_serializable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def class_count_dict(y: np.ndarray) -> Dict[str, int]:
    values, counts = np.unique(y, return_counts=True)

    return {
        str(int(value)): int(count)
        for value, count in zip(values, counts)
    }


def safe_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(roc_auc_score(y_true, probabilities))


def safe_log_loss(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)

    return float(
        log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
        )
    )


# =============================================================================
# Configuration
# =============================================================================

def resolve_configuration(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve training and testing folders.

    Training:
        data_percentage == 100
            dataset_root/<train_dataset>/

        data_percentage in {2, 5, 10, 20, 50}
            dataset_root/data_size/<percentage>_percent/<train_dataset>/

    Testing:
        ALWAYS uses the original fixed independent test:
            dataset_root/<test_dataset>/test.xlsx

    Therefore the test set never changes between data-size experiments.
    """
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

    # --------------------------------------------------------
    # Validate dataset selections
    # --------------------------------------------------------
    if train_choice not in VALID_TRAIN_VALUES:
        raise ValueError(
            "train_dataset must be one of "
            f"{sorted(VALID_TRAIN_VALUES)}, "
            f"but received '{train_choice}'."
        )

    if test_choice not in VALID_DATASETS:
        raise ValueError(
            "test_dataset must be one of "
            f"{sorted(VALID_DATASETS)}, "
            f"but received '{test_choice}'."
        )

    if data_percentage not in VALID_DATA_PERCENTAGES:
        raise ValueError(
            "data_percentage must be one of "
            f"{sorted(VALID_DATA_PERCENTAGES)}, "
            f"but received {data_percentage}."
        )

    # --------------------------------------------------------
    # Resolve training dataset
    # --------------------------------------------------------
    if train_choice == "pre":
        pretrained_config = config.get(
            "pretrained",
            {},
        )

        checkpoint_dir = (
            pretrained_config.get(
                "checkpoint_dir"
            )
        )

        if not checkpoint_dir:
            raise ValueError(
                "pretrained.checkpoint_dir is required "
                "when train_dataset is 'pre'."
            )

        source_dataset = str(
            pretrained_config.get(
                "train_dataset",
                "",
            )
        ).strip().lower()

        if source_dataset not in VALID_DATASETS:
            raise ValueError(
                "pretrained.train_dataset must be one of "
                f"{sorted(VALID_DATASETS)}."
            )

        training_dataset = source_dataset
        mode = "pretrained"

    else:
        training_dataset = train_choice
        checkpoint_dir = None
        mode = "train"

    # --------------------------------------------------------
    # Resolve TRAINING FOLDER
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # TEST ALWAYS COMES FROM ORIGINAL FIXED SPLIT
    # --------------------------------------------------------
    test_folder = (
        dataset_root
        / test_choice
    )

    test_file = (
        test_folder
        / "test.xlsx"
    )

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------
    if not training_folder.is_dir():
        raise NotADirectoryError(
            "Training dataset folder does not exist:\n"
            f"{training_folder}"
        )

    if not test_folder.is_dir():
        raise NotADirectoryError(
            "Test dataset folder does not exist:\n"
            f"{test_folder}"
        )

    if not test_file.exists():
        raise FileNotFoundError(
            "Independent test file does not exist:\n"
            f"{test_file}"
        )

    return {
        "mode": mode,
        "data_percentage": data_percentage,
        "training_dataset": training_dataset,
        "test_dataset": test_choice,
        "training_folder": training_folder,
        "test_folder": test_folder,
        "test_file": test_file,
        "checkpoint_dir": (
            Path(checkpoint_dir)
            if checkpoint_dir is not None
            else None
        ),
    }

# =============================================================================
# Dataset loading
# =============================================================================

def normalize_labels(
    labels: pd.Series,
    label_column: str,
) -> pd.Series:
    """
    Convert supported label representations to:

        no event -> 0
        pacer    -> 1
    """
    numeric_labels = pd.to_numeric(
        labels,
        errors="coerce",
    )

    non_missing_mask = labels.notna()

    numeric_conversion_complete = (
        numeric_labels[non_missing_mask].notna().all()
    )

    if numeric_conversion_complete:
        normalized = numeric_labels

    else:
        normalized_strings = (
            labels.astype("string")
            .str.strip()
            .str.lower()
        )

        label_mapping = {
            "no event": 0,
            "no_event": 0,
            "noevent": 0,
            "0": 0,
            "pacer": 1,
            "pacemaker": 1,
            "1": 1,
        }

        normalized = normalized_strings.map(label_mapping)

    invalid_mask = labels.notna() & normalized.isna()

    if invalid_mask.any():
        invalid_values = sorted(
            labels.loc[invalid_mask]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unsupported values in label column '{label_column}': "
            f"{invalid_values}"
        )

    return normalized


def load_excel_dataset(
    excel_path: Path,
    id_column: str,
    label_column: str,
    required_feature_columns: Optional[Sequence[str]] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    pd.DataFrame,
]:
    """
    Load one fold file or test.xlsx.

    All columns except ID and LABEL are used as prediction features.
    """
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Excel file does not exist: {excel_path}"
        )

    dataframe = pd.read_excel(excel_path)

    missing_required_columns = sorted(
        {id_column, label_column} - set(dataframe.columns)
    )

    if missing_required_columns:
        raise ValueError(
            f"{excel_path} is missing required columns: "
            f"{missing_required_columns}"
        )

    dataframe[label_column] = normalize_labels(
        dataframe[label_column],
        label_column,
    )

    dataframe = dataframe.dropna(
        subset=[label_column]
    ).reset_index(drop=True)

    dataframe[label_column] = dataframe[label_column].astype(int)

    available_labels = sorted(
        dataframe[label_column].unique().tolist()
    )

    if available_labels != [0, 1]:
        raise ValueError(
            f"{excel_path} must contain both labels 0 and 1. "
            f"Found: {available_labels}"
        )

    if dataframe[id_column].isna().any():
        missing_id_rows = dataframe.index[
            dataframe[id_column].isna()
        ].tolist()

        raise ValueError(
            f"{excel_path} contains missing IDs in rows: "
            f"{missing_id_rows[:20]}"
        )

    duplicate_mask = dataframe[id_column].duplicated(
        keep=False
    )

    if duplicate_mask.any():
        duplicate_ids = (
            dataframe.loc[duplicate_mask, id_column]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{excel_path} contains duplicate IDs: "
            f"{duplicate_ids[:20]}"
        )

    if required_feature_columns is None:
        feature_columns = [
            column
            for column in dataframe.columns
            if column not in {id_column, label_column}
        ]

    else:
        feature_columns = list(required_feature_columns)

        missing_features = [
            feature
            for feature in feature_columns
            if feature not in dataframe.columns
        ]

        if missing_features:
            raise ValueError(
                f"{excel_path} is missing features used by the "
                f"training dataset: {missing_features}"
            )

        extra_features = [
            column
            for column in dataframe.columns
            if column not in {
                id_column,
                label_column,
                *feature_columns,
            }
        ]

        if extra_features:
            print(
                f"Extra columns in {excel_path.name} will be ignored:"
            )

            for column in extra_features:
                print(f"  - {column}")

    if not feature_columns:
        raise ValueError(
            f"No prediction features were found in {excel_path}."
        )

    feature_dataframe = dataframe[feature_columns].copy()

    non_numeric_columns: List[str] = []

    for column in feature_columns:
        try:
            feature_dataframe[column] = pd.to_numeric(
                feature_dataframe[column],
                errors="raise",
            )
        except (TypeError, ValueError):
            non_numeric_columns.append(column)

    if non_numeric_columns:
        raise ValueError(
            f"{excel_path} contains non-numeric prediction columns: "
            f"{non_numeric_columns}"
        )

    X = feature_dataframe.astype(np.float32).to_numpy()

    y = dataframe[label_column].to_numpy(
        dtype=np.int64
    )

    ids = dataframe[id_column].astype(str).to_numpy()

    return X, y, ids, feature_columns, dataframe


def get_fold_path(
    dataset_folder: Path,
    fold_number: int,
) -> Path:
    """
    The constructed dataset uses fold1.xlsx ... fold5.xlsx.

    Alternative underscore naming is accepted to avoid unnecessary failures.
    """
    candidates = [
        dataset_folder / f"fold{fold_number}.xlsx",
        dataset_folder / f"fold_{fold_number}.xlsx",
    ]

    existing_candidates = [
        path
        for path in candidates
        if path.exists()
    ]

    if len(existing_candidates) == 1:
        return existing_candidates[0]

    if len(existing_candidates) > 1:
        raise ValueError(
            f"Multiple files were found for fold {fold_number}: "
            f"{existing_candidates}"
        )

    raise FileNotFoundError(
        f"Could not find fold{fold_number}.xlsx in "
        f"{dataset_folder}"
    )


def load_predefined_folds(
    dataset_folder: Path,
    number_of_folds: int,
    id_column: str,
    label_column: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load fold1.xlsx through fold5.xlsx.

    test.xlsx is not loaded here and can never enter training or validation.
    """
    fold_datasets: List[Dict[str, Any]] = []

    feature_columns: Optional[List[str]] = None
    previously_seen_ids: set[str] = set()

    for fold_number in range(1, number_of_folds + 1):
        fold_path = get_fold_path(
            dataset_folder,
            fold_number,
        )

        (
            X,
            y,
            ids,
            loaded_feature_columns,
            dataframe,
        ) = load_excel_dataset(
            excel_path=fold_path,
            id_column=id_column,
            label_column=label_column,
            required_feature_columns=feature_columns,
        )

        if feature_columns is None:
            feature_columns = loaded_feature_columns

        current_ids = set(ids)
        overlapping_ids = sorted(
            previously_seen_ids.intersection(current_ids)
        )

        if overlapping_ids:
            raise ValueError(
                f"IDs occur in multiple predefined folds. "
                f"Overlap detected in fold {fold_number}: "
                f"{overlapping_ids[:20]}"
            )

        previously_seen_ids.update(current_ids)

        fold_datasets.append(
            {
                "fold_number": fold_number,
                "path": fold_path,
                "X": X,
                "y": y,
                "ids": ids,
                "dataframe": dataframe,
            }
        )

        print(
            f"Loaded {fold_path}: "
            f"{len(y)} samples, "
            f"class counts {class_count_dict(y)}"
        )

    if feature_columns is None:
        raise RuntimeError(
            "No fold files were loaded."
        )

    return fold_datasets, feature_columns


def concatenate_folds(
    fold_datasets: Sequence[Dict[str, Any]],
    excluded_fold: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_folds = [
        fold_data
        for fold_data in fold_datasets
        if fold_data["fold_number"] != excluded_fold
    ]

    if not selected_folds:
        raise ValueError(
            "No folds remain after excluding the validation fold."
        )

    X = np.concatenate(
        [fold_data["X"] for fold_data in selected_folds],
        axis=0,
    )

    y = np.concatenate(
        [fold_data["y"] for fold_data in selected_folds],
        axis=0,
    )

    ids = np.concatenate(
        [fold_data["ids"] for fold_data in selected_folds],
        axis=0,
    )

    return X, y, ids


# =============================================================================
# Metrics and standard plots
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

    no_event_mask = y_true == 0
    pacer_mask = y_true == 1

    no_event_accuracy = (
        float(np.mean(predictions[no_event_mask] == 0))
        if np.any(no_event_mask)
        else float("nan")
    )

    pacer_accuracy = (
        float(np.mean(predictions[pacer_mask] == 1))
        if np.any(pacer_mask)
        else float("nan")
    )

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, predictions)
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
        "no_event_accuracy": no_event_accuracy,
        "pacer_accuracy": pacer_accuracy,
        "true_negatives": int(confusion[0, 0]),
        "false_positives": int(confusion[0, 1]),
        "false_negatives": int(confusion[1, 0]),
        "true_positives": int(confusion[1, 1]),
    }

    return metrics, predictions, confusion


def save_confusion_matrix(
    confusion: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(7, 6))

    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Greys",
        cbar=False,
        xticklabels=["No event", "Pacer"],
        yticklabels=["No event", "Pacer"],
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_calibration(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    save_path: Path,
    title: str,
    number_of_bins: int,
) -> None:
    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        probabilities,
        n_bins=number_of_bins,
        strategy="uniform",
    )

    plt.figure(figsize=(7, 7))

    plt.plot(
        mean_predicted,
        fraction_positive,
        "o-",
        label="Model",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        color="gray",
        label="Perfect calibration",
    )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_loss_curves(
    train_losses: List[float],
    validation_losses: List[float],
    save_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))

    epochs = np.arange(
        1,
        len(train_losses) + 1,
    )

    plt.plot(
        epochs,
        train_losses,
        label="Train loss",
    )

    plt.plot(
        epochs,
        validation_losses,
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =============================================================================
# TabPFN evaluation model
# =============================================================================

def make_evaluation_classifier(
    trained_classifier: TabPFNClassifier,
    model_configuration: Dict[str, Any],
    number_of_estimators: int,
    number_of_context_samples: Optional[int],
) -> TabPFNClassifier:
    evaluation_configuration = {
        **model_configuration,
        "n_estimators": number_of_estimators,
        "inference_config": {
            "SUBSAMPLE_SAMPLES": number_of_context_samples,
        },
    }

    return clone_model_for_evaluation(
        trained_classifier,
        evaluation_configuration,
        TabPFNClassifier,
    )


# =============================================================================
# Feature metadata
# =============================================================================

def load_feature_metadata(
    metadata_path: Path,
    feature_columns: Sequence[str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    metadata = pd.read_excel(metadata_path)

    required_columns = {
        "Feature",
        "Group",
        "Mapping Name",
    }

    missing_columns = (
        required_columns - set(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Feature metadata is missing columns: "
            f"{sorted(missing_columns)}"
        )

    metadata["Feature"] = metadata["Feature"].astype(str)
    metadata["Group"] = metadata["Group"].astype(str)
    metadata["Mapping Name"] = (
        metadata["Mapping Name"].astype(str)
    )

    invalid_groups = (
        set(metadata["Group"]) - set(VALID_GROUPS)
    )

    if invalid_groups:
        raise ValueError(
            f"Invalid groups in feature metadata: "
            f"{sorted(invalid_groups)}"
        )

    metadata_features = set(metadata["Feature"])

    missing_features = (
        set(feature_columns) - metadata_features
    )

    if missing_features:
        raise ValueError(
            "The following model features are missing from the "
            "feature metadata file:\n"
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

    return feature_to_group, feature_to_pretty_name


# =============================================================================
# Permutation feature importance
# =============================================================================

def calculate_permutation_importance(
    classifier: TabPFNClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    number_of_repeats: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Permute one test feature at a time.

    Importance is defined as the increase in test log loss after permutation.
    """
    baseline_probabilities = classifier.predict_proba(
        X_test
    )[:, 1]

    baseline_loss = safe_log_loss(
        y_test,
        baseline_probabilities,
    )

    number_of_features = X_test.shape[1]

    importance_deltas = np.zeros(
        number_of_features,
        dtype=np.float64,
    )

    repeat_standard_deviations = np.zeros(
        number_of_features,
        dtype=np.float64,
    )

    for feature_index in tqdm(
        range(number_of_features),
        desc="Permutation importance",
        leave=False,
    ):
        repeat_losses: List[float] = []

        for repeat_number in range(number_of_repeats):
            random_generator = np.random.RandomState(
                seed
                + feature_index * 1009
                + repeat_number
            )

            permuted_X = X_test.copy()

            permutation_indices = random_generator.permutation(
                len(permuted_X)
            )

            permuted_X[:, feature_index] = (
                permuted_X[
                    permutation_indices,
                    feature_index,
                ]
            )

            permuted_probabilities = (
                classifier.predict_proba(
                    permuted_X
                )[:, 1]
            )

            permuted_loss = safe_log_loss(
                y_test,
                permuted_probabilities,
            )

            repeat_losses.append(
                permuted_loss - baseline_loss
            )

        importance_deltas[feature_index] = float(
            np.mean(repeat_losses)
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


def plot_fold_feature_importance(
    importance_table: pd.DataFrame,
    save_path: Path,
    title: str,
) -> None:
    top_features = (
        importance_table
        .nlargest(15, "importance_percent")
        .sort_values(
            "importance_percent",
            ascending=True,
        )
    )

    colors = [
        GROUP_COLORS[group]
        for group in top_features["group"]
    ]

    plt.figure(figsize=(14, 11))

    plt.barh(
        top_features["feature_pretty"],
        top_features["importance_percent"],
        color=colors,
        edgecolor="black",
        linewidth=1.0,
    )

    plt.xlabel("Importance (%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_aggregate_feature_importance(
    feature_table: pd.DataFrame,
    error_column: str,
    save_path: Path,
    title: str,
) -> None:
    top_features = (
        feature_table
        .nlargest(15, "importance_mean")
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
        for group in top_features["group"]
    ]

    plt.figure(figsize=(15, 12))

    plt.barh(
        y_positions,
        top_features["importance_mean"],
        xerr=top_features[error_column],
        color=colors,
        edgecolor="black",
        linewidth=1.0,
        capsize=6,
    )

    plt.yticks(
        y_positions,
        top_features["feature_pretty"],
    )

    plt.xlabel("Importance (%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_group_importance(
    group_table: pd.DataFrame,
    mean_column: str,
    standard_deviation_column: str,
    save_path: Path,
    title: str,
) -> None:
    sorted_table = group_table.sort_values(
        mean_column,
        ascending=True,
    )

    y_positions = np.arange(
        len(sorted_table)
    )

    colors = [
        GROUP_COLORS[group]
        for group in sorted_table["group"]
    ]

    plt.figure(figsize=(13, 9))

    plt.barh(
        y_positions,
        sorted_table[mean_column],
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
        sorted_table["group"],
    )

    plt.xlabel("Importance (%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def aggregate_feature_importance(
    raw_importances: List[np.ndarray],
    repeat_standard_deviations: List[np.ndarray],
    feature_columns: Sequence[str],
    feature_to_group: Dict[str, str],
    feature_to_pretty_name: Dict[str, str],
    output_folder: Path,
) -> None:
    raw_matrix = np.vstack(raw_importances)

    repeat_std_matrix = np.vstack(
        repeat_standard_deviations
    )

    mean_raw_importance = raw_matrix.mean(
        axis=0
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
            np.abs(mean_raw_importance)
        )
    ) or 1.0

    importance_mean = (
        100.0
        * np.abs(mean_raw_importance)
        / normalization_denominator
    )

    fold_importance_std = (
        100.0
        * np.abs(fold_raw_std)
        / normalization_denominator
    )

    repeat_importance_std = (
        100.0
        * np.abs(
            repeat_std_matrix.mean(axis=0)
        )
        / normalization_denominator
    )

    feature_table = pd.DataFrame(
        {
            "feature": feature_columns,
            "feature_pretty": [
                feature_to_pretty_name[feature]
                for feature in feature_columns
            ],
            "group": [
                feature_to_group[feature]
                for feature in feature_columns
            ],
            "importance_mean": importance_mean,
            "repeat_std": repeat_importance_std,
            "fold_std": fold_importance_std,
            "raw_mean": mean_raw_importance,
            "raw_fold_std": fold_raw_std,
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
        feature_table=feature_table,
        error_column="repeat_std",
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
        feature_table=feature_table,
        error_column="fold_std",
        save_path=(
            output_folder
            / "features_top15_fold_std.png"
        ),
        title=(
            "Feature Importances — "
            "Fold Model Variability"
        ),
    )

    group_total_rows: List[List[float]] = []
    group_average_rows: List[List[float]] = []

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

        group_totals: List[float] = []
        group_averages: List[float] = []

        for group in VALID_GROUPS:
            group_indices = [
                index
                for index, feature
                in enumerate(feature_columns)
                if feature_to_group[feature] == group
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
                group_totals.append(0.0)
                group_averages.append(0.0)

        group_total_rows.append(group_totals)
        group_average_rows.append(group_averages)

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
                group_total_matrix.mean(axis=0)
            ),
            "total_importance_std": (
                group_total_matrix.std(
                    axis=0,
                    ddof=standard_deviation_ddof,
                )
            ),
            "average_per_feature_mean": (
                group_average_matrix.mean(axis=0)
            ),
            "average_per_feature_std": (
                group_average_matrix.std(
                    axis=0,
                    ddof=standard_deviation_ddof,
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
        group_table=group_table,
        mean_column="total_importance_mean",
        standard_deviation_column=(
            "total_importance_std"
        ),
        save_path=(
            output_folder
            / "groups_total_importance.png"
        ),
        title="Group Total Importance",
    )

    plot_group_importance(
        group_table=group_table,
        mean_column="average_per_feature_mean",
        standard_deviation_column=(
            "average_per_feature_std"
        ),
        save_path=(
            output_folder
            / "groups_average_per_feature.png"
        ),
        title="Group Average Importance per Feature",
    )


# =============================================================================
# Fine-tuning
# =============================================================================

def fine_tune_fold_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    fold_number: int,
    fold_seed: int,
    fold_output_folder: Path,
    model_configuration: Dict[str, Any],
    device: str,
    number_of_epochs: int,
    patience: int,
    batch_size: int,
    meta_batch_size: int,
    learning_rate: float,
    number_of_validation_estimators: int,
    number_of_context_samples: Optional[int],
) -> Tuple[
    TabPFNClassifier,
    float,
    int,
]:
    """
    Fine-tune one fold model.

    Fine-tuning:
        Other four fold files.

    Validation:
        Current fold file.
    """
    if len(np.unique(y_train)) != 2:
        raise RuntimeError(
            f"Fold {fold_number}: training data does not "
            "contain both classes."
        )

    if len(np.unique(y_validation)) != 2:
        raise RuntimeError(
            f"Fold {fold_number}: validation data does not "
            "contain both classes."
        )

    classifier = TabPFNClassifier(
        **model_configuration
    )

    classifier._initialize_model_variables()

    optimizer = Adam(
        classifier.model_.parameters(),
        lr=learning_rate,
    )

    loss_function = torch.nn.CrossEntropyLoss()

    best_validation_loss = float("inf")
    patience_counter = 0

    best_model_path = (
        fold_output_folder / "best_model.pt"
    )

    train_loss_curve: List[float] = []
    validation_loss_curve: List[float] = []

    def episode_splitter(
        episode_X: np.ndarray,
        episode_y: np.ndarray,
    ):
        return train_test_split(
            episode_X,
            episode_y,
            test_size=0.5,
            random_state=fold_seed,
            stratify=episode_y,
        )

    training_episodes = (
        classifier.get_preprocessed_datasets(
            X_train,
            y_train,
            episode_splitter,
            batch_size,
        )
    )

    training_loader = DataLoader(
        training_episodes,
        batch_size=meta_batch_size,
        collate_fn=meta_dataset_collator,
    )

    for epoch in range(
        1,
        number_of_epochs + 1,
    ):
        print(
            f"\nFold {fold_number} | "
            f"Epoch {epoch}/{number_of_epochs}"
        )

        classifier.model_.train()

        epoch_losses: List[float] = []
        skipped_episodes = 0

        for (
            X_context,
            X_query,
            y_context,
            y_query,
            categorical_indices,
            configurations,
        ) in tqdm(
            training_loader,
            desc=(
                f"Fold {fold_number}, "
                f"epoch {epoch}"
            ),
        ):
            if isinstance(y_context, torch.Tensor):
                y_context = y_context.to(
                    device=device,
                    dtype=torch.long,
                )
            else:
                y_context = torch.as_tensor(
                    np.asarray(y_context),
                    dtype=torch.long,
                    device=device,
                )

            if isinstance(y_query, torch.Tensor):
                y_query = y_query.to(
                    device=device,
                    dtype=torch.long,
                )
            else:
                y_query = torch.as_tensor(
                    np.asarray(y_query),
                    dtype=torch.long,
                    device=device,
                )

            if (
                len(torch.unique(y_context)) != 2
                or len(torch.unique(y_query)) != 2
            ):
                skipped_episodes += 1
                continue

            optimizer.zero_grad(
                set_to_none=True
            )

            classifier.fit_from_preprocessed(
                X_context,
                y_context,
                categorical_indices,
                configurations,
            )

            logits = classifier.forward(
                X_query,
                return_logits=True,
            )

            loss = loss_function(
                logits,
                y_query,
            )

            loss.backward()
            optimizer.step()

            epoch_losses.append(
                float(loss.item())
            )

        if not epoch_losses:
            raise RuntimeError(
                f"Fold {fold_number}, epoch {epoch}: "
                "all training episodes were skipped."
            )

        mean_train_loss = float(
            np.mean(epoch_losses)
        )

        train_loss_curve.append(
            mean_train_loss
        )

        print(
            f"Training loss: {mean_train_loss:.6f}"
        )

        print(
            f"Skipped episodes: {skipped_episodes}"
        )

        classifier.model_.eval()

        validation_classifier = (
            make_evaluation_classifier(
                trained_classifier=classifier,
                model_configuration=(
                    model_configuration
                ),
                number_of_estimators=(
                    number_of_validation_estimators
                ),
                number_of_context_samples=(
                    number_of_context_samples
                ),
            )
        )

        validation_classifier.fit(
            X_train,
            y_train,
        )

        validation_probabilities = (
            validation_classifier.predict_proba(
                X_validation
            )[:, 1]
        )

        validation_loss = safe_log_loss(
            y_validation,
            validation_probabilities,
        )

        validation_loss_curve.append(
            validation_loss
        )

        print(
            f"Validation loss: "
            f"{validation_loss:.6f}"
        )

        del validation_classifier
        clean_gpu_memory()

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            patience_counter = 0

            torch.save(
                clone_state_dict_to_cpu(
                    classifier.model_
                ),
                best_model_path,
            )

            print(
                "New best checkpoint saved."
            )

        else:
            patience_counter += 1

            print(
                f"Patience: "
                f"{patience_counter}/{patience}"
            )

            if patience_counter >= patience:
                print("Early stopping.")
                break

    if not best_model_path.exists():
        raise RuntimeError(
            f"No checkpoint was saved for "
            f"fold {fold_number}."
        )

    loss_table = pd.DataFrame(
        {
            "epoch": np.arange(
                1,
                len(train_loss_curve) + 1,
            ),
            "train_loss": train_loss_curve,
            "validation_loss": (
                validation_loss_curve
            ),
        }
    )

    loss_table.to_csv(
        fold_output_folder
        / "loss_curves.csv",
        index=False,
    )

    plot_loss_curves(
        train_losses=train_loss_curve,
        validation_losses=(
            validation_loss_curve
        ),
        save_path=(
            fold_output_folder
            / "loss_plot.png"
        ),
    )

    best_state = torch.load(
        best_model_path,
        map_location=device,
    )

    classifier.model_.load_state_dict(
        best_state
    )

    classifier.model_.eval()

    return (
        classifier,
        best_validation_loss,
        len(train_loss_curve),
    )


def load_pretrained_fold_model(
    checkpoint_path: Path,
    model_configuration: Dict[str, Any],
    device: str,
) -> TabPFNClassifier:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: "
            f"{checkpoint_path}"
        )

    classifier = TabPFNClassifier(
        **model_configuration
    )

    classifier._initialize_model_variables()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    classifier.model_.load_state_dict(
        checkpoint
    )

    classifier.model_.eval()

    return classifier


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    with open(
        "config.json",
        "r",
        encoding="utf-8",
    ) as config_file:
        config = json.load(config_file)

    resolved = resolve_configuration(config)

    id_column = config["id_col"]
    label_column = config["label_col"]

    base_output_folder = Path(
        config["output_dir"]
    )

    data_percentage = int(
        resolved["data_percentage"]
    )

    output_folder = (
        base_output_folder
        / f"{data_percentage}_percent"
    )

    number_of_folds = int(
        config["n_folds"]
    )

    seed = int(
        config["random_seed"]
    )

    device = str(
        config["device"]
    )

    fine_tuning_config = config["finetuning"]
    evaluation_config = config.get(
        "evaluation",
        {},
    )
    importance_config = config.get(
        "feature_importance",
        {},
    )

    number_of_epochs = int(
        fine_tuning_config["epochs"]
    )

    learning_rate = float(
        fine_tuning_config["learning_rate"]
    )

    batch_size = int(
        fine_tuning_config["batch_size"]
    )

    meta_batch_size = int(
        fine_tuning_config["meta_batch_size"]
    )

    patience = int(
        fine_tuning_config["patience"]
    )

    decision_threshold = float(
        evaluation_config.get(
            "threshold",
            0.5,
        )
    )

    validation_estimators = int(
        evaluation_config.get(
            "validation_n_estimators",
            1,
        )
    )

    test_estimators = int(
        evaluation_config.get(
            "n_estimators",
            1,
        )
    )

    calibration_bins = int(
        evaluation_config.get(
            "calibration_bins",
            10,
        )
    )

    number_of_context_samples = (
        evaluation_config.get(
            "n_inference_context_samples",
            None,
        )
    )

    importance_enabled = bool(
        importance_config.get(
            "enabled",
            True,
        )
    )

    importance_repeats = int(
        importance_config.get(
            "n_repeats",
            5,
        )
    )

    metadata_path = Path(
        importance_config.get(
            "metadata_path",
            (
                "/home/ubuntu/TAVI_new/"
                "dataset/new_features_table.xlsx"
            ),
        )
    )

    if number_of_folds != 5:
        raise ValueError(
            "The new dataset construction contains exactly "
            "five predefined folds. Set n_folds to 5."
        )

    ensure_dir(output_folder)
    set_global_seed(seed)

    feature_importance_folder = (
        output_folder / "feature_importance"
    )

    if importance_enabled:
        ensure_dir(
            feature_importance_folder
        )

    print("\n" + "=" * 80)
    print("LOADING PREDEFINED TRAINING FOLDS")
    print("=" * 80)

    predefined_folds, feature_columns = (
        load_predefined_folds(
            dataset_folder=(
                resolved["training_folder"]
            ),
            number_of_folds=number_of_folds,
            id_column=id_column,
            label_column=label_column,
        )
    )

    print("\n" + "=" * 80)
    print("LOADING INDEPENDENT TEST.XLSX")
    print("=" * 80)

    (
        X_test,
        y_test,
        test_ids,
        _,
        _,
    ) = load_excel_dataset(
        excel_path=resolved["test_file"],
        id_column=id_column,
        label_column=label_column,
        required_feature_columns=(
            feature_columns
        ),
    )

    (
        X_all_folds,
        y_all_folds,
        all_fold_ids,
    ) = concatenate_folds(
        predefined_folds
    )

    overlapping_train_test_ids = sorted(
        set(all_fold_ids).intersection(
            set(test_ids)
        )
    )

    if overlapping_train_test_ids:
        overlap_path = (
            output_folder
            / "overlapping_train_test_ids.csv"
        )

        pd.DataFrame(
            {
                "ID": (
                    overlapping_train_test_ids
                )
            }
        ).to_csv(
            overlap_path,
            index=False,
        )

        raise ValueError(
            "The five non-test folds and test.xlsx contain "
            f"{len(overlapping_train_test_ids)} overlapping IDs. "
            f"See: {overlap_path}"
        )

    print("\n" + "=" * 80)
    print("RUN SUMMARY")
    print("=" * 80)

    print(
        f"Training data percentage: "
        f"{resolved['data_percentage']}%"
    )

    print(
        "Training dataset: "
        f"{resolved['training_dataset']}"
    )

    print(
        "Training fold folder: "
        f"{resolved['training_folder']}"
    )

    print(
        "Test dataset: "
        f"{resolved['test_dataset']}"
    )

    print(
        "Independent test file: "
        f"{resolved['test_file']}"
    )

    print(
        f"Non-test fold samples: "
        f"{len(y_all_folds)}"
    )

    print(
        f"Independent test samples: "
        f"{len(y_test)}"
    )

    print(
        f"Number of features: "
        f"{len(feature_columns)}"
    )

    print(
        "Final evaluation data: test.xlsx only"
    )

    with open(
        output_folder / "feature_columns.txt",
        "w",
        encoding="utf-8",
    ) as feature_file:
        for index, feature in enumerate(
            feature_columns,
            start=1,
        ):
            feature_file.write(
                f"{index}\t{feature}\n"
            )

    resolved_run_configuration = {
        "mode": resolved["mode"],
        "data_percentage": (
            resolved["data_percentage"]
        ),
        "training_dataset": (
            resolved["training_dataset"]
        ),
        "training_folder": str(
            resolved["training_folder"]
        ),
        "test_dataset": (
            resolved["test_dataset"]
        ),
        "test_file": str(
            resolved["test_file"]
        ),
        "fold_files": [
            str(fold_data["path"])
            for fold_data in predefined_folds
        ],
        "test_policy": (
            "Final evaluation always uses only "
            "test.xlsx from test_dataset."
        ),
        "training_policy": (
            "For model i, use fold i as validation "
            "and the other four folds for fine-tuning."
        ),
        "final_context_policy": (
            "After checkpoint selection, use all five "
            "non-test folds as the TabPFN context for "
            "prediction on test.xlsx."
        ),
    }

    with open(
        output_folder
        / "resolved_run_config.json",
        "w",
        encoding="utf-8",
    ) as resolved_file:
        json.dump(
            {
                **config,
                "resolved": (
                    resolved_run_configuration
                ),
            },
            resolved_file,
            indent=2,
        )

    if importance_enabled:
        (
            feature_to_group,
            feature_to_pretty_name,
        ) = load_feature_metadata(
            metadata_path=metadata_path,
            feature_columns=feature_columns,
        )
    else:
        feature_to_group = {}
        feature_to_pretty_name = {}

    model_configuration = {
        "ignore_pretraining_limits": True,
        "device": device,
        "n_estimators": 1,
        "random_state": seed,
        "inference_precision": (
            torch.float32
        ),
        "fit_mode": "batched",
        "differentiable_input": False,
    }

    all_fold_predictions: List[
        Dict[str, Any]
    ] = []

    fold_metrics: List[
        Dict[str, Any]
    ] = []

    raw_importances: List[
        np.ndarray
    ] = []

    repeat_importance_stds: List[
        np.ndarray
    ] = []

    summed_confusion_matrix = np.zeros(
        (2, 2),
        dtype=np.int64,
    )

    probability_columns: Dict[
        str,
        np.ndarray
    ] = {}

    for fold_number in range(
        1,
        number_of_folds + 1,
    ):
        print("\n" + "=" * 80)
        print(
            f"MODEL {fold_number}/"
            f"{number_of_folds}"
        )
        print("=" * 80)

        fold_seed = seed + fold_number
        set_global_seed(fold_seed)

        fold_output_folder = (
            output_folder
            / f"fold_{fold_number}"
        )

        ensure_dir(fold_output_folder)

        validation_fold = predefined_folds[
            fold_number - 1
        ]

        X_validation = validation_fold["X"]
        y_validation = validation_fold["y"]

        (
            X_train,
            y_train,
            _,
        ) = concatenate_folds(
            predefined_folds,
            excluded_fold=fold_number,
        )

        fold_manifest = {
            "fold_model": fold_number,
            "training_files": [
                str(fold_data["path"])
                for fold_data in predefined_folds
                if (
                    fold_data["fold_number"]
                    != fold_number
                )
            ],
            "validation_file": str(
                validation_fold["path"]
            ),
            "final_context_files": [
                str(fold_data["path"])
                for fold_data in predefined_folds
            ],
            "final_test_file": str(
                resolved["test_file"]
            ),
            "training_samples": int(
                len(y_train)
            ),
            "validation_samples": int(
                len(y_validation)
            ),
            "final_context_samples": int(
                len(y_all_folds)
            ),
            "test_samples": int(
                len(y_test)
            ),
        }

        with open(
            fold_output_folder
            / "fold_manifest.json",
            "w",
            encoding="utf-8",
        ) as manifest_file:
            json.dump(
                fold_manifest,
                manifest_file,
                indent=2,
            )

        if resolved["mode"] == "train":
            (
                classifier,
                best_validation_loss,
                epochs_trained,
            ) = fine_tune_fold_model(
                X_train=X_train,
                y_train=y_train,
                X_validation=X_validation,
                y_validation=y_validation,
                fold_number=fold_number,
                fold_seed=fold_seed,
                fold_output_folder=(
                    fold_output_folder
                ),
                model_configuration=(
                    model_configuration
                ),
                device=device,
                number_of_epochs=(
                    number_of_epochs
                ),
                patience=patience,
                batch_size=batch_size,
                meta_batch_size=(
                    meta_batch_size
                ),
                learning_rate=(
                    learning_rate
                ),
                number_of_validation_estimators=(
                    validation_estimators
                ),
                number_of_context_samples=(
                    number_of_context_samples
                ),
            )

        else:
            checkpoint_path = (
                resolved["checkpoint_dir"]
                / f"fold_{fold_number}"
                / "best_model.pt"
            )

            classifier = (
                load_pretrained_fold_model(
                    checkpoint_path=(
                        checkpoint_path
                    ),
                    model_configuration=(
                        model_configuration
                    ),
                    device=device,
                )
            )

            best_validation_loss = float(
                "nan"
            )

            epochs_trained = 0

            with open(
                fold_output_folder
                / "loaded_checkpoint.txt",
                "w",
                encoding="utf-8",
            ) as checkpoint_file:
                checkpoint_file.write(
                    str(checkpoint_path) + "\n"
                )

        evaluation_classifier = (
            make_evaluation_classifier(
                trained_classifier=classifier,
                model_configuration=(
                    model_configuration
                ),
                number_of_estimators=(
                    test_estimators
                ),
                number_of_context_samples=(
                    number_of_context_samples
                ),
            )
        )

        # All five non-test folds are used as the inference context.
        evaluation_classifier.fit(
            X_all_folds,
            y_all_folds,
        )

        # Final prediction is always performed only on test.xlsx.
        test_probabilities = (
            evaluation_classifier.predict_proba(
                X_test
            )[:, 1]
        )

        (
            metrics,
            test_predictions,
            confusion,
        ) = calculate_metrics(
            y_true=y_test,
            probabilities=test_probabilities,
            threshold=decision_threshold,
        )

        metrics.update(
            {
                "data_percentage": int(
                    resolved["data_percentage"]
                ),
                "fold_model": fold_number,
                "best_validation_loss": (
                    best_validation_loss
                ),
                "epochs_trained": (
                    epochs_trained
                ),
                "training_samples": int(
                    len(y_train)
                ),
                "validation_samples": int(
                    len(y_validation)
                ),
                "final_context_samples": int(
                    len(y_all_folds)
                ),
                "test_samples": int(
                    len(y_test)
                ),
                "training_dataset": (
                    resolved["training_dataset"]
                ),
                "test_dataset": (
                    resolved["test_dataset"]
                ),
                "test_file": str(
                    resolved["test_file"]
                ),
            }
        )

        fold_metrics.append(metrics)

        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.6f}"
        )

        print(
            f"AUC-ROC: "
            f"{metrics['auc_roc']:.6f}"
        )

        print(
            f"Log loss: "
            f"{metrics['log_loss']:.6f}"
        )

        print(
            f"Pacer accuracy: "
            f"{metrics['pacer_accuracy']:.6f}"
        )

        print(
            f"No-event accuracy: "
            f"{metrics['no_event_accuracy']:.6f}"
        )

        with open(
            fold_output_folder
            / "test_metrics.json",
            "w",
            encoding="utf-8",
        ) as metrics_file:
            json.dump(
                {
                    key: to_serializable(value)
                    for key, value
                    in metrics.items()
                },
                metrics_file,
                indent=2,
            )

        fold_prediction_table = pd.DataFrame(
            {
                "ID": test_ids,
                "True": y_test.astype(int),
                "Pred": (
                    test_predictions.astype(int)
                ),
                "Prob": (
                    test_probabilities.astype(float)
                ),
                "FoldModel": fold_number,

                "DataPercentage": int(
                    resolved["data_percentage"]
                ),

                "TrainDataset": (
                    resolved["training_dataset"]
                ),
                "TestDataset": (
                    resolved["test_dataset"]
                ),
                "TestFile": str(
                    resolved["test_file"]
                ),
            }
        )

        fold_prediction_table.to_excel(
            fold_output_folder
            / "test_predictions.xlsx",
            index=False,
        )

        all_fold_predictions.extend(
            fold_prediction_table.to_dict(
                orient="records"
            )
        )

        probability_columns[
            f"Prob_FoldModel_{fold_number}"
        ] = test_probabilities

        save_confusion_matrix(
            confusion=confusion,
            save_path=(
                fold_output_folder
                / "confusion_matrix.png"
            ),
            title=(
                "Independent Test Confusion Matrix — "
                f"Fold Model {fold_number}"
            ),
        )

        plot_calibration(
            y_true=y_test,
            probabilities=test_probabilities,
            save_path=(
                fold_output_folder
                / "calibration.png"
            ),
            title=(
                "Independent Test Calibration — "
                f"Fold Model {fold_number}"
            ),
            number_of_bins=(
                calibration_bins
            ),
        )

        summed_confusion_matrix += confusion

        if importance_enabled:
            (
                raw_importance,
                repeat_std,
                baseline_test_loss,
            ) = calculate_permutation_importance(
                classifier=(
                    evaluation_classifier
                ),
                X_test=X_test,
                y_test=y_test,
                number_of_repeats=(
                    importance_repeats
                ),
                seed=fold_seed,
            )

            raw_importances.append(
                raw_importance
            )

            repeat_importance_stds.append(
                repeat_std
            )

            importance_denominator = float(
                np.sum(
                    np.abs(raw_importance)
                )
            ) or 1.0

            importance_percent = (
                100.0
                * np.abs(raw_importance)
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
                fold_output_folder
                / "feature_importance.csv",
                index=False,
            )

            plot_fold_feature_importance(
                importance_table=(
                    fold_importance_table
                ),
                save_path=(
                    fold_output_folder
                    / "feature_importance_top15.png"
                ),
                title=(
                    "Independent Test Feature Importance — "
                    f"Fold Model {fold_number}"
                ),
            )

        del evaluation_classifier
        del classifier
        clean_gpu_memory()

    # =========================================================================
    # Save all per-model predictions
    # =========================================================================

    all_predictions_table = pd.DataFrame(
        all_fold_predictions
    )

    all_predictions_table.to_excel(
        output_folder
        / "all_predictions.xlsx",
        index=False,
    )

    fold_metrics_table = pd.DataFrame(
        fold_metrics
    )

    fold_metrics_table.to_csv(
        output_folder
        / "fold_metrics.csv",
        index=False,
    )

    # =========================================================================
    # Ensemble predictions
    # =========================================================================

    ensemble_prediction_table = pd.DataFrame(
        {
            "ID": test_ids,
            "True": y_test.astype(int),
            **probability_columns,
        }
    )

    probability_matrix = np.column_stack(
        list(probability_columns.values())
    )

    ensemble_probabilities = (
        probability_matrix.mean(axis=1)
    )

    ensemble_predictions = (
        ensemble_probabilities
        >= decision_threshold
    ).astype(int)

    ensemble_prediction_table[
        "EnsembleProb"
    ] = ensemble_probabilities

    ensemble_prediction_table[
        "EnsemblePred"
    ] = ensemble_predictions

    ensemble_prediction_table.to_excel(
        output_folder
        / "ensemble_test_predictions.xlsx",
        index=False,
    )

    (
        ensemble_metrics,
        _,
        ensemble_confusion,
    ) = calculate_metrics(
        y_true=y_test,
        probabilities=ensemble_probabilities,
        threshold=decision_threshold,
    )

    with open(
        output_folder
        / "ensemble_test_metrics.json",
        "w",
        encoding="utf-8",
    ) as ensemble_metrics_file:
        json.dump(
            {
                key: to_serializable(value)
                for key, value
                in ensemble_metrics.items()
            },
            ensemble_metrics_file,
            indent=2,
        )

    save_confusion_matrix(
        confusion=ensemble_confusion,
        save_path=(
            output_folder
            / "confusion_matrix_ensemble.png"
        ),
        title=(
            "Independent Test Confusion Matrix — "
            "Five-Model Ensemble"
        ),
    )

    plot_calibration(
        y_true=y_test,
        probabilities=ensemble_probabilities,
        save_path=(
            output_folder
            / "calibration_ensemble.png"
        ),
        title=(
            "Independent Test Calibration — "
            "Five-Model Ensemble"
        ),
        number_of_bins=calibration_bins,
    )

    save_confusion_matrix(
        confusion=summed_confusion_matrix,
        save_path=(
            output_folder
            / "confusion_matrix_sum.png"
        ),
        title=(
            "Summed Independent Test Confusion Matrix "
            "Across Fold Models"
        ),
    )

    # =========================================================================
    # Mean and standard deviation across fold models
    # =========================================================================

    summary_metric_names = [
        "accuracy",
        "f1_score",
        "auc_roc",
        "log_loss",
        "pacer_accuracy",
        "no_event_accuracy",
        "false_negatives",
        "false_positives",
    ]

    mean_std_summary: Dict[
        str,
        Dict[str, float],
    ] = {}

    for metric_name in summary_metric_names:
        metric_values = (
            fold_metrics_table[
                metric_name
            ]
            .astype(float)
            .to_numpy()
        )

        mean_std_summary[metric_name] = {
            "mean": float(
                np.nanmean(metric_values)
            ),
            "std": float(
                np.nanstd(metric_values)
            ),
        }

    summary_payload = {
        "mode": resolved["mode"],
        "data_percentage": int(
            resolved["data_percentage"]
        ),
        "training_dataset": (
            resolved["training_dataset"]
        ),
        "training_folder": str(
            resolved["training_folder"]
        ),
        "test_dataset": (
            resolved["test_dataset"]
        ),
        "test_file": str(
            resolved["test_file"]
        ),
        "test_policy": (
            "Final evaluation always uses only "
            "test.xlsx from test_dataset."
        ),
        "number_of_features": int(
            len(feature_columns)
        ),
        "number_of_non_test_samples": int(
            len(y_all_folds)
        ),
        "number_of_test_samples": int(
            len(y_test)
        ),
        "number_of_fold_models": (
            number_of_folds
        ),
        "mean_std_across_fold_models": (
            mean_std_summary
        ),
        "five_model_ensemble_metrics": (
            ensemble_metrics
        ),
    }

    with open(
        output_folder / "cv_summary.json",
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(
            summary_payload,
            summary_file,
            indent=2,
        )

    with open(
        output_folder / "cv_summary.txt",
        "w",
        encoding="utf-8",
    ) as summary_text_file:
        summary_text_file.write(
            "FINE-TUNED TABPFN SUMMARY\n"
        )

        summary_text_file.write(
            "=" * 72 + "\n\n"
        )

        summary_text_file.write(
            f"Mode: {resolved['mode']}\n"
        )

        summary_text_file.write(
            "Training data percentage: "
            f"{resolved['data_percentage']}%\n"
        )

        summary_text_file.write(
            "Training dataset: "
            f"{resolved['training_dataset']}\n"
        )

        summary_text_file.write(
            "Training folder: "
            f"{resolved['training_folder']}\n"
        )

        summary_text_file.write(
            "Test dataset: "
            f"{resolved['test_dataset']}\n"
        )

        summary_text_file.write(
            "Independent test file: "
            f"{resolved['test_file']}\n"
        )

        summary_text_file.write(
            "Final evaluation uses test.xlsx only.\n\n"
        )

        summary_text_file.write(
            "MEAN ± STD ACROSS FIVE FOLD MODELS\n"
        )

        summary_text_file.write(
            "-" * 72 + "\n"
        )

        for metric_name in summary_metric_names:
            mean_value = (
                mean_std_summary[
                    metric_name
                ]["mean"]
            )

            std_value = (
                mean_std_summary[
                    metric_name
                ]["std"]
            )

            summary_text_file.write(
                f"{metric_name}: "
                f"{mean_value:.6f} "
                f"± {std_value:.6f}\n"
            )

        summary_text_file.write(
            "\nFIVE-MODEL ENSEMBLE\n"
        )

        summary_text_file.write(
            "-" * 72 + "\n"
        )

        for metric_name, value in (
            ensemble_metrics.items()
        ):
            if isinstance(value, float):
                summary_text_file.write(
                    f"{metric_name}: "
                    f"{value:.6f}\n"
                )
            else:
                summary_text_file.write(
                    f"{metric_name}: "
                    f"{value}\n"
                )

    # =========================================================================
    # Aggregate feature importance
    # =========================================================================

    if importance_enabled:
        aggregate_feature_importance(
            raw_importances=raw_importances,
            repeat_standard_deviations=(
                repeat_importance_stds
            ),
            feature_columns=feature_columns,
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

    # =========================================================================
    # Print final result
    # =========================================================================

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    accuracy_mean = mean_std_summary["accuracy"]["mean"]
    accuracy_std = mean_std_summary["accuracy"]["std"]

    f1_mean = mean_std_summary["f1_score"]["mean"]
    f1_std = mean_std_summary["f1_score"]["std"]

    print(
        f"\nTraining data: "
        f"{resolved['data_percentage']}%"
    )

    print("\nMEAN ± STD ACROSS FIVE FOLD MODELS")
    print("-" * 80)

    print(
        f"Accuracy: "
        f"{accuracy_mean:.4f} ± {accuracy_std:.4f}"
    )

    print(
        f"F1 Score: "
        f"{f1_mean:.4f} ± {f1_std:.4f}"
    )

    print("\nFive-model ensemble:")
    print(
        f"Accuracy: "
        f"{ensemble_metrics['accuracy']:.4f}"
    )
    print(
        f"F1 Score: "
        f"{ensemble_metrics['f1_score']:.4f}"
    )

    print(
        f"\nAll results saved to: "
        f"{output_folder}"
    )


if __name__ == "__main__":
    main()