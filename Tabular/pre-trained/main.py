#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pretrained TabPFN evaluation using predefined dataset folds.

Expected folder structure
-------------------------

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

Behavior
--------

1. Select one dataset: "tum", "lmu", or "merged".
2. Concatenate fold1.xlsx through fold5.xlsx.
3. Fit the default pretrained TabPFN once on all five folds.
4. Evaluate only on test.xlsx.
5. Save predictions, metrics, and confusion matrices.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from tabpfn import TabPFNClassifier


# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_ROOT = Path("/home/ubuntu/TAVI_final/dataset/construct/dataset_splits")

# Choose one:
#     "tum"
#     "lmu"
#     "merged"
DATASET = "merged"

ID_COL = "ID"
LABEL_COL = "LABEL"

OUTPUT_DIR = Path(f"results/pretrained_tabpfn_{DATASET}")

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_FOLDS = 5

NEGATIVE_LABEL_NAME = "No Event"
POSITIVE_LABEL_NAME = "Pacemaker"

DECISION_THRESHOLD = 0.5


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# LABEL HANDLING
# =============================================================================

def normalize_labels(
    labels: pd.Series,
    label_col: str,
) -> pd.Series:
    """
    Convert supported labels into:

        No Event -> 0
        Pacemaker -> 1

    Supports both numeric labels and common text variants.
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

        normalized = normalized_strings.map(
            label_mapping
        )

    invalid_mask = labels.notna() & normalized.isna()

    if invalid_mask.any():
        invalid_values = sorted(
            labels.loc[invalid_mask]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unsupported values found in label column "
            f"'{label_col}': {invalid_values}"
        )

    return normalized


# =============================================================================
# DATA LOADING
# =============================================================================

def get_fold_path(
    dataset_folder: Path,
    fold_number: int,
) -> Path:
    """
    Find a fold file.

    The preferred naming is:

        fold1.xlsx
        fold2.xlsx
        ...

    The alternative fold_1.xlsx style is also accepted.
    """
    candidates = [
        dataset_folder / f"fold{fold_number}.xlsx",
        dataset_folder / f"fold_{fold_number}.xlsx",
    ]

    existing_paths = [
        path
        for path in candidates
        if path.exists()
    ]

    if len(existing_paths) == 1:
        return existing_paths[0]

    if len(existing_paths) > 1:
        raise ValueError(
            f"Multiple files found for fold {fold_number}: "
            f"{existing_paths}"
        )

    raise FileNotFoundError(
        f"Could not find fold{fold_number}.xlsx or "
        f"fold_{fold_number}.xlsx inside:\n"
        f"{dataset_folder}"
    )


def load_excel_dataset(
    excel_path: Path,
    id_col: str,
    label_col: str,
    expected_feature_columns: Sequence[str] | None = None,
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    List[str],
]:
    """
    Load one Excel file.

    Returns:
        X dataframe
        y array
        ID array
        feature column list
    """
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Excel file not found: {excel_path}"
        )

    dataframe = pd.read_excel(excel_path)

    missing_required_columns = sorted(
        {id_col, label_col} - set(dataframe.columns)
    )

    if missing_required_columns:
        raise ValueError(
            f"{excel_path} is missing required columns: "
            f"{missing_required_columns}"
        )

    if dataframe[id_col].isna().any():
        missing_rows = dataframe.index[
            dataframe[id_col].isna()
        ].tolist()

        raise ValueError(
            f"{excel_path} contains missing IDs in rows: "
            f"{missing_rows[:20]}"
        )

    duplicate_mask = dataframe[id_col].duplicated(
        keep=False
    )

    if duplicate_mask.any():
        duplicate_ids = (
            dataframe.loc[duplicate_mask, id_col]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{excel_path} contains duplicate IDs: "
            f"{duplicate_ids[:20]}"
        )

    print(f"\nLoading: {excel_path}")
    print("Original label values:")
    print(
        dataframe[label_col].value_counts(
            dropna=False
        )
    )

    dataframe[label_col] = normalize_labels(
        dataframe[label_col],
        label_col,
    )

    dataframe = dataframe.dropna(
        subset=[label_col]
    ).reset_index(drop=True)

    dataframe[label_col] = dataframe[
        label_col
    ].astype(int)

    available_classes = sorted(
        dataframe[label_col].unique().tolist()
    )

    if available_classes != [0, 1]:
        raise ValueError(
            f"{excel_path} must contain both classes 0 and 1. "
            f"Found classes: {available_classes}"
        )

    if expected_feature_columns is None:
        feature_columns = [
            column
            for column in dataframe.columns
            if column not in {id_col, label_col}
        ]

    else:
        feature_columns = list(
            expected_feature_columns
        )

        missing_features = [
            feature
            for feature in feature_columns
            if feature not in dataframe.columns
        ]

        if missing_features:
            raise ValueError(
                f"{excel_path} is missing features used in the "
                f"training folds:\n{missing_features}"
            )

        additional_columns = [
            column
            for column in dataframe.columns
            if column not in {
                id_col,
                label_col,
                *feature_columns,
            }
        ]

        if additional_columns:
            print(
                "The following extra columns will be ignored:"
            )

            for column in additional_columns:
                print(f"  - {column}")

    if not feature_columns:
        raise ValueError(
            f"No feature columns found in {excel_path}."
        )

    feature_dataframe = dataframe[
        feature_columns
    ].copy()

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
        raise TypeError(
            f"{excel_path} contains non-numeric feature columns:\n"
            + "\n".join(non_numeric_columns)
        )

    feature_dataframe = feature_dataframe.astype(
        np.float32
    )

    y = dataframe[label_col].to_numpy(
        dtype=np.int64
    )

    sample_ids = dataframe[id_col].astype(
        str
    ).to_numpy()

    print(
        f"Samples: {len(dataframe)} | "
        f"Features: {len(feature_columns)}"
    )

    print(
        "Class distribution:",
        dict(
            zip(
                *np.unique(
                    y,
                    return_counts=True,
                )
            )
        ),
    )

    return (
        feature_dataframe,
        y,
        sample_ids,
        feature_columns,
    )


def load_all_training_folds(
    dataset_folder: Path,
    id_col: str,
    label_col: str,
    n_folds: int,
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    List[str],
]:
    """
    Load and concatenate fold1.xlsx through fold5.xlsx.
    """
    fold_feature_frames: List[pd.DataFrame] = []
    fold_labels: List[np.ndarray] = []
    fold_ids: List[np.ndarray] = []

    feature_columns: List[str] | None = None

    previously_seen_ids: set[str] = set()

    for fold_number in range(
        1,
        n_folds + 1,
    ):
        fold_path = get_fold_path(
            dataset_folder,
            fold_number,
        )

        (
            X_fold,
            y_fold,
            ids_fold,
            loaded_feature_columns,
        ) = load_excel_dataset(
            excel_path=fold_path,
            id_col=id_col,
            label_col=label_col,
            expected_feature_columns=feature_columns,
        )

        if feature_columns is None:
            feature_columns = (
                loaded_feature_columns
            )

        current_ids = set(ids_fold)

        overlapping_ids = sorted(
            previously_seen_ids.intersection(
                current_ids
            )
        )

        if overlapping_ids:
            raise ValueError(
                "The same IDs occur in multiple fold files. "
                f"Overlap found in fold {fold_number}: "
                f"{overlapping_ids[:20]}"
            )

        previously_seen_ids.update(current_ids)

        fold_feature_frames.append(X_fold)
        fold_labels.append(y_fold)
        fold_ids.append(ids_fold)

    if feature_columns is None:
        raise RuntimeError(
            "No training folds were loaded."
        )

    X_all = pd.concat(
        fold_feature_frames,
        axis=0,
        ignore_index=True,
    )

    y_all = np.concatenate(
        fold_labels,
        axis=0,
    )

    ids_all = np.concatenate(
        fold_ids,
        axis=0,
    )

    return (
        X_all,
        y_all,
        ids_all,
        feature_columns,
    )


# =============================================================================
# PLOTTING
# =============================================================================

def save_confusion_matrix(
    cm: np.ndarray,
    save_path: Path,
    title: str,
    normalized: bool = False,
) -> None:
    """Save a raw-count or row-normalized confusion matrix."""
    if normalized:
        row_sums = cm.sum(
            axis=1,
            keepdims=True,
        )

        plot_matrix = np.divide(
            cm.astype(float),
            row_sums,
            out=np.zeros_like(
                cm,
                dtype=float,
            ),
            where=row_sums != 0,
        )

        annotation_format = ".2f"

    else:
        plot_matrix = cm
        annotation_format = "d"

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        plot_matrix,
        annot=True,
        fmt=annotation_format,
        cmap="Greys",
        cbar=normalized,
        vmin=0 if normalized else None,
        vmax=1 if normalized else None,
        xticklabels=[
            NEGATIVE_LABEL_NAME,
            POSITIVE_LABEL_NAME,
        ],
        yticklabels=[
            NEGATIVE_LABEL_NAME,
            POSITIVE_LABEL_NAME,
        ],
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    set_global_seed(RANDOM_SEED)

    selected_dataset = DATASET.strip().lower()

    valid_datasets = {
        "tum",
        "lmu",
        "merged",
    }

    if selected_dataset not in valid_datasets:
        raise ValueError(
            f"DATASET must be one of "
            f"{sorted(valid_datasets)}, "
            f"but received '{DATASET}'."
        )

    dataset_folder = (
        DATASET_ROOT / selected_dataset
    )

    test_path = (
        dataset_folder / "test.xlsx"
    )

    if not dataset_folder.is_dir():
        raise NotADirectoryError(
            f"Dataset folder does not exist:\n"
            f"{dataset_folder}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test file does not exist:\n"
            f"{test_path}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 72)
    print("PRETRAINED TABPFN")
    print("=" * 72)

    print(f"Dataset: {selected_dataset}")
    print(f"Dataset folder: {dataset_folder}")
    print(f"Device: {DEVICE}")

    # =========================================================================
    # Load all five training folds
    # =========================================================================

    print("\n" + "=" * 72)
    print("LOADING FIVE TRAINING FOLDS")
    print("=" * 72)

    (
        X_train,
        y_train,
        train_ids,
        feature_columns,
    ) = load_all_training_folds(
        dataset_folder=dataset_folder,
        id_col=ID_COL,
        label_col=LABEL_COL,
        n_folds=N_FOLDS,
    )

    # =========================================================================
    # Load independent test.xlsx
    # =========================================================================

    print("\n" + "=" * 72)
    print("LOADING INDEPENDENT TEST SET")
    print("=" * 72)

    (
        X_test,
        y_test,
        test_ids,
        _,
    ) = load_excel_dataset(
        excel_path=test_path,
        id_col=ID_COL,
        label_col=LABEL_COL,
        expected_feature_columns=feature_columns,
    )

    # =========================================================================
    # Leakage checks
    # =========================================================================

    overlapping_ids = sorted(
        set(train_ids).intersection(
            set(test_ids)
        )
    )

    if overlapping_ids:
        overlap_path = (
            OUTPUT_DIR
            / "overlapping_train_test_ids.xlsx"
        )

        pd.DataFrame(
            {"ID": overlapping_ids}
        ).to_excel(
            overlap_path,
            index=False,
        )

        raise ValueError(
            f"Found {len(overlapping_ids)} IDs in both the "
            f"five training folds and test.xlsx.\n"
            f"Overlap saved to: {overlap_path}"
        )

    print("\n" + "=" * 72)
    print("DATA SUMMARY")
    print("=" * 72)

    print(
        f"Training/context samples: "
        f"{len(y_train)}"
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
        "Training class distribution:",
        dict(
            zip(
                *np.unique(
                    y_train,
                    return_counts=True,
                )
            )
        ),
    )

    print(
        "Test class distribution:",
        dict(
            zip(
                *np.unique(
                    y_test,
                    return_counts=True,
                )
            )
        ),
    )

    # Save exact feature order
    feature_path = (
        OUTPUT_DIR / "feature_columns.txt"
    )

    with open(
        feature_path,
        "w",
        encoding="utf-8",
    ) as feature_file:
        for index, column in enumerate(
            feature_columns,
            start=1,
        ):
            feature_file.write(
                f"{index}\t{column}\n"
            )

    # =========================================================================
    # Fit default pretrained TabPFN
    # =========================================================================

    print("\n" + "=" * 72)
    print("FITTING PRETRAINED TABPFN")
    print("=" * 72)

    classifier = TabPFNClassifier(
        device=DEVICE,
        ignore_pretraining_limits=True,
        random_state=RANDOM_SEED,
        inference_precision=torch.float32,
    )

    classifier.fit(
        X_train.to_numpy(
            dtype=np.float32
        ),
        y_train,
    )

    # =========================================================================
    # Test only on test.xlsx
    # =========================================================================

    print("\n" + "=" * 72)
    print("EVALUATING ON TEST.XLSX")
    print("=" * 72)

    probability_matrix = (
        classifier.predict_proba(
            X_test.to_numpy(
                dtype=np.float32
            )
        )
    )

    print(
        "TabPFN classes:",
        classifier.classes_,
    )

    print(
        "Probability matrix shape:",
        probability_matrix.shape,
    )

    if 1 not in classifier.classes_:
        raise ValueError(
            "Positive pacemaker class 1 is missing from "
            f"TabPFN classes: {classifier.classes_}"
        )

    positive_class_index = list(
        classifier.classes_
    ).index(1)

    pacemaker_probabilities = (
        probability_matrix[
            :,
            positive_class_index,
        ]
    )

    predictions = (
        pacemaker_probabilities
        >= DECISION_THRESHOLD
    ).astype(int)

    # =========================================================================
    # Metrics
    # =========================================================================

    accuracy = float(
        accuracy_score(
            y_test,
            predictions,
        )
    )

    f1 = float(
        f1_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0,
        )
    )

    auc_roc = float(
        roc_auc_score(
            y_test,
            pacemaker_probabilities,
        )
    )

    auc_pr = float(
        average_precision_score(
            y_test,
            pacemaker_probabilities,
        )
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    no_event_accuracy = (
        float(tn / (tn + fp))
        if (tn + fp) > 0
        else float("nan")
    )

    pacemaker_accuracy = (
        float(tp / (tp + fn))
        if (tp + fn) > 0
        else float("nan")
    )

    # =========================================================================
    # Save predictions
    # =========================================================================

    prediction_table = pd.DataFrame(
        {
            "ID": test_ids,
            "True": y_test.astype(int),
            "True Label": np.where(
                y_test == 1,
                POSITIVE_LABEL_NAME,
                NEGATIVE_LABEL_NAME,
            ),
            "Pred": predictions.astype(int),
            "Predicted Label": np.where(
                predictions == 1,
                POSITIVE_LABEL_NAME,
                NEGATIVE_LABEL_NAME,
            ),
            "Pacemaker Probability": (
                pacemaker_probabilities.astype(float)
            ),
            "Correct": (
                predictions == y_test
            ),
        }
    )

    prediction_path = (
        OUTPUT_DIR
        / "tabpfn_test_predictions.xlsx"
    )

    prediction_table.to_excel(
        prediction_path,
        index=False,
    )

    # =========================================================================
    # Save confusion matrices
    # =========================================================================

    raw_cm_path = (
        OUTPUT_DIR
        / "confusion_matrix_test.png"
    )

    normalized_cm_path = (
        OUTPUT_DIR
        / "confusion_matrix_test_normalized.png"
    )

    save_confusion_matrix(
        cm=cm,
        save_path=raw_cm_path,
        title=(
            f"Pretrained TabPFN — "
            f"{selected_dataset.upper()} Test Set"
        ),
        normalized=False,
    )

    save_confusion_matrix(
        cm=cm,
        save_path=normalized_cm_path,
        title=(
            f"Pretrained TabPFN — "
            f"{selected_dataset.upper()} Test Set (Normalized)"
        ),
        normalized=True,
    )

    # =========================================================================
    # Save metrics
    # =========================================================================

    metrics: Dict[str, object] = {
        "dataset": selected_dataset,
        "dataset_folder": str(
            dataset_folder
        ),
        "training_files": [
            str(
                get_fold_path(
                    dataset_folder,
                    fold_number,
                )
            )
            for fold_number in range(
                1,
                N_FOLDS + 1,
            )
        ],
        "test_file": str(test_path),
        "training_samples": int(
            len(y_train)
        ),
        "test_samples": int(
            len(y_test)
        ),
        "number_of_features": int(
            len(feature_columns)
        ),
        "decision_threshold": float(
            DECISION_THRESHOLD
        ),
        "accuracy": accuracy,
        "f1_score": f1,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "no_event_accuracy": (
            no_event_accuracy
        ),
        "pacemaker_accuracy": (
            pacemaker_accuracy
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    metrics_json_path = (
        OUTPUT_DIR
        / "evaluation_metrics.json"
    )

    with open(
        metrics_json_path,
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            metrics,
            metrics_file,
            indent=2,
        )

    summary_text = f"""
PRETRAINED TABPFN TEST RESULTS
========================================================================

Dataset                 : {selected_dataset}
Training/context folder : {dataset_folder}
Independent test file   : {test_path}

Training/context samples: {len(y_train)}
Independent test samples: {len(y_test)}
Number of features      : {len(feature_columns)}

The model was fitted once using fold1.xlsx through fold5.xlsx.
Final evaluation was performed only on test.xlsx.

FINAL TEST METRICS
------------------------------------------------------------------------
Accuracy          : {accuracy:.4f}
F1 Score          : {f1:.4f}
AUC-ROC           : {auc_roc:.4f}
AUC-PR            : {auc_pr:.4f}
No Event Accuracy : {no_event_accuracy:.4f}
Pacemaker Accuracy: {pacemaker_accuracy:.4f}

CONFUSION MATRIX
------------------------------------------------------------------------
Rows = true labels
Columns = predicted labels

                       Pred No Event    Pred Pacemaker
True No Event             {tn:>8}         {fp:>8}
True Pacemaker            {fn:>8}         {tp:>8}

SAVED FILES
------------------------------------------------------------------------
Predictions:
{prediction_path}

Metrics:
{metrics_json_path}

Raw confusion matrix:
{raw_cm_path}

Normalized confusion matrix:
{normalized_cm_path}
"""

    summary_path = (
        OUTPUT_DIR
        / "evaluation_metrics.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as summary_file:
        summary_file.write(
            summary_text.strip() + "\n"
        )

    # =========================================================================
    # Final terminal output
    # =========================================================================

    print("\n" + "=" * 72)
    print("FINAL TEST RESULTS")
    print("=" * 72)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nAdditional metrics:")
    print(f"AUC-ROC  : {auc_roc:.4f}")
    print(f"AUC-PR   : {auc_pr:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    print(
        f"\nPredictions saved to: "
        f"{prediction_path}"
    )

    print(
        f"Metrics saved to: "
        f"{summary_path}"
    )

    print(
        f"Confusion matrices saved to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()

