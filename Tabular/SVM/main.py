#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SVM Evaluation with Five Predefined Training Folds and Independent Test Set
---------------------------------------------------------------------------

Expected structure:

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

Behavior:

1. Load fold1.xlsx through fold5.xlsx.
2. Concatenate them into one training set.
3. Fit one SVM pipeline on the complete training set.
4. Evaluate only on test.xlsx.
5. Save predictions, metrics, calibration, and confusion matrices.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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

OUTPUT_DIR = Path(f"results/svm_{DATASET}")

N_FOLDS = 5
RANDOM_SEED = 42
DECISION_THRESHOLD = 0.5

NEGATIVE_LABEL_NAME = "No Event"
POSITIVE_LABEL_NAME = "Pacemaker"


# =============================================================================
# UTILITIES
# =============================================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_labels(
    labels: pd.Series,
    label_col: str,
) -> pd.Series:
    """
    Convert labels to:

        No Event -> 0
        Pacemaker -> 1
    """
    numeric_labels = pd.to_numeric(
        labels,
        errors="coerce",
    )

    non_missing_mask = labels.notna()

    numeric_conversion_complete = (
        numeric_labels[non_missing_mask]
        .notna()
        .all()
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

    invalid_mask = (
        labels.notna()
        & normalized.isna()
    )

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


def get_fold_path(
    dataset_folder: Path,
    fold_number: int,
) -> Path:
    """
    Accept both:

        fold1.xlsx
        fold_1.xlsx
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
        f"fold_{fold_number}.xlsx in:\n"
        f"{dataset_folder}"
    )


# =============================================================================
# DATA LOADING
# =============================================================================

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
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Excel file not found: {excel_path}"
        )

    dataframe = pd.read_excel(excel_path)

    required_columns = {
        id_col,
        label_col,
    }

    missing_columns = sorted(
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{excel_path} is missing required columns: "
            f"{missing_columns}"
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
            dataframe.loc[
                duplicate_mask,
                id_col,
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{excel_path} contains duplicate IDs: "
            f"{duplicate_ids[:20]}"
        )

    dataframe[label_col] = normalize_labels(
        dataframe[label_col],
        label_col,
    )

    dataframe = dataframe.dropna(
        subset=[label_col]
    ).reset_index(drop=True)

    dataframe[label_col] = (
        dataframe[label_col].astype(int)
    )

    available_classes = sorted(
        dataframe[label_col]
        .unique()
        .tolist()
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
            if column not in {
                id_col,
                label_col,
            }
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
                f"{excel_path} is missing training features:\n"
                f"{missing_features}"
            )

        extra_columns = [
            column
            for column in dataframe.columns
            if column not in {
                id_col,
                label_col,
                *feature_columns,
            }
        ]

        if extra_columns:
            print(
                f"\nExtra columns in {excel_path.name} "
                f"will be ignored:"
            )

            for column in extra_columns:
                print(f"  - {column}")

    if not feature_columns:
        raise ValueError(
            f"No prediction features found in {excel_path}."
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
            f"{excel_path} contains non-numeric features:\n"
            + "\n".join(non_numeric_columns)
        )

    feature_dataframe = feature_dataframe.astype(
        np.float64
    )

    y = dataframe[label_col].to_numpy(
        dtype=np.int64
    )

    ids = dataframe[id_col].astype(
        str
    ).to_numpy()

    print(
        f"Loaded {excel_path.name}: "
        f"{len(y)} samples | "
        f"{len(feature_columns)} features | "
        f"class counts "
        f"{dict(zip(*np.unique(y, return_counts=True)))}"
    )

    return (
        feature_dataframe,
        y,
        ids,
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
    feature_frames: List[pd.DataFrame] = []
    label_arrays: List[np.ndarray] = []
    id_arrays: List[np.ndarray] = []

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
                f"IDs occur in multiple training folds. "
                f"Overlap found in fold {fold_number}: "
                f"{overlapping_ids[:20]}"
            )

        previously_seen_ids.update(
            current_ids
        )

        feature_frames.append(X_fold)
        label_arrays.append(y_fold)
        id_arrays.append(ids_fold)

    if feature_columns is None:
        raise RuntimeError(
            "No training folds were loaded."
        )

    X_train = pd.concat(
        feature_frames,
        axis=0,
        ignore_index=True,
    )

    y_train = np.concatenate(
        label_arrays,
        axis=0,
    )

    train_ids = np.concatenate(
        id_arrays,
        axis=0,
    )

    return (
        X_train,
        y_train,
        train_ids,
        feature_columns,
    )


# =============================================================================
# PLOTS
# =============================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Path,
    title: str,
    normalized: bool = False,
) -> None:
    if normalized:
        row_sums = cm.sum(
            axis=1,
            keepdims=True,
        )

        matrix_to_plot = np.divide(
            cm.astype(float),
            row_sums,
            out=np.zeros_like(
                cm,
                dtype=float,
            ),
            where=row_sums != 0,
        )

        value_format = ".2f"

    else:
        matrix_to_plot = cm
        value_format = "d"

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        matrix_to_plot,
        annot=True,
        fmt=value_format,
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


def plot_calibration(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    fraction_positive, mean_predicted = (
        calibration_curve(
            y_true,
            probabilities,
            n_bins=10,
            strategy="uniform",
        )
    )

    plt.figure(figsize=(7, 7))

    plt.plot(
        mean_predicted,
        fraction_positive,
        "o-",
        label="SVM",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        color="gray",
        label="Perfect calibration",
    )

    plt.xlabel(
        "Mean predicted probability"
    )

    plt.ylabel(
        "Fraction of positives"
    )

    plt.title(title)
    plt.legend()
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

    selected_dataset = (
        DATASET.strip().lower()
    )

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
        DATASET_ROOT
        / selected_dataset
    )

    test_path = (
        dataset_folder
        / "test.xlsx"
    )

    if not dataset_folder.is_dir():
        raise NotADirectoryError(
            f"Dataset folder does not exist:\n"
            f"{dataset_folder}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Independent test file does not exist:\n"
            f"{test_path}"
        )

    ensure_dir(OUTPUT_DIR)

    print("\n" + "=" * 72)
    print("SVM EVALUATION")
    print("=" * 72)

    print(
        f"Dataset: {selected_dataset}"
    )

    print(
        f"Dataset folder: {dataset_folder}"
    )

    # =========================================================================
    # Load five training folds
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
        expected_feature_columns=(
            feature_columns
        ),
    )

    # =========================================================================
    # Leakage check
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
            {
                "ID": overlapping_ids
            }
        ).to_excel(
            overlap_path,
            index=False,
        )

        raise ValueError(
            f"Found {len(overlapping_ids)} IDs in both "
            f"the training folds and test.xlsx.\n"
            f"Saved overlap list to: {overlap_path}"
        )

    print("\n" + "=" * 72)
    print("DATA SUMMARY")
    print("=" * 72)

    print(
        f"Training samples: "
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

    feature_path = (
        OUTPUT_DIR
        / "feature_columns.txt"
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
    # SVM pipeline
    # =========================================================================

    print("\n" + "=" * 72)
    print("FITTING SVM")
    print("=" * 72)

    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    model.fit(
        X_train.to_numpy(),
        y_train,
    )

    # =========================================================================
    # Independent test evaluation
    # =========================================================================

    print("\n" + "=" * 72)
    print("EVALUATING ON TEST.XLSX")
    print("=" * 72)

    probability_matrix = (
        model.predict_proba(
            X_test.to_numpy()
        )
    )

    svm_classes = (
        model.named_steps["svm"].classes_
    )

    if 1 not in svm_classes:
        raise ValueError(
            f"Pacemaker class 1 is missing from "
            f"SVM classes: {svm_classes}"
        )

    positive_class_index = list(
        svm_classes
    ).index(1)

    probabilities = (
        probability_matrix[
            :,
            positive_class_index,
        ]
    )

    predictions = (
        probabilities
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
            probabilities,
        )
    )

    auc_pr = float(
        average_precision_score(
            y_test,
            probabilities,
        )
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    total_samples = (
        tn + fp + fn + tp
    )

    false_positive_fraction = (
        fp / total_samples
    )

    false_negative_fraction = (
        fn / total_samples
    )

    no_event_accuracy = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else float("nan")
    )

    pacemaker_accuracy = (
        tp / (tp + fn)
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
                probabilities.astype(float)
            ),
            "Correct": (
                predictions == y_test
            ),
        }
    )

    predictions_path = (
        OUTPUT_DIR
        / "svm_test_predictions.xlsx"
    )

    prediction_table.to_excel(
        predictions_path,
        index=False,
    )

    # =========================================================================
    # Save confusion matrices
    # =========================================================================

    raw_confusion_path = (
        OUTPUT_DIR
        / "confusion_matrix_test.png"
    )

    normalized_confusion_path = (
        OUTPUT_DIR
        / "confusion_matrix_test_normalized.png"
    )

    plot_confusion_matrix(
        cm=cm,
        save_path=raw_confusion_path,
        title=(
            f"SVM — "
            f"{selected_dataset.upper()} Test Set"
        ),
        normalized=False,
    )

    plot_confusion_matrix(
        cm=cm,
        save_path=normalized_confusion_path,
        title=(
            f"SVM — "
            f"{selected_dataset.upper()} Test Set "
            f"(Normalized)"
        ),
        normalized=True,
    )

    # =========================================================================
    # Save calibration plot
    # =========================================================================

    calibration_path = (
        OUTPUT_DIR
        / "calibration_test.png"
    )

    plot_calibration(
        y_true=y_test,
        probabilities=probabilities,
        save_path=calibration_path,
        title=(
            f"SVM Calibration — "
            f"{selected_dataset.upper()} Test Set"
        ),
    )

    # =========================================================================
    # Save metrics
    # =========================================================================

    metrics = {
        "dataset": selected_dataset,
        "dataset_folder": str(
            dataset_folder
        ),
        "test_file": str(
            test_path
        ),
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
        "no_event_accuracy": float(
            no_event_accuracy
        ),
        "pacemaker_accuracy": float(
            pacemaker_accuracy
        ),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "false_negative_rate": float(
            false_negative_rate
        ),
        "false_positive_fraction": float(
            false_positive_fraction
        ),
        "false_negative_fraction": float(
            false_negative_fraction
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
SVM INDEPENDENT TEST RESULTS
========================================================================

Dataset              : {selected_dataset}
Training folder      : {dataset_folder}
Independent test file: {test_path}

The SVM was trained once on fold1.xlsx through fold5.xlsx.
Final evaluation was performed only on test.xlsx.

DATA
------------------------------------------------------------------------
Training samples     : {len(y_train)}
Test samples         : {len(y_test)}
Number of features   : {len(feature_columns)}

FINAL TEST METRICS
------------------------------------------------------------------------
Accuracy              : {accuracy:.4f}
F1 Score              : {f1:.4f}
AUC-ROC               : {auc_roc:.4f}
AUC-PR                : {auc_pr:.4f}
No Event Accuracy     : {no_event_accuracy:.4f}
Pacemaker Accuracy    : {pacemaker_accuracy:.4f}
False Positive Rate   : {false_positive_rate * 100:.2f}%
False Negative Rate   : {false_negative_rate * 100:.2f}%
False Positive Fraction: {false_positive_fraction * 100:.2f}%
False Negative Fraction: {false_negative_fraction * 100:.2f}%

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
{predictions_path}

Metrics:
{metrics_json_path}

Raw confusion matrix:
{raw_confusion_path}

Normalized confusion matrix:
{normalized_confusion_path}

Calibration plot:
{calibration_path}
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
            summary_text.strip()
            + "\n"
        )

    # =========================================================================
    # Final output
    # =========================================================================

    print("\n" + "=" * 72)
    print("FINAL TEST RESULTS")
    print("=" * 72)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"AUC-ROC  : {auc_roc:.4f}"
    )

    print(
        f"AUC-PR   : {auc_pr:.4f}"
    )

    print("\nConfusion matrix:")
    print(cm)

    print(
        f"\nAll results saved to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()