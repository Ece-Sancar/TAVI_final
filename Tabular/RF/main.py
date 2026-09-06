#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random Forest evaluation with five predefined training folds
and one independent test set.

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
2. Concatenate all five folds into one training set.
3. Train one Random Forest model.
4. Evaluate only on test.xlsx.
5. Save confusion matrices, predictions, metrics,
   classification report, and feature importance.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


# =============================================================================
# CONSTANTS
# =============================================================================

VALID_DATASETS = {
    "tum",
    "lmu",
    "merged",
}

NEGATIVE_LABEL_NAME = "No Event"
POSITIVE_LABEL_NAME = "Pacemaker"


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# =============================================================================
# LABEL PROCESSING
# =============================================================================

def normalize_labels(
    labels: pd.Series,
    label_col: str,
) -> pd.Series:
    """
    Convert supported labels into:

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
            f"Unsupported values in label column "
            f"'{label_col}': {invalid_values}"
        )

    return normalized


# =============================================================================
# PATH HELPERS
# =============================================================================

def get_fold_path(
    dataset_folder: Path,
    fold_number: int,
) -> Path:
    """
    Accept both naming styles:

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
    """
    Load one fold file or test.xlsx.
    """
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Excel file not found: {excel_path}"
        )

    dataframe = pd.read_excel(excel_path)

    missing_required_columns = sorted(
        {id_col, label_col}
        - set(dataframe.columns)
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

    dataframe[label_col] = dataframe[
        label_col
    ].astype(int)

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

    labels = dataframe[label_col].to_numpy(
        dtype=np.int64
    )

    ids = dataframe[id_col].astype(
        str
    ).to_numpy()

    print(
        f"Loaded {excel_path.name}: "
        f"{len(labels)} samples | "
        f"{len(feature_columns)} features | "
        f"class counts "
        f"{dict(zip(*np.unique(labels, return_counts=True)))}"
    )

    return (
        feature_dataframe,
        labels,
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
    """
    Load and concatenate all five non-test fold files.
    """
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
# PLOTTING
# =============================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Path,
    title: str,
    normalized: bool = False,
) -> None:
    """
    Save raw-count or row-normalized confusion matrix.
    """
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


def plot_feature_importance(
    importance_dataframe: pd.DataFrame,
    save_path: Path,
    top_k: int,
) -> None:
    """
    Plot the top-k impurity-based feature importances.
    """
    top_dataframe = (
        importance_dataframe
        .head(top_k)
        .sort_values(
            "importance",
            ascending=True,
        )
    )

    plt.figure(figsize=(13, 10))

    sns.barplot(
        data=top_dataframe,
        x="importance",
        y="feature",
        orient="h",
        color="gray",
    )

    plt.title(
        f"Random Forest Top {top_k} Feature Importances"
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
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

def main(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)

    selected_dataset = (
        args.dataset.strip().lower()
    )

    if selected_dataset not in VALID_DATASETS:
        raise ValueError(
            f"--dataset must be one of "
            f"{sorted(VALID_DATASETS)}, "
            f"but received '{args.dataset}'."
        )

    dataset_root = Path(
        args.dataset_root
    )

    dataset_folder = (
        dataset_root
        / selected_dataset
    )

    test_path = (
        dataset_folder
        / "test.xlsx"
    )

    output_dir = (
        Path(args.output_root)
        / f"rf_{selected_dataset}"
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

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 72)
    print("RANDOM FOREST EVALUATION")
    print("=" * 72)

    print(
        f"Dataset: {selected_dataset}"
    )

    print(
        f"Dataset folder: {dataset_folder}"
    )

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
        feature_names,
    ) = load_all_training_folds(
        dataset_folder=dataset_folder,
        id_col=args.id_col,
        label_col=args.label_col,
        n_folds=args.n_folds,
    )

    # =========================================================================
    # Load independent test set
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
        id_col=args.id_col,
        label_col=args.label_col,
        expected_feature_columns=(
            feature_names
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
            output_dir
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
            f"Overlap saved to: {overlap_path}"
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
        f"{len(feature_names)}"
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

    feature_columns_path = (
        output_dir
        / "feature_columns.txt"
    )

    with open(
        feature_columns_path,
        "w",
        encoding="utf-8",
    ) as feature_file:
        for index, feature in enumerate(
            feature_names,
            start=1,
        ):
            feature_file.write(
                f"{index}\t{feature}\n"
            )

    # =========================================================================
    # Train Random Forest
    # =========================================================================

    print("\n" + "=" * 72)
    print("FITTING RANDOM FOREST")
    print("=" * 72)

    random_forest = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )

    random_forest.fit(
        X_train.to_numpy(),
        y_train,
    )

    # =========================================================================
    # Evaluate on test.xlsx
    # =========================================================================

    print("\n" + "=" * 72)
    print("EVALUATING ON TEST.XLSX")
    print("=" * 72)

    predictions = random_forest.predict(
        X_test.to_numpy()
    )

    probability_matrix = (
        random_forest.predict_proba(
            X_test.to_numpy()
        )
    )

    if 1 not in random_forest.classes_:
        raise ValueError(
            f"Pacemaker class 1 is missing from "
            f"Random Forest classes: "
            f"{random_forest.classes_}"
        )

    positive_class_index = list(
        random_forest.classes_
    ).index(1)

    pacemaker_probabilities = (
        probability_matrix[
            :,
            positive_class_index,
        ]
    )

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
        tn / (tn + fp)
        if (tn + fp) > 0
        else float("nan")
    )

    pacemaker_accuracy = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else float("nan")
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else float("nan")
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else float("nan")
    )

    total_samples = int(
        tn + fp + fn + tp
    )

    false_positive_fraction = (
        fp / total_samples
        if total_samples > 0
        else float("nan")
    )

    false_negative_fraction = (
        fn / total_samples
        if total_samples > 0
        else float("nan")
    )

    # =========================================================================
    # Save predictions
    # =========================================================================

    prediction_dataframe = pd.DataFrame(
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
        output_dir
        / "rf_test_predictions.xlsx"
    )

    prediction_dataframe.to_excel(
        prediction_path,
        index=False,
    )

    # =========================================================================
    # Save classification report
    # =========================================================================

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=[
            NEGATIVE_LABEL_NAME,
            POSITIVE_LABEL_NAME,
        ],
        output_dict=True,
        zero_division=0,
    )

    report_dataframe = (
        pd.DataFrame(report).transpose()
    )

    report_path = (
        output_dir
        / "classification_report.csv"
    )

    report_dataframe.to_csv(
        report_path
    )

    # =========================================================================
    # Save confusion matrices
    # =========================================================================

    raw_confusion_path = (
        output_dir
        / "confusion_matrix_test.png"
    )

    normalized_confusion_path = (
        output_dir
        / "confusion_matrix_test_normalized.png"
    )

    plot_confusion_matrix(
        cm=cm,
        save_path=raw_confusion_path,
        title=(
            f"Random Forest — "
            f"{selected_dataset.upper()} Test Set"
        ),
        normalized=False,
    )

    plot_confusion_matrix(
        cm=cm,
        save_path=normalized_confusion_path,
        title=(
            f"Random Forest — "
            f"{selected_dataset.upper()} Test Set "
            f"(Normalized)"
        ),
        normalized=True,
    )

    # =========================================================================
    # Feature importance
    # =========================================================================

    feature_importance_dataframe = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": (
                    random_forest.feature_importances_
                ),
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    feature_importance_csv_path = (
        output_dir
        / "feature_importance.csv"
    )

    feature_importance_plot_path = (
        output_dir
        / "feature_importance.png"
    )

    feature_importance_dataframe.to_csv(
        feature_importance_csv_path,
        index=False,
    )

    plot_feature_importance(
        importance_dataframe=(
            feature_importance_dataframe
        ),
        save_path=(
            feature_importance_plot_path
        ),
        top_k=args.top_k,
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
            len(feature_names)
        ),
        "n_estimators": int(
            args.n_estimators
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
        output_dir
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
RANDOM FOREST INDEPENDENT TEST RESULTS
========================================================================

Dataset              : {selected_dataset}
Training folder      : {dataset_folder}
Independent test file: {test_path}

The Random Forest was trained once on fold1.xlsx through fold5.xlsx.
Final evaluation was performed only on test.xlsx.

DATA
------------------------------------------------------------------------
Training samples     : {len(y_train)}
Test samples         : {len(y_test)}
Number of features   : {len(feature_names)}
Number of trees      : {args.n_estimators}

FINAL TEST METRICS
------------------------------------------------------------------------
Accuracy               : {accuracy:.4f}
F1 Score               : {f1:.4f}
AUC-ROC                : {auc_roc:.4f}
AUC-PR                 : {auc_pr:.4f}
No Event Accuracy      : {no_event_accuracy:.4f}
Pacemaker Accuracy     : {pacemaker_accuracy:.4f}
False Positive Rate    : {false_positive_rate * 100:.2f}%
False Negative Rate    : {false_negative_rate * 100:.2f}%
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
{prediction_path}

Classification report:
{report_path}

Metrics:
{metrics_json_path}

Raw confusion matrix:
{raw_confusion_path}

Normalized confusion matrix:
{normalized_confusion_path}

Feature-importance table:
{feature_importance_csv_path}

Feature-importance plot:
{feature_importance_plot_path}
"""

    summary_path = (
        output_dir
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
    # Final terminal output
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
        f"{output_dir}"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Random Forest baseline using five predefined "
            "training folds and test.xlsx."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "tum",
            "lmu",
            "merged",
        ],
        default="lmu",
        help=(
            "Dataset folder to use. The model is trained on "
            "fold1.xlsx through fold5.xlsx and tested on test.xlsx."
        ),
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/ubuntu/TAVI_final/dataset/construct/dataset_splits",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="results",
    )

    parser.add_argument(
        "--id_col",
        type=str,
        default="ID",
    )

    parser.add_argument(
        "--label_col",
        type=str,
        default="LABEL",
    )

    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    arguments = parser.parse_args()

    main(arguments)