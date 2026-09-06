
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TabFM evaluation using five predefined training folds
and one independent test set.

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

1. Select "tum", "lmu", or "merged".
2. Load fold1.xlsx through fold5.xlsx.
3. Concatenate all five folds into one training set.
4. Fit one TabFM classifier.
5. Evaluate only on test.xlsx.
6. Save predictions, metrics, classification report,
   and confusion matrices.
"""

import argparse
import gc
import json
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch

from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
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

MISSING_CATEGORY_VALUE = "__MISSING__"


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# =============================================================================
# TABFM
# =============================================================================

def load_tabfm_model(backend: str):
    backend = backend.lower().strip()

    if backend != "pytorch":
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            "Only the 'pytorch' backend is supported."
        )

    from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. TabFM would run on CPU.\n"
            f"PyTorch version: {torch.__version__}\n"
            f"PyTorch CUDA version: {torch.version.cuda}\n"
            "Install a CUDA-enabled PyTorch build compatible with your "
            "NVIDIA driver."
        )

    device = "cuda:0"

    print(f"Loading TabFM on {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime used by PyTorch: {torch.version.cuda}")

    model = tabfm_v1_0_0.load(
        model_type="classification",
        device=device,
    )

    return model


# =============================================================================
# LABEL HANDLING
# =============================================================================

def normalize_labels(
    labels: pd.Series,
    target_column: str,
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
            f"Unsupported values found in target column "
            f"{target_column!r}: {invalid_values}"
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
        f"fold_{fold_number}.xlsx inside:\n"
        f"{dataset_folder}"
    )


# =============================================================================
# DATA LOADING
# =============================================================================

def load_excel_dataset(
    excel_path: Path,
    sheet_name,
    target_column: str,
    id_column: str,
    expected_feature_columns: Sequence[str] | None = None,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    np.ndarray,
    List[str],
]:
    """
    Load one fold file or test.xlsx.

    All columns except the target and ID columns are used as features.
    """
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Excel file not found: {excel_path}"
        )

    print(f"\nReading dataset from: {excel_path}")

    dataframe = pd.read_excel(
        excel_path,
        sheet_name=sheet_name,
        engine="openpyxl",
    )

    dataframe = dataframe.dropna(
        how="all"
    ).reset_index(drop=True)

    missing_required_columns = sorted(
        {
            target_column,
            id_column,
        }
        - set(dataframe.columns)
    )

    if missing_required_columns:
        raise ValueError(
            f"{excel_path} is missing required columns: "
            f"{missing_required_columns}"
        )

    missing_target_mask = dataframe[
        target_column
    ].isna()

    number_missing_targets = int(
        missing_target_mask.sum()
    )

    if number_missing_targets > 0:
        print(
            f"Removing {number_missing_targets} rows because "
            f"{target_column!r} is missing."
        )

        dataframe = dataframe.loc[
            ~missing_target_mask
        ].reset_index(drop=True)

    if dataframe[id_column].isna().any():
        missing_id_rows = dataframe.index[
            dataframe[id_column].isna()
        ].tolist()

        raise ValueError(
            f"{excel_path} contains missing IDs in rows: "
            f"{missing_id_rows[:20]}"
        )

    duplicate_mask = dataframe[
        id_column
    ].duplicated(keep=False)

    if duplicate_mask.any():
        duplicate_ids = (
            dataframe.loc[
                duplicate_mask,
                id_column,
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{excel_path} contains duplicate IDs: "
            f"{duplicate_ids[:20]}"
        )

    dataframe[target_column] = normalize_labels(
        dataframe[target_column],
        target_column,
    )

    dataframe = dataframe.dropna(
        subset=[target_column]
    ).reset_index(drop=True)

    dataframe[target_column] = dataframe[
        target_column
    ].astype(int)

    available_classes = sorted(
        dataframe[target_column]
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
                target_column,
                id_column,
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
                f"{excel_path} is missing features used by "
                f"the training folds:\n{missing_features}"
            )

        extra_columns = [
            column
            for column in dataframe.columns
            if column not in {
                target_column,
                id_column,
                *feature_columns,
            }
        ]

        if extra_columns:
            print(
                f"Extra columns in {excel_path.name} "
                f"will be ignored:"
            )

            for column in extra_columns:
                print(f"  - {column}")

    if not feature_columns:
        raise ValueError(
            f"No feature columns remain in {excel_path}."
        )

    X = dataframe[
        feature_columns
    ].copy()

    y = dataframe[
        target_column
    ].copy()

    ids = dataframe[
        id_column
    ].astype(str).to_numpy()

    X.columns = X.columns.astype(str)

    print(
        f"Samples: {len(dataframe)} | "
        f"Features: {X.shape[1]} | "
        f"Class counts: "
        f"{y.value_counts().sort_index().to_dict()}"
    )

    return (
        X,
        y,
        ids,
        feature_columns,
    )


def load_all_training_folds(
    dataset_folder: Path,
    sheet_name,
    target_column: str,
    id_column: str,
    number_of_folds: int,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    np.ndarray,
    List[str],
]:
    """
    Load fold1.xlsx through fold5.xlsx and concatenate them.
    """
    feature_frames: List[pd.DataFrame] = []
    target_series: List[pd.Series] = []
    id_arrays: List[np.ndarray] = []

    feature_columns: List[str] | None = None
    previously_seen_ids: set[str] = set()

    for fold_number in range(
        1,
        number_of_folds + 1,
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
            sheet_name=sheet_name,
            target_column=target_column,
            id_column=id_column,
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
                f"Overlap detected in fold {fold_number}: "
                f"{overlapping_ids[:20]}"
            )

        previously_seen_ids.update(
            current_ids
        )

        feature_frames.append(X_fold)
        target_series.append(y_fold)
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

    y_train = pd.concat(
        target_series,
        axis=0,
        ignore_index=True,
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
# FEATURE PREPARATION
# =============================================================================

def prepare_train_test_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Prepare train and test features without leaking information from
    test.xlsx into training.

    Numerical columns:
      - Convert invalid values to NaN
      - Replace infinity with NaN
      - Fill missing values using the training-set median
      - Use 0 if the entire training column is missing

    Categorical columns:
      - Convert values to strings
      - Replace missing values with an explicit category

    TabFM still performs its own ordinal encoding and scaling during fit().
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and test feature columns are not in the same order."
        )

    for column in X_train.columns:
        train_column = X_train[column]
        test_column = X_test[column]

        if is_datetime64_any_dtype(
            train_column
        ):
            X_train[column] = (
                train_column.astype("string")
                .fillna(MISSING_CATEGORY_VALUE)
                .astype(str)
            )

            X_test[column] = (
                test_column.astype("string")
                .fillna(MISSING_CATEGORY_VALUE)
                .astype(str)
            )

        elif is_bool_dtype(
            train_column
        ):
            X_train[column] = (
                train_column.astype("string")
                .fillna(MISSING_CATEGORY_VALUE)
                .astype(str)
            )

            X_test[column] = (
                test_column.astype("string")
                .fillna(MISSING_CATEGORY_VALUE)
                .astype(str)
            )

        elif is_numeric_dtype(
            train_column
        ):
            train_numeric = pd.to_numeric(
                train_column,
                errors="coerce",
            ).replace(
                [np.inf, -np.inf],
                np.nan,
            )

            test_numeric = pd.to_numeric(
                test_column,
                errors="coerce",
            ).replace(
                [np.inf, -np.inf],
                np.nan,
            )

            fill_value = train_numeric.median()

            if pd.isna(fill_value):
                fill_value = 0.0

            X_train[column] = (
                train_numeric
                .fillna(fill_value)
                .astype(float)
            )

            X_test[column] = (
                test_numeric
                .fillna(fill_value)
                .astype(float)
            )

        else:
            X_train[column] = (
                train_column.astype("string")
                .fillna(MISSING_CATEGORY_VALUE)
                .astype(str)
            )

            X_test[column] = (
                test_column.astype("string")
                .fillna(MISSING_CATEGORY_VALUE)
                .astype(str)
            )

    return X_train, X_test


# =============================================================================
# PLOTTING
# =============================================================================

def plot_confusion_matrix(
    confusion: np.ndarray,
    save_path: Path,
    title: str,
    normalized: bool = False,
) -> None:
    """
    Save a raw-count or row-normalized confusion matrix.
    """
    if normalized:
        row_sums = confusion.sum(
            axis=1,
            keepdims=True,
        )

        matrix_to_plot = np.divide(
            confusion.astype(float),
            row_sums,
            out=np.zeros_like(
                confusion,
                dtype=float,
            ),
            where=row_sums != 0,
        )

        value_format = ".2f"

    else:
        matrix_to_plot = confusion
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
            f"but received {args.dataset!r}."
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
        / f"tabfm_{selected_dataset}"
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

    print("\n" + "=" * 80)
    print("TABFM EVALUATION")
    print("=" * 80)

    print(
        f"Dataset: {selected_dataset}"
    )

    print(
        f"Dataset folder: {dataset_folder}"
    )

    print(
        f"Backend: {args.backend}"
    )

    # =========================================================================
    # Load all five training folds
    # =========================================================================

    print("\n" + "=" * 80)
    print("LOADING FIVE TRAINING FOLDS")
    print("=" * 80)

    (
        X_train,
        y_train,
        train_ids,
        feature_columns,
    ) = load_all_training_folds(
        dataset_folder=dataset_folder,
        sheet_name=args.sheet_name,
        target_column=args.target_column,
        id_column=args.id_column,
        number_of_folds=args.n_folds,
    )

    # =========================================================================
    # Load test.xlsx
    # =========================================================================

    print("\n" + "=" * 80)
    print("LOADING INDEPENDENT TEST SET")
    print("=" * 80)

    (
        X_test,
        y_test,
        test_ids,
        _,
    ) = load_excel_dataset(
        excel_path=test_path,
        sheet_name=args.sheet_name,
        target_column=args.target_column,
        id_column=args.id_column,
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
            f"the five training folds and test.xlsx.\n"
            f"Overlap saved to: {overlap_path}"
        )

    # =========================================================================
    # Prepare features
    # =========================================================================

    X_train, X_test = prepare_train_test_features(
        X_train,
        X_test,
    )

    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)

    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Independent test samples: "
        f"{len(X_test)}"
    )

    print(
        f"Number of features: "
        f"{X_train.shape[1]}"
    )

    print(
        "Training class distribution:"
    )

    print(
        y_train.value_counts()
        .sort_index()
    )

    print(
        "\nTest class distribution:"
    )

    print(
        y_test.value_counts()
        .sort_index()
    )

    feature_path = (
        output_dir
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
    # Load and fit TabFM
    # =========================================================================

    print("\n" + "=" * 80)
    print("LOADING TABFM FOUNDATION MODEL")
    print("=" * 80)

    tabfm_model = load_tabfm_model(
        args.backend
    )

    from tabfm import TabFMClassifier

    print("\n" + "=" * 80)
    print("FITTING TABFM")
    print("=" * 80)

    classifier = TabFMClassifier(
        model=tabfm_model,
        n_estimators=4,
    )

    classifier.fit(
        X_train,
        y_train.to_numpy(),
    )

    # =========================================================================
    # Test evaluation
    # =========================================================================

    print("\n" + "=" * 80)
    print("EVALUATING ON TEST.XLSX")
    print("=" * 80)

    predictions = np.asarray(
        classifier.predict(
            X_test
        )
    ).reshape(-1)

    if len(predictions) != len(y_test):
        raise RuntimeError(
            f"Model returned {len(predictions)} predictions "
            f"for {len(y_test)} test samples."
        )

    y_test_array = y_test.to_numpy()

    accuracy = float(
        accuracy_score(
            y_test_array,
            predictions,
        )
    )

    f1 = float(
        f1_score(
            y_test_array,
            predictions,
            average="binary",
            pos_label=1,
            zero_division=0,
        )
    )

    confusion = confusion_matrix(
        y_test_array,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = confusion.ravel()

    no_event_accuracy = (
        float(
            tn / (tn + fp)
        )
        if (tn + fp) > 0
        else float("nan")
    )

    pacemaker_accuracy = (
        float(
            tp / (tp + fn)
        )
        if (tp + fn) > 0
        else float("nan")
    )

    # =========================================================================
    # Save predictions
    # =========================================================================

    prediction_table = pd.DataFrame(
        {
            "ID": test_ids,
            "True": y_test_array.astype(int),
            "True Label": np.where(
                y_test_array == 1,
                POSITIVE_LABEL_NAME,
                NEGATIVE_LABEL_NAME,
            ),
            "Pred": predictions.astype(int),
            "Predicted Label": np.where(
                predictions == 1,
                POSITIVE_LABEL_NAME,
                NEGATIVE_LABEL_NAME,
            ),
            "Correct": (
                predictions == y_test_array
            ),
        }
    )

    prediction_path = (
        output_dir
        / "tabfm_test_predictions.xlsx"
    )

    prediction_table.to_excel(
        prediction_path,
        index=False,
    )

    # =========================================================================
    # Classification report
    # =========================================================================

    report = classification_report(
        y_test_array,
        predictions,
        labels=[0, 1],
        target_names=[
            NEGATIVE_LABEL_NAME,
            POSITIVE_LABEL_NAME,
        ],
        output_dict=True,
        zero_division=0,
    )

    report_table = (
        pd.DataFrame(report).transpose()
    )

    report_path = (
        output_dir
        / "classification_report.csv"
    )

    report_table.to_csv(
        report_path
    )

    # =========================================================================
    # Confusion matrices
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
        confusion=confusion,
        save_path=raw_confusion_path,
        title=(
            f"TabFM — "
            f"{selected_dataset.upper()} Test Set"
        ),
        normalized=False,
    )

    plot_confusion_matrix(
        confusion=confusion,
        save_path=normalized_confusion_path,
        title=(
            f"TabFM — "
            f"{selected_dataset.upper()} Test Set "
            f"(Normalized)"
        ),
        normalized=True,
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
        "backend": args.backend,
        "training_samples": int(
            len(X_train)
        ),
        "test_samples": int(
            len(X_test)
        ),
        "number_of_features": int(
            X_train.shape[1]
        ),
        "accuracy": accuracy,
        "f1_score": f1,
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
TABFM INDEPENDENT TEST RESULTS
========================================================================

Dataset              : {selected_dataset}
Training folder      : {dataset_folder}
Independent test file: {test_path}
Backend              : {args.backend}

The TabFM classifier was trained once on fold1.xlsx through fold5.xlsx.
Final evaluation was performed only on test.xlsx.

DATA
------------------------------------------------------------------------
Training samples     : {len(X_train)}
Test samples         : {len(X_test)}
Number of features   : {X_train.shape[1]}

FINAL TEST METRICS
------------------------------------------------------------------------
Accuracy          : {accuracy:.4f}
F1 Score          : {f1:.4f}
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

Classification report:
{report_path}

Metrics:
{metrics_json_path}

Raw confusion matrix:
{raw_confusion_path}

Normalized confusion matrix:
{normalized_confusion_path}
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

    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"No Event Accuracy : "
        f"{no_event_accuracy:.4f}"
    )

    print(
        f"Pacemaker Accuracy: "
        f"{pacemaker_accuracy:.4f}"
    )

    print("\nConfusion matrix:")
    print(confusion)

    print(
        f"\nAll results saved to: "
        f"{output_dir}"
    )

    del classifier
    del tabfm_model
    gc.collect()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "TabFM evaluation using five predefined "
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
        default="merged",
        help=(
            "The model is trained on fold1.xlsx through "
            "fold5.xlsx and tested on test.xlsx."
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
        "--sheet_name",
        default=0,
    )

    parser.add_argument(
        "--target_column",
        type=str,
        default="LABEL",
    )

    parser.add_argument(
        "--id_column",
        type=str,
        default="ID",
    )

    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=[
            "pytorch",
        ],
        default="pytorch",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    arguments = parser.parse_args()

    main(arguments)
