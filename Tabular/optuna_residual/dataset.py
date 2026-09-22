#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset utilities for the residual tabular MLP WITHOUT missingness masks.

Preprocessing policy
--------------------
For every fold model:

1. Load the original clinical variables from the predefined Excel files.
2. Fit per-feature medians ONLY on the four training folds.
3. Impute missing train/validation/test values with those training-fold medians.
4. Fit StandardScaler ONLY on the imputed four training folds.
5. Feed ONLY the standardized clinical values to the MLP.

No missingness-mask columns are created or appended.

This keeps validation/test preprocessing completely independent from their
own statistics and makes the no-mask experiment directly comparable with the
previous residual-MLP experiment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset


CLASS_MAPPING = {
    "no event": 0,
    "no_event": 0,
    "noevent": 0,
    "0": 0,
    "pacer": 1,
    "pacemaker": 1,
    "1": 1,
}


def normalise_id(value) -> Optional[str]:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass

    return text


def normalise_label(value) -> Optional[int]:
    if pd.isna(value):
        return None

    try:
        numeric = int(float(value))
        if numeric in (0, 1):
            return numeric
    except (TypeError, ValueError):
        pass

    return CLASS_MAPPING.get(
        str(value).strip().lower()
    )


def normalise_sex(
    series: pd.Series,
) -> pd.Series:
    """
    Same SEX normalization used in the combined-model preprocessing.
    """
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    observed = set(
        numeric.dropna().unique().tolist()
    )

    if observed and observed.issubset({0, 1}):
        return numeric.astype(float)

    if observed and observed.issubset({1, 2}):
        return numeric.map(
            {
                1: 0.0,
                2: 1.0,
            }
        )

    text = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    mapped = text.map(
        {
            "m": 0.0,
            "male": 0.0,
            "f": 1.0,
            "female": 1.0,
            "0": 0.0,
            "1": 1.0,
            "2": 1.0,
        }
    )

    return mapped.where(
        mapped.notna(),
        numeric,
    )


def get_fold_path(
    dataset_folder: Path,
    fold_number: int,
) -> Path:
    candidates = [
        dataset_folder / f"fold{fold_number}.xlsx",
        dataset_folder / f"fold_{fold_number}.xlsx",
    ]

    existing = [
        path
        for path in candidates
        if path.exists()
    ]

    if len(existing) == 1:
        return existing[0]

    if len(existing) > 1:
        raise ValueError(
            f"Multiple files were found for fold {fold_number}: "
            f"{existing}"
        )

    raise FileNotFoundError(
        f"Could not find fold{fold_number}.xlsx in "
        f"{dataset_folder}"
    )


def load_excel_file(
    excel_path: Path,
    id_column: str,
    label_column: str,
    expected_feature_columns: Optional[
        Sequence[str]
    ] = None,
) -> Dict[str, Any]:
    """
    Load one predefined fold/test Excel file.

    Returns RAW clinical values with NaNs preserved. Preprocessing is performed
    separately after the train/validation split is known.
    """
    excel_path = Path(
        excel_path
    )

    if not excel_path.exists():
        raise FileNotFoundError(
            f"Excel file does not exist: {excel_path}"
        )

    dataframe = pd.read_excel(
        excel_path
    ).copy()

    required = {
        id_column,
        label_column,
    }

    missing_required = (
        required - set(dataframe.columns)
    )

    if missing_required:
        raise ValueError(
            f"{excel_path} is missing required columns: "
            f"{sorted(missing_required)}"
        )

    if "SEX" in dataframe.columns:
        dataframe["SEX"] = normalise_sex(
            dataframe["SEX"]
        )

    dataframe["_LABEL_NUMERIC"] = (
        dataframe[label_column]
        .map(normalise_label)
    )

    dataframe["_NORMALISED_ID"] = (
        dataframe[id_column]
        .map(normalise_id)
    )

    if dataframe["_LABEL_NUMERIC"].isna().any():
        bad_values = (
            dataframe.loc[
                dataframe[
                    "_LABEL_NUMERIC"
                ].isna(),
                label_column,
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unsupported/missing labels in {excel_path}: "
            f"{bad_values[:20]}"
        )

    if dataframe[
        "_NORMALISED_ID"
    ].isna().any():
        raise ValueError(
            f"{excel_path} contains missing/invalid IDs."
        )

    duplicate_mask = (
        dataframe[
            "_NORMALISED_ID"
        ]
        .duplicated(
            keep=False
        )
    )

    if duplicate_mask.any():
        duplicate_ids = (
            dataframe.loc[
                duplicate_mask,
                "_NORMALISED_ID",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"{excel_path} contains duplicate IDs: "
            f"{duplicate_ids[:20]}"
        )

    dataframe[
        "_LABEL_NUMERIC"
    ] = dataframe[
        "_LABEL_NUMERIC"
    ].astype(int)

    excluded = {
        id_column,
        label_column,
        "_LABEL_NUMERIC",
        "_NORMALISED_ID",
    }

    if expected_feature_columns is None:
        feature_columns = [
            column
            for column in dataframe.columns
            if column not in excluded
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
                f"{excel_path} is missing model features: "
                f"{missing_features}"
            )

        extra_features = [
            column
            for column in dataframe.columns
            if (
                column not in excluded
                and column not in feature_columns
            )
        ]

        if extra_features:
            raise ValueError(
                f"{excel_path} contains extra model features "
                f"not present in the training schema: "
                f"{extra_features}"
            )

    if not feature_columns:
        raise ValueError(
            f"No prediction features found in {excel_path}."
        )

    raw_features = (
        dataframe[
            feature_columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .reset_index(
            drop=True
        )
    )

    y = dataframe[
        "_LABEL_NUMERIC"
    ].to_numpy(
        dtype=np.int64
    )

    ids = dataframe[
        "_NORMALISED_ID"
    ].astype(str).to_numpy()

    if set(
        np.unique(y).tolist()
    ) != {0, 1}:
        raise ValueError(
            f"{excel_path} must contain both binary labels. "
            f"Found {np.unique(y).tolist()}."
        )

    return {
        "path": excel_path,
        "raw_X": raw_features,
        "y": y,
        "ids": ids,
        "feature_columns": feature_columns,
    }


def load_predefined_folds(
    dataset_folder: Path,
    number_of_folds: int,
    id_column: str,
    label_column: str,
) -> Tuple[
    List[Dict[str, Any]],
    List[str],
]:
    folds: List[
        Dict[str, Any]
    ] = []

    feature_columns: Optional[
        List[str]
    ] = None

    seen_ids: set[str] = set()

    for fold_number in range(
        1,
        number_of_folds + 1,
    ):
        path = get_fold_path(
            dataset_folder,
            fold_number,
        )

        fold = load_excel_file(
            excel_path=path,
            id_column=id_column,
            label_column=label_column,
            expected_feature_columns=(
                feature_columns
            ),
        )

        if feature_columns is None:
            feature_columns = list(
                fold[
                    "feature_columns"
                ]
            )

        current_ids = set(
            fold[
                "ids"
            ].tolist()
        )

        overlap = (
            seen_ids
            & current_ids
        )

        if overlap:
            raise ValueError(
                "IDs occur in more than one development fold: "
                f"{sorted(overlap)[:20]}"
            )

        seen_ids.update(
            current_ids
        )

        fold[
            "fold_number"
        ] = fold_number

        folds.append(
            fold
        )

    if feature_columns is None:
        raise RuntimeError(
            "No predefined folds were loaded."
        )

    return (
        folds,
        feature_columns,
    )


def concatenate_folds(
    folds: Sequence[
        Dict[str, Any]
    ],
    excluded_fold: Optional[
        int
    ] = None,
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    selected = [
        fold
        for fold in folds
        if (
            excluded_fold is None
            or fold[
                "fold_number"
            ] != excluded_fold
        )
    ]

    if not selected:
        raise ValueError(
            "No folds remain after exclusion."
        )

    raw_X = pd.concat(
        [
            fold[
                "raw_X"
            ]
            for fold in selected
        ],
        axis=0,
        ignore_index=True,
    )

    y = np.concatenate(
        [
            fold[
                "y"
            ]
            for fold in selected
        ],
        axis=0,
    )

    ids = np.concatenate(
        [
            fold[
                "ids"
            ]
            for fold in selected
        ],
        axis=0,
    )

    return (
        raw_X,
        y,
        ids,
    )


def fit_preprocessor(
    raw_train: pd.DataFrame,
    columns: Sequence[str],
) -> Tuple[
    Dict[str, Any],
    np.ndarray,
]:
    """
    NO-MASK preprocessing.

    Fit medians and StandardScaler only on training folds.
    """
    columns = list(
        columns
    )

    train = (
        raw_train[
            columns
        ]
        .copy()
    )

    medians = (
        train.median(
            axis=0,
            skipna=True,
        )
        .fillna(
            0.0
        )
    )

    imputed_train = (
        train.fillna(
            medians
        )
    )

    scaler = StandardScaler()
    scaler.fit(
        imputed_train
    )

    transformed_train = (
        scaler.transform(
            imputed_train
        )
        .astype(
            np.float32
        )
    )

    if not np.isfinite(
        transformed_train
    ).all():
        raise FloatingPointError(
            "Non-finite values remain after training preprocessing."
        )

    state = {
        "original_columns": columns,
        "medians": {
            key: float(value)
            for key, value
            in medians.items()
        },
        "scaler_mean": (
            scaler.mean_
            .astype(float)
            .tolist()
        ),
        "scaler_scale": (
            scaler.scale_
            .astype(float)
            .tolist()
        ),
        "missing_imputation": (
            "training-fold median"
        ),
        "scaler_fit": (
            "training folds only"
        ),
        "missingness_mask": False,
        "value_feature_count": len(
            columns
        ),
        "mask_feature_count": 0,
        "model_input_dim": len(
            columns
        ),
        "reported_feature_names": (
            columns
        ),
    }

    return (
        state,
        transformed_train,
    )


def transform_tabular(
    raw_data: pd.DataFrame,
    preprocessing_state: Dict[
        str,
        Any
    ],
) -> np.ndarray:
    """
    Apply a previously fitted NO-MASK preprocessor.
    """
    columns = list(
        preprocessing_state[
            "original_columns"
        ]
    )

    aligned = (
        raw_data.reindex(
            columns=columns
        )
        .copy()
    )

    medians = pd.Series(
        preprocessing_state[
            "medians"
        ],
        index=columns,
        dtype=np.float64,
    )

    imputed = aligned.fillna(
        medians
    )

    scaler_mean = np.asarray(
        preprocessing_state[
            "scaler_mean"
        ],
        dtype=np.float64,
    )

    scaler_scale = np.asarray(
        preprocessing_state[
            "scaler_scale"
        ],
        dtype=np.float64,
    )

    scaler_scale = np.where(
        scaler_scale == 0,
        1.0,
        scaler_scale,
    )

    transformed = (
        (
            imputed.to_numpy(
                dtype=np.float64
            )
            - scaler_mean
        )
        / scaler_scale
    ).astype(
        np.float32
    )

    if not np.isfinite(
        transformed
    ).all():
        raise FloatingPointError(
            "Non-finite values remain after preprocessing."
        )

    return transformed


def save_preprocessor(
    state: Dict[str, Any],
    path: Path,
) -> None:
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            state,
            handle,
            indent=2,
        )


def compute_binary_class_weights(
    y_train: np.ndarray,
) -> np.ndarray:
    classes = np.unique(
        y_train
    )

    if set(
        classes.tolist()
    ) != {0, 1}:
        raise RuntimeError(
            f"Training data must contain both classes. "
            f"Found {classes.tolist()}."
        )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    result = np.ones(
        2,
        dtype=np.float32,
    )

    for class_id, weight in zip(
        classes,
        weights,
    ):
        result[
            int(class_id)
        ] = float(
            weight
        )

    return result


class TabularDataset(
    Dataset
):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ids: np.ndarray,
    ):
        self.X = torch.as_tensor(
            np.asarray(
                X,
                dtype=np.float32,
            ),
            dtype=torch.float32,
        )

        self.y = torch.as_tensor(
            np.asarray(
                y,
                dtype=np.int64,
            ),
            dtype=torch.long,
        )

        self.ids = np.asarray(
            ids
        ).astype(str)

        if not (
            len(self.X)
            == len(self.y)
            == len(self.ids)
        ):
            raise ValueError(
                "X, y, and IDs must have the same length."
            )

    def __len__(
        self,
    ) -> int:
        return len(
            self.y
        )

    def __getitem__(
        self,
        index: int,
    ):
        return (
            self.X[
                index
            ],
            self.y[
                index
            ],
            self.ids[
                index
            ],
        )


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=(
            batch_size
        ),
        shuffle=(
            shuffle
        ),
        num_workers=(
            num_workers
        ),
        pin_memory=(
            torch.cuda.is_available()
        ),
        persistent_workers=(
            num_workers > 0
        ),
        drop_last=False,
    )
