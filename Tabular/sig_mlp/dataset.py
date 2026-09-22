#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

CLASS_MAPPING = {
    "no event": 0, "no_event": 0, "noevent": 0, "0": 0,
    "pacer": 1, "pacemaker": 1, "1": 1,
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
    return CLASS_MAPPING.get(str(value).strip().lower())

def normalise_sex(series: pd.Series) -> pd.Series:
    # Same behaviour as the combined-model dataset utility.
    numeric = pd.to_numeric(series, errors="coerce")
    observed = set(numeric.dropna().unique().tolist())
    if observed and observed.issubset({0, 1}):
        return numeric.astype(float)
    if observed and observed.issubset({1, 2}):
        return numeric.map({1: 0.0, 2: 1.0})

    text = series.astype(str).str.strip().str.lower()
    mapped = text.map({
        "m": 0.0, "male": 0.0,
        "f": 1.0, "female": 1.0,
        "0": 0.0, "1": 1.0, "2": 1.0,
    })
    return mapped.where(mapped.notna(), numeric)

def get_fold_path(dataset_folder: Path, fold_number: int) -> Path:
    candidates = [
        dataset_folder / f"fold{fold_number}.xlsx",
        dataset_folder / f"fold_{fold_number}.xlsx",
    ]
    existing = [p for p in candidates if p.exists()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise ValueError(f"Multiple files found for fold {fold_number}: {existing}")
    raise FileNotFoundError(
        f"Could not find fold{fold_number}.xlsx or fold_{fold_number}.xlsx "
        f"inside {dataset_folder}"
    )

def load_excel_rows(
    excel_paths: Sequence[Path],
    id_column: str,
    label_column: str,
    expected_feature_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    if not excel_paths:
        raise ValueError("No Excel files supplied.")

    frames = []
    for path in excel_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Excel file does not exist: {path}")
        frame = pd.read_excel(path).copy()
        missing = {id_column, label_column} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        frames.append(frame)

    df = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    if "SEX" in df.columns:
        df["SEX"] = normalise_sex(df["SEX"])

    df["_LABEL_NUMERIC"] = df[label_column].map(normalise_label)
    df["_NORMALISED_ID"] = df[id_column].map(normalise_id)

    if df["_LABEL_NUMERIC"].isna().any():
        bad = df.loc[df["_LABEL_NUMERIC"].isna(), label_column].astype(str).unique().tolist()
        raise ValueError(f"Unsupported/missing labels: {bad[:20]}")
    if df["_NORMALISED_ID"].isna().any():
        raise ValueError("Missing or invalid IDs were found.")

    duplicate_mask = df["_NORMALISED_ID"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_ids = df.loc[duplicate_mask, "_NORMALISED_ID"].drop_duplicates().tolist()
        raise ValueError(f"Duplicate patient IDs found: {duplicate_ids[:20]}")

    df["_LABEL_NUMERIC"] = df["_LABEL_NUMERIC"].astype(int)

    excluded = {id_column, label_column, "_LABEL_NUMERIC", "_NORMALISED_ID"}
    if expected_feature_columns is None:
        feature_columns = [c for c in df.columns if c not in excluded]
    else:
        feature_columns = list(expected_feature_columns)
        missing_features = [c for c in feature_columns if c not in df.columns]
        if missing_features:
            raise ValueError(
                "Dataset is missing features required by the training schema:\n"
                f"{missing_features}"
            )
        extras = [c for c in df.columns if c not in excluded and c not in feature_columns]
        if extras:
            raise ValueError(
                "Dataset contains additional model feature columns not present in "
                f"the training schema:\n{extras}"
            )

    raw = (
        df[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .reset_index(drop=True)
    )
    y = df["_LABEL_NUMERIC"].to_numpy(dtype=np.int64)
    ids = df["_NORMALISED_ID"].astype(str).to_numpy()
    return raw, y, ids, feature_columns

def fit_preprocessor(
    raw_train: pd.DataFrame,
    columns: Sequence[str],
) -> Tuple[Dict, np.ndarray]:
    # This mirrors the combined-model preprocessing exactly:
    # train-only median -> train-only StandardScaler -> append missingness mask.
    columns = list(columns)
    train = raw_train[columns].copy()

    missing_mask = train.isna().astype(np.float32).to_numpy()
    medians = train.median(axis=0, skipna=True).fillna(0.0)
    imputed_train = train.fillna(medians)

    scaler = StandardScaler()
    scaler.fit(imputed_train)
    standardized = scaler.transform(imputed_train).astype(np.float32)

    transformed = np.concatenate(
        [standardized, missing_mask],
        axis=1,
    ).astype(np.float32)

    if not np.isfinite(transformed).all():
        raise FloatingPointError("Non-finite values remain after preprocessing.")

    state = {
        "original_columns": columns,
        "medians": {k: float(v) for k, v in medians.items()},
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "missing_imputation": "training-set median",
        "scaler_fit": "training folds only",
        "missingness_mask": True,
        "value_feature_count": len(columns),
        "mask_feature_count": len(columns),
        "model_input_dim": 2 * len(columns),
        "reported_feature_names": columns,
    }
    return state, transformed

def transform_tabular(raw_data: pd.DataFrame, state: Dict) -> np.ndarray:
    """
    Apply the preprocessing that was actually used when the model was trained.

    Supports both:
      - no-mask models: standardized values only
      - mask models: standardized values + missingness masks
    """
    columns = list(state["original_columns"])
    aligned = raw_data.reindex(columns=columns).copy()

    missing_mask = aligned.isna().astype(np.float32).to_numpy()
    medians = pd.Series(state["medians"], index=columns, dtype=np.float64)
    imputed = aligned.fillna(medians)

    mean = np.asarray(state["scaler_mean"], dtype=np.float64)
    scale = np.asarray(state["scaler_scale"], dtype=np.float64)
    scale = np.where(scale == 0, 1.0, scale)

    standardized = (
        (imputed.to_numpy(dtype=np.float64) - mean) / scale
    ).astype(np.float32)

    use_missingness_mask = bool(
        state.get("missingness_mask", False)
    )

    if use_missingness_mask:
        transformed = np.concatenate(
            [standardized, missing_mask],
            axis=1,
        ).astype(np.float32)
    else:
        transformed = standardized.astype(np.float32)

    if not np.isfinite(transformed).all():
        raise FloatingPointError(
            "Non-finite values remain after preprocessing."
        )

    expected_width = int(
        state.get("model_input_dim", transformed.shape[1])
    )

    if transformed.shape[1] != expected_width:
        raise ValueError(
            "Preprocessing width mismatch: saved preprocessor expects "
            f"{expected_width} model inputs, but transformation produced "
            f"{transformed.shape[1]}. "
            f"missingness_mask={use_missingness_mask}."
        )

    return transformed

def save_preprocessor(state: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")

def load_preprocessor(path: Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def compute_binary_class_weights(y_train: np.ndarray) -> np.ndarray:
    classes = np.unique(y_train)
    if set(classes.tolist()) != {0, 1}:
        raise RuntimeError(f"Training data must contain classes 0 and 1; found {classes}.")
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    result = np.ones(2, dtype=np.float32)
    for class_id, weight in zip(classes, weights):
        result[int(class_id)] = float(weight)
    return result

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, ids: np.ndarray):
        self.X = torch.as_tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y, dtype=np.int64), dtype=torch.long)
        self.ids = np.asarray(ids).astype(str)
        if not (len(self.X) == len(self.y) == len(self.ids)):
            raise ValueError("X, y and IDs must have the same length.")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.ids[index]

def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
