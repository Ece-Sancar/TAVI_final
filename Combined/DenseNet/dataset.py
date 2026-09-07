#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset utilities for the combined CT + tabular model.

Experimental protocol
---------------------

For fold model i:

    Training:
        The other four predefined fold Excel files.

    Validation:
        fold{i}.xlsx

    Final testing:
        test.xlsx from the configured test dataset.

Tabular preprocessing
---------------------

The original dataset features are used directly.

Missing values:
    Median imputation fitted ONLY on the four training folds.

Scaling:
    StandardScaler fitted ONLY on the four training folds.

Missingness is represented internally by one binary mask value per original
feature. These masks are appended to the MLP input but are not exposed as
separate clinical feature names and are not reported as separate features.

No additional class balancing/downsampling is performed.
The predefined Excel files determine the patients used in each split.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# CONSTANTS
# =============================================================================

CLASS_MAPPING = {
    "no event": 0,
    "pacer": 1,
    "pacemaker": 1,
}

ID_COLUMN = "ID"
LABEL_COLUMN = "LABEL"


# =============================================================================
# BASIC HELPERS
# =============================================================================

def _normalise_id(value) -> Optional[str]:
    """
    Convert Excel IDs to the same representation used by PNG stems.

    Examples:
        123
        123.0
        "123"

    all become:
        "123"
    """
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


def _normalise_sex(
    series: pd.Series,
) -> pd.Series:
    """
    Preserve the behavior of the original implementation.

    Supported:
        0/1
        1/2
        M/F
        male/female
    """
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    observed = set(
        numeric.dropna()
        .unique()
        .tolist()
    )

    if observed and observed.issubset(
        {0, 1}
    ):
        return numeric.astype(float)

    if observed and observed.issubset(
        {1, 2}
    ):
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


def _normalise_label(
    value,
) -> Optional[int]:
    if pd.isna(value):
        return None

    # Numeric labels
    try:
        numeric = int(float(value))

        if numeric in (0, 1):
            return numeric

    except (TypeError, ValueError):
        pass

    text = str(value).strip().lower()

    return CLASS_MAPPING.get(text)


# =============================================================================
# DATASET
# =============================================================================

class Valve2DDataset(Dataset):
    """
    Dataset returning:

        image
        label
        tabular features
        patient/sample ID
    """

    def __init__(
        self,
        samples,
        tabular_features,
        tabular_columns,
        augment=False,
        target_size=(448, 448),
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
    ):
        self.samples = list(samples)

        self.labels = [
            int(sample["label"])
            for sample in samples
        ]

        self.used_ids_all = [
            sample["id"]
            for sample in samples
        ]

        self.tabular_features = np.asarray(
            tabular_features,
            dtype=np.float32,
        )

        self.tabular_columns = list(
            tabular_columns
        )

        self.augment = bool(augment)

        self.target_size = tuple(
            target_size
        )

        self.image_mean = torch.tensor(
            image_mean,
            dtype=torch.float32,
        ).view(
            3,
            1,
            1,
        )

        self.image_std = torch.tensor(
            image_std,
            dtype=torch.float32,
        ).view(
            3,
            1,
            1,
        )

        if (
            len(self.samples)
            != len(self.tabular_features)
        ):
            raise ValueError(
                "Samples and tabular feature rows "
                "must have identical lengths."
            )

    def __len__(self):
        return len(self.samples)

    def _resize(
        self,
        image,
    ):
        if (
            image.shape[-2:]
            == self.target_size
        ):
            return image

        return F.interpolate(
            image.unsqueeze(0),
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    @staticmethod
    def _augment_grayscale(
        image,
    ):
        """
        Same conservative CT augmentation as the original model.

        No flips.
        No rotations.
        """
        array = (
            image.squeeze(0)
            .cpu()
            .numpy()
        )

        # Contrast
        if np.random.rand() < 0.50:
            contrast = np.random.uniform(
                0.90,
                1.10,
            )

            image_mean = float(
                array.mean()
            )

            array = np.clip(
                (
                    array
                    - image_mean
                )
                * contrast
                + image_mean,
                0.0,
                1.0,
            )

        # Brightness
        if np.random.rand() < 0.35:
            brightness = np.random.uniform(
                0.95,
                1.05,
            )

            array = np.clip(
                array * brightness,
                0.0,
                1.0,
            )

        # Gaussian noise
        if np.random.rand() < 0.30:
            noise_std = np.random.uniform(
                0.0,
                0.01,
            )

            noise = np.random.normal(
                0.0,
                noise_std,
                array.shape,
            )

            array = np.clip(
                array + noise,
                0.0,
                1.0,
            )

        return torch.from_numpy(
            array.astype(np.float32)
        ).unsqueeze(0)

    def __getitem__(
        self,
        index,
    ):
        sample = self.samples[index]

        image = cv2.imread(
            sample["img"],
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise RuntimeError(
                "Failed to read image: "
                f"{sample['img']}"
            )

        image = torch.from_numpy(
            image.astype(
                np.float32
            )
            / 255.0
        ).unsqueeze(0)

        image = self._resize(
            image
        )

        if self.augment:
            image = (
                self._augment_grayscale(
                    image
                )
            )

        # Preserve pretrained DenseNet input format.
        image = image.repeat(
            3,
            1,
            1,
        )

        image = (
            image
            - self.image_mean
        ) / self.image_std

        label = torch.tensor(
            sample["label"],
            dtype=torch.long,
        )

        tabular = torch.from_numpy(
            self.tabular_features[
                index
            ]
        )

        return (
            image,
            label,
            tabular,
            sample["id"],
        )


# =============================================================================
# PREDEFINED-FOLD DATA MODULE
# =============================================================================

class Valve2DDataModule(
    pl.LightningDataModule
):
    """
    DataModule using explicit Excel files.

    train_excel_paths:
        four fold files

    validation_excel_path:
        current held-out fold

    test_excel_path:
        fixed independent test.xlsx
    """

    def __init__(
        self,
        data_root,
        train_excel_paths,
        validation_excel_path,
        test_excel_path,
        batch_size=16,
        num_workers=8,
        target_size=(448, 448),
        seed=42,
        image_mean=(
            0.485,
            0.456,
            0.406,
        ),
        image_std=(
            0.229,
            0.224,
            0.225,
        ),
        artifacts_dir=None,
        test_data_root=None,
    ):
        super().__init__()

        self.data_root = str(
            data_root
        )

        self.test_data_root = str(
            test_data_root
            if test_data_root
            is not None
            else data_root
        )

        self.train_excel_paths = [
            Path(path)
            for path
            in train_excel_paths
        ]

        self.validation_excel_path = Path(
            validation_excel_path
        )

        self.test_excel_path = Path(
            test_excel_path
        )

        self.batch_size = int(
            batch_size
        )

        self.num_workers = int(
            num_workers
        )

        self.target_size = tuple(
            target_size
        )

        self.seed = int(seed)

        self.image_mean = tuple(
            image_mean
        )

        self.image_std = tuple(
            image_std
        )

        self.artifacts_dir = (
            str(artifacts_dir)
            if artifacts_dir
            is not None
            else None
        )

        self.class_weights = None

        self.tabular_columns = None
        self.model_input_dim = None
        self.preprocessing_state = None

    # =========================================================================
    # Image indexing
    # =========================================================================

    @staticmethod
    def _build_png_index(
        data_root: str,
    ) -> Dict[str, str]:
        root = Path(
            data_root
        )

        if not root.is_dir():
            raise FileNotFoundError(
                "Image directory does not exist: "
                f"{root}"
            )

        # Preserve original behavior:
        # PNG files from the given image root.
        png_by_id = {}

        for path in root.iterdir():
            if (
                path.is_file()
                and path.suffix.lower()
                == ".png"
            ):
                png_by_id[
                    _normalise_id(path.stem)
                ] = str(path)

        if not png_by_id:
            raise RuntimeError(
                "No PNG images were found in: "
                f"{root}"
            )

        return png_by_id

    # =========================================================================
    # Excel loading
    # =========================================================================

    def _load_excel_rows(
        self,
        excel_paths: Sequence[Path],
        data_root: str,
        expected_columns: Optional[
            Sequence[str]
        ] = None,
    ) -> Tuple[
        List[Dict],
        pd.DataFrame,
        List[str],
    ]:
        """
        Load one or more Excel files and find corresponding PNGs.
        """
        if not excel_paths:
            raise ValueError(
                "No Excel files supplied."
            )

        frames = []

        for excel_path in excel_paths:
            excel_path = Path(
                excel_path
            )

            if not excel_path.is_file():
                raise FileNotFoundError(
                    "Excel file does not exist: "
                    f"{excel_path}"
                )

            frame = pd.read_excel(
                excel_path
            ).copy()

            required = {
                ID_COLUMN,
                LABEL_COLUMN,
            }

            missing = required.difference(
                frame.columns
            )

            if missing:
                raise ValueError(
                    f"{excel_path} is missing "
                    f"required columns: "
                    f"{sorted(missing)}"
                )

            frames.append(
                frame
            )

        dataframe = pd.concat(
            frames,
            axis=0,
            ignore_index=True,
            sort=False,
        )

        if "SEX" in dataframe.columns:
            dataframe["SEX"] = (
                _normalise_sex(
                    dataframe["SEX"]
                )
            )

        dataframe["_LABEL_NUMERIC"] = (
            dataframe[LABEL_COLUMN]
            .map(_normalise_label)
        )

        dataframe = dataframe.loc[
            dataframe[
                "_LABEL_NUMERIC"
            ].notna()
        ].copy()

        dataframe[
            "_LABEL_NUMERIC"
        ] = dataframe[
            "_LABEL_NUMERIC"
        ].astype(int)

        # Preserve the old duplicate-ID behavior.
        dataframe[
            "_NORMALISED_ID"
        ] = dataframe[
            ID_COLUMN
        ].map(
            _normalise_id
        )

        dataframe = dataframe.loc[
            dataframe[
                "_NORMALISED_ID"
            ].notna()
        ].copy()

        dataframe = (
            dataframe.drop_duplicates(
                subset=[
                    "_NORMALISED_ID"
                ],
                keep="first",
            )
            .reset_index(
                drop=True
            )
        )

        excluded = {
            ID_COLUMN,
            LABEL_COLUMN,
            "_LABEL_NUMERIC",
            "_NORMALISED_ID",
        }

        if expected_columns is None:
            tabular_columns = [
                column
                for column
                in dataframe.columns
                if column
                not in excluded
            ]

        else:
            tabular_columns = list(
                expected_columns
            )

            missing_features = [
                column
                for column
                in tabular_columns
                if column
                not in dataframe.columns
            ]

            if missing_features:
                raise ValueError(
                    "Dataset is missing tabular "
                    "features required by the "
                    "training data:\n"
                    f"{missing_features}"
                )

        if not tabular_columns:
            raise ValueError(
                "No tabular feature columns "
                "were found."
            )

        png_by_id = (
            self._build_png_index(
                data_root
            )
        )

        samples = []
        raw_rows = []

        for row_idx, row in (
            dataframe.iterrows()
        ):
            sample_id = row[
                "_NORMALISED_ID"
            ]

            if (
                sample_id
                not in png_by_id
            ):
                continue

            samples.append(
                {
                    "id": sample_id,
                    "img": (
                        png_by_id[
                            sample_id
                        ]
                    ),
                    "label": int(
                        row[
                            "_LABEL_NUMERIC"
                        ]
                    ),
                }
            )

            raw_rows.append(
                row_idx
            )

        if not samples:
            raise RuntimeError(
                "No matching Excel rows and "
                "PNG files were found."
            )

        raw_tabular = (
            dataframe.loc[
                raw_rows,
                tabular_columns,
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .reset_index(
                drop=True
            )
        )

        return (
            samples,
            raw_tabular,
            tabular_columns,
        )

    # =========================================================================
    # Tabular preprocessing
    # =========================================================================

    @staticmethod
    def _fit_preprocessor(
        raw_train: pd.DataFrame,
        columns: Sequence[str],
    ) -> Tuple[
        Dict,
        np.ndarray,
    ]:
        """
        Fit preprocessing using ONLY the four training folds.

        For every original clinical feature:
          1. missing values are imputed with the training-fold median;
          2. the imputed value is standardized with a StandardScaler fitted
             only on the training folds;
          3. a binary missingness mask is appended internally to the MLP input.

        The mask columns are NOT exposed as separate feature names. Therefore
        feature-importance outputs still contain only the original Excel
        feature names.
        """
        columns = list(columns)
        train = raw_train[columns].copy()

        # Missingness is measured BEFORE imputation.
        missing_mask = (
            train.isna()
            .astype(np.float32)
            .to_numpy()
        )

        medians = (
            train.median(axis=0, skipna=True)
            .fillna(0.0)
        )

        imputed_train = train.fillna(medians)

        scaler = StandardScaler()
        scaler.fit(imputed_train)

        standardized_values = (
            scaler.transform(imputed_train)
            .astype(np.float32)
        )

        # Internal MLP input: [standardized values | missingness masks].
        transformed_train = np.concatenate(
            [standardized_values, missing_mask],
            axis=1,
        ).astype(np.float32)

        if not np.isfinite(transformed_train).all():
            raise FloatingPointError(
                "Non-finite values remain after train preprocessing."
            )

        state = {
            "original_columns": columns,
            "medians": {
                key: float(value)
                for key, value in medians.items()
            },
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

        return state, transformed_train

    @staticmethod
    def transform_tabular(
        raw_data: pd.DataFrame,
        preprocessing_state: Dict,
    ) -> np.ndarray:
        """
        Apply one fold's training-only preprocessing.

        The returned tensor contains standardized imputed values followed by
        binary missingness masks. The original raw feature names remain the
        only reported clinical features.
        """
        columns = list(preprocessing_state["original_columns"])

        aligned = raw_data.reindex(columns=columns).copy()

        missing_mask = (
            aligned.isna()
            .astype(np.float32)
            .to_numpy()
        )

        medians = pd.Series(
            preprocessing_state["medians"],
            index=columns,
            dtype=np.float64,
        )

        imputed = aligned.fillna(medians)

        scaler_mean = np.asarray(
            preprocessing_state["scaler_mean"],
            dtype=np.float64,
        )
        scaler_scale = np.asarray(
            preprocessing_state["scaler_scale"],
            dtype=np.float64,
        )
        scaler_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)

        standardized_values = (
            (imputed.to_numpy(dtype=np.float64) - scaler_mean)
            / scaler_scale
        ).astype(np.float32)

        transformed = np.concatenate(
            [standardized_values, missing_mask],
            axis=1,
        ).astype(np.float32)

        if not np.isfinite(transformed).all():
            raise FloatingPointError(
                "Non-finite values remain after preprocessing."
            )

        return transformed

    # =========================================================================
    # Setup
    # =========================================================================

    def setup(
        self,
        stage=None,
    ):
        (
            train_samples,
            raw_train,
            train_columns,
        ) = self._load_excel_rows(
            excel_paths=(
                self.train_excel_paths
            ),
            data_root=(
                self.data_root
            ),
            expected_columns=None,
        )

        (
            validation_samples,
            raw_validation,
            _,
        ) = self._load_excel_rows(
            excel_paths=[
                self.validation_excel_path
            ],
            data_root=(
                self.data_root
            ),
            expected_columns=(
                train_columns
            ),
        )

        (
            test_samples,
            raw_test,
            _,
        ) = self._load_excel_rows(
            excel_paths=[
                self.test_excel_path
            ],
            data_root=(
                self.test_data_root
            ),
            expected_columns=(
                train_columns
            ),
        )

        (
            preprocessing_state,
            train_features,
        ) = self._fit_preprocessor(
            raw_train=raw_train,
            columns=train_columns,
        )

        validation_features = (
            self.transform_tabular(
                raw_validation,
                preprocessing_state,
            )
        )

        test_features = (
            self.transform_tabular(
                raw_test,
                preprocessing_state,
            )
        )

        self.preprocessing_state = (
            preprocessing_state
        )

        # Only original Excel features are exposed as clinical feature names.
        self.tabular_columns = list(train_columns)
        self.model_input_dim = int(preprocessing_state["model_input_dim"])

        train_labels = np.asarray(
            [
                sample["label"]
                for sample
                in train_samples
            ],
            dtype=np.int64,
        )

        validation_labels = np.asarray(
            [
                sample["label"]
                for sample
                in validation_samples
            ],
            dtype=np.int64,
        )

        test_labels = np.asarray(
            [
                sample["label"]
                for sample
                in test_samples
            ],
            dtype=np.int64,
        )

        if (
            len(
                np.unique(
                    train_labels
                )
            )
            != 2
        ):
            raise RuntimeError(
                "Training folds do not "
                "contain both classes."
            )

        if (
            len(
                np.unique(
                    validation_labels
                )
            )
            != 2
        ):
            raise RuntimeError(
                "Validation fold does not "
                "contain both classes."
            )

        if (
            len(
                np.unique(
                    test_labels
                )
            )
            != 2
        ):
            raise RuntimeError(
                "Independent test set does not "
                "contain both classes."
            )

        # Preserve original class-weight behavior.
        classes = np.unique(
            train_labels
        )

        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=train_labels,
        )

        class_weights = np.ones(
            2,
            dtype=np.float32,
        )

        for (
            class_id,
            weight,
        ) in zip(
            classes,
            weights,
        ):
            class_weights[
                int(class_id)
            ] = float(weight)

        self.class_weights = (
            class_weights
        )

        common_train = dict(
            samples=train_samples,
            tabular_features=(
                train_features
            ),
            tabular_columns=(
                train_columns
            ),
            target_size=(
                self.target_size
            ),
            image_mean=(
                self.image_mean
            ),
            image_std=(
                self.image_std
            ),
        )

        common_validation = dict(
            samples=validation_samples,
            tabular_features=(
                validation_features
            ),
            tabular_columns=(
                train_columns
            ),
            target_size=(
                self.target_size
            ),
            image_mean=(
                self.image_mean
            ),
            image_std=(
                self.image_std
            ),
        )

        common_test = dict(
            samples=test_samples,
            tabular_features=(
                test_features
            ),
            tabular_columns=(
                train_columns
            ),
            target_size=(
                self.target_size
            ),
            image_mean=(
                self.image_mean
            ),
            image_std=(
                self.image_std
            ),
        )

        self.train_dataset = (
            Valve2DDataset(
                augment=True,
                **common_train,
            )
        )

        self.validation_dataset = (
            Valve2DDataset(
                augment=False,
                **common_validation,
            )
        )

        self.test_dataset = (
            Valve2DDataset(
                augment=False,
                **common_test,
            )
        )

        self.train_samples = (
            train_samples
        )

        self.validation_samples = (
            validation_samples
        )

        self.test_samples = (
            test_samples
        )

        self.train_ids = [
            sample["id"]
            for sample
            in train_samples
        ]

        self.validation_ids = [
            sample["id"]
            for sample
            in validation_samples
        ]

        self.test_ids = [
            sample["id"]
            for sample
            in test_samples
        ]

        # -----------------------------------------------------
        # Leakage checks
        # -----------------------------------------------------

        train_set = set(
            self.train_ids
        )

        validation_set = set(
            self.validation_ids
        )

        test_set = set(
            self.test_ids
        )

        train_val_overlap = (
            train_set
            & validation_set
        )

        train_test_overlap = (
            train_set
            & test_set
        )

        val_test_overlap = (
            validation_set
            & test_set
        )

        if train_val_overlap:
            raise RuntimeError(
                "Training and validation "
                "contain overlapping IDs: "
                f"{list(train_val_overlap)[:20]}"
            )

        if train_test_overlap:
            raise RuntimeError(
                "Training and independent test "
                "contain overlapping IDs: "
                f"{list(train_test_overlap)[:20]}"
            )

        if val_test_overlap:
            raise RuntimeError(
                "Validation and independent test "
                "contain overlapping IDs: "
                f"{list(val_test_overlap)[:20]}"
            )

        print()
        print(
            "DataModule split summary"
        )

        print(
            "-" * 70
        )

        print(
            f"Train:      "
            f"{len(self.train_dataset)}"
        )

        print(
            f"Validation: "
            f"{len(self.validation_dataset)}"
        )

        print(
            f"Test:       "
            f"{len(self.test_dataset)}"
        )

        print(
            f"Clinical features:  {len(train_columns)}"
        )
        print(
            f"MLP input width:    {self.model_input_dim} "
            f"({len(train_columns)} values + {len(train_columns)} masks)"
        )

        print(
            "Train label counts: "
            f"{dict(zip(
                *np.unique(
                    train_labels,
                    return_counts=True
                )
            ))}"
        )

        print(
            "Validation label counts: "
            f"{dict(zip(
                *np.unique(
                    validation_labels,
                    return_counts=True
                )
            ))}"
        )

        print(
            "Test label counts: "
            f"{dict(zip(
                *np.unique(
                    test_labels,
                    return_counts=True
                )
            ))}"
        )

        # -----------------------------------------------------
        # Save artifacts
        # -----------------------------------------------------

        if self.artifacts_dir:
            os.makedirs(
                self.artifacts_dir,
                exist_ok=True,
            )

            preprocessor_path = (
                Path(
                    self.artifacts_dir
                )
                / "tabular_preprocessor.json"
            )

            with preprocessor_path.open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    preprocessing_state,
                    handle,
                    indent=2,
                )

            columns_path = (
                Path(
                    self.artifacts_dir
                )
                / "used_tabular_columns.txt"
            )

            columns_path.write_text(
                "\n".join(
                    train_columns
                )
                + "\n",
                encoding="utf-8",
            )

    # =========================================================================
    # Dataloaders
    # =========================================================================

    def train_dataloader(
        self,
    ):
        return DataLoader(
            self.train_dataset,
            batch_size=(
                self.batch_size
            ),
            shuffle=True,
            num_workers=(
                self.num_workers
            ),
            pin_memory=True,
            persistent_workers=(
                self.num_workers > 0
            ),
            drop_last=False,
        )

    def val_dataloader(
        self,
    ):
        return DataLoader(
            self.validation_dataset,
            batch_size=(
                self.batch_size
            ),
            shuffle=False,
            num_workers=(
                self.num_workers
            ),
            pin_memory=True,
            persistent_workers=(
                self.num_workers > 0
            ),
            drop_last=False,
        )

    def test_dataloader(
        self,
    ):
        return DataLoader(
            self.test_dataset,
            batch_size=(
                self.batch_size
            ),
            shuffle=False,
            num_workers=(
                self.num_workers
            ),
            pin_memory=True,
            persistent_workers=(
                self.num_workers > 0
            ),
            drop_last=False,
        )