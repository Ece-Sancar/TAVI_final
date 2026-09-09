#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset


CLASS_MAPPING = {
    "no event": 0,
    "pacer": 1,
    "pacemaker": 1,
}

ID_COLUMN = "ID"
LABEL_COLUMN = "LABEL"


def _normalise_id(value) -> Optional[str]:
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


def _normalise_label(value) -> Optional[int]:
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


# =============================================================================
# DATASET
# =============================================================================

class Valve2DImageDataset(Dataset):

    def __init__(
        self,
        samples,
        augment=False,
        target_size=(448, 448),
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
    ):
        self.samples = list(samples)

        self.labels = [
            int(s["label"])
            for s in self.samples
        ]

        self.used_ids_all = [
            s["id"]
            for s in self.samples
        ]

        # Keep all original parameters unchanged.
        # They are intentionally not used for preprocessing.
        self.augment = bool(
            augment
        )

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

        # Will be set by the DataModule using a global
        # minimum across train + validation + test.
        self.crop_height = None
        self.crop_width = None

    def __len__(self):
        return len(
            self.samples
        )

    def _resize(
        self,
        image,
    ):
        """
        Preserved unchanged for compatibility.

        NOT USED in __getitem__.
        """
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
        Preserved unchanged for compatibility.

        NOT USED in __getitem__.
        """
        array = (
            image.squeeze(0)
            .cpu()
            .numpy()
        )

        if np.random.rand() < 0.50:
            contrast = np.random.uniform(
                0.90,
                1.10,
            )

            m = float(
                array.mean()
            )

            array = np.clip(
                (
                    array
                    - m
                )
                * contrast
                + m,
                0.0,
                1.0,
            )

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
            array.astype(
                np.float32
            )
        ).unsqueeze(0)

    def __getitem__(
        self,
        index,
    ):
        sample = self.samples[
            index
        ]

        image = cv2.imread(
            sample["img"],
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise RuntimeError(
                f"Failed to read image: "
                f"{sample['img']}"
            )

        # =============================================================
        # CENTER CROP ONLY
        # =============================================================
        #
        # No resizing.
        # No interpolation.
        #
        # Only the outer pixels are removed so that every image
        # has exactly the same dimensions.
        # =============================================================

        if (
            self.crop_height is None
            or self.crop_width is None
        ):
            raise RuntimeError(
                "Crop size has not been set by "
                "Valve2DImageDataModule."
            )

        height, width = image.shape

        if (
            height < self.crop_height
            or width < self.crop_width
        ):
            raise RuntimeError(
                f"Image {sample['img']} has size "
                f"{height}x{width}, which is smaller "
                f"than configured crop "
                f"{self.crop_height}x{self.crop_width}."
            )

        top = (
            height
            - self.crop_height
        ) // 2

        left = (
            width
            - self.crop_width
        ) // 2

        image = image[
            top:
            top + self.crop_height,
            left:
            left + self.crop_width,
        ]

        # =============================================================
        # NO INTENSITY PREPROCESSING
        # =============================================================
        #
        # Original uint8:
        #     0 ... 255
        #
        # becomes float32:
        #     0.0 ... 255.0
        #
        # NO /255
        # NO normalization
        # NO augmentation
        # =============================================================

        image = torch.from_numpy(
            image.astype(
                np.float32
            )
        ).unsqueeze(0)

        # =============================================================
        # GRAYSCALE -> 3 CHANNELS
        # =============================================================
        #
        # Required by the image backbone.
        # Pixel intensities are unchanged.
        # =============================================================

        image = image.repeat(
            3,
            1,
            1,
        )

        label = torch.tensor(
            sample["label"],
            dtype=torch.long,
        )

        return (
            image,
            label,
            sample["id"],
        )


# =============================================================================
# DATA MODULE
# =============================================================================

class Valve2DImageDataModule(
    pl.LightningDataModule
):

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
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
        artifacts_dir=None,
        test_data_root=None,
    ):
        super().__init__()

        self.data_root = str(
            data_root
        )

        self.test_data_root = str(
            test_data_root
            if test_data_root is not None
            else data_root
        )

        self.train_excel_paths = [
            Path(p)
            for p in train_excel_paths
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

        self.seed = int(
            seed
        )

        self.image_mean = tuple(
            image_mean
        )

        self.image_std = tuple(
            image_std
        )

        self.artifacts_dir = (
            str(artifacts_dir)
            if artifacts_dir is not None
            else None
        )

        self.class_weights = None

    # =========================================================================
    # PNG INDEX
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
                f"Image directory does not exist: "
                f"{root}"
            )

        png_by_id = {}

        for path in root.iterdir():

            if (
                path.is_file()
                and path.suffix.lower() == ".png"
            ):

                sid = _normalise_id(
                    path.stem
                )

                if sid is not None:
                    png_by_id[
                        sid
                    ] = str(path)

        if not png_by_id:
            raise RuntimeError(
                f"No PNG images were found in: "
                f"{root}"
            )

        return png_by_id

    # =========================================================================
    # EXCEL LOADING
    # =========================================================================

    def _load_excel_rows(
        self,
        excel_paths: Sequence[Path],
        data_root: str,
    ) -> List[Dict]:

        frames = []

        for excel_path in excel_paths:

            excel_path = Path(
                excel_path
            )

            if not excel_path.is_file():
                raise FileNotFoundError(
                    f"Excel file does not exist: "
                    f"{excel_path}"
                )

            frame = pd.read_excel(
                excel_path
            ).copy()

            missing = {
                ID_COLUMN,
                LABEL_COLUMN,
            }.difference(
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

        dataframe[
            "_LABEL_NUMERIC"
        ] = dataframe[
            LABEL_COLUMN
        ].map(
            _normalise_label
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
        ].astype(
            int
        )

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
            dataframe
            .drop_duplicates(
                subset=[
                    "_NORMALISED_ID"
                ],
                keep="first",
            )
            .reset_index(
                drop=True
            )
        )

        png_by_id = (
            self._build_png_index(
                data_root
            )
        )

        samples = []

        for _, row in (
            dataframe.iterrows()
        ):

            sid = row[
                "_NORMALISED_ID"
            ]

            if sid in png_by_id:

                samples.append(
                    {
                        "id": sid,
                        "img": (
                            png_by_id[
                                sid
                            ]
                        ),
                        "label": int(
                            row[
                                "_LABEL_NUMERIC"
                            ]
                        ),
                    }
                )

        if not samples:
            raise RuntimeError(
                "No matching Excel rows and "
                "PNG files were found."
            )

        return samples

    # =========================================================================
    # FIND GLOBAL CROP SIZE
    # =========================================================================

    @staticmethod
    def _find_global_crop_size(
        samples,
    ):
        """
        Find the smallest height and width across ALL supplied images.

        Every image will subsequently be center-cropped to this exact size.

        No resizing or interpolation is performed.
        """

        min_height = None
        min_width = None

        max_height = None
        max_width = None

        for sample in samples:

            image = cv2.imread(
                sample["img"],
                cv2.IMREAD_GRAYSCALE,
            )

            if image is None:
                raise RuntimeError(
                    "Failed to read image while "
                    "determining crop size: "
                    f"{sample['img']}"
                )

            height, width = (
                image.shape
            )

            if min_height is None:
                min_height = height
                min_width = width
                max_height = height
                max_width = width

            else:
                min_height = min(
                    min_height,
                    height,
                )

                min_width = min(
                    min_width,
                    width,
                )

                max_height = max(
                    max_height,
                    height,
                )

                max_width = max(
                    max_width,
                    width,
                )

        if (
            min_height is None
            or min_width is None
        ):
            raise RuntimeError(
                "Could not determine image crop size."
            )

        return (
            min_height,
            min_width,
            max_height,
            max_width,
        )

    # =========================================================================
    # SETUP
    # =========================================================================

    def setup(
        self,
        stage=None,
    ):

        train_samples = (
            self._load_excel_rows(
                self.train_excel_paths,
                self.data_root,
            )
        )

        validation_samples = (
            self._load_excel_rows(
                [
                    self.validation_excel_path
                ],
                self.data_root,
            )
        )

        test_samples = (
            self._load_excel_rows(
                [
                    self.test_excel_path
                ],
                self.test_data_root,
            )
        )

        # =============================================================
        # LABEL CHECKS
        # =============================================================

        train_labels = np.asarray(
            [
                s["label"]
                for s in train_samples
            ],
            dtype=np.int64,
        )

        validation_labels = np.asarray(
            [
                s["label"]
                for s in validation_samples
            ],
            dtype=np.int64,
        )

        test_labels = np.asarray(
            [
                s["label"]
                for s in test_samples
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

        # =============================================================
        # CLASS WEIGHTS
        # =============================================================

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

        for cid, w in zip(
            classes,
            weights,
        ):
            class_weights[
                int(cid)
            ] = float(w)

        self.class_weights = (
            class_weights
        )

        # =============================================================
        # GLOBAL CROP SIZE
        # =============================================================
        #
        # IMPORTANT:
        #
        # Use train + validation + test together ONLY for determining
        # image dimensions.
        #
        # No pixel values, labels, statistics, or learned parameters
        # are taken from test data.
        #
        # This gives one identical spatial size for every split.
        # =============================================================

        all_samples = (
            train_samples
            + validation_samples
            + test_samples
        )

        (
            crop_height,
            crop_width,
            max_height,
            max_width,
        ) = self._find_global_crop_size(
            all_samples
        )

        # =============================================================
        # DATASETS
        # =============================================================

        common = dict(
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
            Valve2DImageDataset(
                train_samples,
                augment=True,
                **common,
            )
        )

        self.validation_dataset = (
            Valve2DImageDataset(
                validation_samples,
                augment=False,
                **common,
            )
        )

        self.test_dataset = (
            Valve2DImageDataset(
                test_samples,
                augment=False,
                **common,
            )
        )

        # -------------------------------------------------------------
        # SAME CROP SIZE FOR ALL THREE SPLITS
        # -------------------------------------------------------------

        self.train_dataset.crop_height = (
            crop_height
        )

        self.train_dataset.crop_width = (
            crop_width
        )

        self.validation_dataset.crop_height = (
            crop_height
        )

        self.validation_dataset.crop_width = (
            crop_width
        )

        self.test_dataset.crop_height = (
            crop_height
        )

        self.test_dataset.crop_width = (
            crop_width
        )

        # =============================================================
        # IDS
        # =============================================================

        self.train_ids = [
            s["id"]
            for s in train_samples
        ]

        self.validation_ids = [
            s["id"]
            for s in validation_samples
        ]

        self.test_ids = [
            s["id"]
            for s in test_samples
        ]

        # =============================================================
        # LEAKAGE CHECKS
        # =============================================================

        train_set = set(
            self.train_ids
        )

        val_set = set(
            self.validation_ids
        )

        test_set = set(
            self.test_ids
        )

        if train_set & val_set:
            raise RuntimeError(
                "Training and validation contain "
                "overlapping IDs: "
                f"{list(train_set & val_set)[:20]}"
            )

        if train_set & test_set:
            raise RuntimeError(
                "Training and independent test "
                "contain overlapping IDs: "
                f"{list(train_set & test_set)[:20]}"
            )

        if val_set & test_set:
            raise RuntimeError(
                "Validation and independent test "
                "contain overlapping IDs: "
                f"{list(val_set & test_set)[:20]}"
            )

        # =============================================================
        # SUMMARY
        # =============================================================

        print(
            "\nDataModule split summary"
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
            f"Original image size range: "
            f"{crop_height}x{crop_width} "
            f"to "
            f"{max_height}x{max_width}"
        )

        print(
            f"Center crop size: "
            f"{crop_height}x{crop_width}"
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

        # =============================================================
        # IMAGE DIAGNOSTIC
        # =============================================================

        image, label, sid = (
            self.train_dataset[
                0
            ]
        )

        print()
        print(
            "Image diagnostic"
        )

        print(
            "-" * 70
        )

        print(
            f"ID:             {sid}"
        )

        print(
            f"Tensor shape:   "
            f"{tuple(image.shape)}"
        )

        print(
            f"Minimum value:  "
            f"{image.min().item():.4f}"
        )

        print(
            f"Maximum value:  "
            f"{image.max().item():.4f}"
        )

        print(
            "Top-left pixel: "
            f"{image[:, 0, 0].tolist()}"
        )

    # =========================================================================
    # DATALOADERS
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