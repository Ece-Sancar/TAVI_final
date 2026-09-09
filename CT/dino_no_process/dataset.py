#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
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


# ============================================================
# DATASET
# ============================================================

class Valve2DImageDataset(Dataset):

    def __init__(self, samples):
        self.samples = list(samples)

        self.labels = [
            int(s["label"])
            for s in self.samples
        ]

        self.used_ids_all = [
            s["id"]
            for s in self.samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        # ----------------------------------------------------
        # READ ORIGINAL PNG
        # ----------------------------------------------------

        image = cv2.imread(
            sample["img"],
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise RuntimeError(
                f"Failed to read image: {sample['img']}"
            )

        # ----------------------------------------------------
        # NO PREPROCESSING
        # ----------------------------------------------------
        #
        # Original PNG values are kept exactly as 0-255.
        #
        # uint8 [H,W]
        #       ↓
        # float32 [1,H,W]
        #
        # No:
        #   - resizing
        #   - /255
        #   - normalization
        #   - interpolation
        #   - augmentation
        #   - intensity transformation
        #

        image = torch.from_numpy(
            image.astype(np.float32)
        ).unsqueeze(0)

        # ----------------------------------------------------
        # DINO expects 3 channels
        # ----------------------------------------------------
        #
        # No intensity change occurs here.
        # The same grayscale image is simply copied
        # into R, G and B channels.
        #

        image = image.repeat(3, 1, 1)

        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )

        return image, label, sample["id"]


# ============================================================
# DATA MODULE
# ============================================================

class Valve2DImageDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_root,
        train_excel_paths,
        validation_excel_path,
        test_excel_path,
        batch_size=16,
        num_workers=8,
        seed=42,
        artifacts_dir=None,
        test_data_root=None,
    ):

        super().__init__()

        self.data_root = str(data_root)

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

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.seed = int(seed)

        self.artifacts_dir = (
            str(artifacts_dir)
            if artifacts_dir is not None
            else None
        )

        self.class_weights = None

    # ========================================================
    # PNG INDEX
    # ========================================================

    @staticmethod
    def _build_png_index(data_root: str) -> Dict[str, str]:

        root = Path(data_root)

        if not root.is_dir():
            raise FileNotFoundError(
                f"Image directory does not exist: {root}"
            )

        png_by_id = {}

        for path in root.iterdir():

            if (
                path.is_file()
                and path.suffix.lower() == ".png"
            ):

                sid = _normalise_id(path.stem)

                if sid is not None:
                    png_by_id[sid] = str(path)

        if not png_by_id:
            raise RuntimeError(
                f"No PNG images were found in: {root}"
            )

        return png_by_id

    # ========================================================
    # LOAD EXCEL
    # ========================================================

    def _load_excel_rows(
        self,
        excel_paths: Sequence[Path],
        data_root: str,
    ) -> List[Dict]:

        frames = []

        for excel_path in excel_paths:

            excel_path = Path(excel_path)

            if not excel_path.is_file():
                raise FileNotFoundError(
                    f"Excel file does not exist: {excel_path}"
                )

            frame = pd.read_excel(
                excel_path
            ).copy()

            missing = {
                ID_COLUMN,
                LABEL_COLUMN,
            }.difference(frame.columns)

            if missing:
                raise ValueError(
                    f"{excel_path} is missing required columns: "
                    f"{sorted(missing)}"
                )

            frames.append(frame)

        dataframe = pd.concat(
            frames,
            axis=0,
            ignore_index=True,
            sort=False,
        )

        # ----------------------------------------------------
        # LABELS
        # ----------------------------------------------------

        dataframe["_LABEL_NUMERIC"] = (
            dataframe[LABEL_COLUMN]
            .map(_normalise_label)
        )

        dataframe = dataframe.loc[
            dataframe["_LABEL_NUMERIC"].notna()
        ].copy()

        dataframe["_LABEL_NUMERIC"] = (
            dataframe["_LABEL_NUMERIC"]
            .astype(int)
        )

        # ----------------------------------------------------
        # IDs
        # ----------------------------------------------------

        dataframe["_NORMALISED_ID"] = (
            dataframe[ID_COLUMN]
            .map(_normalise_id)
        )

        dataframe = dataframe.loc[
            dataframe["_NORMALISED_ID"].notna()
        ].copy()

        dataframe = (
            dataframe
            .drop_duplicates(
                subset=["_NORMALISED_ID"],
                keep="first",
            )
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # MATCH PNG FILES
        # ----------------------------------------------------

        png_by_id = self._build_png_index(
            data_root
        )

        samples = []

        for _, row in dataframe.iterrows():

            sid = row["_NORMALISED_ID"]

            if sid in png_by_id:

                samples.append(
                    {
                        "id": sid,
                        "img": png_by_id[sid],
                        "label": int(
                            row["_LABEL_NUMERIC"]
                        ),
                    }
                )

        if not samples:
            raise RuntimeError(
                "No matching Excel rows and PNG files were found."
            )

        return samples

    # ========================================================
    # SETUP
    # ========================================================

    def setup(self, stage=None):

        train_samples = self._load_excel_rows(
            self.train_excel_paths,
            self.data_root,
        )

        validation_samples = self._load_excel_rows(
            [self.validation_excel_path],
            self.data_root,
        )

        test_samples = self._load_excel_rows(
            [self.test_excel_path],
            self.test_data_root,
        )

        train_labels = np.asarray(
            [s["label"] for s in train_samples],
            dtype=np.int64,
        )

        validation_labels = np.asarray(
            [s["label"] for s in validation_samples],
            dtype=np.int64,
        )

        test_labels = np.asarray(
            [s["label"] for s in test_samples],
            dtype=np.int64,
        )

        # ----------------------------------------------------
        # CHECK BOTH CLASSES EXIST
        # ----------------------------------------------------

        if len(np.unique(train_labels)) != 2:
            raise RuntimeError(
                "Training folds do not contain both classes."
            )

        if len(np.unique(validation_labels)) != 2:
            raise RuntimeError(
                "Validation fold does not contain both classes."
            )

        if len(np.unique(test_labels)) != 2:
            raise RuntimeError(
                "Independent test set does not contain both classes."
            )

        # ----------------------------------------------------
        # CLASS WEIGHTS
        # ----------------------------------------------------

        classes = np.unique(train_labels)

        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=train_labels,
        )

        class_weights = np.ones(
            2,
            dtype=np.float32,
        )

        for cid, weight in zip(
            classes,
            weights,
        ):
            class_weights[int(cid)] = float(weight)

        self.class_weights = class_weights

        # ----------------------------------------------------
        # DATASETS
        # ----------------------------------------------------

        self.train_dataset = Valve2DImageDataset(
            train_samples
        )

        self.validation_dataset = Valve2DImageDataset(
            validation_samples
        )

        self.test_dataset = Valve2DImageDataset(
            test_samples
        )

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

        # ----------------------------------------------------
        # LEAKAGE CHECK
        # ----------------------------------------------------

        train_set = set(self.train_ids)
        val_set = set(self.validation_ids)
        test_set = set(self.test_ids)

        if train_set & val_set:
            raise RuntimeError(
                "Training and validation contain overlapping IDs: "
                f"{list(train_set & val_set)[:20]}"
            )

        if train_set & test_set:
            raise RuntimeError(
                "Training and independent test contain overlapping IDs: "
                f"{list(train_set & test_set)[:20]}"
            )

        if val_set & test_set:
            raise RuntimeError(
                "Validation and independent test contain overlapping IDs: "
                f"{list(val_set & test_set)[:20]}"
            )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        print("\nDataModule split summary")
        print("-" * 70)

        print(
            f"Train:      {len(self.train_dataset)}"
        )

        print(
            f"Validation: {len(self.validation_dataset)}"
        )

        print(
            f"Test:       {len(self.test_dataset)}"
        )

        print(
            "Train label counts:",
            dict(
                zip(
                    *np.unique(
                        train_labels,
                        return_counts=True,
                    )
                )
            ),
        )

        print(
            "Validation label counts:",
            dict(
                zip(
                    *np.unique(
                        validation_labels,
                        return_counts=True,
                    )
                )
            ),
        )

        print(
            "Test label counts:",
            dict(
                zip(
                    *np.unique(
                        test_labels,
                        return_counts=True,
                    )
                )
            ),
        )

        # ----------------------------------------------------
        # IMAGE DIAGNOSTIC
        # ----------------------------------------------------

        image, _, sid = self.train_dataset[0]

        print("\nRaw image diagnostic")
        print("-" * 70)
        print(f"ID:    {sid}")
        print(f"Shape: {tuple(image.shape)}")
        print(f"Min:   {image.min().item():.4f}")
        print(f"Max:   {image.max().item():.4f}")
        print(
            "Top-left pixel:",
            image[:, 0, 0].tolist(),
        )

    # ========================================================
    # DATALOADERS
    # ========================================================

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )

    def val_dataloader(self):

        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )

    def test_dataloader(self):

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )