#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified combined RadioDINO CT + tabular experiment pipeline.

Features
--------

✓ predefined train/validation folds
✓ fixed independent test.xlsx
✓ TUM / LMU / merged training
✓ TUM / LMU / merged testing
✓ 2 / 5 / 10 / 20 / 50 / 100 percent training data
✓ checkpoint-only ("pre") mode
✓ fixed 0.5 decision threshold for every fold and ensemble
✓ five-fold independent-test mean ± std
✓ five-model probability ensemble
✓ ensemble threshold fixed at 0.5
✓ tabular permutation importance
✓ mean ± std importance across fold models
✓ feature-group colors
✓ per-fold Grad-CAM for every test sample
✓ normalized mean Grad-CAM across fold models
✓ leakage checks
✓ run manifests

The experimental pipeline is identical to the final DenseNet version; only the image backbone is RadioDINO.
"""

import argparse
import gc
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch

try:
    import wandb
except ImportError:
    wandb = None

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)

from pytorch_lightning.loggers import (
    WandbLogger,
)

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset import (
    Valve2DDataModule,
)

from model import (
    Valve2DRadioDINOModel,
)


# =============================================================================
# CONSTANTS
# =============================================================================

CLASS_NAMES = [
    "no_event",
    "pacemaker",
]

VALID_DATASETS = {
    "tum",
    "lmu",
    "merged",
}

VALID_TRAIN_VALUES = (
    VALID_DATASETS
    | {"pre"}
)

VALID_PERCENTAGES = {
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
    "Electrocardiographic": "#1A1A1A",
}

BASE_FONTSIZE = 18
TITLE_FONTSIZE = 22
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


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def set_global_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )


def clean_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_dir(
    path,
) -> None:
    Path(path).mkdir(
        parents=True,
        exist_ok=True,
    )


def save_json(
    data,
    path,
) -> None:
    path = Path(path)

    ensure_dir(
        path.parent
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            data,
            handle,
            indent=2,
        )


def save_lines(
    path,
    values,
) -> None:
    path = Path(path)

    ensure_dir(
        path.parent
    )

    path.write_text(
        "\n".join(
            str(value)
            for value in values
        )
        + "\n",
        encoding="utf-8",
    )


def mean_std(
    values,
) -> Tuple[
    float,
    float,
]:
    array = np.asarray(
        values,
        dtype=float,
    )

    return (
        float(
            np.nanmean(array)
        ),
        float(
            np.nanstd(
                array,
                ddof=0,
            )
        ),
    )


def safe_auc(
    targets,
    probabilities,
) -> float:
    if (
        len(
            np.unique(
                targets
            )
        )
        < 2
    ):
        return float("nan")

    return float(
        roc_auc_score(
            targets,
            probabilities,
        )
    )


def safe_log_loss(
    targets,
    probabilities,
) -> float:
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    probabilities = np.clip(
        probabilities,
        1e-7,
        1.0 - 1e-7,
    )

    return float(
        log_loss(
            targets,
            probabilities,
            labels=[0, 1],
        )
    )


# =============================================================================
# CONFIGURATION / PATH RESOLUTION
# =============================================================================

def get_fold_path(
    folder: Path,
    fold_number: int,
) -> Path:
    candidates = [
        folder
        / f"fold{fold_number}.xlsx",

        folder
        / f"fold_{fold_number}.xlsx",
    ]

    existing = [
        path
        for path in candidates
        if path.exists()
    ]

    if len(existing) == 1:
        return existing[0]

    if len(existing) > 1:
        raise RuntimeError(
            "Multiple fold files found: "
            f"{existing}"
        )

    raise FileNotFoundError(
        f"Could not locate fold "
        f"{fold_number} in {folder}"
    )


def resolve_configuration(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    dataset_root = Path(
        config["dataset_root"]
    )

    train_choice = str(
        config["train_dataset"]
    ).strip().lower()

    test_choice = str(
        config["test_dataset"]
    ).strip().lower()

    percentage = int(
        config.get(
            "data_percentage",
            100,
        )
    )

    if (
        train_choice
        not in VALID_TRAIN_VALUES
    ):
        raise ValueError(
            "train_dataset must be one of "
            f"{sorted(VALID_TRAIN_VALUES)}."
        )

    if (
        test_choice
        not in VALID_DATASETS
    ):
        raise ValueError(
            "test_dataset must be one of "
            f"{sorted(VALID_DATASETS)}."
        )

    if (
        percentage
        not in VALID_PERCENTAGES
    ):
        raise ValueError(
            "data_percentage must be one of "
            f"{sorted(VALID_PERCENTAGES)}."
        )

    # ---------------------------------------------------------
    # Resolve source training dataset
    # ---------------------------------------------------------

    if train_choice == "pre":
        pretrained_config = (
            config.get(
                "pretrained",
                {},
            )
        )

        source_dataset = str(
            pretrained_config.get(
                "train_dataset",
                "",
            )
        ).strip().lower()

        if (
            source_dataset
            not in VALID_DATASETS
        ):
            raise ValueError(
                "When train_dataset='pre', "
                "pretrained.train_dataset must "
                "be tum, lmu or merged."
            )

        checkpoint_root = Path(
            pretrained_config[
                "checkpoint_dir"
            ]
        )

        training_dataset = (
            source_dataset
        )

        mode = "pretrained"

    else:
        training_dataset = (
            train_choice
        )

        checkpoint_root = None

        mode = "train"

    # ---------------------------------------------------------
    # Training folds
    # ---------------------------------------------------------

    if percentage == 100:
        training_folder = (
            dataset_root
            / training_dataset
        )

    else:
        training_folder = (
            dataset_root
            / "data_size"
            / f"{percentage}_percent"
            / training_dataset
        )

    # ---------------------------------------------------------
    # Test is ALWAYS the original fixed test set.
    # ---------------------------------------------------------

    test_folder = (
        dataset_root
        / test_choice
    )

    test_file = (
        test_folder
        / "test.xlsx"
    )

    if not training_folder.is_dir():
        raise NotADirectoryError(
            "Training folder does not exist:\n"
            f"{training_folder}"
        )

    if not test_file.is_file():
        raise FileNotFoundError(
            "Independent test file "
            "does not exist:\n"
            f"{test_file}"
        )

    fold_files = [
        get_fold_path(
            training_folder,
            fold_number,
        )
        for fold_number
        in range(
            1,
            6,
        )
    ]

    # ---------------------------------------------------------
    # Output folders
    # ---------------------------------------------------------

    output_root = Path(
        config["output_dir"]
    )

    output_folder = (
        output_root
        / f"{percentage}_percent"
    )

    if mode == "train":
        checkpoint_output_root = (
            Path(
                config[
                    "checkpoint_dir"
                ]
            )
            / f"{percentage}_percent"
        )

    else:
        checkpoint_output_root = (
            checkpoint_root
        )

    return {
        "mode": mode,
        "percentage": percentage,
        "training_dataset": (
            training_dataset
        ),
        "test_dataset": (
            test_choice
        ),
        "training_folder": (
            training_folder
        ),
        "test_file": (
            test_file
        ),
        "fold_files": (
            fold_files
        ),
        "output_folder": (
            output_folder
        ),
        "checkpoint_root": (
            checkpoint_output_root
        ),
    }


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def find_existing_checkpoint(
    checkpoint_root: Path,
    fold_number: int,
) -> Path:
    fold_folder = (
        checkpoint_root
        / f"fold{fold_number}"
    )

    preferred = [
        fold_folder / "best-checkpoint.ckpt",
    ]

    for path in preferred:
        if path.is_file():
            return path

    candidates = sorted(
        fold_folder.glob(
            "best-checkpoint*.ckpt"
        ),
        key=lambda path: (
            path.stat().st_mtime
        ),
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            "No checkpoint found for "
            f"fold {fold_number} in "
            f"{fold_folder}"
        )

    return candidates[0]


# =============================================================================
# PREDICTIONS / METRICS
# =============================================================================

@torch.inference_mode()
def collect_predictions(
    model,
    data_loader,
    device,
):
    model = model.to(
        device
    )

    model.eval()

    probabilities = []
    targets = []
    sample_ids = []

    for (
        images,
        batch_targets,
        tabular,
        batch_ids,
    ) in data_loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        tabular = tabular.to(
            device,
            non_blocking=True,
        )

        logits = model(
            images,
            tabular,
        )

        batch_probabilities = (
            torch.softmax(
                logits,
                dim=1,
            )[:, 1]
        )

        probabilities.extend(
            batch_probabilities
            .cpu()
            .numpy()
            .tolist()
        )

        targets.extend(
            batch_targets
            .cpu()
            .numpy()
            .tolist()
        )

        sample_ids.extend(
            [
                str(value)
                for value
                in batch_ids
            ]
        )

    return (
        np.asarray(
            probabilities,
            dtype=np.float64,
        ),
        np.asarray(
            targets,
            dtype=np.int64,
        ),
        sample_ids,
    )


def calculate_metrics(
    targets,
    probabilities,
    threshold,
):
    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    confusion = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    )

    (
        tn,
        fp,
        fn,
        tp,
    ) = confusion.ravel()

    return {
        "threshold": float(
            threshold
        ),

        "accuracy": float(
            accuracy_score(
                targets,
                predictions,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                targets,
                predictions,
            )
        ),

        "auc_roc": safe_auc(
            targets,
            probabilities,
        ),

        "auc_pr": float(
            average_precision_score(
                targets,
                probabilities,
            )
        ),

        "f1": float(
            f1_score(
                targets,
                predictions,
                zero_division=0,
            )
        ),

        "precision": float(
            precision_score(
                targets,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                targets,
                predictions,
                zero_division=0,
            )
        ),

        "log_loss": (
            safe_log_loss(
                targets,
                probabilities,
            )
        ),

        "true_negatives": int(
            tn
        ),

        "false_positives": int(
            fp
        ),

        "false_negatives": int(
            fn
        ),

        "true_positives": int(
            tp
        ),
    }


# =============================================================================
# STANDARD EVALUATION OUTPUTS
# =============================================================================

def save_confusion_matrix(
    targets,
    predictions,
    path,
    title,
):
    matrix = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    )

    plt.figure(
        figsize=(7, 6)
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greys",
        cbar=False,
        xticklabels=[
            "No Event",
            "Pacemaker",
        ],
        yticklabels=[
            "No Event",
            "Pacemaker",
        ],
    )

    plt.title(
        title
    )

    plt.ylabel(
        "True label"
    )

    plt.xlabel(
        "Predicted label"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
    )

    plt.close()


def save_evaluation(
    targets,
    probabilities,
    sample_ids,
    threshold,
    output_dir,
    phase,
    fold=None,
):
    ensure_dir(
        output_dir
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    metrics = calculate_metrics(
        targets,
        probabilities,
        threshold,
    )

    frame = pd.DataFrame(
        {
            "ID": sample_ids,

            "true_label": targets,

            "true_class": [
                CLASS_NAMES[
                    int(value)
                ]
                for value
                in targets
            ],

            "prob_pacemaker": (
                probabilities
            ),

            "predicted_label": (
                predictions
            ),

            "predicted_class": [
                CLASS_NAMES[
                    int(value)
                ]
                for value
                in predictions
            ],

            "threshold": float(
                threshold
            ),
        }
    )

    if fold is not None:
        frame.insert(
            0,
            "fold",
            fold,
        )

    frame.to_csv(
        Path(output_dir)
        / f"{phase}_predictions.csv",
        index=False,
    )

    report = classification_report(
        targets,
        predictions,
        labels=[0, 1],
        target_names=[
            "No Event",
            "Pacemaker",
        ],
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(
        report
    ).transpose().to_csv(
        Path(output_dir)
        / (
            f"{phase}_"
            "classification_report.csv"
        )
    )

    save_json(
        metrics,
        Path(output_dir)
        / f"{phase}_metrics.json",
    )

    save_confusion_matrix(
        targets,
        predictions,
        Path(output_dir)
        / (
            f"{phase}_"
            "confusion_matrix.png"
        ),
        (
            phase.replace(
                "_",
                " ",
            ).title()
            + " confusion matrix"
        ),
    )

    # ROC
    fpr, tpr, _ = roc_curve(
        targets,
        probabilities,
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=(
            f"AUC = "
            f"{metrics['auc_roc']:.3f}"
        ),
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "k--",
    )

    plt.xlabel(
        "False positive rate"
    )

    plt.ylabel(
        "True positive rate"
    )

    plt.legend(
        loc="lower right"
    )

    plt.tight_layout()

    plt.savefig(
        Path(output_dir)
        / f"{phase}_roc_curve.png",
        dpi=300,
    )

    plt.close()

    # PR
    precision, recall, _ = (
        precision_recall_curve(
            targets,
            probabilities,
        )
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        recall,
        precision,
        label=(
            f"AP = "
            f"{metrics['auc_pr']:.3f}"
        ),
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.legend(
        loc="lower left"
    )

    plt.tight_layout()

    plt.savefig(
        Path(output_dir)
        / f"{phase}_pr_curve.png",
        dpi=300,
    )

    plt.close()

    return (
        metrics,
        frame,
    )


# =============================================================================
# PATIENT-LEVEL MODALITY IMPORTANCE
# =============================================================================

@torch.inference_mode()
def calculate_modality_importance_percentages(
    model,
    data_loader,
    device,
):
    """
    Calculate image-vs-tabular contribution percentages for every patient.

    The final fusion classifier is linear after concatenating the image and
    tabular embeddings. Therefore the binary class logit margin can be exactly
    decomposed into an image-branch contribution and a tabular-branch
    contribution (plus the classifier bias).

    We report relative absolute branch contributions, excluding the shared
    bias:

        image_pct   = |image contribution| / (|image| + |tabular|) * 100
        tabular_pct = |tabular contribution| / (|image| + |tabular|) * 100

    These two percentages sum to 100 for each patient and fold.
    """
    model = model.to(device)
    model.eval()

    linear_layer = model.classifier[-1]
    if not isinstance(linear_layer, torch.nn.Linear):
        raise TypeError(
            "Expected the final combined classifier layer to be nn.Linear."
        )

    # Difference between pacemaker and no-event classifier weights.
    margin_weights = (
        linear_layer.weight[1] - linear_layer.weight[0]
    )

    image_percentages = []
    tabular_percentages = []
    sample_ids = []

    for images, _, tabular, batch_ids in data_loader:
        images = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)

        image_features = model.extract_image_features(images)
        tabular_features = model.extract_tabular_features(tabular)

        image_dim = image_features.shape[1]
        tabular_dim = tabular_features.shape[1]

        if margin_weights.numel() != image_dim + tabular_dim:
            raise RuntimeError(
                "Fusion classifier width does not match image + tabular embeddings."
            )

        image_weights = margin_weights[:image_dim]
        tabular_weights = margin_weights[image_dim:]

        image_contribution = image_features @ image_weights
        tabular_contribution = tabular_features @ tabular_weights

        image_abs = torch.abs(image_contribution)
        tabular_abs = torch.abs(tabular_contribution)
        denominator = image_abs + tabular_abs

        # If both branch contributions are numerically zero, assign 50/50.
        image_pct = torch.where(
            denominator > 1e-12,
            100.0 * image_abs / denominator,
            torch.full_like(denominator, 50.0),
        )
        tabular_pct = 100.0 - image_pct

        image_percentages.extend(image_pct.cpu().numpy().tolist())
        tabular_percentages.extend(tabular_pct.cpu().numpy().tolist())
        sample_ids.extend([str(value) for value in batch_ids])

    return (
        np.asarray(image_percentages, dtype=np.float64),
        np.asarray(tabular_percentages, dtype=np.float64),
        sample_ids,
    )


# =============================================================================
# FEATURE METADATA
# =============================================================================

def load_feature_metadata(
    metadata_path: Path,
    feature_names: Sequence[str],
):
    metadata = pd.read_excel(
        metadata_path
    )

    required = {
        "Feature",
        "Group",
        "Mapping Name",
    }

    missing = required.difference(
        metadata.columns
    )

    if missing:
        raise ValueError(
            "Feature metadata is missing: "
            f"{sorted(missing)}"
        )

    metadata[
        "Feature"
    ] = metadata[
        "Feature"
    ].astype(str)

    metadata[
        "Group"
    ] = metadata[
        "Group"
    ].astype(str)

    metadata[
        "Mapping Name"
    ] = metadata[
        "Mapping Name"
    ].astype(str)

    missing_features = (
        set(feature_names)
        - set(
            metadata[
                "Feature"
            ]
        )
    )

    if missing_features:
        raise ValueError(
            "Features missing from metadata:\n"
            f"{sorted(missing_features)}"
        )

    feature_to_group = dict(
        zip(
            metadata[
                "Feature"
            ],
            metadata[
                "Group"
            ],
        )
    )

    feature_to_name = dict(
        zip(
            metadata[
                "Feature"
            ],
            metadata[
                "Mapping Name"
            ],
        )
    )

    for feature in feature_names:
        group = feature_to_group[
            feature
        ]

        if (
            group
            not in GROUP_COLORS
        ):
            raise ValueError(
                f"Unsupported feature group "
                f"{group!r} for "
                f"{feature!r}"
            )

    return (
        feature_to_group,
        feature_to_name,
    )


# =============================================================================
# PERMUTATION IMPORTANCE
# =============================================================================

def calculate_permutation_importance(
    model,
    test_dataset,
    test_loader,
    device,
    targets,
    baseline_probabilities,
    n_repeats,
    seed,
):
    """
    Tabular permutation importance for the complete multimodal model.

    CT images remain unchanged.

    One tabular feature is shuffled across test patients.

    Importance:
        increase in test log loss.

    IMPORTANT:
        A dedicated num_workers=0 DataLoader is used because modifying
        test_dataset.tabular_features does not propagate to persistent
        DataLoader worker processes.
    """

    baseline_loss = safe_log_loss(
        targets,
        baseline_probabilities,
    )

    original_features = (
        test_dataset
        .tabular_features
        .copy()
    )

    total_model_input_features = original_features.shape[1]
    number_of_features = len(test_dataset.tabular_columns)

    if total_model_input_features != 2 * number_of_features:
        raise RuntimeError(
            "Expected tabular model input to contain standardized values "
            "followed by one missingness mask per original feature. "
            f"Got width={total_model_input_features}, "
            f"original_features={number_of_features}."
        )

    mean_deltas = np.zeros(
        number_of_features,
        dtype=np.float64,
    )

    repeat_stds = np.zeros(
        number_of_features,
        dtype=np.float64,
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    # Use num_workers=0 so changes to
    # test_dataset.tabular_features are seen immediately.
    # ---------------------------------------------------------
    importance_loader = DataLoader(
        test_dataset,
        batch_size=test_loader.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    try:
        for feature_index in tqdm(
            range(number_of_features),
            desc="Permutation importance",
            leave=False,
        ):
            deltas = []

            for repeat in range(
                n_repeats
            ):
                rng = np.random.default_rng(
                    seed
                    + feature_index * 1009
                    + repeat
                )

                permuted_features = (
                    original_features.copy()
                )

                permutation = rng.permutation(
                    len(permuted_features)
                )

                # Permute the standardized value AND its corresponding
                # missingness mask together. This reports one importance for
                # the original clinical feature rather than separate mask
                # variables.
                mask_index = number_of_features + feature_index

                permuted_features[:, feature_index] = (
                    original_features[permutation, feature_index]
                )
                permuted_features[:, mask_index] = (
                    original_features[permutation, mask_index]
                )

                # This now works because the importance
                # DataLoader uses num_workers=0.
                test_dataset.tabular_features = (
                    permuted_features
                )

                (
                    probabilities,
                    permuted_targets,
                    permuted_ids,
                ) = collect_predictions(
                    model,
                    importance_loader,
                    device,
                )

                if not np.array_equal(
                    permuted_targets,
                    targets,
                ):
                    raise RuntimeError(
                        "Targets changed during "
                        "permutation importance."
                    )

                permuted_loss = safe_log_loss(
                    targets,
                    probabilities,
                )

                delta = (
                    permuted_loss
                    - baseline_loss
                )

                deltas.append(
                    delta
                )

            mean_deltas[
                feature_index
            ] = float(
                np.mean(deltas)
            )

            if n_repeats > 1:
                repeat_stds[
                    feature_index
                ] = float(
                    np.std(
                        deltas,
                        ddof=1,
                    )
                )

    finally:
        # Always restore the original test features.
        test_dataset.tabular_features = (
            original_features
        )

    return (
        mean_deltas,
        repeat_stds,
        baseline_loss,
    )

def plot_fold_importance(
    table,
    save_path,
    title,
    top_k,
):
    top = (
        table.nlargest(
            top_k,
            "importance_percent",
        )
        .sort_values(
            "importance_percent",
            ascending=True,
        )
    )

    colors = [
        GROUP_COLORS[
            group
        ]
        for group
        in top["group"]
    ]

    plt.figure(
        figsize=(14, 11)
    )

    plt.barh(
        top[
            "feature_pretty"
        ],
        top[
            "importance_percent"
        ],
        color=colors,
        edgecolor="black",
        linewidth=1.0,
    )

    plt.xlabel(
        "Importance (%)"
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def aggregate_importance(
    fold_importances,
    feature_names,
    feature_to_group,
    feature_to_name,
    output_dir,
    top_k,
):
    matrix = np.vstack(
        fold_importances
    )

    mean_raw = np.mean(
        matrix,
        axis=0,
    )

    std_raw = np.std(
        matrix,
        axis=0,
        ddof=0,
    )

    # Use absolute magnitude for normalized
    # contribution visualization.
    denominator = float(
        np.sum(
            np.abs(
                mean_raw
            )
        )
    )

    if denominator == 0:
        denominator = 1.0

    mean_percent = (
        np.abs(
            mean_raw
        )
        / denominator
        * 100.0
    )

    std_percent = (
        np.abs(
            std_raw
        )
        / denominator
        * 100.0
    )

    table = pd.DataFrame(
        {
            "feature": (
                feature_names
            ),

            "feature_pretty": [
                feature_to_name[
                    feature
                ]
                for feature
                in feature_names
            ],

            "group": [
                feature_to_group[
                    feature
                ]
                for feature
                in feature_names
            ],

            "importance_mean": (
                mean_percent
            ),

            "importance_std": (
                std_percent
            ),

            "raw_delta_log_loss_mean": (
                mean_raw
            ),

            "raw_delta_log_loss_std": (
                std_raw
            ),
        }
    ).sort_values(
        "importance_mean",
        ascending=False,
    )

    table.to_csv(
        Path(output_dir)
        / (
            "feature_importance_"
            "mean_std.csv"
        ),
        index=False,
    )

    top = (
        table.nlargest(
            top_k,
            "importance_mean",
        )
        .sort_values(
            "importance_mean",
            ascending=True,
        )
    )

    y = np.arange(
        len(top)
    )

    colors = [
        GROUP_COLORS[
            group
        ]
        for group
        in top["group"]
    ]

    plt.figure(
        figsize=(15, 12)
    )

    plt.barh(
        y,
        top[
            "importance_mean"
        ],
        xerr=top[
            "importance_std"
        ],
        color=colors,
        edgecolor="black",
        linewidth=1.0,
        capsize=6,
    )

    plt.yticks(
        y,
        top[
            "feature_pretty"
        ],
    )

    plt.xlabel(
        "Permutation importance (%)"
    )

    plt.title(
        "Tabular Feature Importance — "
        "Mean ± SD Across Fold Models"
    )

    plt.tight_layout()

    plt.savefig(
        Path(output_dir)
        / (
            "feature_importance_"
            "mean_std.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# =============================================================================
# GRAD-CAM
# =============================================================================

def normalize_cam(
    cam: np.ndarray,
) -> np.ndarray:
    cam = np.asarray(
        cam,
        dtype=np.float32,
    )

    cam = cam - np.min(
        cam
    )

    maximum = np.max(
        cam
    )

    if maximum > 0:
        cam = cam / maximum

    return cam.astype(
        np.float32
    )


def calculate_gradcam(
    model,
    image,
    tabular,
    device,
    target_class=1,
):
    """
    RadioDINO gradient-weighted patch attribution for one image.

    The final transformer block exposes tokens [B, N, C]. The spatial patch
    tokens are converted back to a square patch grid so the rest of the
    Grad-CAM saving/overlay/ensemble pipeline stays identical to DenseNet.
    """
    model = model.to(device)
    model.eval()

    image = image.unsqueeze(0).to(device)
    tabular = tabular.unsqueeze(0).to(device)
    image.requires_grad_(True)

    activations = {}
    gradients = {}
    target_layer = model.gradcam_target_layer()

    def forward_hook(module, inputs, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(image, tabular)
        score = logits[0, target_class]
        score.backward()

        activation = activations["value"]
        gradient = gradients["value"]

        if activation.ndim != 3 or gradient.ndim != 3:
            raise RuntimeError(
                "RadioDINO Grad-CAM expected token tensors [B, N, C], "
                f"but received activation={tuple(activation.shape)}, "
                f"gradient={tuple(gradient.shape)}."
            )

        activation = activation[0]
        gradient = gradient[0]
        token_count = activation.shape[0]

        prefix_tokens = int(
            getattr(model.backbone, "num_prefix_tokens", 1)
        )
        spatial_tokens = token_count - prefix_tokens
        grid_size = int(round(math.sqrt(spatial_tokens)))

        if grid_size * grid_size != spatial_tokens:
            prefix_tokens = 1
            spatial_tokens = token_count - 1
            grid_size = int(round(math.sqrt(spatial_tokens)))

        if grid_size * grid_size != spatial_tokens:
            raise RuntimeError(
                "Could not reshape RadioDINO patch tokens into a square grid. "
                f"token_count={token_count}, prefix_tokens={prefix_tokens}, "
                f"spatial_tokens={spatial_tokens}."
            )

        patch_activation = activation[prefix_tokens:]
        patch_gradient = gradient[prefix_tokens:]

        weights = patch_gradient.mean(dim=0)
        cam_tokens = torch.sum(
            patch_activation * weights.unsqueeze(0),
            dim=1,
        )
        cam_tokens = torch.relu(cam_tokens)
        cam = cam_tokens.reshape(grid_size, grid_size)
        cam = cam.detach().cpu().numpy()
        cam = normalize_cam(cam)

    finally:
        forward_handle.remove()
        backward_handle.remove()

    return cam
def denormalize_ct(
    image_tensor,
    image_mean,
    image_std,
):
    """
    Recover grayscale CT representation from
    normalized repeated-RGB image tensor.
    """
    image = (
        image_tensor[0]
        .detach()
        .cpu()
        .numpy()
    )

    image = (
        image
        * float(
            image_std[0]
        )
        + float(
            image_mean[0]
        )
    )

    return np.clip(
        image,
        0.0,
        1.0,
    )


def create_gradcam_overlay(
    grayscale_image,
    cam,
):
    height, width = (
        grayscale_image.shape
    )

    resized_cam = cv2.resize(
        cam,
        (
            width,
            height,
        ),
        interpolation=(
            cv2.INTER_LINEAR
        ),
    )

    heatmap = (
        np.clip(
            resized_cam,
            0.0,
            1.0,
        )
        * 255
    ).astype(
        np.uint8
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    base = (
        grayscale_image
        * 255
    ).astype(
        np.uint8
    )

    base = cv2.cvtColor(
        base,
        cv2.COLOR_GRAY2BGR,
    )

    overlay = cv2.addWeighted(
        base,
        0.55,
        heatmap,
        0.45,
        0,
    )

    return overlay


def prediction_category(
    true_label,
    predicted_label,
):
    if (
        true_label == 1
        and predicted_label == 1
    ):
        return (
            "label_pacer_predict_pacer"
        )

    if (
        true_label == 0
        and predicted_label == 1
    ):
        return (
            "label_no_predict_pacer"
        )

    if (
        true_label == 1
        and predicted_label == 0
    ):
        return (
            "label_pacer_predict_no"
        )

    return (
        "label_no_predict_no"
    )


def generate_fold_gradcams(
    model,
    dataset,
    probabilities,
    threshold,
    output_root,
    fold_number,
    device,
    image_mean,
    image_std,
    cam_sums,
):
    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    fold_root = (
        Path(output_root)
        / f"fold{fold_number}"
    )

    ensure_dir(
        fold_root
    )

    print(
        f"Generating Grad-CAM: "
        f"fold {fold_number}"
    )

    for index in tqdm(
        range(
            len(dataset)
        ),
        desc=(
            f"Grad-CAM fold "
            f"{fold_number}"
        ),
    ):
        (
            image,
            label,
            tabular,
            sample_id,
        ) = dataset[index]

        label_int = int(
            label.item()
        )

        prediction_int = int(
            predictions[
                index
            ]
        )

        cam = calculate_gradcam(
            model=model,
            image=image,
            tabular=tabular,
            device=device,
            target_class=1,
        )

        # cam is normalized independently before
        # entering fold average.
        if sample_id not in cam_sums:
            cam_sums[
                sample_id
            ] = np.zeros_like(
                cam,
                dtype=np.float32,
            )

        cam_sums[
            sample_id
        ] += cam

        grayscale = denormalize_ct(
            image,
            image_mean,
            image_std,
        )

        overlay = (
            create_gradcam_overlay(
                grayscale,
                cam,
            )
        )

        category = (
            prediction_category(
                label_int,
                prediction_int,
            )
        )

        category_folder = (
            fold_root
            / category
        )

        ensure_dir(
            category_folder
        )

        cv2.imwrite(
            str(
                category_folder
                / f"{sample_id}.png"
            ),
            overlay,
        )


def save_ensemble_gradcams(
    dataset,
    cam_sums,
    ensemble_predictions,
    output_root,
    number_of_models,
    image_mean,
    image_std,
):
    ensemble_root = (
        Path(output_root)
        / "ensemble"
    )

    ensure_dir(
        ensemble_root
    )

    for index in tqdm(
        range(
            len(dataset)
        ),
        desc="Ensemble Grad-CAM",
    ):
        (
            image,
            label,
            _,
            sample_id,
        ) = dataset[index]

        if sample_id not in cam_sums:
            raise RuntimeError(
                "Missing accumulated Grad-CAM "
                f"for sample {sample_id}."
            )

        average_cam = (
            cam_sums[
                sample_id
            ]
            / float(
                number_of_models
            )
        )

        average_cam = (
            normalize_cam(
                average_cam
            )
        )

        grayscale = denormalize_ct(
            image,
            image_mean,
            image_std,
        )

        overlay = (
            create_gradcam_overlay(
                grayscale,
                average_cam,
            )
        )

        true_label = int(
            label.item()
        )

        predicted_label = int(
            ensemble_predictions[
                index
            ]
        )

        category = (
            prediction_category(
                true_label,
                predicted_label,
            )
        )

        category_folder = (
            ensemble_root
            / category
        )

        ensure_dir(
            category_folder
        )

        cv2.imwrite(
            str(
                category_folder
                / f"{sample_id}.png"
            ),
            overlay,
        )


# =============================================================================
# SUMMARY
# =============================================================================

def summarise_fold_metrics(
    fold_results,
):
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "auc_roc",
        "auc_pr",
        "f1",
        "precision",
        "recall",
        "log_loss",
        "false_negatives",
        "false_positives",
    ]

    summary = {}

    for metric_name in (
        metric_names
    ):
        (
            mean_value,
            std_value,
        ) = mean_std(
            [
                fold[
                    "test_metrics"
                ][
                    metric_name
                ]
                for fold
                in fold_results
            ]
        )

        summary[
            metric_name
        ] = {
            "mean": mean_value,
            "std": std_value,
        }

    return summary


# =============================================================================
# MAIN
# =============================================================================

def main(
    config,
):
    resolved = (
        resolve_configuration(
            config
        )
    )

    seed = int(
        config.get(
            "seed",
            42,
        )
    )

    n_folds = int(
        config.get(
            "n_folds",
            5,
        )
    )

    if n_folds != 5:
        raise ValueError(
            "This experiment requires "
            "exactly five predefined folds."
        )

    set_global_seed(
        seed
    )

    pl.seed_everything(
        seed,
        workers=True,
    )

    output_dir = Path(
        resolved[
            "output_folder"
        ]
    )

    ensure_dir(
        output_dir
    )

    if resolved["mode"] == "train":
        ensure_dir(
            resolved[
                "checkpoint_root"
            ]
        )

    # ---------------------------------------------------------
    # Config values
    # ---------------------------------------------------------

    data_root = str(
        config[
            "data_root"
        ]
    )

    test_data_root = str(
        config.get(
            "test_data_root",
            data_root,
        )
    )

    image_size = int(
        config.get(
            "img_size",
            448,
        )
    )

    image_mean = tuple(
        config.get(
            "image_mean",
            [
                0.485,
                0.456,
                0.406,
            ],
        )
    )

    image_std = tuple(
        config.get(
            "image_std",
            [
                0.229,
                0.224,
                0.225,
            ],
        )
    )

    batch_size = int(
        config.get(
            "batch_size",
            16,
        )
    )

    num_workers = int(
        config.get(
            "num_workers",
            8,
        )
    )

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------

    importance_config = (
        config.get(
            "feature_importance",
            {},
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

    importance_top_k = int(
        importance_config.get(
            "top_k",
            15,
        )
    )

    metadata_path = Path(
        importance_config.get(
            "metadata_path",
            (
                "/home/ubuntu/"
                "TAVI_new/dataset/"
                "new_features_table.xlsx"
            ),
        )
    )

    importance_dir = (
        output_dir
        / "feature_importance"
    )

    if importance_enabled:
        ensure_dir(
            importance_dir
        )

    # ---------------------------------------------------------
    # Grad-CAM
    # ---------------------------------------------------------

    gradcam_config = (
        config.get(
            "gradcam",
            {},
        )
    )

    gradcam_enabled = bool(
        gradcam_config.get(
            "enabled",
            True,
        )
    )

    gradcam_save_per_fold = bool(
        gradcam_config.get(
            "save_per_fold",
            True,
        )
    )

    gradcam_save_ensemble = bool(
        gradcam_config.get(
            "save_ensemble_average",
            True,
        )
    )

    gradcam_dir = (
        output_dir
        / "gradcam"
    )

    if gradcam_enabled:
        ensure_dir(
            gradcam_dir
        )

    # ---------------------------------------------------------
    # Save resolved configuration
    # ---------------------------------------------------------

    run_manifest = {
        "mode": (
            resolved[
                "mode"
            ]
        ),

        "train_dataset": (
            resolved[
                "training_dataset"
            ]
        ),

        "test_dataset": (
            resolved[
                "test_dataset"
            ]
        ),

        "data_percentage": (
            resolved[
                "percentage"
            ]
        ),

        "training_folder": str(
            resolved[
                "training_folder"
            ]
        ),

        "test_file": str(
            resolved[
                "test_file"
            ]
        ),

        "fold_files": [
            str(path)
            for path
            in resolved[
                "fold_files"
            ]
        ],

        "test_policy": (
            "Only the fixed original "
            "test_dataset/test.xlsx "
            "is used for final testing."
        ),
    }

    save_json(
        {
            **config,
            "resolved": (
                run_manifest
            ),
        },
        output_dir
        / "resolved_config.json",
    )

    print()
    print("=" * 80)
    print("COMBINED RADIODINO CT + TABULAR EXPERIMENT")
    print("=" * 80)

    print(
        f"Mode:             "
        f"{resolved['mode']}"
    )

    print(
        f"Training dataset: "
        f"{resolved['training_dataset']}"
    )

    print(
        f"Test dataset:     "
        f"{resolved['test_dataset']}"
    )

    print(
        f"Data percentage:  "
        f"{resolved['percentage']}%"
    )

    print(
        f"Training folder:  "
        f"{resolved['training_folder']}"
    )

    print(
        f"Independent test: "
        f"{resolved['test_file']}"
    )

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device:           "
        f"{device}"
    )

    # ---------------------------------------------------------
    # Containers
    # ---------------------------------------------------------

    fold_results = []

    fold_probabilities = []

    raw_fold_importances = []

    fold_image_importance_percentages = []
    fold_tabular_importance_percentages = []

    reference_test_ids = None
    reference_test_targets = None
    reference_test_dataset = None
    reference_feature_names = None

    # Low-resolution normalized Grad-CAM sum
    # for each patient.
    cam_sums = {}

    # =========================================================
    # FIVE FOLD MODELS
    # =========================================================

    for fold_index in range(
        n_folds
    ):
        fold_number = (
            fold_index + 1
        )

        print()
        print("=" * 80)
        print(
            f"FOLD MODEL "
            f"{fold_number}/{n_folds}"
        )
        print("=" * 80)

        fold_seed = (
            seed
            + fold_number
        )

        set_global_seed(
            fold_seed
        )

        pl.seed_everything(
            fold_seed,
            workers=True,
        )

        fold_output_dir = (
            output_dir
            / f"fold{fold_number}"
        )

        ensure_dir(
            fold_output_dir
        )

        validation_file = (
            resolved[
                "fold_files"
            ][
                fold_index
            ]
        )

        training_files = [
            path
            for index, path
            in enumerate(
                resolved[
                    "fold_files"
                ]
            )
            if index
            != fold_index
        ]

        # -----------------------------------------------------
        # DataModule
        # -----------------------------------------------------

        data_module = (
            Valve2DDataModule(
                data_root=data_root,
                test_data_root=(
                    test_data_root
                ),
                train_excel_paths=(
                    training_files
                ),
                validation_excel_path=(
                    validation_file
                ),
                test_excel_path=(
                    resolved[
                        "test_file"
                    ]
                ),
                batch_size=(
                    batch_size
                ),
                num_workers=(
                    num_workers
                ),
                target_size=(
                    image_size,
                    image_size,
                ),
                seed=(
                    fold_seed
                ),
                image_mean=(
                    image_mean
                ),
                image_std=(
                    image_std
                ),
                artifacts_dir=(
                    fold_output_dir
                ),
            )
        )

        data_module.setup()

        # -----------------------------------------------------
        # Save fold manifest
        # -----------------------------------------------------

        fold_manifest = {
            "fold": fold_number,

            "training_files": [
                str(path)
                for path
                in training_files
            ],

            "validation_file": str(
                validation_file
            ),

            "test_file": str(
                resolved[
                    "test_file"
                ]
            ),

            "training_samples": int(
                len(
                    data_module
                    .train_dataset
                )
            ),

            "validation_samples": int(
                len(
                    data_module
                    .validation_dataset
                )
            ),

            "test_samples": int(
                len(
                    data_module
                    .test_dataset
                )
            ),

            "tabular_features": int(
                len(
                    data_module
                    .tabular_columns
                )
            ),
        }

        save_json(
            fold_manifest,
            fold_output_dir
            / "fold_manifest.json",
        )

        save_lines(
            fold_output_dir
            / "train_ids.txt",
            data_module.train_ids,
        )

        save_lines(
            fold_output_dir
            / "validation_ids.txt",
            data_module.validation_ids,
        )

        save_lines(
            fold_output_dir
            / "test_ids.txt",
            data_module.test_ids,
        )

        # -----------------------------------------------------
        # Ensure identical test set/order.
        # -----------------------------------------------------

        current_test_ids = list(
            data_module.test_ids
        )

        current_test_targets = np.asarray(
            data_module
            .test_dataset
            .labels,
            dtype=np.int64,
        )

        if (
            reference_test_ids
            is None
        ):
            reference_test_ids = (
                current_test_ids
            )

            reference_test_targets = (
                current_test_targets
            )

            reference_test_dataset = (
                data_module
                .test_dataset
            )

            reference_feature_names = list(
                data_module
                .tabular_columns
            )

        else:
            if (
                current_test_ids
                != reference_test_ids
            ):
                raise RuntimeError(
                    "Independent test sample "
                    "order changed between folds."
                )

            if not np.array_equal(
                current_test_targets,
                reference_test_targets,
            ):
                raise RuntimeError(
                    "Independent test labels "
                    "changed between folds."
                )

            if list(
                data_module
                .tabular_columns
            ) != list(
                reference_feature_names
            ):
                raise RuntimeError(
                    "Tabular feature list "
                    "changed between folds."
                )

        # =====================================================
        # TRAIN OR LOAD
        # =====================================================

        if (
            resolved["mode"]
            == "train"
        ):
            fold_checkpoint_dir = (
                resolved[
                    "checkpoint_root"
                ]
                / f"fold{fold_number}"
            )

            ensure_dir(
                fold_checkpoint_dir
            )

            model = (
                Valve2DRadioDINOModel(
                    learning_rate=float(
                        config[
                            "learning_rate"
                        ]
                    ),

                    class_weights=(
                        data_module
                        .class_weights
                        .tolist()
                    ),

                    dropout_rate_image=float(
                        config[
                            "dropout_rate_image"
                        ]
                    ),

                    dropout_rate_tabular=float(
                        config[
                            "dropout_rate_tabular"
                        ]
                    ),

                    weight_decay=float(
                        config[
                            "weight_decay"
                        ]
                    ),

                    label_smoothing=float(
                        config[
                            "label_smoothing"
                        ]
                    ),

                    warmup_epochs=int(
                        config[
                            "warmup_epochs"
                        ]
                    ),

                    min_lr_factor=float(
                        config[
                            "min_lr_factor"
                        ]
                    ),

                    freeze_bn=bool(
                        config[
                            "freeze_bn"
                        ]
                    ),

                    unfreeze_epoch=int(
                        config[
                            "unfreeze_epoch"
                        ]
                    ),

                    unfreeze_last_n_blocks=int(
                        config[
                            "unfreeze_last_n_blocks"
                        ]
                    ),

                    backbone_lr_multiplier=float(
                        config[
                            "backbone_lr_multiplier"
                        ]
                    ),

                    max_epochs=int(
                        config[
                            "max_epochs"
                        ]
                    ),

                    tabular_in=int(
                        data_module.model_input_dim
                    ),

                    model_name=str(
                        config.get(
                            "model_name",
                            "hf_hub:Snarcy/RadioDino-s8",
                        )
                    ),

                    embed_dim=int(
                        config.get(
                            "embed_dim",
                            384,
                        )
                    ),
                )
            )

            checkpoint_callback = (
                ModelCheckpoint(
                    dirpath=(
                        fold_checkpoint_dir
                    ),
                    filename=(
                        "best-checkpoint"
                    ),
                    monitor=(
                        config[
                            "checkpoint_metric"
                        ]
                    ),
                    mode=(
                        config[
                            "checkpoint_mode"
                        ]
                    ),
                    save_top_k=1,
                    save_last=True,
                )
            )

            callbacks = [
                checkpoint_callback,

                EarlyStopping(
                    monitor=(
                        config[
                            "early_stopping_metric"
                        ]
                    ),
                    mode=(
                        config[
                            "early_stopping_mode"
                        ]
                    ),
                    patience=int(
                        config[
                            "patience"
                        ]
                    ),
                    min_delta=float(
                        config[
                            "early_stopping_min_delta"
                        ]
                    ),
                    verbose=True,
                ),

                LearningRateMonitor(
                    logging_interval="epoch"
                ),
            ]

            if bool(
                config.get(
                    "use_swa",
                    False,
                )
            ):
                callbacks.append(
                    StochasticWeightAveraging(
                        swa_lrs=float(
                            config[
                                "swa_lr"
                            ]
                        ),
                        swa_epoch_start=(
                            config[
                                "swa_epoch_start"
                            ]
                        ),
                    )
                )

            # ---------------------------------------------
            # WandB
            # ---------------------------------------------

            logger = None

            if bool(
                config.get(
                    "use_wandb",
                    True,
                )
            ):
                if wandb is None:
                    raise ImportError(
                        "use_wandb=true but "
                        "wandb is not installed."
                    )

                logger = WandbLogger(
                    project=(
                        config[
                            "wandb_project"
                        ]
                    ),
                    name=(
                        f"{config['run_name']}"
                        f"_{resolved['percentage']}"
                        f"pct_fold{fold_number}"
                    ),
                    group=(
                        f"{config['run_name']}"
                        f"_{resolved['percentage']}"
                        "pct"
                    ),
                    log_model=False,
                )

                logger.log_hyperparams(
                    {
                        **config,
                        "fold": (
                            fold_number
                        ),
                        "data_percentage": (
                            resolved[
                                "percentage"
                            ]
                        ),
                    }
                )

            trainer = pl.Trainer(
                max_epochs=int(
                    config[
                        "max_epochs"
                    ]
                ),

                logger=logger,

                callbacks=callbacks,

                accelerator=(
                    "gpu"
                    if torch.cuda.is_available()
                    else "cpu"
                ),

                devices=(
                    int(
                        config.get(
                            "num_gpus",
                            1,
                        )
                    )
                    if torch.cuda.is_available()
                    else 1
                ),

                precision=(
                    "16-mixed"
                    if torch.cuda.is_available()
                    else "32-true"
                ),

                accumulate_grad_batches=int(
                    config[
                        "accumulate_grad_batches"
                    ]
                ),

                gradient_clip_val=0.5,

                deterministic=True,

                log_every_n_steps=5,
            )

            trainer.fit(
                model,
                datamodule=(
                    data_module
                ),
            )

            best_path = (
                checkpoint_callback
                .best_model_path
            )

            if not best_path:
                raise RuntimeError(
                    "No best checkpoint "
                    f"saved for fold "
                    f"{fold_number}."
                )

            best_model = (
                Valve2DRadioDINOModel
                .load_from_checkpoint(
                    best_path,
                    map_location=(
                        device
                    ),
                )
            )

        else:
            # =============================================
            # PRETRAINED MODE
            # =============================================

            logger = None
            trainer = None

            best_path = (
                find_existing_checkpoint(
                    resolved[
                        "checkpoint_root"
                    ],
                    fold_number,
                )
            )

            print(
                "Loading checkpoint: "
                f"{best_path}"
            )

            best_model = (
                Valve2DRadioDINOModel
                .load_from_checkpoint(
                    str(best_path),
                    map_location=(
                        device
                    ),
                )
            )

            expected_tabular_width = int(
                best_model
                .hparams
                .tabular_in
            )

            actual_tabular_width = int(
                data_module.model_input_dim
            )

            if (
                expected_tabular_width
                != actual_tabular_width
            ):
                raise ValueError(
                    f"Fold {fold_number} "
                    f"checkpoint expects "
                    f"{expected_tabular_width} "
                    "tabular features, but "
                    "the reconstructed dataset "
                    f"has {actual_tabular_width}."
                )

        best_model = (
            best_model.to(
                device
            )
        )

        # =====================================================
        # VALIDATION EVALUATION — FIXED THRESHOLD 0.5
        # =====================================================

        # The same fixed threshold is used for validation, every fold test,
        # and the five-model ensemble. No threshold is fitted/calibrated.
        threshold = 0.5
        best_model.decision_threshold = 0.5

        (
            val_probabilities,
            val_targets,
            val_ids,
        ) = collect_predictions(
            best_model,
            data_module.val_dataloader(),
            device,
        )

        validation_metrics, _ = save_evaluation(
            targets=val_targets,
            probabilities=val_probabilities,
            sample_ids=val_ids,
            threshold=0.5,
            output_dir=fold_output_dir,
            phase="validation",
            fold=fold_number,
        )

        save_json(
            {
                "threshold": 0.5,
                "policy": "Fixed threshold; no calibration or optimization.",
                "checkpoint": str(best_path),
            },
            fold_output_dir / "decision_threshold.json",
        )

        # =====================================================
        # FIXED INDEPENDENT TEST
        # =====================================================

        (
            test_probabilities,
            test_targets,
            test_ids,
        ) = collect_predictions(
            best_model,
            data_module
            .test_dataloader(),
            device,
        )

        if (
            test_ids
            != reference_test_ids
        ):
            raise RuntimeError(
                "Independent test ordering "
                "changed during prediction."
            )

        if not np.array_equal(
            test_targets,
            reference_test_targets,
        ):
            raise RuntimeError(
                "Independent test labels "
                "changed during prediction."
            )

        (
            test_metrics,
            test_frame,
        ) = save_evaluation(
            targets=test_targets,
            probabilities=test_probabilities,
            sample_ids=test_ids,
            threshold=0.5,
            output_dir=fold_output_dir,
            phase="test",
            fold=fold_number,
        )

        fold_probabilities.append(
            test_probabilities
        )

        # =====================================================
        # PATIENT-LEVEL IMAGE VS TABULAR IMPORTANCE
        # =====================================================
        (
            image_importance_pct,
            tabular_importance_pct,
            modality_ids,
        ) = calculate_modality_importance_percentages(
            best_model,
            data_module.test_dataloader(),
            device,
        )

        if modality_ids != reference_test_ids:
            raise RuntimeError(
                "Sample order changed during modality-importance calculation."
            )

        fold_image_importance_percentages.append(image_importance_pct)
        fold_tabular_importance_percentages.append(tabular_importance_pct)

        pd.DataFrame(
            {
                "ID": reference_test_ids,
                "Image_Importance_Pct": image_importance_pct,
                "Tabular_Importance_Pct": tabular_importance_pct,
            }
        ).to_excel(
            fold_output_dir / "modality_importance_per_patient.xlsx",
            index=False,
        )

        fold_result = {
            "fold": fold_number,
            "threshold": 0.5,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "best_checkpoint": str(best_path),
        }

        fold_results.append(
            fold_result
        )

        print()
        print(
            f"Fold {fold_number} "
            "independent test:"
        )

        print(
            f"Accuracy: "
            f"{test_metrics['accuracy']:.4f}"
        )

        print(
            f"F1:       "
            f"{test_metrics['f1']:.4f}"
        )

        print(
            f"AUC-ROC:  "
            f"{test_metrics['auc_roc']:.4f}"
        )

        # =====================================================
        # PERMUTATION IMPORTANCE
        # =====================================================

        if importance_enabled:
            (
                raw_importance,
                repeat_std,
                baseline_loss,
            ) = (
                calculate_permutation_importance(
                    model=(
                        best_model
                    ),
                    test_dataset=(
                        data_module
                        .test_dataset
                    ),
                    test_loader=(
                        data_module
                        .test_dataloader()
                    ),
                    device=(
                        device
                    ),
                    targets=(
                        test_targets
                    ),
                    baseline_probabilities=(
                        test_probabilities
                    ),
                    n_repeats=(
                        importance_repeats
                    ),
                    seed=(
                        fold_seed
                    ),
                )
            )

            raw_fold_importances.append(
                raw_importance
            )

            if (
                reference_feature_names
                is None
            ):
                raise RuntimeError(
                    "Feature names were not "
                    "initialized."
                )

            if fold_number == 1:
                (
                    feature_to_group,
                    feature_to_name,
                ) = (
                    load_feature_metadata(
                        metadata_path=(
                            metadata_path
                        ),
                        feature_names=(
                            reference_feature_names
                        ),
                    )
                )

            denominator = float(
                np.sum(
                    np.abs(
                        raw_importance
                    )
                )
            )

            if denominator == 0:
                denominator = 1.0

            importance_percent = (
                np.abs(
                    raw_importance
                )
                / denominator
                * 100.0
            )

            fold_importance_table = (
                pd.DataFrame(
                    {
                        "feature": (
                            reference_feature_names
                        ),

                        "feature_pretty": [
                            feature_to_name[
                                feature
                            ]
                            for feature
                            in (
                                reference_feature_names
                            )
                        ],

                        "group": [
                            feature_to_group[
                                feature
                            ]
                            for feature
                            in (
                                reference_feature_names
                            )
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
                            baseline_loss
                        ),
                    }
                )
                .sort_values(
                    "importance_percent",
                    ascending=False,
                )
            )

            fold_importance_table.to_csv(
                fold_output_dir
                / (
                    "feature_importance.csv"
                ),
                index=False,
            )

            plot_fold_importance(
                table=(
                    fold_importance_table
                ),
                save_path=(
                    fold_output_dir
                    / (
                        "feature_importance_"
                        "top15.png"
                    )
                ),
                title=(
                    "Independent Test "
                    "Permutation Importance — "
                    f"Fold {fold_number}"
                ),
                top_k=(
                    importance_top_k
                ),
            )

        # =====================================================
        # GRAD-CAM
        # =====================================================

        if (
            gradcam_enabled
            and gradcam_save_per_fold
        ):
            generate_fold_gradcams(
                model=(
                    best_model
                ),
                dataset=(
                    data_module
                    .test_dataset
                ),
                probabilities=(
                    test_probabilities
                ),
                threshold=0.5,
                output_root=(
                    gradcam_dir
                ),
                fold_number=(
                    fold_number
                ),
                device=(
                    device
                ),
                image_mean=(
                    image_mean
                ),
                image_std=(
                    image_std
                ),
                cam_sums=(
                    cam_sums
                ),
            )

        # -----------------------------------------------------
        # WandB final metrics
        # -----------------------------------------------------

        if logger is not None:
            logger.experiment.log(
                {
                    "decision_threshold": 0.5,

                    **{
                        (
                            "test/"
                            + key
                        ): value
                        for key, value
                        in test_metrics.items()
                        if isinstance(
                            value,
                            (
                                int,
                                float,
                            ),
                        )
                    },
                }
            )

            logger.experiment.finish()

            if wandb is not None:
                wandb.finish()

        # -----------------------------------------------------
        # Cleanup
        # -----------------------------------------------------

        del best_model
        del data_module

        if (
            resolved["mode"]
            == "train"
        ):
            del model
            del trainer

        clean_memory()

    # =========================================================
    # FOLD SUMMARY
    # =========================================================

    save_json(
        fold_results,
        output_dir
        / "fold_results.json",
    )

    fold_summary = (
        summarise_fold_metrics(
            fold_results
        )
    )

    # =========================================================
    # FIVE-MODEL ENSEMBLE
    # =========================================================

    probability_matrix = np.stack(
        fold_probabilities,
        axis=0,
    )

    ensemble_probabilities = (
        probability_matrix.mean(
            axis=0
        )
    )

    # Important:
    # no ensemble threshold is optimized
    # using test labels.
    ensemble_threshold = 0.5

    (
        ensemble_metrics,
        ensemble_frame,
    ) = save_evaluation(
        targets=(
            reference_test_targets
        ),
        probabilities=(
            ensemble_probabilities
        ),
        sample_ids=(
            reference_test_ids
        ),
        threshold=(
            ensemble_threshold
        ),
        output_dir=(
            output_dir
        ),
        phase=(
            "ensemble_test"
        ),
        fold=None,
    )

    ensemble_predictions = (
        ensemble_probabilities
        >= ensemble_threshold
    ).astype(int)

    # Add all five fold probabilities.
    ensemble_table = pd.DataFrame(
        {
            "ID": (
                reference_test_ids
            ),

            "true_label": (
                reference_test_targets
            ),
        }
    )

    for fold_number, probabilities in enumerate(
        fold_probabilities,
        start=1,
    ):
        ensemble_table[
            (
                f"fold{fold_number}_"
                "prob_pacemaker"
            )
        ] = probabilities

    ensemble_table[
        "ensemble_prob_pacemaker"
    ] = ensemble_probabilities

    ensemble_table[
        "ensemble_prediction"
    ] = ensemble_predictions

    ensemble_table.to_csv(
        output_dir
        / "ensemble_predictions.csv",
        index=False,
    )

    # =========================================================
    # PATIENT-LEVEL MODALITY IMPORTANCE EXCEL
    # =========================================================
    # The ensemble values are the arithmetic mean of the five fold-specific
    # relative branch-importance percentages for each patient.
    modality_table = pd.DataFrame(
        {
            "ID": reference_test_ids,
            "true_label": reference_test_targets,
        }
    )

    image_importance_matrix = np.stack(
        fold_image_importance_percentages,
        axis=0,
    )
    tabular_importance_matrix = np.stack(
        fold_tabular_importance_percentages,
        axis=0,
    )

    for fold_number in range(1, n_folds + 1):
        modality_table[
            f"Fold{fold_number}_Image_Importance_Pct"
        ] = image_importance_matrix[fold_number - 1]
        modality_table[
            f"Fold{fold_number}_Tabular_Importance_Pct"
        ] = tabular_importance_matrix[fold_number - 1]

    modality_table["Ensemble_Image_Importance_Pct"] = (
        image_importance_matrix.mean(axis=0)
    )
    modality_table["Ensemble_Tabular_Importance_Pct"] = (
        tabular_importance_matrix.mean(axis=0)
    )

    # Helpful prediction context in the same workbook.
    modality_table["Ensemble_Prob_Pacemaker"] = ensemble_probabilities
    modality_table["Ensemble_Prediction"] = ensemble_predictions

    modality_table.to_excel(
        output_dir / "test_patient_modality_importance.xlsx",
        index=False,
    )

    # =========================================================
    # AGGREGATED FEATURE IMPORTANCE
    # =========================================================

    if importance_enabled:
        aggregate_importance(
            fold_importances=(
                raw_fold_importances
            ),
            feature_names=(
                reference_feature_names
            ),
            feature_to_group=(
                feature_to_group
            ),
            feature_to_name=(
                feature_to_name
            ),
            output_dir=(
                importance_dir
            ),
            top_k=(
                importance_top_k
            ),
        )

    # =========================================================
    # AVERAGED GRAD-CAM
    # =========================================================

    if (
        gradcam_enabled
        and gradcam_save_per_fold
        and gradcam_save_ensemble
    ):
        save_ensemble_gradcams(
            dataset=(
                reference_test_dataset
            ),
            cam_sums=(
                cam_sums
            ),
            ensemble_predictions=(
                ensemble_predictions
            ),
            output_root=(
                gradcam_dir
            ),
            number_of_models=(
                n_folds
            ),
            image_mean=(
                image_mean
            ),
            image_std=(
                image_std
            ),
        )

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    summary = {
        "mode": (
            resolved[
                "mode"
            ]
        ),

        "train_dataset": (
            resolved[
                "training_dataset"
            ]
        ),

        "test_dataset": (
            resolved[
                "test_dataset"
            ]
        ),

        "data_percentage": (
            resolved[
                "percentage"
            ]
        ),

        "fold_mean_std": (
            fold_summary
        ),

        "five_model_ensemble": (
            ensemble_metrics
        ),
    }

    save_json(
        summary,
        output_dir
        / "cv_summary.json",
    )

    with (
        output_dir
        / "cv_results.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "COMBINED RADIODINO CT + TABULAR RESULTS\n"
        )

        handle.write(
            "=" * 80
            + "\n\n"
        )

        handle.write(
            "Training dataset: "
            f"{resolved['training_dataset']}\n"
        )

        handle.write(
            "Test dataset: "
            f"{resolved['test_dataset']}\n"
        )

        handle.write(
            "Data percentage: "
            f"{resolved['percentage']}%\n\n"
        )

        handle.write(
            "MEAN ± STD ACROSS FIVE "
            "FOLD MODELS\n"
        )

        handle.write(
            "-" * 80
            + "\n"
        )

        for (
            metric_name,
            values,
        ) in fold_summary.items():
            handle.write(
                f"{metric_name}: "
                f"{values['mean']:.6f} "
                f"± "
                f"{values['std']:.6f}\n"
            )

        handle.write(
            "\nFIVE-MODEL ENSEMBLE\n"
        )

        handle.write(
            "-" * 80
            + "\n"
        )

        for (
            metric_name,
            value,
        ) in ensemble_metrics.items():
            handle.write(
                f"{metric_name}: "
                f"{value}\n"
            )

    # =========================================================
    # TERMINAL
    # =========================================================

    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(
        f"Training dataset: "
        f"{resolved['training_dataset']}"
    )

    print(
        f"Test dataset:     "
        f"{resolved['test_dataset']}"
    )

    print(
        f"Training size:    "
        f"{resolved['percentage']}%"
    )

    print()
    print(
        "MEAN ± STD ACROSS "
        "FIVE FOLD MODELS"
    )

    print(
        "-" * 80
    )

    print(
        "Accuracy: "
        f"{fold_summary['accuracy']['mean']:.4f} "
        "± "
        f"{fold_summary['accuracy']['std']:.4f}"
    )

    print(
        "F1 Score: "
        f"{fold_summary['f1']['mean']:.4f} "
        "± "
        f"{fold_summary['f1']['std']:.4f}"
    )

    print(
        "AUC-ROC:  "
        f"{fold_summary['auc_roc']['mean']:.4f} "
        "± "
        f"{fold_summary['auc_roc']['std']:.4f}"
    )

    print()
    print(
        "FIVE-MODEL ENSEMBLE"
    )

    print(
        "-" * 80
    )

    print(
        "Accuracy: "
        f"{ensemble_metrics['accuracy']:.4f}"
    )

    print(
        "F1 Score: "
        f"{ensemble_metrics['f1']:.4f}"
    )

    print(
        "AUC-ROC:  "
        f"{ensemble_metrics['auc_roc']:.4f}"
    )

    print(
        f"\nResults saved to: "
        f"{output_dir}"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Combined RadioDINO CT + tabular "
            "predefined-fold experiment."
        )
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default="config.json",
    )

    parsed = parser.parse_args()

    config_path = Path(
        parsed.config_file
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            "Configuration file "
            f"not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        configuration = json.load(
            handle
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(
            0
        )

    main(
        configuration
    )