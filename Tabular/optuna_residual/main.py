#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Residual tabular MLP WITHOUT missingness masks + Optuna hyperparameter tuning.

Important experimental design
-----------------------------
Optuna NEVER sees test.xlsx.

For every Optuna trial:
    Fold i validation:
        training   = other four predefined development folds
        validation = fold i

    Objective:
        mean validation metric across all five development folds

After Optuna finishes:
    1. freeze the best hyperparameters;
    2. train five final fold models with those hyperparameters;
    3. evaluate every model on the SAME fixed independent test.xlsx;
    4. calculate a five-model probability ensemble;
    5. calculate post-hoc permutation feature importance.

This prevents test-set performance from driving hyperparameter selection.
"""

from __future__ import annotations

import gc
import json
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch

from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

from dataset import (
    TabularDataset,
    compute_binary_class_weights,
    concatenate_folds,
    fit_preprocessor,
    load_excel_file,
    load_predefined_folds,
    make_dataloader,
    save_preprocessor,
    transform_tabular,
)
from model import ResidualTabularMLP


# =============================================================================
# PLOT SETTINGS -- SAME AS TABPFN
# =============================================================================

BASE_FONTSIZE = 18
TITLE_FONTSIZE = 24
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


VALID_DATASETS = {
    "tum",
    "lmu",
    "merged",
}

VALID_DATA_PERCENTAGES = {
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
    "Electrocardiographic": "#1a1a1a",
}


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def set_global_seed(
    seed: int,
) -> None:
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    pl.seed_everything(
        seed,
        workers=True,
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed(
            seed
        )
        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_dir(
    path: Path,
) -> None:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def safe_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    if len(
        np.unique(
            y_true
        )
    ) < 2:
        return float(
            "nan"
        )

    return float(
        roc_auc_score(
            y_true,
            probabilities,
        )
    )


def safe_log_loss(
    y_true: np.ndarray,
    probabilities: np.ndarray,
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
            y_true,
            probabilities,
            labels=[
                0,
                1,
            ],
        )
    )


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Tuple[
    Dict[str, float],
    np.ndarray,
    np.ndarray,
]:
    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    confusion = confusion_matrix(
        y_true,
        predictions,
        labels=[
            0,
            1,
        ],
    )

    tn, fp, fn, tp = (
        confusion.ravel()
    )

    return (
        {
            "accuracy": float(
                accuracy_score(
                    y_true,
                    predictions,
                )
            ),
            "f1_score": float(
                f1_score(
                    y_true,
                    predictions,
                    pos_label=1,
                    zero_division=0,
                )
            ),
            "auc_roc": safe_auc(
                y_true,
                probabilities,
            ),
            "log_loss": safe_log_loss(
                y_true,
                probabilities,
            ),
            "no_event_accuracy": (
                float(
                    tn
                    / (
                        tn + fp
                    )
                )
                if (
                    tn + fp
                ) > 0
                else float(
                    "nan"
                )
            ),
            "pacer_accuracy": (
                float(
                    tp
                    / (
                        tp + fn
                    )
                )
                if (
                    tp + fn
                ) > 0
                else float(
                    "nan"
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
        },
        predictions,
        confusion,
    )


def resolve_paths(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    dataset_root = Path(
        config[
            "dataset_root"
        ]
    )

    train_dataset = str(
        config[
            "train_dataset"
        ]
    ).strip().lower()

    test_dataset = str(
        config[
            "test_dataset"
        ]
    ).strip().lower()

    data_percentage = int(
        config.get(
            "data_percentage",
            100,
        )
    )

    if train_dataset not in VALID_DATASETS:
        raise ValueError(
            f"train_dataset must be one of "
            f"{sorted(VALID_DATASETS)}."
        )

    if test_dataset not in VALID_DATASETS:
        raise ValueError(
            f"test_dataset must be one of "
            f"{sorted(VALID_DATASETS)}."
        )

    if data_percentage not in (
        VALID_DATA_PERCENTAGES
    ):
        raise ValueError(
            "data_percentage must be one of "
            f"{sorted(VALID_DATA_PERCENTAGES)}."
        )

    if data_percentage == 100:
        training_folder = (
            dataset_root
            / train_dataset
        )
    else:
        training_folder = (
            dataset_root
            / "data_size"
            / f"{data_percentage}_percent"
            / train_dataset
        )

    test_file = (
        dataset_root
        / test_dataset
        / "test.xlsx"
    )

    if not training_folder.is_dir():
        raise NotADirectoryError(
            f"Training folder does not exist: "
            f"{training_folder}"
        )

    if not test_file.exists():
        raise FileNotFoundError(
            f"Independent test file does not exist: "
            f"{test_file}"
        )

    return {
        "dataset_root": (
            dataset_root
        ),
        "train_dataset": (
            train_dataset
        ),
        "test_dataset": (
            test_dataset
        ),
        "data_percentage": (
            data_percentage
        ),
        "training_folder": (
            training_folder
        ),
        "test_file": (
            test_file
        ),
    }


# =============================================================================
# PREDICTION
# =============================================================================

@torch.no_grad()
def predict_probabilities(
    model: ResidualTabularMLP,
    X: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    actual_device = torch.device(
        "cuda"
        if (
            device.startswith(
                "cuda"
            )
            and torch.cuda.is_available()
        )
        else "cpu"
    )

    model = model.to(
        actual_device
    )

    model.eval()

    X_tensor = torch.as_tensor(
        X,
        dtype=torch.float32,
    )

    outputs: List[
        np.ndarray
    ] = []

    for start in range(
        0,
        len(X_tensor),
        batch_size,
    ):
        batch = X_tensor[
            start:
            start
            + batch_size
        ].to(
            actual_device
        )

        logits = model(
            batch
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=1,
            )[:, 1]
        )

        outputs.append(
            probabilities
            .detach()
            .cpu()
            .numpy()
        )

    return np.concatenate(
        outputs,
        axis=0,
    )


# =============================================================================
# FEATURE METADATA -- TABPFN MATCH
# =============================================================================

def load_feature_metadata(
    metadata_path: Path,
    feature_columns: Sequence[str],
) -> Tuple[
    Dict[str, str],
    Dict[str, str],
]:
    metadata = pd.read_excel(
        metadata_path
    )

    required = {
        "Feature",
        "Group",
        "Mapping Name",
    }

    missing = (
        required
        - set(
            metadata.columns
        )
    )

    if missing:
        raise ValueError(
            "Feature metadata is missing columns: "
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

    invalid_groups = (
        set(
            metadata[
                "Group"
            ]
        )
        - set(
            VALID_GROUPS
        )
    )

    if invalid_groups:
        raise ValueError(
            "Invalid feature groups in metadata: "
            f"{sorted(invalid_groups)}"
        )

    metadata_features = set(
        metadata[
            "Feature"
        ]
    )

    missing_features = (
        set(
            feature_columns
        )
        - metadata_features
    )

    if missing_features:
        raise ValueError(
            "These model features are missing from "
            "the metadata workbook:\n"
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

    feature_to_pretty_name = dict(
        zip(
            metadata[
                "Feature"
            ],
            metadata[
                "Mapping Name"
            ],
        )
    )

    return (
        feature_to_group,
        feature_to_pretty_name,
    )


# =============================================================================
# MODEL / TRAINER CREATION
# =============================================================================

def build_model(
    input_dim: int,
    parameters: Dict[
        str,
        Any
    ],
    class_weights: np.ndarray,
) -> ResidualTabularMLP:
    return ResidualTabularMLP(
        tabular_in=(
            input_dim
        ),
        hidden_dim=int(
            parameters[
                "hidden_dim"
            ]
        ),
        bottleneck_dim=int(
            parameters[
                "bottleneck_dim"
            ]
        ),
        embedding_dim=int(
            parameters[
                "embedding_dim"
            ]
        ),
        n_residual_blocks=int(
            parameters[
                "n_residual_blocks"
            ]
        ),
        dropout_rate_tabular=float(
            parameters[
                "dropout_rate_tabular"
            ]
        ),
        learning_rate=float(
            parameters[
                "learning_rate"
            ]
        ),
        weight_decay=float(
            parameters[
                "weight_decay"
            ]
        ),
        optimizer_name=str(
            parameters[
                "optimizer"
            ]
        ),
        scheduler_type=str(
            parameters[
                "scheduler_type"
            ]
        ),
        scheduler_factor=float(
            parameters.get(
                "scheduler_factor",
                0.5,
            )
        ),
        scheduler_patience=int(
            parameters.get(
                "scheduler_patience",
                5,
            )
        ),
        scheduler_min_lr=float(
            parameters.get(
                "scheduler_min_lr",
                1e-6,
            )
        ),
        class_weights=(
            class_weights
        ),
    )


def make_trainer(
    checkpoint_dir: Path,
    max_epochs: int,
    early_stopping_patience: int,
    device: str,
    logger,
    enable_progress_bar: bool,
) -> Tuple[
    pl.Trainer,
    ModelCheckpoint,
]:
    checkpoint_callback = (
        ModelCheckpoint(
            dirpath=(
                checkpoint_dir
            ),
            filename=(
                "best_model"
            ),
            monitor=(
                "val_loss"
            ),
            mode="min",
            save_top_k=1,
            save_last=False,
        )
    )

    early_stopping = (
        EarlyStopping(
            monitor=(
                "val_loss"
            ),
            mode="min",
            patience=(
                early_stopping_patience
            ),
            min_delta=0.0,
        )
    )

    accelerator = (
        "gpu"
        if (
            device.startswith(
                "cuda"
            )
            and torch.cuda.is_available()
        )
        else "cpu"
    )

    trainer = pl.Trainer(
        max_epochs=(
            max_epochs
        ),
        accelerator=(
            accelerator
        ),
        devices=1,
        deterministic=True,
        callbacks=[
            checkpoint_callback,
            early_stopping,
        ],
        logger=(
            logger
        ),
        enable_progress_bar=(
            enable_progress_bar
        ),
        enable_model_summary=False,
        log_every_n_steps=1,
    )

    return (
        trainer,
        checkpoint_callback,
    )


# =============================================================================
# OPTUNA PARAMETER SAMPLING
# =============================================================================

def sample_parameters(
    trial: optuna.Trial,
    config: Dict[
        str,
        Any
    ],
) -> Dict[str, Any]:
    """
    Search only over the compact set of parameters that are useful for this
    dataset/architecture.

    The search space comes from config.json so it is fully reproducible.
    """
    space = config[
        "optuna"
    ][
        "search_space"
    ]

    parameters = {
        "hidden_dim": (
            trial.suggest_categorical(
                "hidden_dim",
                space[
                    "hidden_dim"
                ],
            )
        ),
        "bottleneck_dim": (
            trial.suggest_categorical(
                "bottleneck_dim",
                space[
                    "bottleneck_dim"
                ],
            )
        ),
        "embedding_dim": (
            trial.suggest_categorical(
                "embedding_dim",
                space[
                    "embedding_dim"
                ],
            )
        ),
        "n_residual_blocks": (
            trial.suggest_categorical(
                "n_residual_blocks",
                space[
                    "n_residual_blocks"
                ],
            )
        ),
        "dropout_rate_tabular": (
            trial.suggest_categorical(
                "dropout_rate_tabular",
                space[
                    "dropout_rate_tabular"
                ],
            )
        ),
        "learning_rate": (
            trial.suggest_categorical(
                "learning_rate",
                space[
                    "learning_rate"
                ],
            )
        ),
        "batch_size": (
            trial.suggest_categorical(
                "batch_size",
                space[
                    "batch_size"
                ],
            )
        ),
        "optimizer": (
            trial.suggest_categorical(
                "optimizer",
                space[
                    "optimizer"
                ],
            )
        ),
        "weight_decay": (
            trial.suggest_categorical(
                "weight_decay",
                space[
                    "weight_decay"
                ],
            )
        ),
        "scheduler_type": (
            trial.suggest_categorical(
                "scheduler_type",
                space[
                    "scheduler_type"
                ],
            )
        ),
        "early_stopping_patience": (
            trial.suggest_categorical(
                "early_stopping_patience",
                space[
                    "early_stopping_patience"
                ],
            )
        ),
    }

    if (
        parameters[
            "scheduler_type"
        ]
        == "plateau"
    ):
        parameters[
            "scheduler_factor"
        ] = (
            trial.suggest_categorical(
                "scheduler_factor",
                space[
                    "scheduler_factor"
                ],
            )
        )

        parameters[
            "scheduler_patience"
        ] = (
            trial.suggest_categorical(
                "scheduler_patience",
                space[
                    "scheduler_patience"
                ],
            )
        )

    else:
        parameters[
            "scheduler_factor"
        ] = 0.5

        parameters[
            "scheduler_patience"
        ] = 5

    parameters[
        "scheduler_min_lr"
    ] = float(
        config[
            "training"
        ].get(
            "scheduler_min_lr",
            1e-6,
        )
    )

    return parameters


# =============================================================================
# ONE FOLD TRAINING
# =============================================================================

def prepare_fold_data(
    folds: Sequence[
        Dict[str, Any]
    ],
    fold_number: int,
    feature_columns: Sequence[str],
) -> Dict[str, Any]:
    raw_train, y_train, train_ids = (
        concatenate_folds(
            folds,
            excluded_fold=(
                fold_number
            ),
        )
    )

    validation_fold = folds[
        fold_number
        - 1
    ]

    raw_validation = (
        validation_fold[
            "raw_X"
        ]
    )

    y_validation = (
        validation_fold[
            "y"
        ]
    )

    validation_ids = (
        validation_fold[
            "ids"
        ]
    )

    (
        preprocessing_state,
        X_train,
    ) = fit_preprocessor(
        raw_train,
        feature_columns,
    )

    X_validation = (
        transform_tabular(
            raw_validation,
            preprocessing_state,
        )
    )

    class_weights = (
        compute_binary_class_weights(
            y_train
        )
    )

    return {
        "X_train": (
            X_train
        ),
        "y_train": (
            y_train
        ),
        "train_ids": (
            train_ids
        ),
        "X_validation": (
            X_validation
        ),
        "y_validation": (
            y_validation
        ),
        "validation_ids": (
            validation_ids
        ),
        "preprocessing_state": (
            preprocessing_state
        ),
        "class_weights": (
            class_weights
        ),
    }


def train_and_evaluate_validation_fold(
    folds: Sequence[
        Dict[str, Any]
    ],
    fold_number: int,
    feature_columns: Sequence[str],
    parameters: Dict[
        str,
        Any
    ],
    config: Dict[
        str,
        Any
    ],
    seed: int,
) -> Dict[str, float]:
    """
    Used only during Optuna.

    No test-set data enters this function.
    """
    set_global_seed(
        seed
    )

    data = prepare_fold_data(
        folds=(
            folds
        ),
        fold_number=(
            fold_number
        ),
        feature_columns=(
            feature_columns
        ),
    )

    batch_size = int(
        parameters[
            "batch_size"
        ]
    )

    train_loader = (
        make_dataloader(
            TabularDataset(
                data[
                    "X_train"
                ],
                data[
                    "y_train"
                ],
                data[
                    "train_ids"
                ],
            ),
            batch_size=(
                batch_size
            ),
            shuffle=True,
            num_workers=int(
                config[
                    "training"
                ][
                    "num_workers"
                ]
            ),
        )
    )

    validation_loader = (
        make_dataloader(
            TabularDataset(
                data[
                    "X_validation"
                ],
                data[
                    "y_validation"
                ],
                data[
                    "validation_ids"
                ],
            ),
            batch_size=(
                batch_size
            ),
            shuffle=False,
            num_workers=int(
                config[
                    "training"
                ][
                    "num_workers"
                ]
            ),
        )
    )

    with tempfile.TemporaryDirectory(
        prefix=(
            f"resmlp_trial_fold"
            f"{fold_number}_"
        )
    ) as temporary_directory:
        temp_path = Path(
            temporary_directory
        )

        model = build_model(
            input_dim=int(
                data[
                    "preprocessing_state"
                ][
                    "model_input_dim"
                ]
            ),
            parameters=(
                parameters
            ),
            class_weights=(
                data[
                    "class_weights"
                ]
            ),
        )

        (
            trainer,
            checkpoint_callback,
        ) = make_trainer(
            checkpoint_dir=(
                temp_path
            ),
            max_epochs=int(
                config[
                    "training"
                ][
                    "max_epochs"
                ]
            ),
            early_stopping_patience=int(
                parameters[
                    "early_stopping_patience"
                ]
            ),
            device=str(
                config[
                    "device"
                ]
            ),
            logger=False,
            enable_progress_bar=False,
        )

        trainer.fit(
            model,
            train_dataloaders=(
                train_loader
            ),
            val_dataloaders=(
                validation_loader
            ),
        )

        best_checkpoint = Path(
            checkpoint_callback.best_model_path
        )

        if not best_checkpoint.exists():
            raise RuntimeError(
                "Optuna fold did not create a best checkpoint."
            )

        model = (
            ResidualTabularMLP
            .load_from_checkpoint(
                best_checkpoint,
                class_weights=(
                    data[
                        "class_weights"
                    ]
                ),
            )
        )

        probabilities = (
            predict_probabilities(
                model=model,
                X=(
                    data[
                        "X_validation"
                    ]
                ),
                batch_size=(
                    batch_size
                ),
                device=str(
                    config[
                        "device"
                    ]
                ),
            )
        )

        metrics, _, _ = (
            calculate_metrics(
                y_true=(
                    data[
                        "y_validation"
                    ]
                ),
                probabilities=(
                    probabilities
                ),
                threshold=float(
                    config[
                        "evaluation"
                    ].get(
                        "threshold",
                        0.5,
                    )
                ),
            )
        )

        metrics[
            "best_val_loss_logged"
        ] = float(
            checkpoint_callback.best_model_score
            .detach()
            .cpu()
            .item()
        )

        metrics[
            "epochs_trained"
        ] = int(
            trainer.current_epoch
            + 1
        )

        del model
        del trainer
        clean_memory()

    return metrics


# =============================================================================
# OPTUNA OBJECTIVE
# =============================================================================

def make_objective(
    folds: Sequence[
        Dict[str, Any]
    ],
    feature_columns: Sequence[str],
    config: Dict[
        str,
        Any
    ],
):
    objective_metric = str(
        config[
            "optuna"
        ].get(
            "objective_metric",
            "auc_roc",
        )
    )

    valid_metrics = {
        "auc_roc",
        "accuracy",
        "f1_score",
        "log_loss",
    }

    if objective_metric not in (
        valid_metrics
    ):
        raise ValueError(
            "optuna.objective_metric must be one of "
            f"{sorted(valid_metrics)}."
        )

    def objective(
        trial: optuna.Trial,
    ) -> float:
        parameters = (
            sample_parameters(
                trial,
                config,
            )
        )

        fold_values: List[
            float
        ] = []

        fold_metrics: List[
            Dict[str, float]
        ] = []

        for fold_number in range(
            1,
            int(
                config[
                    "n_folds"
                ]
            )
            + 1,
        ):
            fold_seed = (
                int(
                    config[
                        "random_seed"
                    ]
                )
                + trial.number
                * 1000
                + fold_number
            )

            metrics = (
                train_and_evaluate_validation_fold(
                    folds=(
                        folds
                    ),
                    fold_number=(
                        fold_number
                    ),
                    feature_columns=(
                        feature_columns
                    ),
                    parameters=(
                        parameters
                    ),
                    config=(
                        config
                    ),
                    seed=(
                        fold_seed
                    ),
                )
            )

            fold_metrics.append(
                metrics
            )

            fold_values.append(
                float(
                    metrics[
                        objective_metric
                    ]
                )
            )

            current_mean = float(
                np.nanmean(
                    fold_values
                )
            )

            # Optuna pruning assumes larger-is-better for the reported value
            # here. Invert log loss when necessary.
            report_value = (
                -current_mean
                if (
                    objective_metric
                    == "log_loss"
                )
                else current_mean
            )

            trial.report(
                report_value,
                step=(
                    fold_number
                ),
            )

            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_value = float(
            np.nanmean(
                fold_values
            )
        )

        std_value = float(
            np.nanstd(
                fold_values
            )
        )

        trial.set_user_attr(
            "fold_metrics",
            fold_metrics,
        )

        trial.set_user_attr(
            "objective_mean",
            mean_value,
        )

        trial.set_user_attr(
            "objective_std",
            std_value,
        )

        return mean_value

    return objective


# =============================================================================
# TRAINING CURVES
# =============================================================================

def save_training_curves(
    trainer: pl.Trainer,
    fold_output: Path,
) -> None:
    logger = trainer.logger

    if (
        logger is None
        or not hasattr(
            logger,
            "log_dir",
        )
    ):
        return

    metrics_path = (
        Path(
            logger.log_dir
        )
        / "metrics.csv"
    )

    if not metrics_path.exists():
        return

    metrics = pd.read_csv(
        metrics_path
    )

    required = {
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
    }

    if not required.issubset(
        metrics.columns
    ):
        return

    def last_non_missing(
        series: pd.Series,
    ):
        values = series.dropna()

        if values.empty:
            return np.nan

        return values.iloc[-1]

    history = (
        metrics.loc[
            metrics[
                "epoch"
            ].notna()
        ]
        .groupby(
            "epoch",
            as_index=False,
        )
        .agg(
            {
                "train_loss": (
                    last_non_missing
                ),
                "train_accuracy": (
                    last_non_missing
                ),
                "val_loss": (
                    last_non_missing
                ),
                "val_accuracy": (
                    last_non_missing
                ),
            }
        )
        .sort_values(
            "epoch"
        )
        .reset_index(
            drop=True
        )
    )

    history = history.dropna(
        subset=[
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
        ]
    ).copy()

    if history.empty:
        return

    history[
        "epoch"
    ] = (
        history[
            "epoch"
        ].astype(int)
        + 1
    )

    history.to_csv(
        fold_output
        / "training_history.csv",
        index=False,
    )

    # Loss
    figure, axis = plt.subplots(
        figsize=(
            10,
            7,
        )
    )

    axis.plot(
        history[
            "epoch"
        ],
        history[
            "train_loss"
        ],
        linewidth=2.5,
        label="Training",
    )

    axis.plot(
        history[
            "epoch"
        ],
        history[
            "val_loss"
        ],
        linewidth=2.5,
        label="Validation",
    )

    axis.set_xlabel(
        "Epoch"
    )
    axis.set_ylabel(
        "Cross-entropy loss"
    )
    axis.set_title(
        "Training and Validation Loss",
        pad=18,
    )
    axis.legend(
        frameon=False
    )
    axis.spines[
        "top"
    ].set_visible(
        False
    )
    axis.spines[
        "right"
    ].set_visible(
        False
    )
    axis.grid(
        axis="y",
        linestyle=":",
        linewidth=1.0,
        alpha=0.5,
    )

    figure.tight_layout()

    figure.savefig(
        fold_output
        / "training_validation_loss.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # Accuracy
    figure, axis = plt.subplots(
        figsize=(
            10,
            7,
        )
    )

    axis.plot(
        history[
            "epoch"
        ],
        history[
            "train_accuracy"
        ]
        * 100.0,
        linewidth=2.5,
        label="Training",
    )

    axis.plot(
        history[
            "epoch"
        ],
        history[
            "val_accuracy"
        ]
        * 100.0,
        linewidth=2.5,
        label="Validation",
    )

    axis.set_xlabel(
        "Epoch"
    )
    axis.set_ylabel(
        "Accuracy (%)"
    )
    axis.set_ylim(
        0,
        100,
    )
    axis.set_title(
        "Training and Validation Accuracy",
        pad=18,
    )
    axis.legend(
        frameon=False
    )
    axis.spines[
        "top"
    ].set_visible(
        False
    )
    axis.spines[
        "right"
    ].set_visible(
        False
    )
    axis.grid(
        axis="y",
        linestyle=":",
        linewidth=1.0,
        alpha=0.5,
    )

    figure.tight_layout()

    figure.savefig(
        fold_output
        / "training_validation_accuracy.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# STANDARD PLOTS
# =============================================================================

def save_confusion_matrix(
    confusion: np.ndarray,
    path: Path,
    title: str,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            7,
            6,
        )
    )

    axis.imshow(
        confusion,
        cmap="Greys",
    )

    for row in range(
        2
    ):
        for column in range(
            2
        ):
            axis.text(
                column,
                row,
                str(
                    confusion[
                        row,
                        column,
                    ]
                ),
                ha="center",
                va="center",
                fontsize=18,
            )

    axis.set_xticks(
        [
            0,
            1,
        ],
        [
            "No event",
            "Pacer",
        ],
    )

    axis.set_yticks(
        [
            0,
            1,
        ],
        [
            "No event",
            "Pacer",
        ],
    )

    axis.set_xlabel(
        "Predicted label"
    )

    axis.set_ylabel(
        "True label"
    )

    axis.set_title(
        title
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_calibration_plot(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    path: Path,
    title: str,
    number_of_bins: int,
) -> None:
    fraction_positive, mean_predicted = (
        calibration_curve(
            y_true,
            probabilities,
            n_bins=(
                number_of_bins
            ),
            strategy="uniform",
        )
    )

    figure, axis = plt.subplots(
        figsize=(
            7,
            7,
        )
    )

    axis.plot(
        mean_predicted,
        fraction_positive,
        "o-",
        label="Model",
    )

    axis.plot(
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
        "--",
        color="gray",
        label="Perfect calibration",
    )

    axis.set_xlabel(
        "Mean predicted probability"
    )

    axis.set_ylabel(
        "Fraction of positives"
    )

    axis.set_title(
        title
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def calculate_permutation_importance(
    model: ResidualTabularMLP,
    X_test: np.ndarray,
    y_test: np.ndarray,
    number_of_repeats: int,
    seed: int,
    batch_size: int,
    device: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    float,
]:
    """
    No-mask model:
        one model column == one original clinical feature.

    Importance = increase in test log loss after permuting that feature.
    """
    baseline_probabilities = (
        predict_probabilities(
            model=model,
            X=X_test,
            batch_size=(
                batch_size
            ),
            device=(
                device
            ),
        )
    )

    baseline_loss = safe_log_loss(
        y_test,
        baseline_probabilities,
    )

    number_of_features = (
        X_test.shape[
            1
        ]
    )

    importance_deltas = np.zeros(
        number_of_features,
        dtype=np.float64,
    )

    repeat_std = np.zeros(
        number_of_features,
        dtype=np.float64,
    )

    for feature_index in range(
        number_of_features
    ):
        repeat_values: List[
            float
        ] = []

        for repeat_number in range(
            number_of_repeats
        ):
            rng = np.random.RandomState(
                seed
                + feature_index
                * 1009
                + repeat_number
            )

            permutation = rng.permutation(
                len(
                    X_test
                )
            )

            permuted = X_test.copy()

            permuted[
                :,
                feature_index,
            ] = X_test[
                permutation,
                feature_index,
            ]

            probabilities = (
                predict_probabilities(
                    model=model,
                    X=permuted,
                    batch_size=(
                        batch_size
                    ),
                    device=(
                        device
                    ),
                )
            )

            permuted_loss = safe_log_loss(
                y_test,
                probabilities,
            )

            repeat_values.append(
                permuted_loss
                - baseline_loss
            )

        importance_deltas[
            feature_index
        ] = float(
            np.mean(
                repeat_values
            )
        )

        if number_of_repeats > 1:
            repeat_std[
                feature_index
            ] = float(
                np.std(
                    repeat_values,
                    ddof=1,
                )
            )

    return (
        importance_deltas,
        repeat_std,
        baseline_loss,
    )


def plot_fold_feature_importance(
    table: pd.DataFrame,
    path: Path,
    title: str,
) -> None:
    top = (
        table
        .nlargest(
            15,
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
        in top[
            "group"
        ]
    ]

    plt.figure(
        figsize=(
            14,
            11,
        )
    )

    plt.barh(
        top[
            "feature_pretty"
        ],
        top[
            "importance_percent"
        ],
        color=(
            colors
        ),
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
        path,
        dpi=300,
    )

    plt.close()


def plot_aggregate_feature_importance(
    table: pd.DataFrame,
    error_column: str,
    path: Path,
    title: str,
) -> None:
    top = (
        table
        .nlargest(
            15,
            "importance_mean",
        )
        .sort_values(
            "importance_mean",
            ascending=True,
        )
    )

    positions = np.arange(
        len(
            top
        )
    )

    colors = [
        GROUP_COLORS[
            group
        ]
        for group
        in top[
            "group"
        ]
    ]

    plt.figure(
        figsize=(
            15,
            12,
        )
    )

    plt.barh(
        positions,
        top[
            "importance_mean"
        ],
        xerr=(
            top[
                error_column
            ]
        ),
        color=(
            colors
        ),
        edgecolor="black",
        linewidth=1.0,
        capsize=6,
    )

    plt.yticks(
        positions,
        top[
            "feature_pretty"
        ],
    )

    plt.xlabel(
        "Importance (%)"
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
    )

    plt.close()


def plot_group_importance(
    table: pd.DataFrame,
    mean_column: str,
    std_column: str,
    path: Path,
    title: str,
) -> None:
    sorted_table = (
        table.sort_values(
            mean_column,
            ascending=True,
        )
    )

    positions = np.arange(
        len(
            sorted_table
        )
    )

    colors = [
        GROUP_COLORS[
            group
        ]
        for group
        in sorted_table[
            "group"
        ]
    ]

    plt.figure(
        figsize=(
            13,
            9,
        )
    )

    plt.barh(
        positions,
        sorted_table[
            mean_column
        ],
        xerr=(
            sorted_table[
                std_column
            ]
        ),
        color=(
            colors
        ),
        edgecolor="black",
        linewidth=1.2,
        capsize=7,
    )

    plt.yticks(
        positions,
        sorted_table[
            "group"
        ],
    )

    plt.xlabel(
        "Importance (%)"
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
    )

    plt.close()


def aggregate_feature_importance(
    raw_importances: List[
        np.ndarray
    ],
    repeat_standard_deviations: List[
        np.ndarray
    ],
    feature_columns: Sequence[str],
    feature_to_group: Dict[
        str,
        str
    ],
    feature_to_pretty_name: Dict[
        str,
        str
    ],
    output_folder: Path,
) -> pd.DataFrame:
    raw_matrix = np.vstack(
        raw_importances
    )

    repeat_std_matrix = np.vstack(
        repeat_standard_deviations
    )

    mean_raw = raw_matrix.mean(
        axis=0
    )

    if raw_matrix.shape[
        0
    ] > 1:
        fold_raw_std = raw_matrix.std(
            axis=0,
            ddof=1,
        )
    else:
        fold_raw_std = np.zeros(
            raw_matrix.shape[
                1
            ]
        )

    denominator = float(
        np.sum(
            np.abs(
                mean_raw
            )
        )
    ) or 1.0

    importance_mean = (
        100.0
        * np.abs(
            mean_raw
        )
        / denominator
    )

    fold_std = (
        100.0
        * np.abs(
            fold_raw_std
        )
        / denominator
    )

    repeat_std = (
        100.0
        * np.abs(
            repeat_std_matrix.mean(
                axis=0
            )
        )
        / denominator
    )

    feature_table = (
        pd.DataFrame(
            {
                "feature": (
                    feature_columns
                ),
                "feature_pretty": [
                    feature_to_pretty_name[
                        feature
                    ]
                    for feature
                    in feature_columns
                ],
                "group": [
                    feature_to_group[
                        feature
                    ]
                    for feature
                    in feature_columns
                ],
                "importance_mean": (
                    importance_mean
                ),
                "repeat_std": (
                    repeat_std
                ),
                "fold_std": (
                    fold_std
                ),
                "raw_mean": (
                    mean_raw
                ),
                "raw_fold_std": (
                    fold_raw_std
                ),
            }
        )
        .sort_values(
            "importance_mean",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    feature_table.to_csv(
        output_folder
        / "feature_importance_average.csv",
        index=False,
    )

    plot_aggregate_feature_importance(
        table=(
            feature_table
        ),
        error_column=(
            "repeat_std"
        ),
        path=(
            output_folder
            / "features_top15_repeat_std.png"
        ),
        title=(
            "Feature Importances — "
            "Permutation Repeat Variability"
        ),
    )

    plot_aggregate_feature_importance(
        table=(
            feature_table
        ),
        error_column=(
            "fold_std"
        ),
        path=(
            output_folder
            / "features_top15_fold_std.png"
        ),
        title=(
            "Feature Importances — "
            "Fold Model Variability"
        ),
    )

    group_total_rows: List[
        List[float]
    ] = []

    group_average_rows: List[
        List[float]
    ] = []

    for fold_importance in raw_matrix:
        absolute = np.abs(
            fold_importance
        )

        fold_denominator = float(
            absolute.sum()
        ) or 1.0

        fold_percentages = (
            100.0
            * absolute
            / fold_denominator
        )

        totals: List[
            float
        ] = []

        averages: List[
            float
        ] = []

        for group in VALID_GROUPS:
            indices = [
                index
                for index, feature
                in enumerate(
                    feature_columns
                )
                if (
                    feature_to_group[
                        feature
                    ]
                    == group
                )
            ]

            if indices:
                totals.append(
                    float(
                        fold_percentages[
                            indices
                        ].sum()
                    )
                )

                averages.append(
                    float(
                        fold_percentages[
                            indices
                        ].mean()
                    )
                )

            else:
                totals.append(
                    0.0
                )

                averages.append(
                    0.0
                )

        group_total_rows.append(
            totals
        )

        group_average_rows.append(
            averages
        )

    total_matrix = np.asarray(
        group_total_rows,
        dtype=np.float64,
    )

    average_matrix = np.asarray(
        group_average_rows,
        dtype=np.float64,
    )

    ddof = (
        1
        if (
            raw_matrix.shape[
                0
            ]
            > 1
        )
        else 0
    )

    group_table = pd.DataFrame(
        {
            "group": (
                VALID_GROUPS
            ),
            "total_importance_mean": (
                total_matrix.mean(
                    axis=0
                )
            ),
            "total_importance_std": (
                total_matrix.std(
                    axis=0,
                    ddof=(
                        ddof
                    ),
                )
            ),
            "average_per_feature_mean": (
                average_matrix.mean(
                    axis=0
                )
            ),
            "average_per_feature_std": (
                average_matrix.std(
                    axis=0,
                    ddof=(
                        ddof
                    ),
                )
            ),
        }
    )

    group_table.to_csv(
        output_folder
        / "group_importance_average.csv",
        index=False,
    )

    plot_group_importance(
        table=(
            group_table
        ),
        mean_column=(
            "total_importance_mean"
        ),
        std_column=(
            "total_importance_std"
        ),
        path=(
            output_folder
            / "groups_total_importance.png"
        ),
        title=(
            "Group Total Importance"
        ),
    )

    plot_group_importance(
        table=(
            group_table
        ),
        mean_column=(
            "average_per_feature_mean"
        ),
        std_column=(
            "average_per_feature_std"
        ),
        path=(
            output_folder
            / "groups_average_per_feature.png"
        ),
        title=(
            "Group Average Importance per Feature"
        ),
    )

    return feature_table


# =============================================================================
# FINAL TRAINING WITH BEST HYPERPARAMETERS
# =============================================================================

def run_final_models(
    folds: Sequence[
        Dict[str, Any]
    ],
    feature_columns: Sequence[str],
    test_data: Dict[str, Any],
    best_parameters: Dict[
        str,
        Any
    ],
    config: Dict[
        str,
        Any
    ],
    output_folder: Path,
    feature_to_group: Dict[
        str,
        str
    ],
    feature_to_pretty_name: Dict[
        str,
        str
    ],
) -> Dict[str, Any]:
    threshold = float(
        config[
            "evaluation"
        ].get(
            "threshold",
            0.5,
        )
    )

    calibration_bins = int(
        config[
            "evaluation"
        ].get(
            "calibration_bins",
            10,
        )
    )

    device = str(
        config[
            "device"
        ]
    )

    fold_metrics: List[
        Dict[str, Any]
    ] = []

    probability_columns: Dict[
        str,
        np.ndarray
    ] = {}

    raw_importances: List[
        np.ndarray
    ] = []

    repeat_importance_stds: List[
        np.ndarray
    ] = []

    for fold_number in range(
        1,
        int(
            config[
                "n_folds"
            ]
        )
        + 1,
    ):
        print(
            "\n"
            + "=" * 80
        )

        print(
            f"FINAL MODEL "
            f"{fold_number}/"
            f"{config['n_folds']}"
        )

        print(
            "=" * 80
        )

        fold_seed = (
            int(
                config[
                    "random_seed"
                ]
            )
            + fold_number
        )

        set_global_seed(
            fold_seed
        )

        data = prepare_fold_data(
            folds=(
                folds
            ),
            fold_number=(
                fold_number
            ),
            feature_columns=(
                feature_columns
            ),
        )

        X_test = transform_tabular(
            test_data[
                "raw_X"
            ],
            data[
                "preprocessing_state"
            ],
        )

        batch_size = int(
            best_parameters[
                "batch_size"
            ]
        )

        train_loader = make_dataloader(
            TabularDataset(
                data[
                    "X_train"
                ],
                data[
                    "y_train"
                ],
                data[
                    "train_ids"
                ],
            ),
            batch_size=(
                batch_size
            ),
            shuffle=True,
            num_workers=int(
                config[
                    "training"
                ][
                    "num_workers"
                ]
            ),
        )

        validation_loader = make_dataloader(
            TabularDataset(
                data[
                    "X_validation"
                ],
                data[
                    "y_validation"
                ],
                data[
                    "validation_ids"
                ],
            ),
            batch_size=(
                batch_size
            ),
            shuffle=False,
            num_workers=int(
                config[
                    "training"
                ][
                    "num_workers"
                ]
            ),
        )

        fold_output = (
            output_folder
            / f"fold_{fold_number}"
        )

        ensure_dir(
            fold_output
        )

        save_preprocessor(
            data[
                "preprocessing_state"
            ],
            fold_output
            / "tabular_preprocessor.json",
        )

        (
            fold_output
            / "used_tabular_columns.txt"
        ).write_text(
            "\n".join(
                feature_columns
            )
            + "\n",
            encoding="utf-8",
        )

        model = build_model(
            input_dim=int(
                data[
                    "preprocessing_state"
                ][
                    "model_input_dim"
                ]
            ),
            parameters=(
                best_parameters
            ),
            class_weights=(
                data[
                    "class_weights"
                ]
            ),
        )

        logger = pl.loggers.CSVLogger(
            save_dir=str(
                fold_output
            ),
            name=(
                "training_log"
            ),
        )

        (
            trainer,
            checkpoint_callback,
        ) = make_trainer(
            checkpoint_dir=(
                fold_output
            ),
            max_epochs=int(
                config[
                    "training"
                ][
                    "max_epochs"
                ]
            ),
            early_stopping_patience=int(
                best_parameters[
                    "early_stopping_patience"
                ]
            ),
            device=(
                device
            ),
            logger=(
                logger
            ),
            enable_progress_bar=True,
        )

        trainer.fit(
            model,
            train_dataloaders=(
                train_loader
            ),
            val_dataloaders=(
                validation_loader
            ),
        )

        save_training_curves(
            trainer=(
                trainer
            ),
            fold_output=(
                fold_output
            ),
        )

        best_checkpoint = Path(
            checkpoint_callback.best_model_path
        )

        model = (
            ResidualTabularMLP
            .load_from_checkpoint(
                best_checkpoint,
                class_weights=(
                    data[
                        "class_weights"
                    ]
                ),
            )
        )

        probabilities = predict_probabilities(
            model=model,
            X=X_test,
            batch_size=(
                batch_size
            ),
            device=(
                device
            ),
        )

        (
            metrics,
            predictions,
            confusion,
        ) = calculate_metrics(
            y_true=(
                test_data[
                    "y"
                ]
            ),
            probabilities=(
                probabilities
            ),
            threshold=(
                threshold
            ),
        )

        metrics[
            "fold_model"
        ] = fold_number

        metrics[
            "best_validation_loss"
        ] = float(
            checkpoint_callback
            .best_model_score
            .detach()
            .cpu()
            .item()
        )

        metrics[
            "epochs_trained"
        ] = int(
            trainer.current_epoch
            + 1
        )

        fold_metrics.append(
            metrics
        )

        probability_columns[
            f"Prob_FoldModel_"
            f"{fold_number}"
        ] = probabilities

        pd.DataFrame(
            {
                "ID": (
                    test_data[
                        "ids"
                    ]
                ),
                "True": (
                    test_data[
                        "y"
                    ]
                ),
                "Pred": (
                    predictions
                ),
                "Prob": (
                    probabilities
                ),
                "Correct": (
                    predictions
                    == test_data[
                        "y"
                    ]
                ),
            }
        ).to_excel(
            fold_output
            / "test_predictions.xlsx",
            index=False,
        )

        save_confusion_matrix(
            confusion=(
                confusion
            ),
            path=(
                fold_output
                / "confusion_matrix.png"
            ),
            title=(
                "Independent Test Confusion Matrix — "
                f"Fold Model {fold_number}"
            ),
        )

        save_calibration_plot(
            y_true=(
                test_data[
                    "y"
                ]
            ),
            probabilities=(
                probabilities
            ),
            path=(
                fold_output
                / "calibration.png"
            ),
            title=(
                "Independent Test Calibration — "
                f"Fold Model {fold_number}"
            ),
            number_of_bins=(
                calibration_bins
            ),
        )

        if bool(
            config[
                "feature_importance"
            ].get(
                "enabled",
                True,
            )
        ):
            (
                raw_importance,
                repeat_std,
                baseline_loss,
            ) = calculate_permutation_importance(
                model=model,
                X_test=(
                    X_test
                ),
                y_test=(
                    test_data[
                        "y"
                    ]
                ),
                number_of_repeats=int(
                    config[
                        "feature_importance"
                    ].get(
                        "n_repeats",
                        5,
                    )
                ),
                seed=(
                    fold_seed
                ),
                batch_size=(
                    batch_size
                ),
                device=(
                    device
                ),
            )

            raw_importances.append(
                raw_importance
            )

            repeat_importance_stds.append(
                repeat_std
            )

            denominator = float(
                np.sum(
                    np.abs(
                        raw_importance
                    )
                )
            ) or 1.0

            importance_percent = (
                100.0
                * np.abs(
                    raw_importance
                )
                / denominator
            )

            fold_importance_table = (
                pd.DataFrame(
                    {
                        "feature": (
                            feature_columns
                        ),
                        "feature_pretty": [
                            feature_to_pretty_name[
                                feature
                            ]
                            for feature
                            in feature_columns
                        ],
                        "group": [
                            feature_to_group[
                                feature
                            ]
                            for feature
                            in feature_columns
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
                fold_output
                / "feature_importance.csv",
                index=False,
            )

            plot_fold_feature_importance(
                table=(
                    fold_importance_table
                ),
                path=(
                    fold_output
                    / "feature_importance_top15.png"
                ),
                title=(
                    "Independent Test Feature Importance — "
                    f"Fold Model {fold_number}"
                ),
            )

        del model
        del trainer
        clean_memory()

    fold_metrics_table = pd.DataFrame(
        fold_metrics
    )

    fold_metrics_table.to_csv(
        output_folder
        / "fold_metrics.csv",
        index=False,
    )

    metric_names = [
        "accuracy",
        "f1_score",
        "auc_roc",
        "log_loss",
        "pacer_accuracy",
        "no_event_accuracy",
    ]

    mean_std = {}

    for metric_name in metric_names:
        values = (
            fold_metrics_table[
                metric_name
            ]
            .astype(float)
            .to_numpy()
        )

        mean_std[
            metric_name
        ] = {
            "mean": float(
                np.nanmean(
                    values
                )
            ),
            "std": float(
                np.nanstd(
                    values
                )
            ),
        }

    probability_matrix = np.column_stack(
        list(
            probability_columns.values()
        )
    )

    ensemble_probabilities = (
        probability_matrix.mean(
            axis=1
        )
    )

    (
        ensemble_metrics,
        ensemble_predictions,
        ensemble_confusion,
    ) = calculate_metrics(
        y_true=(
            test_data[
                "y"
            ]
        ),
        probabilities=(
            ensemble_probabilities
        ),
        threshold=(
            threshold
        ),
    )

    pd.DataFrame(
        {
            "ID": (
                test_data[
                    "ids"
                ]
            ),
            "True": (
                test_data[
                    "y"
                ]
            ),
            **probability_columns,
            "EnsembleProb": (
                ensemble_probabilities
            ),
            "EnsemblePred": (
                ensemble_predictions
            ),
        }
    ).to_excel(
        output_folder
        / "ensemble_test_predictions.xlsx",
        index=False,
    )

    save_confusion_matrix(
        confusion=(
            ensemble_confusion
        ),
        path=(
            output_folder
            / "confusion_matrix_ensemble.png"
        ),
        title=(
            "Independent Test Confusion Matrix — "
            "Five-Model Ensemble"
        ),
    )

    feature_table = None

    if (
        raw_importances
        and bool(
            config[
                "feature_importance"
            ].get(
                "enabled",
                True,
            )
        )
    ):
        importance_folder = (
            output_folder
            / "feature_importance"
        )

        ensure_dir(
            importance_folder
        )

        feature_table = (
            aggregate_feature_importance(
                raw_importances=(
                    raw_importances
                ),
                repeat_standard_deviations=(
                    repeat_importance_stds
                ),
                feature_columns=(
                    feature_columns
                ),
                feature_to_group=(
                    feature_to_group
                ),
                feature_to_pretty_name=(
                    feature_to_pretty_name
                ),
                output_folder=(
                    importance_folder
                ),
            )
        )

    return {
        "fold_metrics": (
            fold_metrics_table
        ),
        "mean_std": (
            mean_std
        ),
        "ensemble_metrics": (
            ensemble_metrics
        ),
        "feature_table": (
            feature_table
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    with open(
        "config.json",
        "r",
        encoding="utf-8",
    ) as handle:
        config = json.load(
            handle
        )

    resolved = resolve_paths(
        config
    )

    if int(
        config[
            "n_folds"
        ]
    ) != 5:
        raise ValueError(
            "This project expects exactly five predefined folds."
        )

    output_folder = (
        Path(
            config[
                "output_dir"
            ]
        )
        / (
            f"{resolved['data_percentage']}"
            "_percent"
        )
    )

    ensure_dir(
        output_folder
    )

    set_global_seed(
        int(
            config[
                "random_seed"
            ]
        )
    )

    # -------------------------------------------------------------------------
    # Load DEVELOPMENT folds only. Test data is intentionally not loaded yet.
    # -------------------------------------------------------------------------
    folds, feature_columns = (
        load_predefined_folds(
            dataset_folder=(
                resolved[
                    "training_folder"
                ]
            ),
            number_of_folds=int(
                config[
                    "n_folds"
                ]
            ),
            id_column=(
                config[
                    "id_col"
                ]
            ),
            label_column=(
                config[
                    "label_col"
                ]
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Feature metadata
    # -------------------------------------------------------------------------
    metadata_path = Path(
        config[
            "feature_importance"
        ][
            "metadata_path"
        ]
    )

    (
        feature_to_group,
        feature_to_pretty_name,
    ) = load_feature_metadata(
        metadata_path=(
            metadata_path
        ),
        feature_columns=(
            feature_columns
        ),
    )

    # -------------------------------------------------------------------------
    # Optuna
    # -------------------------------------------------------------------------
    objective_metric = str(
        config[
            "optuna"
        ].get(
            "objective_metric",
            "auc_roc",
        )
    )

    direction = (
        "minimize"
        if objective_metric
        == "log_loss"
        else "maximize"
    )

    sampler = TPESampler(
        seed=int(
            config[
                "optuna"
            ].get(
                "sampler_seed",
                config[
                    "random_seed"
                ],
            )
        )
    )

    pruner = MedianPruner(
        n_startup_trials=int(
            config[
                "optuna"
            ].get(
                "n_startup_trials",
                5,
            )
        ),
        n_warmup_steps=int(
            config[
                "optuna"
            ].get(
                "n_warmup_folds",
                2,
            )
        ),
    )

    study = optuna.create_study(
        direction=(
            direction
        ),
        sampler=(
            sampler
        ),
        pruner=(
            pruner
        ),
        study_name=(
            config[
                "optuna"
            ].get(
                "study_name",
                "residual_mlp_no_mask",
            )
        ),
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "OPTUNA HYPERPARAMETER TUNING"
    )

    print(
        "=" * 80
    )

    print(
        "IMPORTANT: Optuna uses development folds only; "
        "test.xlsx is not used for tuning."
    )

    study.optimize(
        make_objective(
            folds=(
                folds
            ),
            feature_columns=(
                feature_columns
            ),
            config=(
                config
            ),
        ),
        n_trials=int(
            config[
                "optuna"
            ][
                "n_trials"
            ]
        ),
        timeout=(
            config[
                "optuna"
            ].get(
                "timeout_seconds",
                None,
            )
        ),
        gc_after_trial=True,
        show_progress_bar=True,
    )

    trials_table = (
        study.trials_dataframe()
    )

    trials_table.to_csv(
        output_folder
        / "optuna_trials.csv",
        index=False,
    )

    best_trial = (
        study.best_trial
    )

    # Reconstruct parameters, including default scheduler fields that are not
    # present in trial.params when scheduler_type == "none".
    best_parameters = dict(
        best_trial.params
    )

    if (
        best_parameters[
            "scheduler_type"
        ]
        == "none"
    ):
        best_parameters[
            "scheduler_factor"
        ] = 0.5

        best_parameters[
            "scheduler_patience"
        ] = 5

    best_parameters[
        "scheduler_min_lr"
    ] = float(
        config[
            "training"
        ].get(
            "scheduler_min_lr",
            1e-6,
        )
    )

    best_validation_payload = {
        "objective_metric": (
            objective_metric
        ),
        "objective_direction": (
            direction
        ),
        "best_trial_number": int(
            best_trial.number
        ),
        "best_objective_value": float(
            best_trial.value
        ),
        "best_objective_std": (
            best_trial.user_attrs.get(
                "objective_std"
            )
        ),
        "best_parameters": (
            best_parameters
        ),
        "fold_validation_metrics": (
            best_trial.user_attrs.get(
                "fold_metrics",
                []
            )
        ),
    }

    with (
        output_folder
        / "best_optuna_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            best_validation_payload,
            handle,
            indent=2,
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BEST OPTUNA CONFIG"
    )

    print(
        "=" * 80
    )

    for key, value in (
        best_parameters.items()
    ):
        print(
            f"{key}: {value}"
        )

    print(
        "\nValidation objective "
        f"({objective_metric}): "
        f"{best_trial.value:.4f}"
    )

    if (
        best_trial.user_attrs.get(
            "objective_std"
        )
        is not None
    ):
        print(
            "Validation fold std: "
            f"{best_trial.user_attrs['objective_std']:.4f}"
        )

    # -------------------------------------------------------------------------
    # NOW load the independent test set. This happens only after tuning is done.
    # -------------------------------------------------------------------------
    test_data = load_excel_file(
        excel_path=(
            resolved[
                "test_file"
            ]
        ),
        id_column=(
            config[
                "id_col"
            ]
        ),
        label_column=(
            config[
                "label_col"
            ]
        ),
        expected_feature_columns=(
            feature_columns
        ),
    )

    development_ids = set()

    for fold in folds:
        development_ids.update(
            fold[
                "ids"
            ].tolist()
        )

    test_ids = set(
        test_data[
            "ids"
        ].tolist()
    )

    overlap = (
        development_ids
        & test_ids
    )

    if overlap:
        raise RuntimeError(
            "Independent test IDs overlap with development folds: "
            f"{sorted(overlap)[:20]}"
        )

    # -------------------------------------------------------------------------
    # Final five-model experiment using the frozen best hyperparameters.
    # -------------------------------------------------------------------------
    final_output = (
        output_folder
        / "best_config_final"
    )

    ensure_dir(
        final_output
    )

    final_results = run_final_models(
        folds=(
            folds
        ),
        feature_columns=(
            feature_columns
        ),
        test_data=(
            test_data
        ),
        best_parameters=(
            best_parameters
        ),
        config=(
            config
        ),
        output_folder=(
            final_output
        ),
        feature_to_group=(
            feature_to_group
        ),
        feature_to_pretty_name=(
            feature_to_pretty_name
        ),
    )

    summary = {
        "best_optuna_config": (
            best_validation_payload
        ),
        "test_mean_std": (
            final_results[
                "mean_std"
            ]
        ),
        "test_ensemble_metrics": (
            final_results[
                "ensemble_metrics"
            ]
        ),
    }

    with (
        output_folder
        / "final_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # FINAL PRINT
    # -------------------------------------------------------------------------
    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL BEST-CONFIG RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        "\nBEST HYPERPARAMETERS"
    )

    for key, value in (
        best_parameters.items()
    ):
        print(
            f"  {key}: {value}"
        )

    print(
        "\nVALIDATION CV USED FOR OPTUNA SELECTION"
    )

    print(
        f"  {objective_metric}: "
        f"{best_trial.value:.4f}"
        + (
            " ± "
            f"{best_trial.user_attrs['objective_std']:.4f}"
            if (
                best_trial.user_attrs.get(
                    "objective_std"
                )
                is not None
            )
            else ""
        )
    )

    print(
        "\nINDEPENDENT TEST — MEAN ± STD ACROSS FIVE FOLD MODELS"
    )

    for metric_name, values in (
        final_results[
            "mean_std"
        ].items()
    ):
        print(
            f"  {metric_name}: "
            f"{values['mean']:.4f} "
            f"± {values['std']:.4f}"
        )

    print(
        "\nINDEPENDENT TEST — FIVE-MODEL ENSEMBLE"
    )

    for metric_name in [
        "accuracy",
        "f1_score",
        "auc_roc",
        "log_loss",
        "pacer_accuracy",
        "no_event_accuracy",
    ]:
        print(
            f"  {metric_name}: "
            f"{final_results['ensemble_metrics'][metric_name]:.4f}"
        )

    feature_table = (
        final_results[
            "feature_table"
        ]
    )

    if feature_table is not None:
        print(
            "\nTOP 15 FEATURE IMPORTANCES"
        )

        print(
            "  "
            + "-" * 72
        )

        for _, row in (
            feature_table
            .head(15)
            .iterrows()
        ):
            print(
                f"  {row['feature_pretty']}: "
                f"{row['importance_mean']:.2f}% "
                f"[{row['group']}]"
            )

    print(
        "\nSaved outputs:"
    )

    print(
        f"  {output_folder / 'best_optuna_config.json'}"
    )

    print(
        f"  {output_folder / 'optuna_trials.csv'}"
    )

    print(
        f"  {output_folder / 'final_summary.json'}"
    )

    print(
        f"  {final_output / 'feature_importance' / 'feature_importance_average.csv'}"
    )


if __name__ == "__main__":
    main()
