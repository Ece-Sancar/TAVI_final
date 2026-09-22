#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plain tabular MLP without residual connections or missingness masks.

The model input consists only of standardized, median-imputed clinical values.

Architecture rule
-----------------
For hidden_dims=[64, 32, 16], the tabular network is exactly:

    Linear(tabular_in, 64)
    LayerNorm(64)
    GELU()
    Dropout(dropout_rate_tabular)

    Linear(64, 32)
    LayerNorm(32)
    GELU()
    Dropout(dropout_rate_tabular)

    Linear(32, 16)
    LayerNorm(16)

followed by a final Linear(16, 2) classifier.

Optuna can vary hidden_dims while the model family always remains a plain MLP.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn


class TabularMLP(pl.LightningModule):
    def __init__(
        self,
        tabular_in: int,
        hidden_dims: Sequence[int],
        dropout_rate_tabular: float,
        learning_rate: float,
        weight_decay: float,
        optimizer_name: str,
        scheduler_type: str,
        scheduler_factor: float,
        scheduler_patience: int,
        scheduler_min_lr: float,
        class_weights,
    ):
        super().__init__()

        hidden_dims = [
            int(dim)
            for dim in hidden_dims
        ]

        if len(hidden_dims) < 2:
            raise ValueError(
                "hidden_dims must contain at least two layers."
            )

        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError(
                "All hidden layer dimensions must be positive."
            )

        self.save_hyperparameters(
            ignore=[
                "class_weights"
            ]
        )

        layers = []
        previous_dim = int(
            tabular_in
        )

        for layer_index, hidden_dim in enumerate(
            hidden_dims
        ):
            layers.append(
                nn.Linear(
                    previous_dim,
                    hidden_dim,
                )
            )

            layers.append(
                nn.LayerNorm(
                    hidden_dim
                )
            )

            # Match the requested architecture exactly:
            # GELU + Dropout after every hidden layer except the last one.
            if layer_index < len(hidden_dims) - 1:
                layers.append(
                    nn.GELU()
                )

                layers.append(
                    nn.Dropout(
                        dropout_rate_tabular
                    )
                )

            previous_dim = hidden_dim

        self.tabular_net = nn.Sequential(
            *layers
        )

        self.classifier = nn.Linear(
            hidden_dims[-1],
            2,
        )

        class_weights_tensor = (
            torch.as_tensor(
                np.asarray(
                    class_weights,
                    dtype=np.float32,
                ),
                dtype=torch.float32,
            )
        )

        if class_weights_tensor.numel() != 2:
            raise ValueError(
                "class_weights must have exactly two values."
            )

        self.register_buffer(
            "class_weights_tensor",
            class_weights_tensor,
        )

        self.loss_function = (
            nn.CrossEntropyLoss(
                weight=(
                    self.class_weights_tensor
                )
            )
        )

        self.learning_rate = float(
            learning_rate
        )

        self.weight_decay = float(
            weight_decay
        )

        self.optimizer_name = str(
            optimizer_name
        ).strip().lower()

        self.scheduler_type = str(
            scheduler_type
        ).strip().lower()

        self.scheduler_factor = float(
            scheduler_factor
        )

        self.scheduler_patience = int(
            scheduler_patience
        )

        self.scheduler_min_lr = float(
            scheduler_min_lr
        )

    def extract_embedding(
        self,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        return self.tabular_net(
            tabular
        )

    def forward(
        self,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.extract_embedding(
            tabular
        )

        return self.classifier(
            embedding
        )

    def _shared_step(
        self,
        batch,
        stage: str,
    ):
        X, y, _ = batch

        logits = self(
            X
        )

        loss = self.loss_function(
            logits,
            y,
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        accuracy = (
            predictions == y
        ).float().mean()

        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=(
                stage != "train"
            ),
            batch_size=(
                len(y)
            ),
        )

        self.log(
            f"{stage}_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=(
                stage != "train"
            ),
            batch_size=(
                len(y)
            ),
        )

        return loss

    def training_step(
        self,
        batch,
        batch_idx,
    ):
        return self._shared_step(
            batch,
            "train",
        )

    def validation_step(
        self,
        batch,
        batch_idx,
    ):
        return self._shared_step(
            batch,
            "val",
        )

    def configure_optimizers(
        self,
    ):
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=(
                    self.learning_rate
                ),
                weight_decay=(
                    self.weight_decay
                ),
            )

        elif (
            self.optimizer_name
            == "adamw"
        ):
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=(
                    self.learning_rate
                ),
                weight_decay=(
                    self.weight_decay
                ),
            )

        else:
            raise ValueError(
                "optimizer_name must be 'adam' or 'adamw'. "
                f"Received {self.optimizer_name!r}."
            )

        if self.scheduler_type == "none":
            return optimizer

        if self.scheduler_type == "plateau":
            scheduler = (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=(
                        self.scheduler_factor
                    ),
                    patience=(
                        self.scheduler_patience
                    ),
                    min_lr=(
                        self.scheduler_min_lr
                    ),
                )
            )

            return {
                "optimizer": (
                    optimizer
                ),
                "lr_scheduler": {
                    "scheduler": (
                        scheduler
                    ),
                    "monitor": (
                        "val_loss"
                    ),
                    "interval": (
                        "epoch"
                    ),
                    "frequency": 1,
                },
            }

        raise ValueError(
            "scheduler_type must be 'none' or 'plateau'. "
            f"Received {self.scheduler_type!r}."
        )
