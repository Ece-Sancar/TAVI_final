#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Residual tabular MLP without missingness masks.

The model input consists only of standardized, median-imputed clinical values.
"""

from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn


class ResidualBlock(
    nn.Module
):
    def __init__(
        self,
        dim: int,
        dropout_rate: float,
    ):
        super().__init__()

        self.residual_path = (
            nn.Sequential(
                nn.Linear(
                    dim,
                    dim,
                ),
                nn.LayerNorm(
                    dim
                ),
                nn.GELU(),
                nn.Dropout(
                    dropout_rate
                ),
                nn.Linear(
                    dim,
                    dim,
                ),
                nn.Dropout(
                    dropout_rate
                ),
            )
        )

        self.output_norm = (
            nn.LayerNorm(
                dim
            )
        )

        self.output_activation = (
            nn.GELU()
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        update = self.residual_path(
            x
        )

        x = x + update

        x = self.output_norm(
            x
        )

        x = self.output_activation(
            x
        )

        return x


class ResidualTabularMLP(
    pl.LightningModule
):
    def __init__(
        self,
        tabular_in: int,
        hidden_dim: int,
        bottleneck_dim: int,
        embedding_dim: int,
        n_residual_blocks: int,
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

        self.save_hyperparameters(
            ignore=[
                "class_weights"
            ]
        )

        self.input_projection = (
            nn.Sequential(
                nn.Linear(
                    tabular_in,
                    hidden_dim,
                ),
                nn.LayerNorm(
                    hidden_dim
                ),
                nn.GELU(),
                nn.Dropout(
                    dropout_rate_tabular
                ),
            )
        )

        self.residual_blocks = (
            nn.Sequential(
                *[
                    ResidualBlock(
                        dim=hidden_dim,
                        dropout_rate=(
                            dropout_rate_tabular
                        ),
                    )
                    for _ in range(
                        n_residual_blocks
                    )
                ]
            )
        )

        self.embedding_head = (
            nn.Sequential(
                nn.Linear(
                    hidden_dim,
                    bottleneck_dim,
                ),
                nn.LayerNorm(
                    bottleneck_dim
                ),
                nn.GELU(),
                nn.Dropout(
                    dropout_rate_tabular
                ),
                nn.Linear(
                    bottleneck_dim,
                    embedding_dim,
                ),
                nn.LayerNorm(
                    embedding_dim
                ),
            )
        )

        self.classifier = nn.Linear(
            embedding_dim,
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
        x = self.input_projection(
            tabular
        )

        x = self.residual_blocks(
            x
        )

        return self.embedding_head(
            x
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
