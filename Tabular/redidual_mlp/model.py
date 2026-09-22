#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Residual tabular MLP for binary pacemaker-outcome prediction.

Input preprocessing is handled in dataset.py and produces:

    [standardized imputed clinical values | binary missingness masks]

If there are F original clinical variables, the model receives 2F inputs.

Architecture
------------
Input (2F)
    -> Linear(2F, hidden_dim)
    -> LayerNorm
    -> GELU
    -> Dropout

    -> ResidualBlock(hidden_dim) x n_residual_blocks

    -> Linear(hidden_dim, bottleneck_dim)
    -> LayerNorm
    -> GELU
    -> Dropout

    -> Linear(bottleneck_dim, embedding_dim)
    -> LayerNorm

    -> tabular embedding

Standalone classifier:
    embedding -> Linear(embedding_dim, 2)

The residual connection allows the network to learn refinements around an
identity mapping rather than forcing every hidden layer to completely rewrite
the representation.
"""

from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Fully connected residual block with constant hidden width.

    x
      -> Linear
      -> LayerNorm
      -> GELU
      -> Dropout
      -> Linear
      -> Dropout
      + x
      -> LayerNorm
      -> GELU
    """

    def __init__(
        self,
        dim: int,
        dropout_rate: float,
    ):
        super().__init__()

        self.residual_path = nn.Sequential(
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

        self.output_norm = nn.LayerNorm(
            dim
        )

        self.output_activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        residual_update = self.residual_path(
            x
        )

        x = x + residual_update

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
    """
    Residual MLP whose final tabular representation can later be reused
    directly as a multimodal tabular embedding.
    """

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
        class_weights,
    ):
        super().__init__()

        self.save_hyperparameters(
            ignore=[
                "class_weights"
            ]
        )

        if tabular_in <= 0:
            raise ValueError(
                "tabular_in must be > 0."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be > 0."
            )

        if bottleneck_dim <= 0:
            raise ValueError(
                "bottleneck_dim must be > 0."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be > 0."
            )

        if n_residual_blocks < 1:
            raise ValueError(
                "n_residual_blocks must be >= 1."
            )

        if not (
            0.0
            <= dropout_rate_tabular
            < 1.0
        ):
            raise ValueError(
                "dropout_rate_tabular must be in [0, 1)."
            )

        # ------------------------------------------------------------------
        # Input projection
        # ------------------------------------------------------------------
        self.input_projection = nn.Sequential(
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

        # ------------------------------------------------------------------
        # Residual representation learning
        # ------------------------------------------------------------------
        self.residual_blocks = nn.Sequential(
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

        # ------------------------------------------------------------------
        # Final clinical embedding
        # ------------------------------------------------------------------
        self.embedding_head = nn.Sequential(
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

        # The encoder is deliberately separated from the standalone
        # classification head so that exactly this encoder can later be
        # reused inside DenseNet/DINO multimodal models.
        self.classifier = nn.Linear(
            embedding_dim,
            2,
        )

        weights = torch.as_tensor(
            np.asarray(
                class_weights,
                dtype=np.float32,
            ),
            dtype=torch.float32,
        )

        if weights.numel() != 2:
            raise ValueError(
                "class_weights must contain exactly two values."
            )

        self.register_buffer(
            "class_weights_tensor",
            weights,
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

    def extract_embedding(
        self,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the final clinical representation before classification.

        Shape:
            [batch_size, embedding_dim]
        """
        x = self.input_projection(
            tabular
        )

        x = self.residual_blocks(
            x
        )

        embedding = self.embedding_head(
            x
        )

        return embedding

    def forward(
        self,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.extract_embedding(
            tabular
        )

        logits = self.classifier(
            embedding
        )

        return logits

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
            batch_size=len(y),
        )

        self.log(
            f"{stage}_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=(
                stage != "train"
            ),
            batch_size=len(y),
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
                lr=self.learning_rate,
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
                lr=self.learning_rate,
                weight_decay=(
                    self.weight_decay
                ),
            )

        else:
            raise ValueError(
                "optimizer must be 'adam' or 'adamw', "
                f"got {self.optimizer_name!r}"
            )

        return optimizer
