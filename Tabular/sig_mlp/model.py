#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

class TabularMLP(pl.LightningModule):
    def __init__(
        self,
        tabular_in: int,
        dropout_rate_tabular: float,
        learning_rate: float,
        weight_decay: float,
        optimizer_name: str,
        class_weights,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_in, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_rate_tabular),

            nn.Linear(32, 16),
            nn.LayerNorm(16),
        )

        self.classifier = nn.Linear(16, 2)

        weights = torch.as_tensor(
            np.asarray(class_weights, dtype=np.float32),
            dtype=torch.float32,
        )
        self.register_buffer("class_weights_tensor", weights)
        self.loss_function = nn.CrossEntropyLoss(weight=self.class_weights_tensor)

        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.optimizer_name = str(optimizer_name).strip().lower()

    def forward(self, tabular):
        embedding = self.tabular_net(tabular)
        return self.classifier(embedding)

    def extract_embedding(self, tabular):
        return self.tabular_net(tabular)

    def _shared_step(self, batch, stage: str):
        X, y, _ = batch
        logits = self(X)
        loss = self.loss_function(logits, y)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y).float().mean()

        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=(stage != "train"),
            batch_size=len(y),
        )
        self.log(
            f"{stage}_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=(stage != "train"),
            batch_size=len(y),
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            return torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        raise ValueError(
            f"optimizer must be 'adam' or 'adamw', got {self.optimizer_name!r}"
        )
