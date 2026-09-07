#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import (
    AUROC,
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


class Valve2DRadioDINOModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1.2e-4,
        class_weights=None,
        dropout_rate_image=0.30,
        dropout_rate_tabular=0.25,
        num_classes=2,
        weight_decay=1e-3,
        label_smoothing=0.0,
        warmup_epochs=5,
        min_lr_factor=0.01,
        freeze_bn=True,
        unfreeze_epoch=20,
        unfreeze_last_n_blocks=1,
        backbone_lr_multiplier=0.03,
        max_epochs=250,
        tabular_in=45,
        decision_threshold=0.5,
        model_name="hf_hub:Snarcy/RadioDino-s8",
        embed_dim=384,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.class_names = ["no_event", "pacemaker"]
        self.decision_threshold = float(decision_threshold)
        self._partial_unfreeze_applied = False

        # RadioDINO image branch
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
        )
        inferred_dim = getattr(self.backbone, "num_features", None)
        if inferred_dim is not None:
            embed_dim = int(inferred_dim)
        self.image_feature_dim = int(embed_dim)

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self.image_norm = nn.LayerNorm(self.image_feature_dim)

        # EXACT same tabular branch as the final DenseNet implementation.
        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_in, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate_tabular),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_rate_tabular),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate_image),
            nn.Linear(self.image_feature_dim + 16, num_classes),
        )

        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(
            weight=weight_tensor,
            label_smoothing=label_smoothing,
        )

        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_f1 = BinaryF1Score()

    def _unfreeze_requested_backbone_part(self):
        n_blocks = int(self.hparams.unfreeze_last_n_blocks)
        if n_blocks <= 0:
            print("RadioDINO remains fully frozen because unfreeze_last_n_blocks <= 0.")
            return

        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            raise RuntimeError(
                "Partial RadioDINO unfreezing requested, but this backbone has no 'blocks' attribute."
            )
        if n_blocks > len(blocks):
            raise ValueError(
                f"Requested {n_blocks} transformer blocks, but backbone has only {len(blocks)}."
            )

        for block in blocks[-n_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

        norm = getattr(self.backbone, "norm", None)
        if norm is not None:
            for parameter in norm.parameters():
                parameter.requires_grad = True

        trainable = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )
        print(
            f"Unfroze final {n_blocks} RadioDINO transformer block(s); "
            f"trainable backbone parameters={trainable:,}."
        )

    def on_train_epoch_start(self):
        if (
            self.hparams.unfreeze_epoch >= 0
            and self.current_epoch >= self.hparams.unfreeze_epoch
            and not self._partial_unfreeze_applied
        ):
            self._unfreeze_requested_backbone_part()
            self._partial_unfreeze_applied = True

    def extract_image_features(self, image):
        features = self.backbone(image)
        if features.ndim != 2:
            raise RuntimeError(
                f"Expected pooled RadioDINO embeddings [B, D], received {tuple(features.shape)}"
            )
        if features.shape[1] != self.image_feature_dim:
            raise RuntimeError(
                f"RadioDINO produced {features.shape[1]} features, expected {self.image_feature_dim}."
            )
        return self.image_norm(features)

    def extract_tabular_features(self, tabular):
        return self.tabular_net(tabular)

    def forward(self, image, tabular):
        image_features = self.extract_image_features(image)
        tabular_features = self.extract_tabular_features(tabular)
        logits = self.classifier(torch.cat([image_features, tabular_features], dim=1))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite logits were produced.")
        return logits

    def gradcam_target_layer(self):
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None or len(blocks) == 0:
            raise RuntimeError("RadioDINO backbone exposes no transformer blocks.")
        final_block = blocks[-1]
        target = getattr(final_block, "norm1", None)
        if target is None:
            target = getattr(final_block, "norm", None)
        if target is None:
            raise RuntimeError(
                "Could not identify a token-preserving normalization layer in the final RadioDINO block."
            )
        return target

    def _shared_step(self, batch, stage):
        images, targets, tabular, _ = batch
        logits = self(images, tabular)
        loss = self.criterion(logits, targets)
        probabilities = torch.softmax(logits, dim=1)[:, 1]

        if stage == "train":
            self.train_accuracy(probabilities, targets)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
            self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        elif stage == "val":
            self.val_accuracy(probabilities, targets)
            self.val_auroc(probabilities, targets)
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
            self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
            self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        else:
            self.test_accuracy(probabilities, targets)
            self.test_auroc(probabilities, targets)
            self.test_precision(probabilities, targets)
            self.test_recall(probabilities, targets)
            self.test_f1(probabilities, targets)
            self.log("test_loss", loss, on_epoch=True, batch_size=images.size(0))
            self.log("test_acc", self.test_accuracy, on_epoch=True, batch_size=images.size(0))
            self.log("test_auroc", self.test_auroc, on_epoch=True, batch_size=images.size(0))
            self.log("test_precision", self.test_precision, on_epoch=True, batch_size=images.size(0))
            self.log("test_recall", self.test_recall, on_epoch=True, batch_size=images.size(0))
            self.log("test_f1", self.test_f1, on_epoch=True, batch_size=images.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        backbone_parameters = list(self.backbone.parameters())
        head_parameters = (
            list(self.image_norm.parameters())
            + list(self.tabular_net.parameters())
            + list(self.classifier.parameters())
        )

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_parameters,
                    "lr": self.hparams.learning_rate * self.hparams.backbone_lr_multiplier,
                    "name": "backbone",
                },
                {
                    "params": head_parameters,
                    "lr": self.hparams.learning_rate,
                    "name": "heads",
                },
            ],
            weight_decay=self.hparams.weight_decay,
        )

        def learning_rate_factor(epoch):
            warmup_epochs = max(1, int(self.hparams.warmup_epochs))
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)

            total_decay_epochs = max(1, int(self.hparams.max_epochs) - warmup_epochs)
            progress = min(1.0, (epoch - warmup_epochs) / total_decay_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(
                self.hparams.min_lr_factor
                + (1.0 - self.hparams.min_lr_factor) * cosine
            )

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=[learning_rate_factor, learning_rate_factor],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint.get("state_dict", {})
        for key in list(state_dict):
            if key.startswith("criterion."):
                del state_dict[key]

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, **kwargs):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict = checkpoint.get("state_dict", {})
        for key in list(state_dict):
            if key.startswith("criterion."):
                del state_dict[key]
        hyperparameters = dict(checkpoint.get("hyper_parameters", {}))
        hyperparameters.update(kwargs)
        model = cls(**hyperparameters)
        model.load_state_dict(state_dict, strict=False)
        model.decision_threshold = float(
            checkpoint.get(
                "decision_threshold",
                hyperparameters.get("decision_threshold", 0.5),
            )
        )
        return model
