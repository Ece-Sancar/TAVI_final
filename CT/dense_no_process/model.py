#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import AUROC,BinaryAccuracy,BinaryF1Score,BinaryPrecision,BinaryRecall
from torchvision.models import DenseNet121_Weights,densenet121

class Valve2DDenseNetImageModel(pl.LightningModule):
    def __init__(self, learning_rate=1.2e-4, class_weights=None, dropout_rate_image=0.30, num_classes=2, weight_decay=1e-3, label_smoothing=0.0, warmup_epochs=5, min_lr_factor=0.01, freeze_bn=True, unfreeze_epoch=20, unfreeze_last_n_blocks=1, backbone_lr_multiplier=0.03, max_epochs=250, decision_threshold=0.5):
        super().__init__(); self.save_hyperparameters(); self.class_names=["no_event","pacemaker"]; self.decision_threshold=float(decision_threshold); self._partial_unfreeze_applied=False
        self.densenet=densenet121(weights=DenseNet121_Weights.DEFAULT, drop_rate=0.0)
        image_feature_dim=int(self.densenet.classifier.in_features); self.densenet.classifier=nn.Identity()
        for p in self.densenet.features.parameters(): p.requires_grad=False
        if self.hparams.freeze_bn: self._freeze_backbone_batch_norm()
        self.image_norm=nn.LayerNorm(image_feature_dim)
        self.classifier=nn.Sequential(nn.Dropout(dropout_rate_image),nn.Linear(image_feature_dim,num_classes))
        weight_tensor=None if class_weights is None else torch.tensor(class_weights,dtype=torch.float32)
        self.criterion=nn.CrossEntropyLoss(weight=weight_tensor,label_smoothing=label_smoothing)
        self.train_accuracy=BinaryAccuracy(); self.val_accuracy=BinaryAccuracy(); self.test_accuracy=BinaryAccuracy(); self.val_auroc=AUROC(task="binary"); self.test_auroc=AUROC(task="binary"); self.test_precision=BinaryPrecision(); self.test_recall=BinaryRecall(); self.test_f1=BinaryF1Score()

    def _freeze_backbone_batch_norm(self):
        for module in self.densenet.features.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                if module.weight is not None: module.weight.requires_grad=False
                if module.bias is not None: module.bias.requires_grad=False
    def train(self, mode=True):
        super().train(mode)
        if mode and self.hparams.freeze_bn: self._freeze_backbone_batch_norm()
        return self
    def _unfreeze_requested_backbone_part(self):
        n_blocks=int(self.hparams.unfreeze_last_n_blocks)
        if n_blocks<=0:
            print("DenseNet remains fully frozen because unfreeze_last_n_blocks <= 0."); return
        if n_blocks>4: raise ValueError("DenseNet121 has four dense blocks; requested more than four.")
        first_block=5-n_blocks; names=[]
        for block_index in range(first_block,5):
            block_name=f"denseblock{block_index}"; names.append(block_name)
            for p in getattr(self.densenet.features,block_name).parameters(): p.requires_grad=True
            if block_index<4:
                transition_name=f"transition{block_index}"; names.append(transition_name)
                for p in getattr(self.densenet.features,transition_name).parameters(): p.requires_grad=True
        names.append("norm5")
        for p in self.densenet.features.norm5.parameters(): p.requires_grad=True
        if self.hparams.freeze_bn: self._freeze_backbone_batch_norm()
        trainable=sum(p.numel() for p in self.densenet.features.parameters() if p.requires_grad)
        print(f"Unfroze DenseNet modules {names}; trainable backbone parameters={trainable:,}.")
    def on_train_epoch_start(self):
        if self.hparams.unfreeze_epoch>=0 and self.current_epoch>=self.hparams.unfreeze_epoch and not self._partial_unfreeze_applied:
            self._unfreeze_requested_backbone_part(); self._partial_unfreeze_applied=True
    def extract_image_features(self,image):
        features=self.densenet.features(image); features=F.relu(features,inplace=False); features=F.adaptive_avg_pool2d(features,output_size=(1,1)).flatten(1); return self.image_norm(features)
    def forward(self,image):
        logits=self.classifier(self.extract_image_features(image))
        if not torch.isfinite(logits).all(): raise FloatingPointError("Non-finite logits were produced.")
        return logits
    def gradcam_target_layer(self): return self.densenet.features.norm5
    def _shared_step(self,batch,stage):
        images,targets,_=batch; logits=self(images); loss=self.criterion(logits,targets); probabilities=torch.softmax(logits,dim=1)[:,1]
        if stage=="train":
            self.train_accuracy(probabilities,targets); self.log("train_loss",loss,on_step=False,on_epoch=True,prog_bar=True,batch_size=images.size(0)); self.log("train_acc",self.train_accuracy,on_step=False,on_epoch=True,prog_bar=True,batch_size=images.size(0))
        elif stage=="val":
            self.val_accuracy(probabilities,targets); self.val_auroc(probabilities,targets); self.log("val_loss",loss,on_step=False,on_epoch=True,prog_bar=True,batch_size=images.size(0)); self.log("val_acc",self.val_accuracy,on_step=False,on_epoch=True,prog_bar=True,batch_size=images.size(0)); self.log("val_auroc",self.val_auroc,on_step=False,on_epoch=True,prog_bar=True,batch_size=images.size(0))
        else:
            self.test_accuracy(probabilities,targets); self.test_auroc(probabilities,targets); self.test_precision(probabilities,targets); self.test_recall(probabilities,targets); self.test_f1(probabilities,targets)
            self.log("test_loss",loss,on_epoch=True,batch_size=images.size(0)); self.log("test_acc",self.test_accuracy,on_epoch=True,batch_size=images.size(0)); self.log("test_auroc",self.test_auroc,on_epoch=True,batch_size=images.size(0)); self.log("test_precision",self.test_precision,on_epoch=True,batch_size=images.size(0)); self.log("test_recall",self.test_recall,on_epoch=True,batch_size=images.size(0)); self.log("test_f1",self.test_f1,on_epoch=True,batch_size=images.size(0))
        return loss
    def training_step(self,batch,batch_idx): return self._shared_step(batch,"train")
    def validation_step(self,batch,batch_idx): return self._shared_step(batch,"val")
    def test_step(self,batch,batch_idx): return self._shared_step(batch,"test")
    def configure_optimizers(self):
        backbone_parameters=list(self.densenet.features.parameters()); head_parameters=list(self.image_norm.parameters())+list(self.classifier.parameters())
        optimizer=torch.optim.AdamW([{"params":backbone_parameters,"lr":self.hparams.learning_rate*self.hparams.backbone_lr_multiplier,"name":"backbone"},{"params":head_parameters,"lr":self.hparams.learning_rate,"name":"heads"}],weight_decay=self.hparams.weight_decay)
        def learning_rate_factor(epoch):
            warmup_epochs=max(1,int(self.hparams.warmup_epochs))
            if epoch<warmup_epochs: return float(epoch+1)/float(warmup_epochs)
            total_decay_epochs=max(1,int(self.hparams.max_epochs)-warmup_epochs); progress=min(1.0,(epoch-warmup_epochs)/total_decay_epochs); cosine=0.5*(1.0+math.cos(math.pi*progress))
            return float(self.hparams.min_lr_factor+(1.0-self.hparams.min_lr_factor)*cosine)
        scheduler=LambdaLR(optimizer,lr_lambda=[learning_rate_factor,learning_rate_factor])
        return {"optimizer":optimizer,"lr_scheduler":{"scheduler":scheduler,"interval":"epoch"}}
