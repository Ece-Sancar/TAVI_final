#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import AUROC, BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall

class Valve2DRadioDINOImageModel(pl.LightningModule):
    def __init__(self, learning_rate=1.2e-4, class_weights=None, dropout_rate_image=0.30,
                 num_classes=2, weight_decay=1e-3, label_smoothing=0.0, warmup_epochs=5,
                 min_lr_factor=0.01, freeze_bn=True, unfreeze_epoch=20,
                 unfreeze_last_n_blocks=1, backbone_lr_multiplier=0.03, max_epochs=250,
                 decision_threshold=0.5, model_name='hf_hub:Snarcy/RadioDino-s8', embed_dim=384):
        super().__init__()
        self.save_hyperparameters()
        self.class_names=['no_event','pacemaker']
        self.decision_threshold=float(decision_threshold)
        self._partial_unfreeze_applied=False

        self.backbone=timm.create_model(model_name, pretrained=True, num_classes=0)
        inferred=getattr(self.backbone,'num_features',None)
        if inferred is not None:
            embed_dim=int(inferred)
        self.image_feature_dim=int(embed_dim)
        for p in self.backbone.parameters():
            p.requires_grad=False
        self.image_norm=nn.LayerNorm(self.image_feature_dim)
        self.classifier=nn.Sequential(nn.Dropout(dropout_rate_image), nn.Linear(self.image_feature_dim,num_classes))

        wt=None if class_weights is None else torch.tensor(class_weights,dtype=torch.float32)
        self.criterion=nn.CrossEntropyLoss(weight=wt,label_smoothing=label_smoothing)
        self.train_accuracy=BinaryAccuracy(); self.val_accuracy=BinaryAccuracy(); self.test_accuracy=BinaryAccuracy()
        self.val_auroc=AUROC(task='binary'); self.test_auroc=AUROC(task='binary')
        self.test_precision=BinaryPrecision(); self.test_recall=BinaryRecall(); self.test_f1=BinaryF1Score()

    def _unfreeze_requested_backbone_part(self):
        n=int(self.hparams.unfreeze_last_n_blocks)
        if n<=0:
            print('RadioDINO remains fully frozen because unfreeze_last_n_blocks <= 0.')
            return
        blocks=getattr(self.backbone,'blocks',None)
        if blocks is None:
            raise RuntimeError("Partial RadioDINO unfreezing requested, but backbone has no 'blocks' attribute.")
        if n>len(blocks):
            raise ValueError(f'Requested {n} blocks, backbone has {len(blocks)}.')
        for block in blocks[-n:]:
            for p in block.parameters(): p.requires_grad=True
        norm=getattr(self.backbone,'norm',None)
        if norm is not None:
            for p in norm.parameters(): p.requires_grad=True
        trainable=sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f'Unfroze final {n} RadioDINO transformer block(s); trainable backbone parameters={trainable:,}.')

    def on_train_epoch_start(self):
        if self.hparams.unfreeze_epoch>=0 and self.current_epoch>=self.hparams.unfreeze_epoch and not self._partial_unfreeze_applied:
            self._unfreeze_requested_backbone_part(); self._partial_unfreeze_applied=True

    def extract_image_features(self,image):
        features=self.backbone(image)
        if features.ndim!=2:
            raise RuntimeError(f'Expected pooled embeddings [B,D], got {tuple(features.shape)}')
        return self.image_norm(features)

    def forward(self,image):
        logits=self.classifier(self.extract_image_features(image))
        if not torch.isfinite(logits).all(): raise FloatingPointError('Non-finite logits were produced.')
        return logits

    def gradcam_target_layer(self):
        blocks=getattr(self.backbone,'blocks',None)
        if blocks is None or len(blocks)==0: raise RuntimeError('RadioDINO backbone exposes no transformer blocks.')
        final=blocks[-1]
        target=getattr(final,'norm1',None)
        if target is None: target=getattr(final,'norm',None)
        if target is None: raise RuntimeError('Could not identify token-preserving norm layer in final RadioDINO block.')
        return target

    def _shared_step(self,batch,stage):
        images,targets,_=batch
        logits=self(images); loss=self.criterion(logits,targets); probs=torch.softmax(logits,dim=1)[:,1]
        if stage=='train':
            self.train_accuracy(probs,targets)
            self.log('train_loss',loss,on_step=False,on_epoch=True,prog_bar=True,batch_size=images.size(0))
            self.log('train_acc',self.train_accuracy,on_step=False,on_epoch=True,prog_bar=True,batch_size=images.size(0))
        elif stage=='val':
            self.val_accuracy(probs,targets); self.val_auroc(probs,targets)
            self.log('val_loss',loss,on_step=False,on_epoch=True,prog_bar=True,batch_size=images.size(0))
            self.log('val_acc',self.val_accuracy,on_step=False,on_epoch=True,prog_bar=True,batch_size=images.size(0))
            self.log('val_auroc',self.val_auroc,on_step=False,on_epoch=True,prog_bar=True,batch_size=images.size(0))
        else:
            self.test_accuracy(probs,targets); self.test_auroc(probs,targets); self.test_precision(probs,targets); self.test_recall(probs,targets); self.test_f1(probs,targets)
            self.log('test_loss',loss,on_epoch=True,batch_size=images.size(0)); self.log('test_acc',self.test_accuracy,on_epoch=True,batch_size=images.size(0))
            self.log('test_auroc',self.test_auroc,on_epoch=True,batch_size=images.size(0)); self.log('test_precision',self.test_precision,on_epoch=True,batch_size=images.size(0))
            self.log('test_recall',self.test_recall,on_epoch=True,batch_size=images.size(0)); self.log('test_f1',self.test_f1,on_epoch=True,batch_size=images.size(0))
        return loss

    def training_step(self,batch,batch_idx): return self._shared_step(batch,'train')
    def validation_step(self,batch,batch_idx): return self._shared_step(batch,'val')
    def test_step(self,batch,batch_idx): return self._shared_step(batch,'test')

    def configure_optimizers(self):
        optimizer=torch.optim.AdamW([
            {'params':list(self.backbone.parameters()),'lr':self.hparams.learning_rate*self.hparams.backbone_lr_multiplier,'name':'backbone'},
            {'params':list(self.image_norm.parameters())+list(self.classifier.parameters()),'lr':self.hparams.learning_rate,'name':'heads'}],
            weight_decay=self.hparams.weight_decay)
        def factor(epoch):
            warm=max(1,int(self.hparams.warmup_epochs))
            if epoch<warm: return float(epoch+1)/float(warm)
            total=max(1,int(self.hparams.max_epochs)-warm); progress=min(1.0,(epoch-warm)/total)
            cosine=0.5*(1.0+math.cos(math.pi*progress))
            return float(self.hparams.min_lr_factor+(1.0-self.hparams.min_lr_factor)*cosine)
        scheduler=LambdaLR(optimizer,lr_lambda=[factor,factor])
        return {'optimizer':optimizer,'lr_scheduler':{'scheduler':scheduler,'interval':'epoch'}}

    def on_load_checkpoint(self,checkpoint):
        sd=checkpoint.get('state_dict',{})
        for k in list(sd):
            if k.startswith('criterion.'): del sd[k]

    @classmethod
    def load_from_checkpoint(cls,checkpoint_path,map_location=None,**kwargs):
        checkpoint=torch.load(checkpoint_path,map_location=map_location)
        sd=checkpoint.get('state_dict',{})
        for k in list(sd):
            if k.startswith('criterion.'): del sd[k]
        hp=dict(checkpoint.get('hyper_parameters',{})); hp.update(kwargs)
        model=cls(**hp); model.load_state_dict(sd,strict=False)
        model.decision_threshold=float(checkpoint.get('decision_threshold',hp.get('decision_threshold',0.5)))
        return model
