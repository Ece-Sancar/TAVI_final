#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""CT-only DenseNet pipeline matching the final combined experiment protocol."""

import argparse, gc, json, random
from pathlib import Path
from typing import Any, Dict, Tuple

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
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, log_loss, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from tqdm import tqdm

from dataset import Valve2DImageDataModule
from model import Valve2DRadioDINOImageModel

CLASS_NAMES=["no_event","pacemaker"]
VALID_DATASETS={"tum","lmu","merged"}
VALID_TRAIN_VALUES=VALID_DATASETS|{"pre"}
VALID_PERCENTAGES={2,5,10,20,50,100}
plt.rcParams.update({"font.size":18,"axes.titlesize":22,"axes.labelsize":20,"xtick.labelsize":16,"ytick.labelsize":16,"legend.fontsize":18})

def set_global_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def clean_memory():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def ensure_dir(path): Path(path).mkdir(parents=True,exist_ok=True)

def save_json(data,path):
    path=Path(path); ensure_dir(path.parent)
    with path.open("w",encoding="utf-8") as f: json.dump(data,f,indent=2)

def save_lines(path,values):
    path=Path(path); ensure_dir(path.parent); path.write_text("\n".join(str(v) for v in values)+"\n",encoding="utf-8")

def mean_std(values):
    a=np.asarray(values,dtype=float); return float(np.nanmean(a)),float(np.nanstd(a,ddof=0))

def safe_auc(y,p):
    return float("nan") if len(np.unique(y))<2 else float(roc_auc_score(y,p))

def safe_log_loss(y,p):
    p=np.clip(np.asarray(p,dtype=np.float64),1e-7,1.0-1e-7); return float(log_loss(y,p,labels=[0,1]))

def get_fold_path(folder,fold_number):
    candidates=[Path(folder)/f"fold{fold_number}.xlsx",Path(folder)/f"fold_{fold_number}.xlsx"]
    existing=[p for p in candidates if p.exists()]
    if len(existing)==1:return existing[0]
    if len(existing)>1:raise RuntimeError(f"Multiple fold files found: {existing}")
    raise FileNotFoundError(f"Could not locate fold {fold_number} in {folder}")

def resolve_configuration(config:Dict[str,Any]):
    dataset_root=Path(config["dataset_root"]); train_choice=str(config["train_dataset"]).strip().lower(); test_choice=str(config["test_dataset"]).strip().lower(); percentage=int(config.get("data_percentage",100))
    if train_choice not in VALID_TRAIN_VALUES: raise ValueError(f"train_dataset must be one of {sorted(VALID_TRAIN_VALUES)}")
    if test_choice not in VALID_DATASETS: raise ValueError(f"test_dataset must be one of {sorted(VALID_DATASETS)}")
    if percentage not in VALID_PERCENTAGES: raise ValueError(f"data_percentage must be one of {sorted(VALID_PERCENTAGES)}")
    if train_choice=="pre":
        pc=config.get("pretrained",{}); training_dataset=str(pc.get("train_dataset","")).strip().lower()
        if training_dataset not in VALID_DATASETS: raise ValueError("pretrained.train_dataset must be tum, lmu or merged")
        checkpoint_root=Path(pc["checkpoint_dir"]); mode="pretrained"
    else:
        training_dataset=train_choice; checkpoint_root=None; mode="train"
    training_folder=dataset_root/training_dataset if percentage==100 else dataset_root/"data_size"/f"{percentage}_percent"/training_dataset
    test_file=dataset_root/test_choice/"test.xlsx"
    if not training_folder.is_dir(): raise NotADirectoryError(f"Training folder does not exist:\n{training_folder}")
    if not test_file.is_file(): raise FileNotFoundError(f"Independent test file does not exist:\n{test_file}")
    fold_files=[get_fold_path(training_folder,i) for i in range(1,6)]
    output_folder=Path(config["output_dir"])/f"{percentage}_percent"
    checkpoint_output_root=(Path(config["checkpoint_dir"])/f"{percentage}_percent") if mode=="train" else checkpoint_root
    return {"mode":mode,"percentage":percentage,"training_dataset":training_dataset,"test_dataset":test_choice,"training_folder":training_folder,"test_file":test_file,"fold_files":fold_files,"output_folder":output_folder,"checkpoint_root":checkpoint_output_root}

def find_existing_checkpoint(checkpoint_root,fold_number):
    fold_folder=Path(checkpoint_root)/f"fold{fold_number}"; preferred=fold_folder/"best-checkpoint.ckpt"
    if preferred.is_file(): return preferred
    candidates=sorted(fold_folder.glob("best-checkpoint*.ckpt"),key=lambda p:p.stat().st_mtime,reverse=True)
    if not candidates: raise FileNotFoundError(f"No checkpoint found for fold {fold_number} in {fold_folder}")
    return candidates[0]

@torch.inference_mode()
def collect_predictions(model,data_loader,device):
    model=model.to(device); model.eval(); probs=[]; targets=[]; ids=[]
    for images,batch_targets,batch_ids in data_loader:
        images=images.to(device,non_blocking=True); logits=model(images); p=torch.softmax(logits,dim=1)[:,1]
        probs.extend(p.cpu().numpy().tolist()); targets.extend(batch_targets.cpu().numpy().tolist()); ids.extend([str(x) for x in batch_ids])
    return np.asarray(probs,dtype=np.float64),np.asarray(targets,dtype=np.int64),ids

def calculate_metrics(targets,probabilities,threshold=0.5):
    pred=(probabilities>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(targets,pred,labels=[0,1]).ravel()
    return {"threshold":float(threshold),"accuracy":float(accuracy_score(targets,pred)),"balanced_accuracy":float(balanced_accuracy_score(targets,pred)),"auc_roc":safe_auc(targets,probabilities),"auc_pr":float(average_precision_score(targets,probabilities)),"f1":float(f1_score(targets,pred,zero_division=0)),"precision":float(precision_score(targets,pred,zero_division=0)),"recall":float(recall_score(targets,pred,zero_division=0)),"log_loss":safe_log_loss(targets,probabilities),"true_negatives":int(tn),"false_positives":int(fp),"false_negatives":int(fn),"true_positives":int(tp)}

def save_confusion_matrix(targets,predictions,path,title):
    matrix=confusion_matrix(targets,predictions,labels=[0,1]); plt.figure(figsize=(7,6)); sns.heatmap(matrix,annot=True,fmt="d",cmap="Greys",cbar=False,xticklabels=["No Event","Pacemaker"],yticklabels=["No Event","Pacemaker"]); plt.title(title); plt.ylabel("True label"); plt.xlabel("Predicted label"); plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()

def save_evaluation(targets,probabilities,sample_ids,threshold,output_dir,phase,fold=None):
    ensure_dir(output_dir); pred=(probabilities>=threshold).astype(int); metrics=calculate_metrics(targets,probabilities,threshold)
    frame=pd.DataFrame({"ID":sample_ids,"true_label":targets,"true_class":[CLASS_NAMES[int(v)] for v in targets],"prob_pacemaker":probabilities,"predicted_label":pred,"predicted_class":[CLASS_NAMES[int(v)] for v in pred],"threshold":float(threshold)})
    if fold is not None: frame.insert(0,"fold",fold)
    frame.to_csv(Path(output_dir)/f"{phase}_predictions.csv",index=False)
    pd.DataFrame(classification_report(targets,pred,labels=[0,1],target_names=["No Event","Pacemaker"],output_dict=True,zero_division=0)).transpose().to_csv(Path(output_dir)/f"{phase}_classification_report.csv")
    save_json(metrics,Path(output_dir)/f"{phase}_metrics.json"); save_confusion_matrix(targets,pred,Path(output_dir)/f"{phase}_confusion_matrix.png",phase.replace("_"," ").title()+" confusion matrix")
    fpr,tpr,_=roc_curve(targets,probabilities); plt.figure(figsize=(7,6)); plt.plot(fpr,tpr,label=f"AUC = {metrics['auc_roc']:.3f}"); plt.plot([0,1],[0,1],"k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(Path(output_dir)/f"{phase}_roc_curve.png",dpi=300); plt.close()
    precision,recall,_=precision_recall_curve(targets,probabilities); plt.figure(figsize=(7,6)); plt.plot(recall,precision,label=f"AP = {metrics['auc_pr']:.3f}"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(loc="lower left"); plt.tight_layout(); plt.savefig(Path(output_dir)/f"{phase}_pr_curve.png",dpi=300); plt.close(); return metrics,frame

def normalize_cam(cam):
    cam=np.asarray(cam,dtype=np.float32); cam=cam-np.min(cam); m=np.max(cam); return (cam/m if m>0 else cam).astype(np.float32)

def calculate_gradcam(model, image, device, target_class=1):
    """Gradient-weighted RadioDINO patch-token map, saved through the same overlay pipeline."""
    import math
    model=model.to(device); model.eval()
    image=image.unsqueeze(0).to(device); image.requires_grad_(True)
    activations={}; gradients={}; target_layer=model.gradcam_target_layer()
    def fh(module,inputs,output): activations['value']=output
    def bh(module,grad_input,grad_output): gradients['value']=grad_output[0]
    h1=target_layer.register_forward_hook(fh); h2=target_layer.register_full_backward_hook(bh)
    try:
        model.zero_grad(set_to_none=True)
        logits=model(image); logits[0,target_class].backward()
        a=activations['value']; g=gradients['value']
        if a.ndim!=3 or g.ndim!=3:
            raise RuntimeError(f'Expected token tensors [B,N,C], got {tuple(a.shape)} and {tuple(g.shape)}')
        a=a[0]; g=g[0]; n=int(a.shape[0])
        prefix=int(getattr(model.backbone,'num_prefix_tokens',1)); spatial=n-prefix; grid=int(round(math.sqrt(spatial)))
        if grid*grid!=spatial:
            prefix=1; spatial=n-1; grid=int(round(math.sqrt(spatial)))
        if grid*grid!=spatial:
            raise RuntimeError(f'Could not reshape RadioDINO tokens: token_count={n}, prefix={prefix}, spatial={spatial}')
        pa=a[prefix:]; pg=g[prefix:]; weights=pg.mean(dim=0)
        cam_tokens=torch.relu(torch.sum(pa*weights.unsqueeze(0),dim=1))
        cam=cam_tokens.reshape(grid,grid).detach().cpu().numpy()
        return normalize_cam(cam)
    finally:
        h1.remove(); h2.remove()

def denormalize_ct(image_tensor,image_mean,image_std):
    image=image_tensor[0].detach().cpu().numpy(); image=image*float(image_std[0])+float(image_mean[0]); return np.clip(image,0.0,1.0)

def create_gradcam_overlay(grayscale_image,cam):
    h,w=grayscale_image.shape; resized=cv2.resize(cam,(w,h),interpolation=cv2.INTER_LINEAR); heatmap=(np.clip(resized,0,1)*255).astype(np.uint8); heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET); base=(grayscale_image*255).astype(np.uint8); base=cv2.cvtColor(base,cv2.COLOR_GRAY2BGR); return cv2.addWeighted(base,0.55,heatmap,0.45,0)

def prediction_category(true_label,predicted_label):
    if true_label==1 and predicted_label==1:return "label_pacer_predict_pacer"
    if true_label==0 and predicted_label==1:return "label_no_predict_pacer"
    if true_label==1 and predicted_label==0:return "label_pacer_predict_no"
    return "label_no_predict_no"

def generate_fold_gradcams(model,dataset,probabilities,threshold,output_root,fold_number,device,image_mean,image_std,cam_sums):
    predictions=(probabilities>=threshold).astype(int); fold_root=Path(output_root)/f"fold{fold_number}"; ensure_dir(fold_root); print(f"Generating Grad-CAM: fold {fold_number}")
    for index in tqdm(range(len(dataset)),desc=f"Grad-CAM fold {fold_number}"):
        image,label,sample_id=dataset[index]; label_int=int(label.item()); prediction_int=int(predictions[index]); cam=calculate_gradcam(model,image,device,target_class=1)
        if sample_id not in cam_sums: cam_sums[sample_id]=np.zeros_like(cam,dtype=np.float32)
        cam_sums[sample_id]+=cam; overlay=create_gradcam_overlay(denormalize_ct(image,image_mean,image_std),cam); folder=fold_root/prediction_category(label_int,prediction_int); ensure_dir(folder); cv2.imwrite(str(folder/f"{sample_id}.png"),overlay)

def save_ensemble_gradcams(dataset,cam_sums,ensemble_predictions,output_root,number_of_models,image_mean,image_std):
    root=Path(output_root)/"ensemble"; ensure_dir(root)
    for index in tqdm(range(len(dataset)),desc="Ensemble Grad-CAM"):
        image,label,sample_id=dataset[index]
        if sample_id not in cam_sums: raise RuntimeError(f"Missing accumulated Grad-CAM for sample {sample_id}")
        cam=normalize_cam(cam_sums[sample_id]/float(number_of_models)); overlay=create_gradcam_overlay(denormalize_ct(image,image_mean,image_std),cam); folder=root/prediction_category(int(label.item()),int(ensemble_predictions[index])); ensure_dir(folder); cv2.imwrite(str(folder/f"{sample_id}.png"),overlay)

def summarise_fold_metrics(fold_results):
    names=["accuracy","balanced_accuracy","auc_roc","auc_pr","f1","precision","recall","log_loss","false_negatives","false_positives"]; out={}
    for name in names:
        m,s=mean_std([f["test_metrics"][name] for f in fold_results]); out[name]={"mean":m,"std":s}
    return out

def main(config):
    resolved=resolve_configuration(config); seed=int(config.get("seed",42)); n_folds=int(config.get("n_folds",5))
    if n_folds!=5: raise ValueError("This experiment requires exactly five predefined folds.")
    set_global_seed(seed); pl.seed_everything(seed,workers=True); output_dir=Path(resolved["output_folder"]); ensure_dir(output_dir)
    if resolved["mode"]=="train": ensure_dir(resolved["checkpoint_root"])
    data_root=str(config["data_root"]); test_data_root=str(config.get("test_data_root",data_root)); image_size=int(config.get("img_size",448)); image_mean=tuple(config.get("image_mean",[0.485,0.456,0.406])); image_std=tuple(config.get("image_std",[0.229,0.224,0.225])); batch_size=int(config.get("batch_size",16)); num_workers=int(config.get("num_workers",8))
    gc_cfg=config.get("gradcam",{}); gradcam_enabled=bool(gc_cfg.get("enabled",True)); gradcam_save_per_fold=bool(gc_cfg.get("save_per_fold",True)); gradcam_save_ensemble=bool(gc_cfg.get("save_ensemble_average",True)); gradcam_dir=output_dir/"gradcam"; ensure_dir(gradcam_dir) if gradcam_enabled else None
    manifest={"mode":resolved["mode"],"train_dataset":resolved["training_dataset"],"test_dataset":resolved["test_dataset"],"data_percentage":resolved["percentage"],"training_folder":str(resolved["training_folder"]),"test_file":str(resolved["test_file"]),"fold_files":[str(p) for p in resolved["fold_files"]],"test_policy":"Only the fixed original test_dataset/test.xlsx is used for final testing."}; save_json({**config,"resolved":manifest},output_dir/"resolved_config.json")
    print("\n"+"="*80+"\nCT-ONLY RADIODINO EXPERIMENT\n"+"="*80); print(f"Mode:             {resolved['mode']}\nTraining dataset: {resolved['training_dataset']}\nTest dataset:     {resolved['test_dataset']}\nData percentage:  {resolved['percentage']}%\nTraining folder:  {resolved['training_folder']}\nIndependent test: {resolved['test_file']}")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Device:           {device}")
    fold_results=[]; fold_probabilities=[]; reference_test_ids=None; reference_test_targets=None; reference_test_dataset=None; cam_sums={}
    for fold_index in range(n_folds):
        fold_number=fold_index+1; print("\n"+"="*80+f"\nFOLD MODEL {fold_number}/{n_folds}\n"+"="*80); fold_seed=seed+fold_number; set_global_seed(fold_seed); pl.seed_everything(fold_seed,workers=True); fold_output_dir=output_dir/f"fold{fold_number}"; ensure_dir(fold_output_dir)
        validation_file=resolved["fold_files"][fold_index]; training_files=[p for i,p in enumerate(resolved["fold_files"]) if i!=fold_index]
        dm=Valve2DImageDataModule(data_root=data_root,test_data_root=test_data_root,train_excel_paths=training_files,validation_excel_path=validation_file,test_excel_path=resolved["test_file"],batch_size=batch_size,num_workers=num_workers,target_size=(image_size,image_size),seed=fold_seed,image_mean=image_mean,image_std=image_std,artifacts_dir=fold_output_dir); dm.setup()
        save_json({"fold":fold_number,"training_files":[str(p) for p in training_files],"validation_file":str(validation_file),"test_file":str(resolved["test_file"]),"training_samples":len(dm.train_dataset),"validation_samples":len(dm.validation_dataset),"test_samples":len(dm.test_dataset)},fold_output_dir/"fold_manifest.json"); save_lines(fold_output_dir/"train_ids.txt",dm.train_ids); save_lines(fold_output_dir/"validation_ids.txt",dm.validation_ids); save_lines(fold_output_dir/"test_ids.txt",dm.test_ids)
        current_ids=list(dm.test_ids); current_targets=np.asarray(dm.test_dataset.labels,dtype=np.int64)
        if reference_test_ids is None: reference_test_ids=current_ids; reference_test_targets=current_targets; reference_test_dataset=dm.test_dataset
        else:
            if current_ids!=reference_test_ids: raise RuntimeError("Independent test sample order changed between folds.")
            if not np.array_equal(current_targets,reference_test_targets): raise RuntimeError("Independent test labels changed between folds.")
        if resolved["mode"]=="train":
            fold_checkpoint_dir=Path(resolved["checkpoint_root"])/f"fold{fold_number}"; ensure_dir(fold_checkpoint_dir)
            model=Valve2DRadioDINOImageModel(learning_rate=float(config["learning_rate"]),class_weights=dm.class_weights.tolist(),dropout_rate_image=float(config["dropout_rate_image"]),weight_decay=float(config["weight_decay"]),label_smoothing=float(config["label_smoothing"]),warmup_epochs=int(config["warmup_epochs"]),min_lr_factor=float(config["min_lr_factor"]),freeze_bn=bool(config["freeze_bn"]),unfreeze_epoch=int(config["unfreeze_epoch"]),unfreeze_last_n_blocks=int(config["unfreeze_last_n_blocks"]),backbone_lr_multiplier=float(config["backbone_lr_multiplier"]),max_epochs=int(config["max_epochs"]))
            ckpt=ModelCheckpoint(dirpath=fold_checkpoint_dir,filename="best-checkpoint",monitor=config["checkpoint_metric"],mode=config["checkpoint_mode"],save_top_k=1,save_last=True); callbacks=[ckpt,EarlyStopping(monitor=config["early_stopping_metric"],mode=config["early_stopping_mode"],patience=int(config["patience"]),min_delta=float(config["early_stopping_min_delta"]),verbose=True),LearningRateMonitor(logging_interval="epoch")]
            if bool(config.get("use_swa",False)): callbacks.append(StochasticWeightAveraging(swa_lrs=float(config["swa_lr"]),swa_epoch_start=config["swa_epoch_start"]))
            logger=None
            if bool(config.get("use_wandb",True)):
                if wandb is None: raise ImportError("use_wandb=true but wandb is not installed.")
                logger=WandbLogger(project=config["wandb_project"],name=f"{config['run_name']}_{resolved['percentage']}pct_fold{fold_number}",group=f"{config['run_name']}_{resolved['percentage']}pct",log_model=False); logger.log_hyperparams({**config,"fold":fold_number,"data_percentage":resolved["percentage"]})
            trainer=pl.Trainer(max_epochs=int(config["max_epochs"]),logger=logger,callbacks=callbacks,accelerator="gpu" if torch.cuda.is_available() else "cpu",devices=int(config.get("num_gpus",1)) if torch.cuda.is_available() else 1,precision="16-mixed" if torch.cuda.is_available() else "32-true",accumulate_grad_batches=int(config["accumulate_grad_batches"]),gradient_clip_val=0.5,deterministic=True,log_every_n_steps=5); trainer.fit(model,datamodule=dm); best_path=ckpt.best_model_path
            if not best_path: raise RuntimeError(f"No best checkpoint saved for fold {fold_number}.")
            best_model=Valve2DRadioDINOImageModel.load_from_checkpoint(best_path,map_location=device)
        else:
            logger=None; trainer=None; best_path=find_existing_checkpoint(resolved["checkpoint_root"],fold_number); print(f"Loading checkpoint: {best_path}"); best_model=Valve2DRadioDINOImageModel.load_from_checkpoint(str(best_path),map_location=device)
        best_model=best_model.to(device); best_model.decision_threshold=0.5
        val_p,val_y,val_ids=collect_predictions(best_model,dm.val_dataloader(),device); validation_metrics,_=save_evaluation(val_y,val_p,val_ids,0.5,fold_output_dir,"validation",fold_number); save_json({"threshold":0.5,"policy":"Fixed threshold; no calibration or optimization.","checkpoint":str(best_path)},fold_output_dir/"decision_threshold.json")
        test_p,test_y,test_ids=collect_predictions(best_model,dm.test_dataloader(),device)
        if test_ids!=reference_test_ids or not np.array_equal(test_y,reference_test_targets): raise RuntimeError("Independent test set changed during prediction.")
        test_metrics,_=save_evaluation(test_y,test_p,test_ids,0.5,fold_output_dir,"test",fold_number); fold_probabilities.append(test_p); fold_results.append({"fold":fold_number,"threshold":0.5,"validation_metrics":validation_metrics,"test_metrics":test_metrics,"best_checkpoint":str(best_path)}); print(f"\nFold {fold_number} independent test:\nAccuracy: {test_metrics['accuracy']:.4f}\nF1:       {test_metrics['f1']:.4f}\nAUC-ROC:  {test_metrics['auc_roc']:.4f}")
        if gradcam_enabled and gradcam_save_per_fold: generate_fold_gradcams(best_model,dm.test_dataset,test_p,0.5,gradcam_dir,fold_number,device,image_mean,image_std,cam_sums)
        if logger is not None:
            logger.experiment.log({"decision_threshold":0.5,**{f"test/{k}":v for k,v in test_metrics.items() if isinstance(v,(int,float))}}); logger.experiment.finish(); wandb.finish() if wandb is not None else None
        del best_model,dm
        if resolved["mode"]=="train": del model,trainer
        clean_memory()
    save_json(fold_results,output_dir/"fold_results.json"); fold_summary=summarise_fold_metrics(fold_results); matrix=np.stack(fold_probabilities,axis=0); ensemble_probabilities=matrix.mean(axis=0); ensemble_metrics,_=save_evaluation(reference_test_targets,ensemble_probabilities,reference_test_ids,0.5,output_dir,"ensemble_test",None); ensemble_predictions=(ensemble_probabilities>=0.5).astype(int)
    table=pd.DataFrame({"ID":reference_test_ids,"true_label":reference_test_targets});
    for fold_number,p in enumerate(fold_probabilities,start=1): table[f"fold{fold_number}_prob_pacemaker"]=p
    table["ensemble_prob_pacemaker"]=ensemble_probabilities; table["ensemble_prediction"]=ensemble_predictions; table.to_csv(output_dir/"ensemble_predictions.csv",index=False)
    if gradcam_enabled and gradcam_save_per_fold and gradcam_save_ensemble: save_ensemble_gradcams(reference_test_dataset,cam_sums,ensemble_predictions,gradcam_dir,n_folds,image_mean,image_std)
    summary={"mode":resolved["mode"],"train_dataset":resolved["training_dataset"],"test_dataset":resolved["test_dataset"],"data_percentage":resolved["percentage"],"fold_mean_std":fold_summary,"five_model_ensemble":ensemble_metrics}; save_json(summary,output_dir/"cv_summary.json")
    with (output_dir/"cv_results.txt").open("w",encoding="utf-8") as h:
        h.write("CT-ONLY RADIODINO RESULTS\n"+"="*80+"\n\n"); h.write(f"Training dataset: {resolved['training_dataset']}\nTest dataset: {resolved['test_dataset']}\nData percentage: {resolved['percentage']}%\n\nMEAN ± STD ACROSS FIVE FOLD MODELS\n"+"-"*80+"\n");
        for name,values in fold_summary.items(): h.write(f"{name}: {values['mean']:.6f} ± {values['std']:.6f}\n")
        h.write("\nFIVE-MODEL ENSEMBLE\n"+"-"*80+"\n");
        for name,value in ensemble_metrics.items(): h.write(f"{name}: {value}\n")
    print("\n"+"="*80+"\nFINAL RESULTS\n"+"="*80); print(f"Training dataset: {resolved['training_dataset']}\nTest dataset:     {resolved['test_dataset']}\nTraining size:    {resolved['percentage']}%\n\nMEAN ± STD ACROSS FIVE FOLD MODELS\n"+"-"*80); print(f"Accuracy: {fold_summary['accuracy']['mean']:.4f} ± {fold_summary['accuracy']['std']:.4f}\nF1 Score: {fold_summary['f1']['mean']:.4f} ± {fold_summary['f1']['std']:.4f}\nAUC-ROC:  {fold_summary['auc_roc']['mean']:.4f} ± {fold_summary['auc_roc']['std']:.4f}\n\nFIVE-MODEL ENSEMBLE\n"+"-"*80); print(f"Accuracy: {ensemble_metrics['accuracy']:.4f}\nF1 Score: {ensemble_metrics['f1']:.4f}\nAUC-ROC:  {ensemble_metrics['auc_roc']:.4f}\n\nResults saved to: {output_dir}")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="CT-only RadioDINO predefined-fold experiment."); parser.add_argument("--config_file",type=str,default="config.json"); args=parser.parse_args(); config_path=Path(args.config_file)
    if not config_path.is_file(): raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r",encoding="utf-8") as h: configuration=json.load(h)
    if torch.cuda.is_available(): torch.cuda.set_device(0)
    main(configuration)
