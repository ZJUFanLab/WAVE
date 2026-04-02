import torch
import torch.nn.functional as F
import numpy as np
import logging
import sys
import os
import scanpy as sc
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
from scipy.stats import pearsonr
from collections import defaultdict, OrderedDict

from utils import seed_everything, log_config
from model import WAVE
from load_dataset import SingleDrugDataset, single_cell_collate
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)] 
)

logger = logging.getLogger()


def compute_detailed_metrics(preds, truths, unperts, prefix=""):
    results = {}
    
    def safe_pcc(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return 0.0
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                pcc, _ = pearsonr(a, b)
            return pcc if not np.isnan(pcc) else 0.0
        except:
            return 0.0

    results[f'{prefix}Pearson_Global'] = safe_pcc(preds.flatten(), truths.flatten())
    
    ss_res_abs = np.sum((truths - preds) ** 2)
    ss_tot_abs = np.sum((truths - np.mean(truths)) ** 2)
    results[f'{prefix}R2_Global'] = 1 - ss_res_abs / (ss_tot_abs + 1e-8)

    gene_pccs = []
    for g in range(truths.shape[1]):
        if np.std(truths[:, g]) > 1e-9:
            p = safe_pcc(preds[:, g], truths[:, g])
            gene_pccs.append(p)
    results[f'{prefix}Mean_Gene_PCC'] = np.mean(gene_pccs) if gene_pccs else 0.0

    delta_preds = preds - unperts
    delta_truths = truths - unperts
    
    results[f'{prefix}Delta_Pearson_Global'] = safe_pcc(delta_preds.flatten(), delta_truths.flatten())

    top_deg_pccs = []
    for n in range(truths.shape[0]):
        diff = np.abs(delta_truths[n, :])
        top_indices = np.argsort(diff)[-50:]
        p = safe_pcc(delta_preds[n, top_indices], delta_truths[n, top_indices])
        top_deg_pccs.append(p)
    results[f'{prefix}Top50_DEG_Delta_PCC'] = np.mean(top_deg_pccs)

    delta_pred_c = delta_preds - np.mean(delta_preds, axis=1, keepdims=True)
    delta_truth_c = delta_truths - np.mean(delta_truths, axis=1, keepdims=True)
    delta_pred_ss = np.sum(delta_pred_c**2, axis=1)
    delta_truth_ss = np.sum(delta_truth_c**2, axis=1)
    
    mask = (delta_pred_ss > 1e-12) & (delta_truth_ss > 1e-12)
    if np.sum(mask) == 0: 
        results[f'{prefix}Delta_Mean_Cell_PCC'] = 0.0
    else:
        num = np.sum(delta_pred_c[mask] * delta_truth_c[mask], axis=1)
        den = np.sqrt(delta_pred_ss[mask] * delta_truth_ss[mask])
        results[f'{prefix}Delta_Mean_Cell_PCC'] = np.mean(num / (den + 1e-8))

    ss_res_cell = np.sum((delta_truths - delta_preds)**2, axis=1)
    ss_tot_cell = delta_truth_ss
    r2_mask = ss_tot_cell > 1e-12
    r2_scores = np.zeros_like(ss_res_cell)
    r2_scores[r2_mask] = 1.0 - (ss_res_cell[r2_mask] / ss_tot_cell[r2_mask])
    results[f'{prefix}Delta_Mean_Cell_R2'] = np.mean(r2_scores)
    
    return results

def loss_fct(pred, target, mu, logvar, alpha=1.0, beta=0.01, huber_delta=1.0):
    recon_loss = F.huber_loss(pred, target, delta=huber_delta)

    pred_mean = pred - pred.mean(dim=1, keepdim=True)
    target_mean = target - target.mean(dim=1, keepdim=True)
    numerator = torch.sum(pred_mean * target_mean, dim=1)
    denominator = torch.sqrt(torch.sum(pred_mean ** 2, dim=1) * torch.sum(target_mean ** 2, dim=1) + 1e-8)
    pearson_loss = 1 - torch.mean(numerator / denominator)

    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)

    weighted_pearson = alpha * pearson_loss
    weighted_kld = beta * kld_loss
    total_loss = recon_loss + weighted_pearson + weighted_kld

    return total_loss, recon_loss, weighted_pearson, weighted_kld

def train_epoch(model, loader, optimizer, cfg, current_beta):
    model.train()
    total_loss_sum, recon_loss_sum, pearson_loss_sum, kld_loss_sum = 0.0, 0.0, 0.0, 0.0
    
    for batch in loader:
        unpert_expr = batch['unpert_expr'].to(cfg.device)
        drug_fp = batch['drug_fp'].to(cfg.device)
        pert_expr = batch['pert_expr'].to(cfg.device)

        optimizer.zero_grad()
        pred, mu, logvar, _ = model(unpert_expr, drug_fp)
        
        loss, recon, pcc, kld = loss_fct(
            pred, pert_expr, mu, logvar, 
            alpha=cfg.alpha_pearson, 
            beta=current_beta,
            huber_delta=getattr(cfg, 'huber_delta', 1.0)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss_sum += loss.item()
        recon_loss_sum += recon.item()
        pearson_loss_sum += pcc.item()
        kld_loss_sum += kld.item()
        
    num_batches = len(loader)
    return (
        total_loss_sum / num_batches, 
        recon_loss_sum / num_batches, 
        pearson_loss_sum / num_batches, 
        kld_loss_sum / num_batches
    )

def validate(model, loader, cfg):
    model.eval()
    all_preds, all_truths, all_unpert = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            unpert = batch['unpert_expr'].to(cfg.device)
            drug = batch['drug_fp'].to(cfg.device)
            truth = batch['pert_expr'].to(cfg.device)
            
            pred, _, _, _ = model(unpert, drug)
            
            all_preds.append(pred.cpu().numpy())
            all_truths.append(truth.cpu().numpy())
            all_unpert.append(unpert.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    truths = np.concatenate(all_truths, axis=0)
    unperts = np.concatenate(all_unpert, axis=0)
    
    delta_pred = preds - unperts
    delta_truth = truths - unperts
    
    delta_pred_c = delta_pred - np.mean(delta_pred, axis=1, keepdims=True)
    delta_truth_c = delta_truth - np.mean(delta_truth, axis=1, keepdims=True)
    delta_pred_ss = np.sum(delta_pred_c**2, axis=1)
    delta_truth_ss = np.sum(delta_truth_c**2, axis=1)
    
    mask = (delta_pred_ss > 1e-12) & (delta_truth_ss > 1e-12)
    if np.sum(mask) == 0: 
        mean_pcc = 0.0
    else:
        num = np.sum(delta_pred_c[mask] * delta_truth_c[mask], axis=1)
        den = np.sqrt(delta_pred_ss[mask] * delta_truth_ss[mask])
        mean_pcc = np.mean(num / (den + 1e-8))

    ss_res = np.sum((delta_truth - delta_pred)**2, axis=1)
    ss_tot = delta_truth_ss
    r2_mask = ss_tot > 1e-12
    r2_scores = np.zeros_like(ss_res)
    r2_scores[r2_mask] = 1.0 - (ss_res[r2_mask] / ss_tot[r2_mask])
    mean_r2 = np.mean(r2_scores)
    
    return mean_pcc, mean_r2

def test_and_save(cfg, model):
    logger.info(f"Loading test data from manifest: {cfg.test_path}")
    test_set = SingleDrugDataset(cfg.test_path, cfg, mode='test') 
    
    test_loader = DataLoader(
        test_set, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        collate_fn=single_cell_collate
    )
    
    all_preds, all_truths, all_unperts = [], [], []
    model.eval()
    logger.info("Starting inference...")
    
    with torch.no_grad():
        for batch in test_loader:
            if batch is None: continue
                
            unpert = batch['unpert_expr'].to(cfg.device) 
            truth = batch['pert_expr'].to(cfg.device)
            drug = batch['drug_fp'].to(cfg.device)
            
            if drug.dim() == 1:
                drug = drug.unsqueeze(0).expand(unpert.size(0), -1)
            
            pred, _, _, _ = model(unpert, drug)
            
            all_preds.append(pred.cpu().numpy())
            all_truths.append(truth.cpu().numpy())
            all_unperts.append(unpert.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_truths = np.concatenate(all_truths, axis=0)
    all_unperts = np.concatenate(all_unperts, axis=0)

    metrics = compute_detailed_metrics(all_preds, all_truths, all_unperts, prefix="")

def run_single_fold(fold_id):
    logger.info(f"🚀 STARTING {fold_id}")
    cfg = Config(fold_id=fold_id)
    seed_everything(cfg.seed)
    log_config(cfg, logger)

    train_set = SingleDrugDataset(cfg.train_path, cfg, mode='train')
    val_set = SingleDrugDataset(cfg.val_path, cfg, mode='val')
    
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, 
        num_workers=cfg.num_workers, collate_fn=single_cell_collate
    )
    
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=cfg.num_workers, collate_fn=single_cell_collate
    )
    
    model = WAVE(cfg).to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_score = -float('inf')
    counter = 0
    min_lr_counter = 0

    kl_warmup_epochs = getattr(cfg, 'kl_anneal_epochs', 23) 
    target_beta = cfg.beta_kl

    for epoch in range(cfg.epochs):
        if epoch < kl_warmup_epochs:
            current_beta = target_beta * ((epoch + 1) / kl_warmup_epochs)
        else:
            current_beta = target_beta

        train_loss, recon_loss, pcc_loss, kld_loss = train_epoch(model, train_loader, optimizer, cfg, current_beta)
        
        mean_pcc, mean_r2 = validate(model, val_loader, cfg)

        scheduler.step()

        recon_pct = (recon_loss / train_loss) * 100 if train_loss > 0 else 0
        pcc_pct = (pcc_loss / train_loss) * 100 if train_loss > 0 else 0
        kld_pct = (kld_loss / train_loss) * 100 if train_loss > 0 else 0

        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(
            f"[{fold_id}] Ep {epoch+1:03d} | LR: {current_lr:.1e} | "
            f"Loss: {train_loss:.4f} [Rec:{recon_pct:.1f}% Pcc:{pcc_pct:.1f}% KLD:{kld_pct:.1f}%] | "
            f"MeanPCC: {mean_pcc:.4f} | R2: {mean_r2:.4f}"
        )

        if mean_pcc > (best_score + getattr(cfg, 'min_delta', 0.0)):
            best_score = mean_pcc
            torch.save(model.state_dict(), cfg.model_save_path)
            logger.info(f"--> Best model saved (MeanPCC: {best_score:.4f})")
            counter = 0 
            min_lr_counter = 0  
        else:
            counter += 1
            if counter % 5 == 0:
                logger.info(f"--> No improvement for {counter} epochs.")
            

        if counter >= getattr(cfg, 'patience', 30):
            logger.info(f"Early stop triggered ({getattr(cfg, 'patience', 30)})，training stop at epoch {epoch+1}!")
            break
        
    logger.info(f"[{fold_id}] Finished training. Best MeanPCC: {best_score:.4f}. Starting testing...")
    
    model.load_state_dict(torch.load(cfg.model_save_path, weights_only=True))
    test_and_save(cfg, model)
    
    del model, optimizer, scheduler, train_loader, val_loader, train_set, val_set
    torch.cuda.empty_cache()
    logger.info(f"Finished {fold_id}.\n")

def main():
    folds_to_run = ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]
    for fold in folds_to_run:
        try:
            run_single_fold(fold)
        except Exception as e:
            logger.error(f"Error occurred in {fold}: {e}")
            continue

if __name__ == "__main__":
    main()