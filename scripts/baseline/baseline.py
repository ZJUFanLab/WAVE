import torch
import numpy as np
import argparse
import os
import pandas as pd
from torch.utils.data import DataLoader
import glob
from collections import defaultdict
from scipy.stats import pearsonr

from config import Config
from load_dataset import GeneDrugDataset 

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def get_args():
    parser = argparse.ArgumentParser(description="Professional Evaluation with Baselines and Biological Metrics")
    parser.add_argument('--split_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--eval_file', type=str, default='test.h5ad')
    parser.add_argument('--batch_size', type=int, default=256)
    return parser.parse_args()

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
    
    delta_preds = preds - unperts
    delta_truths = truths - unperts
    results[f'{prefix}Delta_Pearson_Global'] = safe_pcc(delta_preds.flatten(), delta_truths.flatten())

    gene_pccs = []
    for g in range(truths.shape[1]):
        p = safe_pcc(preds[:, g], truths[:, g])
        if np.std(truths[:, g]) > 1e-9:
            gene_pccs.append(p)
    results[f'{prefix}Mean_Gene_PCC'] = np.mean(gene_pccs) if gene_pccs else 0.0

    sample_pccs = [safe_pcc(preds[n, :], truths[n, :]) for n in range(truths.shape[0])]
    results[f'{prefix}Mean_Sample_PCC'] = np.mean(sample_pccs)

    ss_res = np.sum((truths - preds) ** 2)
    ss_tot = np.sum((truths - np.mean(truths)) ** 2)
    results[f'{prefix}R2_Global'] = 1 - ss_res / (ss_tot + 1e-8)

    top_deg_pccs = []
    for n in range(truths.shape[0]):
        diff = np.abs(delta_truths[n, :])
        top_indices = np.argsort(diff)[-50:]
        p = safe_pcc(delta_preds[n, top_indices], delta_truths[n, top_indices])
        top_deg_pccs.append(p)
    results[f'{prefix}Top50_DEG_Delta_PCC'] = np.mean(top_deg_pccs)
    
    return results

def get_train_statistics(train_loader):
    print("Computing training statistics...")
    drug_delta_sum = defaultdict(lambda: 0)
    drug_count = defaultdict(lambda: 0)
    cell_expr_sum = defaultdict(lambda: 0)
    cell_count = defaultdict(lambda: 0)
    
    all_pert_sum = 0
    total_samples = 0
    global_delta_sum = 0

    for batch in train_loader:
        unpert = batch['unpert_expr'].numpy()
        pert = batch['pert_expr'].numpy()
        smiles_list = batch['smiles']
        cell_list = batch['cell']
        deltas = pert - unpert
        
        all_pert_sum += np.sum(pert, axis=0)
        global_delta_sum += np.sum(deltas, axis=0)
        total_samples += len(smiles_list)
        
        for i in range(len(smiles_list)):
            smi, ctype = smiles_list[i], cell_list[i]
            drug_delta_sum[smi] += deltas[i]
            drug_count[smi] += 1
            cell_expr_sum[ctype] += pert[i]
            cell_count[ctype] += 1

    global_mean_pert = all_pert_sum / total_samples
    global_mean_delta = global_delta_sum / total_samples
    per_drug_mean_delta = {s: v / drug_count[s] for s, v in drug_delta_sum.items()}
    per_cell_mean_expr = {c: v / cell_count[c] for c, v in cell_expr_sum.items()}
    
    return global_mean_pert, global_mean_delta, per_drug_mean_delta, per_cell_mean_expr

def process_single_fold(fold_name, train_path, eval_path, batch_size):
    print(f"--- [{fold_name}] Evaluation Start ---")
    config = Config(fold_id=fold_name)
    train_loader = DataLoader(GeneDrugDataset(train_path, config), batch_size=batch_size)
    eval_loader = DataLoader(GeneDrugDataset(eval_path, config), batch_size=batch_size)
    
    g_mean_pert, g_mean_delta, p_drug_delta, p_cell_expr = get_train_statistics(train_loader)
    
    all_truths, all_unperts = [], []
    eval_cells, eval_smiles = [], []
    
    for batch in eval_loader:
        all_truths.append(batch['pert_expr'].numpy())
        all_unperts.append(batch['unpert_expr'].numpy())
        eval_cells.extend(batch['cell']) 
        eval_smiles.extend(batch['smiles']) 
    
    all_truths = np.concatenate(all_truths, axis=0)
    all_unperts = np.concatenate(all_unperts, axis=0)
    N, G = all_truths.shape

    preds_id = all_unperts
    
    preds_g_mean = np.tile(g_mean_pert, (N, 1))
    
    preds_c_mean = []
    for i, c in enumerate(eval_cells):
        preds_c_mean.append(p_cell_expr.get(c, g_mean_pert))
    preds_c_mean = np.array(preds_c_mean)

    preds_d_mean = []
    for i, s in enumerate(eval_smiles):
        delta = p_drug_delta.get(s, g_mean_delta)
        preds_d_mean.append(all_unperts[i] + delta)
    preds_d_mean = np.array(preds_d_mean)

    methods = {
        "Baseline: Identity": preds_id,
        "Baseline: Global Mean": preds_g_mean,
        "Baseline: Cell Mean": preds_c_mean,
        "Baseline: Drug Mean Delta": preds_d_mean
    }
    
    fold_results = []
    for name, p_val in methods.items():
        m = compute_detailed_metrics(p_val, all_truths, all_unperts)
        m["Method"] = name
        m["Fold"] = fold_name
        fold_results.append(m)
        
    return fold_results

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    fold_dirs = sorted(glob.glob(os.path.join(args.split_dir, "Fold*")))
    
    all_results = []
    for fold_dir in fold_dirs:
        fold_name = os.path.basename(fold_dir)
        train_path = os.path.join(fold_dir, "train.h5ad")
        eval_path = os.path.join(fold_dir, args.eval_file)
        if os.path.exists(train_path) and os.path.exists(eval_path):
            res = process_single_fold(fold_name, train_path, eval_path, args.batch_size)
            all_results.extend(res)

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.output_dir, "detailed_baselines.csv"), index=False)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = df.groupby("Method")[numeric_cols].agg(['mean', 'std'])
    
    print("\n" + "="*80)
    print("FINAL SUMMARY (Averaged across Folds)")
    print("="*80)
    core_metrics = ["Delta_Pearson_Global", "Mean_Gene_PCC", "Top50_DEG_Delta_PCC", "R2_Global"]

    present_metrics = [m for m in core_metrics if m in summary.columns.get_level_values(0)]
    print(summary[present_metrics].xs('mean', axis=1, level=1))
    
    summary.to_csv(os.path.join(args.output_dir, "summary_stats.csv"))
if __name__ == "__main__":
    main()