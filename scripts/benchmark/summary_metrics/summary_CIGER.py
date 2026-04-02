import os, sys, re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import scanpy as sc


def calc_rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2, axis=1))

def calc_sse(y_pred, y_true):
    return np.sum((y_pred - y_true) ** 2, axis=1)

def calc_mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2, axis=1)

def calc_mae(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true), axis=1)

def calc_r2(preds, targets):
    ss_res = np.sum((targets - preds) ** 2, axis=1)
    ss_tot = np.sum((targets - np.mean(targets, axis=1, keepdims=True)) ** 2, axis=1)
    return 1 - ss_res / ss_tot

def calc_pearson(preds, targets):
    r_values = []
    for p, t in zip(preds, targets):
        if np.std(p) == 0 or np.std(t) == 0:
            r_values.append(np.nan)
        else:
            r_values.append(pearsonr(p, t)[0])
    return np.array(r_values)

def compute_pearson_top50(true_expr, gene_expr, output):
    pearson_list = []
    for i in range(true_expr.shape[0]):
        diff_expr = true_expr[i] - gene_expr[i]
        top50_indices = np.argsort(np.abs(diff_expr))[::-1][:50]
        true_top50 = true_expr[i, top50_indices]
        output_top50 = output[i, top50_indices]
        if np.std(output_top50) == 0 or np.std(true_top50) == 0:
            pearson_list.append(np.nan)
        else:
            pearson_list.append(pearsonr(true_top50, output_top50)[0])
    return np.array(pearson_list)


def compute_precision_recall_at_k(delta_pred, delta_truth, k=100):
    n_samples, n_genes = delta_truth.shape
    k = min(k, n_genes)

    pos_prec = np.zeros(n_samples)
    neg_prec = np.zeros(n_samples)
    pos_recall = np.zeros(n_samples)
    neg_recall = np.zeros(n_samples)

    for i in range(n_samples):
        truth_pos = set(np.argsort(delta_truth[i])[-k:])
        pred_pos  = set(np.argsort(delta_pred[i])[-k:])
        truth_neg = set(np.argsort(delta_truth[i])[:k])
        pred_neg  = set(np.argsort(delta_pred[i])[:k])

        pos_overlap = len(pred_pos & truth_pos)
        neg_overlap = len(pred_neg & truth_neg)

        pos_prec[i]   = pos_overlap / k
        neg_prec[i]   = neg_overlap / k
        pos_recall[i] = pos_overlap / len(truth_pos) if len(truth_pos) > 0 else 0.0
        neg_recall[i] = neg_overlap / len(truth_neg) if len(truth_neg) > 0 else 0.0

    return {
        f'positive_precision_at_{k}': pos_prec,
        f'negative_precision_at_{k}': neg_prec,
        f'positive_recall_at_{k}':    pos_recall,
        f'negative_recall_at_{k}':    neg_recall,
    }


# ===== Configuration =====
SOFTWARE = "CIGER"
Folds = ['Fold1']
K_LIST = [50, 100]

CIGER_BASE = "/path/to/CIGER"
ADATA_PATH = "/path/to/test_L1000_level4_benchmark.h5ad"

# ===== Load sample IDs for CIGER (subset of test set) =====
sample_id = pd.read_csv(
    os.path.join(CIGER_BASE, "Fold1", "test_sample_ids.txt"), header=None
)[0].tolist()

# ===== Load truth from CIGER's own csv =====
truth_path = os.path.join(CIGER_BASE, "Fold1", "train_val_test.csv")
truth_df = pd.read_csv(truth_path, index_col=0).iloc[:, 4:]
truth_df = truth_df[truth_df.index.isin(sample_id)]
truth = truth_df.values

# ===== Load control (unpert_expr) from adata, subset to matching samples =====
adata = sc.read_h5ad(ADATA_PATH)
adata = adata[adata.obs.index.isin(sample_id)]
ctrl = adata.layers['unpert_expr']

# ===== Per-fold mean lists (for summary) =====
fold_means = {
    'RMSE': [], 'SSE': [], 'MSE': [], 'MAE': [],
    'R2': [], 'Pearson': [], 'Delta_x_pearson': [], 'top50': [],
}
for k in K_LIST:
    fold_means[f'positive_precision_at_{k}'] = []
    fold_means[f'negative_precision_at_{k}'] = []
    fold_means[f'positive_recall_at_{k}'] = []
    fold_means[f'negative_recall_at_{k}'] = []

# ===== Per-sample lists (for detailed output) =====
all_samples = {key: [] for key in fold_means}
all_samples['Fold'] = []
all_samples['Cell'] = []

for Fold in Folds:
    pred_f = os.path.join(CIGER_BASE, Fold, "prediction.txt")
    pred = np.loadtxt(pred_f, delimiter=",")

    delta_pred  = pred - ctrl
    delta_truth = truth - ctrl

    # Basic metrics
    rmse    = calc_rmse(pred, truth)
    sse     = calc_sse(pred, truth)
    mse     = calc_mse(pred, truth)
    mae     = calc_mae(pred, truth)
    r2      = calc_r2(pred, truth)
    pearson = calc_pearson(pred, truth)
    delta_p = calc_pearson(delta_pred, delta_truth)
    top50   = compute_pearson_top50(truth, ctrl, pred)

    # Precision@k and Recall@k
    pk_results = {}
    for k in K_LIST:
        pk = compute_precision_recall_at_k(delta_pred, delta_truth, k=k)
        pk_results.update(pk)

    # Fold means (nan-robust)
    fold_means['RMSE'].append(np.nanmean(rmse))
    fold_means['SSE'].append(np.nanmean(sse))
    fold_means['MSE'].append(np.nanmean(mse))
    fold_means['MAE'].append(np.nanmean(mae))
    fold_means['R2'].append(np.nanmean(r2))
    fold_means['Pearson'].append(np.nanmean(pearson))
    fold_means['Delta_x_pearson'].append(np.nanmean(delta_p))
    fold_means['top50'].append(np.nanmean(top50))
    for key in pk_results:
        fold_means[key].append(np.nanmean(pk_results[key]))

    # Per-sample accumulation
    n = len(rmse)
    all_samples['RMSE']             += rmse.tolist()
    all_samples['SSE']              += sse.tolist()
    all_samples['MSE']              += mse.tolist()
    all_samples['MAE']              += mae.tolist()
    all_samples['R2']               += r2.tolist()
    all_samples['Pearson']          += pearson.tolist()
    all_samples['Delta_x_pearson']  += delta_p.tolist()
    all_samples['top50']            += top50.tolist()
    for key in pk_results:
        all_samples[key] += pk_results[key].tolist()
    all_samples['Fold'] += [Fold] * n
    all_samples['Cell'] += adata.obs['cell'].tolist()


# ===== Summary table =====
results = []
for metric, values in fold_means.items():
    results.append({
        'Metric': metric,
        'Mean': f"{np.nanmean(values):.5f}",
        'Std':  f"{np.nanstd(values):.5f}"
    })

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv(f"{SOFTWARE}_metrics_sum.csv", index=False)

# ===== Per-sample table =====
metrics_df = pd.DataFrame(all_samples)
metrics_df.to_csv(f"{SOFTWARE}_metrics.csv", index=False)

print(f"\nSaved {SOFTWARE}_metrics_sum.csv and {SOFTWARE}_metrics.csv")
print(f"Total samples: {len(metrics_df)}")

# ===== Per-cell summary table =====
metric_cols = [c for c in metrics_df.columns if c not in ('Fold', 'Cell')]
cell_summary = metrics_df.groupby('Cell')[metric_cols].agg(['mean', 'std'])
cell_summary.columns = [f"{col}_{stat}" for col, stat in cell_summary.columns]
cell_summary = cell_summary.reset_index()

cell_summary.to_csv(f"{SOFTWARE}_metrics_per_cell.csv", index=False)
print(f"\nSaved {SOFTWARE}_metrics_per_cell.csv ({len(cell_summary)} cell lines)")
print(cell_summary.to_string())