import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import matplotlib.colors as mcolors
import matplotlib
import os

sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams['axes.unicode_minus'] = False 

plt.rcParams['svg.fonttype'] = 'none' 
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

matplotlib.use('Agg')

def save_plot(output_dir, filename_base, dpi=300):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    png_path = os.path.join(output_dir, f"{filename_base}.png")
    svg_path = os.path.join(output_dir, f"{filename_base}.svg")
    
    plt.savefig(png_path, dpi=dpi, bbox_inches='tight', transparent=False)
    plt.savefig(svg_path, format='svg', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Saved: {png_path} and {svg_path}")

def load_gene_mapping(mapping_path):
    if not os.path.exists(mapping_path):
        print(f"Warning: Mapping file not found at {mapping_path}. Using raw IDs.")
        return {}
    
    print(f"Loading gene mapping from {mapping_path}...")
    try:
        df = pd.read_csv(mapping_path, sep='\t')
        df.columns = df.columns.str.strip()
        return dict(zip(df['ensembl_gene_id'], df['hgnc_symbol']))
    except Exception as e:
        print(f"Error loading mapping file: {e}")
        return {}

def get_readable_name(ensembl_id, mapping_dict):
    ensembl_id = str(ensembl_id).strip()
    
    if mapping_dict and ensembl_id in mapping_dict:
        symbol = mapping_dict[ensembl_id]
        if pd.notna(symbol) and str(symbol).lower() != 'nan':
            return symbol
            
    return ensembl_id

def generate_paper_figures(h5ad_path, mapping_path, output_dir="./"):
    print(f"Loading data from: {h5ad_path} ...")
    adata = sc.read_h5ad(h5ad_path)
    
    gene_map = load_gene_mapping(mapping_path)
    
    X_true = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X_pred = adata.layers['pred']
    X_unpert = adata.layers['unpert_expr']
    
    color_pred = "#fd6969"
    color_true = "#f7ca49"
    color_ctrl = "#dcdcdc"
    start_color = "#f7ca49"
    
    warm_cmap = mcolors.LinearSegmentedColormap.from_list("warm", [start_color, color_pred], N=256)

    print("\nGenerating Figure 1: Global Correlation...")
    
    delta_true = X_true - X_unpert
    delta_pred = X_pred - X_unpert
    
    flat_true = X_true.flatten()
    flat_pred = X_pred.flatten()
    flat_delta_true = delta_true.flatten()
    flat_delta_pred = delta_pred.flatten()
    
    pearson_abs, _ = pearsonr(flat_true, flat_pred)
    pearson_delta, _ = pearsonr(flat_delta_true, flat_delta_pred)
    
    if len(flat_true) > 50000:
        idx = np.random.choice(len(flat_true), 50000, replace=False)
    else:
        idx = np.arange(len(flat_true))

    fig_1a, ax_1a = plt.subplots(figsize=(7, 6))
    hb1 = ax_1a.hexbin(flat_true[idx], flat_pred[idx], gridsize=60, cmap=warm_cmap, mincnt=1, bins='log', edgecolors='none')
    fig_1a.colorbar(hb1, ax=ax_1a, label='Log10(Count + 1)')
    min_v, max_v = flat_true[idx].min(), flat_true[idx].max()
    ax_1a.plot([min_v, max_v], [min_v, max_v], color='#333333', linestyle='--', lw=1.5)
    ax_1a.set_title(f"Absolute Expression\nPearson r = {pearson_abs:.3f}")
    ax_1a.set_xlabel("True Expression")
    ax_1a.set_ylabel("Predicted Expression")
    plt.tight_layout()
    save_plot(output_dir, "Figure1A_Absolute_Expression")

    fig_1b, ax_1b = plt.subplots(figsize=(7, 6))
    hb2 = ax_1b.hexbin(flat_delta_true[idx], flat_delta_pred[idx], gridsize=60, cmap=warm_cmap, mincnt=1, bins='log', edgecolors='none')
    fig_1b.colorbar(hb2, ax=ax_1b, label='Log10(Count + 1)')
    min_d, max_d = flat_delta_true[idx].min(), flat_delta_true[idx].max()
    ax_1b.plot([min_d, max_d], [min_d, max_d], color='#333333', linestyle='--', lw=1.5)
    ax_1b.set_title(f"Delta Expression\nPearson r = {pearson_delta:.3f}")
    ax_1b.set_xlabel("True Delta (Perturbed - Control)")
    ax_1b.set_ylabel("Predicted Delta (Predicted - Control)")
    plt.tight_layout()
    save_plot(output_dir, "Figure1B_Delta_Expression")

    print("\nGenerating Figure 2: Gene-wise Metrics...")
    
    gene_vars = np.var(X_true, axis=0)
    threshold = np.percentile(gene_vars, 50) 
    hvg_indices = np.where(gene_vars > threshold)[0]
    
    r2_filtered = []
    pearson_filtered = []
    
    for i in hvg_indices:
        r2 = r2_score(X_true[:, i], X_pred[:, i])
        p, _ = pearsonr(X_true[:, i], X_pred[:, i])
        r2_filtered.append(r2)
        pearson_filtered.append(p)
        
    fig_2a, ax_2a = plt.subplots(figsize=(7, 6))
    sns.histplot(pearson_filtered, bins=30, kde=True, color=color_true, ax=ax_2a)
    ax_2a.axvline(np.mean(pearson_filtered), color=color_pred, linestyle='--', label=f'Mean: {np.mean(pearson_filtered):.2f}')
    ax_2a.set_title("Gene-wise Pearson Correlation")
    ax_2a.legend()
    plt.tight_layout()
    save_plot(output_dir, "Figure2A_Gene_Pearson")
    
    fig_2b, ax_2b = plt.subplots(figsize=(7, 6))
    sns.histplot(r2_filtered, bins=30, kde=True, color=color_pred, ax=ax_2b)
    ax_2b.axvline(np.mean(r2_filtered), color="#333333", linestyle='--', label=f'Mean: {np.mean(r2_filtered):.2f}')
    ax_2b.set_title("Gene-wise R2 Score")
    ax_2b.legend()
    plt.tight_layout()
    save_plot(output_dir, "Figure2B_Gene_R2")


    print("\nGenerating Figure 3: Case Study...")
    
    pert_counts = adata.obs['perturbation'].value_counts()
    valid_perts = [p for p in pert_counts.index if 'ctrl' not in p.lower() and 'dmso' not in p.lower()]
    
    if len(valid_perts) > 0:
        target_drug = valid_perts[0]
        drug_indices = adata.obs['perturbation'] == target_drug
        
        mean_true = np.mean(X_true[drug_indices], axis=0)
        mean_unpert = np.mean(X_unpert[drug_indices], axis=0)
        
        lfc_true = mean_true - mean_unpert 
        top_indices = np.argsort(np.abs(lfc_true))[-6:][::-1]
        
        plot_data = []
        for i, gene_idx in enumerate(top_indices):
            raw_ensembl_id = adata.var['ensembl_id'].iloc[gene_idx]
            display_name = get_readable_name(raw_ensembl_id, gene_map)
            
            vals_true = X_true[drug_indices, gene_idx]
            vals_pred = X_pred[drug_indices, gene_idx]
            vals_ctrl = X_unpert[drug_indices, gene_idx]
            
            df_gene = pd.DataFrame({
                'Expression': np.concatenate([vals_ctrl, vals_true, vals_pred]),
                'Condition': ['Control']*len(vals_ctrl) + ['True Perturbed']*len(vals_true) + ['Predicted']*len(vals_pred),
                'Gene': display_name
            })
            plot_data.append(df_gene)
            
        df_plot = pd.concat(plot_data)
        
        fig_3, ax_3 = plt.subplots(figsize=(10, 6))
        my_pal = {"Control": color_ctrl, "True Perturbed": color_true, "Predicted": color_pred}
        
        sns.boxplot(x='Gene', y='Expression', hue='Condition', data=df_plot, 
                    palette=my_pal, showfliers=False, linewidth=1.5, ax=ax_3)
        
        ax_3.set_title(f"Top Responsive Genes for {target_drug}", fontsize=15, pad=20)
        ax_3.set_ylabel("Log Expression", fontsize=12)
        ax_3.set_xlabel("Genes (Mapped)", fontsize=12)
        ax_3.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_3.tick_params(axis='x', rotation=30) 
        plt.tight_layout()
        save_plot(output_dir, "Figure3_Case_Study")

    print("\nGenerating Figure 4: Variance Comparison...")
    
    std_true = np.std(X_true, axis=0)
    std_pred = np.std(X_pred, axis=0)
    
    fig_4, ax_4 = plt.subplots(figsize=(7, 7))
    ax_4.scatter(std_true, std_pred, alpha=0.3, s=5, c='#6A5ACD', edgecolors='none')
    
    max_val = max(np.max(std_true), np.max(std_pred))
    ax_4.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Identity (x=y)')
    
    m, b = np.polyfit(std_true, std_pred, 1)
    ax_4.plot(std_true, m*std_true + b, 'k-', lw=1.5, label=f'Fit (slope={m:.2f})')
    
    ax_4.set_xlabel("Standard Deviation of True Expression")
    ax_4.set_ylabel("Standard Deviation of Predicted Expression")
    ax_4.set_title("Variance Shrinkage Analysis\n(Demonstrating Denoising Effect)")
    ax_4.legend()
    ax_4.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_plot(output_dir, "Figure4_Variance_Comparison")
    
    print(f"\n✅ All figures saved in both PNG and SVG formats.")
    print(f"Summary Metrics:")
    print(f"1. Delta Pearson r : {pearson_delta:.4f}")
    print(f"2. Variance Slope  : {m:.4f} (Ideal range: 0.8-0.9 for denoised prediction)")

file_path = "./test_pred.h5ad"
mapping_file_path = "./ensembl_to_symbol_mapping.txt"

generate_paper_figures(file_path, mapping_file_path)