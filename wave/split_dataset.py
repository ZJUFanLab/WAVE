import os
import numpy as np
import scanpy as sc
from sklearn.model_selection import train_test_split, KFold
import shutil


def split_dataset(pert_ctrl, method="random_split", test_size=0.2, n_splits=5, random_state=1, output_dir="output_split"):
    """
    Split dataset based on the specified method: 'random_split', 'cell_split', or 'smiles_split'.

    Parameters:
        pert_ctrl (AnnData): Input dataset in AnnData format.
        method (str): Splitting method. Options: 'random_split', 'cell_split', 'smiles_split'.
        test_size (float): Proportion of the test set.
        n_splits (int): Number of folds for cross-validation.
        random_state (int): Random seed for reproducibility.
        output_dir (str): Directory to save split datasets.
    """
    os.makedirs(output_dir, exist_ok=True)

    if method == "random_split":
        print("Splitting data: random sample-based split")
        train_data, test_data = train_test_split(pert_ctrl, test_size=test_size, random_state=random_state)

    elif method == "cell_split":
        print("Splitting data: cell-based split")
        unique_cells = pert_ctrl.obs['cell'].unique()
        np.random.seed(random_state)
        test_cells = np.random.choice(unique_cells, int(len(unique_cells) * test_size), replace=False)
        train_cells = [cell for cell in unique_cells if cell not in test_cells]

        train_data = pert_ctrl[pert_ctrl.obs['cell'].isin(train_cells)].copy()
        test_data = pert_ctrl[pert_ctrl.obs['cell'].isin(test_cells)].copy()

    elif method == "smiles_split":
        print("Splitting data: SMILES-based split")
        unique_smiles = pert_ctrl.obs['smiles'].unique()
        np.random.seed(random_state)
        test_smiles = np.random.choice(unique_smiles, int(len(unique_smiles) * test_size), replace=False)
        train_smiles = [smi for smi in unique_smiles if smi not in test_smiles]

        train_data = pert_ctrl[pert_ctrl.obs['smiles'].isin(train_smiles)].copy()
        test_data = pert_ctrl[pert_ctrl.obs['smiles'].isin(test_smiles)].copy()

    else:
        raise ValueError(f"Invalid splitting method: {method}. Choose from 'random_split', 'cell_split', 'smiles_split'.")

    # Save test data
    test_data_path = os.path.join(output_dir, "test.h5ad")
    test_data.write_h5ad(test_data_path)

    # 5-fold cross-validation for training data with separate SMILES or cell lines in each fold
    if method in ["cell_split", "smiles_split"]:
        unique_groups = train_data.obs['cell'].unique() if method == "cell_split" else train_data.obs['smiles'].unique()
        np.random.seed(random_state)
        np.random.shuffle(unique_groups)

        folds = np.array_split(unique_groups, n_splits)

        for fold, val_groups in enumerate(folds):
            fold_dir = os.path.join(output_dir, f"Fold{fold + 1}")
            os.makedirs(fold_dir, exist_ok=True)

            train_groups = [g for g in unique_groups if g not in val_groups]
            train_fold_data = train_data[train_data.obs['cell'].isin(train_groups)] if method == "cell_split" else train_data[train_data.obs['smiles'].isin(train_groups)]
            val_fold_data = train_data[train_data.obs['cell'].isin(val_groups)] if method == "cell_split" else train_data[train_data.obs['smiles'].isin(val_groups)]

            train_fold_data.write_h5ad(os.path.join(fold_dir, "train.h5ad"))
            val_fold_data.write_h5ad(os.path.join(fold_dir, "val.h5ad"))
            shutil.copy(test_data_path, os.path.join(fold_dir, "test.h5ad"))

    else:  # For random_split, use traditional k-fold split
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        train_indices = np.arange(train_data.shape[0])

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_indices)):
            fold_dir = os.path.join(output_dir, f"Fold{fold + 1}")
            os.makedirs(fold_dir, exist_ok=True)

            train_fold_data = train_data[train_idx, :].copy()
            val_fold_data = train_data[val_idx, :].copy()

            train_fold_data.write_h5ad(os.path.join(fold_dir, "train.h5ad"))
            val_fold_data.write_h5ad(os.path.join(fold_dir, "val.h5ad"))
            shutil.copy(test_data_path, os.path.join(fold_dir, "test.h5ad"))

    print(f"Data split completed using method: {method}. Results saved in {output_dir}")


# Example usage with command-line arguments
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split dataset into train, validation, and test sets.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input AnnData file (e.g., level3_cp_ctrl.h5ad).")
    parser.add_argument("--method", type=str, choices=["random_split", "cell_split", "smiles_split"], required=True,
                        help="Splitting method: 'random_split', 'cell_split', or 'smiles_split'.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the test set (default: 0.2).")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for cross-validation (default: 5).")
    parser.add_argument("--random_state", type=int, default=1, help="Random seed for reproducibility (default: 1).")
    parser.add_argument("--output_dir", type=str, default="output_split", help="Directory to save the split datasets.")

    args = parser.parse_args()

    # Load input AnnData file
    pert_ctrl = sc.read_h5ad(args.input)

    # Call the split function
    split_dataset(
        pert_ctrl=pert_ctrl,
        method=args.method,
        test_size=args.test_size,
        n_splits=args.n_splits,
        random_state=args.random_state,
        output_dir=args.output_dir
    )

