import torch
import numpy as np
from torch.utils.data import Dataset
import scanpy as sc

class GeneDrugDataset(Dataset):
    def __init__(self, adata_path):
        adata = sc.read_h5ad(adata_path)
        self.unpert_expr = adata.layers['unpert_expr']
        self.pert_expr = adata.X
        self.smiles = adata.obs['smiles'].tolist()

        from rdkit import Chem
        from rdkit.Chem import AllChem
        self.drug_fps = []
        for smi in self.smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                self.drug_fps.append(np.array(fp))
            else:
                self.drug_fps.append(np.zeros(2048))

    def __len__(self):
        return len(self.unpert_expr)

    def __getitem__(self, idx):
        return {
            'unpert_expr': torch.tensor(self.unpert_expr[idx], dtype=torch.float32),
            'drug_fp': torch.tensor(self.drug_fps[idx], dtype=torch.float32),
            'pert_expr': torch.tensor(self.pert_expr[idx], dtype=torch.float32)
        }
