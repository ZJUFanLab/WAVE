# load_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import os
import pandas as pd
from utils import morgan_fp

class SingleDrugDataset(Dataset):
    def __init__(self, manifest_path: str, config, mode: str = 'train'):
        self.cfg = config
        self.mode = mode
        
        self.manifest = pd.read_csv(manifest_path)
        
        target_dose = getattr(self.cfg, 'target_dose', None)
        if target_dose is not None:
            self.manifest = self.manifest[self.manifest['dose'] == target_dose].reset_index(drop=True)
            if mode == 'test':
                logging.info(f"[{mode}] Applied dose filter: {target_dose}. Count: {len(self.manifest)}")
        else:
            if mode == 'train' and 'dose' in self.manifest.columns:
                 most_common_dose = self.manifest['dose'].mode()[0]
                 self.manifest = self.manifest[self.manifest['dose'] == most_common_dose].reset_index(drop=True)

        drug_mapping_df = pd.read_csv(self.cfg.drug_mapping_path)
        drug_to_smiles_dict = dict(zip(drug_mapping_df['drug'], drug_mapping_df['mapped_smiles']))
        unique_drugs = self.manifest['drug_name'].unique()
        self.drug_fp_map = {}
        for drug in unique_drugs:
            smiles = drug_to_smiles_dict.get(drug, "")
            self.drug_fp_map[drug] = self._precompute_fp(smiles)

    def _precompute_fp(self, smiles):
        if not isinstance(smiles, str): smiles = ""
        fp_numpy = morgan_fp(smiles, n_bits=self.cfg.n_bits)
        return torch.tensor(fp_numpy, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        drug_name = row['drug_name']
        
        c_filename = os.path.basename(row['control_vector_path'])
        p_filename = os.path.basename(row['perturbation_vector_path'])
        
        c_path = os.path.join(self.cfg.vector_dir, c_filename)
        p_path = os.path.join(self.cfg.vector_dir, p_filename)
        
        try:
            unpert_mat = np.load(c_path) 
            pert_mat = np.load(p_path)
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self))

        n_ctrl = unpert_mat.shape[0]
        n_pert = pert_mat.shape[0]
        
        if self.mode in ['train']:
            replace = n_ctrl < n_pert
            ctrl_indices = np.random.choice(n_ctrl, size=n_pert, replace=replace)
        else:
            rng = np.random.RandomState(idx) 
            replace = n_ctrl < n_pert
            ctrl_indices = rng.choice(n_ctrl, size=n_pert, replace=replace)
            
        unpert_sample = unpert_mat[ctrl_indices, :]
        pert_sample = pert_mat

        unpert_tensor = torch.tensor(unpert_sample, dtype=torch.float32)
        pert_tensor = torch.tensor(pert_sample, dtype=torch.float32)
        
        drug_fp = self.drug_fp_map.get(drug_name, torch.zeros(self.cfg.n_bits, dtype=torch.float32))

        pert_h5_path = row['perturbation_h5ad_path']

        return {
            'unpert_expr': unpert_tensor,
            'drug_fp': drug_fp,
            'pert_expr': pert_tensor,
            'pert_h5_path': pert_h5_path
        }
    
    def __len__(self) -> int:
        return len(self.manifest)
    
def single_cell_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    unpert_list = []
    pert_list = []
    drug_list = []
    
    for item in batch:
        n_cells = item['unpert_expr'].shape[0]
        
        unpert_list.append(item['unpert_expr'])
        pert_list.append(item['pert_expr'])
        
        d_fp = item['drug_fp'].unsqueeze(0).expand(n_cells, -1)
        drug_list.append(d_fp)

    return {
        'unpert_expr': torch.cat(unpert_list, dim=0),
        'pert_expr': torch.cat(pert_list, dim=0),
        'drug_fp': torch.cat(drug_list, dim=0)
    }