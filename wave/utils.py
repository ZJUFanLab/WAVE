import torch
import numpy as np
import random
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def morgan_fp(
    smiles: str, 
    radius: int = 2, 
    n_bits: int = 2048
) -> np.ndarray:
    """
    Convert a SMILES string to Morgan fingerprint using RDKit's recommended MorganGenerator.

    Parameters:
    - smiles (str): SMILES string of the compound
    - radius (int): Radius for Morgan fingerprint (default: 2)
    - n_bits (int): Length of fingerprint bit vector (default: 2048)

    Returns:
    - np.ndarray: 1D binary fingerprint array (dtype=uint8)
                  Returns zero vector if SMILES is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return np.zeros(n_bits, dtype=np.uint8)
    
    try:
        
        generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = generator.GetFingerprint(mol)
        return np.array(fp, dtype=np.uint8)
    
    except AttributeError:
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.uint8)
