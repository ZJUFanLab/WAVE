import torch
import numpy as np
import random
import json
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
import json
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
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return np.zeros(n_bits, dtype=np.uint8)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    
    return np.array(fp, dtype=np.uint8)

def log_config(cfg, logger=None):
    config_dict = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    config_json = json.dumps(config_dict, indent=4, ensure_ascii=False, sort_keys=True, default=str)
    
    line_width = 86
    msg = "\n" + "="*line_width + "\n"
    msg += f"CONFIG FOR: {getattr(cfg, 'fold_id', 'Unknown')}".center(line_width) + "\n"
    msg += "-"*line_width + "\n"
    msg += config_json
    msg += "\n" + "="*line_width + "\n"
    
    if logger:
        logger.info(msg)
    else:
        print(msg)