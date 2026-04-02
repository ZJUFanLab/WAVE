# config.py
import os
import torch

class Config:
    def __init__(self, fold_id="fold_0"):
        self.base_dir = "../tahoe"
        self.vector_dir = os.path.join(self.base_dir, "cv_folds_processed_CELL/per_cell_normalized_vectors")
        
        self.fold_id = fold_id
        self.fold_dir = os.path.join(self.base_dir, f"cv_folds_processed_CELL/{self.fold_id}")

        self.train_path = os.path.join(self.fold_dir, "train_manifest.csv")
        self.val_path   = os.path.join(self.fold_dir, "val_manifest.csv")
        self.test_path  = os.path.join(self.fold_dir, "test_manifest.csv")

        self.drug_mapping_path = os.path.join(self.base_dir, "final_drug_mapping.csv")
        self.gene_list_path = os.path.join(self.base_dir, "tahoe_gene_hvg_list.csv")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(current_dir, "model_weights")
        os.makedirs(self.output_dir, exist_ok=True)

        self.model_save_path = os.path.join(self.output_dir, f"best_model_{fold_id}.pth")
        self.result_save_path = os.path.join(self.output_dir, f"test_predictions_{fold_id}.h5ad")

        self.seed = 42
        self.device = "cuda"
        self.epochs = 300
        self.batch_size = 128
        self.lr = 0.0002
        self.weight_decay = 7e-05
        self.dropout_rate = 0.05
        
        self.num_workers = 4
        self.patience = 25
        self.min_delta = 1e-6 
        self.input_dim = 2000
        self.drug_input_dim = 2048
        self.latent_dim = 128
        
        self.vae_hidden_dims = [512, 256]          
        
        self.drug_hidden_dims = [512] 
        self.drug_output_dim = 512
        
        self.fusion_hidden_dims = [512, 256]       
        self.output_dim = 2000
        self.huber_delta = 1.0
        
        self.alpha_pearson = 12
        
        self.beta_kl = 1.5e-4
        
        self.kl_anneal_epochs = 10

        self.radius = 2
        self.n_bits = 2048

    def to_dict(self):
        return self.__dict__