import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import logging
import argparse
from load_dataset import GeneDrugDataset
from utils import seed_everything
from model import WAVE
import sys
import os
import scanpy as sc
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

for path in [current_dir, parent_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)


def loss_fct(pred, target, mu, logvar, alpha=1.0, beta=0.01):
    mse_loss = F.mse_loss(pred, target)

    # Pearson r loss
    pred_mean = pred - pred.mean(dim=1, keepdim=True)
    target_mean = target - target.mean(dim=1, keepdim=True)
    numerator = torch.sum(pred_mean * target_mean, dim=1)
    denominator = torch.sqrt(torch.sum(pred_mean ** 2, dim=1) * torch.sum(target_mean ** 2, dim=1))
    pearson_loss = 1 - torch.mean(numerator / denominator)

    # KL loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)

    return mse_loss + alpha * pearson_loss + beta * kld_loss



def train_model(model, train_loader, val_loader, optimizer, device, epochs, outdir, logger):
    model.to(device)
    best_pearson_delta = -float('inf')  # Record the best ΔGene Pearson correlation
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Train
        for batch in train_loader:
            unpert_expr = batch['unpert_expr'].to(device)
            drug_fp = batch['drug_fp'].to(device)
            pert_expr = batch['pert_expr'].to(device)

            optimizer.zero_grad()
            pred, mu, logvar, _ = model(unpert_expr, drug_fp)
            loss = loss_fct(pred, pert_expr, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validate
        with torch.no_grad():
            all_preds, all_truths, all_unpert_expr = [], [], []
            
            for batch in val_loader:
                unpert_expr = batch['unpert_expr'].to(device)  # baseline gene expression
                drug_fp = batch['drug_fp'].to(device)
                pert_expr = batch['pert_expr'].to(device)  # true perturbated gene expression
                
                pred, mu, logvar, _ = model(unpert_expr, drug_fp)  # prediction
        
                # save data to list
                all_preds.append(pred.cpu().numpy())
                all_truths.append(pert_expr.cpu().numpy())
                all_unpert_expr.append(unpert_expr.cpu().numpy())
        
            # covert list to numpy.array
            all_preds = np.concatenate(all_preds, axis=0)
            all_truths = np.concatenate(all_truths, axis=0)
            all_unpert_expr = np.concatenate(all_unpert_expr, axis=0)
        

            pearson = np.corrcoef(all_preds.flatten(), all_truths.flatten())[0, 1]
        

            delta_pred = all_preds - all_unpert_expr  
            delta_truth = all_truths - all_unpert_expr  
            pearson_delta = np.corrcoef(delta_pred.flatten(), delta_truth.flatten())[0, 1]
        

            ss_res = np.sum((all_truths - all_preds) ** 2)
            ss_tot = np.sum((all_truths - np.mean(all_truths)) ** 2)
            r2 = 1 - ss_res / ss_tot
        

            log_message = (
                f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Pearson: {pearson:.4f} "
                f"| R²: {r2:.4f} | ΔGene Pearson: {pearson_delta:.4f}"
            )
            logger.info(log_message)


            if pearson_delta > best_pearson_delta:
                best_pearson_delta = pearson_delta
                torch.save(model.state_dict(), "best_model.pth")
                logger.info(f"Get a better model (ΔGene Pearson: {pearson_delta:.4f})，save as best_model.pth")


def test_model(model_path, test_dataset, test_loader, device, outdir, logger):
    test_adata = sc.read_h5ad(test_dataset)
    test_adata.uns['model_path'] = model_path
    model_state_dict = torch.load(model_path)
    model = WAVE().to(device)
    model.load_state_dict(model_state_dict)
    
    
    with torch.no_grad():
        all_preds, all_truths, all_unpert_expr = [], [], []
        
        for batch in test_loader:
            unpert_expr = batch['unpert_expr'].to(device)  # baseline gene expression
            drug_fp = batch['drug_fp'].to(device)
            pert_expr = batch['pert_expr'].to(device)  # true perturbated gene expression
            
            pred, mu, logvar, _ = model(unpert_expr, drug_fp)  # prediction
        
            # save data to list
            all_preds.append(pred.cpu().numpy())
            all_truths.append(pert_expr.cpu().numpy())
            all_unpert_expr.append(unpert_expr.cpu().numpy())
        
        # covert list to numpy.array
        all_preds = np.concatenate(all_preds, axis=0)
        all_truths = np.concatenate(all_truths, axis=0)
        all_unpert_expr = np.concatenate(all_unpert_expr, axis=0)
        
        
        pearson = np.corrcoef(all_preds.flatten(), all_truths.flatten())[0, 1]
        
        
        delta_pred = all_preds - all_unpert_expr  
        delta_truth = all_truths - all_unpert_expr  
        pearson_delta = np.corrcoef(delta_pred.flatten(), delta_truth.flatten())[0, 1]
        
        
        ss_res = np.sum((all_truths - all_preds) ** 2)
        ss_tot = np.sum((all_truths - np.mean(all_truths)) ** 2)
        r2 = 1 - ss_res / ss_tot
            
        
        sample_pearson = np.array([
            np.corrcoef(all_preds[i], all_truths[i])[0, 1] for i in range(all_preds.shape[0])
        ])
        
        
        sample_pearson_delta = np.array([
            np.corrcoef(delta_pred[i], delta_truth[i])[0, 1] for i in range(delta_pred.shape[0])
        ])
        
        
        sample_r2 = np.array([
            1 - np.sum((all_truths[i] - all_preds[i]) ** 2) / np.sum((all_truths[i] - np.mean(all_truths[i])) ** 2)
            for i in range(all_truths.shape[0])
        ])
        
        
        log_message = (
            f"Test Results | Pearson: {pearson:.4f} | R²: {r2:.4f} | ΔGene Pearson: {pearson_delta:.4f}"
        )
        

        logger.info(log_message)

        test_adata.layers['pred'] = all_preds
        test_adata.uns['metrics_df'] = pd.DataFrame({'r2':sample_r2, 'pearson':sample_pearson, 'delta_x_pearson':sample_pearson_delta})
        
        test_adata.write_h5ad(outdir+"/test_pred.h5ad")
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WAVE.",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  

    parser.add_argument('--outdir', type=str, help="Output directory for saving results")
    parser.add_argument('--seed', type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument('--train_dataset', type=str, help="Path to training dataset(.h5ad)")
    parser.add_argument('--val_dataset', type=str, help="Path to validation dataset(.h5ad)")
    parser.add_argument('--test_dataset', type=str, help="Path to test dataset(.h5ad)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--epochs', type=int, default=300, help="Maximum number of training epochs")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use for computation ('cpu' or 'cuda')")

    args = parser.parse_args()

    outdir = args.outdir
    seed = args.seed
    train_dataset = args.train_dataset
    val_dataset = args.val_dataset
    test_dataset = args.test_dataset
    batch_size = args.batch_size
    epochs = args.epochs
    device = args.device

    seed_everything(seed)

    train_set = GeneDrugDataset(train_dataset)
    val_set = GeneDrugDataset(val_dataset)
    test_set = GeneDrugDataset(test_dataset)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=0, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0, shuffle=False)
    
    model = WAVE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    logging.basicConfig(
                        level=logging.INFO,
                        format="%(asctime)s | %(message)s",
                        handlers=[
                            logging.StreamHandler(),
                            logging.FileHandler(outdir + "/training_log.txt", mode='w')
                                ]
    )
    logger = logging.getLogger()
    
    
    # train model
    logger.info("Start training.")
    
    model.train()
    train_model(model, train_loader, val_loader, optimizer, device, epochs, outdir, logger)
    logger.info("Completed training.")
        
    logger.info("Start test.")
    best_model_path = outdir+"/best_model.pth"
    test_model(best_model_path, test_dataset, test_loader, device, outdir, logger)
    logger.info("Completed test.")
