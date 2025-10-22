##  Model Architecture

WAVE consists of three main components:

* **GeneVAE**
  Encodes baseline (control) gene expression profiles into a latent space using a variational autoencoder.

* **DrugNN**
  Maps drug molecular fingerprints (e.g., ECFP4) into a low-dimensional embedding vector.

* **GeneDrugFusion**
  Concatenates gene latent vectors and drug embeddings, then predicts the expression **delta** (i.e., perturbation effect), which is added to the VAE-reconstructed baseline to generate the final perturbed profile.



---

## Installation & Dependencies

WAVE is implemented in Python 3.8+. You can install the required dependencies with:

```bash
pip install torch
pip install scanpy anndata pandas numpy scipy
```

> If running on GPU, ensure CUDA is properly configured.

---

## Input Data Format

WAVE expects an `.h5ad` file as input (AnnData format) with the following:

* `.X`: baseline (control) gene expression matrix (cells × genes)
* `.layers['ctrl']`: control expression (optional if `.X` used)
* `.layers['pert']`: drug-treated expression (ground truth)
* `.obsm['drug_fp']`: drug molecular fingerprints (e.g., ECFP vectors, shape: cells × 2048)

---

## Dataset Splitting

Use the provided utility script to split the dataset into training, validation, and test sets:

```bash
python /path/to/wave/split_dataset.py \
  --input /path/to/level3_cp_ctrl.h5ad \
  --method random_split \
  --output_dir .
```

This will create 5 fold directories containing:

* `train.h5ad`
* `val.h5ad`
* `test.h5ad`

---

## Training & Evaluation

### Step 1: Navigate to the data fold directory

```bash
cd Fold1
```

### Step 2: Train the model

```bash
python /path/to/wave/train.py \
  --outdir . \
  --train_dataset train.h5ad \
  --val_dataset val.h5ad \
  --test_dataset test.h5ad
```

During training, the model will log the following metrics:

* `Loss`: total objective
* `Pearson`: correlation between predicted and actual expression
* `Delta Pearson`: correlation of predicted vs actual **expression change**
* `R²`: coefficient of determination (goodness of fit)

The best model (lowest validation loss) will be saved as `best_model.pth`.

---

## Output

After training, the model outputs predictions to:

```
test_pred.h5ad
```

This includes:

* `.layers['pred']`: predicted perturbed gene expression
* `.uns['metrics_df']`: per-cell performance metrics (Pearson, R², Delta Pearson)

---

## File Structure

```
wave/
├── model.py              # Defines the model architecture
├── train.py              # Main training & evaluation script
├── utils.py              # Utilities (e.g., seed control)
├── load_dataset.py       # Custom PyTorch Dataset loader
├── split_dataset.py      # Dataset splitting tool
└── README.md             # Project documentation
```