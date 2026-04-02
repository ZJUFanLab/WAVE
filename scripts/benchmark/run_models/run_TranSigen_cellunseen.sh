python train_TranSiGen_full_data.py \
  --data_path train_val_test.h5 \
  --molecule_path /path/to/KPGT_emb2501.pickle \
  --molecule_feature KPGT \
  --initialization_model initial \
  --split_data_type cells_split \
  --n_epochs 50 \
  --n_latent 100 \
  --molecule_feature_embed_dim 400 \
  --batch_size 16 \
  --learning_rate 1e-3 \
  --beta 0.1 \
  --dropout 0.1 \
  --weight_decay 1e-5 \
  --train_flag True \
  --eval_metric True \
  --predict_profile True \
  --seed 1 \
  --outdir .

