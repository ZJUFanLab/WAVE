python /path/to/wave/split_dataset.py --input /path/to/level3_cp_ctrl.h5ad --method random_split --output_dir .
cd Fold1
python /path/to/wave/train.py --outdir . --train_dataset train.h5ad --val_dataset val.h5ad --test_dataset test.h5ad