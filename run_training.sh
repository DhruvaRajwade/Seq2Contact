#!/usr/bin/env bash
set -e
source "path/to/your/virtualenv/bin/activate"
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 -u main_train.py --log_suffix 'my_suffix' --data_mode 'dna' --lr 0.00001 --split_mode 'sequence_similarity' --finetune_protein_model --finetune_na_model --num_epochs 1000  --save_freq 300 --seed 42 --na_upper_threshold 200 --protein_upper_threshold 1000 --train_ratio 0.8 --train_batch_size 8 --eval_batch_size 8 --dna_num_layers_to_unfreeze 1 --protein_num_layers_to_unfreeze 1

