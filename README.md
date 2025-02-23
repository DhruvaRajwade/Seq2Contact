# Understanding Protein-DNA Interactions by Paying Attention to Protein and Genomics Foundation Models -- Code

This is the code corresponding to the experiments conducted for the work "Understanding Protein-DNA Interactions by Paying Attention to Genomic Foundation Models" where we use the ESM2 and GPN foundation models coupled with a Cross-Attention module to learn Protein-DNA contacts at single amino acid and single nucleotide resolution. 

## Requirements

Create an environment (users will be prompted to select `Conda` or `Virtualenv` )and install all dependencies by running the `create_env.sh` script. To do so, follow the below steps:
1. `chmod +x ./create_env.sh`
2. `./create_env.sh`
3. You will be prompted to choose Conda or Virtualenv, and an environment name, and the script should install everything else.
  

## Datasets

We extracted Protein-Nucleic acid complex data from the [NAKB](https://www.nakb.org/) database. Our processed dataset (pickle) can be downloaded [here](https://drive.google.com/file/d/1nHrB8WiYuLdJpiP-L3lls9cKp0ipWXi1/view?usp=sharing). We provide a helper Jupyter notebook [(`notebooks/read_and_save_data.ipynb`)](notebooks/read_and_save_data.ipynb) to create your own dataset and process it to be compatible with our pipeline. 

## Files

### ├── notebooks/
- `af3_read.ipynb`: Notebook for reading and analyzing AlphaFold 3 data.
- `read_and_save_data.ipynb`: Notebook to download 3D complex pdbs from NAKB, process to obtain sequences and contact_maps and save them into a pickled list. 
- `inference.ipynb`: Notebook to run inference on trained models and analyze predictions.

### ├── utils/
  - `loss.py` file: Implements the BCE and Weighted BCE loss with support for attention masking
  - `data.py` file: Code for all data handling operations including creating custom dataloaders for handling variable length `(sequence,contact_map)` pairs. 
  - `model.py` file: Implements the Cross-Attention module for contact-map prediction
  - `finetune.py` file: Custom wrappers for the ESM2 and GPN models allowing a user to select the number of layers to unfreeze and the model size (for the ESM2 family)
  - `util.py` file: Helper functions to set a seed for reproducibility, and to compute distance and contact maps from 3D co-ordinates
  - `train.py` file: Code to train the whole pipeline, and to evaluate it. Saves 10 randomly sampled attention map overlays (GT vs pred), all model gradients, loss values and metrics for training and evaluation sets. 
  - `plots.py` file: Code for plotting metrics as well as contact maps. 
    
### └── main.py   
  - `main_train.py` file: Wrapper file for an end to end run.

### └── run_training.sh
  - `run_training.sh`: Script to run the `main.py` file in background using `nohup`

### Misc:
  -  `checkpoints/` directory (user needs to create this): Model checkpoints will be saved here by default.
  -  `runs/` directory (user needs to create this): Tensorboard logs saved will be saved here by default.
  -  `logs/` directory (user needs to create this): `stdout` and `stderr` files for any run will be merged and saved here.
  - `data/dna_protein_dataset.pkl`: Our processed DNA-Protein complex dataset, available at [this link]((https://drive.google.com/file/d/1nHrB8WiYuLdJpiP-L3lls9cKp0ipWXi1/view?usp=sharing)). 
  - `data/train_test_clusters.pkl`: Train test split indices based on DNA sequence-similarity based clustering (using `mmseq2`) .
  - `data/AF3_pdbs.txt`: Subset of AlphaFold3 test set which we used for our baseline comparison. 


## Usage

### Training using default hyperparemeters (Save your dataset as dna_xxx.pkl in the Data directory)

To train a model using default hyperparameters and save checkpoints, tensorboard logs and evaluate, use: (Note: The CUBLAS part is for reproducibility)

`CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 -u main_train.py --log_suffix 'my_suffix'  --lr 0.00001 --num_epochs 1000 --train_batch_size 8 --eval_batch_size 8 --finetune_protein_model --finetune_na_model --dna_num_layers_to_unfreeze 1 --protein_num_layers_to_unfreeze 1`

To train a model in the background using `nohup`, set your environment path in the `run_training.sh` file, verify the argparse agruments you want for the run and use:

`nohup ./run_training.sh > logs/my_suffix.out 2>&1 & `



### Further Documentation

See the code documentation for more details. `main_train.py` can be called with the
`-h` option for additional help.

### Hyperparameters

Relevant Hyperparameters for setting up training:
 - `d_k`: Dimension of the key vector in our cross-attention module
 - `protein_num_layers_to_unfreeze` : These many last layers of the Protein model will be kept trainable
 - `dna_num_layers_to_unfreeze` : These many last layers of the DNA model will be kept trainable
 - `log_suffix`: This will be appended to the path of your model's checkpoints and runs 

### Citation
Please consider citing our work using the following bibtex entry,
```bibtex
@inproceedings{seq2contact,
  title     = {Understanding Protein-DNA Interactions by Paying Attention to Protein and Genomics Foundation Models},
  author    = {Rajwade, Dhruva and Wang, Erica, and Satpathy, Aryan and Brace, Alexander and Guo, Hongyu and Ramanathan, Arvind and Liu, Shengchao and Anandkumar, Anima},
  booktitle = {Foundation Models for Science Workshop, NeurIPS},
  year      = {2024},
  url       = {https://openreview.net/pdf?id=NifnqNdpaf}
}
```
