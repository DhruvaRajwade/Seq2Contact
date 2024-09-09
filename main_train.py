"""
Description: This script is used to train the protein-NA binding model using the ESM and GPN models. The script uses argparse to take in the following arguments:

              --seed: Random seed
              --data_mode: Mode for loading and processing data
              --split_mode: Mode for splitting data
              --lower_threshold: Lower threshold for data processing
              --na_upper_threshold: Upper threshold for NA length
              --protein_upper_threshold: Upper threshold for protein length
              --train_ratio: Ratio of training data
              --train_batch_size: Batch size for training
              --eval_batch_size: Batch size for evaluation
              --d_k: Dimension of key
              --num_epochs: Number of epochs to train
              --lr: Learning rate
              --loss_fn: Loss function, Options are: bce_loss, weighted_bce_loss
              --device: Device to train on
              --log_suffix: Suffix for log directory
              --log_dir: Directory for logs
              --save_freq: Frequency of saving model checkpoints
              --save_path: Path to save model checkpoints
              --use_tqdm: Use tqdm for progress bar
              --resume_from_checkpoint: Resume training from checkpoint
              --checkpoint_path: Path to checkpoint to resume from
              --finetune_protein_model: Finetune protein model or Freeze
              --finetune_na_model: Finetune NA model or Freeze
              --protein_num_layers_to_unfreeze: Number of layers to unfreeze in the protein model
              --dna_num_layers_to_unfreeze: Number of layers to unfreeze in the protein model

"""
import argparse
from utils.data import *
from utils.loss import *
from utils.model import *
from utils.train import *
from utils.plots import *
from utils.util import *
from utils.finetune import *
from torch.utils.data import DataLoader
from torch.utils.data import Subset as Subset


def main(args):
    set_seed(args.seed)

    data = load_and_process_data(mode=args.data_mode, lower_threshold=args.lower_threshold, na_upper_threshold=args.na_upper_threshold, protein_upper_threshold=args.protein_upper_threshold)


    if 'dna' in args.data_mode:
        split_mode = 'dna'

    elif 'rna' or 'rev_transcript' in args.data_mode:
        print('Support for RNA and Reverse Transcription coming soon') 
        return None
    
    if args.split_mode == 'sequence_similarity':
        train_dset, eval_dset = sequence_similarity_split(data, mode = split_mode)
    
    else:
        raise ValueError(f"Split mode {args.split_mode} not recognized")

        

    train_dataloader = DataLoader(train_dset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_sequences)
    eval_dataloader = DataLoader(eval_dset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_sequences)

    
    """
    Available ESM Models are:

        "esm2_t48_15B_UR50D": 5120
        "esm2_t36_3B_UR50D": 2560
        "esm2_t33_650M_UR50D": 1280
        "esm2_t30_150M_UR50D": 640
        "esm2_t12_35M_UR50D": 480
        "esm2_t6_8M_UR50D": 320

    """

    protein_model = ESMModel(model_name='esm2_t6_8M_UR50D',unfreeze_last_n_layers=args.protein_num_layers_to_unfreeze)
    d_model_q = protein_model.embedding_dim



    if 'dna' in args.data_mode:
    #if 'dna' in args.data_mode:
        na_model = GPNModel(unfreeze_last_n_layers=args.dna_num_layers_to_unfreeze)
        d_model_kv = 512

    else: 
        print('Support for RNA and Reverse Transcription coming soon')

    binding_model = CustomCrossAttention(d_model_q, d_model_kv, args.d_k).to(args.device)

    
    model, _, _, _ = train_model(protein_model, na_model, binding_model, args.lr, train_dataloader, eval_dataloader, args.loss_fn, args.num_epochs, device=args.device, log_suffix=args.log_suffix, log_dir=args.log_dir, save_freq=args.save_freq, save_path=args.save_path, use_tqdm=args.use_tqdm, resume_from_checkpoint=args.resume_from_checkpoint, checkpoint_path=args.checkpoint_path, train_protein_model=args.finetune_protein_model, train_na_model=args.finetune_na_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Protein-NA Binding Model')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_mode', type=str, default='dna', help='Mode for loading and processing data')
    parser.add_argument('--split_mode', type=str, default='sequence_similarity', help='Mode for splitting data')
    parser.add_argument('--lower_threshold', type=int, default=10, help='Lower threshold for data processing')
    parser.add_argument('--na_upper_threshold', type=int, default=100, help='Upper threshold for NA length')
    parser.add_argument('--protein_upper_threshold', type=int, default=1000, help='Upper threshold for protein length')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of training data')
    parser.add_argument('--train_batch_size', type=int, default=12, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--d_k', type=int, default=32, help='Dimension of key')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--loss_fn', type=str, default='weighted_bce_loss', help='Loss function, Options are: bce_loss, weighted_bce_loss')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train on')
    parser.add_argument('--log_suffix', type=str, default='lmao_ded', help='Suffix for log directory')
    parser.add_argument('--log_dir', type=str, default='./runs', help='Directory for logs')
    parser.add_argument('--save_freq', type=int, default=50, help='Frequency of saving model checkpoints')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='Path to save model checkpoints')
    parser.add_argument('--use_tqdm', type=bool, default=False, help='Use tqdm for progress bar')
    parser.add_argument('--resume_from_checkpoint', type=bool, default=False, help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints', help='Path to checkpoint to resume from')
    parser.add_argument('--finetune_protein_model',default=False,action='store_true', help='Finetune protein model or Freeze')
    parser.add_argument('--finetune_na_model', default=False,action='store_true', help='Finetune NA model or Freeze')
    parser.add_argument('--protein_num_layers_to_unfreeze', type=int, default=1, help='Number of layers to unfreeze in the protein model')
    parser.add_argument('--dna_num_layers_to_unfreeze', type=int, default=1, help='Number of layers to unfreeze in the protein model')

    args = parser.parse_args()
    main(args)