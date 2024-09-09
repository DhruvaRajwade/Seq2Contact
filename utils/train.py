import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import datetime
import torch.nn.utils
import time
import os
from utils.data import *
from utils.loss import *
from utils.model import *
from utils.train import *
from utils.plots import *
from utils.util import *
from utils.finetune import *
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset as Subset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef,precision_recall_curve, auc #average_precision_score, roc_auc_score,


import os
import torch
import torch.optim as optim
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from tqdm import tqdm


def evaluate(prot_model,na_model, binding_model, dataloader, device='cuda'):
    predicted_attention_maps = []
    gt_attention_maps = []
    protein_lens = []
    rna_lens = []
    binding_model.eval()
    with torch.no_grad():
        for protein_seqs, na_seqs, contact_maps, _ in dataloader:
            protein_seqs = [('',i) for i in protein_seqs]

            protein_embeddings = prot_model(protein_seqs) ## List of tensors of shape (L, d_model)
            na_embeddings = na_model(na_seqs)
                ## List of tensors of shape (L, d_model)
            padded_protein_embeddings, padded_na_embeddings, padded_contact_maps, protein_lengths, rna_lengths = collate_embeddings(protein_embeddings, na_embeddings, contact_maps)
            padded_protein_embeddings, padded_na_embeddings = padded_protein_embeddings.to(device), padded_na_embeddings.to(device)
            mask = create_mask(protein_lengths, rna_lengths, padded_protein_embeddings.size(1), padded_na_embeddings.size(1)).unsqueeze(1).float().to(device)  
        
            _, predicted_attention = binding_model(padded_protein_embeddings, padded_na_embeddings, mask)
        
            predicted_attention_maps.append(predicted_attention.cpu())
            gt_attention_maps.append(padded_contact_maps.cpu())
            protein_lens.append(protein_lengths)
            rna_lens.append(rna_lengths)
            
    return predicted_attention_maps, gt_attention_maps, protein_lens, rna_lens

def get_batch_metrics(batch_gt, batch_pred, protein_lengths, na_lengths):

    net_f1,net_prauc,net_precision,net_recall,net_mcc= 0,0,0,0,0

    for i in range(batch_gt.shape[0]):
        target = batch_gt[i][:protein_lengths[i], :na_lengths[i]].flatten()
        prediction = batch_pred[i][:protein_lengths[i], :na_lengths[i]].flatten()
        net_f1 += f1_score(target, prediction > 0.5)
        precision, recall, _ = precision_recall_curve(target, prediction)
        prauc = auc(recall, precision)
        net_prauc += prauc
        net_mcc += matthews_corrcoef(target, prediction  > 0.5)
      
    return net_f1 / batch_gt.shape[0], net_prauc / batch_gt.shape[0], net_mcc / batch_gt.shape[0] #n


def train_model(protein_model, na_model, model, lr, train_dataloader, eval_dataloader, loss_fn, num_epochs, device='cuda', log_suffix='', log_dir='./runs', save_freq=1000, save_path='./checkpoints', use_tqdm=True, resume_from_checkpoint=False, checkpoint_path=None, train_protein_model=True, train_na_model=True):
    if log_suffix == '':
        log_suffix = datetime.now().strftime("%Y%m%d-%H%M%S")

    ## Make directory with log_suffix as name inside save_path
    os.makedirs(f'{save_path}/{log_suffix}', exist_ok=True)
    model = model.to(device)

    protein_model.verbose()
    print("")
    na_model.verbose()
    print("")
    
    # Freeze parameters of models that are not being trained
    if not train_protein_model:
        for param in protein_model.model.parameters():
            param.requires_grad = False
    if not train_na_model:
        for param in na_model.model.parameters():
            param.requires_grad = False

    param_groups = [{'params': model.parameters(), 'lr': lr}]
    
    if train_protein_model:
        param_groups.append({'params': protein_model.parameters(), 'lr': lr / 10})
        print("Protein Model is in train mode")
    
    else:
        pass

    if train_na_model:
        param_groups.append({'params': na_model.parameters(), 'lr': lr / 10})
        print("NA Model is in train mode")

    else:
        pass

    optimizer = optim.Adam(param_groups)

    
    start_epoch = 0

    if resume_from_checkpoint:
        model.to(device), protein_model.to(device), na_model.to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if train_protein_model:
            protein_model.model.load_state_dict(checkpoint['protein_model_state_dict'])
        if train_na_model:
            na_model.model.load_state_dict(checkpoint['na_model_state_dict'])

        a = checkpoint['optimizer_state_dict']

        fused = a['param_groups'][1]
        l1, l2 = fused['params'][:-8], fused['params'][-8:]
        a['param_groups'].append(fused.copy())
        a['param_groups'][1]['params'] = l1
        a['param_groups'][2]['params'] = l2
        a['param_groups'][0]['lr'] = 1e-6  
        optimizer.load_state_dict(a)
        start_epoch = checkpoint['epoch'] + 1
        log_suffix = checkpoint['log_suffix']
        print(f'Resuming training from epoch {start_epoch}')

    
    all_losses_train = []
    all_gradients = []
    all_losses_eval = []


    log_dir = f'{log_dir}/{log_suffix}'
    writer = SummaryWriter(log_dir=log_dir)
    subset_indices = np.random.choice(len(eval_dataloader.dataset), 10, replace=False)
    subset_dataset = Subset(eval_dataloader.dataset, subset_indices)
    print(f'Number of Samples in the Training_Dataset: {len(train_dataloader.dataset)}')
    print(f'Number of Samples in the Evaluation_Dataset: {len(eval_dataloader.dataset)}')
    subset_dataloader = torch.utils.data.DataLoader(subset_dataset, batch_size=1, shuffle=False, collate_fn=collate_sequences)
    
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()

        #gpu_usage_start = torch.cuda.memory_allocated(device)
       
        protein_model.train(train_protein_model)
        na_model.train(train_na_model)
        model.train()

        total_loss, total_loss_eval, total_gradients, total_gradients_protein, total_gradients_na, total_f1, total_prauc, total_precision, total_recall, total_mcc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if use_tqdm:
            progress_bar = tqdm(total=len(train_dataloader) + len(eval_dataloader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
        
        for protein_seqs, na_seqs, contact_maps, _ in train_dataloader:
            optimizer.zero_grad()

            protein_seqs = [('', i) for i in protein_seqs]
            protein_embeddings = protein_model(protein_seqs)  # List of tensors of shape (L, d_model)
            na_embeddings = na_model(na_seqs)  # List of tensors of shape (L, d_model)
       
            padded_protein_embeddings, padded_na_embeddings, padded_contact_maps, protein_lengths, rna_lengths = collate_embeddings(protein_embeddings, na_embeddings, contact_maps)

            padded_protein_embeddings, padded_na_embeddings = padded_protein_embeddings.to(device), padded_na_embeddings.to(device)
            mask = create_mask(protein_lengths, rna_lengths, padded_protein_embeddings.size(1), padded_na_embeddings.size(1)).unsqueeze(1).float().to(device)

            output, predicted_attention = model(padded_protein_embeddings, padded_na_embeddings, mask)

            if loss_fn == 'weighted_bce_loss':
                loss = weighted_bce_loss(predicted_attention, padded_contact_maps, mask, pos_weight=None)
            elif loss_fn == 'vanilla_bce_loss':
                loss = vanilla_bce_loss(predicted_attention, padded_contact_maps, mask)

            loss.backward()
            batch_f1, batch_prauc, batch_mcc = get_batch_metrics(padded_contact_maps.cpu().numpy(), predicted_attention.detach().cpu().numpy(), protein_lengths, rna_lengths)
            total_f1 += batch_f1
            total_prauc += batch_prauc
            total_mcc += batch_mcc

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            total_gradients += total_norm
            total_norm_protein = 0

            if train_protein_model:

                for p in protein_model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm_protein += param_norm.item() ** 2
                total_norm_protein = total_norm_protein ** 0.5

            else:
                pass

            total_gradients_protein += total_norm_protein

            total_norm_na = 0

            if train_na_model:
            
                for p in na_model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm_na += param_norm.item() ** 2
                total_norm_na = total_norm_na ** 0.5

            else:
                pass

            total_gradients_na += total_norm_na

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()

            total_loss += loss.item()

            if use_tqdm:
                progress_bar.update(1)
                progress_bar.set_postfix({'Train Loss': total_loss / len(train_dataloader), 'Eval Loss': total_loss_eval / len(eval_dataloader) if len(eval_dataloader) > 0 else 0})

        gpu_usage_end = torch.cuda.memory_allocated(device)
        gpu_usage = (gpu_usage_end) / (1024 * 1024)  # Convert to MB
        
        
        writer.add_scalar('Loss/Train', total_loss / len(train_dataloader), epoch)
        writer.add_scalar('Gradients/Binding_Model', total_gradients / len(train_dataloader), epoch)
        writer.add_scalar('Gradients/Protein_Model', total_gradients_protein / len(train_dataloader), epoch)
        writer.add_scalar('Gradients/NA_Model', total_gradients_na / len(train_dataloader), epoch)
        writer.add_scalar('Metrics/F1', total_f1 / len(train_dataloader), epoch)
        writer.add_scalar('Metrics/PR_AUC', total_prauc / len(train_dataloader), epoch)
        writer.add_scalar('Metrics/MCC', total_mcc / len(train_dataloader), epoch)
        #writer.add_scalar('GPU Usage (MB)', gpu_usage, epoch)
        all_gradients.append(total_gradients / len(train_dataloader))
        all_losses_train.append(total_loss / len(train_dataloader))

        total_f1_eval, total_prauc_eval, total_precision_eval, total_recall_eval, total_mcc_eval = 0, 0, 0, 0, 0
        model.eval()
        with torch.no_grad():
            for protein_seqs, na_seqs, contact_maps, _ in eval_dataloader:
                protein_seqs = [('', i) for i in protein_seqs]
                protein_embeddings = protein_model(protein_seqs)  # List of tensors of shape (L, d_model)
                na_embeddings = na_model(na_seqs)  # List of tensors of shape (L, d_model)
                padded_protein_embeddings, padded_na_embeddings, padded_contact_maps, protein_lengths, rna_lengths = collate_embeddings(protein_embeddings, na_embeddings, contact_maps)
                padded_protein_embeddings, padded_na_embeddings = padded_protein_embeddings.to(device), padded_na_embeddings.to(device)
                mask = create_mask(protein_lengths, rna_lengths, padded_protein_embeddings.size(1), padded_na_embeddings.size(1)).unsqueeze(1).float().to(device)

                output, predicted_attention = model(padded_protein_embeddings, padded_na_embeddings, mask)

                loss = weighted_bce_loss(predicted_attention, padded_contact_maps.squeeze(), mask, pos_weight=None)
                total_loss_eval += loss.item()
                batch_f1, batch_prauc, batch_mcc = get_batch_metrics(padded_contact_maps.cpu().numpy(), predicted_attention.detach().cpu().numpy(), protein_lengths, rna_lengths)

                total_f1_eval += batch_f1
                total_prauc_eval += batch_prauc
                total_mcc_eval += batch_mcc

                if use_tqdm:
                    progress_bar.update(1)
                    progress_bar.set_postfix({'Train Loss': total_loss / len(train_dataloader), 'Eval Loss': total_loss_eval / len(eval_dataloader)})

            writer.add_scalar('Loss/Eval', total_loss_eval / len(eval_dataloader), epoch)
            writer.add_scalar('Metrics/F1/Eval', total_f1_eval / len(eval_dataloader), epoch)
            writer.add_scalar('Metrics/PR_AUC/Eval', total_prauc_eval / len(eval_dataloader), epoch)
            writer.add_scalar('Metrics/MCC/Eval', total_mcc_eval / len(eval_dataloader), epoch)

        if use_tqdm:
            progress_bar.close()

        all_losses_eval.append(total_loss_eval / len(eval_dataloader))

        if not use_tqdm:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss / len(train_dataloader):.4f}, Eval Loss: {total_loss_eval / len(eval_dataloader):.4f}, Time: {time.time() - start_time:.2f} sec')
        
        counter = 0
        if epoch % 5 == 0:
            initial_preds, gt, prot_lens, rna_lens = evaluate(protein_model, na_model, model, subset_dataloader)
            initial_preds = [torch.where(p > 0, 1.0, 0.0) for p in initial_preds]

            for idx in range(len(initial_preds)):
                fig = overlay_attention_maps(gt[idx], initial_preds[idx].squeeze(0), int(prot_lens[idx][0]), int(rna_lens[idx][0]), plot=False)
                writer.add_figure(f'Predictions/Sample_{counter}', fig, epoch)
                counter += 1

        if epoch % save_freq == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'protein_model_state_dict': protein_model.model.state_dict() if train_protein_model else None,
                'na_model_state_dict': na_model.model.state_dict() if train_na_model else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'log_suffix': log_suffix
            }
            torch.save(checkpoint, f'{save_path}/{log_suffix}/Epochs:{epoch}.pt')

    # Save the final model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'protein_model_state_dict': protein_model.model.state_dict() if train_protein_model else None,
        'na_model_state_dict': na_model.model.state_dict() if train_na_model else None,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs - 1,
        'log_suffix': log_suffix
    }
    torch.save(checkpoint, f'{save_path}/{log_suffix}/Final_Model_Epochs:{num_epochs}.pt')
    writer.close()
    return model, all_losses_train, all_losses_eval, all_gradients
