import torch
import torch.nn as nn

def create_mask(protein_lengths, rna_lengths, max_protein_len, max_rna_len):
    protein_mask = torch.arange(max_protein_len).expand(len(protein_lengths), max_protein_len) < torch.tensor(protein_lengths).unsqueeze(1)
    rna_mask = torch.arange(max_rna_len).expand(len(rna_lengths), max_rna_len) < torch.tensor(rna_lengths).unsqueeze(1)
    mask = protein_mask.unsqueeze(2) & rna_mask.unsqueeze(1)
    return mask.float()

def calculate_pos_weight(gt_attention, mask):
    num_positive = torch.sum(gt_attention)
    #num_negative = gt_attention.numel() - num_positive
    num_negative = gt_attention.numel() - num_positive - (1-mask).sum()
    pos_weight = num_negative / num_positive
    #pos_weight = torch.tensor(100)
    return pos_weight


def vanilla_bce_loss(predicted_attention, gt_attention, mask):

    predicted_attention = predicted_attention.squeeze()
    predicted_attention =torch.sigmoid(predicted_attention)

    bce_loss = nn.BCELoss(reduction='none')
    
    loss = bce_loss(predicted_attention, gt_attention)
   
    mask = mask.squeeze()
    loss = loss * mask
    return loss.sum() / mask.sum()


def weighted_bce_loss(predicted_attention, gt_attention, mask, pos_weight=None):
    predicted_attention = predicted_attention.squeeze()
    #predicted_attention = predicted_attention
    bce_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(20))
    loss = bce_loss(predicted_attention, gt_attention)

    mask = mask.squeeze()
    loss = loss * mask
    
    return loss.sum()/mask.sum() 
