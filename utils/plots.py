import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import torch.nn as nn
import torch
from utils.loss import weighted_bce_loss, vanilla_bce_loss
import torch.nn.functional as F

def smooth(data, alpha=0.9):
    smoothed_data = []
    for i, value in enumerate(data):
        if i == 0:
            smoothed_data.append(value)
        else:
            smoothed_value = alpha * smoothed_data[-1] + (1 - alpha) * value
            smoothed_data.append(smoothed_value)
    return smoothed_data


def plot_metrics(train_loss_list, val_loss_list, grad_list):
    # Plot training loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.plot(smooth(train_loss_list), label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Iterations')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(smooth(val_loss_list), label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Validation Loss Over Iterations')
    plt.legend()


    # Plot gradients
    plt.subplot(1, 3, 3)
    plt.plot(smooth(grad_list), label='Gradients')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Magnitude')
    plt.title('Gradient Magnitude Over Iterations')
    plt.legend()

    plt.show()

def visualize_attention_maps(gt_attention, predicted_attention, protein_len, rna_len, plot=True):

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    predicted_attention = predicted_attention.squeeze(0)   
    gt_attention = gt_attention.squeeze(0) 
    ax = axes[0]
    cax = ax.matshow(gt_attention[:protein_len, :rna_len], cmap='viridis')
    fig.colorbar(cax, ax=ax)
    ax.set_title(f'Ground Truth Contact Map| loss:{torch.mean(F.binary_cross_entropy(gt_attention.view(-1), predicted_attention.view(-1))):.2f}')
    ax.set_xlabel('NA Sequence')
    ax.set_ylabel('Protein Sequence')

    ax = axes[1]
    cax = ax.matshow(predicted_attention[:protein_len, :rna_len], cmap='viridis')
    fig.colorbar(cax, ax=ax)
    ax.set_title('Predicted Contact Map')
    ax.set_xlabel('NA Sequence')
    ax.set_ylabel('Protein Sequence')
    if not plot:
        plt.close()
        return fig
    else:
        plt.show()    

def overlay_attention_maps(gt_attention, predicted_attention, protein_len, rna_len, plot=True):
    fig, axes = plt.subplots(1, 3, figsize=(7, 7), constrained_layout=True)
    predicted_attention = predicted_attention.squeeze(0).cpu().numpy()
    gt_attention = gt_attention.squeeze(0).cpu().numpy()

    # Ground Truth Contact Map
    ax = axes[0]
    cax = ax.matshow(gt_attention[:protein_len, :rna_len], cmap='viridis')
    #fig.colorbar(cax, ax=ax)
    ax.set_title(f'GT')
    ax.set_xlabel('NA Seq')
    ax.set_ylabel('Protein Seq')

    # Overlay Plot
    ax = axes[1]
    overlay = np.ones((protein_len, rna_len, 3))
    for i in range(protein_len):
        for j in range(rna_len):
            if gt_attention[i, j] > 0.5 and predicted_attention[i, j] > 0.5:
                overlay[i, j] = [0, 0, 0]  # Purple for matching non-zero points
            elif gt_attention[i, j] > 0.5 and predicted_attention[i, j] <= 0.5:
                overlay[i, j] = [0, 0, 1]  # Blue for ground truth contacts
            elif gt_attention[i, j] <= 0.5 and predicted_attention[i, j] > 0.5:
                overlay[i, j] = [1, 0, 0]  # Red for incorrect predictions
    ax.imshow(overlay)
    ax.set_title('Overlay')
    ax.set_xlabel('NA Seq')
    ax.set_ylabel('Protein Seq')

    # Predicted Contact Map
    ax = axes[2]
    cax = ax.matshow(predicted_attention[:protein_len, :rna_len], cmap='viridis')
    #fig.colorbar(cax, ax=ax)
    ax.set_title('Pred')
    ax.set_xlabel('NA Seq')
    ax.set_ylabel('Protein Seq')

    if not plot:
        plt.close()
        return fig
    else:
        plt.show()


