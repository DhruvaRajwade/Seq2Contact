
import torch
import numpy as np
import random


import os


def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_distance_matrix(arr1, arr2):
    """
    Calculate the pairwise Euclidean distance matrix between two arrays.
    """
    return np.linalg.norm(arr1[:, np.newaxis] - arr2, axis=2)
## Distance matrix to contact map
def distance_matrix_to_contact_map(distance_matrix, threshold=6):
    ## Set values below threshold to 1 and above to 0
    contact_map = np.zeros_like(distance_matrix)
    contact_map[distance_matrix < threshold] = 1
    return contact_map

def clip_data(data, mode='dna'):
    na_key = 'dna_seq' if mode == 'dna' else 'rna_seq'
    data1 = data
    ### For all dictionaries in data1 only keep the keys 'pdb_id', 'dna_seq', 'protein_seq' and `complex_contact_map`
    for i in range(len(data1)):
        data1[i] = {key: data1[i][key] for key in ['pdb_id', na_key, 'protein_seq', 'complex_contact_map']}
    return data1