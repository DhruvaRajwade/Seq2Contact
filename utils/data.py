import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from utils.util import calculate_distance_matrix, distance_matrix_to_contact_map, clip_data
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset as Subset
import os




def load_and_process_data(mode='dna', lower_threshold=10, na_upper_threshold=100, protein_upper_threshold=1000, dataset_dir=None):
    filtered_data = []
    
    # Set the default dataset directory if none is provided
    if dataset_dir is None:
        dataset_dir = './data/'
    
    if 'rna' in mode:
        print( 'Support for RNA and Reverse Transcription coming soon')
        return None
             
    elif 'dna' in mode:
        file_path = os.path.join(dataset_dir, 'dna_protein_dataset.pkl')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        data = pickle.load(open(file_path, 'rb'))
        for i in range(len(data)):
            if (len(data[i]['dna_seq']) > lower_threshold and len(data[i]['protein_seq']) > lower_threshold and 
                len(data[i]['dna_seq']) < na_upper_threshold and len(data[i]['protein_seq']) < protein_upper_threshold and 
                np.sum(data[i]['complex_contact_map']) > 0):
                filtered_data.append(data[i])
        return clip_data(filtered_data, mode='dna')            
    else:
        raise ValueError(f"Mode {mode} not recognized")
    

class ProteinNADataset(Dataset):
    def __init__(self, protein_seqs, na_seqs, contact_maps, pdb_ids):
        self.protein_seqs = protein_seqs
        self.na_seqs = na_seqs
        self.contact_maps = contact_maps
        self.pdb_ids = pdb_ids


    def __len__(self):
        return len(self.protein_seqs)

    def __getitem__(self, idx):
        return self.protein_seqs[idx], self.na_seqs[idx], self.contact_maps[idx], self.pdb_ids[idx]
    
def collate_sequences(batch):
    protein_seqs, na_seqs, contact_maps, pdb_ids = zip(*batch)
    return list(protein_seqs), list(na_seqs), list(contact_maps), list(pdb_ids)


def collate_embeddings(proteins, rnas, contacts):
    
    
    protein_lengths = [len(p) for p in proteins]
    rna_lengths = [len(r) for r in rnas]

    max_protein_len = max(protein_lengths)
    max_rna_len = max(rna_lengths)

    padded_proteins = torch.zeros(len(proteins), max_protein_len, proteins[0].size(1))
    padded_rnas = torch.zeros(len(rnas), max_rna_len, rnas[0].size(1))
    padded_contacts = torch.zeros(len(contacts), max_protein_len, max_rna_len).to('cuda')

    for i, (p, r, c) in enumerate(zip(proteins, rnas, contacts)):
        padded_proteins[i, :protein_lengths[i], :] = p
        padded_rnas[i, :rna_lengths[i], :] = r
        padded_contacts[i, :protein_lengths[i], :rna_lengths[i]] = c

    return padded_proteins, padded_rnas, padded_contacts, protein_lengths, rna_lengths



def sequence_similarity_split(data, split_path='./train_test_clusters.pkl', mode='dna'):

    if mode=='rna':
        pass
    else:
        na_key = 'dna_seq'
    #split_path = os.path.abspath(split_path)
      
    train_protein_sequences, train_na_sequences, train_contact_maps, train_pdbs = [], [], [],[]
    test_protein_sequences, test_na_sequences, test_contact_maps, test_pdbs= [], [], [], []
    with open(split_path, 'rb') as infile:
        train_clusters, test_clusters = pickle.load(infile)
    for d in data:
        if d['pdb_id'] in train_clusters:
            train_protein_sequences.append(d['protein_seq'])
            train_na_sequences.append(d[na_key])
            train_contact_maps.append(torch.tensor(d['complex_contact_map']).T)
            train_pdbs.append(d['pdb_id'])
        else:
            test_protein_sequences.append(d['protein_seq'])
            test_na_sequences.append(d[na_key])
            test_contact_maps.append(torch.tensor(d['complex_contact_map']).T)
            test_pdbs.append(d['pdb_id'])

    train_dataset = ProteinNADataset(train_protein_sequences, train_na_sequences, train_contact_maps, train_pdbs)
    test_dataset = ProteinNADataset(test_protein_sequences, test_na_sequences, test_contact_maps, test_pdbs)

    return train_dataset, test_dataset        

        
    
def get_contact_maps(dataset):
    return [i for (_,_,i,_) in dataset]


def collate_fn(batch):
    proteins, rnas, contacts = zip(*batch)
    
    protein_lengths = [len(p) for p in proteins]
    rna_lengths = [len(r) for r in rnas]

    max_protein_len = max(protein_lengths)
    max_rna_len = max(rna_lengths)

    padded_proteins = torch.zeros(len(proteins), max_protein_len, proteins[0].size(1))
    padded_rnas = torch.zeros(len(rnas), max_rna_len, rnas[0].size(1))
    padded_contacts = torch.zeros(len(contacts), max_protein_len, max_rna_len)

    for i, (p, r, c) in enumerate(zip(proteins, rnas, contacts)):
        padded_proteins[i, :protein_lengths[i], :] = p
        padded_rnas[i, :rna_lengths[i], :] = r
        padded_contacts[i, :protein_lengths[i], :rna_lengths[i]] = c

    return padded_proteins, padded_rnas, padded_contacts, protein_lengths, rna_lengths



class VariableLengthDataset(Dataset):
    def __init__(self, protein_embeddings, rna_embeddings, contact_maps):
        self.protein_embeddings = protein_embeddings
        self.rna_embeddings = rna_embeddings
        self.contact_maps = contact_maps

    def __len__(self):
        return len(self.protein_embeddings)

    def __getitem__(self, idx):
        return self.protein_embeddings[idx], self.rna_embeddings[idx], self.contact_maps[idx]


