import pandas as pd
import numpy as np
import torch as t
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from misc.Audio2Tensor import convert_mp3_to_tensor,convert_all_mp3_to_tensor,Audio2Tensor
import glob
import os
class AudioDatasetter(Dataset):

    def __init__(self, directory, sr=22050, n_mels=64, hop_length=512):
        self.directory = directory
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.file_paths = glob.glob(os.path.join(directory, '*.mp3'))

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            log_melspec = Audio2Tensor(file_path, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length)
            return log_melspec, file_path
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None, file_path
        
    
# Collate function to handle None values
def collate_fn(batch):

    # Filter out None values
    batch = [(data, path) for data, path in batch if data is not None]
    if not batch:
        return t.tensor([]), []
    
    # Separate data and file paths
    data, paths = zip(*batch)
    return t.stack(data), paths