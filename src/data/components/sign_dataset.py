from torch.utils.data import Dataset, sampler
from torch import nn
import torch
import math 
import torch.nn.functional as F
from collections import OrderedDict

class ASLDataFrameDataset(Dataset):
    def __init__(self, df, dataset, labels, transform=None):
        self.df = df
        self.X = dataset
        self.y = labels
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Use df_index as idx due to folds splitting
        df_index = self.df.index.values[idx]
        x = self.X[df_index]
        y = self.y[df_index]
        x = torch.Tensor(x)
        y = torch.Tensor([y]).long()

        if self.transform:
            x = self.transform(x)
        return x, y

class ASLDataNPYDataset(Dataset):
    def __init__(self, array, dataset, labels, transform=None):
        self.array = array
        self.X = dataset
        self.y = labels
        self.transform = transform

    def __len__(self):
        return len(self.array)

    def __getitem__(self, idx):
        # Use df_index as idx due to folds splitting
        df_index = self.df.index.values[idx]
        x = self.X[df_index]
        y = self.y[df_index]
        x = torch.Tensor(x)
        y = torch.Tensor([y]).long()

        if self.transform:
            x = self.transform(x)
        return x, y

class ASLDataNPYDataset(Dataset):
    def __init__(self, X, y, NON_EMPTY_FRAME_IDXS):
        self.X = X
        self.y = y
        self.non_empty_frame_idxs = NON_EMPTY_FRAME_IDXS
        
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.non_empty_frame_idxs[idx]
    
    