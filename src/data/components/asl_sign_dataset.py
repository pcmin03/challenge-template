import torch
import numpy as np

from torch.utils.data import Dataset
from utils.asl_utils import read_kaggle_csv_by_part, read_christ_csv_by_part, load_relevant_data_subset, read_json_file

class ASLDataset(Dataset):
    def __init__(self, mode, train_path, pr_train_path, json_path, num_fold, 
                 fold_type, val_fold, max_length, random, use_feature, transform=None):
        self.mode = mode
        self.train_file = train_path
        self.pr_train_file = pr_train_path
        self.sign_to_idx = read_json_file(json_path)
        self.num_fold = num_fold
        self.fold_type = fold_type
        self.val_fold = val_fold
        self.transform = transform
        self.max_length = max_length
        self.random  = random
        self.use_feature = use_feature
        self.meta_df = self.get_meta_df()
        
    def get_meta_df(self):
        
        if self.fold_type == 'hwc':
            df = read_kaggle_csv_by_part(num_fold=self.num_fold, TRAIN_FILE=self.train_file, 
                                         SIGN_TO_IDX=self.sign_to_idx, random=self.random)
            
        elif self.fold_type == 'christ':
            df = read_christ_csv_by_part(PR_TRAIN_FILE=self.pr_train_file , TRAIN_FILE=self.train_file)
        
        if self.mode == 'train':
            meta_df = df[df.fold!=self.val_fold]
        
        elif self.mode == 'val':
            meta_df = df[df.fold==self.val_fold]
            
        return meta_df.reset_index(drop=True)
    
    def pre_process(self, xyz):
        # xyz = xyz - xyz[~torch.isnan(xyz)].mean(0,keepdims=True) #noramlisation to common mean
        # xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)
        LIP = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ]
        lip = xyz[:, LIP]
        lhand = xyz[:, 468:489]
        rhand = xyz[:, 522:543]
        xyz = torch.cat([ #(none, 82, 3)
            lip,
            lhand,
            rhand,
        ],1)
        xyz[torch.isnan(xyz)] = 0
        xyz = xyz[:self.max_length]
        
        return xyz
    
    def __len__(self):
        return len(self.meta_df)
    
    
    def __getitem__(self, idx):

        elem = self.meta_df.iloc[idx]
        if self.use_feature:
            xyz = torch.from_numpy(np.load(elem.feature_npy_path))
        else:
            pq_path = elem.path
            xyz = load_relevant_data_subset(f'/opt/sign/data/data/sign_data/asl-signs/{pq_path}')
            xyz = xyz - xyz[~np.isnan(xyz)].mean(0,keepdims=True) #noramlisation to common maen
            xyz = xyz / xyz[~np.isnan(xyz)].std(0, keepdims=True)
            
            if self.transform is not None and self.mode=='train':
                xyz = self.transform(xyz)

            xyz = torch.from_numpy(xyz).float()
            xyz = self.pre_process(xyz)


        data = {}
        data['index'] = idx
        data['d'] = elem
        data['xyz'] = xyz
        data['label'] = elem.label
        return data