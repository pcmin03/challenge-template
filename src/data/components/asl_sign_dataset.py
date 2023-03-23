import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset
from utils.asl_utils import read_kaggle_csv_by_part, read_christ_csv_by_part, load_relevant_data_subset, read_json_file
from src.data.components.augmentation import do_random_affine


class ASLDataset(Dataset):
    def __init__(self, mode, train_path, pr_train_path, json_path, num_fold, 
                 fold_type, val_fold, max_length, random, use_feature, offset, transform=None):
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
        if offset: #https://www.kaggle.com/code/dschettler8845/gislr-how-to-ensemble/notebook
            offset_ = (np.arange(1000)-self.max_length)//2
            offset_ = np.clip(offset_, 0, 1000).tolist()
            self.offset = nn.Parameter(torch.LongTensor(offset_),requires_grad=False)
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
    
    def pre_process(self, xyz, L):
        
        #todo use tril:210
        triu_index = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 28,
            29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
            45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 67, 68, 69, 70, 71, 72, 73, 74,
            75, 76, 77, 78, 79, 80, 81, 82, 83, 89, 90, 91, 92,
            93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 111,
            112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
            125, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
            145, 146, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,
            166, 167, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
            188, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 221,
            222, 223, 224, 225, 226, 227, 228, 229, 230, 243, 244, 245, 246,
            247, 248, 249, 250, 251, 265, 266, 267, 268, 269, 270, 271, 272,
            287, 288, 289, 290, 291, 292, 293, 309, 310, 311, 312, 313, 314,
            331, 332, 333, 334, 335, 353, 354, 355, 356, 375, 376, 377, 397,
            398, 419,
        ]
        
        LIP = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ]
        SPOSE = [504, 502, 500, 501, 503, 505, 512, 513]
        
        lhand = xyz[:, 468:489]
        rhand = xyz[:, 522:543]
        
        # add distance
        lhand2 = lhand[:, :21, :2]
        ld = lhand2.reshape(-1, 21, 1, 2) - lhand2.reshape(-1, 1, 21, 2)
        ld = np.sqrt((ld ** 2).sum(-1))
        ld = ld.reshape(L, -1)
        ld = ld[:,triu_index]

        rhand2 = rhand[:, :21, :2]
        rd = rhand2.reshape(-1, 21, 1, 2) - rhand2.reshape(-1, 1, 21, 2)
        rd = np.sqrt((rd ** 2).sum(-1))
        rd = rd.reshape(L, -1)
        rd = rd[:,triu_index]

        xyz = torch.cat([  # (none, 82, 3)
            lhand,
            rhand,
            xyz[:, LIP],
            xyz[:, SPOSE],
        ], 1).contiguous()
        
        dxyz = F.pad(xyz[:-1] - xyz[1:], [0, 0, 0, 0, 0, 1])
        # (L, 960)
        x = torch.cat([
            xyz.reshape(L,-1),
            dxyz.reshape(L,-1),
            rd.reshape(L,-1),
            ld.reshape(L,-1),
        ], -1)
        
        x[torch.isnan(x)] = 0
        
        return x
    
    def __len__(self):
        return len(self.meta_df)
    
    
    def __getitem__(self, idx):

        elem = self.meta_df.iloc[idx]
        if self.use_feature:
            xyz = torch.from_numpy(np.load(elem.feature_npy_path))
        else:
            pq_path = elem.path
            xyz = load_relevant_data_subset(f'/opt/sign/challenge-template/data/{pq_path}')
            L = len(xyz)
            if L > self.max_length:
                #xyz = xyz[:self.max_length] #first
                #xyz = xyz[-self.max_length:] #last
                i = self.offset[L]
                xyz = xyz[i:i+self.max_length] #center
                
            L = len(xyz)
            not_nan_xyz = xyz[~np.isnan(xyz)]
            
            if len(not_nan_xyz) != 0:
                not_nan_xyz_mean = not_nan_xyz.mean(0, keepdims=True)  # noramlisation to common maen
                not_nan_xyz_std  = not_nan_xyz.std(0, keepdims=True)
                xyz -= not_nan_xyz_mean
                xyz /= not_nan_xyz_std

            if self.transform and self.mode=='train':
                xyz = do_random_affine(xyz)

            xyz = torch.from_numpy(xyz).float()
            xyz_features = self.pre_process(xyz, L)

        data = {}
        data['index'] = idx
        data['d'] = elem
        data['xyz'] = xyz_features
        data['label'] = elem.label
        return data