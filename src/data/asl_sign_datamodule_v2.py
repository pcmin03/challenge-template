import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit

import pandas as pd
import numpy as np

from src.data.components.asl_sign_dataset_v2 import gen_dataset
from src.data.components.preprocess.asl_preprocess_v2 import get_file_path, get_x_y

class ASLDataModule_v2(pl.LightningDataModule):
    def __init__(
        self,
        SEED: int,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        dataset_cfg: dict,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.seed = SEED
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
    
        self.dataset_cfg = dataset_cfg
        self.file_path = self.dataset_cfg.train_path
        self.train = self.read_csv()
        self.n_samples = len(self.train)
        
        self.preprocess_flag = self.dataset_cfg.preprocess
        self.num_classes = self.dataset_cfg.num_classes
        self.input_size = self.dataset_cfg.INPUT_SIZE
        self.n_dims = self.dataset_cfg.n_dims
        
        idxes = self.landmark_indexing()
        self.hand_idx0 = idxes[0]
        self.landmark_idx0 = idxes[1]
        self.n_cols = idxes[2]
        self.lips_idxs = idxes[3]
        self.hands_idxs = idxes[4]
        self.left_hand_idxs = idxes[5]
        self.right_hand_idxs = idxes[6]
        self.pose_idxs = idxes[7]

        self.X, self.y, self.NON_EMPTY_FRAME_IDXS = self.preprocess()
        
        split_folds = self.split_fold()
        self.X_train, self.X_val = split_folds[0], split_folds[1]
        self.NON_EMPTY_FRAME_IDXS_TRAIN, self.NON_EMPTY_FRAME_IDXS_VAL = split_folds[2], split_folds[3]
        self.y_train, self.y_val = split_folds[4], split_folds[5]
        
        self.sampler = self.sampler()
        
    def read_csv(self,):
        train = pd.read_csv(self.file_path)
        train['file_path'] = train['path'].apply(get_file_path)
        # Add ordinally Encoded Sign (assign number to each sign name)
        train['sign_ord'] = train['sign'].astype('category').cat.codes
        return train
    
    def preprocess(self,):
        if self.preprocess_flag:
            X, y, NON_EMPTY_FRAME_IDXS = get_x_y(self.train, self.n_samples, self.input_size, 
                                                 self.n_cols, self.n_dims, self.hand_idx0, 
                                                 self.landmark_idx0)
        else:
            X = np.load('/opt/sign/transformer/X.npy')
            y = np.load('/opt/sign/transformer/y.npy')
            NON_EMPTY_FRAME_IDXS = np.load('/opt/sign/transformer/NON_EMPTY_FRAME_IDXS.npy')
        return X, y, NON_EMPTY_FRAME_IDXS
    
    def split_fold(self,):
        # Split data based on participant id
        splitter = GroupShuffleSplit(test_size=0.10, n_splits=2, random_state=self.seed)

        PARTICIPANT_IDS = self.train['participant_id']

        train_idxs, val_idxs = next(splitter.split(self.X, self.y, groups=PARTICIPANT_IDS))

        X_train = self.X[train_idxs]
        X_val = self.X[val_idxs]
        NON_EMPTY_FRAME_IDXS_TRAIN = self.NON_EMPTY_FRAME_IDXS[train_idxs]
        NON_EMPTY_FRAME_IDXS_VAL = self.NON_EMPTY_FRAME_IDXS[val_idxs]
        y_train = self.y[train_idxs]
        y_val = self.y[val_idxs]
        
        return (X_train, X_val, NON_EMPTY_FRAME_IDXS_TRAIN, NON_EMPTY_FRAME_IDXS_VAL, y_train, y_val)
    
    def landmark_indexing(self,):
        USE_TYPES = ['left_hand', 'pose', 'right_hand']
        START_IDX = 468
        LIPS_IDXS0 = np.array([
                61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
            ])
        # Landmark indices in original data
        LEFT_HAND_IDXS0 = np.arange(468,489)
        RIGHT_HAND_IDXS0 = np.arange(522,543)
        POSE_IDXS0 = np.arange(502, 512)
        LANDMARK_IDXS0 = np.concatenate((LIPS_IDXS0, LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0, POSE_IDXS0))
        HAND_IDXS0 = np.concatenate((LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0), axis=0)
        N_COLS = LANDMARK_IDXS0.size
        # Landmark indices in processed data
        LIPS_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, LIPS_IDXS0)).squeeze()
        LEFT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, LEFT_HAND_IDXS0)).squeeze()
        RIGHT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, RIGHT_HAND_IDXS0)).squeeze()
        HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, HAND_IDXS0)).squeeze()
        POSE_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, POSE_IDXS0)).squeeze()
        
        
        return (HAND_IDXS0, LANDMARK_IDXS0, N_COLS, LIPS_IDXS, HAND_IDXS, LEFT_HAND_IDXS, RIGHT_HAND_IDXS, POSE_IDXS)

    def sampler(self,):
         ## Weigted smapler
        CLASS2IDXS_train = {}
        for i in range(self.num_classes):
            CLASS2IDXS_train[i] = np.argwhere(self.y_train == i).squeeze().astype(np.int32)
            
        class_weights = {i:1/len(CLASS2IDXS_train[i]) for i in range(self.num_classes)}
        samples_weight = np.array([class_weights[t] for t in self.y_train])
        samples_weight = torch.from_numpy(samples_weight)
        ## train dataset용
        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(samples_weight), num_samples=len(samples_weight), replacement=True)
        return sampler

    def setup(self, stage=None):
        self.train_dataset = gen_dataset(self.X_train, self.y_train, self.NON_EMPTY_FRAME_IDXS_TRAIN, self.input_size,
                                         self.n_dims, self.n_cols, self.lips_idxs, self.hands_idxs, self.left_hand_idxs, self.right_hand_idxs,
                                         self.pose_idxs)
        self.val_dataset = gen_dataset(self.X_val, self.y_val, self.NON_EMPTY_FRAME_IDXS_VAL, self.input_size, 
                                       self.n_dims, self.n_cols, self.lips_idxs, self.hands_idxs, self.left_hand_idxs, self.right_hand_idxs,
                                        self.pose_idxs)
    
    def train_dataloader(self):
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def _dataloader(self, dataset, train=False):
        return DataLoader(
            dataset,
            sampler=self.sampler if train else None,
            batch_size=self.train_batch_size if train else self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )