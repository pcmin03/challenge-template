
from pathlib import Path
import numpy as np 

from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, WeightedRandomSampler
from omegaconf import DictConfig
from src.utils.asl_utils import read_json_file
from sklearn.model_selection import StratifiedGroupKFold
from src.data.components.sign_dataset import ASLDataNPYDataset
import pandas as pd
import hydra
class ASLDataModule(LightningDataModule):

    def __init__(
        self,
        csv_path : str = '/opt/rsna/data/sign_data/asl-signs/train.csv',
        json_path : str = '/opt/rsna/data/sign_data/asl-signs/sign_to_prediction_index_map.json',
        npy_path : str = '/opt/rsna/data/sign_data/',
        npy_name: str = 'X.npy',
        lab_name: str = 'y.npy',
        non_emp_name : str = 'NON_EMPTY_FRAME_IDXS.npy',
        val_fold : int = 0,
        test_fold : int = 1,
        batch_size : int = 100,
        # preprocess : DictConfig=None,
    ):
        super().__init__()
        csv_path = Path(csv_path)
        abs_path = csv_path.parent

        df = pd.read_csv(str(csv_path))
        label_map = read_json_file(json_path)


        # sign to heatmap
        df['label'] = df['sign'].map(label_map)
        df['abs_path'] = abs_path / df['path']

        # dataframe check fold
        if 'fold' not in df: 
            sgkf = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
            for i, (_, valid_index) in enumerate(sgkf.split(df.path, df.label, df.participant_id)):
                df.loc[valid_index,'fold'] = i
    
        df['abs_path'] = df['abs_path'].map(str)

        # make train, valid, test dataframe
        self.train_df = df[~df['fold'].isin([val_fold,test_fold])]
        self.valid_df = df[df['fold'].isin([val_fold])]
        self.test_df = df[df['fold'].isin([test_fold])]

        self.npy_path = Path(npy_path)
        self.label_map = label_map
        self.save_hyperparameters(logger=False)

    @property
    def num_classes(self):
        return len(self.label_map)

    def prepare_data(self):
        
        npy_data = self.npy_path/self.hparams.npy_name
        npy_label = self.npy_path/self.hparams.lab_name
        non_emp_lab = self.npy_path/self.hparams.non_emp_name

        if npy_data.exists() and npy_label.exists():
            self.npy_data = np.load(npy_data)
            self.npy_label = np.load(npy_label)
            self.non_empty_frame = np.load(non_emp_lab)
        else:
            pass
            #TODO : make npy file function
        pass
        
    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        train_idxs = np.array(self.train_df.index)
        valid_idxs = np.array(self.valid_df.index)
        test_idxs = np.array(self.test_df.index)

        # load and split datasets only if not loaded already
        self.data_train = ASLDataNPYDataset(self.npy_data[train_idxs], self.npy_label[train_idxs], self.non_empty_frame[train_idxs])
        self.data_val = ASLDataNPYDataset(self.npy_data[valid_idxs], self.npy_label[valid_idxs], self.non_empty_frame[valid_idxs])
        self.data_test = ASLDataNPYDataset(self.npy_data[test_idxs], self.npy_label[test_idxs], self.non_empty_frame[test_idxs])

        CLASS2IDXS_train = {}
        for i in range(250):
            CLASS2IDXS_train[i] = np.argwhere(self.npy_label[train_idxs] == i).squeeze().astype(np.int32)
            
        class_weights = {i:1/len(CLASS2IDXS_train[i]) for i in range(250)}
        samples_weight = np.array([class_weights[t] for t in self.npy_label[train_idxs]])
        samples_weight = torch.from_numpy(samples_weight)
        
        ## train dataset용
        self.sampler = WeightedRandomSampler(weights=torch.DoubleTensor(samples_weight), num_samples=len(samples_weight), replacement=True)
        
    def train_dataloader(self):
        self.train_loader =  DataLoader(dataset=self.data_train,
                                        batch_size=self.hparams.batch_size,
                                        shuffle=True,
                                        num_workers = 4,
                                        pin_memory = True,
                                        prefetch_factor =  4,
                                        persistent_workers = True,
                                        )

        return self.train_loader

    def val_dataloader(self):
        self.val_loader = DataLoader(dataset=self.data_val,
                                    batch_size=self.hparams.batch_size, 
                                    shuffle=False,
                                    num_workers = 4,
                                    pin_memory = True,
                                    prefetch_factor =  4,
                                    persistent_workers = True
                                    )
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = DataLoader(dataset=self.data_test,
                                    batch_size=self.hparams.batch_size,
                                    shuffle=False,
                                    num_workers = 4,
                                    pin_memory = True,
                                    prefetch_factor =  4,
                                    persistent_workers = True,)

        return self.test_loader

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass



