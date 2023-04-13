
from pathlib import Path
import numpy as np 

from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from omegaconf import DictConfig
from src.utils.asl_utils import read_json_file
from sklearn.model_selection import StratifiedGroupKFold
from src.data.components.sign_dataset import ASLDataFrameDataset
import pandas as pd
import hydra
class ASLDataModule(LightningDataModule):

    def __init__(
        self,
        csv_path : str = '/opt/rsna/data/sign_data/asl-signs/train.csv',
        json_path : str = '/opt/rsna/data/sign_data/asl-signs/sign_to_prediction_index_map.json',
        npy_path : str = '/opt/rsna/data/sign_data/',
        npy_name: str = 'small_feature_data.npy',
        lab_name: str = 'small_feature_labels.npy',
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
        self.train_df = df[~df['fold'].isin([val_fold,test_fold])].reset_index(drop=True)
        self.valid_df = df[df['fold'].isin([val_fold])].reset_index(drop=True)
        self.test_df = df[df['fold'].isin([test_fold])].reset_index(drop=True)
        self.npy_path = Path(npy_path)
        self.label_map = label_map
        self.save_hyperparameters(logger=False)

    @property
    def num_classes(self):
        return len(self.label_map)

    def prepare_data(self):
        
        npy_data = self.npy_path/self.hparams.npy_name
        npy_label = self.npy_path/self.hparams.lab_name
        if npy_data.exists() and npy_label.exists():
            self.npy_data = np.load(npy_data)
            self.npy_label = np.load(npy_label)
        else:
            pass
            #TODO : make npy file function
        pass
        
    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        self.data_train = ASLDataFrameDataset(self.train_df, self.npy_data, self.npy_label)
        self.data_val = ASLDataFrameDataset(self.valid_df, self.npy_data, self.npy_label)
        self.data_test = ASLDataFrameDataset(self.test_df, self.npy_data, self.npy_label)

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



