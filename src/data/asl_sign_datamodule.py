import torch
import pytorch_lightning as pl

from src.data.components.asl_sign_dataset import ASLDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils.asl_utils import null_collate
import numpy as np

class ASLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        dataset_cfg: dict,
    ):
        super().__init__()

        self.save_hyperparameters()

        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_transform, self.val_transform = self._init_transforms()
        
        self.dataset_cfg = dataset_cfg
    
    def _init_transforms(self):
        train_transform = None
        val_transform = None

        return train_transform, val_transform

    def setup(self, stage=None):
        self.train_dataset = ASLDataset(**self.dataset_cfg, mode='train', transform=self.train_transform)
        self.val_dataset = ASLDataset(**self.dataset_cfg, mode='val', transform=self.val_transform)
    def train_dataloader(self):
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def _dataloader(self, dataset, train=False):
        return DataLoader(
            dataset,
            sampler=RandomSampler(dataset) if train else SequentialSampler(dataset),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=train,
            worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
            collate_fn=null_collate,
        )