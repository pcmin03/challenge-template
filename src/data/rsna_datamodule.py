from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from omegaconf import DictConfig

class MMGDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_cfg : DictConfig,
        loader_cfg : DictConfig
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        

        
    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        pass
        
    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        self.data_train = hydra.utils.instantiate(self.hparams.data_cfg, mode='train')
        self.data_val = hydra.utils.instantiate(self.hparams.data_cfg, mode='valid')
        self.data_test = hydra.utils.instantiate(self.hparams.data_cfg, mode='valid')

    def train_dataloader(self):
        self.train_loader =  hydra.utils.instantiate(self.hparams.loader, 
                                    dataset=self.data_test,
                                    shuffle=True)

        return self.train_loader

    def val_dataloader(self):
        self.val_loader = hydra.utils.instantiate(self.hparams.loader, 
                                            dataset=self.data_val,
                                            shuffle=False)
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = hydra.utils.instantiate(self.hparams.loader, 
                                    dataset=self.data_test,
                                    shuffle=False)

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


if __name__ == "__main__":
    _ = MMGDataModule()
