from typing import Any, List

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, LightningDataModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import AUROC, F1Score
from torchmetrics import MetricCollection
from omegaconf import DictConfig
from timm.optim import create_optimizer_v2

class ASLModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        net: torch.nn.Module,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.net = net
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        self.arc_loss = torch.nn.CrossEntropyLoss()
        self._init_metrics()

    def _init_metrics(self):
        # metric objects for calculating and averaging accuracy across batches
        metrics = {
            "acc": Accuracy(task='multiclass',num_classes=250),
            "f1": F1Score(task='multiclass',num_classes=250),
            "arc_acc": Accuracy(task='multiclass',num_classes=250),
        }
        metric_collection = MetricCollection(metrics)
        self.metrics = torch.nn.ModuleDict(
                {
                    "train_metrics": metric_collection.clone(prefix="train_"),
                    "valid_metrics": metric_collection.clone(prefix="val_"),
                    "test_metrics": metric_collection.clone(prefix="test_"),
                    "train_arc_metrics": metric_collection.clone(prefix="train_"),
                    "valid_arc_metrics": metric_collection.clone(prefix="val_"),
                    "test_arc_metrics": metric_collection.clone(prefix="test_"),
                }
            )
    def _init_losses(self,): 
        pass

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.net(x, y)
    
    def _forward_pass(self, batch):
        x, y = batch
        y = y.view(-1)
        y_pred = self(x,y)

        return x, y, y_pred

    def _shared_step(self, batch: Any, stage: str):
        x, y, y_pred = self._forward_pass(batch)

        loss = self.criterion(y_pred[0].float(), y.long())
        arcface_loss = self.criterion(y_pred[1].float(), y.long())
        loss = 0.5 * loss + 0.5 * arcface_loss

        self.metrics[f"{stage}_metrics"](y_pred[0].float(), y.long())
        self.metrics[f"{stage}_arc_metrics"](y_pred[1].float(), y.long())
        self.log(f"{stage}_loss", loss, batch_size=len(x))
        self.log_dict(self.metrics[f"{stage}_metrics"], batch_size=len(x))
        
        return loss, y_pred, y

    def training_step(self, batch: Any, batch_idx: int):
        # print(print(dir(self.trainer)))
        loss, preds, targets = self._shared_step(batch, 'train')
        return loss

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self._shared_step(batch, 'valid')
        return loss

    def validation_epoch_end(self, outputs: List[Any]):
        pass
        
    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self._shared_step(batch, 'test')
        return loss

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self._init_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "valid_loss"
                },
            }
        return {"optimizer": optimizer}

    def _init_scheduler(self, optimizer):
        if self.hparams.scheduler.name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.scheduler.max_epochs,
                eta_min=self.hparams.scheduler.eta_min,
            )
        elif self.hparams.scheduler.name == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.scheduler.max_epochs // 5,
                gamma=0.95,
            )
        elif self.hparams.scheduler.name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.hparams.scheduler.factor,
                patience=self.hparams.scheduler.patience,
                verbose=False,
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.hparams.scheduler.name}")
        return scheduler


    # def _init_optimizer(self):
    #     return create_optimizer_v2(
    #         self.parameters(),
    #         opt=self.hparams.optimizer,
    #         lr=self.hparams.learning_rate,
    #         weight_decay=self.hparams.weight_decay,
    #     )

    # def _init_scheduler(self, optimizer):
    #     if self.hparams.scheduler == "CosineAnnealingLR":
    #         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #             optimizer,
    #             T_max=self.hparams.max_epochs,
    #             eta_min=self.hparams.eta_min,
    #         )
    #     elif self.hparams.scheduler == "StepLR":
    #         scheduler = torch.optim.lr_scheduler.StepLR(
    #             optimizer,
    #             step_size=self.hparams.max_epochs // 5,
    #             gamma=0.95,
    #         )
    #     else:
    #         raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")
    #     return scheduler