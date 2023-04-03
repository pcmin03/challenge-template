import hydra

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import torchmetrics
from torchmetrics import MetricCollection

from timm.optim import create_optimizer_v2
from optimizers.optimizer import Lookahead

import warnings

warnings.filterwarnings(action='ignore')


class ASLModule(pl.LightningModule):
    def __init__(self,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler,
                net: torch.nn.Module, 
                **kwargs):
        super().__init__()

        self.save_hyperparameters()
        
        self.model = hydra.utils.instantiate(net)
        
        self.preprocessor = hydra.utils.instantiate(self.hparams.preprocessor, 
                                                    _recursive_=False)
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        self.metrics = self._init_metrics()
    

    def _init_metrics(self):
        metrics = {
            "acc": torchmetrics.classification.MulticlassAccuracy(
                num_classes=250
            ),
        }
        metric_collection = MetricCollection(metrics)

        return torch.nn.ModuleDict(
            {
                "train_metrics": metric_collection.clone(prefix="train_"),
                "val_metrics": metric_collection.clone(prefix="val_"),
            }
        )

    def configure_optimizers(self):
        optimizer = self._init_optimizer()

        scheduler = self._init_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        } if scheduler else {"optimizer": optimizer,} 

    def _init_optimizer(self):
        opt_lower = self.hparams.optimizer.opt_name.lower()
        opt_split = opt_lower.split('_')
        if len(opt_split) > 1:
            if opt_split[0] == 'lookahead':
                optimizer = create_optimizer_v2(
                                    model_or_params = self.parameters(),
                                    opt=opt_split[-1],
                                    lr=self.hparams.optimizer.learning_rate,
                                    weight_decay=self.hparams.optimizer.weight_decay,
                                )
                optimizer = Lookahead(optimizer, alpha=0.5, k=5)
        else:
            print(self.parameters())
            optimizer = create_optimizer_v2(
                                    model_or_params = self.parameters(),
                                    opt=self.hparams.optimizer.opt_name,
                                    lr=self.hparams.optimizer.learning_rate,
                                    weight_decay=self.hparams.optimizer.weight_decay,
                                 )
        return optimizer

    def _init_scheduler(self, optimizer):
        if self.hparams.scheduler.sch_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.scheduler.max_epochs,
                eta_min=self.hparams.scheduler.eta_min,
            )
        elif self.hparams.scheduler.sch_name == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.scheduler.max_epochs // 5,
                gamma=0.95,
            )
        else:
            scheduler = None
        return scheduler

    def forward(self, x, non_empty_frame_idxs, mask):
        x = self.preprocessor(x)
        x = x[0], x[1], x[2], x[3], non_empty_frame_idxs
        return self.model(x, mask)

    def training_step(self, batch):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")
    
    def predict_step(self, batch, batch_idx):
        labels, logits = self._forward_pass(batch)
        probs = torch.softmax(logits, dim=-1)
        probs = torch.Tensor(probs)
        preds = torch.argmax(logits, dim=-1) 
        return probs, preds, labels

    def _shared_step(self, batch, stage):
        y, y_logit = self._forward_pass(batch)

        loss = self.loss_fn(y_logit, y)
        
        y_pred = torch.softmax(y_logit, dim=-1)
        self.metrics[f"{stage}_metrics"](y_pred, y)

        self._log(stage, loss)

        return loss

    def _forward_pass(self, batch):
        x = batch['X']
        y = batch['label']
        non_empty_frame_idxs = batch['non_empty_frame_idxs']
        non_empty_frame_idxs = torch.tensor(non_empty_frame_idxs)
        mask = torch.ne(non_empty_frame_idxs, -1).float()
        mask = torch.unsqueeze(mask, dim=2)
        y_logit = self.forward(x, non_empty_frame_idxs, mask)

        return y, y_logit

    def _log(self, stage, loss):
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
        self.log_dict(self.metrics[f"{stage}_metrics"], on_epoch=True, prog_bar=True, on_step=False)