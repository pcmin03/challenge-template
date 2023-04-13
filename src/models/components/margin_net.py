# From https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py
# Added type annotations, device, and 16bit support
from torch import nn
import torch

class ASLLinearModel(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        first_out_features: int,
        num_classes: int,
        num_blocks: int,
        drop_rate: float,
    ):
        super(ASLLinearModel, self).__init__()

        blocks = []
        out_features = first_out_features
        for idx in range(num_blocks):
            if idx == num_blocks - 1:
                out_features = num_classes

            blocks.append(self._make_block(in_features, out_features, drop_rate))

            in_features = out_features
            out_features = out_features // 2

        self.model = nn.Sequential(*blocks)
        print(self.model)

    def _make_block(self, in_features, out_features, drop_rate):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.model(x)