import torch
import math
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
# From https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py
# Added type annotations, device, and 16bit support
class ArcMarginProduct(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float,
        margin: float,
        easy_margin: bool,
        ls_eps: float,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
        self, input: torch.Tensor, label: torch.Tensor, device: str = "cuda"
    ) -> torch.Tensor:
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # # Enable 16 bit precision
        # cosine = cosine.to(torch.float32)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output

class Model(nn.Module):
    def __init__(self,
                    in_features: int,
                    num_classes: int,
                    scale: int,
                    margin: float,
                    drop_rate: float,
                    use_arcface: bool,
                    ):
        super(Model, self).__init__()
        if in_features > 1000: 
            self.lin_bn_mish = nn.Sequential(
                OrderedDict(
                    [
                        ("lin_mish1", lin_bn_mish(in_features, 1024, drop_rate)),
                        ("lin_mish2", lin_bn_mish(1024, 512, drop_rate)),
                        ("lin_mish3", lin_bn_mish(512, 250, drop_rate)),
                        ("lin_mish4", lin_bn_mish(250, 128, drop_rate)),
                    ]
                )
            )
        else: 
            self.lin_bn_mish = nn.Sequential(
                OrderedDict(
                    [
                        ("lin_mish1", lin_bn_mish(in_features, 512, drop_rate)),
                        ("lin_mish2", lin_bn_mish(512, 256, drop_rate)),
                        ("lin_mish3", lin_bn_mish(256, 256, drop_rate)),
                        ("lin_mish4", lin_bn_mish(256, 128, drop_rate)),
                    ]
                )
            )

        self.final = ArcMarginProduct(
            in_features=128,
            out_features=num_classes,
            scale=scale,
            margin=margin,
            easy_margin=False,
            ls_eps=0.0,
        )
        self.fc_probs = nn.Linear(128, num_classes)
        self.use_arcface = use_arcface

    def forward(self, x, label):
        feature = self.lin_bn_mish(x)
        if self.use_arcface:
            arcface = self.final(feature, label)
            probs = self.fc_probs(feature)
            return probs, arcface
        else:
            probs = self.fc_probs(feature)
            return probs, probs


def lin_bn_mish(input_dim, output_dim, drop_rate):
    return nn.Sequential(
        OrderedDict(
            [
                ("lin", nn.Linear(input_dim, output_dim, bias=False)),
                ("bn", nn.BatchNorm1d(output_dim)),
                ("dropout", nn.Dropout(drop_rate)),
                ("relu", nn.Mish()),
            ]
        )
    )
