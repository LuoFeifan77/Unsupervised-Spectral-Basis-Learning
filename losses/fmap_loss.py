# from tkinter import N
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class SquaredFrobeniusLoss(nn.Module):  # 单向的！
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, a, b):
        loss = torch.sum(torch.abs(a - b) ** 2, dim=(-2, -1))
        return self.loss_weight * torch.mean(loss)


 
@LOSS_REGISTRY.register()
class MRSLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(MRSLoss, self).__init__()
        assert loss_weight >= 0
        self.loss_weight = loss_weight

    def forward(self, loss):
        losses = dict()
        losses['mrsloss'] = self.loss_weight * loss
        return losses

