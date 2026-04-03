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
class RfmnetLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(RfmnetLoss, self).__init__()
        assert loss_weight >= 0
        self.loss_weight = loss_weight

    def forward(self, loss):
        losses = dict()
        losses['rfmnet'] = self.loss_weight * loss
        return losses


@LOSS_REGISTRY.register()
class SelfLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(SelfLoss, self).__init__()
        assert loss_weight >= 0
        self.loss_weight = loss_weight

    def forward(self, loss):
        losses = dict()
        losses['self'] = self.loss_weight * loss
        return losses



@LOSS_REGISTRY.register()
class PartialFmapsLoss(nn.Module):
    def __init__(self, w_bij=1.0, w_orth=1.0):
        """
        Init PartialFmapsLoss
        Args:
            w_bij (float, optional): Bijectivity penalty weight. Default 1.0.
            w_orth (float, optional): Orthogonality penalty weight. Default 1.0.
        """
        super(PartialFmapsLoss, self).__init__()
        assert w_bij >= 0 and w_orth >= 0, 'Loss weight should be non-negative.'
        self.w_bij = w_bij
        self.w_orth = w_orth

    def forward(self, C_fp, C_pf, evals_full, evals_partial):
        assert C_fp.shape[0] == 1, 'Currently, only support batch size = 1'
        criterion = SquaredFrobeniusLoss()
        C_fp, C_pf = C_fp[0], C_pf[0]
        evals_full, evals_partial = evals_full[0], evals_partial[0]

        ## compute area ratio between full shape and partial shape r
        
        # r = min((evals_partial < evals_full.max()).sum(), C_fp.shape[0] - 1) 
        # eye = torch.zeros_like(C_fp)
        # eye[torch.arange(0, r + 1), torch.arange(0, r + 1)] = 1.0

        eye = torch.eye(C_fp.shape[0], C_fp.shape[0], device=C_fp.device)

        if self.w_bij > 0:
            bijectivity_loss = self.w_bij * criterion(torch.matmul(C_fp, C_pf), eye)
        else:
            bijectivity_loss = 0.0

        if self.w_orth > 0:
            orthogonality_loss = self.w_orth * criterion(torch.matmul(C_fp, C_fp.t()), eye)
        else:
            orthogonality_loss = 0.0

        return {'l_bij': bijectivity_loss, 'l_orth': orthogonality_loss}
