from math import fabs
import torch
import torch.nn as nn

from utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class CoupleSquaredFrobeniusLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, a, b):
        loss = torch.sum(torch.abs(a - b) ** 2, dim=(-2, -1))
        return self.loss_weight * torch.mean(loss)


@LOSS_REGISTRY.register()
class CoupleSURFMNetLoss(nn.Module):
    """
    Loss as presented in the SURFMNet paper.
    Orthogonality + Bijectivity + Laplacian Commutativity
    """

    def __init__(self, w_bij=1.0, w_orth=1.0, w_lap=1e-3, bidirectional_loss=True):
        """
        Init SURFMNetLoss

        Args:
            w_bij (float, optional): Bijectivity penalty weight. Default 1e3.
            w_orth (float, optional): Orthogonality penalty weight. Default 1e3.
            w_lap (float, optional): Laplacian commutativity penalty weight. Default 1.0.
        """
        super(CoupleSURFMNetLoss, self).__init__()
        assert w_bij >= 0 and w_orth >= 0 and w_lap >= 0
        self.w_bij = w_bij
        self.w_orth = w_orth
        self.w_lap = w_lap
        self.bidirectional_loss= bidirectional_loss

    def forward(self, C12, C21, C12_est, C21_est, C12_est_filter, C21_est_filter, evals_1, evals_2):
        """
        Compute bijectivity loss + orthogonality loss
                            + Laplacian commutativity loss
                            + descriptor preservation via commutativity loss

        Args:
            C12 (torch.Tensor): matrix representation of functional map (1->2). Shape: [N, K, K]
            C21 (torch.Tensor): matrix representation of functional map (2->1). Shape: [N, K, K]
            evals_1 (torch.Tensor): eigenvalues of shape 1. Shape [N, K]
            evals_2 (torch.Tensor): eigenvalues of shape 2. Shape [N, K]
        """
        criterion = CoupleSquaredFrobeniusLoss()
        eye = torch.eye(C12.shape[1], C12.shape[2], device=C12.device).unsqueeze(0)
        eye_batch = torch.repeat_interleave(eye, repeats=C12.shape[0], dim=0)

        losses = dict()
        # Bijectivity penalty
        if self.w_bij > 0:
            if self.bidirectional_loss:
                bijectivity_loss = criterion(torch.bmm(C12, C21_est_filter), eye_batch)+ criterion(torch.bmm(C12_est, C21_est), eye_batch) \
                    + criterion(torch.bmm(C21, C12_est_filter), eye_batch) + criterion(torch.bmm(C21_est, C12_est), eye_batch)
                bijectivity_loss *= self.w_bij
                losses['l_freq_bij'] = bijectivity_loss
            else: 
                losses['l_freq_bij'] = 0
            
        # Orthogonality penalty
        if self.w_orth > 0:
            orthogonality_loss = criterion(torch.bmm(C12.transpose(1, 2), C12_est_filter), eye_batch) \
                + criterion(torch.bmm(C12_est.transpose(1, 2), C12_est), eye_batch)
            
            if self.bidirectional_loss :
                orthogonality_loss += criterion(torch.bmm(C21.transpose(1, 2), C21_est_filter), eye_batch) \
                    + criterion(torch.bmm(C21_est.transpose(1, 2), C21_est), eye_batch)
            
            orthogonality_loss *= self.w_orth
            losses['l_freq_orth'] = orthogonality_loss

        # Laplacian commutativity penalty
        if self.w_lap > 0:
            laplacian_loss = criterion(torch.einsum('abc,ac->abc', C12, evals_1),
                                    torch.einsum('ab,abc->abc', evals_2, C12))
            if self.bidirectional_loss :
                laplacian_loss += criterion(torch.einsum('abc,ac->abc', C21, evals_2),
                                            torch.einsum('ab,abc->abc', evals_1, C21))
            laplacian_loss *= self.w_lap
            losses['l_freq_lap'] = laplacian_loss

        return losses


@LOSS_REGISTRY.register()
class CouplePartialFmapsLoss(nn.Module):
    def __init__(self, w_bij=1.0, w_orth=1.0):
        """
        Init PartialFmapsLoss
        Args:
            w_bij (float, optional): Bijectivity penalty weight. Default 1.0.
            w_orth (float, optional): Orthogonality penalty weight. Default 1.0.
        """
        super(CouplePartialFmapsLoss, self).__init__()
        assert w_bij >= 0 and w_orth >= 0, 'Loss weight should be non-negative.'
        self.w_bij = w_bij
        self.w_orth = w_orth

    def forward(self, C_fp, C_pf, evals_full, evals_partial):
        assert C_fp.shape[0] == 1, 'Currently, only support batch size = 1'
        criterion = CouplePartialFmapsLoss()
        C_fp, C_pf = C_fp[0], C_pf[0]
        evals_full, evals_partial = evals_full[0], evals_partial[0]

        eye = torch.eye(C_fp.shape[0], C_fp.shape[0], device=C_fp.device)

        if self.w_bij > 0:
            bijectivity_loss = self.w_bij * criterion(torch.matmul(C_fp, C_pf), eye)
        else:
            bijectivity_loss = 0.0

        if self.w_orth > 0:
            orthogonality_loss = self.w_orth * criterion(torch.matmul(C_fp, C_fp.t()), eye)
        else:
            orthogonality_loss = 0.0

        return {'l_freq_bij': bijectivity_loss, 'l_freq_orth': orthogonality_loss}
