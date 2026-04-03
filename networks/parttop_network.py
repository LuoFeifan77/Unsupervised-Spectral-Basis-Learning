import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.geometry_util import get_all_operators, to_basis, from_basis, compute_hks_autoscale, compute_wks_autoscale, data_augmentation
from utils.registry import NETWORK_REGISTRY

# computer partiality + topology noise

class ResidualBlock(torch.nn.Module):
    """Implement one residual block as presented in FMNet paper."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, out_dim)
        self.bn1 = torch.nn.BatchNorm1d(out_dim)
        self.fc2 = torch.nn.Linear(out_dim, out_dim)
        self.bn2 = torch.nn.BatchNorm1d(out_dim)

        if in_dim != out_dim:
            self.projection = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)  # non implemented in original FMNet paper, suggested in resnet paper
            )
        else:
            self.projection = None

    def forward(self, x):
        x_res = F.relu(self.bn1(self.fc1(x)))
        x_res = self.bn2(self.fc2(x_res))
        if self.projection:
            x = self.projection(x)
        x_res += x
        return F.relu(x_res)


@NETWORK_REGISTRY.register()
class RefineNet(torch.nn.Module):
    """Implement the refine net of FMNet. Take as input hand-crafted descriptors.
       Output learned descriptors well suited to the task of correspondence"""
    def __init__(self, n_residual_blocks=7, in_dim=352):
        super().__init__()
        model = []
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_dim, in_dim)]

        self.model = nn.Sequential(*model)

    # def forward(self, x):
    def forward(self, x):
        """One pass in refine net.

        Arguments:
            x {torch.Tensor} -- input hand-crafted descriptor. Shape: batch-size x num-vertices x num-features

        Returns:
            torch.Tensor -- learned descriptor. Shape: batch-size x num-vertices x num-features
        """


        return self.model(x)



