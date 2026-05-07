import torch
import hashlib
import numpy as np
import scipy
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import gamma as gammaFunc
from torch import Tensor
from utils.registry import NETWORK_REGISTRY


## For inbition Learning without eigenvalues
@NETWORK_REGISTRY.register()
class EigenBasis_Fliters(torch.nn.Module):
    def __init__(self, 
                C_inout = 200) :
        super().__init__()
        self.C_inout = C_inout  
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  # diffusion time for heat kernel
        nn.init.constant_(self.diffusion_time, 0.0)  # 

    def forward(self, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y):   # 
        
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)  #

            #inhibition function: heat kernel
            gs_x = torch.exp(-self.diffusion_time)  
            gs_y = torch.exp(-self.diffusion_time)   
       
            gs_x_inv = torch.exp(self.diffusion_time) 
            gs_y_inv = torch.exp(self.diffusion_time) 

        evecs_trans_x_gs = gs_x_inv.unsqueeze(-1) * evecs_trans_x  # utilize inverse 
        evecs_trans_y_gs = gs_y_inv.unsqueeze(-1) * evecs_trans_y  

        evecs_x_gs = evecs_x * gs_x.unsqueeze(0) 
        evecs_y_gs = evecs_y * gs_y.unsqueeze(0) 

        return gs_x, gs_y, evecs_x_gs, evecs_y_gs, evecs_trans_x_gs, evecs_trans_y_gs

