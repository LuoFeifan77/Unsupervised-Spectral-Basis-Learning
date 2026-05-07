import torch
import hashlib
import numpy as np
import scipy
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import gamma as gammaFunc
from torch import Tensor
from utils.registry import NETWORK_REGISTRY

EPS=1e-7



# heat kernel 
@NETWORK_REGISTRY.register()
class Heat_Kernel_Fliters(torch.nn.Module):
    def __init__(self, 
                C_inout = 200) :
        super().__init__()
        self.C_inout = C_inout  # 与特征值的维度一致
        # self.device = device # do not need 
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  #  # 扩散的时间
        nn.init.constant_(self.diffusion_time, 0.0)

    def forward(self, evals_x, evals_y, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y):   # 为什么没有计算呢
        
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)
        
        # Diffuse
        time = self.diffusion_time

        filter_x = torch.exp(-evals_x * time)  # 就这个就
        filter_y = torch.exp(-evals_y * time)

        inv_filter_x = torch.exp(evals_x * time) 
        inv_filter_y = torch.exp(evals_y * time)

        evecs_x = evecs_x * filter_x
        evecs_y = evecs_y * filter_y

        evecs_trans_x =  evecs_trans_x * inv_filter_x.unsqueeze(-1)
        evecs_trans_y =  evecs_trans_y * inv_filter_y.unsqueeze(-1)

        return filter_x, filter_y, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y


# For EigenBasis Learning without eigenvalues
@NETWORK_REGISTRY.register()
class EigenBasis_Fliters(torch.nn.Module):
    def __init__(self, 
                C_inout = 200,
                tau = 1,
                filter_type ='heat') :
        super().__init__()
        self.C_inout = C_inout  # 与特征值的维度一致
        self.tau = tau
        self.filter_type = filter_type  # 选择不同的滤波器
        # self.nn = torch.nn.Linear(200, 200)
        if self.filter_type =='none':
            self.diffusion_time = torch.ones(C_inout)
        else:
            self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  #  # 扩散的时间
            nn.init.constant_(self.diffusion_time, 0.0)  # 原来是这样定义的

    def forward(self, evals_x, evals_y, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y):   # 为什么没有计算呢
        
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)  # 设置最小值为0

            #inhibition function
            gs_x = torch.exp(-self.diffusion_time)  
            gs_y = torch.exp(-self.diffusion_time)   
       
            gs_x_inv = torch.exp(self.diffusion_time) 
            gs_y_inv = torch.exp(self.diffusion_time) 

        evecs_trans_x_gs = gs_x_inv.unsqueeze(-1) * evecs_trans_x  # utilize inverse 
        evecs_trans_y_gs = gs_y_inv.unsqueeze(-1) * evecs_trans_y  

        evecs_x_gs = evecs_x * gs_x.unsqueeze(0) 
        evecs_y_gs = evecs_y * gs_y.unsqueeze(0) 

        return gs_x, gs_y, evecs_x_gs, evecs_y_gs, evecs_trans_x_gs, evecs_trans_y_gs

