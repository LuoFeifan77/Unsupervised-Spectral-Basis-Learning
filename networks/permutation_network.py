import torch
import torch.nn as nn

from utils.registry import NETWORK_REGISTRY
EPS=1e-5

@NETWORK_REGISTRY.register()
class Similarity(nn.Module):
    def __init__(self, normalise_dim=-1, tau=0.2, hard=False):
        super(Similarity, self).__init__()
        self.dim = normalise_dim
        self.tau = tau
        self.hard = hard

    def forward(self, log_alpha):
        log_alpha = log_alpha / self.tau
        alpha = torch.exp(log_alpha - (torch.logsumexp(log_alpha, dim=self.dim, keepdim=True)))  # -1的正则化
        if self.hard:
            # Straight through.
            index = alpha.max(self.dim, keepdim=True)[1]
            alpha_hard = torch.zeros_like(alpha, memory_format=torch.legacy_contiguous_format).scatter_(self.dim, index, 1.0)
            ret = alpha_hard - alpha.detach() + alpha
        else:
            ret = alpha
        return ret


# @NETWORK_REGISTRY.register()
# class DiffSimilarity(nn.Module):
#     def __init__(self, normalise_dim=-1, tau=1.0, tau_min=1e-4, hard=False):
#         super(DiffSimilarity, self).__init__()
#         self.dim = normalise_dim
#         self.tau_min = tau_min
#         self.hard = hard
#         self.tau = nn.parameter.Parameter(torch.tensor(tau, dtype=torch.float32), requires_grad=True)  # 这个只是一个很小的数值

    # def get_tau(self):
    #     return torch.clamp(self.tau**2, min=self.tau_min)

    # def forward(self, log_alpha):
        
    #     a = self.tau  if isinstance(self.tau, float) else torch.clamp(self.tau**2, min=self.tau_min, max=1.0)   # 
    #     log_alpha = log_alpha / a

    #     alpha = torch.exp(log_alpha - (torch.logsumexp(log_alpha, dim=self.dim, keepdim=True)))  # -1的正则化
    #     if self.hard:
    #         # Straight through.
    #         index = alpha.max(self.dim, keepdim=True)[1]
    #         alpha_hard = torch.zeros_like(alpha, memory_format=torch.legacy_contiguous_format).scatter_(self.dim, index, 1.0)
    #         ret = alpha_hard - alpha.detach() + alpha
    #     else:
    #         ret = alpha
    #     return ret


# class DiffNNSearch(nn.Module):

#     def __init__(self, temp_init=1.0, temp_min=1e-4):
#         super().__init__()
#         self.temp_min = temp_min
#         self.temp = nn.parameter.Parameter(torch.tensor(temp_init, dtype=torch.float32))  # 这个只是一个很小的数值

#     def get_temp(self):
#         return torch.clamp(self.temp**2, min=self.temp_min)

#     def forward(self, feats0, feats1):    # 这个方法也许是个宝贝, 可以用到我的下一个方法中
#         dists = pdists(feats0, feats1, squared=True)
#         dists = torch.softmax(-dists / self.get_temp(), dim=-1)
#         _, indices = torch.max(dists, dim=-1, keepdim=True)
#         if self.training:
#             asgn_diff = dists
#         else:
#             asgn = torch.zeros_like(dists).scatter_(dim=-1, index=indices, value=1.0)
#             asgn_diff = asgn - dists.detach() + dists
#         return asgn_diff, torch.squeeze(indices, dim=-1)






@NETWORK_REGISTRY.register()
class Similarity_sink(nn.Module):
    def __init__(self, normalise_dim=-1, tau=0.2, hard=False):
        super(Similarity_sink, self).__init__()
        self.dim = normalise_dim
        self.tau = tau
        self.hard = hard

    def forward(self, log_alpha):
        log_alpha = log_alpha / self.tau

        # 直接对这里进行改进 行和列 都规范化
        for _ in range(1):
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=-1, keepdim=True))
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=-2, keepdim=True))

        alpha = torch.exp(log_alpha - (torch.logsumexp(log_alpha, dim=self.dim, keepdim=True)))  # -1的正则化

        if self.hard:
            # Straight through.
            index = alpha.max(self.dim, keepdim=True)[1]
            alpha_hard = torch.zeros_like(alpha, memory_format=torch.legacy_contiguous_format).scatter_(self.dim, index, 1.0)
            ret = alpha_hard - alpha.detach() + alpha
        else:
            ret = alpha
        return ret




# 放入sinkhron的方法进来

def sinkhorn(d, sigma=0.1, num_sink=10):
    d = d / d.mean()

    log_p = -d / (2*sigma**2)

    for it in range(num_sink):
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
    log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
    p = torch.exp(log_p)
    log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
    p_adj = torch.exp(log_p).transpose(0, 1)
    
    return p, p_adj
