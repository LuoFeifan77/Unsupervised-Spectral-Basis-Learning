
import os
import sys
import numpy as np
import torch as th
import torch.nn as nn
import pykeops
import time

from einops import rearrange
from tqdm.auto import tqdm
from pathlib import Path
from utils.registry import NETWORK_REGISTRY
import torch.nn.functional as F
from pykeops.torch import Genred


# 这个路径需要调整一下
SOURCE_DIR = Path(os.path.abspath(__file__)).parents[1]
sys.path.append(os.path.join(SOURCE_DIR, 'utils', 'ProjectionUtils'))   # 路径没有导入进来
from projection_utils import nn_query_precise   # 实际上这里没问题


# 将这里搭建成网络, 这个需要训练
@NETWORK_REGISTRY.register()   # 完成初始化和训练
class KernelZoomOut(nn.Module):
    def __init__(self, k_init=20, nit=2, step=10, blur=1e-2, init_blur=1, normalize=False,
                 nn_only=False, precise=False, n_inner=1):#, simple_init=False, init_blur=None):
        super(KernelZoomOut, self).__init__()

        self.nit = nit
        self.k_init = k_init
        self.step = step
        self.n_inner = n_inner

        # self.blur = blur
        self.register_buffer('blur', th.Tensor([blur]))
        self.normalize=normalize

        # self.init_blur = init_blur
        self.register_buffer('init_blur', th.Tensor([init_blur]))    # 这里写这么复杂的意义在哪里
        self.nn_only=nn_only
        self.precise=precise

    @property
    def k_final(self):
        return self.k_init + self.nit * self.step

    def compute_C12(self, T21, k, evects1, evects2, mass2):
        """
        evects1 : (N1, K1) or (B,N1, K1)
        evects2 : (N2, K2) or (B,N2, K2)
        mass2   : (N2) or (B,N2)
        """
        # Linear algebra
        if np.issubdtype(type(k), np.integer):
            k1 = k
            k2 = k
        else:
            k1, k2 = k

        evects1_pb = T21.pull_back(evects1[:, :k1]) # ([B], N2, K1)

        return evects2[:,:k2].mT @ (mass2[..., None] * evects1_pb)

    def compute_T21(self, C12, evects1, evects2, blur=None, faces1=None):  # 
        k2, k1 = C12.shape

        emb1 = evects1[:, :k1] @ C12.mT / k2    # 这种emb 意义在哪里？
        emb2 = evects2[:, :k2] / k2

        emb1 = emb1.contiguous()  # 连续的，邻近的
        emb2 = emb2.contiguous()

        blur = self.blur if blur is None else blur

        # 这写开关也太丰富了. 直接使用这个？
        if self.nn_only:
            if self.precise:
                T21 = EmbPreciseMap(emb1, emb2, faces1)
            else:
                T21 = EmbP2PMap(emb1, emb2)
        else:
            T21 = KernelDistMap(emb1, emb2, blur=blur.item(), normalize=self.normalize)  # 这种方式，计算出来的，为什么呢？

        return T21

    def compute_init(self, F1, F2, faces1=None):
        if self.nn_only:
            if self.precise:
                T21 = EmbPreciseMap(F1, F2, faces1)
            else:
                T21 = EmbP2PMap(F1, F2)
        else:
            T21 = KernelDistMap(F1, F2, blur=self.init_blur.item(), normalize=self.normalize)

        return T21

    # def forward(self, F1, F2, evects1, evects2, mass2, return_T21=True, return_init=False, faces1=None):
    def forward(self, F1, F2, evects1, evects2, mass1, mass2, return_T=False, return_init=False, faces1=None, faces2=None):
        # print(F1.shape, F2.shape, evects1.shape, evects2.shape, mass2.shape)

        # print(f' F1: {F1} \n F2: {F2} \n evects1: {evects1} \n evects2: {evects2} \n mass2: {mass2}')
        # print(f'F1 : {F1.is_contiguous()} \n F2 : {F2.is_contiguous()} \n evects1 : {evects1.is_contiguous()} \n evects2 : {evects2.is_contiguous()} \n mass2 : {mass2.is_contiguous()}')
        if self.nn_only:
            if self.precise:
                T21 = EmbPreciseMap(F1, F2, faces1)   # 这篇论文有很多tips
                T12 = EmbPreciseMap(F2, F1, faces2)
            else:
                T21 = EmbP2PMap(F1, F2)
                T12 = EmbP2PMap(F2, F1)  # 把这个也放上
        else:
            T21 = KernelDistMap(F1, F2, blur=self.init_blur.item(), normalize=self.normalize)
            T12 = KernelDistMap(F2, F1, blur=self.init_blur.item(), normalize=self.normalize) # 

            # 这里用softmax 用下
            # T12, T21 = compute_permutation_matrix(F1.unsqueeze(0), F2.unsqueeze(0), bidirectional=True)  # 使用softmax来计算看看
            # T12, T21 = T12.squeeze(), T21.squeeze()

        # 计算初始的functional maps C12 & C21
        k_curr = self.k_init

        C12 = self.compute_C12(T21, k_curr, evects1, evects2, mass2)
        C21 = self.compute_C12(T12, k_curr, evects2, evects1, mass1)   # 计算双向的

        # if return_init:
        #     C12_init = C12
        #     C21_init = C21

        # C12_list.append(C12)

        # for i in tqdm(range(self.nit), leave=False):
        for i in range(self.nit):

            k_curr = k_curr + self.step
            for _ in range(self.n_inner):   # 这里有refine 过程
                T21 = self.compute_T21(C12, evects1, evects2, faces1=faces1)
                T12 = self.compute_T21(C21, evects2, evects1, faces1=faces2) # 计算另外的逐点映射

                C12 = self.compute_C12(T21, k_curr, evects1, evects2, mass2) 
                C21 = self.compute_C12(T12, k_curr, evects2, evects1, mass1)

            # C12_list.append(C12)

        # if not return_T21:
        #     if return_init:
        #         return [C12_init, C12]
        #     return C12

        # 更新逐点映射
        if return_T: 
            T21 = self.compute_T21(C12, evects1, evects2, faces1=faces1) 
            T12 = self.compute_T21(C21, evects2, evects1, faces1=faces2) 
            return C12, C21, T12, T21  #返回逐点映射
        
        else:
            return C12, C21  

        # if return_init:
        #     return [C12_init, C12], T21

        

def compute_permutation_matrix(feat_x, feat_y, bidirectional=False, normalize=True):
    if normalize:
        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y, dim=-1, p=2)
    similarity = th.bmm(feat_x, feat_y.transpose(1, 2))

    # sinkhorn normalization
    Pxy = F_Similarity(similarity)

    if bidirectional:
        Pyx = F_Similarity(similarity.transpose(1, 2))
        return Pxy, Pyx
    else:
        return Pxy


# class Similarity(nn.Module):
#     def __init__(self, normalise_dim=-1, tau=0.07, hard=False):
#         super(Similarity, self).__init__()
#         self.dim = normalise_dim
#         self.tau = tau
#         self.hard = hard

#     def forward(self, log_alpha):
#         log_alpha = log_alpha / self.tau
#         alpha = th.exp(log_alpha - (th.logsumexp(log_alpha, dim=self.dim, keepdim=True)))  # -1的正则化
#         if self.hard:
#             # Straight through.
#             index = alpha.max(self.dim, keepdim=True)[1]
#             alpha_hard = th.zeros_like(alpha, memory_format=th.legacy_contiguous_format).scatter_(self.dim, index, 1.0)
#             ret = alpha_hard - alpha.detach() + alpha   # 这一步难道用上呢？
#         else:
#             ret = alpha
#         return ret


def F_Similarity(log_alpha, normalise_dim=-1, tau=0.07, hard=False):

    log_alpha = log_alpha / tau
    alpha = th.exp(log_alpha - (th.logsumexp(log_alpha, dim=normalise_dim, keepdim=True)))  # -1的正则化
    if hard:
        # Straight through.
        index = alpha.max(normalise_dim, keepdim=True)[1]
        alpha_hard = th.zeros_like(alpha, memory_format=th.legacy_contiguous_format).scatter_(normalise_dim, index, 1.0)
        ret = alpha_hard - alpha.detach() + alpha   # 难道别人这里已经用上呢？
    else:
        ret = alpha
    return ret




def nn_query(X, Y):  # 这个是可微分的吗
    formula = pykeops.torch.Genred('SqDist(X,Y)',
                    [f'X = Vi({X.shape[-1]})',          # First arg  is a parameter,    of dim 1
                    f'Y = Vj({Y.shape[-1]})',          # Second arg is indexed by "i", of dim
                    ],
                    reduction_op='ArgMin',
                    axis=0)

    return formula(X, Y).squeeze(-1)


class PointWiseMap:
    def __init__(self):
        pass

    def pull_back(self, f):
        pass

    def get_nn(self):
        pass

class P2PMap(PointWiseMap):  # 这个到后续的test有妙用
    """
    Point to point map, as an array or tensor of shape (n2,)
    """
    def __init__(self, p2p_21, n1=None):
        super().__init__()

        assert p2p_21.ndim == 1, "p2p should only have one dimension"
        self.p2p_21 = p2p_21  # (n2, )
        self.n2 = self.p2p_21.shape[0]
        self.n1 = n1

        self.max_ind = self.p2p.max() if n1 is None else n1-1

    def pull_back(self, f):
        """
        Pull back f using the map T.

        Parameters:
        ------------------
        f : (N1,), (N1, p) or (B, N, p)

        Output
        -------------------
        pull_back : (N2, p)  or (B, N2, p)
        """
        if f.shape[0] <= self.max_ind:
            raise ValueError(f'Function f doesn\'t have enough entries, need at least {1+self.max_ind} but only has {f.shape[0]}')

        if f.ndim == 1 or f.ndim == 2:
            f_pb = f[self.p2p_21]  # (n2, k)
        elif f.ndim == 3:
            f_pb = f[:, self.p2p_21]  # (B, n2, k)
        else:
            raise ValueError('Function is only dim 1, 2 or 3')

        return f_pb

    def get_nn(self):
        return self.p2p_21

class PreciseMap(PointWiseMap):
    """
    Point to barycentric map, using vertex to face and barycentric coordinates.
    """
    def __init__(self, v2face_21, bary_coords, faces1):
        super().__init__()

        # assert P21.ndim == 2, "Precise map should only have two dimension"

        self.v2face_21 = v2face_21
        self.bary_coords = bary_coords  # (N2, 3)
        self.faces1 = faces1

        self.n2 = self.v2face_21.shape[0]
        self.n1 = self.faces1.max()+1

        self._nn_map = None

    def pull_back(self, f):
        """
        Pull back f using the map T.

        Parameters:
        ------------------
        f : (N1,), (N1, p) or (B, N, p)

        Output
        -------------------
        pull_back : (N2, p)  or (B, N2, p)
        """

        # f_pb = self.P21 @ f
        if f.ndim == 1 or f.ndim == 2:
            f_selected = f[self.faces1[self.v2face_21]]  # (N2, 3, p) or (N2, 3)
            # print('Selected', f_selected.max(), self.bary_coords.sum(1).max())
            if f.ndim == 1:
                f_pb = (self.bary_coords * f_selected).sum(1)
            else:
                f_pb = (self.bary_coords.unsqueeze(-1) * f_selected).sum(1)
                # print('Selected2', f_pb.max())

        elif f.ndim == 3:
            f_selected = f[: self.faces1[self.v2face_21]]  # (B, N2, 3, p)
            f_pb = (self.bary_coords.unsqueeze(0).unsqueeze(-1) * f_selected).sum(1)

        return f_pb

    def get_nn(self):
        if self._nn_map is None:
            self._nn_map = th.take_along_dim(self.faces1[self.v2face_21],
                                             self.bary_coords.argmax(1, keepdims=True),
                                             1).squeeze(-1)
            # self._nn_map = nn_query(self.emb1, self.emb2)

        return self._nn_map
        # return self.P21

class EmbP2PMap(P2PMap):
    """
    Point to point map, computed from embeddings.
    """
    def __init__(self, emb1, emb2):
        self.emb1 = emb1.contiguous()  # (N1, K)
        self.emb2 = emb2.contiguous()  # (N2, K)

        p2p_21 = nn_query(self.emb1, self.emb2)  # 这样的数值是不可微分的
        super().__init__(p2p_21, n1=self.emb1.shape[-2])


class EmbPreciseMap(PreciseMap):
    """
    Point to barycentric map, computed from embeddings.
    """
    def __init__(self, emb1, emb2, faces1, clear_cache=True):
        self.emb1 = emb1.contiguous()  # (N1, K)
        self.emb2 = emb2.contiguous()  # (N2, K)

        v2face_21, bary_coords = nn_query_precise(self.emb1, faces1, self.emb2, return_dist=False, batch_size=min(2000, emb2.shape[0]), clear_cache=clear_cache)

        # th.cuda.empty_cache()

        super().__init__(v2face_21, bary_coords, faces1)  # 可微分吗


class KernelDistMapOld(PointWiseMap):
    """
    Map of the the shape exp(- ||X_i - Y_j||_2^2 / blur**2)). Normalized per row.
    """
    def __init__(self, emb1, emb2, normalize=False, blur=None):
        self.emb1 = emb1.contiguous()  # (N1, K)
        self.emb2 = emb2.contiguous()  # (N2, K)


        self.blur = th.ones(1, device=self.emb1.device)

        if blur is not None:
            self.blur = self.blur * blur

        if normalize:
            with th.no_grad():
                self.blur = self.blur * th.sqrt(self.get_maxnorm())

        self.n1 = self.emb1.shape[-2]
        self.n2 = self.emb2.shape[-2]

        # self.batched1 = self.emb1.ndim == 3
        # self.batched2 = self.emb2.ndim == 3

        self.pull_back_formula = self.get_pull_back_formula()
        self._nn_map = None

    def get_maxnorm(self):
        formula = pykeops.torch.Genred('SqDist(X,Y)',
                    [f'X = Vi({self.emb1.shape[-1]})',          # First arg  is a parameter,    of dim 1
                    f'Y = Vj({self.emb2.shape[-1]})',          # Second arg is indexed by "i", of dim
                    ],
                    reduction_op='Max',
                    axis=0)

        max_dist = formula(self.emb1, self.emb2).max()

        return max_dist.squeeze()

    def get_pull_back_formula(self):
        """
        B, N1, 1 -> B, N2, 1
        """

        f = pykeops.torch.Vj(0, 1)  # (B, 1, N1, p)
        emb1_j = pykeops.torch.Vj(1, self.emb1.shape[1])  # (1, 1, N1, K)
        emb2_i = pykeops.torch.Vi(2, self.emb1.shape[1])  # (1, N2, 1, K)
        sqblur = pykeops.torch.Pm(3, 1)  # (B, 1)

        dist = -emb2_i.sqdist(emb1_j) / sqblur  # (B, N2, N1)

        return dist.sumsoftmaxweight(f, axis=1)


    def pull_back(self, f):
        """
        Pull back f using the map T.

        Parameters:
        ------------------
        f : (N1,), (N1, p) or (B, N, p)

        Output
        -------------------
        pull_back : (N2, p)  or (B, N2, p)
        """

        n_func = f.shape[-1] if f.ndim > 1 else 1

        sqblur = 2*th.square(self.blur)

        # print(self.emb2.shape, self.emb1.shape, f.shape)
        # test = ((self.emb2.unsqueeze(1) * self.emb1.unsqueeze(0)).sum(-1).unsqueeze(-1) * f.unsqueeze(0)).sum(1)
        # print(test.shape, f.shape, self.emb2.shape)
        # return test

        if f.ndim == 1:
            f_pb = self.pull_back_formula(f.unsqueeze(-1), self.emb1, self.emb2, sqblur).squeeze(-1)  # (N2, )

        elif f.ndim == 2:
            f_input = f.transpose(0,1).contiguous()  # (p, N)
            f_pb = self.pull_back_formula(f_input, self.emb1.unsqueeze(0), self.emb2.unsqueeze(0), sqblur).squeeze(-1)  # (p, N2)
            f_pb = f_pb.transpose(0,1)  # (N2, p)

        elif f.ndim == 3:
            f_input = rearrange(f, 'B N p -> (B p) N').contiguous()
            f_pb = self.pull_back_formula(f_input, self.emb1.unsqueeze(0), self.emb2.unsqueeze(0), sqblur).squeeze(-1)  # (Bp, N2)
            f_pb = rearrange(f_pb, '(B p) N -> B N p', p=n_func)
        else:
            raise ValueError('Function is only dim 1, 2 or 3')

        return f_pb

    def get_nn(self):
        if self._nn_map is None:
            self._nn_map = nn_query(self.emb1, self.emb2)

        return self._nn_map


class KernelDistMap(PointWiseMap):  # 这个才是目前最有效的，替换softmax
    """
    Map of the the shape exp(- ||X_i - Y_j||_2^2 / blur**2)). Normalized per row.
    """
    def __init__(self, emb1, emb2, blur=None, normalize=False):  #  把位置写反了
        self.emb1 = emb1.contiguous()  # (N1, K)
        self.emb2 = emb2.contiguous()  # (N2, K)

        self.blur = th.ones(1, device=self.emb1.device)
        #print(self.blur, blur)
        if blur is not None:
            self.blur = self.blur * blur

        if normalize:
            with th.no_grad():
                self.blur = self.blur * th.sqrt(self.get_maxnorm())

        self.n1 = self.emb1.shape[-2]
        self.n2 = self.emb2.shape[-2]

        # self.batched1 = self.emb1.ndim == 3
        # self.batched2 = self.emb2.ndim == 3

        # self.pull_back_formula = self.get_pull_back_formula()
        self._nn_map = None

    def get_maxnorm(self):  # 应该这里少了函数
        formula = pykeops.torch.Genred(
            "SqDist(X,Y)",
            [
                f"X = Vi({self.emb1.shape[-1]})",  # First arg  is a parameter,    of dim 1
                f"Y = Vj({self.emb2.shape[-1]})",  # Second arg is indexed by "i", of dim
            ],
            reduction_op="Max",
            axis=0,
        )

        max_dist = formula(self.emb1, self.emb2).max()   # 

        return max_dist.squeeze()

    def get_pull_back_formula(self, dim):
        """
        B, N1, 1 -> B, N2, 1
        """

        f = pykeops.torch.Vj(0, dim)  # (B, 1, N1, p)
        emb1_j = pykeops.torch.Vj(1, self.emb1.shape[1])  # (1, 1, N1, K)
        emb2_i = pykeops.torch.Vi(2, self.emb1.shape[1])  # (1, N2, 1, K)
        sqblur = pykeops.torch.Pm(3, 1)  # (B, 1)

        dist = -emb2_i.sqdist(emb1_j) / sqblur  # (B, N2, N1)

        return dist.sumsoftmaxweight(f, axis=1)  # (B, N2, p)


    def pull_back(self, f):
        """
        Pull back f using the map T.

        Parameters:
        ------------------
        f : (N1,), (N1, p) or (B, N, p)

        Output
        -------------------
        pull_back : (N2, p)  or (B, N2, p)
        """

        n_func = f.shape[-1] if f.ndim > 1 else 1
        pull_back_formula = self.get_pull_back_formula(n_func)

        sqblur = 2*th.square(self.blur)

        # print(self.emb2.shape, self.emb1.shape, f.shape)
        # test = ((self.emb2.unsqueeze(1) * self.emb1.unsqueeze(0)).sum(-1).unsqueeze(-1) * f.unsqueeze(0)).sum(1)
        # print(test.shape, f.shape, self.emb2.shape)
        # return test

        if f.ndim == 1:
            f_pb = pull_back_formula(f.unsqueeze(-1), self.emb1, self.emb2, sqblur).squeeze(-1)  # (N2, )

        elif f.ndim == 2:
            f_input = f.contiguous()  # (p, N)
            # print(f'f {f.is_contiguous()}, emb1 {self.emb1.is_contiguous()}, emb2 {self.emb2.is_contiguous()}')
            f_pb = pull_back_formula(f_input, self.emb1, self.emb2, sqblur) # (N2, p)

            # exit(-1)
            # f_pb = f_pb.transpose(0,1)  # (N2, p)

        elif f.ndim == 3:
            # f_input = rearrange(f, 'B N p -> (B p) N').contiguous()
            f_pb = pull_back_formula(f, self.emb1.unsqueeze(0), self.emb2.unsqueeze(0), sqblur) # (B, N2, p)
            # f_pb = rearrange(f_pb, '(B p) N -> B N p', p=n_func)
        else:
            raise ValueError('Function is only dim 1, 2 or 3')

        return f_pb

    def get_nn(self):
        if self._nn_map is None:
            self._nn_map = nn_query(self.emb1, self.emb2)

        return self._nn_map

