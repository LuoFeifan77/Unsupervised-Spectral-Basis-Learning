from pickle import TRUE
from re import I
import torch
import torch.nn.functional as F
from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, fmap2pointmap
from networks.filter_network import Meyer
import scipy.io as sio
import os.path as osp
import numpy as np

#compute partiality and topology! 

def sinkhorn_correspondences(emb_x, emb_y):
    d = dist_mat(emb_x, emb_y, False)   # 计算嵌入的距离矩阵, 这里计算的软对应矩阵
    return sinkhorn(d)   #


def dist_mat(x, y, inplace=True):
    d = torch.mm(x, y.transpose(0, 1))
    v_x = torch.sum(x ** 2, 1).unsqueeze(1)
    v_y = torch.sum(y ** 2, 1).unsqueeze(0)
    d *= -2
    if inplace:
        d += v_x
        d += v_y
    else:
        d = d + v_x
        d = d + v_y

    return d


def nn_search(y, x):
    d = dist_mat(x, y)
    return torch.argmin(d, dim=1)


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


def normalize_rows(x: torch, ord=None):
    """
    按行归一化矩阵
    """
    # 计算每行的范数
    norm = torch.linalg.norm(x, ord=ord, axis=1, keepdims=True)  # ord=1 : l1 norm
    # [m,n] = x.shape

    # 每行的元素除以该行的范数
    x_normalized = torch.divide(x, norm)

    return x_normalized


@MODEL_REGISTRY.register()
class SHREC16_FAPModel(BaseModel):  # Base model
    def __init__(self, opt):
        self.with_refine = opt.get('refine', -1)
        self.non_isometric = opt.get('non-isometric', False)
        self.partial = opt.get('partial', False)
        self.topo = opt.get('topo', False)
        self.key_words = opt.get('key_words', None)
        self.train_desc_path = opt.get('train_desc_path', None)
        self.test_desc_path = opt.get('test_desc_path', None)
        
        if self.with_refine > 0:
            opt['is_train'] = True
        super(SHREC16_FAPModel, self).__init__(opt)

    def feed_data(self, data):
        #1 get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)
        
        if self.partial:
            # load full shape descriptor:  ../data/SHREC16/cuts/SHOT_V7/
            data_x['desc'] = sio.loadmat(osp.join(self.train_desc_path,'null','SHOT_V7', ''.join(data_x['name'])+'.mat'))['desc'].astype(np.float32)
            data_y['desc'] = sio.loadmat(osp.join(self.train_desc_path, self.key_words,'SHOT_V7', ''.join(data_y['name'])+'.mat'))['desc'].astype(np.float32)
            
            # get spectral operators
            evals_x, evecs_x, evals_y, evecs_y =self.truncate_eigensystem(data_x['evals'], data_x['evecs'], data_y['evals'], data_y['evecs']) # truncated 
            evecs_trans_x = evecs_x.transpose(-2, -1)* data_x['mass'] 
            evecs_trans_y = evecs_y.transpose(-2, -1)* data_y['mass'] #换成新的计算！
        
        elif self.topo:
            # load topo descriptor
            data_x['desc'] = sio.loadmat(osp.join(self.train_desc_path, ''.join(data_x['name'])+'.mat'))['desc'].astype(np.float32)
            data_y['desc'] = sio.loadmat(osp.join(self.train_desc_path, ''.join(data_y['name'])+'.mat'))['desc'].astype(np.float32)
            
            # for topo 
            evals_x = data_x['evals']
            evals_y = data_y['evals']
            evecs_x = data_x['evecs']
            evecs_y = data_y['evecs']
            evecs_trans_x = data_x['evecs_trans']  # [B, K, Nx]
            evecs_trans_y = data_y['evecs_trans']  # [B, K, Ny]


        data_x['desc']= torch.tensor(data_x['desc']).to(self.device)
        data_y['desc']= torch.tensor(data_y['desc']).to(self.device)

        #2 feature extractor for mesh  using fm refineNet
        feat_x = self.networks['feature_extractor'](data_x['desc'])  # [Nx, C]
        feat_y = self.networks['feature_extractor'](data_y['desc'])  # [Ny, C]

        feat_x = feat_x.unsqueeze(0)
        feat_y = feat_y.unsqueeze(0)

        #3 calculate C_desc by desc preservation！
        Cxy, Cyx = self.networks['fmap_net'](feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

        # self.loss_metrics = self.losses['surfmnet_loss'](Cxy, Cyx, evals_x, evals_y)   # 约束描述符保持！
        Pxy, Pyx = self.compute_permutation_matrix(feat_x, feat_y, bidirectional=True)

        #4 calculate C_filter by MCFP
        basis_x, basis_y = self.networks['conv'](evals_x, evals_y)

        # #5 compute filter by Jacobi 
        gs_x, gs_y = self.networks['comb'](basis_x, basis_y)
        
        #6 MSFOP
        Cxy_est = torch.bmm(evecs_trans_y, torch.bmm(Pyx, evecs_x))   # [1, K, K]
        Cyx_est = torch.bmm(evecs_trans_x, torch.bmm(Pxy, evecs_y))   # [1, K, K]

        self.loss_metrics = self.losses['surfmnet_loss'](Cxy, Cyx, evals_x, evals_y)  # for estimation, bij+orth

        #7 MSFOP  # 直接试一下滤波的结果，在非等距的数据上！
        Cxy_filtering = self.MCFP(gs_y, gs_x, Cxy_est)  # [1, K ,K]
        Cyx_filtering = self.MCFP(gs_x, gs_y, Cyx_est)  # [1, K, K]

        #8.1 fmap loss
        self.loss_metrics = self.losses['surfmnet_loss'](Cxy, Cyx, evals_x, evals_y)

        #8.2 frequency awareness couple loss
        self.loss_metrics['l_align'] = self.losses['align_loss'](Cxy, Cxy_filtering)
        
        if not self.partial:
            if Cyx != None:
                self.loss_metrics['l_align'] += self.losses['align_loss'](Cyx, Cyx_filtering)  #添加双向
        
        #8.3 smooth loss
        if 'dirichlet_loss' in self.losses:
            Lx, Ly = data_x['L'], data_y['L']  # 
            verts_x, verts_y = data_x['verts'], data_y['verts']
            self.loss_metrics['l_d'] = self.losses['dirichlet_loss'](torch.bmm(Pxy, verts_y), Lx) + \
                                       self.losses['dirichlet_loss'](torch.bmm(Pyx, verts_x), Ly)


    def validate_single(self, data, timer):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        if self.partial:
            # load full shape descriptor:  ../data/SHREC16/cuts/SHOT_V7/
            data_x['desc'] = sio.loadmat(osp.join(self.test_desc_path,'null','SHOT_V7', ''.join(data_x['name'])+'.mat'))['desc'].astype(np.float32)
            data_y['desc'] = sio.loadmat(osp.join(self.test_desc_path, self.key_words,'SHOT_V7', ''.join(data_y['name'])+'.mat'))['desc'].astype(np.float32)
            
            evals_x, evecs_x, evals_y, evecs_y =self.truncate_eigensystem(data_x['evals'], data_x['evecs'], data_y['evals'], data_y['evecs']) # truncated 
            evecs_trans_x = evecs_x.transpose(-2, -1)* data_x['mass'] 
            evecs_trans_y = evecs_y.transpose(-2, -1)* data_y['mass'] #换成新的计算！

            evecs_trans_x = evecs_trans_x.squeeze() #[K1, NX]
            evecs_trans_y = evecs_trans_y.squeeze() #[K2, Ny]
            evecs_x = evecs_x.squeeze()
            evecs_y = evecs_y.squeeze()


        elif self.topo:
            # load topo descriptor
            data_x['desc'] = sio.loadmat(osp.join(self.test_desc_path, ''.join(data_x['name'])+'.mat'))['desc'].astype(np.float32)
            data_y['desc'] = sio.loadmat(osp.join(self.test_desc_path, ''.join(data_y['name'])+'.mat'))['desc'].astype(np.float32)
            evals_x = data_x['evals']
            evals_y = data_y['evals']
            evecs_trans_x = data_x['evecs_trans'].squeeze()  # [K, Nx]
            evecs_trans_y = data_y['evecs_trans'].squeeze()  # [K, Ny]
            evecs_x = data_x['evecs'].squeeze()
            evecs_y = data_y['evecs'].squeeze()

        data_x['desc']= torch.tensor(data_x['desc']).to(self.device)
        data_y['desc']= torch.tensor(data_y['desc']).to(self.device)

        # get previous network state dict
        if self.with_refine > 0:
            state_dict = {'networks': self._get_networks_state_dict()}

        # start record
        timer.start()

        # test-time refinement
        if self.with_refine > 0:
            self.refine(data)

        #1 feature extractor
        feat_x = self.networks['feature_extractor'](data_x['desc'])  # [Nx, C]
        feat_y = self.networks['feature_extractor'](data_y['desc'])  # [Ny, C]

        feat_x = feat_x.unsqueeze(0) #[1, Nx, C]
        feat_y = feat_y.unsqueeze(0) #[1, Ny, C]

        feat_x = F.normalize(feat_x, dim=-1, p=2)  # 正则化
        feat_y = F.normalize(feat_y, dim=-1, p=2)

        #2 obtian learned filter functions
        basis_x, basis_y = self.networks['conv'](evals_x, evals_y)
        gs_x, gs_y = self.networks['comb'](basis_x, basis_y)

        #3 using nnsearch to compute functional maps
        p2p = nn_query(feat_x, feat_y).squeeze() # nearest neighbour query
        Cxy_est = evecs_trans_y @ evecs_x[p2p] 
        
        for _ in range(5):  # 迭代几次

            Cxy_est = self.MCFP(gs_y, gs_x, Cxy_est.unsqueeze(0)).squeeze()  # [K ,K], using learned filters
            p2p = nn_query(evecs_x@Cxy_est.t(), evecs_y).squeeze()   # p2pyx, 使用nnsearch
            Cxy_est = evecs_trans_y @ evecs_x[p2p]  #[K, K]    # recover Cxy_est

        #6 compute Pyx from functional map
        Pyx = evecs_y @ Cxy_est @ evecs_trans_x

        # finish record
        timer.record()

        # resume previous network state dict, restar
        if self.with_refine > 0:
            self.resume_model(state_dict, net_only=True, verbose=False)

        # return
        return p2p, Pyx, Cxy_est, gs_x, gs_y


    def compute_permutation_matrix(self, feat_x, feat_y, bidirectional=False, normalize=True):
        if normalize:
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)
        similarity = torch.bmm(feat_x, feat_y.transpose(1, 2))

        # sinkhorn normalization
        Pxy = self.networks['permutation'](similarity)

        if bidirectional:
            Pyx = self.networks['permutation'](similarity.transpose(1, 2))
            return Pxy, Pyx
        else:
            return Pxy


    def feat_correspondence(self, feat_x, feat_y, bidirectional=False, normalize=False):
        if normalize:
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)

        feat_x = torch.squeeze(feat_x)
        feat_y = torch.squeeze(feat_y)
        Pxy, Pyx = sinkhorn_correspondences(feat_x, feat_y)
        
        if bidirectional:
            return Pxy, Pyx
        else:
            return Pxy
    
    
    def MCFP(self, gs_x, gs_y, Cyx):
    # input:
    #   gs_x/y: [1, Nf, Kx/Ky]
    #   Cyx : [1, K, K]

        C_new = torch.zeros_like(Cyx)  # 声明成全零元素
        gs_y2 = torch.sum(gs_y**2, dim=1)  # 

        # MWP filters
        Nf = gs_x.size(1)  # 
        for s in range(Nf):
            C_new = C_new + gs_x[:,s,:].t()*Cyx*gs_y[:,s,:]

        C_new=C_new*(1/gs_y2)
    
        return C_new  # [1, K, K]


    def truncate_eigensystem(self, evals_x, evecs_x, evals_y, evecs_y):
        # cut off eigen-system
        # evals [1, K]
        # evecs [1, N, k]
        min_evals = torch.min(evals_x[:, -1], evals_y[:, -1])  # comparing

        idx_1 = torch.sum(evals_x<=min_evals)
        evals_x = evals_x[:, :idx_1]
        evecs_x = evecs_x[:,:,:idx_1]

        idx_2 = torch.sum(evals_y<=min_evals)
        evals_y = evals_y[:,:idx_2]
        evecs_y = evecs_y[:,:,:idx_2]

        return evals_x, evecs_x, evals_y, evecs_y


    def refine(self, data):  # optimal parameters
        
        self.networks['permutation'].hard = False
        self.networks['fmap_net'].bidirectional = True

        with torch.set_grad_enabled(True):
            for _ in range(self.with_refine):
                self.feed_data(data)
                self.optimize_parameters()

        self.networks['permutation'].hard = True
        self.networks['fmap_net'].bidirectional = False



    @torch.no_grad()  # 需要修改
    def validation(self, dataloader, tb_logger, update=True): # 进行验证
        # change permutation prediction status

        if 'permutation' in self.networks:
            self.networks['permutation'].hard = True
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = False

        super(SHREC16_FAPModel, self).validation(dataloader, tb_logger, update)

        if 'permutation' in self.networks:
            self.networks['permutation'].hard = False
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = True

