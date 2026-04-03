from pickle import TRUE
from re import I
import torch
import torch.nn.functional as F
from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, fmap2pointmap
from networks.filter_network import Meyer


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
class FAP_Refine_Model(BaseModel):  # Base model
    def __init__(self, opt):
        self.with_refine = opt.get('refine', -1)
        self.partial = opt.get('partial', False)
        self.non_isometric = opt.get('non-isometric', False)
        self.using_nn_search = opt.get('using_nn_search', False)
        self.norm_filter = opt.get('norm_filter', False)
        self.proper = opt.get('proper', False)
        if self.with_refine > 0:
            opt['is_train'] = True
        super(FAP_Refine_Model, self).__init__(opt)

    def feed_data(self, data):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        #1 feature extractor for mesh 
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x['faces'])  # [B, Nx, C]
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y['faces'])  # [B, Ny, C]

        #2 get spectral operators
        evals_x = data_x['evals']
        evals_y = data_y['evals']
        evecs_x = data_x['evecs']
        evecs_y = data_y['evecs']
        evecs_trans_x = data_x['evecs_trans']  # [B, K, Nx]
        evecs_trans_y = data_y['evecs_trans']  # [B, K, Ny]

        # 使用smooth features
        feat_x = torch.bmm(evecs_x, torch.bmm(evecs_trans_x, feat_x))
        feat_y = torch.bmm(evecs_y, torch.bmm(evecs_trans_y, feat_y))

        #3 calculate C_desc by desc preservation！
        Cxy, Cyx = self.networks['fmap_net'](feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

        # self.loss_metrics = self.losses['surfmnet_loss'](Cxy, Cyx, evals_x, evals_y)   # 约束描述符保持！
        Pxy, Pyx = self.compute_permutation_matrix(feat_x, feat_y, bidirectional=True)

        #4 calculate C_filter by MCFP
        basis_x, basis_y = self.networks['conv'](evals_x, evals_y)

        #5 compute filter by Jacobi 
        gs_x, gs_y = self.networks['comb'](basis_x, basis_y)

        #6 作用在特征值上，
        # gs_x = torch.bmm(torch.softmax(gs_x, dim=1), evals_x.unsqueeze(0))   #学习了权重作用在特征值上
        # gs_y = torch.bmm(torch.softmax(gs_y, dim=1), evals_y.unsqueeze(0)) 

        # gs_x = evals_x.unsqueeze(0) * torch.softmax(gs_x, dim=1)    #学习了权重作用在特征值上
        # gs_y = evals_y.unsqueeze(0) * torch.softmax(gs_y, dim=1)  

        #6 将滤波器前一项设置为tau=1， 满足稳定性条件
        # # 应该是对学出来的滤波器进行L2-normalization
        if self.norm_filter:
            a1, _ = torch.max(gs_x, dim=-1)  # max values
            a2, _ = torch.max(gs_y, dim=-1)

            b1, _ = torch.min(gs_x, dim=-1)  # min values
            b2, _ = torch.min(gs_y, dim=-1)

            gs_x = (gs_x-b1.unsqueeze(-1))/(a1.unsqueeze(-1) - b1.unsqueeze(-1))  # 缩放到 [0,1] 
            gs_y = (gs_y-b2.unsqueeze(-1))/(a2.unsqueeze(-1) - b2.unsqueeze(-1)) 

        #6 non filtering for estimation  skip it
        Cxy_est = torch.bmm(evecs_trans_y, torch.bmm(Pyx, evecs_x))   # [1, K, K]
        Cyx_est = torch.bmm(evecs_trans_x, torch.bmm(Pxy, evecs_y))   # [1, K, K]

        # loss
        self.loss_metrics = self.losses['surfmnet_loss'](Cxy, Cyx, evals_x, evals_y)  # for estimation, bij+orth

        #7 MSFOP  # 直接试一下滤波的结果，在非等距的数据上！
        Cxy_filtering = self.MCFP(gs_y, gs_x, Cxy_est)  # [1, K ,K]
        Cyx_filtering = self.MCFP(gs_x, gs_y, Cyx_est)  # [1, K, K]


        #8.1 frequency awareness couple loss
        self.loss_metrics['l_align'] = self.losses['align_loss'](Cxy, Cxy_filtering)  # key ideas

        #8.2 calculate loss

        if not self.partial:
            if Cyx != None:
                self.loss_metrics['l_align'] += self.losses['align_loss'](Cyx, Cyx_filtering)  #添加双向

        if 'dirichlet_loss' in self.losses:
            Lx, Ly = data_x['L'], data_y['L']  # 这里的loss不对！
            verts_x, verts_y = data_x['verts'], data_y['verts']
            self.loss_metrics['l_d'] = self.losses['dirichlet_loss'](torch.bmm(Pxy, verts_y), Lx) + \
                                       self.losses['dirichlet_loss'](torch.bmm(Pyx, verts_x), Ly)


    def validate_single(self, data, timer):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # get previous network state dict
        if self.with_refine > 0:
            state_dict = {'networks': self._get_networks_state_dict()}

        # start record
        timer.start()

        # test-time refinement
        if self.with_refine > 0:
            self.refine(data)

        #1 feature extractor
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x.get('faces')) #[1, Nx, D]
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y.get('faces')) #[1, Ny, D]

        #2 get spectral operators
        evals_x = data_x['evals']  #[1, K]
        evecs_x = data_x['evecs'].squeeze() 
        evecs_trans_x = data_x['evecs_trans'].squeeze() # [1, K, Nx]
       
        evals_y = (data_y['evals']) 
        evecs_y = (data_y['evecs']).squeeze() # [1, K, Nx]
        evecs_trans_y = (data_y['evecs_trans']).squeeze()  # [1, K, Ny]

        # 使用smooth features
        # feat_x = torch.bmm(evecs_x.unsqueeze(0), torch.bmm(evecs_trans_x.unsqueeze(0), feat_x))
        # feat_y = torch.bmm(evecs_y.unsqueeze(0), torch.bmm(evecs_trans_y.unsqueeze(0), feat_y))

        basis_x, basis_y = self.networks['conv'](evals_x, evals_y)

        gs_x, gs_y = self.networks['comb'](basis_x, basis_y) # [1, 6, 200]

        # gs_x =  torch.softmax(gs_x, dim=-1)    #学习了权重作用在特征值上
        # gs_y =  torch.softmax(gs_y, dim=-1)  

        # tau = 0.1

        # gs_x = gs_x / tau
        # gs_x = torch.exp(gs_x - (torch.logsumexp(gs_x, dim=-1, keepdim=True)))  # [0,1]

        # gs_y = gs_y / tau
        # gs_y = torch.exp(gs_y - (torch.logsumexp(gs_y, dim=-1, keepdim=True)))

        # 应该是对学出来的滤波器进行L2-normalization
        if self.norm_filter:
            a1, _ = torch.max(gs_x, dim=-1)  # max values
            a2, _ = torch.max(gs_y, dim=-1)

            b1, _ = torch.min(gs_x, dim=-1)  # min values
            b2, _ = torch.min(gs_y, dim=-1)

            gs_x = (gs_x-b1.unsqueeze(-1))/(a1.unsqueeze(-1) - b1.unsqueeze(-1))  # 缩放到 [0,1] 
            gs_y = (gs_y-b2.unsqueeze(-1))/(a2.unsqueeze(-1) - b2.unsqueeze(-1)) 
        
        # using 
        feat_x = F.normalize(feat_x, dim=-1, p=2)  # 仅仅对非等距使用
        feat_y = F.normalize(feat_y, dim=-1, p=2)
        p2p = nn_query(feat_x, feat_y).squeeze() # nearest neighbour query
        Cxy_est = evecs_trans_y @ evecs_x[p2p]   #[K, K]

        # Pyx = self.compute_permutation_matrix(feat_y, feat_x, bidirectional=False).squeeze()
        # Cxy_est = evecs_trans_y @ (Pyx @ evecs_x)

        # Cxy_est = torch.bmm(evecs_trans_y, torch.bmm(Pyx, evecs_x))  # [1, K, K]

        # 换一种思路
        if self.non_isometric:   # non_isometric matching 迭代一次即可
            iter_num = 1
        else :                   # near_isometric matching 迭代5次结果更好
            iter_num = 5      
            
        # using learned filter functions to refine
        for _ in range(iter_num):  # 迭代几次

            Cxy_est = self.MCFP(gs_y, gs_x, Cxy_est.unsqueeze(0)).squeeze()  # [K ,K], using learned filters
            # convert functional map to point-to-point map
            # 使用nnsearch
            # if self.proper:  # 看看这一步有没有效果
            #     pmap21_soft = (evecs_y @ Cxy_est) @ evecs_x.t()
            #     pmap21_soft = torch.softmax(pmap21_soft / 0.07, dim=-1)
            #     Cxy_est = evecs_trans_y @ pmap21_soft @ evecs_x

            p2p = nn_query(evecs_x@Cxy_est.t(), evecs_y).squeeze()   # p2pyx, 使用nnsearch

            Cxy_est = evecs_trans_y @ evecs_x[p2p]  #[K, K]    # recover Cxy_est

        Pyx = evecs_y @ Cxy_est @ evecs_trans_x  #置换矩阵

# 添加一层proper 是不是更好一些呢？ 并不好
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

        C_new=C_new*(1/gs_y2)   # 后面对滤波器直接正则化后，不需要除以了！，gs_x, gs_y 按L2
    
        return C_new  # [1, K, K]


    def refine(self, data):  # optimal parameters
        # pass 
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
        super(FAP_Refine_Model, self).validation(dataloader, tb_logger, update)
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = False
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = True
