from pickle import TRUE
from re import I
import torch
import torch.nn.functional as F
from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, fmap2pointmap
from networks.filter_network import Meyer
from networks.attention_network import Attention

@MODEL_REGISTRY.register()
class OnlyLearningBasisModel(BaseModel):  # Base model
    def __init__(self, opt):
        self.with_refine = opt.get('refine', -1)
        self.partial = opt.get('partial', False)
        self.non_isometric = opt.get('non-isometric', False)
        self.using_nn_search = opt.get('using_nn_search', False)
        self.norm_filter = opt.get('norm_filter', False)
        self.sm_feat = opt.get('sm_feat', False)
        if self.with_refine > 0:
            opt['is_train'] = True
        super(OnlyLearningBasisModel, self).__init__(opt)

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

        #2 calculate C_filter by MCFP, obtain basis; 这个函数改一下
        evecs_x_gs, evecs_y_gs, evecs_trans_x_gs, evecs_trans_y_gs, _, _ = self.networks['conv'](evals_x, evals_y, evecs_x, evecs_y, \
                                    evecs_trans_x, evecs_trans_y)  # 使用sigmod function

        # feat_x_sm = torch.bmm(evecs_x, torch.bmm(evecs_trans_x, feat_x))   # 凑成更好的couple. 效果肯定会好一下
        # feat_y_sm = torch.bmm(evecs_y, torch.bmm(evecs_trans_y, feat_y))
        Pxy, Pyx = self.compute_permutation_matrix(feat_x, feat_y, bidirectional=True)  # 应该只是这个使用
        # Pxy, Pyx = self.compute_permutation_matrix(feat_x_sm, feat_y_sm, bidirectional=True)  # 应该只是这个使用

        #### spectral branch & spatial branch
        #3 calculate C_desc by desc preservation！  replace it with learnable basis 
        # using sigmod function to solve the inverse
        Cxy, Cyx = self.networks['fmap_net'](feat_x, feat_y, evals_x, evals_y, evecs_trans_x_gs, evecs_trans_y_gs)
        # resdual term 
        Cxy_est = torch.bmm(evecs_trans_y_gs, torch.bmm(Pyx, evecs_x_gs))   # [1, K, K]
        Cyx_est = torch.bmm(evecs_trans_x_gs, torch.bmm(Pxy, evecs_y_gs))   # [1, K, K]

        # loss
        self.loss_metrics = self.losses['surfmnet_loss'](Cxy, Cyx, evals_x, evals_y)
        self.loss_metrics['l_align'] = self.losses['align_loss'](Cxy, Cxy_est)  # key ideas

        #8.2 calculate loss
        if not self.partial:
            if Cyx != None:
                self.loss_metrics['l_align'] += self.losses['align_loss'](Cyx, Cyx_est)  #添加双向

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
        evecs_x = data_x['evecs']
        evecs_trans_x = data_x['evecs_trans'] # [1, K, Nx]
       
        evals_y = (data_y['evals']) 
        evecs_y = (data_y['evecs'])# [1, K, Nx]
        evecs_trans_y = (data_y['evecs_trans'])  # [1, K, Ny]

        # 使用smooth features
        # if self.sm_feat: 
        feat_x = torch.bmm(evecs_x, torch.bmm(evecs_trans_x, feat_x))
        feat_y = torch.bmm(evecs_y, torch.bmm(evecs_trans_y, feat_y))

        feat_x = F.normalize(feat_x, dim=-1, p=2)  # 仅仅对非等距使用
        feat_y = F.normalize(feat_y, dim=-1, p=2)
        
        evecs_x_gs, evecs_y_gs, evecs_trans_x_gs, evecs_trans_y_gs, gs_x, gs_y = self.networks['conv'](evals_x, evals_y, evecs_x, evecs_y, \
                                    evecs_trans_x, evecs_trans_y) 
        
        # att = self.networks['attention']()
        p2p = nn_query(feat_x, feat_y).squeeze() # nearest neighbour query

        # 换一种思路
        if self.non_isometric:   # non_isometric matching 迭代一次即可
            iter_num = 1
        else :                   # near_isometric matching 迭代5次结果更好
            iter_num = 5      
            
        # using learned filter functions to refine
        for _ in range(iter_num):  # 迭代几次
            Cxy_est = torch.bmm(evecs_trans_y_gs,  evecs_x_gs[:, p2p, :]) 
            p2p = nn_query(torch.bmm(evecs_x_gs, Cxy_est.transpose(1, 2)), evecs_y_gs).squeeze() 

        Pyx = torch.bmm(evecs_y_gs, torch.bmm(Cxy_est, evecs_trans_x_gs))
        # finish record
        timer.record()

        # resume previous network state dict, restar
        if self.with_refine > 0:
            self.resume_model(state_dict, net_only=True, verbose=False)

        # return
        return p2p, Pyx.squeeze(), Cxy_est.squeeze(), gs_x, gs_y


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
    

    def attention_filter(self, p2p, gs_x, gs_y, gs_x_inv, gs_y_inv, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y, att=None):
        
        Cxy_est = 0

        if att is None: 
            for it in range(gs_x_inv.shape[1]):
                # spectral branch
                evecs_trans_y_gs = gs_y_inv[:,it,:].unsqueeze(-1) * evecs_trans_y
                evecs_x_gs = evecs_x * gs_x[:,it,:].unsqueeze(0)  #[1, Nx, k]
                Cxy_est_it = torch.bmm(evecs_trans_y_gs,  evecs_x_gs[:, p2p, :])   # [1, K, K]
                Cxy_est = Cxy_est + Cxy_est_it

        else: 
            for it in range(gs_x_inv.shape[1]):
                # spectral branch
                evecs_trans_y_gs = gs_y_inv[:,it,:].unsqueeze(-1) * evecs_trans_y
                evecs_x_gs = evecs_x * gs_x[:,it,:].unsqueeze(0)  #[1, Nx, k]
                Cxy_est_it = torch.bmm(evecs_trans_y_gs,  evecs_x_gs[:, p2p, :])   # [1, K, K]
                Cxy_est = Cxy_est + att[it] * Cxy_est_it

        p2p = nn_query(torch.bmm(evecs_x, Cxy_est.transpose(1, 2)), evecs_y).squeeze() 

        return p2p, Cxy_est

    
    def attention_filter_only(self, p2p, gs_x, gs_y, gs_x_inv, gs_y_inv, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y, att=None):
        
        Cxy_est = 0
        # spectral branch
        for it in range(gs_x_inv.shape[1]):
            evecs_trans_x_gs = gs_x_inv[:,it,:].unsqueeze(-1) * evecs_trans_x
            evecs_trans_y_gs = gs_y_inv[:,it,:].unsqueeze(-1) * evecs_trans_y
            evecs_x_gs = evecs_x * gs_x[:,it,:].unsqueeze(0)  #[1, Nx, k]
            evecs_y_gs = evecs_y * gs_y[:,it,:].unsqueeze(0)
            Cxy_est_it = torch.bmm(evecs_trans_y_gs,  evecs_x_gs[:, p2p, :])   # [1, K, K]
            Cxy_est = Cxy_est + Cxy_est_it



        p2p = nn_query(torch.bmm(evecs_x_gs, Cxy_est.transpose(1, 2)), evecs_y_gs).squeeze() 
        # p2p = nn_query(torch.bmm(evecs_x, Cxy_est.transpose(1, 2)), evecs_y).squeeze() 
        Pyx = torch.bmm(evecs_y_gs, torch.bmm(Cxy_est, evecs_trans_x_gs))

        return p2p, Pyx, Cxy_est

    def attention_filter_res(self, p2p, evecs_x_gs, evecs_y_gs, evecs_x_gs_r, evecs_y_gs_r, evecs_trans_x_gs, \
                evecs_trans_y_gs, evecs_trans_x_gs_r, evecs_trans_y_gs_r, att=None):

        Cxy_est = torch.bmm(evecs_trans_y_gs,  evecs_x_gs[:, p2p, :])   # [1, K, K]
        Cxy_est += torch.bmm(evecs_trans_y_gs_r,  evecs_x_gs_r[:, p2p, :]) 

        return Cxy_est

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
        super(OnlyLearningBasisModel, self).validation(dataloader, tb_logger, update)
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = False
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = True
