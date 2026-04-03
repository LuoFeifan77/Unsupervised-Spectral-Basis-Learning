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
class MaskBasisModel(BaseModel):  # Base model
    def __init__(self, opt):
        self.with_refine = opt.get('refine', -1)
        self.partial = opt.get('partial', False)
        self.non_isometric = opt.get('non-isometric', False)
        self.using_nn_search = opt.get('using_nn_search', False)
        self.norm_filter = opt.get('norm_filter', False)
        self.sm_feat = opt.get('sm_feat', False)
        if self.with_refine > 0:
            opt['is_train'] = True
        super(MaskBasisModel, self).__init__(opt)

    def feed_data(self, data):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        #1 feature extractor for mesh, 这里的时间参数是共享的吗？
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x['faces'])  # [B, Nx, C]
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y['faces'])  # [B, Ny, C]

        #2 get spectral operators
        evals_x = data_x['evals']
        evals_y = data_y['evals']
        evecs_x = data_x['evecs']
        evecs_y = data_y['evecs']
        evecs_trans_x = data_x['evecs_trans']  # [B, K, Nx]
        evecs_trans_y = data_y['evecs_trans']  # [B, K, Ny]

        # select eigenfunction & eigenvaule        
        _, _, evecs_x_gs, evecs_y_gs, evecs_trans_x_gs, evecs_trans_y_gs = self.networks['basisconv'](evals_x, evals_y, \
                                    evecs_x, evecs_y, evecs_trans_x, evecs_trans_y)
        
        # 这个地方也要调整
        Cxy, Cyx = self.networks['fmap_net'](feat_x, feat_y, evals_x, evals_y, evecs_trans_x_gs, evecs_trans_y_gs)  # 学习的
        Pxy, Pyx = self.compute_permutation_matrix(feat_x, feat_y, bidirectional=True)  # 应该只是这个使用

        # Pxx = self.compute_permutation_matrix(evecs_x_gs, evecs_x_gs, bidirectional=False)
        # Pyy = self.compute_permutation_matrix(evecs_y_gs, evecs_y_gs, bidirectional=False)

        # new basis下的结果
        Cxy_est = torch.bmm(evecs_trans_y_gs, torch.bmm(Pyx, evecs_x_gs))   # [1, K, K]
        Cyx_est = torch.bmm(evecs_trans_x_gs, torch.bmm(Pxy, evecs_y_gs))   # [1, K, K]

        # Cxx_est = torch.bmm(evecs_trans_x_gs, torch.bmm(Pxx, evecs_x_gs))   # [1, K, K]
        # Cyy_est = torch.bmm(evecs_trans_y_gs, torch.bmm(Pyy, evecs_y_gs))   # [1, K, K]

        # spectral branch 
        self.loss_metrics = self.losses['surfmnet_loss'](Cxy, Cyx, [], [],  evals_x, evals_y)

        # align loss
        self.loss_metrics['l_align'] = self.losses['align_loss'](Cxy, Cxy_est) #
        if not self.partial:
            if Cyx != None:
                self.loss_metrics['l_align'] += self.losses['align_loss'](Cyx, Cyx_est)

        # 按照最新的loss测试一下
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
        # feat_x = torch.bmm(evecs_x, torch.bmm(evecs_trans_x, feat_x))
        # feat_y = torch.bmm(evecs_y, torch.bmm(evecs_trans_y, feat_y))

        # evecs_x_gs, evecs_y_gs, evecs_trans_x_gs, evecs_trans_y_gs = self.networks['basisconv']( evecs_x, evecs_y, \
        #                             evecs_trans_x, evecs_trans_y) 
        
        gs_x, gs_y, evecs_x_gs, evecs_y_gs, evecs_trans_x_gs, evecs_trans_y_gs = self.networks['basisconv'](evals_x, evals_y, \
                                    evecs_x, evecs_y, evecs_trans_x, evecs_trans_y)
        
        evecs_x_gs = evecs_x_gs.squeeze()
        evecs_y_gs = evecs_y_gs.squeeze()
        evecs_trans_x_gs = evecs_trans_x_gs.squeeze()
        evecs_trans_y_gs = evecs_trans_y_gs.squeeze()

        # 最后这里使用zoomout

        if self.non_isometric:
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)

            # nearest neighbour query
            p2p = nn_query(feat_x, feat_y).squeeze()  # 难道用这个要好一些？

            # compute Pyx from functional map
            Cxy_est = evecs_trans_y_gs @ evecs_x_gs[p2p]
            Pyx = evecs_y_gs @ Cxy_est @ evecs_trans_x_gs

            # 这个后面在试一试
            # p2p = fmap2pointmap(Cxy_est, evecs_x_gs, evecs_y_gs)

        else:
            # compute Pxy
            Pyx = self.compute_permutation_matrix(feat_y, feat_x, bidirectional=False).squeeze() # soft correspondence 
            Cxy_est = evecs_trans_y_gs @ (Pyx @ evecs_x_gs)

            # convert functional map to point-to-point map
            p2p = fmap2pointmap(Cxy_est, evecs_x_gs, evecs_y_gs)

            # compute Pyx from functional map
            Pyx = evecs_y_gs @ Cxy_est @ evecs_trans_x_gs


        # using learned filter functions to refine
        # for _ in range(iter_num):  # 迭代几次
        #     Cxy_est = torch.bmm(evecs_trans_y_gs,  evecs_x_gs[:, p2p, :]) 
        #     p2p = nn_query(torch.bmm(evecs_x_gs, Cxy_est.transpose(1, 2)), evecs_y_gs).squeeze() 
        # Pyx = torch.bmm(evecs_y_gs, torch.bmm(Cxy_est, evecs_trans_x_gs))


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
        super(MaskBasisModel, self).validation(dataloader, tb_logger, update)
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = False
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = True
