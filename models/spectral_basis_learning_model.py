from pickle import TRUE
from re import I
import torch
import torch.nn.functional as F
from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, fmap2pointmap
from networks.filter_network import Meyer

# For CVPR2026
@MODEL_REGISTRY.register()
class SpectralBasisLearningModel(BaseModel):  # Base model
    def __init__(self, opt):
        self.with_refine = opt.get('refine', -1)
        self.partial = opt.get('partial', False)
        self.non_isometric = opt.get('non-isometric', False)
        self.G_zoomout = opt.get('G_zoomout', False)
        self.using_nn_search = opt.get('using_nn_search', False)
        self.norm_filter = opt.get('norm_filter', False)
        self.sm_feat = opt.get('sm_feat', False)
        if self.with_refine > 0:
            opt['is_train'] = True
        super(SpectralBasisLearningModel, self).__init__(opt)

    def feed_data(self, data):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        #1 feature extractor for mesh, 
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x['faces'])  # [B, Nx, C]
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y['faces'])  # [B, Ny, C]

        #2 get spectral operators
        evals_x = data_x['evals']
        evals_y = data_y['evals']
        evecs_x = data_x['evecs']
        evecs_y = data_y['evecs']
        evecs_trans_x = data_x['evecs_trans']  # [B, K, Nx]
        evecs_trans_y = data_y['evecs_trans']  # [B, K, Ny]

        #3 generate learned basis  for functional maps systems     
        _, _, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y = self.networks['basisconv']\
            (evecs_x, evecs_y, evecs_trans_x, evecs_trans_y)  # 

        #4  compute pointwise maps and advanced functional maps   
        Pyx = self.compute_permutation_matrix(feat_y, feat_x, bidirectional=False)  # 
        Cxy_est = torch.bmm(evecs_trans_y, torch.bmm(Pyx, evecs_x))   # [1, K, K]

        #5 compute multi-resolution spectral loss
        self.step = 50  
        self.iter_num = 2  
        k_curr = 100  
        mrs_loss = 0
        for _ in range(self.iter_num+1):
            mrs_loss += (torch.linalg.norm(evecs_y[:,:,:k_curr]-torch.bmm(Pyx, torch.bmm(evecs_x[:,:,:k_curr], Cxy_est[:,:k_curr,:k_curr].transpose(-2, -1)))))
            k_curr += self.step  # 100, 150, 200
        self.loss_metrics = self.losses['mrsloss'](mrs_loss) 

    def validate_single(self, data, timer):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # get previous network state dict
        if self.with_refine > 0:
            state_dict = {'networks': self._get_networks_state_dict()}

        # start record
        timer.start()

        #1 feature extractor
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x.get('faces')) #[1, Nx, D]
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y.get('faces')) #[1, Ny, D]

        #2 get spectral operators
        evecs_x = data_x['evecs']
        evecs_trans_x = data_x['evecs_trans'] # [1, K, Nx]
       
        evecs_y = (data_y['evecs'])# [1, K, Nx]
        evecs_trans_y = (data_y['evecs_trans'])  # [1, K, Ny]

        #3 generate learned basis     
        gs_x, gs_y, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y = self.networks['basisconv']\
            (evecs_x, evecs_y, evecs_trans_x, evecs_trans_y)   #

        evecs_x = evecs_x.squeeze()
        evecs_y = evecs_y.squeeze()
        evecs_trans_x = evecs_trans_x.squeeze()
        evecs_trans_y = evecs_trans_y.squeeze()

        #4 recover pointwise map from feature scape or functional map
        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y, dim=-1, p=2)

        
        p2p = nn_query(feat_x, feat_y).squeeze()  # nearest neighbour query

        if self.non_isometric:  # 
            # compute Pyx from functional map
            Cxy_est = evecs_trans_y @ evecs_x[p2p]
            Pyx = evecs_y @ Cxy_est @ evecs_trans_x

        else:
            if self.G_zoomout:  # G-zoomout for shrec19
                self.step = 50  # 
                self.iter_num = 2  
                k_curr = 100  # 
                for _ in range(self.iter_num+1):                
                    Cxy_est = evecs_trans_y[:k_curr, :] @ evecs_x[p2p, :k_curr]
                    p2p = fmap2pointmap(Cxy_est, evecs_x[:, :k_curr], evecs_y[:, :k_curr])  
                    k_curr = k_curr + self.step  
            else:
                iter_num = 5  # for faust,scape topkids
                for _ in range(iter_num):
                    Cxy_est = evecs_trans_y @ evecs_x[p2p]
                    p2p = fmap2pointmap(Cxy_est, evecs_x, evecs_y)
            # compute Pyx from functional map
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

    @torch.no_grad()  # 需要修改
    def validation(self, dataloader, tb_logger, update=True): # 进行验证
        # change permutation prediction status
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = True
        super(SpectralBasisLearningModel, self).validation(dataloader, tb_logger, update)
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = False
