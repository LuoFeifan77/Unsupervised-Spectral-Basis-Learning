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


@MODEL_REGISTRY.register()
class MWPModel(BaseModel):  #only for testing
    def __init__(self, opt):
        self.partial = opt.get('partial', False)
        self.train_desc_path = opt.get('train_desc_path', None)
        self.test_desc_path = opt.get('test_desc_path', None)
        super(MWPModel, self).__init__(opt)

    def validate_single(self, data, timer):

        #1 get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # start record
        timer.start()

        if self.partial:
        # SHOT descriptors
            data_x['desc'] = sio.loadmat(osp.join(self.test_desc_path,'null','SHOT_V7', ''.join(data_x['name'])+'.mat'))['desc'].astype(np.float32)
            data_y['desc'] = sio.loadmat(osp.join(self.test_desc_path, self.key_words,'SHOT_V7', ''.join(data_y['name'])+'.mat'))['desc'].astype(np.float32)
            
            # get spectral operators, truncate_eigensystem 
            evals_x, evecs_x, evals_y, evecs_y =self.truncate_eigensystem(data_x['evals'], data_x['evecs'], data_y['evals'], data_y['evecs']) # truncated 
            evecs_trans_x = evecs_x.transpose(-2, -1)* data_x['mass'] 
            evecs_trans_y = evecs_y.transpose(-2, -1)* data_y['mass'] #换成新的计算！
            evecs_trans_x = evecs_trans_x.squeeze() #[K1, NX]
            evecs_trans_y = evecs_trans_y.squeeze() #[K2, Ny]
            evecs_x = evecs_x.squeeze()
            evecs_y = evecs_y.squeeze()
        
        else:
            # for other datasets: topo, SMAL, DT4D  
            data_x['desc'] = sio.loadmat(osp.join(self.test_desc_path, ''.join(data_x['name'])+'.mat'))['desc'].astype(np.float32)
            data_y['desc'] = sio.loadmat(osp.join(self.test_desc_path, ''.join(data_y['name'])+'.mat'))['desc'].astype(np.float32) 
            evals_x = data_x['evals']  # number of eigensystems == 128
            evals_y = data_y['evals']
            evecs_x = data_x['evecs'].squeeze()
            evecs_y = data_y['evecs'].squeeze()
            evecs_trans_x = data_x['evecs_trans'].squeeze()
            evecs_trans_y = data_y['evecs_trans'].squeeze()

        # 1 obtain descriptors
        feat_x =  torch.from_numpy(data_x['desc']).unsqueeze(0)   # from numpy to torch
        feat_y =  torch.from_numpy(data_y['desc']).unsqueeze(0)


        #2 obtian wavelet filter functions
        evals_x_cpu = evals_x.cpu().numpy()
        evals_y_cpu = evals_y.cpu().numpy()

        wavelet_gs_x = Meyer(max(evals_x_cpu[0]), Nf =6)(evals_x_cpu[0])  # evals_x[0] : [,K]; evals_x[1, K]
        wavelet_gs_y = Meyer(max(evals_y_cpu[0]), Nf =6)(evals_y_cpu[0])  # 

        gs_x = wavelet_gs_x.to(self.device)  # numpy to torch
        gs_y = wavelet_gs_y.to(self.device)

        gs_x = gs_x.unsqueeze(0)  #[1,Nf,K]
        gs_y = gs_y.unsqueeze(0)

        p2p = nn_query(feat_x, feat_y).squeeze() # nearest neighbour query

        #3 using nnsearch to compute functional maps
        for _ in range(5):
            Cxy_est = evecs_trans_y @ evecs_x[p2p]  # [K, K]

            #4 MCFP for filtering
            Cxy_filtering = self.MCFP(gs_y, gs_x, Cxy_est.unsqueeze(0))  # [1, K ,K]
            Cxy_filtering = Cxy_filtering.squeeze()  #[K, K]

            #5 convert functional map to point-to-point map
            p2p = fmap2pointmap(Cxy_filtering, evecs_x, evecs_y)

        #6 compute Pyx from functional map
        Pyx = evecs_y @ Cxy_filtering @ evecs_trans_x

        # finish record
        timer.record()

        # return
        return p2p, Pyx, Cxy_filtering, gs_x, gs_y

    
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


    @torch.no_grad()  # 需要修改
    def validation(self, dataloader, tb_logger, update=True): # 进行验证
        # change permutation prediction status

        # pass

        # if 'permutation' in self.networks:
        #     self.networks['permutation'].hard = True
        # if 'fmap_net' in self.networks:
        #     self.networks['fmap_net'].bidirectional = False

        super(MWPModel, self).validation(dataloader, tb_logger, update)  # 调用方法

        # if 'permutation' in self.networks:
        #     self.networks['permutation'].hard = False
        # if 'fmap_net' in self.networks:
        #     self.networks['fmap_net'].bidirectional = True

