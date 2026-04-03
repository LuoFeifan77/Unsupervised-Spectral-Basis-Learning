import torch
import torch.nn.functional as F
import numpy as np
from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, fmap2pointmap
import scipy.io as sio
import os.path as osp

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

# 构建Rfmnet
@MODEL_REGISTRY.register()
class RfmnetModel(BaseModel):   #
    def __init__(self, opt):
        self.with_refine = opt.get('refine', -1)
        self.partial = opt.get('partial', False)
        self.non_isometric = opt.get('non-isometric', False)
        self.train_desc_path = opt.get('train_desc_path', None)  # 
        self.test_desc_path = opt.get('test_desc_path', None)
        self.opt = opt

        if self.with_refine > 0:
            opt['is_train'] = True
        super(RfmnetModel, self).__init__(opt)

    def feed_data(self, data):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # feature extractor for mesh
        if self.train_desc_path is not None:   # for topology
            
            # load SHOT descriptor
            data_x['desc'] = sio.loadmat(osp.join(self.train_desc_path, ''.join(data_x['name'])+'.mat'))['desc'].astype(np.float32)
            data_y['desc'] = sio.loadmat(osp.join(self.train_desc_path, ''.join(data_y['name'])+'.mat'))['desc'].astype(np.float32) 

            data_x['desc'] = to_device(torch.from_numpy(data_x['desc']), self.device)
            data_y['desc'] = to_device(torch.from_numpy(data_y['desc']), self.device)

            feat_x = self.networks['feature_extractor'](data_x['desc'])  # [ Nx, C]
            feat_y = self.networks['feature_extractor'](data_y['desc'])  # [ Ny, C]

            feat_x = feat_x.unsqueeze(0)  #[1, Nx, C]
            feat_y = feat_y.unsqueeze(0)

        else:  # for non-isometric, 
            feat_x = self.networks['feature_extractor'](data_x['verts'], data_x['faces'])  # [B, Nx, C]
            feat_y = self.networks['feature_extractor'](data_y['verts'], data_y['faces'])  # [B, Ny, C]

        # get spectral operators
        evals_x = data_x['evals']
        evals_y = data_y['evals']
        evecs_x = data_x['evecs']
        evecs_y = data_y['evecs']
        evecs_trans_x = data_x['evecs_trans']  # [B, K, Nx]
        evecs_trans_y = data_y['evecs_trans']  # [B, K, Ny]

        # 计算逐点映射
        _, Pyx = self.compute_permutation_matrix(feat_x, feat_y, bidirectional=True)

        # compute C
        Cxy_est = torch.bmm(evecs_trans_y, torch.bmm(Pyx, evecs_x))  
        # Cyx_est = torch.bmm(evecs_trans_x, torch.bmm(Pxy, evecs_y))


        evals_x_cpu = evals_x.cpu().numpy()
        evals_y_cpu = evals_y.cpu().numpy()

        wavelet_gs_x = Meyer(max(evals_x_cpu[0]), Nf =6)(evals_x_cpu[0])  # evals_x[0] : [,K]; evals_x[1, K]
        wavelet_gs_y = Meyer(max(evals_y_cpu[0]), Nf =6)(evals_y_cpu[0])  # 

        gs_x = wavelet_gs_x.to(self.device)  # numpy to torch
        gs_y = wavelet_gs_y.to(self.device)

        gs_x = gs_x.unsqueeze(0)  #[1,Nf,K]
        gs_y = gs_y.unsqueeze(0)


        ## wavelet filtering 
        Cxy_filtering = self.MCFP(gs_y, gs_x, Cxy_est)  # [1, K ,K]
        # Cyx_filtering = self.MCFP(gs_x, gs_y, Cyx_est)  # [1, K, K]

        # wavelet Rfmnet 非常简单
        self.loss_metrics['l_Rfmnet'] = torch.linalg.norm(evecs_y-torch.bmm(Pyx, torch.bmm(evecs_x, Cxy_filtering.transpose(-2, -1))))  # 这个就是couple, 采用我们的滤波器！


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

        # feature extractor
        if self.train_desc_path is not None:   # for topology
            
            # load SHOT descriptor
            data_x['desc'] = sio.loadmat(osp.join(self.train_desc_path, ''.join(data_x['name'])+'.mat'))['desc'].astype(np.float32)
            data_y['desc'] = sio.loadmat(osp.join(self.train_desc_path, ''.join(data_y['name'])+'.mat'))['desc'].astype(np.float32) 

            data_x['desc'] = to_device(torch.from_numpy(data_x['desc']), self.device)
            data_y['desc'] = to_device(torch.from_numpy(data_y['desc']), self.device)

            feat_x = self.networks['feature_extractor'](data_x['desc'])  # [ Nx, C]
            feat_y = self.networks['feature_extractor'](data_y['desc'])  # [ Ny, C]

            feat_x = feat_x.unsqueeze(0)  #[1, Nx, C]
            feat_y = feat_y.unsqueeze(0)
        
        else:
            feat_x = self.networks['feature_extractor'](data_x['verts'], data_x.get('faces'))
            feat_y = self.networks['feature_extractor'](data_y['verts'], data_y.get('faces'))

        # get spectral operators
        evals_x = data_x['evals']  
        evals_y = data_y['evals']  # 将特征值放进来！

        evecs_x = data_x['evecs'].squeeze()
        evecs_y = data_y['evecs'].squeeze()

        evecs_trans_x = data_x['evecs_trans'].squeeze()
        evecs_trans_y = data_y['evecs_trans'].squeeze()


        ## wavelet filter
        evals_x_cpu = evals_x.cpu().numpy()
        evals_y_cpu = evals_y.cpu().numpy()

        wavelet_gs_x = Meyer(max(evals_x_cpu[0]), Nf =6)(evals_x_cpu[0])  # evals_x[0] : [,K]; evals_x[1, K]
        wavelet_gs_y = Meyer(max(evals_y_cpu[0]), Nf =6)(evals_y_cpu[0])  # 

        gs_x = wavelet_gs_x.to(self.device)  # numpy to torch
        gs_y = wavelet_gs_y.to(self.device)

        gs_x = gs_x.unsqueeze(0)  #[1,Nf,K]
        gs_y = gs_y.unsqueeze(0)

        p2p = nn_query(feat_x, feat_y).squeeze() # nearest neighbour query
        Cxy_est = evecs_trans_y @ evecs_x[p2p]   # compute functional maps

        if not self.non_isometric:   # non_isometric matching 迭代一次即可

            # using learned filter functions to refine
            for _ in range(5):  # 迭代几次

                Cxy_est = Cxy_est.unsqueeze(0)  #[1, K, K]
                
                Cxy_est = self.MCFP(gs_y, gs_x, Cxy_est)  # [1, K ,K]
                Cxy_est = Cxy_est.squeeze()  #[K, K]

                # convert functional map to point-to-point map
                p2p = fmap2pointmap(Cxy_est, evecs_x, evecs_y)

                Cxy_est = evecs_trans_y @ evecs_x[p2p]

        Pyx = evecs_y @ Cxy_est @ evecs_trans_x  #置换矩阵


        # finish record
        timer.record()

        # resume previous network state dict
        if self.with_refine > 0:
            self.resume_model(state_dict, net_only=True, verbose=False)

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
        
    def feat_correspondence(self, feat_x, feat_y, bidirectional=False, normalize=True):
        if normalize:
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)

        feat_x = torch.squeeze(feat_x)
        feat_y = torch.squeeze(feat_y)
        Pxy, Pyx = sinkhorn_correspondences(feat_x, feat_y)
        
        if bidirectional:
            return Pxy, Pyx
        else:
            return Pyx

    def refine(self, data):

        self.networks['permutation'].hard = False
        self.networks['fmap_net'].bidirectional = True

        with torch.set_grad_enabled(True):
            for _ in range(self.with_refine):
                self.feed_data(data)
                self.optimize_parameters()

        self.networks['permutation'].hard = True
        self.networks['fmap_net'].bidirectional = False


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
    
    
    @torch.no_grad()
    def validation(self, dataloader, tb_logger, update=True):
        # change permutation prediction status
        # if 'permutation' in self.networks:
        #     self.networks['permutation'].hard = True
        # if 'fmap_net' in self.networks:
        #     self.networks['fmap_net'].bidirectional = False

        super(RfmnetModel, self).validation(dataloader, tb_logger, update)

        # if 'permutation' in self.networks:
        #     self.networks['permutation'].hard = False
        # if 'fmap_net' in self.networks:
        #     self.networks['fmap_net'].bidirectional = True


class Meyer(object):
    def __init__(self, lmax, Nf=6, scales=None):

        self.Nf=Nf

        if scales is None:
            scales = (4./(3 * lmax)) * np.power(2., np.arange(Nf-2, -1, -1))

        if len(scales) != Nf - 1:
            raise ValueError('len(scales) should be Nf-1.')

        self.g = [lambda x: kernel(scales[0] * x, 'scaling_function')]

        for i in range(Nf - 1):
            self.g.append(lambda x, i=i: kernel(scales[i] * x, 'wavelet'))

        def kernel(x, kernel_type):
            r"""
            Evaluates Meyer function and scaling function

            * meyer wavelet kernel: supported on [2/3,8/3]
            * meyer scaling function kernel: supported on [0,4/3]
            """

            x = np.asarray(x)

            l1 = 2/3.
            l2 = 4/3.  # 2*l1
            l3 = 8/3.  # 4*l1

            def v(x):
                return x**4 * (35 - 84*x + 70*x**2 - 20*x**3)

            r1ind = (x < l1)
            r2ind = (x >= l1) * (x < l2)
            r3ind = (x >= l2) * (x < l3)

            # as we initialize r with zero, computed function will implicitly
            # be zero for all x not in one of the three regions defined above
            r = np.zeros(x.shape)
            if kernel_type == 'scaling_function':
                r[r1ind] = 1
                r[r2ind] = np.cos((np.pi/2) * v(np.abs(x[r2ind])/l1 - 1))
            elif kernel_type == 'wavelet':
                r[r2ind] = np.sin((np.pi/2) * v(np.abs(x[r2ind])/l1 - 1))
                r[r3ind] = np.cos((np.pi/2) * v(np.abs(x[r3ind])/l2 - 1))
            else:
                raise ValueError('Unknown kernel type {}'.format(kernel_type))

            return r


    def __call__(self, evals):
        # input:
        #   evals: [K,], pytorch tensor
        # output: 
        #   gs: [Nf,K,], pytorch tensor
        # evals=evals.numpy()
        gs=np.expand_dims(self.g[0](evals),0)

        for s in range(1, self.Nf):
            gs=np.concatenate((gs,np.expand_dims(self.g[s](evals),0)),0)
        
        return torch.from_numpy(gs.astype(np.float32))