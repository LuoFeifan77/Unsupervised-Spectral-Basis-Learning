from pickle import TRUE
from re import I
import torch
import torch.nn.functional as F
from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, fmap2pointmap
from networks.diffzo_network import KernelZoomOut   # 将这个放进来

@MODEL_REGISTRY.register()
class LearningBasisZOModel(BaseModel):  # Base model
    def __init__(self, opt):
        self.with_refine = opt.get('refine', -1)
        self.partial = opt.get('partial', False)
        self.non_isometric = opt.get('non-isometric', False)
        self.using_nn_search = opt.get('using_nn_search', False)
        self.norm_filter = opt.get('norm_filter', False)
        self.sm_feat = opt.get('sm_feat', False)
        if self.with_refine > 0:
            opt['is_train'] = True
        super(LearningBasisZOModel, self).__init__(opt)

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

        #2 calculate C_filter by MCFP, obtain basis; 这个函数改一下
        # evecs_x_gs, evecs_y_gs, evecs_trans_x_gs, evecs_trans_y_gs, _, _= self.networks['basisconv'](evecs_x, evecs_y, \
        #                             evecs_trans_x, evecs_trans_y)  # 使用sigmod function
        
        # gs_x, gs_y = self.networks['eigconv'](evals_x, evals_y) # 使用学习的滤波器函数

        # 采用特征值的效果好一些
        Cxy, Cyx = self.networks['fmap_net'](feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)
        # Cxy, Cyx = self.networks['fmap_net'](feat_x, feat_y, evals_x, evals_y, evecs_trans_x_gs, evecs_trans_y_gs)  # 学习的
        # loss
        self.loss_metrics = self.losses['surfmnet_loss'](Cxy, Cyx, evals_x, evals_y)  # for estimation, bij+orth

        # 让计算逐点映射的alpha是学习的？
        # 这个地方的计算也可以改掉
        Pxy, Pyx = self.compute_permutation_matrix(feat_x, feat_y, bidirectional=True)  # 应该只是这个使用

        # 这里refine 一下看看效果
        # 计算functional maps, 使用一个refine
        # Cxy_est = torch.bmm(evecs_trans_y[:,:k,:], torch.bmm(Pyx, evecs_x[:,:,:k]))   # [1, K, K]
        # Cyx_est = torch.bmm(evecs_trans_x[:,:k,:], torch.bmm(Pxy, evecs_y[:,:,:k]))   # [1, K, K]

        # 计算 Pxy, Pyx 看看效果
        # Pxy = self.compute_permutation_matrix(torch.bmm(evecs_x[:,:,:k], Cxy_est.transpose(1, 2)), evecs_y[:,:,:k], bidirectional=False)
        # Pyx = self.compute_permutation_matrix(torch.bmm(evecs_y[:,:,:k], Cyx_est.transpose(1, 2)), evecs_x[:,:,:k] , bidirectional=False)

        step = 20 
        self.loss_metrics['l_align'] = 0  # 这样的多分辨率效果很差
        for k in range(60, 201, step): 

            # Cxy_est = torch.bmm(evecs_trans_y_gs[:,:k,:], torch.bmm(Pyx, evecs_x_gs[:,:,:k]))   # [1, K, K]
            # Cyx_est = torch.bmm(evecs_trans_x_gs[:,:k,:], torch.bmm(Pxy, evecs_y_gs[:,:,:k]))   # [1, K, K]
            # evecs_trans_y_gs[:,:k,:] = torch.bmm()
            # evecs_trans_x_gs[:,:k,:] = torch.bmm()

            # Cxy_est = torch.bmm(evecs_trans_y_gs[:,:k,:], torch.bmm(Pyx, evecs_x_gs[:,:,:k]))   # [1, K, K]
            # Cyx_est = torch.bmm(evecs_trans_x_gs[:,:k,:], torch.bmm(Pxy, evecs_y_gs[:,:,:k]))   # [1, K, K]

            # 这里refine 一下看看效果
            # 计算functional maps, 使用一个refine
            Cxy_est = torch.bmm(evecs_trans_y[:,:k,:], torch.bmm(Pyx, evecs_x[:,:,:k]))   # [1, K, K]
            Cyx_est = torch.bmm(evecs_trans_x[:,:k,:], torch.bmm(Pxy, evecs_y[:,:,:k]))   # [1, K, K]

            # 计算 Pxy, Pyx 看看效果
            Pxy = self.compute_permutation_matrix(torch.bmm(evecs_x[:,:,:k], Cxy_est.transpose(1, 2)), evecs_y[:,:,:k], bidirectional=False)
            Pyx = self.compute_permutation_matrix(torch.bmm(evecs_y[:,:,:k], Cyx_est.transpose(1, 2)), evecs_x[:,:,:k] , bidirectional=False)

            # 最后仅仅用来对齐C_refine, 回复成高分辨率的
            Cxy_est = torch.bmm(evecs_trans_y, torch.bmm(Pyx, evecs_x))   # [1, K, K]
            Cyx_est = torch.bmm(evecs_trans_x, torch.bmm(Pxy, evecs_y))   # [1, K, K]

            self.loss_metrics['l_align'] += self.losses['align_loss'](Cxy, Cxy_est) + self.losses['align_loss'](Cyx, Cyx_est)

        # 用来初始化
        ##  
        # forward(self, F1, F2, evects1, evects2, mass1, mass2, return_T=False, return_init=False, faces1=None, faces2=None):

        # self.dzo_layer = self.networks['dzo_layer']()  # 初始化, 传递方法
        # 这种对C的refine 效果会变得更差
        # Cxy_est, Cyx_est = self.networks['dzo_layer'](feat_x.squeeze(0), feat_y.squeeze(0), evecs_x[0], evecs_y[0], data_x['mass'][0], data_y['mass'][0],
        #                               return_init=False, return_T=False)  # 用少量的特征值初始，还没有refine


        # K = C01.shape[-1]  #
        # C10_pred_ref = self.compute_C12(feat_y.squeeze(0), feat_x.squeeze(0), evecs_y[0], evecs_x[0], data_x['mass'][0], K)

        # 最后仅仅用来对齐C_refine
        # 直接对齐
        # self.loss_metrics['l_align'] = self.losses['align_loss'](Cxy, Cxy_est.unsqueeze(0))  # 和它对应分辨率的loss 进行对齐

        #8.2 calculate loss
        # if not self.partial:
        #     if Cyx != None:
        #         self.loss_metrics['l_align'] += self.losses['align_loss'](Cyx, Cyx_est.unsqueeze(0))  #添加双向

        # 对比loss
        # 用来提升基函数的表征能力
        # Pxx = self.compute_permutation_matrix(evecs_x_gs, evecs_x_gs, bidirectional=False)
        # Pyy = self.compute_permutation_matrix(evecs_y_gs, evecs_y_gs, bidirectional=False)

        # Cxx_est = torch.bmm(evecs_trans_x_gs, torch.bmm(Pxx, evecs_x_gs))   # [1, K, K]
        # Cyy_est = torch.bmm(evecs_trans_y_gs, torch.bmm(Pyy, evecs_y_gs))   # [1, K, K]
        
        # eye = torch.eye(Cxx_est.shape[1], Cxx_est.shape[2], device=Cxx_est.device).unsqueeze(0)
        # eye_batch = torch.repeat_interleave(eye, repeats=Cxx_est.shape[0], dim=0)

        # self.loss_metrics['l_basis'] = self.losses['align_loss'](Cxx_est, eye_batch) + self.losses['align_loss'](Cyy_est, eye_batch) 
        # self.loss_metrics['l_basis'] *= 10

        # 可视化看下二者之间的差异
        # corr_x = data_x['corr']
        # corr_y = data_y['corr']
        # Cxy_gt = torch.linalg.lstsq(evecs_x[0,corr_x,:], evecs_y[0,corr_y, :]).solution   # 得到groundtruth
        # Cyx_gt = torch.linalg.lstsq(evecs_y[0,corr_y,:], evecs_x[0,corr_x, :]).solution 
        # Cxy_gt = Cxy_gt.squeeze()

        # loss_gt = self.losses['align_loss'](Cxy_est, Cxy_gt) + self.losses['align_loss'](Cyx_est, Cyx_gt) #把他们之间的loss 全部输出看看
        # loss_gt = self.losses['align_loss'](Cxy_est.unsqueeze(0), Cxy_gt) + self.losses['align_loss'](Cyx_est.unsqueeze(0), Cyx_gt)


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
        # evals_x = data_x['evals']  #[1, K]
        evecs_x = data_x['evecs'].squeeze()
        evecs_trans_x = data_x['evecs_trans'].squeeze() # [1, K, Nx]
       
        # evals_y = (data_y['evals']) 
        evecs_y = (data_y['evecs']).squeeze() # [1, K, Nx]
        evecs_trans_y = (data_y['evecs_trans']).squeeze()  # [1, K, Ny]

        # 使用smooth features
        # if self.sm_feat: 
        # feat_x = torch.bmm(evecs_x, torch.bmm(evecs_trans_x, feat_x))
        # feat_y = torch.bmm(evecs_y, torch.bmm(evecs_trans_y, feat_y))

        # evecs_x_gs, evecs_y_gs, evecs_trans_x_gs, evecs_trans_y_gs, gs_x, gs_y = self.networks['basisconv']( evecs_x, evecs_y, \
        #                             evecs_trans_x, evecs_trans_y) 
        
        # evecs_x_gs, evecs_y_gs, evecs_trans_x_gs, evecs_trans_y_gs, _, _= self.networks['basisconv'](evecs_x, evecs_y, \
        #                             evecs_trans_x, evecs_trans_y)  # 使用sigmod function

        # evecs_x_gs = evecs_x_gs.squeeze()
        # evecs_y_gs = evecs_y_gs.squeeze()
        # evecs_trans_x_gs = evecs_trans_x_gs.squeeze()
        # evecs_trans_y_gs = evecs_trans_y_gs.squeeze()

        # 最后这里使用zoomout

        if self.non_isometric:
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)

            # nearest neighbour query
            p2p = nn_query(feat_x, feat_y).squeeze()  # 难道用这个要好一些？

            # compute Pyx from functional map
            # Cxy_est = evecs_trans_y_gs @ evecs_x_gs[p2p]
            # Pyx = evecs_y_gs @ Cxy_est @ evecs_trans_x_gs

            Cxy_est = evecs_trans_y @ evecs_x[p2p]
            Pyx = evecs_y @ Cxy_est @ evecs_trans_x

            # 这个后面在试一试
            # p2p = fmap2pointmap(Cxy_est, evecs_x_gs, evecs_y_gs)

        else:
            # compute Pxy
            Pyx = self.compute_permutation_matrix(feat_y, feat_x, bidirectional=False).squeeze() # soft correspondence 

            # Cxy_est = evecs_trans_y_gs @ (Pyx @ evecs_x_gs)
            # p2p = fmap2pointmap(Cxy_est, evecs_x_gs, evecs_y_gs)
            # Pyx = evecs_y_gs @ Cxy_est @ evecs_trans_x_gs

            # 逐点映射
            Cxy_est = evecs_trans_y @ (Pyx @ evecs_x)
            p2p = fmap2pointmap(Cxy_est, evecs_x, evecs_y)
            Pyx = evecs_y @ Cxy_est @ evecs_trans_x


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
        return p2p, Pyx, Cxy_est


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
    
    # 用来计算Cxy
    def compute_C12(self, F1, F2, evecs1, evecs2, mass2, K):
        # self.dzo_layer(feats0.squeeze(0), feats1.squeeze(0), batch_data['evecs0'][0], batch_data['evecs1'][0], batch_data['mass1'][0])

        T21_init = self.networks['dzo_layer'].compute_init(F1, F2)  # 这里可以用到最后的结果
        C12 = self.networks['dzo_layer'].compute_C12(T21_init, K, evecs1, evecs2, mass2)  # 只是最后求的吗

        return C12


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
        super(LearningBasisZOModel, self).validation(dataloader, tb_logger, update)
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = False
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = True
