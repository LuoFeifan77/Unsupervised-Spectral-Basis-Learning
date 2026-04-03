import torch
import hashlib
import numpy as np
import scipy
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import gamma as gammaFunc
from torch import Tensor
from utils.registry import NETWORK_REGISTRY

EPS=1e-7

def get_norm_weight_formula_torch(a, b, K):   # 生成标准权重进行计算！
    ret = []

    def gammaFunc_torch(x):
        return torch.exp(torch.special.gammaln(x))

    for i in range(K+1):
        term1 = torch.pow(2, a+b+1)/(2*i+a+b+1)
        term2 = gammaFunc_torch(i+a+1)/gammaFunc_torch(i+a+b+1)
        term3 = gammaFunc_torch(i+b+1)/gammaFunc_torch(torch.tensor(i+1))
        ret.append(torch.sqrt(term1*term2*term3))
    return torch.stack(ret)


def get_norm_weight_formula_np(a, b, K):  # 这个就是Jacobi, 这个地方的值，改变了！
    ret = []
    for i in range(K+1):
        term1 = np.power(2, a+b+1)/(2*i+a+b+1)
        term2 = gammaFunc(i+a+1)/gammaFunc(i+a+b+1)
        term3 = gammaFunc(i+b+1)/gammaFunc(i+1)
        ret.append(np.sqrt(term1*term2*term3))
    ret = np.asarray(ret, dtype=np.float32)
    # print(ret, file=sys.stderr)
    return ret


def _gumbel_sigmoid(logits, tau=1, hard=False, eps=1e-10, training = True, threshold = 0.5):
    if training :
        # ~Gumbel(0,1)`
        gumbels1 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels2 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        # Difference of two` gumbels because we apply a sigmoid
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
    else :
        y_soft = logits.sigmoid()

    if hard:
        # Straight through.
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).masked_fill(y_soft > threshold, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


# 这个函数没有用上去
def gumbel_softmax(logits, tau=5.0, dim = -1):
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    
    return y_soft


# 直接对提取的shape feature
# mask each channel of features by using sigmod function learning
@NETWORK_REGISTRY.register()
class feat_mask(torch.nn.Module):
    def __init__(self, dim=256, channel_number=256, tau=5, is_hard=True, threshold=0.5, is_training=True):
        super().__init__()
        # self.channel_number = channel_number
        self.dim = dim
        self.channel_number = channel_number
        self.router = torch.nn.Linear(dim, channel_number)  # 怎么是这样的一个
        self.is_hard = is_hard
        self.tau = tau
        self.threshold = threshold   # 大于0.5 pick这个channel，小于0.5就去掉这个channel
        self.training = is_training

    def forward(self, Feat_x, Feat_y, is_normlize=False): 
        # b, l = x.shape[:2]
        # x = x.mean(dim=1)
        # 输入特征值+特征向量+特征向量的违逆

        logits_x = self.router(Feat_x)  # 输出特征值
        logits_y = self.router(Feat_y)  # 输出特征值

        # 直接用特征值初始化（ 可以使用热扩散+sigmod 来挑选特征数量？） 
        # channel_select_x = _gumbel_sigmoid(logits_x, self.tau, self.is_hard, threshold=self.threshold, training=self.training)
        channel_select_y = _gumbel_sigmoid(logits_y, self.tau, self.is_hard, threshold=self.threshold, training=self.training)

        # channel_select_y = torch.sigmoid(logits_x)

        # 对其中一个进行mask
        # Feat_x = Feat_x * channel_select_x
        Feat_y = Feat_y * channel_select_y

        # Feat_x = torch.bmm(Feat_x, channel_select_x.unsqueeze(0))
        # Feat_y = torch.bmm(Feat_y, channel_select_y.unsqueeze(0))

        return Feat_x, Feat_y


# 第一种，思路是直接作用在特征值上生成{0,1}的mask，然后对应提取特征值和特征向量，以及特征向量的违逆

# 其实我的方法已经可以了
# 第二种，怎么去评价特征函数的不同通道的重要性，


# heat kernel 
@NETWORK_REGISTRY.register()
class Heat_Kernel_Fliters(torch.nn.Module):
    def __init__(self, 
                C_inout = 200) :
        super().__init__()
        self.C_inout = C_inout  # 与特征值的维度一致
        # self.device = device # do not need 
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  #  # 扩散的时间
        nn.init.constant_(self.diffusion_time, 0.0)

    def forward(self, evals_x, evals_y, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y):   # 为什么没有计算呢
        
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)
        
        # Diffuse
        time = self.diffusion_time

        filter_x = torch.exp(-evals_x * time)  # 就这个就
        filter_y = torch.exp(-evals_y * time)

        inv_filter_x = torch.exp(evals_x * time) 
        inv_filter_y = torch.exp(evals_y * time)

        evecs_x = evecs_x * filter_x
        evecs_y = evecs_y * filter_y

        evecs_trans_x =  evecs_trans_x * inv_filter_x.unsqueeze(-1)
        evecs_trans_y =  evecs_trans_y * inv_filter_y.unsqueeze(-1)

        return filter_x, filter_y, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y


# For EigenBasis Learning without eigenvalues
@NETWORK_REGISTRY.register()
class EigenBasis_Fliters(torch.nn.Module):
    def __init__(self, 
                C_inout = 200,
                tau = 1,
                filter_type ='heat') :
        super().__init__()
        self.C_inout = C_inout  # 与特征值的维度一致
        self.tau = tau
        self.filter_type = filter_type  # 选择不同的滤波器
        # self.nn = torch.nn.Linear(200, 200)
        if self.filter_type =='none':
            self.diffusion_time = torch.ones(C_inout)
        else:
            self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  #  # 扩散的时间
            nn.init.constant_(self.diffusion_time, 0.0)  # 原来是这样定义的
            if self.filter_type =='heat_delta':
                self.delta = nn.Parameter(torch.Tensor(C_inout))  # 余项
                nn.init.constant_(self.delta, 0.1)

    def forward(self, evals_x, evals_y, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y):   # 为什么没有计算呢
        
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)  # 设置最小值为0

        if self.filter_type =='sigmod':
        # inverse
            diffusion_coefs_x = torch.exp(-self.diffusion_time) # 不需要用上特征值
            diffusion_coefs_y = torch.exp(-self.diffusion_time)

            gs_x_inv = self.tau + diffusion_coefs_x    
            gs_y_inv = self.tau + diffusion_coefs_y

            # filter invers
            gs_x = 1/ gs_x_inv    # 控制在[0,1]之间
            gs_y = 1/ gs_y_inv

        if self.filter_type =='heat':

            #仅仅改变X
            gs_x = torch.exp(-self.diffusion_time)  #这个数值不在[0,1]之间吗
            gs_y = torch.exp(-self.diffusion_time)   
       
            gs_x_inv = torch.exp(self.diffusion_time) 
            gs_y_inv = torch.exp(self.diffusion_time) 

        if self.filter_type =='heat_delta':
            with torch.no_grad():
                self.delta.data = torch.clamp(self.diffusion_time, min=1e-3)  # 设置最小值为0.001

            gs_x = torch.exp(-self.diffusion_time) + self.delta
            gs_y = torch.exp(-self.diffusion_time) + self.delta
       
            gs_x_inv = 1/ gs_x  #torch.exp(self.diffusion_time) 
            gs_y_inv = 1/ gs_y  #torch.exp(self.diffusion_time) 


        if self.filter_type =='none':

            # to_same device 
            self.diffusion_time = self.diffusion_time.to(evals_x.device)
            
            gs_x = self.diffusion_time  #这个数值不在[0,1]之间吗
            gs_y = self.diffusion_time   
            gs_x_inv = self.diffusion_time
            gs_y_inv = self.diffusion_time


        evecs_trans_x_gs = gs_x_inv.unsqueeze(-1) * evecs_trans_x  # utilize inverse 
        evecs_trans_y_gs = gs_y_inv.unsqueeze(-1) * evecs_trans_y  

        evecs_x_gs = evecs_x * gs_x.unsqueeze(0) 
        evecs_y_gs = evecs_y * gs_y.unsqueeze(0) 

        return gs_x, gs_y, evecs_x_gs, evecs_y_gs, evecs_trans_x_gs, evecs_trans_y_gs


@NETWORK_REGISTRY.register()
class AsyNoEvals_Fliters(torch.nn.Module):
    def __init__(self, 
                C_inout = 200,
                tau = 1,
                filter_type ='sigmod') :
        super().__init__()
        self.C_inout = C_inout  # 与特征值的维度一致
        self.tau = tau
        self.filter_type = filter_type  # 选择不同的滤波器
        self.tx = nn.Parameter(torch.Tensor(C_inout)) 
        self.ty = nn.Parameter(torch.Tensor(C_inout)) 
        nn.init.constant_(self.tx, 0.0)  # 原来是这样定义的
        nn.init.constant_(self.ty, 0.0)  # 测试一下异步的结果

    def forward(self, evals_x :Tensor, evals_y : Tensor, evecs_x :Tensor, evecs_y : Tensor, \
                evecs_trans_x: Tensor, evecs_trans_y: Tensor):   # 为什么没有计算呢
        
        # with torch.no_grad():
        #     self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)  # 设置最小值为0

        # 将特征值缩放到[0,1]
        evals_x = evals_x / torch.max(evals_x)  # 缩放大[0,1]
        evals_y = evals_y / torch.max(evals_y)

        # 凑成 sigmod 形式
        # 但不是0，1之间，这一点还是有点不一样！
        if self.filter_type =='sigmod':
        # inverse
            diffusion_coefs_x = torch.exp(-self.tx) #
            diffusion_coefs_y = torch.exp(-self.ty)

            gs_x = self.tau + diffusion_coefs_x    # 写错了
            gs_y = self.tau + diffusion_coefs_y

            # filter invers
            gs_x_inv = 1/ gs_x    # 控制在[0,1]之间
            gs_y_inv = 1/ gs_y

        if self.filter_type =='heat':

            # 直接使用热核函数
            gs_x = torch.exp(-self.diffusion_time)  #这个数值不在[0,1]之间吗
            gs_y = torch.exp(-self.diffusion_time)   

            gs_x_inv = torch.exp( self.diffusion_time) 
            gs_y_inv = torch.exp( self.diffusion_time) 

        evecs_trans_x_gs = gs_x_inv.unsqueeze(-1) * evecs_trans_x  # utilize inverse 
        evecs_trans_y_gs = gs_y_inv.unsqueeze(-1) * evecs_trans_y  

        evecs_x_gs = evecs_x * gs_x.unsqueeze(0) 
        evecs_y_gs = evecs_y * gs_y.unsqueeze(0) 

        return evecs_x_gs, evecs_y_gs, evecs_trans_x_gs, evecs_trans_y_gs, gs_x, gs_y




# 这个后面可以考虑一下
@NETWORK_REGISTRY.register()
class Feature_Fliters(torch.nn.Module):
    def __init__(self, 
                C_inout = 256) :
        super().__init__()
        self.C_inout = C_inout  # 与特征值的维度一致
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  #  # 扩散的时间
        nn.init.constant_(self.diffusion_time, 0.0)  # 原来是这样定义的

    def forward(self, feat_x :Tensor, feat_y :Tensor):   # 为什么没有计算呢
        
        # with torch.no_grad():
        #     self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        feat_filter = torch.sigmoid(self.diffusion_time)  # 这个做法有点多余！

        feat_x = feat_x * feat_filter
        feat_y = feat_y * feat_filter

        return feat_x, feat_y




@NETWORK_REGISTRY.register()
class ResHeat_Kernel_Fliters(torch.nn.Module):
    def __init__(self, 
                C_inout = 200) :
        super().__init__()
        self.C_inout = C_inout  # 与特征值的维度一致
        # self.device = device # do not need 
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  #  # 扩散的时间
        nn.init.constant_(self.diffusion_time, 0.0)  # 原来是这样定义的

    def forward(self, evals_x :Tensor, evals_y : Tensor, evecs_x :Tensor, evecs_y : Tensor, \
                evecs_trans_x: Tensor, evecs_trans_y: Tensor):   # 为什么没有计算呢
        
        # with torch.no_grad():
        #     self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        # 将特征值缩放到[0,1]
        evals_x = evals_x / max(evals_x)
        evals_y = evals_y / max(evals_y)
        # Diffuse
        time = self.diffusion_time
        diffusion_coefs_x = torch.exp(-evals_x * time)  # 就这个就
        diffusion_coefs_y = torch.exp(-evals_y * time)

        # 凑成 sigmod 形式
        # inverse
        gs_x_inv = 1+ diffusion_coefs_x
        gs_y_inv = 1+ diffusion_coefs_y

        # filter 
        gs_x = 1/ gs_x_inv    # 利用矩阵广播进行计算
        gs_y = 1/ gs_y_inv

        # res term
        gs_x_r = 1 - gs_x   #余项
        gs_y_r = 1 - gs_y

        gs_x_r_inv = 1 + torch.exp(evals_x * time)   #余项
        gs_y_r_inv = 1 + torch.exp(evals_y * time)

        evecs_trans_x_gs = gs_x_inv.unsqueeze(-1) * evecs_trans_x  # utilize inverse 
        evecs_trans_y_gs = gs_y_inv.unsqueeze(-1) * evecs_trans_y  

        evecs_trans_x_gs_r = gs_x_r_inv.unsqueeze(-1) * evecs_trans_x  # utilize inverse 
        evecs_trans_y_gs_r = gs_y_r_inv.unsqueeze(-1) * evecs_trans_y  

        evecs_x_gs = evecs_x * gs_x.unsqueeze(0) 
        evecs_y_gs = evecs_y * gs_y.unsqueeze(0) 
        # resdual term
        evecs_x_gs_r = evecs_x * gs_x_r.unsqueeze(0) 
        evecs_y_gs_r = evecs_y * gs_y_r.unsqueeze(0)


        return evecs_x_gs, evecs_y_gs, evecs_x_gs_r, evecs_y_gs_r, evecs_trans_x_gs, \
                evecs_trans_y_gs, evecs_trans_x_gs_r, evecs_trans_y_gs_r




####-------------------------Jacobi filters------------------------###########

from functools import partial

# 将其申明成类
@NETWORK_REGISTRY.register()
class Learned_Fliters(torch.nn.Module):
    def __init__(self, 
                Poly_Type : str = 'JacobiConv',
                orders: int = 32, 
                alpha:float = 1.0, 
                learnable_bases: bool = True, 
                learnable_alphas : bool = True,
                normalized_bases: bool = True,
                **kwargs) :
        super().__init__()
        
        self.Poly_Type = Poly_Type
        self.orders = orders  # default 30
        self.basealpha = alpha
        self._normalized_bases = normalized_bases
        self.learnable_alphas = learnable_alphas
        self.learnable_bases = learnable_bases
        
        # 对alphas的限制是有效的   

        # 初始化滤波器哦
        #1 choose conv
        if self.Poly_Type == 'JacobiConv':
            conv_fn =  partial(JacobiConv, **kwargs) # 传递学习的滤波器+ 学习参数alphas
            # conv_fn =  partial(JacobiConv, [jacobi_a, jacobi_b])
            jacobi_a=kwargs.get('a', 1.0),  # 系数不用，学习参数a,b
            jacobi_b=kwargs.get('b', 1.0),
            # whether learn a, b or not
            if learnable_bases:
                self._a = nn.Parameter(torch.tensor(jacobi_a), requires_grad=True)   # 定义一个可学习的部分
                self._b = nn.Parameter(torch.tensor(jacobi_b), requires_grad=True)
            else:
                self._a = jacobi_a[0]
                self._b = jacobi_b[0]

        if self.Poly_Type == 'PowerConv':
            conv_fn = PowerConv
        if self.Poly_Type == 'LegendreConv':
            conv_fn = LegendreConv
        if self.Poly_Type == 'ChebyshevConv':
            conv_fn = ChebyshevConv

        self.conv_fn = conv_fn

        #2 whether learn alpha or not and apply restriction on alphas
        if self.learnable_alphas:
            self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(float(min(1 / alpha, 1))),
                         requires_grad=True) for i in range(self.orders)]) # 多项式系数
        else:
            self.alphas = [torch.tensor(float(min(1 / alpha, 1))) for i in range(self.orders)]  # 定义非学习部分
            

    def forward(self, evals_x :Tensor, evals_y : Tensor):   # 为什么没有计算呢

        alphas = [self.basealpha * torch.tanh(_) for _ in self.alphas] if self.learnable_alphas else self.alphas

        num_eig_x = len(evals_x[0])  # 计算特征值的个数
        num_eig_y = len(evals_y[0])

        self.device = evals_x.device
        #1 yield sign == 1
        x= torch.ones(1, num_eig_x, device = self.device)
        y= torch.ones(1, num_eig_y, device = self.device)

        #2 learn a, b
        if self.learnable_bases:
            a = self._a if isinstance(self._a, float) else torch.clamp(self._a, min=-1.0+EPS, max=10.0)  # 学习参数a,b
            b = self._b if isinstance(self._b, float) else torch.clamp(self._b, min=-1.0+EPS, max=10.0)

            xs_x = [self.conv_fn(0, [x], evals_x, alphas, a=a, b=b)]  # 初始化
            xs_y = [self.conv_fn(0, [y], evals_y, alphas, a=a, b=b)]  # 初始化

            for L in range(1, self.orders):  # 这个就是阶数啊

                tx = self.conv_fn(L, xs_x, evals_x, alphas, a=a, b=b)   # 计算L层的卷积结果
                ty = self.conv_fn(L, xs_y, evals_y, alphas, a=a, b=b) 

                xs_x.append(tx)
                xs_y.append(ty)

        else: 
            xs_x = [self.conv_fn(0, [x], evals_x, alphas)]  # 初始化
            xs_y = [self.conv_fn(0, [y], evals_y, alphas)]  # 初始化

            for L in range(1, self.orders):  # 这个就是阶数啊

                tx = self.conv_fn(L, xs_x, evals_x, alphas)   # 计算L层的卷积结果
                ty = self.conv_fn(L, xs_y, evals_y, alphas) 

                xs_x.append(tx)
                xs_y.append(ty)

        xs_x = [x.unsqueeze(1) for x in xs_x]
        xs_y = [y.unsqueeze(1) for y in xs_y]

        basis_x = torch.cat(xs_x, dim=1)   # 分别用来计算这个值
        basis_y = torch.cat(xs_y, dim=1)   # 拼凑

        #3 normalized base
        if self.Poly_Type =='JacobiConv': 
            if self._normalized_bases and self.learnable_bases:
                norms = get_norm_weight_formula_torch(self._a, self._b, self.orders-1).reshape(1, self.orders, 1)
                basis_x = basis_x / (norms + EPS)
                basis_y = basis_y / (norms + EPS)

            if self._normalized_bases and not self.learnable_bases:
                norms = get_norm_weight_formula_np(self._a, self._b, self.orders-1).reshape(1, self.orders, 1)
                norms = torch.from_numpy(norms)
                norms = norms.to(self.device)
                basis_x = basis_x / (norms + EPS)
                basis_y = basis_y / (norms + EPS)

        # basis_x = (basis_x).squeeze()   # 删除维度为1的维度
        # basis_y = (basis_y).squeeze()

    # 这里直接把权重参数放进去就行了


        return basis_x, basis_y



def JacobiBasis_Conv_new(L, xs, adj, alphas, a=1.0, b=1.0, l=-1.0, r=1.0):  # 直接采用别人的方法生成
    '''
    Jacobi Bases. Please refer to our paper for the form of the bases.
    '''
    adj = 2*adj/adj[-1]-1  # 一个向量！

    if L == 0: return xs[0]
    if L == 1:
        coef1 = (a - b) / 2 
        coef1 *= alphas[0]
        coef2 = (a + b + 2) / (r - l)
        coef2 *= alphas[0]
        return coef1 * xs[-1] + coef2 * (adj * xs[-1])  # 这一步和论文里面的相同

    coef_l = 2 * L * (L + a + b) * (2 * L - 2 + a + b)
    coef_lm1_1 = (2 * L + a + b - 1) * (2 * L + a + b) * (2 * L + a + b - 2)
    coef_lm1_2 = (2 * L + a + b - 1) * (a**2 - b**2)
    coef_lm2 = 2 * (L - 1 + a) * (L - 1 + b) * (2 * L + a + b)

    tmp1 = alphas[L - 1] * (coef_lm1_1 / coef_l)  #  那我这里直接把alpha 去掉不久行呢吗
    tmp2 = alphas[L - 1] * (coef_lm1_2 / coef_l)
    tmp3 = alphas[L - 1] * (coef_lm2 / coef_l)

    nx = tmp1 * (adj * xs[-1]) + tmp2 * xs[-1]  # 为什么不是加
    nx -= tmp3 * xs[-2]
    return nx      # 内容也算简单


#---------------- 未修改的 --------------------#
def PowerConv(L, xs, adj, alphas):
    '''
    Monomial bases.
    '''
    if L == 0: return xs[0]
    return alphas[L] * (adj * xs[-1])


def LegendreConv(L, xs, adj, alphas):
    '''
    Legendre bases. Please refer to our paper for the form of the bases.
    '''
    adj = 2*adj/adj[-1]-1  # 一个向量！

    if L == 0: return xs[0]
    nx = (alphas[L - 1] * (2 - 1 / L)) * (adj * xs[-1])
    if L > 1:
        nx -= (alphas[L - 1] * alphas[L - 2] * (1 - 1 / L)) * xs[-2]
    return nx


def ChebyshevConv(L, xs, adj, alphas):
    '''
    Chebyshev Bases. Please refer to our paper for the form of the bases.
    '''
    adj = 2*adj/adj[-1]-1  # 一个向量！
    
    if L == 0: return xs[0]
    nx = (2 * alphas[L - 1]) * (adj * xs[-1])
    if L > 1:
        nx -= (alphas[L - 1] * alphas[L - 2]) * xs[-2]   # 系数分解
    return nx


def JacobiConv(L, xs, adj, alphas, a=1.0, b=1.0, l=-1.0, r=1.0):  # 直接在这上面进行修改的!
    '''
    Jacobi Bases. Please refer to our paper for the form of the bases.
    '''
    adj = 2*adj/adj[0][-1]-1  # 一个向量！这是缩放到[-1, 1];


    if L == 0: return xs[0]
    if L == 1:

        coef1 = (a - b) / 2 
        coef1 *= alphas[0]
        coef2 = (a + b + 2) / (r - l)
        coef2 *= alphas[0]
        return coef1 * xs[-1] + coef2 * (adj * xs[-1])  # 这一步和论文里面的相同

    coef_l = 2 * L * (L + a + b) * (2 * L - 2 + a + b)
    coef_lm1_1 = (2 * L + a + b - 1) * (2 * L + a + b) * (2 * L + a + b - 2)
    coef_lm1_2 = (2 * L + a + b - 1) * (a**2 - b**2)
    coef_lm2 = 2 * (L - 1 + a) * (L - 1 + b) * (2 * L + a + b)
    tmp1 = alphas[L - 1] * (coef_lm1_1 / coef_l)
    tmp2 = alphas[L - 1] * (coef_lm1_2 / coef_l)
    tmp3 = alphas[L - 1] * alphas[L - 2] * (coef_lm2 / coef_l)

    nx = tmp1 * (adj * xs[-1]) + tmp2 * xs[-1]  # 为什么不是加
    nx -= tmp3 * xs[-2]
    return nx


# Mexican_hat filters
class Mexican_hat(object):
    def __init__(self, lmax, Nf=6, scales=None):

        self.Nf = Nf

        if scales is None:
            lpfactor = 20
            lmin = lmax / lpfactor
            # scales = (4./(3 * lmax)) * np.power(2., np.arange(Nf-2, -1, -1))
            t1 = 1
            t2 = 2

            smin = t1 / lmax
            smax = t2 / lmin
            tt = np.linspace(np.log(smax), np.log(smin), Nf)

            scales = np.exp(tt)  # 也许需要调整

        self.gb = [lambda x: (np.exp(-x) * x)]  # high pass filter
        self.gl = [lambda x: (np.exp(-x ** 4))]  # low pass filter

        lminfac = 0.4 * lmin

        self.g = [lambda x: 1.2 * np.exp(-1) * self.gl[0](x / lminfac)]  # 等到尺度函数

        for i in range(Nf - 1):
            self.g.append(lambda x, i=i: self.gb[0](scales[i] * x))


    def __call__(self, evals):
        # input:
        #   evals: [K,], pytorch tensor
        # output:
        #   gs: [Nf,K,], pytorch tensor   返回函数值

        #-----------------数组版本------------------#

        gs = np.expand_dims(self.g[0](evals), 0)  # 扩张维度！
        
        for s in range(1, self.Nf):
            gs = np.concatenate((gs, np.expand_dims(self.g[s](evals), 0)), 0)
        # return gs
        return torch.from_numpy(gs.astype(np.float32))



# Mexican_hat filters
class Mexican_hat_MGCN(object): # Tight Frame
    def __init__(self, lmax, Nf=32, scales=None):

        self.Nf = Nf

        if scales is None:
            lpfactor = 20
            lmin = lmax/lpfactor
            # scales = (4./(3 * lmax)) * np.power(2., np.arange(Nf-2, -1, -1))
            t1 =1
            t2 =2
            smin = t1/lmax
            smax = t2/lmin
            tt = np.linspace(np.log(smax*1.15), np.log(smin*0.1), Nf)

            scales = np.exp(tt) # 也许需要调整

        self.gb = [lambda x: (0.443*x**2*np.exp(1-x**2))]  # high pass filter
        self.gl = [lambda x: (1.004*np.exp(-x**3))]  # low pass filter

        lminfac = 0.52 *lmin

        self.g = [lambda x: 1.2*np.exp(-1)*self.gl[0](x/lminfac)]  # 等到尺度函数

        for i in range(Nf-1):
            self.g.append(lambda x, i=i: self.gb[0](scales[i] * x))


    def __call__(self, evals):
        # input:
        #   evals: [K,], pytorch tensor
        # output:
        #   gs: [Nf,K,], pytorch tensor   返回函数值

        #-----------------数组版本------------------#

        gs = np.expand_dims(self.g[0](evals), 0)  # 扩张维度！
        
        for s in range(1, self.Nf):
            gs = np.concatenate((gs, np.expand_dims(self.g[s](evals), 0)), 0)
        # return gs
        return torch.from_numpy(gs.astype(np.float32))
        # return torch.from_numpy(gs.astype(np.float32), device='cuda')



# Meyer filters
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
