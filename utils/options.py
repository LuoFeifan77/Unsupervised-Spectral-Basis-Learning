import argparse
import random
import yaml
from collections import OrderedDict
from os import path as osp

import torch

from .misc import make_exp_dirs, set_random_seed
from .dist_util import get_dist_info, init_dist


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(opt_path, root_path, is_train=True):
    """Parse option file.

    Args:
        opt_path (str): Option file path.
        root_path (str): Root path.
        is_train (str): Indicate whether in training or not. Default True.

    Returns:
        (dict): Options.
    """
    # read config yaml file
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    opt['is_train'] = is_train

    # set number of gpus
    if opt['num_gpu'] == 'auto':
        opt['num_gpu'] = torch.cuda.device_count()

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)

    if is_train:  # specify training log paths
        experiments_root = osp.join(root_path, 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['log'] = osp.join(experiments_root, 'log')
    else:  # specify test log paths
        results_root = osp.join(root_path, 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = osp.join(results_root, 'log')
        opt['path']['visualization'] = osp.join(results_root, 'visualization')
    
    return opt


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + str(k) + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        elif isinstance(v, list):
            msg = ''
            for iv in v:
                if isinstance(iv, dict):
                    msg += dict2str(iv, indent_level)
                else:
                    msg += '\n' + ' ' * (indent_level * 2) + str(iv)
        else:
            msg += ' ' * (indent_level * 2) + str(k) + ': ' + str(v) + '\n'
    return msg



def parse_options(root_path, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to option YAML file.')

    args = parser.parse_args()
    opt = parse(args.opt, root_path, is_train=is_train)

    # distributed settings
    if opt['backend'] == 'dp':
        opt['dist'] = False
        print('Backend DataParallel.', flush=True)
    elif opt['backend'] == 'ddp':
        opt['dist'] = True
        port = opt.get('port', 29500)
        init_dist(port=port)
        print('Backend DistributedDataParallel.', flush=True)
    else:
        raise ValueError(f'Invalid backend option: {opt["backend"]}, only supports "dp" and "ddp"')

    # set rank and world_size
    opt['rank'], opt['world_size'] = get_dist_info()

    # make experiment directories
    make_exp_dirs(opt)

    # set random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt






def parse_options_debug(root_path, data_type, is_train=True):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()
    
    if is_train:

        # for iso matching
        if data_type=='faust':
            args.opt = 'options/train/faust.yaml'
        if data_type=='scape':    
            args.opt = 'options/train/scape.yaml'

        if data_type == 'faust_aniso':
            args.opt = 'options/train/faust_aniso.yaml'
        if data_type=='scape_aniso':    
            args.opt = 'options/train/scape_aniso.yaml'

        if data_type == 'faust_gene':
            args.opt = 'options/train/faust_gene.yaml'
        if data_type=='scape_gene':    
            args.opt = 'options/train/scape_gene.yaml'
        
        # for non-iso matching
        if data_type=='smal_F':
            args.opt = 'options/train/smal_category_False.yaml'

        # if data_type=='smal':
        #     args.opt = 'options/train/smal.yaml' 
        
        if data_type=='smal':
            # args.opt = 'options_mask/train/smal.yaml' 
            args.opt = 'options/train/smal.yaml'

        if data_type=='dt4d_intra':    
            args.opt = 'options/train/dt4d_intra.yaml'
        if data_type=='dt4d':    
            args.opt = 'options/train/dt4d.yaml'

        # for topology noise 
        if data_type=='topkids':    
            args.opt = 'options/train/topkids.yaml'

        if data_type == 'topkids_diffusion': 
            args.opt = 'options/train/topkids_diffusion.yaml'

        if data_type=='topkid_Rfmnet':    
            args.opt = 'options/train/topkid_Rfmnet.yaml'
        
        if data_type =='shrec16_cuts':
            args.opt = 'options/train/shrec16_cuts.yaml'

        if data_type =='shrec16_holes':
            args.opt = 'options/train/shrec16_holes.yaml'    

    else:
        
        # iso-metric shape matching
        if data_type=='faust':  
            args.opt = 'options/test/faust.yaml'
        
        if data_type=='scape':  
            args.opt = 'options/test/scape.yaml'

        if data_type=='faust_a':  
            args.opt = 'options/test/faust_a.yaml'
    
        if data_type=='scape_a':  
            args.opt = 'options/test/scape_a.yaml'

        if data_type=='shrec19':  
            args.opt = 'options/test/shrec19.yaml'


        # non-isometric shape matching
        if data_type=='smal_F':
            args.opt = 'options/test/smal_category_False.yaml'

        if data_type=='smal':
            args.opt = 'options/test/smal.yaml'

        if data_type=='dt4d_intra':  
            args.opt = 'options/test/dt4d_intraclass.yaml'
        
        if data_type=='dt4d_inter':  
            args.opt = 'options/test/dt4d_interclass.yaml'
        
        if data_type =='shrec16_cuts':
            args.opt = 'options/test/shrec16_cuts.yaml'

        if data_type =='shrec16_holes':
            args.opt = 'options/test/shrec16_holes.yaml'   

        # topology noise
        if data_type=='topkids':    
            args.opt = 'options/test/topkids.yaml' 

        if data_type=='topkids_diffusion':    
            args.opt = 'options/test/topkids_diffusion.yaml'     

        if data_type=='topkids_mwp':
            args.opt = 'options/test/topkids_mwp.yaml'
    
    opt = parse(args.opt, root_path, is_train=is_train)

    # distributed settings
    if opt['backend'] == 'dp':
        opt['dist'] = False
        print('Backend DataParallel.', flush=True)
    elif opt['backend'] == 'ddp':
        opt['dist'] = True
        port = opt.get('port', 29500)
        init_dist(port=port)
        print('Backend DistributedDataParallel.', flush=True)
    else:
        raise ValueError(f'Invalid backend option: {opt["backend"]}, only supports "dp" and "ddp"')

    # set rank and world_size
    opt['rank'], opt['world_size'] = get_dist_info()

    # make experiment directories
    make_exp_dirs(opt)

    # set random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])  # 这个地方是什么意思呢？

    return opt
