# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import sys
os.chdir(sys.path[0])
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import os.path as osp
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash
from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import (collect_env, get_device, get_root_logger,setup_multi_processes)
SAVEPATH='./exps/msrs_vi_ir_meanstd_ConvNext_fusion_0219'

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'

def PrintModelInfo(model):
    """Print the parameter size and shape of model detail"""
    total_params = 0
    for name, param in model.named_parameters():
        num_params = torch.prod(torch.tensor(param.shape)).item() * param.element_size() / (1024 * 1024)  # 转换为MB
        print(f"{name}: {num_params:.4f} MB, Shape: {param.shape}")
        total_params += num_params
    print(f"Total number of parameters: {total_params:.4f} MB")  
    
def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--work-dir', help='the dir to save logs and models',default=SAVEPATH)
    parser.add_argument('--config', help='train config file path',
                        default="./configs/ddp_config.py")
    parser.add_argument('--load-from', help='the checkpoint file to load weights from',
                        default="./exps/Done/msrs_vi_ir_meanstd_Convnetinput4_8102/best_mIoU_iter_64000.pth")
    
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--no-validate',action='store_true',help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus',type=int,help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)',default=1)
    group_gpus.add_argument('--gpu-ids',type=int,nargs='+',help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-id',type=int,default=[0,1],help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--diff_seed',action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument('--deterministic',action='store_true',
        help='whether to set deterministic options for CUDNN backend.',default=True)
    parser.add_argument('--options',nargs='+',action=DictAction)
    parser.add_argument('--cfg-options',nargs='+',action=DictAction)
    parser.add_argument('--launcher',choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--auto-resume',action='store_true',
        help='resume from the latest checkpoint automatically.')
    args = parser.parse_args()
    os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):# set cudnn_benchmark
        torch.backends.cudnn.benchmark = True
    cfg.work_dir = args.work_dir  # update configs according to CLI args if args.work_dir is not None
    cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpu_ids = range(1)
    cfg.auto_resume = args.auto_resume
    distributed = True
    init_dist(args.launcher, **cfg.dist_params)
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    setup_multi_processes(cfg)
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    meta['env_info'] = env_info
    logger.info(f'Distributed training: {distributed}')
    # set random seeds
    cfg.device = get_device()
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    PrintModelInfo(model)
    #model.init_weights()
    datasets = [build_dataset(cfg.data.train)]
    cfg.checkpoint_config.meta = dict(
        mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
        config=cfg.pretty_text,
        CLASSES=datasets[0].CLASSES,
        PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)

if __name__ == '__main__':
    main()
