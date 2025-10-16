import os
import sys
os.chdir(sys.path[0])
import argparse
import shutil
import warnings
warnings.filterwarnings("ignore", message=".*MMCV will release v2.0.0.*")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid.*indexing")
import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,)
from mmcv.utils import DictAction

from model import digit_version
from thop import profile
from model.apis import single_gpu_test, set_random_seed
from model.datasets import build_dataloader, build_dataset
from model.models import build_segmentor
from model.utils import build_difusionseg, get_device,PrintModelInfo,count_params
"""please use RTX4090 to fork the results"""
GPU=0
CONFIG='./configs/DiFusionSeg_config.py'
CHECKPOINT='./exps/BestMSRS/best.pth'
OUT="./out/seg/"

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('--config',default=CONFIG)
    parser.add_argument('--checkpoint',default=CHECKPOINT)
    parser.add_argument('--work-dir')
    parser.add_argument('--aug-test', action='store_true')
    parser.add_argument('--out')
    parser.add_argument('--format-only',action='store_true')
    parser.add_argument('--eval',type=str,nargs='+',default='mIoU')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--show-dir', default=OUT)
    parser.add_argument('--gpu-collect',action='store_true')
    parser.add_argument('--gpu-id',type=int,default=GPU)
    parser.add_argument('--tmpdir')
    parser.add_argument('--options',nargs='+',action=DictAction)
    parser.add_argument('--cfg-options',nargs='+',action=DictAction)
    parser.add_argument('--eval-options',nargs='+',action=DictAction)
    parser.add_argument('--launcher',choices=['none', 'pytorch', 'slurm', 'mpi'],default='none')
    parser.add_argument('--opacity',type=float,default=1)
    parser.add_argument('--seed',type=int,default=2002)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    set_random_seed(args.seed)

    """set cudnn_benchmark"""
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]
    """init distributed env first, since logger depends on the dist info."""
    if args.launcher == 'none':
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    """build the dataloader"""
    dataset = build_dataset(cfg.data.test)
    """The default loader config"""
    loader_cfg = dict(
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        shuffle=False)
    """The overall dataloader settings"""
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader']
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    """build the dataloader"""
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    """build the model and load checkpoint"""
    cfg.model.train_cfg = None
    cfg.device = get_device()
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg')).to(cfg.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = dataset.CLASSES
    model.PALETTE = dataset.PALETTE

    """clean gpu memory when starting a new evaluation."""
    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options
    tmpdir = None
    cfg.device = get_device()
    #if not distributed:
    if not torch.cuda.is_available():
        assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
            'Please use MMCV >= 1.4.4 for CPU training!'
    model = revert_sync_batchnorm(model)
    model = build_difusionseg(model, cfg.device, device_ids=cfg.gpu_ids)
    results = single_gpu_test(
        model,
        data_loader,
        args.show,
        args.show_dir,
        False,
        args.opacity,
        pre_eval=args.eval is not None and not False,
        format_only=args.format_only or False,
        format_args=eval_kwargs)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(results, args.out)
        if args.eval:
            eval_kwargs.update(metric=args.eval)
            metric = dataset.evaluate(results, **eval_kwargs)
            metric_dict = dict(config=args.config, metric=metric)
            if tmpdir is not None and False:
                shutil.rmtree(tmpdir)

if __name__ == '__main__':
    main()
