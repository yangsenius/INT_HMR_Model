# -*- coding: utf-8 -*-

import os
from tracemalloc import start
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
# import SummaryWriter before torchvision due to some unknown bugs related to pytorch
# See https://github.com/pytorch/pytorch/issues/30651
from torch.utils.tensorboard import SummaryWriter
import torchvision
import pprint
import random
import subprocess
import numpy as np
import torch.backends.cudnn as cudnn

import torch.distributed as dist
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.nn import SyncBatchNorm
import horovod.torch as hvd

from lib.core.loss import Loss
from lib.core.trainer import Trainer
from lib.core.config import parse_args
from lib.utils.utils import prepare_output_dir
from lib.data_utils.transforms import *
from lib.models import MAED
from lib.models.tokenpose import Token3d
from lib.dataset.loaders import get_data_loaders
from lib.utils.utils import create_logger, get_optimizer

def main(args, cfg, world_size, rank, local_rank, device, logger):

    # ========= Tensorboard Writer ========= #
    if rank == 0:
        writer = SummaryWriter(log_dir=cfg.LOGDIR)
        writer.add_text('config', pprint.pformat(cfg), 0)
    else:
        writer = None

    # ========= Data Transforms ========= #
    transforms_3d = torchvision.transforms.Compose([
        CropVideo(cfg.DATASET.HEIGHT, cfg.DATASET.WIDTH, cfg.DATASET.ROT_JITTER, cfg.DATASET.SIZE_JITTER, cfg.DATASET.RANDOM_CROP_P, cfg.DATASET.RANDOM_CROP_SIZE), 
        ColorJitterVideo(cfg.DATASET.COLOR_JITTER, cfg.DATASET.COLOR_JITTER, cfg.DATASET.COLOR_JITTER, cfg.DATASET.COLOR_JITTER),
        RandomEraseVideo(cfg.DATASET.ERASE_PROB, cfg.DATASET.ERASE_PART, cfg.DATASET.ERASE_FILL, cfg.DATASET.ERASE_KP, cfg.DATASET.ERASE_MARGIN),
        RandomHorizontalFlipVideo(cfg.DATASET.RANDOM_FLIP),
        StackFrames(),
        ToTensorVideo(),
        NormalizeVideo(),
    ])
    transforms_2d = torchvision.transforms.Compose([
        CropVideo(cfg.DATASET.HEIGHT, cfg.DATASET.WIDTH, cfg.DATASET.ROT_JITTER, cfg.DATASET.SIZE_JITTER, cfg.DATASET.RANDOM_CROP_P, cfg.DATASET.RANDOM_CROP_SIZE),
        RandomEraseVideo(cfg.DATASET.ERASE_PROB, cfg.DATASET.ERASE_PART, cfg.DATASET.ERASE_FILL, cfg.DATASET.ERASE_KP, cfg.DATASET.ERASE_MARGIN),
        RandomHorizontalFlipVideo(cfg.DATASET.RANDOM_FLIP),
        StackFrames(),
        ToTensorVideo(),
        NormalizeVideo(),
    ])
    transforms_img = torchvision.transforms.Compose([
        CropImage(cfg.DATASET.HEIGHT, cfg.DATASET.WIDTH, cfg.DATASET.ROT_JITTER, cfg.DATASET.SIZE_JITTER, cfg.DATASET.RANDOM_CROP_P, cfg.DATASET.RANDOM_CROP_SIZE),
        RandomEraseImage(cfg.DATASET.ERASE_PROB, cfg.DATASET.ERASE_PART, cfg.DATASET.ERASE_FILL, cfg.DATASET.ERASE_KP, cfg.DATASET.ERASE_MARGIN),
        RandomHorizontalFlipImage(cfg.DATASET.RANDOM_FLIP),
        ToTensorImage(),
        NormalizeImage()
    ])
    transforms_val = torchvision.transforms.Compose([
        CropVideo(cfg.DATASET.HEIGHT, cfg.DATASET.WIDTH, default_bbox_scale=cfg.EVAL.BBOX_SCALE),
        StackFrames(),
        ToTensorVideo(),
        NormalizeVideo(),
    ])
    # ========= Dataloaders ========= #
    data_loaders = get_data_loaders(cfg, transforms_3d, transforms_2d, transforms_val, transforms_img, rank, world_size, verbose=rank==0)

    # ========= Compile Loss ========= #
    loss = Loss(
        e_loss_weight=cfg.LOSS.KP_2D_W,
        e_3d_loss_weight=cfg.LOSS.KP_3D_W,
        e_pose_loss_weight=cfg.LOSS.POSE_W,
        e_shape_loss_weight=cfg.LOSS.SHAPE_W,
        e_smpl_norm_loss=cfg.LOSS.SMPL_NORM,
        e_smpl_accl_loss=cfg.LOSS.ACCL_W,
        delta_norm_loss_weight = cfg.LOSS.DELTA_NORM,
        e_temp_loss_weight=cfg.LOSS.TEMP_W,
        device=device
    )

    # ========= Initialize networks, optimizers and lr_schedulers ========= #

    model = Token3d(
        num_blocks=cfg.MODEL.ENCODER.NUM_BLOCKS,
        num_heads=cfg.MODEL.ENCODER.NUM_HEADS,
        st_mode=cfg.MODEL.ENCODER.SPA_TEMP_MODE,
        mask_ratio=cfg.MODEL.MASK_RATIO,
        temporal_layers=cfg.MODEL.TEMPORAL_LAYERS,
        temporal_num_heads=cfg.MODEL.TEMPORAL_NUM_HEADS,
        enable_temp_modeling=cfg.MODEL.ENABLE_TEMP_MODELING,
        enable_temp_embedding=cfg.MODEL.ENABLE_TEMP_EMBEDDING
    )
    
    logger.info("model params:{:.3f}M (/1000^2)".format(
        sum([p.numel() for p in model.parameters()])/1000**2))
    
    #model = SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = model.to(device)


    if args.pretrained != '' and os.path.isfile(args.pretrained):
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        best_performance = checkpoint['performance']
        his_best_performance = checkpoint['history_best_performance'] \
                if 'history_best_performance' in checkpoint else checkpoint['performance']
        if cfg.MODEL.LOAD_PRETRAINED_HEAD:
            state_dict = {}
            for k, w in checkpoint['state_dict'].items():
                if k.startswith('module.'):
                    state_dict[k[len('module.'):]] = w
                elif k in model.state_dict():
                    state_dict[k] = w
                else:
                    continue
            print("load pretrained head")
        else:
            state_dict = {}
            for k, w in checkpoint['state_dict'].items():
                if k.startswith('module.') and not 'head' in k:
                    state_dict[k[len('module.'):]] = w
                elif k in model.state_dict() and not 'head' in k:
                    state_dict[k] = w
                else:
                    continue
            print('not load pretrained head')
        model.load_state_dict(state_dict, strict=False)
        if rank==0:
            logger.info(f'=> Loaded checkpoint from {args.pretrained}...')
            logger.info(f'=> Performance on 3DPW test set {best_performance} history best {his_best_performance}')
        del checkpoint
    elif args.pretrained == '':
        if rank==0:
            logger.info('=> No checkpoint specified.')
    else:
        raise ValueError(f'{args.pretrained} is not a checkpoint path!')

    #model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)

    optimizer = get_optimizer(
        model=model,
        optim_type=cfg.TRAIN.OPTIM.OPTIM,
        lr=cfg.TRAIN.OPTIM.LR,
        weight_decay=cfg.TRAIN.OPTIM.WD,
        momentum=cfg.TRAIN.OPTIM.MOMENTUM,
    )

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    optimizer = hvd.DistributedOptimizer(
                optimizer, named_parameters=model.named_parameters(), backward_passes_per_step=16
    )   
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    
    last_epoch = -1
    best_performance = None
    start_epoch = cfg.TRAIN.START_EPOCH
    if cfg.TRAIN.RESUME:
        model_path = cfg.TRAIN.RESUME
        if osp.isfile(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            start_epoch = checkpoint['epoch']
            renamed = {}
            for k,v in checkpoint['state_dict'].items():
                if k.startswith('module.'):
                    k = k[7:]
                renamed[k] = v
            
            model.load_state_dict(renamed, strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            best_performance = checkpoint['history_best_performance'] \
                if 'history_best_performance' in checkpoint else checkpoint['performance']
            
            if 'history_best_performance' in checkpoint:
                logger.info(f"load history best_performance {best_performance}")
            else:
                logger.info(f"load last epoch performance {best_performance}")
                
            if rank == 0:
                if best_performance is not '':
                    logger.info(f"=> loaded checkpoint '{model_path}' "
                        f"(epoch {start_epoch}, history best performance {best_performance})")
                else:
                    logger.info(f"=> loaded checkpoint '{model_path}' "
                        f"(epoch {start_epoch}")
            
            del checkpoint
            last_epoch = start_epoch - 1
        else:
            if rank == 0:
                logger.info(f"=> no checkpoint found at '{model_path}'")
            
    
    warmup_lr = lambda epoch: (epoch+1) * cfg.TRAIN.OPTIM.WARMUP_FACTOR if epoch < cfg.TRAIN.OPTIM.WARMUP_EPOCH else 0.1**len([m for m in cfg.TRAIN.OPTIM.MILESTONES if m <= epoch])
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=warmup_lr,
        last_epoch=last_epoch
        )

    # ========= Start Training ========= #
    Trainer(
        # Training args
        data_loaders=data_loaders,
        model=model,
        criterion=loss,
        optimizer=optimizer,
        start_epoch=start_epoch,
        end_epoch=cfg.TRAIN.END_EPOCH,
        img_use_freq=cfg.TRAIN.IMG_USE_FREQ,
        device=device,
        lr_scheduler=lr_scheduler,
        resume=cfg.TRAIN.RESUME,
        num_iters_per_epoch=cfg.TRAIN.NUM_ITERS_PER_EPOCH,

        # Validation args
        seqlen=cfg.EVAL.SEQLEN,
        interp=cfg.EVAL.INTERPOLATION,

        # Others
        writer=writer,
        rank=rank,
        debug=cfg.DEBUG,
        logdir=cfg.LOGDIR,
        world_size=world_size,
        debug_freq=cfg.DEBUG_FREQ,
        save_freq=cfg.SAVE_FREQ,
        best_performance = best_performance
    ).fit()


if __name__ == '__main__':
    args, cfg, cfg_file = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)
    
    cfg.LOGDIR = args.logdir
    cfg.TRAIN.RESUME = args.resume
    
    logger = create_logger(cfg.LOGDIR, phase='train')
    
    # ========= Pytorch Setting & Distributed Initialization ========= #
    '''
    if "LOCAL_RANK" in os.environ: # for torch.distributed.launch
        logger.info("Starting training through torch.distributed.launch...")
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
    elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
        logger.info("Starting training through slurm scheduler...")
        world_size = int(os.environ["SLURM_NPROCS"])
        local_rank = int(os.environ['SLURM_LOCALID'])
        rank = int(os.environ['SLURM_PROCID'])
        os.environ['MASTER_ADDR'] = subprocess.check_output("scontrol show hostnames $SLURM_JOB_NODELIST", shell=True).decode("ascii").split("\n")[0]
    else:
        raise NotImplementedError("Invalid launch.")
    '''

    # horovod parallel training
    hvd.init()
    
    rank = hvd.rank()
    local_rank = hvd.local_rank()
    world_size = hvd.size()
    device = torch.device('cuda:'+ str(local_rank))
    
    torch.cuda.set_device(local_rank)
    logger.info(f"Initializing process group... World_size: {world_size}, Rank: {rank}, GPU: {local_rank}")

    #torch.cuda.set_device(local_rank)
    #logger.info(f"Initializing process group... World_size: {world_size}, Rank: {rank}, GPU: {local_rank}, Master_addr: {os.environ['MASTER_ADDR']}")
    #dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

    #device = torch.device('cuda')

    if cfg.SEED_VALUE >= 0:
        if rank==0:
            logger.info(f'Seed value for the experiment {cfg.SEED_VALUE}')
        os.environ['PYTHONHASHSEED'] = str(cfg.SEED_VALUE)
        random.seed(cfg.SEED_VALUE)
        torch.manual_seed(cfg.SEED_VALUE)
        np.random.seed(cfg.SEED_VALUE)
    if rank==0:
        logger.info(pprint.pformat(cfg))

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    main(args, cfg, world_size, rank, local_rank, device, logger)
