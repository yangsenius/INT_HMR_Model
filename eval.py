import os
import torch
import torchvision

from lib.dataset import VideoDataset
from lib.data_utils.transforms import *
from lib.models import MAED
from lib.models.tokenpose import Token3d
from lib.core.evaluate import Evaluator
from lib.core.config import parse_args
from torch.utils.data import DataLoader


def main(cfg, args):
    print(f'...Evaluating on {args.eval_ds.lower()} {args.eval_set.lower()} set...')
    device = "cuda"

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

    print("model params:{:.3f}M (/1000^2)".format(
        sum([p.numel() for p in model.parameters()]) / 1000**2))

    if args.pretrained != '' and os.path.isfile(args.pretrained):
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        # best_performance = checkpoint['performance']
        history_best_performance = checkpoint['history_best_peformance'] \
            if 'history_best_peformance' in checkpoint else checkpoint['performance']
        state_dict = {}
        for k, w in checkpoint['state_dict'].items():
            if k.startswith('module.'):
                state_dict[k[len('module.'):]] = w
            elif k in model.state_dict():
                state_dict[k] = w
            else:
                continue
            
        temp_embedding_shape = state_dict['temporal_pos_embedding'].shape
        if model.temporal_pos_embedding.shape[1] != temp_embedding_shape[1]:
            model.temporal_pos_embedding = torch.nn.Parameter(
                torch.zeros(1, temp_embedding_shape[1], temp_embedding_shape[2]))

        # checkpoint['state_dict'] = {k[len('module.'):]: w for k, w in checkpoint['state_dict'].items() if k.startswith('module.') else}
        model.load_state_dict(state_dict, strict=False)
        print(f'==> Loaded pretrained model from {args.pretrained}...')
        print(
            f'==> History best Performance on 3DPW test set {history_best_performance}')
    else:
        print(f'{args.pretrained} is not a pretrained model!!!!')
        exit()
    
    model = model.to(device)

    transforms = torchvision.transforms.Compose([
        CropVideo(cfg.DATASET.HEIGHT, cfg.DATASET.WIDTH,
                  default_bbox_scale=cfg.EVAL.BBOX_SCALE),
        StackFrames(),
        ToTensorVideo(),
        NormalizeVideo(),
    ])

    test_db = VideoDataset(
        args.eval_ds.lower(),
        set=args.eval_set.lower(),
        transforms=transforms,
        sample_pool=cfg.EVAL.SAMPLE_POOL,
        random_sample=False, random_start=False,
        verbose=True,
        debug=cfg.DEBUG)

    test_loader = DataLoader(
        dataset=test_db,
        batch_size=cfg.EVAL.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    Evaluator().run(
        model=model,
        dataloader=test_loader,
        seqlen=cfg.EVAL.SEQLEN,
        interp=cfg.EVAL.INTERPOLATION,
        save_path=args.output_path,
        device=cfg.DEVICE,
    )


if __name__ == '__main__':
    args, cfg, cfg_file = parse_args()

    main(cfg, args)
