from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger

import dataset
import models

from utils.vis import save_batch_image_with_joints_original_size, get_masked_image
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers

args = parse_args()
reset_config(config, args)
logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

cudnn.benchmark = config.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = config.CUDNN.ENABLED

model = eval('models.'+config.MODEL.NAME+'.get_pose_net_practice')(
        config, is_train=False
    )

this_dir = os.path.dirname(__file__)
shutil.copy2(
    os.path.join(this_dir, '../lib/models', config.MODEL.NAME + '.py'),
    final_output_dir)
    
gpus = [int(i) for i in config.GPUS.split(',')]
model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
train_dataset = eval('dataset.'+config.DATASET.DATASET)(
    config,
    config.DATASET.ROOT,
    config.DATASET.TRAIN_SET,
    True,
    transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
)

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )

## model inference start
criterion = JointsMSELoss(use_target_weight=config.LOSS.USE_TARGET_WEIGHT, use_gain_loss = config.LOSS.USE_GAIN_LOSS)
train_loader = iter(train_loader)
for i in range(10):
    input, target, target_weight, meta = next(train_loader)
    
    input = input.cuda()
    x1, x2, x3, x4, x5, x6, x = model(input)
    
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    dtype = input.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=input.device)
    std = torch.as_tensor(std, dtype=dtype, device=input.device)
    if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    
    input = input[0].mul_(std).add_(mean).mul(255).clamp(0, 255).permute(1,2,0).byte().cpu().numpy()
    
    x = x.sum(dim=1).sub(x.min())
    x = x.div(x.max()).mul(255).clamp(0, 255).permute(1,2,0).byte().cpu().numpy()
    
    x1 = x1.sum(dim=1).sub(x1.min())
    x1 = x1.div(x1.max()).mul(255).clamp(0, 255).permute(1,2,0).byte().cpu().numpy()
    
    x2 = x2.sum(dim=1).sub(x2.min())
    x2 = x2.div(x2.max()).mul(255).clamp(0, 255).permute(1,2,0).byte().cpu().numpy()
    
    x3 = x3.sum(dim=1).sub(x3.min())
    x3 = x3.div(x3.max()).mul(255).clamp(0, 255).permute(1,2,0).byte().cpu().numpy()
    
    x4 = x4.sum(dim=1).sub(x4.min())
    x4 = x4.div(x4.max()).mul(255).clamp(0, 255).permute(1,2,0).byte().cpu().numpy()
    
    x5 = x5.sum(dim=1).sub(x5.min())
    x5 = x5.div(x5.max()).mul(255).clamp(0, 255).permute(1,2,0).byte().cpu().numpy()
    
    x6 = x6.sum(dim=1).sub(x6.min())
    x6 = x6.div(x6.max()).mul(255).clamp(0, 255).permute(1,2,0).byte().cpu().numpy()
    
    pics = [input, x1, x2, x3, x4, x5, x6, x]
    
    max_w = max(input.shape[1], x1.shape[1], x2.shape[1], x3.shape[1], x4.shape[1], x5.shape[1], x6.shape[1], x.shape[1])
    max_h = max(input.shape[0], x1.shape[0], x2.shape[0], x3.shape[0], x4.shape[0], x5.shape[0], x6.shape[0], x.shape[0])
    
    total_w = sum([input.shape[1], x1.shape[1], x2.shape[1], x3.shape[1], x4.shape[1], x5.shape[1], x6.shape[1], x.shape[1]])
    total_h = sum([input.shape[0], x1.shape[0], x2.shape[0], x3.shape[0], x4.shape[0], x5.shape[0], x6.shape[0], x.shape[0]])
    
    canvas = np.zeros(shape=(max_h, total_w, 3))
    current_w = 0
    for j in range(8):
        w = pics[j].shape[1]
        h = pics[j].shape[0]
        canvas[:h, current_w:current_w+w, :] = pics[j]
        current_w += w
    
    #imwrite
    cv2.imwrite(f'{i}_th_image.jpg', canvas)