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

joint_index = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
}


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


def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    '''
    return : image[B, C, H, W] (tenmsor)
    '''
    tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)

    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    # denormalizing
    tensor.mul_(std).add_(mean)

    return tensor


args = parse_args()
reset_config(config, args)
logger, final_output_dir, tb_log_dir = create_logger(
    config, args.cfg, 'train')

cudnn.benchmark = config.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = config.CUDNN.ENABLED

model = eval('models.'+config.MODEL.NAME+'.get_pose_net_gradcam')(
    config, is_train=False
)

this_dir = os.path.dirname(__file__)
shutil.copy2(
    os.path.join(this_dir, '../lib/models', config.MODEL.NAME + '.py'),
    final_output_dir)

gpus = [int(i) for i in config.GPUS.split(',')]
model = model.cuda()

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

# model inference start
criterion = JointsMSELoss(
    use_target_weight=config.LOSS.USE_TARGET_WEIGHT, use_gain_loss=config.LOSS.USE_GAIN_LOSS)
train_loader = iter(train_loader)

# backward print hook
def backward_hook(grad):
    print('backwarding : ', grad.shape)

input, target, target_weight, meta = next(train_loader)
def get_gradcam(model, input, hook_which=4, visualize=False):
    for i in range(1):
        input = input.cuda()

        # for {hook_which}_th activation map
        # output, additional_output = model(input, hook_which)
        output = model(input, hook_which)
        
        # per joint
        for j in range(17):
            
            # highest score
            joint_htmap = output[0, j]
            assert joint_htmap.ndim == 2
            joint_htmap_w = joint_htmap.shape[1]
            joint_htmap_h = joint_htmap.shape[0]

            point = torch.argmax(joint_htmap)
            x = point % joint_htmap_w
            y = point // joint_htmap_w
            highest_score = joint_htmap[y, x]

            # backward
            highest_score.backward(retain_graph=True)
            gradients = model.grad_return().mean(dim=([0, 2, 3]))
            model.zero_grad()
            assert gradients.ndim == 1

            #get activation which is not in the original graph
            activation = model.activation.clone()

            # multiply the weight for the activation
            for c in range(gradients.shape[0]):
                activation[:, c, :, :] *= gradients[c]

            # mean of the activation through channel axis
            heatmap = torch.nn.functional.relu(activation.mean(dim=1))
            max_val = heatmap.div(heatmap.max())
            # heatmap = heatmap / (max_val + 1e-8)
            heatmap.mean().backward(retain_graph=True)
        
        if visualize:    
            # for visualize
            img = denormalize(input[0])
            img_numpy = img.mul(255).permute(1, 2, 0).byte().cpu().numpy()

            heatmap = heatmap.mul(255).permute(1, 2, 0).byte().cpu().numpy()
            heatmap = cv2.resize(heatmap, (img_numpy.shape[1], img_numpy.shape[0]))
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = heatmap * 0.4 + img_numpy

            cv2.imwrite(f'gradcam_pics_2/{joint_index[j]}_{hook_which}_th_heatmap_gradcam.jpg', superimposed_img)


get_gradcam(model, input, hook_which=4)
