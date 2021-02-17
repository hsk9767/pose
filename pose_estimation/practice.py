'''
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

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



model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=True
    )

cudnn.benchmark = config.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = config.CUDNN.ENABLED

checkpoint = torch.load('models/pytorch/pose_coco/pose_resnet_50_256x192.pth.tar', map_location='cpu')

model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=True
    )


model = model.cuda()
'''
# import torch
# import torch.nn as nn
# class dummy(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Conv2d(1, 1, kernel_size=1, stride=1)
#         self.conv_1 = nn.Conv2d(1, 2, kernel_size=1, stride=1)
#         self.grad = None
#     def grad_save(self, grad):
#         self.grad = grad
#     def grad_take(self):
#         return self.grad
#     def forward(self, x):
#         x = self.conv(x)
#         print('x.requires_grad : ', x.requires_grad)
#         x.register_hook(self.grad_save)
#         x = self.conv_1(x)
#         return x
#     def forward_like(self, x):
#         x = self.conv(x)
#         return x

# model = dummy()
# input_ = torch.rand(size=(1,1, 5, 5))
# input_.requires_grad = True

# output_ = model(input_)
# for i in range(2):
#     output_[0][i].mean().backward(retain_graph=True)
#     grad = model.grad_take()
#     weight_grad = model.conv.weight.grad

# print('\n\nnow\n\n')
# print(model.conv.weight)
# print(model.conv_1.weight)

# output__ = model.forward_like(input_)
# output__[0].mean().backward()
# grad=model.grad_take()
# weight_grad_ = model.conv.weight.grad
# print('end')
# for i in range(2):
#     output_[i].backward(retain_graph=True)
#     grad = model.grad_take()
#     output_2 = model(input_).detach()
#     grad = model.grad_take()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from cnn_finetune import make_model

def to_one_hot_like(tensor):
    '''
    tensor : [B, H, W] -> [B, H, W] which is composed of 1 for max_val & 0 for the others
    '''
    assert tensor.ndim == 3
    batch_size = tensor.shape[0]
    
    one_hot_like_tensor = torch.zeros_like(tensor)
    tensor_reshaped = tensor.reshape((batch_size, -1))
    max_vals = tensor_reshaped.max(dim=1)[0]
        
    for b in range(batch_size):
        max_val = max_vals[b]
        one_hot_like_tensor[b] = (tensor[b] >= max_val * 0.9).float()
        one_hot_like_tensor[b] /= one_hot_like_tensor[b].sum()
        print(one_hot_like_tensor[b].sum())
    
    return one_hot_like_tensor


# output_for_argmax = torch.rand(size=(2, 17, 64, 48))
# gradients = output_for_argmax.mean(dim=[2,3], keepdims=True)
# print(gradients[0,0])


# one_hot = torch.ones_like(output_for_argmax)
# batch_size, joint_num = output_for_argmax.shape[:2]
# for b in range(batch_size):
#     for j in range(joint_num):
#         max_val = output_for_argmax[b,j].max()
#         one_hot[b,j] = (output_for_argmax[b,j] >= max_val).float()


input_ = torch.rand(size=(10, 64, 48))
input__ = torch.Tensor([1,2,3,4])
input__.requires_grad = True

# output_ = input__ * 2 * 6
# output_.backward(gradient=torch.Tensor([0.5, 0.5, 0, 0]))
# print(input__.grad)
one_hot = to_one_hot_like(input_)


