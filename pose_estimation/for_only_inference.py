from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import torchvision.models as model_zoo

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import JointsMSELoss
from core.function_classification_loss import train
from core.function import validate
from core.inference import get_max_preds, get_final_preds, get_final_preds_test
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from core.evaluate import accuracy, accuracy_test


import dataset
import models

from utils.vis import save_batch_image_with_joints_original_size, save_batch_image_with_joints, save_batch_heatmaps
import cv2
import csv
import numpy as np
import torch.nn.functional as F

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

def get_batch_mask(heatmap_pred, target_weight, omega=10, face_sigma=0.95, body_sigma=0.7, order=4):
        '''
        heatmap_pred : [B, N, H, W]
        target_weight : [B, N, 1]
        
        return : mask(B, 1, H, W) (tensor, same size with image)
        '''
        # get visible joints
        batch_size, joint_num, height, width = heatmap_pred.shape[:]
        visible_joints = (target_weight == 1.).float().view(
            batch_size, joint_num, 1, 1)

        # for only visible joint
        heatmap_pred *= visible_joints
        heatmap_pred_face = torch.sum(heatmap_pred[:, :5], dim=1)
        heatmap_pred_body = torch.sum(heatmap_pred[:, 5:], dim=1)

        # GAIN eq.4
        denom_face = 1 + (-omega * (heatmap_pred_face - face_sigma)).exp()
        denom_body = 1 + (-omega * (heatmap_pred_body - body_sigma)).exp()
        face_mask = 1 / denom_face
        body_mask = 1 / denom_body

        return torch.unsqueeze(face_mask, dim=1), torch.unsqueeze(body_mask, dim=1)

def get_masked_image(batch_image, heatmap, target_weight, target_resolution=224):
    htmap_h, htmap_w = heatmap.shape[-2:]
    # batch_image = F.interpolate(batch_image,size=(htmap_h, htmap_w))
    batch_image = F.interpolate(batch_image,size=(target_resolution, target_resolution), mode='bilinear')
    
    face_mask, body_mask = get_batch_mask(heatmap, target_weight)
    mask = F.interpolate(face_mask + body_mask, (target_resolution, target_resolution), mode='bilinear')
    masked_image = mask * batch_image
    
    # return F.interpolate(masked_image, (target_resolution, target_resolution))
    return masked_image


args = parse_args()
reset_config(config, args)
logger, final_output_dir, tb_log_dir = create_logger(
    config, args.cfg, 'train')

cudnn.benchmark = config.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = config.CUDNN.ENABLED

# model = eval('models.'+config.MODEL.NAME+'.get_pose_net_second_deconv')(
#     config, is_train=True
# ).cuda()
# model = eval('models.'+config.MODEL.NAME+'.get_pose_net_eca')(
#     config, is_train=False
# ).cuda()
model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
    config, is_train=False
).cuda()

if config.TEST.MODEL_FILE:
    logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
    weight = torch.load(config.TEST.MODEL_FILE)
    weight_keys = weight.keys()
    ## 내가 학습한 모델은 key에 modul.이 붙어 있음
    if 'module' in list(weight.keys())[0]:
        new_weight = dict()
        for key in list(weight_keys):
            new_weight[key[7:]] = weight[key]
        model.load_state_dict(new_weight)
    ## 내가 학습한 모델은 key에 modul.이 붙어 있음
    else:
        # model.load_state_dict(weight['state_dict'])
        model.load_state_dict(weight)
else:
    model_state_file = os.path.join(final_output_dir,
                                    'final_state.pth.tar')
    logger.info('=> loading model from {}'.format(model_state_file))
    model.load_state_dict(torch.load(model_state_file))


second_deconv = eval('models.'+config.MODEL.NAME+'.get_second_deconv')(
        config, pretrained='output/coco/pose_resnet_50/256x192_d256x3_adam_lr1e-3/2021-02-15-12-13/model_best.pth.tar'
    ).cuda()

# classification_model = model_zoo.resnet50(pretrained=False)
# classification_model.load_state_dict(torch.load('models/resnet50-19c8e357.pth'), strict=True)
# classification_model = classification_model.cuda()
# classification_model.requires_grad = False
# classification_model.eval()

# class Resnetperceptual(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 
#         self.init_layer = nn.Sequential(*[
#             classification_model.conv1.eval(), 
#             classification_model.bn1.eval(), 
#         ])
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         blocks = []
#         blocks.append(classification_model.layer1.eval())
#         blocks.append(classification_model.layer2.eval())
#         blocks.append(classification_model.layer3.eval())
#         blocks.append(classification_model.layer4.eval())
#         for bl in blocks:
#             for p in bl:
#                 p.requires_grad = False
#         self.blocks = nn.ModuleList(blocks)
    
#     def forward(self, input, target):
#         loss = 0.0
#         x = self.init_layer(input)
#         y = self.init_layer(target)
        
#         x = self.relu(x)
#         y = self.relu(y)
        
#         x = self.pool(x)
#         y = self.pool(y)
        
#         for block in self.blocks:
#             x = block(x)
#             y = block(y)
#             loss += torch.nn.functional.l1_loss(x, y)
        
#         return loss

this_dir = os.path.dirname(__file__)
shutil.copy2(
    os.path.join(this_dir, '../lib/models', config.MODEL.NAME + '.py'),
    final_output_dir)

gpus = [int(i) for i in config.GPUS.split(',')]
# model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

valid_dataset = eval('dataset.'+config.DATASET.DATASET+'_posefix')(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=config.TEST.BATCH_SIZE*len(gpus),
    shuffle=False,
    num_workers=config.WORKERS,
    pin_memory=True
)
# train_dataset = eval('dataset.'+config.DATASET.DATASET)(
#     config,
#     config.DATASET.ROOT,
#     config.DATASET.TRAIN_SET,
#     True,
#     transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#     ])
# )

# train_loader = torch.utils.data.DataLoader(
#     train_dataset,
#     batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
#     shuffle=config.TRAIN.SHUFFLE,
#     num_workers=config.WORKERS,
#     pin_memory=True
# )

# optimizer = get_optimizer(config, second_deconv)

# model inference start
criterion = JointsMSELoss(
    use_target_weight=config.LOSS.USE_TARGET_WEIGHT, use_gain_loss=config.LOSS.USE_GAIN_LOSS)
# train_loader = iter(train_loader)
valid_loader = iter(valid_loader)

mse_loss = nn.MSELoss()

dists = []
# model.train()
model.eval()
# p_loss = Resnetperceptual()

for i in range(len(valid_loader)):
    input, target, target_weight, meta, one_hot = next(valid_loader)
    
    target_np = target[0, 0].unsqueeze(dim=0).detach().permute(1,2,0).mul(255.).cpu().numpy()
    onehot_np = one_hot[0, 0].unsqueeze(dim=0).detach().permute(1,2,0).mul(255.).cpu().numpy()
    joints = meta['joints']


    input = input.cuda()
    with torch.no_grad():
        output = model(input)
        # output, img_feature = model(input)
        # output = output * second_deconv(img_feature)
    
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        gt_coords = (meta['joints'] / input.shape[2] * output.shape[2]).cuda()
        loss = criterion.get_coord_loss(output, gt_coords, target_weight)
    
    # masked_image_gt = get_masked_image(input, target, target_weight)
    # masked_image_pred = get_masked_image(input, output, target_weight)
    
    # gt_save = denormalize(masked_image_gt.clone().detach())
    # pred_save = denormalize(masked_image_pred.clone().detach())
    # pred_save = pred_save.detach().permute(0,2,3,1).mul(255).byte().cpu().numpy()
    # cv2.imwrite('pred.jpg', pred_save[0])
    
    # gt_score = classification_model(masked_image_gt)
    # pred_score = classification_model(masked_image_pred)
    
    # loss = criterion(output, target, target_weight)
    # loss += p_loss(masked_image_gt, masked_image_pred)
    
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    
    # print('::::', classification_model.conv1.weight)
    
    
    # target_wider = target_wider.cuda(non_blocking=True)
    # loss = criterion(output, target, target_weight)

    # _, gradcam_gt, gradcam_pred = criterion.get_gradcam_loss_partly(
    #     model, output, target_wider, target_weight, visualize=True)
    # for k in range(17):
    #     max_value = gradcam_gt[0, k, :, :].min()

    # gradcam_pred = gradcam_pred.detach().mul(255).permute(0, 2, 3, 1).byte().cpu().numpy()
    # gradcam_gt = gradcam_gt.detach().mul(255).permute(0, 2, 3, 1).byte().cpu().numpy()

    # img = denormalize(input[0])
    # img_h, img_w = img.shape[-2:]
    # for j in range(17):
    #     img_numpy = img.mul(255).permute(1, 2, 0).byte().cpu().numpy()

    #     superimposed_img_pred = cv2.applyColorMap(cv2.resize(np.expand_dims(
    #         gradcam_pred[0, :, :, j], axis=2), (img_w, img_h)), cv2.COLORMAP_JET) * 0.4  + img_numpy * 0.5 
    #     superimposed_img_gt = cv2.applyColorMap(cv2.resize(np.expand_dims(
    #         gradcam_gt[0, :, :, j], axis=2), (img_w, img_h)), cv2.COLORMAP_JET) *0.4 + img_numpy * 0.5

        # cv2.imwrite(
        #     f'gradcam_pics_2/{i}_th_epoch_{joint_index[j]}_all_pred.jpg', np.expand_dims(
        #     heatmap[0, :, :, j], axis=2))
        # cv2.imwrite(
        #     f'gradcam_pics_2/{i}_th_epoch_{joint_index[j]}_all_gt.jpg', np.expand_dims(
        #     gt_heatmap[0, :, :, j], axis=2))
        
        # cv2.imwrite(
        #     f'gradcam_pics_2/{i}_{joint_index[j]}_pred_0.9.jpg', superimposed_img_pred)
        # cv2.imwrite(
        #     f'gradcam_pics_2/{i}_{joint_index[j]}_gt_0.9.jpg', superimposed_img_gt)

    # batch_mask, image, de_image = get_masked_image(input, output, target_weight)
    # batch_mask = batch_mask.detach().mul(255).permute(0, 2, 3, 1).byte().cpu().numpy()
    # de_image_numpy = de_image.mul_(255).permute(0, 2, 3, 1).byte().cpu().numpy()

    # second_output = model(de_image)

    # gain_loss = criterion.get_gain_loss(second_output)
    # loss += gain_loss
    # print('loss :', loss)
    
    ## 하나씩 뽑아본 경우.
#     dist, pred = accuracy_test(output.detach().cpu().numpy(), target.detach().cpu().numpy())
#     final_pred, _ = get_final_preds_test(config, output.detach().clone().cpu().numpy(), meta['center'].numpy(), meta['scale'].numpy())
#     dists.append(dist)
    
#     image_w_joints_gt = save_batch_image_with_joints(input, meta['joints'], meta['joints_vis'], -1, nrow=1)
#     image_w_joints_pred = save_batch_image_with_joints(input, final_pred * 4, meta['joints_vis'], -1, nrow=1)
#     heatmap_gt = save_batch_heatmaps(input, target, -1)
#     heatmap_pred = save_batch_heatmaps(input, output, -1)
    
#     image_with_joints = np.concatenate((image_w_joints_gt, image_w_joints_pred), axis=1)
#     image_heatmaps = np.concatenate((heatmap_gt, heatmap_pred), axis=0)
    
#     cv2.imwrite(f'{i}_th_image.jpg', image_with_joints)
#     cv2.imwrite(f'{i}_th_heatmap.jpg', image_heatmaps)

# with open('dists_double_deconv.csv', 'w', encoding='utf-8', newline='') as file:
#     writer = csv.writer(file)
#     for d in dists:
#         writer.writerow(d)


'''
mask 씌운 이미지 저장하는 것.
for i in range(batch_mask.shape[0]):
    cv2.imwrite(f'{i}_th_batch.jpg', batch_mask[i])
    cv2.imwrite(f'{i}_th_image.jpg', de_image[i])
'''