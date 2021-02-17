# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as model_zoo


classification_model = model_zoo.resnet50(pretrained=False)
classification_model.load_state_dict(torch.load('models/resnet50-19c8e357.pth'), strict=True)
classification_model = classification_model.cuda()
classification_model.requires_grad = False
classification_model.eval()

class Resnetperceptual(nn.Module):
    def __init__(self):
        super().__init__()
        # 
        self.init_layer = nn.Sequential(*[
            classification_model.conv1.eval(), 
            classification_model.bn1.eval(), 
        ])
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        blocks = []
        blocks.append(classification_model.layer1.eval())
        blocks.append(classification_model.layer2.eval())
        blocks.append(classification_model.layer3.eval())
        blocks.append(classification_model.layer4.eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, input, target):
        loss = 0.0
        x = self.init_layer(input)
        y = self.init_layer(target)
        
        x = self.relu(x)
        y = self.relu(y)
        
        x = self.pool(x)
        y = self.pool(y)
        
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        
        return loss


def to_one_hot_like(tensor, threshold=1.0):
    '''
    tensor : [B, H, W] -> [B, H, W] which is composed of 1 for max_val & 0 for the others
    '''
    batch_size = tensor.shape[0]

    one_hot_like_tensor = torch.zeros_like(tensor)
    tensor_reshaped = tensor.reshape((batch_size, -1))
    max_vals = tensor_reshaped.max(dim=1)[0]

    for b in range(batch_size):
        max_val = max_vals[b]
        one_hot_like_tensor[b] = (tensor[b] >= max_val * threshold).float()
        one_hot_like_tensor[b] /= one_hot_like_tensor[b].sum()

    return one_hot_like_tensor


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight, use_gain_loss=True):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight
        self.use_gain_loss = use_gain_loss
        if self.use_gain_loss:
            self.lambda_gain_loss = 0.01
        self.head_idx = [j for j in range(5)]
        self.body_idx = [i for i in range(5, 17)]

        # adjacent joint
        self.how_much_to_adj = 1.0
        self.associated_matrix = {5: [6, 7, 9], 6: [5, 8, 9], 7: [5, 9], 8: [6, 10], 9: [5, 7], 10: [
            6, 8], 11: [12, 13, 15], 12: [11, 14, 16], 13: [11, 15], 14: [12, 16], 15: [11, 13], 16: [12, 13]}
        
        # Cross
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints
    
    def forward_(self, output, target, target_weight):
        batch_size, num_joint = output.shape[:2]
        
        output = output * target_weight.view(batch_size, num_joint, 1, 1)
        target = target * target_weight.view(batch_size, num_joint, 1, 1)
        
        pred_idx = self.get_max_pred_torch(output).cuda()
        gt_idx = self.get_max_pred_torch(target).cuda()
        
        pred_idx = pred_idx * target_weight
        gt_idx = gt_idx * target_weight
        
        # loss weight -> [B, J]
        # loss_weight = F.mse_loss(pred_idx, gt_idx, reduction='none').sum(dim=2).softmax(dim=1).cuda()
        loss_weight = F.l1_loss(pred_idx, gt_idx, reduction='none').sum(dim=2).softmax(dim=1).cuda()
        target_weight = target_weight.view(batch_size, num_joint)
        
        loss = 0.
        for idx in range(num_joint):
            heatmap_pred = output[:, idx, :, :]
            heatmap_gt = target[:, idx, :, :]
            if self.use_target_weight:
                heatmap_pred = heatmap_pred.mul(target_weight[:, idx].view(batch_size, 1, 1))
                heatmap_gt = heatmap_gt.mul(target_weight[:, idx].view(batch_size, 1, 1))
                loss += F.mse_loss(heatmap_pred, heatmap_gt, reduction='none').sum(dim=[1,2]) * \
                    loss_weight[:, idx]
            else:
                loss += F.mse_loss(heatmap_pred, heatmap_gt, reduction='none').sum(dim=[1,2]) * \
                    loss_weight[:, idx]
            
        return 0.001 * loss.mean() / num_joint
    
    def forward__(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
        
        pred_idx = self.get_max_pred_torch(output)
        gt_idx = self.get_max_pred_torch(target)
        # loss weight -> [B, J]
        # loss_weight = F.mse_loss(pred_idx, gt_idx, reduction='none').sum(dim=2).softmax(dim=1).cuda()
        loss_weight = F.l1_loss(pred_idx, gt_idx, reduction='none').sum(dim=2).softmax(dim=1).cuda()
        target_weight = target_weight * loss_weight.unsqueeze(2)

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


    def get_max_pred_torch(self, heatmap):
        B, J, H, W = heatmap.shape
        heatmap_reshaped = heatmap.reshape(shape=(B, J, -1))
        max_idx = heatmap_reshaped.max(dim=2)[1]
        
        preds = torch.zeros(size=(B, J, 2))
        preds[:, :, 0] = max_idx[:, :] % W
        preds[:, :, 1] = (max_idx[:, :] / float(W)).floor_()
        
        return preds # [B, J, 2]
        
        
    
    def get_mse(self, a, b):
        return self.criterion(a, b)

    def get_loss_separately(self, output, target, target_weight, head_weight=0.2):
        assert target_weight is not None

        output *= target_weight.unsqueeze(dim=3)
        target *= target_weight.unsqueeze(dim=3)

        heatmap_body_pred = torch.sum(output[:, self.body_idx, :, :], dim=1)
        heatmap_head_pred = torch.sum(output[:, self.head_idx, :, :], dim=1)
        heatmap_body_gt = torch.sum(target[:, self.body_idx, :, :], dim=1)
        heatmap_head_gt = torch.sum(target[:, self.head_idx, :, :], dim=1)

        loss = head_weight * self.criterion(heatmap_head_gt, heatmap_head_pred) +\
            (1-head_weight) * self.criterion(heatmap_body_gt, heatmap_body_pred)

        return loss

    def get_gain_loss(self, output, target, target_weight):
        return torch.abs(self.lambda_gain_loss * output.mean())

    def get_gradcam_loss(self, model, output, target, target_weight, visualize=False):
        batch_size, joint_num, htmap_h, htmap_w = output.shape
        loss = 0.

        if visualize:
            # placeholder
            gradcam_pred = torch.zeros(size=(batch_size, joint_num, 8, 6))
            gradcam_gt = torch.zeros(size=(batch_size, joint_num, 8, 6))

        # per joint
        for j in range(joint_num):
            joint_htmap = output[:, j]
            assert joint_htmap.ndim == 3

            # one hot tensor for gradient
            one_hot_like = to_one_hot_like(joint_htmap)

            # backward to get grad_CAM
            joint_htmap.backward(gradient=one_hot_like, retain_graph=True)
            gradients = model.grad_return().mean(dim=[2, 3], keepdim=True)
            model.zero_grad()

            # get activation which is not in the original graph
            activation = model.activation.clone()

            # multiply the weight for the activation
            activation = activation.mul(gradients)

            # mean through channel axis
            heatmap = torch.nn.functional.relu(
                activation.mean(dim=1))  # -> [B, H, W]
            assert heatmap.ndim == 3
            max_vals = heatmap.reshape(
                (batch_size, -1)).max(dim=1, keepdims=True)[0].unsqueeze(dim=2)
            heatmap = heatmap.div(max_vals + 1e-12).unsqueeze(dim=1)

            # get the gt
            gt_heatmap = target[:, j, :, :].unsqueeze(dim=1)
            _, _,  gt_h, gt_w = gt_heatmap.shape

            # scale adjust
            # heatmap = torch.nn.functional.interpolate(
            #     heatmap, size=(gt_h, gt_w), mode='bilinear')
            gt_heatmap = torch.nn.functional.interpolate(
                gt_heatmap, size=(8, 6), mode='bilinear')
            max_vals_gt = gt_heatmap.reshape(
                (batch_size, -1)).max(dim=1, keepdims=True)[0].unsqueeze(dim=2).unsqueeze(dim=3)
            gt_heatmap = gt_heatmap.div(max_vals_gt + 1e-12)

            # get loss
            if self.use_target_weight:
                joint_target_weight = target_weight[:, j].unsqueeze(
                    1).unsqueeze(2)
                loss += 0.5 * \
                    self.criterion(heatmap.mul(joint_target_weight),
                                   gt_heatmap.mul(joint_target_weight))
            else:
                loss += 0.5 * self.criterion(heatmap, gt_heatmap)

            # for visualize
            if visualize:
                gradcam_pred[:, j, :, :] = heatmap.detach().squeeze(dim=1)
                gradcam_gt[:, j, :, :] = gt_heatmap.detach().squeeze(dim=1)

        if visualize:
            return 0.01 * loss / joint_num, gradcam_gt, gradcam_pred
        else:
            return 0.01 * loss / joint_num

    def get_gradcam_loss_partly(self, model, output, target, target_weight, visualize=False):
        batch_size, joint_num, htmap_h, htmap_w = output.shape
        loss = 0.

        if visualize:
            # placeholder
            gradcam_pred = torch.zeros(size=(batch_size, joint_num, 64, 48))
            gradcam_gt = torch.zeros(size=(batch_size, joint_num, 64, 48))

        # per joint
        for j in range(joint_num):
            joint_htmap = output[:, j]
            assert joint_htmap.ndim == 3

            # one hot tensor for gradient
            one_hot_like = to_one_hot_like(joint_htmap)

            # backward to get grad_CAM
            joint_htmap.backward(gradient=one_hot_like, retain_graph=True)
            gradients = model.grad_return()
            model.zero_grad()

            # get activation which is not in the original graph
            activation = model.activation.clone()

            # multiply the weight for the activation
            activation = activation.mul(gradients)

            # mean through channel axis
            heatmap = torch.nn.functional.relu(
                activation.mean(dim=1))  # -> [B, H, W]
            assert heatmap.ndim == 3
            heatmap = torch.nn.functional.interpolate(
                heatmap.unsqueeze(dim=1), size=(htmap_h, htmap_w), mode='bilinear')
            max_vals = heatmap.reshape(
                (batch_size, -1)).max(dim=1, keepdims=True)[0].unsqueeze(dim=2)
            min_vals = heatmap.reshape(
                (batch_size, -1)).min(dim=1, keepdims=True)[0].unsqueeze(dim=2)
            heatmap = heatmap.squeeze(dim=1).sub(min_vals).div(
                max_vals + 1e-12).unsqueeze(dim=1)

            # get the gt
            # if j < 5:
            #     gt_heatmap = target[:, j, :, :].unsqueeze(dim=1)
            #     _, _,  gt_h, gt_w = gt_heatmap.shape
            # else:
            #     target_ = target * (target > 0.95).float()
            #     gt_heatmap = target_[:, self.associated_matrix[j], :, :].sum(
            #         dim=1, keepdims=True) * self.how_much_to_adj + target_[:, j, :, :].unsqueeze(dim=1)
            #     _, _,  gt_h, gt_w = gt_heatmap.shape

            # scale adjust
            # heatmap = torch.nn.functional.interpolate(
            #     heatmap, size=(gt_h, gt_w), mode='bilinear')
            # gt_heatmap = torch.nn.functional.interpolate(
            #     gt_heatmap, size=(8, 6), mode='bilinear')
            # max_vals_gt = gt_heatmap.reshape(
            #     (batch_size, -1)).max(dim=1, keepdims=True)[0].unsqueeze(dim=2).unsqueeze(dim=3)
            # gt_heatmap = gt_heatmap.div(max_vals_gt + 1e-12)

            # masking : to compare only target_part
            if visualize:
                heatmap_ = heatmap.clone().detach()
            mask = (gt_heatmap >= 0.95).float()
            heatmap = mask * heatmap
            gt_heatmap = gt_heatmap * mask

            # get loss
            if self.use_target_weight:
                joint_target_weight = target_weight[:, j].unsqueeze(
                    1).unsqueeze(2)
                loss += 0.5 * \
                    self.criterion(heatmap.mul(joint_target_weight),
                                    gt_heatmap.mul(joint_target_weight))
            else:
                loss += 0.5 * self.criterion(heatmap, gt_heatmap)

            # for visualize
            if visualize:
                gradcam_pred[:, j, :, :] = heatmap_.detach().squeeze(dim=1)
                gradcam_gt[:, j, :, :] = gt_heatmap.detach().squeeze(dim=1)

        if visualize:
            return 0.1 * loss / joint_num, gradcam_gt, gradcam_pred
        else:
            return 0.1 * loss / joint_num
    
    def get_cross_entropy_loss(self, output , target, target_weight):
        batch_size, num_joint, h, w = output.shape
        output = output.reshape(shape=(batch_size, num_joint, -1))
        target = target.reshape(shape=(batch_size, num_joint, -1))
        
        logprobs = F.log_softmax(output, dim=2)
        loss = -(logprobs * target) / output.shape[0]
        loss = loss * target_weight
        return loss.mean()
        
    def soft_argmax(self, heatmap):
        batch_size, num_joint, h, w = heatmap.shape
        heatmap = heatmap.reshape((batch_size, num_joint, -1))
        heatmap = F.softmax(heatmap * 100, dim=2)
        heatmap = heatmap.reshape((batch_size, num_joint, h, w))
        
        accu_x = heatmap.sum(dim=2)
        accu_y = heatmap.sum(dim=3)
        
        accu_x = accu_x * torch.arange(w).float().cuda()[None, None, :]
        accu_y = accu_y * torch.arange(h).float().cuda()[None, None, :]
        
        accu_x = accu_x.sum(dim=2, keepdim=True)
        accu_y = accu_y.sum(dim=2, keepdim=True)
        
        coord_out = torch.cat((accu_x, accu_y), dim=2)
        
        return coord_out
    
    def get_coord_loss(self, output, gt_coords, target_weight):
        pred_coord = self.soft_argmax(output)
        loss = torch.abs(gt_coords[:, :, :-1] - pred_coord) * target_weight
        return loss.mean()
