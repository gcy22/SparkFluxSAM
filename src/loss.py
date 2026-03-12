import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class StructureLoss(nn.Module):
    def __init__(self, weight_dice=1.0, weight_focal=1.0, weight_bce=0.2):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.weight_bce = weight_bce

    def forward(self, pred, mask):
        smooth = 1e-5
        pred_sigmoid = torch.sigmoid(pred)

        intersection = (pred_sigmoid * mask).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
        dice_loss = 1 - (2. * intersection + smooth) / (union + smooth)
        dice_loss = dice_loss.mean()

        bce_none = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

        p_t = pred_sigmoid * mask + (1 - pred_sigmoid) * (1 - mask)

        alpha = 0.25
        gamma = 2.0
        alpha_t = alpha * mask + (1 - alpha) * (1 - mask)

        focal_loss = (alpha_t * (1 - p_t) ** gamma * bce_none).mean()

        bce_loss = bce_none.mean()

        total_loss = (self.weight_dice * dice_loss) + \
                     (self.weight_focal * focal_loss) + \
                     (self.weight_bce * bce_loss)

        return total_loss


def calculate_dice(pred_mask, gt_mask):
    smooth = 1e-5
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    score = (2. * intersection) / (union + smooth)
    return score.item()


def calculate_iou(pred_mask, gt_mask):
    smooth = 1e-5
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum() - intersection
    score = (intersection) / (union + smooth)
    return score.item()


def calculate_hd95(pred_mask_np, gt_mask_np):
    if pred_mask_np.dtype != np.uint8:
        pred_mask_np = (pred_mask_np > 0.5).astype(np.uint8)
    if gt_mask_np.dtype != np.uint8:
        gt_mask_np = (gt_mask_np > 0.5).astype(np.uint8)
    if pred_mask_np.sum() == 0 or gt_mask_np.sum() == 0:
        return 100.0

    try:
        pred_contours, _ = cv2.findContours(pred_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        gt_contours, _ = cv2.findContours(gt_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(pred_contours) == 0 or len(gt_contours) == 0:
            return 100.0

        pred_points = np.concatenate(pred_contours, axis=0).reshape(-1, 2)
        gt_points = np.concatenate(gt_contours, axis=0).reshape(-1, 2)

        if len(pred_points) == 0 or len(gt_points) == 0:
            return 100.0
        dt_gt = cv2.distanceTransform(1 - gt_mask_np, cv2.DIST_L2, 5)
        d_pred_gt = [dt_gt[pt[1], pt[0]] for pt in pred_points]  # y, x

        dt_pred = cv2.distanceTransform(1 - pred_mask_np, cv2.DIST_L2, 5)
        d_gt_pred = [dt_pred[pt[1], pt[0]] for pt in gt_points]

        all_distances = d_pred_gt + d_gt_pred

        if len(all_distances) == 0:
            return 100.0

        hd95 = np.percentile(all_distances, 95)
        return float(hd95)

    except Exception as e:
        print(f"HD95 Error: {e}")
        return 100.0