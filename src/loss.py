import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalDiceLoss(nn.Module):
    def __init__(self, num_classes=4, gamma=2.0, alpha=0.25, dice_weight=0.5, smooth=1e-5):
        super(FocalDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, preds, targets):
        log_probs = F.log_softmax(preds, dim=1)
        probs = torch.exp(log_probs)
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        ce_loss = -(self.alpha * targets_onehot * (1 - probs) ** self.gamma * log_probs)
        focal_loss = ce_loss.sum(dim=1).mean()

        pred_soft = probs
        intersection = (pred_soft * targets_onehot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))
        dice_loss = 1 - ((2 * intersection + self.smooth) / (union + self.smooth)).mean()

        total_loss = (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss
        return total_loss
