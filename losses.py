import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dice_score import dice_loss

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=2.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, true):
        pred = torch.sigmoid(pred)
        
        # Calculate true positives, false positives, and false negatives
        tp = (pred * true).sum()
        fp = (pred * (1 - true)).sum()
        fn = ((1 - pred) * true).sum()
        
        # Calculate Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Apply focal weighting
        return (1 - tversky) ** self.gamma

class CombinedLoss(nn.Module):
    def __init__(self, pos_weight=15.0, ft_weight=0.7, dice_weight=0.3):
        super().__init__()
        self.pos_weight = pos_weight
        self.ft_weight = ft_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.ft = FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=2.5)

    def forward(self, pred, true):
        # Calculate individual losses
        bce_loss = self.bce(pred, true)
        ft_loss = self.ft(pred, true)
        dice = dice_loss(pred, true)
        
        # Dynamic weighting: penalize over-segmentation more
        with torch.no_grad():
            pred_sigmoid = torch.sigmoid(pred)
            pred_ratio = (pred_sigmoid > 0.5).float().mean()
            true_ratio = true.float().mean()
            
            if pred_ratio > true_ratio * 1.5:  # Over-predicting foreground
                ft_weight = max(0.8, self.ft_weight + 0.2)
                dice_weight = max(0.4, self.dice_weight + 0.1)
            elif pred_ratio < true_ratio * 0.5:  # Under-predicting foreground
                ft_weight = min(0.6, self.ft_weight - 0.1)
                dice_weight = min(0.2, self.dice_weight - 0.1)
            else:
                ft_weight = self.ft_weight
                dice_weight = self.dice_weight
        
        # Combine losses with dynamic weights
        total_loss = (1 - ft_weight - dice_weight) * bce_loss + \
                    ft_weight * ft_loss + \
                    dice_weight * dice
        
        return total_loss 