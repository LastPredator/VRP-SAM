import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (B, 1, H, W) logits
        # target: (B, H, W) or (B, 1, H, W) 0/1
        
        # Ensure float32 for Dice stability
        pred = pred.float()
        pred = torch.sigmoid(pred)
        
        if target.dim() == 3:
            target = target.unsqueeze(1)
            
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        loss = 1 - (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        # pred: logits
        
        if target.dim() == 3:
            target = target.unsqueeze(1)
            
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        # Use BCEWithLogits for stability and AMP safety
        ce_loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')
        
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss
            
        return loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
    def forward(self, pred, target):
        return self.dice_weight * self.dice_loss(pred, target) + self.focal_weight * self.focal_loss(pred, target)
