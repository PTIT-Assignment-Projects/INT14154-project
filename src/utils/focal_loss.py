"""
Focal Loss for handling class imbalance in multi-label classification.
Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
        return (focal_weight * bce).mean()


def smooth_bce_with_logits(logits, targets, smoothing=0.1, pos_weight=None):
    """Label smoothing BCE for better calibration."""
    targets_smoothed = targets * (1 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(
        logits, targets_smoothed, pos_weight=pos_weight
    )