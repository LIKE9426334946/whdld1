import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = -1, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        """
        logits: (B,C,H,W)
        target: (B,H,W) long
        """
        b, c, h, w = logits.shape
        probs = F.softmax(logits, dim=1)

        # mask ignore
        if self.ignore_index >= 0:
            valid = (target != self.ignore_index)
        else:
            valid = torch.ones_like(target, dtype=torch.bool)

        # one-hot target
        target_clamped = target.clone()
        target_clamped[~valid] = 0
        target_1h = F.one_hot(target_clamped, num_classes=self.num_classes).permute(0,3,1,2).float()

        valid = valid.unsqueeze(1)  # (B,1,H,W)
        probs = probs * valid
        target_1h = target_1h * valid

        dims = (0,2,3)
        inter = (probs * target_1h).sum(dims)
        denom = probs.sum(dims) + target_1h.sum(dims)

        dice = (2 * inter + self.smooth) / (denom + self.smooth)

        # optionally ignore class itself by index (common: ignore void)
        if self.ignore_index >= 0 and self.ignore_index < self.num_classes:
            dice = torch.cat([dice[:self.ignore_index], dice[self.ignore_index+1:]], dim=0)

        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, num_classes: int, ce_weight=1.0, dice_weight=1.0, ignore_index=-1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index if ignore_index >= 0 else -100)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce_w = ce_weight
        self.dice_w = dice_weight

    def forward(self, logits, target):
        return self.ce_w * self.ce(logits, target) + self.dice_w * self.dice(logits, target)