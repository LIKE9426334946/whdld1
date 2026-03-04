import torch

class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = -1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        preds: (B,H,W) long
        target: (B,H,W) long
        """
        preds = preds.view(-1)
        target = target.view(-1)
        if self.ignore_index >= 0:
            mask = target != self.ignore_index
            preds = preds[mask]
            target = target[mask]

        k = (target >= 0) & (target < self.num_classes)
        target = target[k]
        preds = preds[k]
        idx = target * self.num_classes + preds
        mat = torch.bincount(idx, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        self.mat += mat.cpu()

    def compute(self):
        mat = self.mat.float()
        tp = torch.diag(mat)
        fp = mat.sum(0) - tp
        fn = mat.sum(1) - tp
        denom_iou = tp + fp + fn
        iou = tp / torch.clamp(denom_iou, min=1.0)

        pa = tp / torch.clamp(mat.sum(1), min=1.0)  # per-class accuracy
        precision = tp / torch.clamp(tp + fp, min=1.0)
        recall = tp / torch.clamp(tp + fn, min=1.0)

        miou = iou.mean().item()
        mpa = pa.mean().item()
        mp = precision.mean().item()
        mr = recall.mean().item()

        return {
            "miou": miou,
            "mpa": mpa,
            "precision": mp,
            "recall": mr,
            "iou_per_class": iou.tolist(),
            "pa_per_class": pa.tolist(),
            "confusion_matrix": self.mat.clone()
        }

    def reset(self):
        self.mat.zero_()