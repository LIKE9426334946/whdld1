import os
import numpy as np
from PIL import Image
import torch

def id_to_color(mask_id: np.ndarray, colors: np.ndarray) -> np.ndarray:
    # mask_id: (H,W) int
    return colors[mask_id]  # (H,W,3)

def save_prediction_vis(save_path: str, image_tensor: torch.Tensor, gt_id: torch.Tensor, pred_id: torch.Tensor,
                        colors: np.ndarray, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), alpha=0.5):
    """
    image_tensor: (3,H,W) normalized
    gt_id/pred_id: (H,W)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img = image_tensor.detach().cpu().numpy().transpose(1,2,0)
    img = (img * np.array(std) + np.array(mean))
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    gt = gt_id.detach().cpu().numpy().astype(np.int64)
    pr = pred_id.detach().cpu().numpy().astype(np.int64)

    gt_rgb = id_to_color(gt, colors).astype(np.uint8)
    pr_rgb = id_to_color(pr, colors).astype(np.uint8)
    overlay = (img * (1 - alpha) + pr_rgb * alpha).astype(np.uint8)

    # concat horizontally: img | gt | pred | overlay
    panel = np.concatenate([img, gt_rgb, pr_rgb, overlay], axis=1)
    Image.fromarray(panel).save(save_path)