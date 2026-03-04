# datasets/whdld.py
import os
from typing import List, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# 类别顺序（id=0..5）：
# 0 vegetation, 1 water, 2 road, 3 building, 4 pavement, 5 bare_soil
WHDLD_CLASS_NAMES = ["vegetation", "water", "road", "building", "pavement", "bare_soil"]
WHDLD_COLORS = np.array([
    [0, 255, 0],       # vegetation
    [0, 0, 255],       # water
    [255, 255, 0],     # road
    [255, 0, 0],       # building
    [192, 192, 0],     # pavement
    [128, 128, 128],   # bare soil
], dtype=np.uint8)

def mask_rgb_to_id(mask_rgb: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """
    mask_rgb: (H,W,3) uint8
    colors: (C,3) uint8
    return: (H,W) int64
    """
    h, w, _ = mask_rgb.shape
    mask_flat = mask_rgb.reshape(-1, 3).astype(np.int16)
    colors_int = colors.astype(np.int16)

    matches = (mask_flat[:, None, :] == colors_int[None, :, :]).all(axis=2)  # (N,C)
    has_match = matches.any(axis=1)

    mask_id = matches.argmax(axis=1).reshape(h, w).astype(np.int64)

    # 若出现未知颜色（理论上不应出现），这里先置为0（vegetation）
    if not has_match.all():
        mask_id_flat = mask_id.reshape(-1)
        mask_id_flat[~has_match] = 0
        mask_id = mask_id_flat.reshape(h, w)

    return mask_id

class WHDLDDataset(Dataset):
    def __init__(
        self,
        root: str,
        split_files: List[str],          # ["wh0001", ...] 或带后缀都行
        images_dir: str = "Images",
        masks_dir: str = "ImagesPNG",
        transforms=None,
        colors: Optional[np.ndarray] = None,
    ):
        self.root = root
        self.images_dir = os.path.join(root, images_dir)
        self.masks_dir = os.path.join(root, masks_dir)
        self.transforms = transforms

        self.class_names = WHDLD_CLASS_NAMES
        self.colors = (colors if colors is not None else WHDLD_COLORS).astype(np.uint8)
        self.num_classes = len(self.class_names)

        # ids: 不带后缀
        self.ids = [os.path.splitext(os.path.basename(f))[0] for f in split_files]

        self.img_paths = [os.path.join(self.images_dir, f"{i}.jpg") for i in self.ids]
        self.mask_paths = [os.path.join(self.masks_dir, f"{i}.png") for i in self.ids]

        # quick check
        if len(self.ids) == 0:
            raise ValueError("Empty split_files.")
        for p in self.img_paths[:5]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Image not found: {p}")
        for p in self.mask_paths[:5]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Mask not found: {p}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        # mask -> numpy RGB
        if isinstance(mask, Image.Image):
            mask_np = np.array(mask, dtype=np.uint8)
        else:
            m = mask
            if m.ndim == 3 and m.shape[0] == 3:
                mask_np = m.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            elif m.ndim == 3 and m.shape[-1] == 3:
                mask_np = m.cpu().numpy().astype(np.uint8)
            else:
                raise ValueError("Unexpected mask tensor shape.")

        mask_id = mask_rgb_to_id(mask_np, self.colors)
        mask_id = torch.from_numpy(mask_id).long()

        return img, mask_id, os.path.basename(img_path)