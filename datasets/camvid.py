import os
import csv
from typing import Tuple, List, Dict, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

def read_class_dict(csv_path: str) -> Tuple[List[str], np.ndarray, Dict[Tuple[int,int,int], int]]:
    """
    Returns:
      class_names: list[str]
      colors: np.ndarray shape (C,3) uint8
      color2id: dict[(r,g,b)->class_id]
    """
    class_names = []
    colors = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            r, g, b = int(row["r"]), int(row["g"]), int(row["b"])
            class_names.append(name)
            colors.append([r, g, b])
    colors = np.array(colors, dtype=np.uint8)
    color2id = {tuple(c.tolist()): i for i, c in enumerate(colors)}
    return class_names, colors, color2id

def mask_rgb_to_id(mask_rgb: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """
    mask_rgb: (H,W,3) uint8
    colors: (C,3) uint8
    Returns: mask_id (H,W) int64
    Vectorized color matching.
    """
    h, w, _ = mask_rgb.shape
    mask_flat = mask_rgb.reshape(-1, 3).astype(np.int16)  # (N,3)
    colors_int = colors.astype(np.int16)                  # (C,3)

    # Compute match by broadcasting: (N,1,3) == (1,C,3) -> (N,C,3) -> all -> (N,C)
    matches = (mask_flat[:, None, :] == colors_int[None, :, :]).all(axis=2)
    mask_id = matches.argmax(axis=1).reshape(h, w).astype(np.int64)

    # Safety: check unmapped pixels (no True in row) -> would argmax to 0 incorrectly
    # If your dataset is clean, this is fine. Otherwise, set unknown to 0 or ignore.
    has_match = matches.any(axis=1)
    if not has_match.all():
        mask_id_flat = mask_id.reshape(-1)
        mask_id_flat[~has_match] = 0
        mask_id = mask_id_flat.reshape(h, w)

    return mask_id

class CamVidDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        class_dict_csv: str,
        transforms=None,
        img_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg")
    ):
        """
        root/
          train/, train_labels/, val/, val_labels/, test/, test_labels/
        split: "train" | "val" | "test"
        """
        assert split in ["train", "val", "test"]
        self.root = root
        self.split = split
        self.transforms = transforms

        self.class_names, self.colors, self.color2id = read_class_dict(os.path.join(root, class_dict_csv))

        img_dir = os.path.join(root, split)
        mask_dir = os.path.join(root, f"{split}_labels")

        self.img_paths = []
        for fn in sorted(os.listdir(img_dir)):
            if os.path.splitext(fn.lower())[1] in img_exts:
                self.img_paths.append(os.path.join(img_dir, fn))
        self.mask_paths = []
        for img_path in self.img_paths:
            base = os.path.splitext(os.path.basename(img_path))[0]  # 0001TP_009210
            mask_name = f"{base}_L.png"
            mask_path = os.path.join(mask_dir, mask_name)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Missing mask: {mask_path}")
            self.mask_paths.append(mask_path)

        # Basic existence check
        missing = [m for m in self.mask_paths if not os.path.exists(m)]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} masks in {mask_dir}. Example: {missing[0]}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        # img: Tensor float32 (3,H,W)
        # mask: still PIL or Tensor? We'll standardize: output mask_id Tensor long (H,W)
        if isinstance(mask, Image.Image):
            mask_np = np.array(mask, dtype=np.uint8)
        else:
            # if transforms converted mask to Tensor (H,W,3) or (3,H,W), handle it
            mask_t = mask
            if mask_t.ndim == 3 and mask_t.shape[0] == 3:
                mask_np = mask_t.permute(1,2,0).cpu().numpy().astype(np.uint8)
            elif mask_t.ndim == 3 and mask_t.shape[-1] == 3:
                mask_np = mask_t.cpu().numpy().astype(np.uint8)
            else:
                raise ValueError("Unexpected mask tensor shape, expected RGB mask.")

        mask_id = mask_rgb_to_id(mask_np, self.colors)
        mask_id = torch.from_numpy(mask_id).long()

        return img, mask_id, os.path.basename(img_path)