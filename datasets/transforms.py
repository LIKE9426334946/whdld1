import random
from typing import Tuple
from PIL import Image
import torchvision.transforms.functional as TF
import torch

class ComposeDual:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class ResizeDual:
    def __init__(self, size_hw: Tuple[int,int]):
        self.h, self.w = size_hw

    def __call__(self, img, mask):
        img = TF.resize(img, [self.h, self.w], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.h, self.w], interpolation=TF.InterpolationMode.NEAREST)
        return img, mask

class RandomHorizontalFlipDual:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        return img, mask

class RandomRotationDual:
    def __init__(self, degrees=10):
        self.degrees = degrees

    def __call__(self, img, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)
        return img, mask

class RandomCropDual:
    def __init__(self, size_hw: Tuple[int,int]):
        self.th, self.tw = size_hw

    def __call__(self, img, mask):
        w, h = img.size
        if w == self.tw and h == self.th:
            return img, mask
        if w < self.tw or h < self.th:
            # pad if needed
            pad_w = max(0, self.tw - w)
            pad_h = max(0, self.th - h)
            img = TF.pad(img, [0, 0, pad_w, pad_h], fill=0)
            mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=0)
            w, h = img.size

        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)
        img = TF.crop(img, i, j, self.th, self.tw)
        mask = TF.crop(mask, i, j, self.th, self.tw)
        return img, mask

class ColorJitterImageOnly:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
        self.jitter = torch.nn.Sequential()  # placeholder, using TF.adjust_* below
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img, mask):
        # minimal jitter implementation using torchvision functional
        b = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        c = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        s = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        h = random.uniform(-self.hue, self.hue)

        img = TF.adjust_brightness(img, b)
        img = TF.adjust_contrast(img, c)
        img = TF.adjust_saturation(img, s)
        img = TF.adjust_hue(img, h)
        return img, mask

class ToTensorNormalize:
    def __init__(self, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        img = TF.to_tensor(img)
        img = TF.normalize(img, self.mean, self.std)
        # keep mask as PIL RGB here; later in dataset convert to id
        return img, mask

def build_transforms(split: str, img_size_hw: Tuple[int,int]):
    # img_size_hw: (H,W)
    if split == "train":
        return ComposeDual([
            ResizeDual(img_size_hw),
            RandomHorizontalFlipDual(0.5),
            RandomRotationDual(10),
            # 如果你想做 crop，建议 crop size = img_size_hw（相当于随机裁剪+padding效果）
            # RandomCropDual(img_size_hw),
            ColorJitterImageOnly(0.2,0.2,0.2,0.05),
            ToTensorNormalize(),
        ])
    else:
        return ComposeDual([
            ResizeDual(img_size_hw),
            ToTensorNormalize(),
        ])