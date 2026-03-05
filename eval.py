import os
import yaml
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from datasets.transforms import build_transforms
from models.unet_resnet_attn import UNetResNet34Attn
from losses import CombinedLoss
from utils.metrics import ConfusionMatrix

from utils.split import make_splits, read_split
from datasets.whdld import WHDLDDataset, WHDLD_COLORS

def load_cfg(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(split="test"):
    cfg = load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = cfg["data"]["dataset"]
    if dataset_name != "whdld":
        raise ValueError("This eval.py is configured for WHDLD. If you need CamVid too, tell me and I merge both.")

    root = cfg["data"]["root"]
    images_dir = cfg["data"]["images_dir"]
    masks_dir = cfg["data"]["masks_dir"]
    ratio = tuple(cfg["data"]["split_ratio"])
    img_size = tuple(cfg["data"]["img_size"])

    # ✅ split 文件统一放在 runs/splits（可写/可读）
    split_dir = os.path.join(cfg["train"]["save_dir"], "splits")
    os.makedirs(split_dir, exist_ok=True)

    tr_path = os.path.join(split_dir, "train.txt")
    va_path = os.path.join(split_dir, "val.txt")
    te_path = os.path.join(split_dir, "test.txt")

    # 如果 splits 不存在，就生成到 runs/splits
    if not (os.path.exists(tr_path) and os.path.exists(va_path) and os.path.exists(te_path)):
        make_splits(root=root, images_dir=images_dir, ratio=ratio, seed=cfg["seed"], out_dir=split_dir)

    if split == "train":
        ids = read_split(tr_path)
    elif split == "val":
        ids = read_split(va_path)
    elif split == "test":
        ids = read_split(te_path)
    else:
        raise ValueError("split must be one of: train/val/test")

    # checkpoint：默认 runs/best.pt
    ckpt_path = cfg["eval"]["checkpoint"] or os.path.join(cfg["train"]["save_dir"], "best.pt")
    print("Using checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # colors 优先从 checkpoint 取，取不到就用固定 WHDLD_COLORS
    colors = ckpt.get("colors", WHDLD_COLORS)

    # dataset & dataloader
    tf = build_transforms(split, img_size)
    ds = WHDLDDataset(
        root=root,
        split_files=ids,
        images_dir=images_dir,
        masks_dir=masks_dir,
        transforms=tf,
        colors=colors
    )
    num_classes = 6

    ignore_index = cfg["loss"]["ignore_index"]
    if ignore_index < 0 or ignore_index >= num_classes:
        ignore_index = -1

    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)

    # model
    model = UNetResNet34Attn(
        num_classes=num_classes,
        simam_in_encoder=cfg["model"]["simam_in_encoder"],
        cbam_in_decoder=cfg["model"]["cbam_in_decoder"],
        pretrained=False
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    criterion = CombinedLoss(
        num_classes=num_classes,
        ce_weight=cfg["loss"]["ce_weight"],
        dice_weight=cfg["loss"]["dice_weight"],
        ignore_index=ignore_index
    ).to(device)

    cm = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
    total_loss = 0.0

    with torch.no_grad():
        for img, mask_id, _ in tqdm(dl, desc=f"Eval [{split}]"):
            img = img.to(device)
            mask_id = mask_id.to(device)

            logits = model(img)
            total_loss += criterion(logits, mask_id).item()

            pred = logits.argmax(dim=1)
            cm.update(pred.cpu(), mask_id.cpu())

    stats = cm.compute()
    total_loss /= max(1, len(dl))
    print(f"[{split.upper()}] loss={total_loss:.4f} mIoU={stats['miou']:.4f} mPA={stats['mpa']:.4f} "
          f"P={stats['precision']:.4f} R={stats['recall']:.4f}")

    # 可选：打印每类 IoU / PA
    class_names = ["vegetation", "water", "road", "building", "pavement", "bare_soil"]
    for i in range(num_classes):
        print(f"  [{i}] {class_names[i]:<12s} IoU={stats['iou_per_class'][i]:.4f}  PA={stats['pa_per_class'][i]:.4f}")

if __name__ == "__main__":
    main("test")
