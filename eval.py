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
    root = cfg["data"]["root"]
    img_size = tuple(cfg["data"]["img_size"])
    tf = build_transforms(split, img_size)

    ckpt_path = cfg["eval"]["checkpoint"] or os.path.join(cfg["train"]["save_dir"], "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    colors = ckpt.get("colors", None)

    if dataset_name == "whdld":
        images_dir = cfg["data"]["images_dir"]
        masks_dir = cfg["data"]["masks_dir"]
        ratio = tuple(cfg["data"]["split_ratio"])

        split_dir = os.path.join(root, "splits")
        te_path = os.path.join(split_dir, "test.txt")
        if not os.path.exists(te_path):
            make_splits(root=root, images_dir=images_dir, ratio=ratio, seed=cfg["seed"])
        test_ids = read_split(te_path)

        if colors is None:
            colors = WHDLD_COLORS
        ds = WHDLDDataset(root=root, split_files=test_ids, images_dir=images_dir, masks_dir=masks_dir,
                          transforms=tf, colors=colors)
        num_classes = 6

    else:
        raise ValueError("This eval.py template focuses on WHDLD; if you need CamVid too, I can merge both.")

    ignore_index = cfg["loss"]["ignore_index"]
    if ignore_index < 0 or ignore_index >= num_classes:
        ignore_index = -1

    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)

    model = UNetResNet34Attn(
        num_classes=num_classes,
        simam_in_encoder=cfg["model"]["simam_in_encoder"],
        cbam_in_decoder=cfg["model"]["cbam_in_decoder"],
        pretrained=False
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    criterion = CombinedLoss(num_classes=num_classes,
                             ce_weight=cfg["loss"]["ce_weight"],
                             dice_weight=cfg["loss"]["dice_weight"],
                             ignore_index=ignore_index).to(device)

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

if __name__ == "__main__":
    main("test")