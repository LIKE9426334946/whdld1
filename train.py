import os
import yaml
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils.seed import set_seed
from datasets.transforms import build_transforms
from models.unet_resnet_attn import UNetResNet34Attn
from losses import CombinedLoss
from utils.metrics import ConfusionMatrix
from utils.visualize import save_prediction_vis

from utils.split import make_splits, read_split
from datasets.whdld import WHDLDDataset, WHDLD_COLORS

def load_cfg(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_datasets(cfg):
    dataset_name = cfg["data"]["dataset"]
    root = cfg["data"]["root"]
    img_size = tuple(cfg["data"]["img_size"])

    tf_train = build_transforms("train", img_size)
    tf_val = build_transforms("val", img_size)

    if dataset_name == "camvid":
        from datasets.camvid import CamVidDataset
        class_dict = cfg["data"]["class_dict"]
        ds_train = CamVidDataset(root=root, split="train", class_dict_csv=class_dict, transforms=tf_train)
        ds_val = CamVidDataset(root=root, split="val", class_dict_csv=class_dict, transforms=tf_val)
        num_classes = len(ds_train.class_names)
        colors = ds_train.colors

    elif dataset_name == "whdld":
        images_dir = cfg["data"]["images_dir"]
        masks_dir  = cfg["data"]["masks_dir"]
        ratio = tuple(cfg["data"]["split_ratio"])
    
        # ✅ 写到 runs/splits（Kaggle可写）
        split_dir = os.path.join(cfg["train"]["save_dir"], "splits")
        os.makedirs(split_dir, exist_ok=True)
    
        tr_path = os.path.join(split_dir, "train.txt")
        va_path = os.path.join(split_dir, "val.txt")
        te_path = os.path.join(split_dir, "test.txt")
    
        if not (os.path.exists(tr_path) and os.path.exists(va_path) and os.path.exists(te_path)):
            make_splits(root=root, images_dir=images_dir, ratio=ratio, seed=cfg["seed"], out_dir=split_dir)
    
        train_ids = read_split(tr_path)
        val_ids   = read_split(va_path)
    
        ds_train = WHDLDDataset(
            root=root, split_files=train_ids,
            images_dir=images_dir, masks_dir=masks_dir,
            transforms=tf_train, colors=WHDLD_COLORS
        )
        ds_val = WHDLDDataset(
            root=root, split_files=val_ids,
            images_dir=images_dir, masks_dir=masks_dir,
            transforms=tf_val, colors=WHDLD_COLORS
        )
    
        num_classes = 6
        colors = WHDLD_COLORS

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return ds_train, ds_val, num_classes, colors

def main():
    cfg = load_cfg()
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train, ds_val, num_classes, colors = build_datasets(cfg)

    ignore_index = cfg["loss"]["ignore_index"]
    if ignore_index < 0 or ignore_index >= num_classes:
        ignore_index = -1

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True
    )

    model = UNetResNet34Attn(
        num_classes=num_classes,
        simam_in_encoder=cfg["model"]["simam_in_encoder"],
        cbam_in_decoder=cfg["model"]["cbam_in_decoder"],
        pretrained=True
    ).to(device)

    criterion = CombinedLoss(
        num_classes=num_classes,
        ce_weight=cfg["loss"]["ce_weight"],
        dice_weight=cfg["loss"]["dice_weight"],
        ignore_index=ignore_index
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])

    use_amp = bool(cfg["train"]["amp"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    save_dir = cfg["train"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best.pt")
    last_path = os.path.join(save_dir, "last.pt")
    vis_dir = os.path.join(save_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    best_metric = -1.0
    best_metric_name = cfg["train"]["save_best_metric"]

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        # ---- train ----
        model.train()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{cfg['train']['epochs']} [train]")
        running = 0.0

        for it, (img, mask_id, _) in enumerate(pbar, 1):
            img = img.to(device, non_blocking=True)
            mask_id = mask_id.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(img)
                loss = criterion(logits, mask_id)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            if it % cfg["train"]["log_interval"] == 0:
                pbar.set_postfix(loss=running / it, lr=opt.param_groups[0]["lr"])

        sched.step()

        # ---- val ----
        model.eval()
        cm = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
        val_loss = 0.0

        with torch.no_grad():
            for vi, (img, mask_id, name) in enumerate(tqdm(dl_val, desc=f"Epoch {epoch} [val]"), 1):
                img = img.to(device)
                mask_id = mask_id.to(device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(img)
                    loss = criterion(logits, mask_id)
                val_loss += loss.item()

                pred = logits.argmax(dim=1)
                cm.update(pred.cpu(), mask_id.cpu())

                if vi <= 8:
                    save_prediction_vis(
                        save_path=os.path.join(vis_dir, f"ep{epoch:03d}_{name[0]}"),
                        image_tensor=img[0].cpu(),
                        gt_id=mask_id[0].cpu(),
                        pred_id=pred[0].cpu(),
                        colors=colors,
                        alpha=0.5
                    )

        stats = cm.compute()
        val_loss /= max(1, len(dl_val))

        print(f"[VAL] epoch={epoch} loss={val_loss:.4f} "
              f"mIoU={stats['miou']:.4f} mPA={stats['mpa']:.4f} "
              f"P={stats['precision']:.4f} R={stats['recall']:.4f}")

        # save last
        torch.save({"model": model.state_dict(), "epoch": epoch, "stats": stats, "cfg": cfg, "colors": colors}, last_path)

        # save best
        current = stats["miou"] if best_metric_name == "miou" else stats["mpa"]
        if current > best_metric:
            best_metric = current
            torch.save({"model": model.state_dict(), "epoch": epoch, "stats": stats, "cfg": cfg, "colors": colors}, best_path)
            print(f"Saved best to {best_path} ({best_metric_name}={best_metric:.4f})")

if __name__ == "__main__":

    main()
