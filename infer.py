import os
import yaml
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as TF

from models.unet_resnet_attn import UNetResNet34Attn

MEAN = (0.485,0.456,0.406)
STD  = (0.229,0.224,0.225)

def load_cfg(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def id_to_color(mask_id: np.ndarray, colors: np.ndarray) -> np.ndarray:
    return colors[mask_id]

def main():
    cfg = load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = cfg["model"]["num_classes"]
    img_size = tuple(cfg["data"]["img_size"])
    ckpt_path = cfg["infer"]["checkpoint"] or os.path.join(cfg["train"]["save_dir"], "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    colors = ckpt.get("colors", None)
    if colors is None:
        raise ValueError("Checkpoint has no colors. Re-train or save colors in train.py.")

    model = UNetResNet34Attn(
        num_classes=num_classes,
        simam_in_encoder=cfg["model"]["simam_in_encoder"],
        cbam_in_decoder=cfg["model"]["cbam_in_decoder"],
        pretrained=False
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    in_path = cfg["infer"]["input_path"]
    out_dir = cfg["infer"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    img_pil = Image.open(in_path).convert("RGB")
    img_rs = TF.resize(img_pil, list(img_size), interpolation=TF.InterpolationMode.BILINEAR)
    x = TF.to_tensor(img_rs)
    x = TF.normalize(x, MEAN, STD).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)

    pred_rgb = id_to_color(pred, np.array(colors, dtype=np.uint8)).astype(np.uint8)

    alpha = float(cfg["infer"]["overlay_alpha"])
    img_np = np.array(img_rs, dtype=np.uint8)
    overlay = (img_np * (1-alpha) + pred_rgb * alpha).astype(np.uint8)

    base = os.path.splitext(os.path.basename(in_path))[0]
    Image.fromarray(pred_rgb).save(os.path.join(out_dir, f"{base}_pred.png"))
    Image.fromarray(overlay).save(os.path.join(out_dir, f"{base}_overlay.png"))
    img_rs.save(os.path.join(out_dir, f"{base}_input_resized.png"))

    print("Saved to:", out_dir)

if __name__ == "__main__":
    main()