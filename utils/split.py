# utils/split.py
import os, random
from typing import List

def make_splits(root: str, images_dir: str, ratio=(0.8,0.1,0.1), seed: int = 42, out_dir: str = None):
    img_dir = os.path.join(root, images_dir)
    names = [os.path.splitext(fn)[0] for fn in sorted(os.listdir(img_dir))
             if fn.lower().endswith((".jpg",".jpeg",".png"))]

    random.seed(seed)
    random.shuffle(names)

    n = len(names)
    n_train = int(n * ratio[0])
    n_val = int(n * ratio[1])

    train_ids = names[:n_train]
    val_ids = names[n_train:n_train+n_val]
    test_ids = names[n_train+n_val:]

    split_dir = out_dir if out_dir is not None else os.path.join(root, "splits")
    os.makedirs(split_dir, exist_ok=True)

    def dump(path: str, items: List[str]):
        with open(path, "w") as f:
            for x in items:
                f.write(x + "\n")

    dump(os.path.join(split_dir, "train.txt"), train_ids)
    dump(os.path.join(split_dir, "val.txt"), val_ids)
    dump(os.path.join(split_dir, "test.txt"), test_ids)

    return train_ids, val_ids, test_ids, split_dir

def read_split(path: str):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]
