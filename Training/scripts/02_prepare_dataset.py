"""
Step 2 - Build the ARM-ONLY YOLO-pose dataset (train/val split + data.yaml).

Takes the images in data/raw/ and the keypoint labels written by 01_label.py
or import_arm_boxes.py (data/labels/) and lays them out the way Ultralytics
pose training expects:

    data/pose/
      images/train  images/val
      labels/train  labels/val
      data.yaml

data.yaml declares kpt_shape: [2, 3] (2 keypoints - proximal/distal - each
x/y/visibility) and the single class (arm). Only images that have a label file
are included; an empty label file is kept as a background negative.
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RAW = DATA / "raw"
LABELS = DATA / "labels"
POSE = DATA / "pose"

CLASSES = ["arm"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of images held out for validation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs = []
    for img in sorted(RAW.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = LABELS / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))

    if not pairs:
        print(f"No labeled images found. Run scripts/01_label.py first (need .txt files in {LABELS}).")
        return 1

    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_frac)) if len(pairs) > 1 else 0
    val = pairs[:n_val]
    train = pairs[n_val:]

    if POSE.exists():
        shutil.rmtree(POSE)
    for split in ("train", "val"):
        (POSE / "images" / split).mkdir(parents=True, exist_ok=True)
        (POSE / "labels" / split).mkdir(parents=True, exist_ok=True)

    def place(items, split):
        for img, lbl in items:
            shutil.copy2(img, POSE / "images" / split / img.name)
            shutil.copy2(lbl, POSE / "labels" / split / f"{img.stem}.txt")

    place(train, "train")
    place(val, "val")

    data_yaml = {
        "path": str(POSE.resolve()),
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [2, 3],         # 2 keypoints (proximal, distal), each (x, y, visibility)
        "flip_idx": [0, 1],          # collinear points, no L/R pairs -> identity
        "names": {i: n for i, n in enumerate(CLASSES)},
    }
    (POSE / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    print(f"Wrote {POSE / 'data.yaml'}")
    print(f"  train={len(train)}  val={len(val)}  classes={CLASSES}  kpt_shape={data_yaml['kpt_shape']}")
    print("Next: python scripts/03_train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
