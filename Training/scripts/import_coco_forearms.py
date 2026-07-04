"""
Derive forearm training data from COCO person keypoints — the only bulk public
source of REAL proximal/distal keypoint supervision (elbow -> wrist).

Use as STAGE-1 pretraining only (see README "Two-stage recipe"): these are
human forearms labeled as the arm class, which teaches the keypoint head the
elbow->wrist axis. The final model must then be fine-tuned WITHOUT this data
(mannequin captures + negatives) so it stops firing on real arms.

Get the source files (val2017 is plenty at ~2.7k person images):
    http://images.cocodataset.org/zips/val2017.zip
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip

    python scripts/import_coco_forearms.py \
        --images ~/datasets/coco/val2017 \
        --annotations ~/datasets/coco/annotations/person_keypoints_val2017.json \
        --max 800

For each person with a visible elbow+wrist pair, emits one arm instance:
box around the forearm segment (padded), kpt0 = elbow (proximal, v=2),
kpt1 = wrist (distal, v=2). Images -> data/raw/ prefixed coco__.
"""
import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RAW = DATA / "raw"
LABELS = DATA / "labels"

# COCO keypoint indices.
L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 7, 8, 9, 10


def forearm_line(ex, ey, wx, wy, img_w, img_h, min_len_px):
    import math
    length = math.hypot(wx - ex, wy - ey)
    if length < min_len_px:
        return None

    # Box: forearm segment padded ~22% of its length (arm thickness + margin).
    pad = max(0.22 * length, 12.0)
    x0 = max(0.0, min(ex, wx) - pad)
    y0 = max(0.0, min(ey, wy) - pad)
    x1 = min(img_w - 1.0, max(ex, wx) + pad)
    y1 = min(img_h - 1.0, max(ey, wy) + pad)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None

    cx, cy = (x0 + x1) / 2 / img_w, (y0 + y1) / 2 / img_h
    bw, bh = (x1 - x0) / img_w, (y1 - y0) / img_h
    return (f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} "
            f"{ex / img_w:.6f} {ey / img_h:.6f} 2 "
            f"{wx / img_w:.6f} {wy / img_h:.6f} 2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="COCO images dir (e.g. val2017/)")
    ap.add_argument("--annotations", required=True,
                    help="person_keypoints_*.json matching the images dir")
    ap.add_argument("--max", type=int, default=800, help="cap imported images (0 = all)")
    ap.add_argument("--min-arm-px", type=float, default=28.0,
                    help="skip forearms shorter than this in pixels")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images_dir = Path(args.images).expanduser()
    ann_path = Path(args.annotations).expanduser()
    if not images_dir.is_dir():
        raise SystemExit(f"{images_dir} is not a directory.")
    if not ann_path.exists():
        raise SystemExit(f"{ann_path} not found.")

    print(f"Loading {ann_path.name} ...")
    coco = json.loads(ann_path.read_text())
    images = {im["id"]: im for im in coco["images"]}

    per_image = defaultdict(list)
    for ann in coco["annotations"]:
        kpts = ann.get("keypoints")
        if not kpts or ann.get("num_keypoints", 0) == 0:
            continue
        im = images.get(ann["image_id"])
        if im is None:
            continue
        for elbow, wrist in ((L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST)):
            ev, wv = kpts[elbow * 3 + 2], kpts[wrist * 3 + 2]
            if ev != 2 or wv != 2:      # both must be labeled AND visible
                continue
            line = forearm_line(kpts[elbow * 3], kpts[elbow * 3 + 1],
                                kpts[wrist * 3], kpts[wrist * 3 + 1],
                                im["width"], im["height"], args.min_arm_px)
            if line:
                per_image[ann["image_id"]].append(line)

    image_ids = list(per_image.keys())
    random.seed(args.seed)
    random.shuffle(image_ids)
    if args.max:
        image_ids = image_ids[: args.max]

    RAW.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)

    imported = arms = missing = 0
    for image_id in image_ids:
        im = images[image_id]
        src = images_dir / im["file_name"]
        if not src.exists():
            missing += 1
            continue
        name = f"coco__{src.stem}"
        shutil.copy2(src, RAW / f"{name}{src.suffix.lower()}")
        lines = per_image[image_id]
        (LABELS / f"{name}.txt").write_text("\n".join(lines) + "\n")
        imported += 1
        arms += len(lines)

    print(f"Imported {imported} image(s), {arms} forearm instance(s) with REAL "
          f"elbow/wrist keypoints -> {RAW}")
    if missing:
        print(f"  {missing} annotated image(s) missing from {images_dir} (wrong split?).")
    print("Reminder: stage-1 pretraining only - fine-tune without coco__ data afterwards.")
    print("Next: python scripts/02_prepare_dataset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
