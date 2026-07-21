"""
Step 3 - Close-up crop augmentation for the TRAIN split.

    python scripts/02_prepare_dataset.py      # build the split first
    python scripts/03_augment_closeups.py     # then add close-ups to train
    python scripts/05_train.py --name arm_pose_v6

WHY
---
The captured images all frame the arm from a similar distance: it covers ~3% of
the frame. In the headset it often fills a third of the view or more, and
detection falls apart there - measured on v4, sweeping the same test images
from their original framing to a close-up:

    arm 3% of frame -> 100% detected      arm 32% of frame -> 65% detected
    arm 19% of frame ->  93% detected     arm 55% of frame -> 50% detected

Ultralytics' scale augmentation cannot close that gap. `scale=0.5` resizes by a
random factor in [0.5, 1.5], so at most 1.5x linear (2.25x area) - from 3%
occupancy that reaches ~7%. Mosaic makes it worse, tiling four images together
so objects come out smaller still. This script fills the missing range offline
by cropping in on the arm and recomputing the labels.

IMPORTANT
---------
This runs AFTER 02_prepare_dataset.py and only touches the train split, so no
synthetic image can reach val or test - those must keep measuring real framing.
Re-running 02_prepare_dataset.py deletes data/pose entirely, so re-run this
afterwards. Running it twice is safe: existing zoom_* files are cleared first.

Keypoints that fall outside a crop are marked visibility 0 rather than clamped
to the edge - a wrist that is out of frame has no true position, and inventing
one at the border would teach the model to place keypoints on frame edges.
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from pose_label_io import (format_label, has_keypoints, montage, parse_label,
                           plan_crop, sharpness)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
POSE = DATA / "pose"
PREFIX = "zoom_"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def transform(instances, crop, out_w, out_h, min_visible):
    """Map instances into the crop. Returns None if the arm is mostly gone."""
    cx1, cy1, cx2, cy2 = crop
    sx = out_w / (cx2 - cx1)
    sy = out_h / (cy2 - cy1)

    kept = []
    for cls, box, kpts in instances:
        moved = np.array([(box[0] - cx1) * sx, (box[1] - cy1) * sy,
                          (box[2] - cx1) * sx, (box[3] - cy1) * sy])
        clipped = np.array([max(0.0, moved[0]), max(0.0, moved[1]),
                            min(out_w, moved[2]), min(out_h, moved[3])])
        bw, bh = clipped[2] - clipped[0], clipped[3] - clipped[1]
        if bw <= 2 or bh <= 2:
            continue                     # outside the crop entirely
        full = (moved[2] - moved[0]) * (moved[3] - moved[1])
        if full > 0 and (bw * bh) / full < min_visible:
            continue                     # only a sliver survived

        new_kpts = []
        for kx, ky, kv in kpts:
            nx, ny = (kx - cx1) * sx, (ky - cy1) * sy
            if kv == 0 or not (0 <= nx < out_w and 0 <= ny < out_h):
                new_kpts.append((0.0, 0.0, 0))   # out of frame -> not visible
            else:
                new_kpts.append((nx, ny, kv))
        kept.append((cls, clipped, new_kpts))

    return kept or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=("train", "val", "test"),
                    help="split to augment; val/test are refused unless --force")
    ap.add_argument("--per-image", type=int, default=2, help="crops generated per source image")
    ap.add_argument("--min-occupancy", type=float, default=0.10,
                    help="lowest fraction of the frame the arm should cover")
    ap.add_argument("--max-occupancy", type=float, default=0.55,
                    help="highest fraction of the frame the arm should cover")
    ap.add_argument("--jitter", type=float, default=0.15,
                    help="crop centre offset, as a fraction of crop size")
    ap.add_argument("--min-visible", type=float, default=0.35,
                    help="minimum fraction of the arm box that must remain in frame")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="allow augmenting val/test")
    args = ap.parse_args()

    if args.split != "train" and not args.force:
        print(f"Refusing to augment '{args.split}': synthetic crops in val/test would "
              f"stop them measuring real framing. Pass --force to override.")
        return 1

    img_dir = POSE / "images" / args.split
    lbl_dir = POSE / "labels" / args.split
    if not img_dir.is_dir():
        print(f"{img_dir} missing - run scripts/02_prepare_dataset.py first.")
        return 1

    # Clear any previous run so repeated invocations don't accumulate.
    removed = 0
    for d in (img_dir, lbl_dir):
        for p in list(d.iterdir()):
            if p.name.startswith(PREFIX):
                p.unlink()
                removed += 1

    sources = []
    for lbl in sorted(lbl_dir.iterdir()):
        if lbl.suffix != ".txt" or lbl.name.startswith(PREFIX):
            continue
        text = lbl.read_text()
        insts = [i for i in parse_label(text, 1, 1) if any(k[2] for k in i[2])]
        if not insts:
            continue                     # box-only image: no pose signal to preserve
        img = next((p for p in img_dir.glob(f"{lbl.stem}.*") if p.suffix.lower() in IMG_EXTS), None)
        if img:
            sources.append((img, lbl))

    if not sources:
        print(f"No keypoint-labeled images in {lbl_dir}.")
        return 1

    rng = random.Random(args.seed)
    made = skipped = kpt_dropped = kpt_total = 0
    occupancies, samples = [], []

    for img_path, lbl_path in sources:
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        instances = parse_label(lbl_path.read_text(), w, h)
        target = next((i for i in instances if any(k[2] for k in i[2])), None)
        if target is None:
            continue

        for k in range(args.per_image):
            occ = rng.uniform(args.min_occupancy, args.max_occupancy)
            crop = plan_crop(w, h, target[1], occ, args.jitter, rng)
            if crop is None:
                skipped += 1
                continue
            new_inst = transform(instances, crop, w, h, args.min_visible)
            if new_inst is None:
                skipped += 1
                continue

            x1, y1, x2, y2 = crop
            patch = im[int(y1):int(y2), int(x1):int(x2)]
            if patch.size == 0:
                skipped += 1
                continue
            patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)

            stem = f"{PREFIX}{img_path.stem}_{k}"
            cv2.imwrite(str(img_dir / f"{stem}{img_path.suffix}"), patch)
            (lbl_dir / f"{stem}.txt").write_text(format_label(new_inst, w, h))
            made += 1

            for _, box, kpts in new_inst:
                occupancies.append((box[2] - box[0]) * (box[3] - box[1]) / (w * h))
                for _, _, kv in kpts:
                    kpt_total += 1
                    kpt_dropped += (kv == 0)
            if len(samples) < 8:
                samples.append((patch, new_inst))

    if samples:
        preview = DATA / "preview_closeup.jpg"
        montage(samples, preview)

    occ = np.array(occupancies) if occupancies else np.array([0.0])
    print(f"Source images (with keypoints): {len(sources)}")
    if removed:
        print(f"Cleared {removed} file(s) from a previous run.")
    print(f"Crops written : {made}   skipped: {skipped}")
    print(f"Arm occupancy : median {100*np.median(occ):.1f}%  "
          f"range {100*occ.min():.1f}%-{100*occ.max():.1f}%   (source images ~3%)")
    print(f"Keypoints out of frame -> visibility 0: {kpt_dropped}/{kpt_total} "
          f"({100*kpt_dropped/max(1,kpt_total):.1f}%)")
    if samples:
        print(f"Preview: {DATA / 'preview_closeup.jpg'}")
    print(f"\n{args.split} split now has {len(list(img_dir.iterdir()))} images.")
    print("Next: python scripts/04_augment_handview.py   (optional hand-POV views)")
    print("      python scripts/05_train.py --name arm_pose_v6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
