"""
Step 2 - Build the ARM-ONLY YOLO-pose dataset (train/val/test split + data.yaml).

Takes the images in data/raw/ and the keypoint labels written by 01_label.py
or the import_* scripts (data/labels/) and lays them out the way Ultralytics
pose training expects:

    data/pose/
      images/train  images/val  images/test
      labels/train  labels/val  labels/test
      data.yaml

data.yaml declares kpt_shape: [2, 3] (2 keypoints - proximal/distal - each
x/y/visibility) and the single class (arm). Only images that have a label file
are included; an empty label file is kept as a background negative.

The split is GROUP-aware: consecutive frames of the same video clip (phleb__*,
v1__*, v2__*) and burst captures from the same day (arm_YYYYMMDD_*) are
near-duplicates, so the whole group is assigned to a single split. A plain
per-image shuffle leaks neighbouring frames across splits and inflates
val/test metrics.

Labels in data/labels/ arrive in mixed formats, so each file is normalized to
the 11-column 2-keypoint layout (class cx cy w h k0x k0y v0 k1x k1y v1):

  * 11 columns          - kept as-is (manual arm_2 labels, phleb import)
  * 17 columns, class 0 - 4-keypoint layout from 05_convert_*; kpt2/kpt3 are
                          padding, so the first two keypoints are kept
  * class != 0          - foreign annotation (v1__/v2__ files label the NEEDLE,
                          not the arm). If a file has nothing else, the image
                          is EXCLUDED: its unlabeled visible arm would train
                          the model that arms are background.

A genuinely empty label file is still kept as a background negative.
"""
import argparse
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RAW = DATA / "raw"
LABELS = DATA / "labels"
POSE = DATA / "pose"

CLASSES = ["arm"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Filename stem -> group key. Frames that must not straddle splits share a key.
# Checked in order; first match wins. Unmatched stems are their own group
# (independent stills are safe to split individually).
GROUP_PATTERNS = [
    # phleb__<roboflow split>__frame_0007_79_jpg.rf.<hash> -> clip frame_0007.
    # The upstream Roboflow split tag is ignored: the same clip's frames appear
    # under train/valid/test up there, which is exactly the leak we're fixing.
    (re.compile(r"^phleb__(?:train|valid|test)__(frame_\d+)"), "phleb/{0}"),
    # v1__frames_3_frame_000042 / v2__veni2_frame_000014 -> one group per video
    (re.compile(r"^(v\d+)__(.+?)_frame_\d+$"), "{0}/{1}"),
    # arm_20260706_220940_237 -> burst captures, ~1 frame/sec; group by day
    (re.compile(r"^arm_(\d{8})_\d{6}_\d+$"), "arm/{0}"),
]


def group_key(stem: str) -> str:
    for pattern, fmt in GROUP_PATTERNS:
        m = pattern.match(stem)
        if m:
            return fmt.format(*m.groups())
    return stem


def normalize_label(lbl: Path):
    """Normalize one label file to 11-column 2-keypoint lines.

    Returns (text, action) where action is one of "kept", "converted",
    "negative", or (None, "excluded") when the image must not enter the
    dataset (only foreign-class annotations, or malformed lines).
    """
    raw = lbl.read_text().strip()
    if not raw:
        return "", "negative"

    out, converted, foreign = [], False, 0
    for line in raw.splitlines():
        parts = line.split()
        if not parts:
            continue
        if int(float(parts[0])) != 0:
            foreign += 1
            continue
        if len(parts) == 11:
            out.append(" ".join(parts))
        elif len(parts) == 17:
            # 4-keypoint layout; kpt2/kpt3 are padding -> keep box + kpt0/kpt1
            out.append(" ".join(parts[:11]))
            converted = True
        else:
            return None, "excluded"

    if not out:
        # Only needle/foreign annotations: the visible arm is unlabeled, so
        # keeping this image would supervise "arm = background".
        return None, "excluded"
    return "\n".join(out) + "\n", ("converted" if converted else "kept")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.15, help="fraction of images held out for validation")
    ap.add_argument("--test-frac", type=float, default=0.15, help="fraction of images held out for the test set")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.val_frac + args.test_frac >= 1.0:
        print("val-frac + test-frac must be < 1.")
        return 1

    pairs = []
    stats = {"kept": 0, "converted": 0, "negative": 0, "excluded": 0}
    for img in sorted(RAW.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = LABELS / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        text, action = normalize_label(lbl)
        stats[action] += 1
        if text is not None:
            pairs.append((img, text))

    if not pairs:
        print(f"No labeled images found. Run scripts/01_label.py first (need .txt files in {LABELS}).")
        return 1

    # Bucket images into groups, then assign whole groups to splits.
    groups = defaultdict(list)
    for img, text in pairs:
        groups[group_key(img.stem)].append((img, text))

    rng = random.Random(args.seed)
    order = sorted(groups)
    rng.shuffle(order)

    total = len(pairs)
    targets = {
        "train": total * (1.0 - args.val_frac - args.test_frac),
        "val": total * args.val_frac,
        "test": total * args.test_frac,
    }
    assigned = {"train": [], "val": [], "test": []}
    for key in order:
        # Greedy: put the group where the remaining deficit is largest, so the
        # handful of very large video groups can't blow one split's budget.
        split = max(targets, key=lambda s: targets[s] - len(assigned[s]))
        assigned[split].extend(groups[key])

    if POSE.exists():
        shutil.rmtree(POSE)
    for split in assigned:
        (POSE / "images" / split).mkdir(parents=True, exist_ok=True)
        (POSE / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, text in assigned[split]:
            shutil.copy2(img, POSE / "images" / split / img.name)
            (POSE / "labels" / split / f"{img.stem}.txt").write_text(text)

    data_yaml = {
        "path": str(POSE.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "kpt_shape": [2, 3],         # 2 keypoints (proximal, distal), each (x, y, visibility)
        "flip_idx": [0, 1],          # collinear points, no L/R pairs -> identity
        "names": {i: n for i, n in enumerate(CLASSES)},
    }
    (POSE / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    n_groups = {s: len({group_key(img.stem) for img, _ in items}) for s, items in assigned.items()}
    print(f"Wrote {POSE / 'data.yaml'}")
    print(f"  labels: kept={stats['kept']}  converted 17->11 cols={stats['converted']}  "
          f"background negatives={stats['negative']}  excluded (foreign class)={stats['excluded']}")
    print(f"  groups={len(groups)}  classes={CLASSES}  kpt_shape={data_yaml['kpt_shape']}")
    for split in ("train", "val", "test"):
        print(f"  {split}={len(assigned[split])} images in {n_groups[split]} groups")
    print("Next: python scripts/03_train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
