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
x/y/visibility) and the single class (arm).

Labels in data/labels/ arrive in mixed formats, so each file is normalized to
the 11-column 2-keypoint layout (class cx cy w h k0x k0y v0 k1x k1y v1):

  * 11 columns          - kept as-is (manual arm_2 labels, phleb import)
  * 17 columns, class 0 - 4-keypoint layout from import_arm_segmentation_to_pose.py;
                          kpt2/kpt3 are
                          padding, so the first two keypoints are kept
  * class != 0          - foreign annotation (v1__/v2__ files label the NEEDLE,
                          not the arm). If a file has nothing else, the image
                          is EXCLUDED: its unlabeled visible arm would train
                          the model that arms are background.

A genuinely empty label file is kept as a background negative.

SPLITTING
---------
The dataset has two tiers of supervision and each is split differently:

  * KEYPOINT-BEARING images (the manually-labeled arm_ captures) are the only
    source of pose supervision, so all three splits must contain some or the
    pose metrics are undefined. These are burst captures (~1 frame/sec), so
    they are grouped into contiguous SESSIONS (gap > --session-gap starts a
    new one) and each session is assigned WHOLE to one split. A session is a
    single scene - one room, one lighting setup, one mannequin pose - so
    splitting one across train and val would measure "same scene, new angle"
    while reporting it as generalization. Whole sessions keep val/test
    genuinely unseen, which is the only way the pose numbers mean anything.

    --cut-sessions overrides this and cuts long sessions into contiguous
    blocks (with a --guard frame gap dropped at each boundary). That buys a
    larger val set at the cost of an optimistic one; use it only if you need
    a smoother val curve and will not quote the result.

  * BOX-ONLY images (phleb video frames, web stills) supervise the detector.
    They are assigned whole-group, where a group is a video clip or a single
    still, so neighbouring frames never straddle a split.

Both tiers respect group integrity: a per-image shuffle would put near-
duplicate frames on both sides of a split and inflate val/test metrics.
"""
import argparse
import random
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RAW = DATA / "raw"
LABELS = DATA / "labels"
POSE = DATA / "pose"

CLASSES = ["arm"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")
# Order used to break ties when two splits are equally short of target. test is
# filled before val so the reported holdout gets the larger capture session.
FILL_ORDER = ("train", "test", "val")

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

CAPTURE_TS = re.compile(r"^arm_(\d{8})_(\d{6})_\d+$")


def group_key(stem: str) -> str:
    for pattern, fmt in GROUP_PATTERNS:
        m = pattern.match(stem)
        if m:
            return fmt.format(*m.groups())
    return stem


def capture_time(stem: str):
    """Timestamp of an arm_YYYYMMDD_HHMMSS_mmm burst capture, else None."""
    m = CAPTURE_TS.match(stem)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def has_visible_keypoints(text: str) -> bool:
    """True if any instance carries a keypoint with visibility > 0."""
    for line in text.splitlines():
        p = line.split()
        if len(p) == 11 and (p[7] != "0" or p[10] != "0"):
            return True
    return False


def normalize_label(lbl: Path):
    """Normalize one label file to 11-column 2-keypoint lines.

    Returns (text, action) where action is one of "kept", "converted",
    "negative", or (None, "excluded") when the image must not enter the
    dataset (only foreign-class annotations, or malformed lines).
    """
    raw = lbl.read_text().strip()
    if not raw:
        return "", "negative"

    out = []
    converted = False
    for line in raw.splitlines():
        parts = line.split()
        if not parts:
            continue
        if int(float(parts[0])) != 0:
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


def build_sessions(items, gap_seconds):
    """Split timestamped captures into contiguous sessions.

    items: list of (img_path, text). Returns a list of sessions, each a
    time-ordered list of items. Untimestamped items become singleton sessions.
    """
    timed, untimed = [], []
    for img, text in items:
        ts = capture_time(img.stem)
        (timed if ts is not None else untimed).append((ts, img, text))
    timed.sort(key=lambda r: r[0])

    sessions, current, prev = [], [], None
    for ts, img, text in timed:
        if prev is not None and (ts - prev).total_seconds() > gap_seconds:
            sessions.append(current)
            current = []
        current.append((img, text))
        prev = ts
    if current:
        sessions.append(current)
    sessions.extend([[(img, text)] for _, img, text in untimed])
    return sessions


def neediest(assigned, targets):
    """Split furthest below its target; FILL_ORDER breaks ties."""
    return max(FILL_ORDER, key=lambda s: targets[s] - len(assigned[s]))


def split_sessions(sessions, fracs, guard, min_cut, cut):
    """Assign capture sessions across splits.

    By default each session goes whole to the split furthest below target, so
    no scene appears in more than one split. With cut=True, sessions of at
    least min_cut frames are instead cut into contiguous train/val/test blocks
    (dropping `guard` frames at each boundary) - larger val, but val and test
    then share a scene with train.
    """
    assigned = {s: [] for s in SPLITS}
    total = sum(len(s) for s in sessions)
    targets = {s: total * fracs[s] for s in SPLITS}
    dropped = 0

    for sess in sorted(sessions, key=len, reverse=True):
        if cut and len(sess) >= min_cut:
            n = len(sess)
            a = int(n * fracs["train"])
            b = a + guard
            c = b + int(n * fracs["val"])
            d = c + guard
            assigned["train"].extend(sess[:a])
            assigned["val"].extend(sess[b:c])
            assigned["test"].extend(sess[d:])
            dropped += min(guard, max(0, n - a)) + min(guard, max(0, n - c))
        else:
            assigned[neediest(assigned, targets)].extend(sess)
    return assigned, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.15, help="fraction of images held out for validation")
    ap.add_argument("--test-frac", type=float, default=0.15, help="fraction of images held out for the test set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--session-gap", type=float, default=120.0,
                    help="seconds between captures that starts a new session")
    ap.add_argument("--cut-sessions", action="store_true",
                    help="cut long capture sessions across splits (larger but optimistic val)")
    ap.add_argument("--guard", type=int, default=5,
                    help="frames dropped at each in-session boundary when --cut-sessions is set")
    ap.add_argument("--min-cut", type=int, default=60,
                    help="with --cut-sessions, sessions at least this long are cut")
    args = ap.parse_args()

    if args.val_frac + args.test_frac >= 1.0:
        print("val-frac + test-frac must be < 1.")
        return 1
    fracs = {
        "train": 1.0 - args.val_frac - args.test_frac,
        "val": args.val_frac,
        "test": args.test_frac,
    }

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

    # Tier 1: the timestamped burst captures. These carry nearly all of the
    # keypoint supervision, so they are spread across all three splits or the
    # pose metrics are undefined. Selected on timestamp rather than on "has
    # keypoints" so that the few keypoint-less frames of a session travel with
    # the rest of their scene instead of leaking it into another split.
    kpt_items = [(i, t) for i, t in pairs if capture_time(i.stem) is not None]
    box_items = [(i, t) for i, t in pairs if capture_time(i.stem) is None]

    sessions = build_sessions(kpt_items, args.session_gap)
    assigned, guard_dropped = split_sessions(
        sessions, fracs, args.guard, args.min_cut, args.cut_sessions)

    # Tier 2: box-only groups -> whole groups, greedily filling toward the
    # global per-split targets already partly consumed by the keypoint tier.
    groups = defaultdict(list)
    for img, text in box_items:
        groups[group_key(img.stem)].append((img, text))

    rng = random.Random(args.seed)
    order = sorted(groups)
    rng.shuffle(order)

    total = len(kpt_items) - guard_dropped + len(box_items)
    targets = {s: total * fracs[s] for s in SPLITS}
    for key in order:
        # Greedy: put the group where the remaining deficit is largest, so the
        # handful of very large video groups can't blow one split's budget.
        assigned[neediest(assigned, targets)].extend(groups[key])

    if POSE.exists():
        shutil.rmtree(POSE)
    for split in SPLITS:
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

    print(f"Wrote {POSE / 'data.yaml'}")
    print(f"  labels: kept={stats['kept']}  converted 17->11 cols={stats['converted']}  "
          f"background negatives={stats['negative']}  excluded (foreign class)={stats['excluded']}")
    print(f"  keypoint sessions={len(sessions)}  guard frames dropped={guard_dropped}")
    print(f"  box-only groups={len(groups)}  classes={CLASSES}  kpt_shape={data_yaml['kpt_shape']}")
    for split in SPLITS:
        items = assigned[split]
        n_kpt = sum(1 for _, t in items if has_visible_keypoints(t))
        print(f"  {split:<5} {len(items):>6} images  ({n_kpt} with keypoints)")
    print("Next: python scripts/03_augment_closeups.py   (then 04_augment_handview.py, 05_train.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
