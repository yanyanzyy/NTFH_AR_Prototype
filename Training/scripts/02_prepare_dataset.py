"""Builds the YOLO-pose training dataset from labeled mannequin images,
optionally mixed with real-human images from COCO val2017 so the fine-tuned
model RETAINS its ability to detect real human arms (needed for the future
patient phase — without mixing, fine-tuning would 'forget' humans).

Reads
-----
  dataset/raw/         images captured on the headset
  dataset/raw_labels/  labels written by 01_label_arm.py

Writes
------
  dataset/yolo/images/{train,val}/
  dataset/yolo/labels/{train,val}/
  dataset/yolo/data.yaml

COCO mixing downloads val2017 images (~780 MB) + annotations (~240 MB) on the
first run into dataset/coco_cache/ and converts person annotations to
YOLO-pose labels. Skip it with --no-coco (NOT recommended unless you only
ever target the mannequin).

Run:  python 02_prepare_dataset.py
      python 02_prepare_dataset.py --no-coco --oversample 1
"""

import argparse
import json
import random
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # Training/
RAW_DIR = ROOT / "dataset" / "raw"
LABEL_DIR = ROOT / "dataset" / "raw_labels"
OUT_DIR = ROOT / "dataset" / "yolo"
COCO_CACHE = ROOT / "dataset" / "coco_cache"

COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def download(url: str, dest: Path):
    import requests
    from tqdm import tqdm
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def ensure_coco_annotations() -> Path:
    """Returns the path of person_keypoints_val2017.json, downloading if needed."""
    ann_json = COCO_CACHE / "person_keypoints_val2017.json"
    if ann_json.exists():
        return ann_json
    ann_zip = COCO_CACHE / "annotations_trainval2017.zip"
    if not ann_zip.exists():
        download(COCO_ANN_URL, ann_zip)
    with zipfile.ZipFile(ann_zip) as zf:
        member = "annotations/person_keypoints_val2017.json"
        with zf.open(member) as src, open(ann_json, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return ann_json


def ensure_coco_images() -> Path:
    """Returns the val2017 image zip path, downloading if needed (extracted lazily)."""
    img_zip = COCO_CACHE / "val2017.zip"
    if not img_zip.exists():
        download(COCO_IMAGES_URL, img_zip)
    return img_zip


def coco_to_yolo_entries(ann_json: Path, max_images: int, seed: int):
    """Yields (file_name, [label_line, ...]) for COCO images containing labeled persons."""
    print("Converting COCO person annotations…")
    data = json.loads(ann_json.read_text())
    images = {im["id"]: im for im in data["images"]}

    per_image = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd") or ann.get("num_keypoints", 0) < 1:
            continue
        per_image[ann["image_id"]].append(ann)

    ids = sorted(per_image.keys())
    random.Random(seed).shuffle(ids)
    ids = ids[:max_images]

    entries = []
    for img_id in ids:
        im = images[img_id]
        w, h = im["width"], im["height"]
        lines = []
        for ann in per_image[img_id]:
            x, y, bw, bh = ann["bbox"]
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            nw, nh = bw / w, bh / h
            if nw <= 0 or nh <= 0:
                continue
            kps = ann["keypoints"]  # 17 * (x, y, v)
            kp_vals = []
            for k in range(17):
                kx, ky, kv = kps[k * 3], kps[k * 3 + 1], kps[k * 3 + 2]
                kp_vals += [kx / w, ky / h, float(kv)]
            vals = [0, cx, cy, nw, nh] + kp_vals
            lines.append(" ".join(f"{v:.6f}" if isinstance(v, float) else str(v)
                                  for v in vals))
        if lines:
            entries.append((im["file_name"], lines))
    print(f"  {len(entries)} COCO images with person keypoints selected.")
    return entries


def extract_coco_images(img_zip: Path, file_names, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    from tqdm import tqdm
    with zipfile.ZipFile(img_zip) as zf:
        for name in tqdm(file_names, desc="Extracting COCO images"):
            target = dest / name
            if target.exists():
                continue
            with zf.open(f"val2017/{name}") as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def write_split(pairs, split, prefix=""):
    """pairs: list of (image_path, label_text). Copies into dataset/yolo/<split>."""
    img_dir = OUT_DIR / "images" / split
    lbl_dir = OUT_DIR / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for img_path, label_text, name in pairs:
        stem = f"{prefix}{name}"
        shutil.copy2(img_path, img_dir / f"{stem}{img_path.suffix.lower()}")
        (lbl_dir / f"{stem}.txt").write_text(label_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-coco", action="store_true",
                    help="skip mixing COCO human images (mannequin-only model)")
    ap.add_argument("--coco-max", type=int, default=1500,
                    help="max COCO person images to mix in (default 1500)")
    ap.add_argument("--val-split", type=float, default=0.15,
                    help="fraction of mannequin images held out for validation")
    ap.add_argument("--oversample", type=int, default=3,
                    help="times each mannequin training image is duplicated, balancing "
                         "it against the larger COCO set (default 3)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ── Collect labeled mannequin images ───────────────────────────────────────
    if not LABEL_DIR.exists():
        sys.exit(f"{LABEL_DIR} not found — run 01_label_arm.py first.")
    mannequin = []
    for lbl in sorted(LABEL_DIR.glob("*.txt")):
        img = next((RAW_DIR / f"{lbl.stem}{ext}" for ext in IMG_EXTS
                    if (RAW_DIR / f"{lbl.stem}{ext}").exists()), None)
        if img is None:
            print(f"  WARNING: label without image, skipping: {lbl.name}")
            continue
        mannequin.append((img, lbl.read_text(), lbl.stem))
    if not mannequin:
        sys.exit("No labeled images found — run 01_label_arm.py first.")
    print(f"{len(mannequin)} labeled mannequin image(s) "
          f"({sum(1 for _, t, _ in mannequin if not t.strip())} background).")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    rng = random.Random(args.seed)
    rng.shuffle(mannequin)
    n_val = max(1, int(len(mannequin) * args.val_split))
    man_val, man_train = mannequin[:n_val], mannequin[n_val:]

    # Oversample mannequin training images so they aren't drowned out by COCO.
    train_pairs = []
    for rep in range(max(1, args.oversample)):
        for img, text, name in man_train:
            train_pairs.append((img, text, f"{name}_r{rep}" if rep else name))

    write_split(train_pairs, "train", prefix="man_")
    write_split(man_val, "val", prefix="man_")

    # ── COCO human mixing ──────────────────────────────────────────────────────
    if not args.no_coco:
        ann_json = ensure_coco_annotations()
        img_zip = ensure_coco_images()
        entries = coco_to_yolo_entries(ann_json, args.coco_max, args.seed)

        coco_img_dir = COCO_CACHE / "val2017"
        extract_coco_images(img_zip, [n for n, _ in entries], coco_img_dir)

        rng.shuffle(entries)
        n_coco_val = max(1, len(entries) // 10)
        coco_pairs = [(coco_img_dir / n, "\n".join(lines) + "\n", Path(n).stem)
                      for n, lines in entries]
        write_split(coco_pairs[n_coco_val:], "train", prefix="coco_")
        write_split(coco_pairs[:n_coco_val], "val", prefix="coco_")

    # ── data.yaml ──────────────────────────────────────────────────────────────
    yaml_text = (
        f"path: {OUT_DIR.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "kpt_shape: [17, 3]\n"
        "flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]\n"
        "names:\n"
        "  0: person\n"
    )
    (OUT_DIR / "data.yaml").write_text(yaml_text)

    n_train = len(list((OUT_DIR / "images" / "train").iterdir()))
    n_val_total = len(list((OUT_DIR / "images" / "val").iterdir()))
    print(f"\nDataset ready: {n_train} train / {n_val_total} val images")
    print(f"  {OUT_DIR / 'data.yaml'}")
    print("Next: python 03_train.py")


if __name__ == "__main__":
    main()
